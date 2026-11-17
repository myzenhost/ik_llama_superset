#!/usr/bin/env python3
# GPTQ int4 -> Q4_0 / Q4_1 exact repack (gguf/gptq_repack.py). numpy only, no model needed.
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf  # noqa: E402
from gguf import gptq_repack  # noqa: E402

K, N, GS = 256, 24, 128


def pack(q: np.ndarray, z: np.ndarray, zero_offset: int):
    """GPTQ layout: qweight int32 [K/8, N] (8 nibbles along K), qzeros int32 [G, N/8] (8 along N)."""
    k, n = q.shape
    qw = np.zeros((k // 8, n), dtype=np.uint32)
    for j in range(8):
        qw |= q[j::8, :].astype(np.uint32) << np.uint32(4 * j)
    zs = (z.astype(np.int32) - zero_offset).astype(np.uint32)  # stored zero
    qz = np.zeros((z.shape[0], n // 8), dtype=np.uint32)
    for j in range(8):
        qz |= zs[:, j::8] << np.uint32(4 * j)
    return qw.view(np.int32), qz.view(np.int32)


def make(seed=0, sym=False, zero_offset=1, qlo=2, qhi=13):
    rng = np.random.default_rng(seed)
    q = rng.integers(qlo, qhi + 1, size=(K, N)).astype(np.uint8)  # does not span 0..15 per block
    g = K // GS
    z = np.full((g, N), 8, dtype=np.int16) if sym else rng.integers(3, 13, size=(g, N)).astype(np.int16)
    scales = (rng.random((g, N)) * 0.02 + 0.003).astype(np.float16)
    gidx = (np.arange(K) // GS).astype(np.int32)
    qw, qz = pack(q, z, zero_offset)
    # reference: scale * (q - zero) in float, rows = output features
    ref = (scales.astype(np.float32)[gidx] * (q.astype(np.float32) - z[gidx].astype(np.float32))).T
    return qw, qz, scales, gidx, ref


def deq(res):
    qtype, raw = res
    return gguf.quants.dequantize(raw, qtype)


def test_unpack_roundtrip():
    qw, qz, *_ = make()
    q, _ = gptq_repack.unpack_gptq_int4(qw, qz)
    assert q.shape == (K, N) and q.min() >= 2 and q.max() <= 13


def test_asym_repacks_to_q4_1_exactly():
    qw, qz, s, gi, ref = make()
    res = gptq_repack.repack_gptq_int4(qw, qz, s, gi, 1)
    assert res is not None and res[0] == gguf.GGMLQuantizationType.Q4_1
    err = np.abs(deq(res) - ref).max()
    assert err <= np.abs(ref).max() * 2.0 ** -10, err
    assert gptq_repack.matches_float(res[1], res[0], ref)


def test_float_path_is_not_exact_control():
    # the path being replaced: quantise the float result again. Blocks that do not use the whole
    # 0..15 range get a different scale, so the error is far above the exact repack error.
    qw, qz, s, gi, ref = make()
    exact = np.abs(deq(gptq_repack.repack_gptq_int4(qw, qz, s, gi, 1)) - ref).max()
    again = gguf.quants.dequantize(gguf.quants.quantize(ref.astype(np.float32), gguf.GGMLQuantizationType.Q4_1),
                                   gguf.GGMLQuantizationType.Q4_1)
    assert np.abs(again - ref).max() > exact * 8


def test_symmetric_repacks_to_q4_0():
    qw, qz, s, gi, ref = make(sym=True)
    res = gptq_repack.repack_gptq_int4(qw, qz, s, gi, 1)
    assert res is not None and res[0] == gguf.GGMLQuantizationType.Q4_0
    assert np.abs(deq(res) - ref).max() <= np.abs(ref).max() * 2.0 ** -10
    assert res[1].shape == (N, (K // 32) * 18)


def test_gptq_v2_zero_offset():
    qw, qz, s, gi, ref = make(zero_offset=0)
    res = gptq_repack.repack_gptq_int4(qw, qz, s, gi, 0)
    assert res is not None and np.abs(deq(res) - ref).max() <= np.abs(ref).max() * 2.0 ** -10


def test_negative_control_zero_off_by_one_fails():
    qw, qz, s, gi, ref = make()
    bad = gptq_repack.repack_gptq_int4(qw, qz, s, gi, 0)  # forgot the +1 of the v1 format
    assert bad is not None
    assert np.abs(deq(bad) - ref).max() > np.abs(ref).max() * 2.0 ** -6
    assert not gptq_repack.matches_float(bad[1], bad[0], ref)


def test_non_sequential_g_idx_falls_back():
    qw, qz, s, gi, _ = make()
    perm = np.random.default_rng(1).permutation(K).astype(np.int32)
    assert gptq_repack.repack_gptq_int4(qw, qz, s, gi[perm], 1) is None


def test_group_not_multiple_of_32_falls_back():
    rng = np.random.default_rng(2)
    k, n, gs = 64, 8, 16
    q = rng.integers(0, 16, size=(k, n)).astype(np.uint8)
    z = np.full((k // gs, n), 8, dtype=np.int16)
    qw, qz = pack(q, z, 1)
    s = np.full((k // gs, n), 0.01, dtype=np.float16)
    assert gptq_repack.repack_gptq_int4(qw, qz, s, (np.arange(k) // gs).astype(np.int32), 1) is None


def test_permuted_tensor_is_not_accepted_by_guard():
    qw, qz, s, gi, ref = make()
    res = gptq_repack.repack_gptq_int4(qw, qz, s, gi, 1)
    permuted = ref.reshape(N, K // 2, 2)[:, :, ::-1].reshape(N, K)  # converter-style column swap
    assert not gptq_repack.matches_float(res[1], res[0], permuted)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
