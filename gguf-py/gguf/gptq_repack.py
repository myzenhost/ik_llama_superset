"""Exact repack of GPTQ int4 tensors into ggml Q4_0 / Q4_1 blocks.

A GPTQ int4 weight is value = scale * (q - zero) with q in 0..15, one (scale, zero) pair per
group of `group_size` input columns. Dequantising it to float and letting the converter quantise
it again adds a second rounding on top of GPTQ's calibrated grid (a Q4_x block recomputes its own
scale from the block min/max, which differs from the stored scale whenever the block does not
use the full 0..15 range). The stored integers are instead written into the blocks unchanged:

  * every zero == 8  -> Q4_0, d = scale, q unchanged   (value = d * (q - 8))
  * otherwise        -> Q4_1, d = scale, m = -scale * zero (value = d * q + m), m rounded to fp16

Only numpy is used so the routine can be tested without torch. A tensor that cannot be repacked
(act-order g_idx that is not sequential, group size not a multiple of 32, bits != 4, columns not
a multiple of 32) returns None and the caller keeps the float path.
"""
from __future__ import annotations

import numpy as np

from .constants import GGMLQuantizationType

QK = 32  # ggml Q4_0 / Q4_1 block size
PACK = 8  # int4 values per int32


def unpack_gptq_int4(qweight: np.ndarray, qzeros: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """qweight int32 [K/8, N] -> q uint8 [K, N]; qzeros int32 [G, N/8] -> z uint8 [G, N] (as stored)."""
    if qweight.dtype != np.int32 or qzeros.dtype != np.int32:
        raise ValueError("GPTQ int4 packs 8 values per int32")
    shifts = (np.arange(PACK, dtype=np.int32) * 4)
    # [K/8, 8, N]: row i*8+j of the unpacked weight is nibble j of word i
    w = (qweight[:, None, :] >> shifts[None, :, None]) & 0xF
    q = w.reshape(qweight.shape[0] * PACK, qweight.shape[1]).astype(np.uint8)
    # [G, N/8, 8]: column i*8+j is nibble j of word i
    z = (qzeros[:, :, None] >> shifts[None, None, :]) & 0xF
    z = z.reshape(qzeros.shape[0], qzeros.shape[1] * PACK).astype(np.uint8)
    return q, z


def repack_gptq_int4(qweight: np.ndarray, qzeros: np.ndarray, scales: np.ndarray, g_idx: np.ndarray,
                     zero_offset: int = 1) -> tuple[GGMLQuantizationType, np.ndarray] | None:
    """Return (Q4_0 or Q4_1, raw uint8 [N, K/32 * block_bytes]) for a GPTQ tensor, or None.

    zero_offset is 1 for checkpoint_format "gptq" (zeros are stored minus one) and 0 for "gptq_v2".
    The result is in the layout of dequant_gptq(...).T, i.e. rows are output features and each row
    runs along the input dimension, which is the layout ggml expects for a weight matrix.
    """
    q, z = unpack_gptq_int4(qweight, qzeros)
    k, n = q.shape
    n_groups = scales.shape[0]
    if scales.shape != (n_groups, n) or z.shape != (n_groups, n) or g_idx.shape != (k,):
        return None
    if k % QK != 0 or k % n_groups != 0:
        return None
    group_size = k // n_groups
    if group_size % QK != 0:
        return None
    # act-order: only legal when the group map is the plain contiguous one
    if not np.array_equal(g_idx.astype(np.int64), np.arange(k, dtype=np.int64) // group_size):
        return None

    zeros = z.astype(np.int16) + zero_offset  # [G, N], the true zero point
    if int(zeros.min()) < 0 or int(zeros.max()) > 15 + zero_offset:
        return None
    nb = k // QK
    blk_group = (np.arange(nb) * QK) // group_size  # [nb]
    d = scales.astype(np.float32)[blk_group, :].T  # [N, nb]
    zb = zeros[blk_group, :].T  # [N, nb]

    qt = q.T.reshape(n, nb, QK)  # [N, nb, 32]
    lo, hi = qt[..., : QK // 2], qt[..., QK // 2:]
    qs = (lo | (hi << np.uint8(4))).astype(np.uint8)  # [N, nb, 16]
    d16 = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(n, nb, 2)

    if int(zeros.min()) == 8 and int(zeros.max()) == 8:
        raw = np.concatenate([d16, qs], axis=-1)
        qtype = GGMLQuantizationType.Q4_0
    else:
        m = np.ascontiguousarray((-(d * zb.astype(np.float32))).astype(np.float16)).view(np.uint8).reshape(n, nb, 2)
        raw = np.concatenate([d16, m, qs], axis=-1)
        qtype = GGMLQuantizationType.Q4_1
    return qtype, np.ascontiguousarray(raw.reshape(n, -1))


def matches_float(raw: np.ndarray, qtype: GGMLQuantizationType, reference: np.ndarray) -> bool:
    """True when dequantising `raw` reproduces `reference` ([N, K] float) within fp16 rounding of d/m.

    Guards a converter against tensors that were permuted/split after dequantisation (rope permute,
    expert merge): those no longer match the repacked layout and fall back to the float path.
    """
    from . import quants

    if reference.ndim != 2:
        return False
    got = quants.dequantize(raw, qtype)
    if got.shape != reference.shape:
        return False
    ref = reference.astype(np.float32)
    # d and m are stored in fp16: a relative 2^-10 of the block's largest magnitude is the budget
    nb = ref.shape[1] // QK
    mag = np.abs(ref).reshape(ref.shape[0], nb, QK).max(axis=-1, keepdims=True)
    tol = np.maximum(mag * (2.0 ** -9), 1e-30)
    err = np.abs(got.astype(np.float32) - ref).reshape(ref.shape[0], nb, QK)
    return bool((err <= tol).all())
