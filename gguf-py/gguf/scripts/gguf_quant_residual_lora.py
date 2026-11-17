#!/usr/bin/env python3
"""Residual-LoRA compensation generator for a quantised GGUF (row S69-ENGREV-RKNN-QUANT-RESIDUAL-LORA-2026-10-27-01).

For every 2-D weight W in a high-precision reference GGUF whose same-named tensor in the quantised GGUF is a quantised type
Q, form W' = dequant(Q), E = W - W', take the top-r SVD E ~ U S Vt and write the adapter tensors
    <name>.lora_a = sqrt(S) Vt   (numpy [r, in]  -> GGUF ne [in, r])
    <name>.lora_b = U sqrt(S)    (numpy [out, r] -> GGUF ne [r, out])
so that  W' + B A  ~=  W  (same layout convert_lora_to_gguf.py writes; lora_alpha 0 so the loader scale is the --lora-scaled user
scale, 1.0 by default: src/llama-adapter.h get_scale). SPEC from the rknn-llm SDK description of its normal_rN / grq_rN build
option (rank-N compensation LoRA exported next to the quantised model); the algorithm is plain truncated SVD, no code copied
(rule 1.87). The adapter is exact only for the SAME quantised file.
Skipped (reported): 1-D tensors, tensors that are not quantised in the quant file, >2-D tensors (fused MoE experts: LoRA on
those is unverified in this fork), tensors absent from either file, shape mismatches.
Usage: gguf_quant_residual_lora.py REF.gguf QUANT.gguf OUT.gguf --rank 32 [--outtype f32|f16]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

logger = logging.getLogger("gguf-quant-residual-lora")

_PLAIN = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)


def _as_f32(t) -> np.ndarray:
    """Full float32 matrix [rows, cols] of a GGUFReader tensor, dequantising when needed."""
    qt = t.tensor_type
    if qt == gguf.GGMLQuantizationType.F32:
        arr = np.asarray(t.data, dtype=np.float32)
    elif qt == gguf.GGMLQuantizationType.F16:
        arr = np.asarray(t.data).astype(np.float32)
    elif qt == gguf.GGMLQuantizationType.BF16:
        u = np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16
        arr = u.view(np.float32)
    else:
        arr = dequantize(np.asarray(t.data), qt).astype(np.float32)
    return arr.reshape(tuple(int(x) for x in reversed(t.shape)))


def residual_factors(w: np.ndarray, wq: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(A [r,in], B [out,r], ||E||_F, ||E - BA||_F) for E = w - wq, r clipped to min(out, in)."""
    e = (w - wq).astype(np.float64)
    u, s, vt = np.linalg.svd(e, full_matrices=False)
    r = min(rank, s.shape[0])
    rs = np.sqrt(s[:r])
    b = (u[:, :r] * rs).astype(np.float32)
    a = (rs[:, None] * vt[:r, :]).astype(np.float32)
    return a, b, float(np.linalg.norm(e)), float(np.linalg.norm(e - b.astype(np.float64) @ a.astype(np.float64)))


def build(ref_path: Path, quant_path: Path, out_path: Path, rank: int, outtype: str = "f32") -> dict:
    if rank < 1:
        raise ValueError("--rank must be >= 1 (a rank-0 adapter has no tensors; the base is its own control)")
    ref = gguf.GGUFReader(str(ref_path), "r")
    qnt = gguf.GGUFReader(str(quant_path), "r")
    ref_by = {t.name: t for t in ref.tensors}
    arch_f = qnt.get_field(gguf.Keys.General.ARCHITECTURE)
    arch = bytes(arch_f.parts[arch_f.data[0]]).decode("utf-8") if arch_f is not None else "llama"
    dt = np.float16 if outtype == "f16" else np.float32
    writer = gguf.GGUFWriter(str(out_path), arch)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, 0.0)
    stats = {"done": 0, "skipped": {}, "e_before": 0.0, "e_after": 0.0}

    def skip(n, why):
        stats["skipped"][n] = why

    for t in qnt.tensors:
        n = t.name
        if len(t.shape) == 1:
            skip(n, "1-D")
            continue
        if len(t.shape) != 2:
            skip(n, f"{len(t.shape)}-D (fused experts unverified)")
            continue
        if t.tensor_type in _PLAIN:
            skip(n, f"not quantised ({t.tensor_type.name})")
            continue
        r = ref_by.get(n)
        if r is None:
            skip(n, "absent from reference")
            continue
        if tuple(int(x) for x in r.shape) != tuple(int(x) for x in t.shape):
            skip(n, "shape mismatch")
            continue
        try:
            wq = _as_f32(t)
        except NotImplementedError as ex:
            skip(n, f"no dequant: {ex}")
            continue
        a, b, e0, e1 = residual_factors(_as_f32(r), wq, rank)
        writer.add_tensor(n + ".lora_a", a.astype(dt))
        writer.add_tensor(n + ".lora_b", b.astype(dt))
        stats["done"] += 1
        stats["e_before"] += e0 * e0
        stats["e_after"] += e1 * e1
    if stats["done"] == 0:
        raise RuntimeError("no eligible tensor: " + str(stats["skipped"])[:400])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    stats["e_before"] = stats["e_before"] ** 0.5
    stats["e_after"] = stats["e_after"] ** 0.5
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("ref", type=Path)
    p.add_argument("quant", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--outtype", choices=["f32", "f16"], default="f32")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    st = build(a.ref, a.quant, a.out, a.rank, a.outtype)
    logger.info("tensors=%d skipped=%d residual ||E||=%.6g -> %.6g (rank %d)", st["done"], len(st["skipped"]), st["e_before"], st["e_after"], a.rank)
    return 0


if __name__ == "__main__":
    sys.exit(main())
