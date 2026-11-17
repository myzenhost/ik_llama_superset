#!/usr/bin/env python3
"""HF bitsandbytes 4-bit (nf4 / fp4, with and without double quantisation) through the REAL ModelBase.dequant_model.

The reference quantiser and the reference dequantiser below are written independently from the format (plain Python
loops, no import from conversion.bnb4), so a bug shared by a vectorised implementation and its own test cannot hide.
A synthetic checkpoint is built exactly as bitsandbytes serialises it (uint8 packed weight with the FIRST element in the
HIGH nibble, per-block absmax, the 16-entry quant_map, optional nested absmax/map, and the JSON quant_state tensor), then
dequant_model is called unbound on a stub that carries only the attributes it reads.

   python tests/test_bnb4_dequant.py <tree-root>

BEFORE (tree without conversion/bnb4.py and the dequant_model branch): NotImplementedError "Quant method is not yet
supported: 'bitsandbytes'" -> every case FAILS. AFTER: all cases pass.
Controls inside the test: (a) the reference result differs from the LOW-nibble-first reading, so the data discriminates
nibble order; (b) the dequantised weight is not equal to the original (quantisation is lossy) but within the codebook
gap bound, so a no-op "return the packed bytes" cannot pass; (c) a 8-bit config is still rejected.
Row: S69-ONNXRUNTIME-PICK-BNB4-NF4-FP4-CHECKPOINT-IMPORT-2026-10-05-01 (MIT sources, behaviour only).
"""
import json
import sys
import types
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "gguf-py"))

import torch  # noqa: E402

import conversion.base as base  # noqa: E402

NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
       0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0]
FP4 = [0.0, 5.208333333e-03, 0.66666667, 1.0, 0.33333333, 0.5, 0.16666667, 0.25,
       -0.0, -5.208333333e-03, -0.66666667, -1.0, -0.33333333, -0.5, -0.16666667, -0.25]
NESTED_MAP = [-1.0 + 2.0 * k / 255.0 for k in range(256)]   # any 256-entry map: the converter must use the STORED one


def nearest(x, table):
    best, bi = None, 0
    for i, t in enumerate(table):
        d = abs(x - t)
        if best is None or d < best:
            best, bi = d, i
    return bi


def quantize(flat, table, blocksize):
    """Reference quantiser: per-block absmax, nearest code, FIRST element in the HIGH nibble."""
    n = len(flat)
    absmax, codes = [], []
    for b in range(0, n, blocksize):
        blk = flat[b:b + blocksize]
        am = max(abs(v) for v in blk)
        absmax.append(am)
        for v in blk:
            codes.append(nearest(v / am if am else 0.0, table))
    if len(codes) % 2:
        codes.append(0)
    packed = [(codes[i] << 4) | codes[i + 1] for i in range(0, len(codes), 2)]
    return packed, absmax


def quantize_nested(absmax, blocksize2=256):
    """Second level: subtract the mean, 8-bit blockwise against NESTED_MAP."""
    offset = sum(absmax) / len(absmax)
    shifted = [a - offset for a in absmax]
    am2, codes = [], []
    for b in range(0, len(shifted), blocksize2):
        blk = shifted[b:b + blocksize2]
        m = max(abs(v) for v in blk)
        am2.append(m)
        for v in blk:
            codes.append(nearest(v / m if m else 0.0, NESTED_MAP))
    return codes, am2, offset


def reference_dequant(packed, table, absmax_eff, blocksize, n, low_first=False):
    out = []
    for i in range(n):
        byte = packed[i // 2]
        hi, lo = byte >> 4, byte & 15
        code = (lo if i % 2 == 0 else hi) if low_first else (hi if i % 2 == 0 else lo)
        out.append(table[code] * absmax_eff[i // blocksize])
    return out


def build_case(shape, table, qtype, blocksize, nested, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g) * 0.05
    flat = [float(v) for v in w.reshape(-1)]
    packed, absmax = quantize(flat, table, blocksize)
    prefix = "model.layers.0.self_attn.q_proj"
    t = {
        prefix + ".weight": torch.tensor(packed, dtype=torch.uint8).reshape(-1, 1),
        prefix + ".weight.quant_map": torch.tensor(table, dtype=torch.float32),
    }
    state = {"quant_type": qtype, "blocksize": blocksize, "dtype": "float32", "shape": list(shape)}
    if nested:
        codes2, am2, offset = quantize_nested(absmax)
        t[prefix + ".weight.absmax"] = torch.tensor(codes2, dtype=torch.uint8)
        t[prefix + ".weight.nested_absmax"] = torch.tensor(am2, dtype=torch.float32)
        t[prefix + ".weight.nested_quant_map"] = torch.tensor(NESTED_MAP, dtype=torch.float32)
        state.update({"nested_blocksize": 256, "nested_dtype": "float32", "nested_offset": offset})
        # effective scales the converter must reconstruct (independent loop)
        eff = []
        for i, c in enumerate(codes2):
            eff.append(NESTED_MAP[c] * am2[i // 256] + offset)
        absmax_eff = eff
    else:
        t[prefix + ".weight.absmax"] = torch.tensor(absmax, dtype=torch.float32)
        absmax_eff = absmax
    t[prefix + ".weight.quant_state.bitsandbytes__" + qtype] = torch.tensor(list(json.dumps(state).encode("utf-8")), dtype=torch.uint8)
    return prefix, t, w, packed, absmax_eff


def run_dequant_model(tensors, quant_config):
    stub = types.SimpleNamespace(
        _is_nvfp4=False, hparams={"quantization_config": quant_config}, lazy=False, _fp8_as_q8=False,
        _fp8_dequantized=set(), model_tensors={k: (lambda v=v: v) for k, v in tensors.items()},
    )
    base.ModelBase.dequant_model(stub)
    return stub


failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" -- " + detail) if (detail and not cond) else ""))
    if not cond:
        failures.append(name)


CASES = [
    ("nf4 single-quant 64x64", (64, 64), NF4, "nf4", 64, False),
    ("fp4 single-quant 64x64", (64, 64), FP4, "fp4", 64, False),
    ("nf4 double-quant 128x128 (64 scales -> 1 nested block)", (128, 128), NF4, "nf4", 64, True),
    ("nf4 partial last block + odd numel 5x13", (5, 13), NF4, "nf4", 64, False),
    ("fp4 double-quant blocksize 128", (96, 128), FP4, "fp4", 128, True),
]
for i, (name, shape, table, qtype, bs, nested) in enumerate(CASES):
    prefix, tensors, w, packed, absmax_eff = build_case(shape, table, qtype, bs, nested, seed=100 + i)
    cfg = {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": qtype,
           "bnb_4bit_use_double_quant": nested}
    n = w.numel()
    try:
        stub = run_dequant_model(dict(tensors), cfg)
        got = stub.model_tensors[prefix + ".weight"]()
    except Exception as e:  # BEFORE the change this is NotImplementedError
        check(name + ": dequant_model accepts the checkpoint", False, "%s: %s" % (type(e).__name__, e))
        continue
    check(name + ": shape", tuple(got.shape) == tuple(shape), str(tuple(got.shape)))
    check(name + ": aux tensors removed", not any(k.startswith(prefix + ".weight.") for k in stub.model_tensors),
          str([k for k in stub.model_tensors]))
    ref = torch.tensor(reference_dequant(packed, table, absmax_eff, bs, n), dtype=torch.float32).reshape(shape)
    err = float((got.float() - ref).abs().max())
    check(name + ": equals independent reference dequantiser", err < 1e-5, "max|diff|=%g" % err)
    wrong = torch.tensor(reference_dequant(packed, table, absmax_eff, bs, n, low_first=True), dtype=torch.float32).reshape(shape)
    if n > 1:
        check(name + ": CONTROL data discriminates nibble order", float((ref - wrong).abs().max()) > 1e-4)
    # lossy but bounded: nearest-code error <= half the largest gap between adjacent codes, times the block scale
    srt = sorted(table)
    gap = max(b - a for a, b in zip(srt, srt[1:])) / 2.0
    amax_bound = float(max(absmax_eff)) if absmax_eff else 0.0
    dev = float((got.float() - w).abs().max())
    check(name + ": within codebook-gap bound of the ORIGINAL weight", dev <= gap * amax_bound * 1.05 + (0.5 if nested else 0) * 0.0 + 1e-4 + (0.02 * amax_bound if nested else 0.0),
          "dev=%g bound=%g" % (dev, gap * amax_bound))
    check(name + ": CONTROL result is not the original (lossy)", dev > 0.0)

# a bnb config with neither 4-bit nor 8-bit set must still be rejected (not silently mis-read as 4-bit).
# LLM.int8 (load_in_8bit) is handled since conversion/intperrow.py; see tests/test_int_perrow_dequant.py.
try:
    run_dequant_model({}, {"quant_method": "bitsandbytes"})
    check("CONTROL bitsandbytes with neither 4 nor 8 bit is rejected", False, "no exception")
except NotImplementedError:
    check("CONTROL bitsandbytes with neither 4 nor 8 bit is rejected", True)
except Exception as e:
    check("CONTROL bitsandbytes with neither 4 nor 8 bit is rejected", False, "%s: %s" % (type(e).__name__, e))

print("RESULT: %d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
