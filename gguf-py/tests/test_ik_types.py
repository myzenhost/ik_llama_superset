"""gguf-py knows the ik-main quant types and the DeepSeek-V4.1 architecture (ik-main merge 2026-09-29, rows S69-IKMERGE-GGUFPY-CONVERTER-DELTAS-NOT-PORTED-*).

Every expected number below comes from the C side, not from this package:
  * ggml.h / llama.h enum values are PARSED from the tree (not typed here);
  * 2180 = ggml_row_size(IQ4_KS_R16, 4096), printed by tests/test-iqk-golden ("ggml_row_size(type,4096)=2180");
  * 2240 = 16 * ggml_row_size(IQ4_KS_R16, 256), the bytes that test says the r16 quantizer WROTE for 4096 floats.
Run:  python -m pytest gguf-py/tests/test_ik_types.py   (or python gguf-py/tests/test_ik_types.py)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# optional argv[1] / IK_GGUF_PY = a gguf-py directory to test instead of this one (used to prove the test FAILS on the unported tree)
GGUF_PY = os.path.abspath(os.environ.get("IK_GGUF_PY") or (sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else os.path.join(HERE, "..")))
sys.path.insert(0, GGUF_PY)
ROOT = os.path.normpath(os.path.join(GGUF_PY, ".."))

import gguf  # noqa: E402
from gguf.constants import (  # noqa: E402
    GGML_QUANT_SIZES, GGML_ROW_META_SIZES, GGMLQuantizationType as T, LlamaFileType as F,
    MODEL_ARCH, MODEL_ARCH_NAMES, MODEL_TENSOR, MODEL_TENSORS, MODEL_TENSOR_SKIP, TENSOR_NAMES,
)
from gguf.quants import quant_row_bytes, quant_shape_from_byte_shape, quant_shape_to_byte_shape  # noqa: E402


def _c_enum(path, prefix):
    """{NAME: value} for every `prefix NAME = <int>` in a C header (comments stripped)."""
    txt = open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"^\s*" + prefix + r"([A-Z0-9_]+)\s*=\s*(\d+)", txt, re.M):
        out[m.group(1)] = int(m.group(2))
    return out


C_TYPES = _c_enum("ggml/include/ggml.h", "GGML_TYPE_")
C_FTYPES = _c_enum("include/llama.h", "LLAMA_FTYPE_MOSTLY_")


def test_control_c_parse_reaches_the_headers():
    # positive controls: known-present entries must parse with their known values; an invented name must be absent
    assert C_TYPES["Q8_0"] == 8 and C_TYPES["IQ4_KS_R4"] == 344, C_TYPES.get("IQ4_KS_R4")
    assert "ZZ_INVENTED_TYPE" not in C_TYPES
    assert len(C_TYPES) > 60 and len(C_FTYPES) > 40, (len(C_TYPES), len(C_FTYPES))


def test_new_quant_type_ids_match_the_c_headers():
    assert C_TYPES["IQ4_KS_R16"] == 345 and C_TYPES["Q1_0_G128_R8"] == 159
    assert int(T.IQ4_KS_R16) == C_TYPES["IQ4_KS_R16"]
    assert int(T.Q1_0_G128_R8) == C_TYPES["Q1_0_G128_R8"]
    assert int(F.MOSTLY_IQ4_KS_R16) == C_FTYPES["IQ4_KS_R16"] == 346
    assert int(F.MOSTLY_Q1_0_G128_R8) == C_FTYPES["Q1_0_G128_R8"] == 160


def test_no_quant_type_id_collides_and_every_shared_name_matches_c():
    ids = [int(t) for t in T]
    assert len(ids) == len(set(ids)), "two GGMLQuantizationType names share a value"
    mismatched = {}
    for t in T:
        if t.name in C_TYPES and int(t) != C_TYPES[t.name]:
            mismatched[t.name] = (int(t), C_TYPES[t.name])
    # the fork renumbers Q1_0/Q2_0 on purpose (constants.py comments); nothing else may differ
    assert set(mismatched) <= {"Q1_0", "Q2_0"}, mismatched


def test_row_bytes_match_the_c_row_size():
    # IQ4_KS_R16: block 32, type_size 17, row_meta 4 -> 4 + 4096/32*17 = 2180 (ggml_row_size printed by test-iqk-golden)
    assert GGML_QUANT_SIZES[T.IQ4_KS_R16] == (32, 17) and GGML_ROW_META_SIZES[T.IQ4_KS_R16] == 4
    assert quant_row_bytes(4096, T.IQ4_KS_R16) == 2180
    # the bytes the r16 quantizer writes for 4096 floats = 16 rows of 256 (measured: 2240)
    assert 16 * quant_row_bytes(256, T.IQ4_KS_R16) == 2240
    # Q1_0_G128_R8: block 128, type_size (8*2 + 128)/8 = 18, no row meta
    assert GGML_QUANT_SIZES[T.Q1_0_G128_R8] == (128, 18) and T.Q1_0_G128_R8 not in GGML_ROW_META_SIZES
    assert quant_row_bytes(512, T.Q1_0_G128_R8) == 4 * 18
    # control: an existing type still gives its known value (Q8_0: 34 bytes per 32)
    assert quant_row_bytes(4096, T.Q8_0) == 4096 // 32 * 34


def test_shape_round_trip():
    for t, n in ((T.IQ4_KS_R16, 4096), (T.Q1_0_G128_R8, 1024), (T.IQ4_KS_R4, 2048), (T.Q8_0, 1024)):
        b = quant_shape_to_byte_shape((7, n), t)
        assert quant_shape_from_byte_shape(b, t) == (7, n), (t, b)
    # control: a row size that is not a multiple of the block must be refused, not rounded
    try:
        quant_shape_to_byte_shape((1, 100), T.IQ4_KS_R16)
    except ValueError:
        pass
    else:
        raise AssertionError("100 is not a multiple of 32 and must raise")


def test_deepseek41_arch_and_engram_tensors():
    assert MODEL_ARCH_NAMES[MODEL_ARCH.DEEPSEEK41] == "deepseek41"
    # names are what src/llama-load-tensors.cpp asks for (blk.%d.engram_embd.weight etc.)
    assert TENSOR_NAMES[MODEL_TENSOR.ENGRAM_EMBD].format(bid=3) == "blk.3.engram_embd"
    assert TENSOR_NAMES[MODEL_TENSOR.ENGRAM_WKV].format(bid=3) == "blk.3.engram_wkv"
    assert TENSOR_NAMES[MODEL_TENSOR.ENGRAM_K].format(bid=3) == "blk.3.engram_k"
    assert TENSOR_NAMES[MODEL_TENSOR.ENGRAM_Q].format(bid=3) == "blk.3.engram_q"
    assert TENSOR_NAMES[MODEL_TENSOR.FFN_EXP_PROBS_B_VL].format(bid=3) == "blk.3.exp_probs_b_vl"
    for t in (MODEL_TENSOR.ENGRAM_EMBD, MODEL_TENSOR.ENGRAM_K, MODEL_TENSOR.ENGRAM_Q, MODEL_TENSOR.ENGRAM_WKV, MODEL_TENSOR.FFN_EXP_PROBS_B_VL):
        assert t in MODEL_TENSORS[MODEL_ARCH.DEEPSEEK41] and t in MODEL_TENSORS[MODEL_ARCH.DEEPSEEK4]
    assert MODEL_TENSORS[MODEL_ARCH.DEEPSEEK41] is MODEL_TENSORS[MODEL_ARCH.DEEPSEEK4]
    assert MODEL_TENSOR_SKIP[MODEL_ARCH.DEEPSEEK41] == [MODEL_TENSOR.ROPE_FREQS, MODEL_TENSOR.ATTN_ROT_EMBD]
    # control: the plain arch must not gain the new name by accident
    assert MODEL_ARCH_NAMES[MODEL_ARCH.DEEPSEEK4] == "deepseek4"


def test_engram_tensor_names_exist_in_the_c_side():
    # the name table is in llama-model.cpp; exp_probs_b_vl is only loaded by name in llama-load-tensors.cpp,
    # as "blk.%d.exp_probs_b_vl.bias", so accept the name followed by either the closing quote or a suffix dot
    src = "".join(open(os.path.join(ROOT, "src", f), encoding="utf-8", errors="replace").read()
                  for f in ("llama-model.cpp", "llama-load-tensors.cpp"))
    for name in ("blk.%d.engram_embd", "blk.%d.engram_k", "blk.%d.engram_q", "blk.%d.engram_wkv", "blk.%d.exp_probs_b_vl"):
        assert re.search('"' + re.escape(name) + '[".]', src), name
    assert not re.search('"' + re.escape("blk.%d.zz_invented_tensor") + '[".]', src)  # control


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001 -- a failing test is REPORTED and counted, never swallowed
                fails += 1
                print("FAIL", name, repr(e)[:300])
    print("%d failed" % fails)
    sys.exit(1 if fails else 0)
