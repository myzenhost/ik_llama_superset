"""Inkling (S69 P4): gguf-py, the converter and the C++ loader agree on the architecture's names. PROGRESS-OK: a few regex scans, milliseconds.

Cross-language, because each side was typed separately and a one-character drift loads nothing:
  * every gguf tensor name gguf-py can write for MODEL_ARCH.INKLING equals a name in the C++ LLM_TENSOR_NAMES block for LLM_ARCH_INKLING (and back);
  * every inkling.* KV the converter writes equals an LLM_KV_INKLING_* entry of src/llama-arch.cpp (and back);
  * the HF checkpoint names the converter handles map to the gguf names the C++ loader asks for;
  * TensorNameMap no longer leaks one arch's block mappings into the next one built in the same process (fixed in the same commit).
Run:  python -m pytest gguf-py/tests/test_inkling_registration.py   (or python gguf-py/tests/test_inkling_registration.py [<gguf-py dir>])
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GGUF_PY = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else os.path.join(HERE, ".."))
sys.path.insert(0, GGUF_PY)
ROOT = os.path.normpath(os.path.join(GGUF_PY, ".."))

from gguf.constants import MODEL_ARCH, MODEL_ARCH_NAMES, MODEL_TENSORS, TENSOR_NAMES  # noqa: E402
from gguf.tensor_mapping import get_tensor_name_map  # noqa: E402


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _cpp_inkling_tensor_names():
    txt = _read("src/llama-model.cpp")
    i = txt.index("        LLM_ARCH_INKLING,\n")
    j = txt.index("    {\n        LLM_ARCH_MINIMAX_M2,", i)
    block = txt[i:j]
    names = re.findall(r'\{\s*LLM_TENSOR_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}', block)
    return {n.replace("%d", "{bid}") for n in names}


def test_control_cpp_block_is_found_and_not_empty():
    names = _cpp_inkling_tensor_names()
    assert len(names) >= 25, names
    assert "blk.{bid}.shortconv_k" in names and "token_embd" in names


def test_arch_name():
    assert MODEL_ARCH_NAMES[MODEL_ARCH.INKLING] == "inkling"
    assert re.search(r'\{\s*LLM_ARCH_INKLING\s*,\s*"inkling"\s*\}', _read("src/llama-arch.cpp"))


def test_tensor_names_agree_with_the_cpp_table():
    py = {TENSOR_NAMES[t] for t in MODEL_TENSORS[MODEL_ARCH.INKLING]}
    cpp = _cpp_inkling_tensor_names()
    assert py == cpp, ("only in gguf-py:", sorted(py - cpp), "only in C++:", sorted(cpp - py))


def test_kv_names_agree_with_the_converter_and_the_cpp_table():
    conv = _read("conversion/inkling.py")
    written = set(re.findall(r'f"\{arch\}\.([a-z_]+)"', conv))
    arch_cpp = _read("src/llama-arch.cpp")
    cpp = set(re.findall(r'LLM_KV_INKLING_[A-Z_]+\s*,\s*"%s\.([a-z_]+)"', arch_cpp))
    assert len(written) == 9 and len(cpp) == 9, (written, cpp)
    assert written == cpp, ("converter only:", sorted(written - cpp), "C++ only:", sorted(cpp - written))


def test_hf_checkpoint_names_map_to_what_the_loader_asks_for():
    m = get_tensor_name_map(MODEL_ARCH.INKLING, 2)
    want = {
        "model.embed_tokens.weight": "token_embd.weight", "model.embed_norm.weight": "token_embd_norm.weight",
        "model.norm.weight": "output_norm.weight", "model.unembed.weight": "output.weight",
        "model.layers.1.attn_norm.weight": "blk.1.attn_norm.weight", "model.layers.1.attn.wq_du.weight": "blk.1.attn_q.weight",
        "model.layers.1.attn.wk_dv.weight": "blk.1.attn_k.weight", "model.layers.1.attn.wv_dv.weight": "blk.1.attn_v.weight",
        "model.layers.1.attn.wr_du.weight": "blk.1.attn_r.weight", "model.layers.1.attn.wo_ud.weight": "blk.1.attn_output.weight",
        "model.layers.1.attn.q_norm.weight": "blk.1.attn_q_norm.weight", "model.layers.1.attn.k_norm.weight": "blk.1.attn_k_norm.weight",
        "model.layers.1.attn.rel_logits_proj.weight": "blk.1.attn_rel_proj.weight",
        "model.layers.1.attn.k_sconv.weight": "blk.1.shortconv_k.weight", "model.layers.1.attn.v_sconv.weight": "blk.1.shortconv_v.weight",
        "model.layers.1.attn_sconv.weight": "blk.1.shortconv_attn.weight", "model.layers.1.mlp_sconv.weight": "blk.1.shortconv_mlp.weight",
        "model.layers.1.mlp_norm.weight": "blk.1.ffn_norm.weight", "model.layers.1.mlp.w2_md.weight": "blk.1.ffn_down.weight",
        "model.layers.1.mlp.gate.weight": "blk.1.ffn_gate_inp.weight",
    }
    for hf, ggml_name in want.items():
        assert m.get_name(hf, try_suffixes=(".weight", ".bias")) == ggml_name, (hf, m.get_name(hf, try_suffixes=(".weight", ".bias")))
    # a name that must NOT map is reported as unmapped, not silently routed
    assert m.get_name("model.layers.1.attn.no_such_tensor.weight", try_suffixes=(".weight", ".bias")) is None


def test_name_map_does_not_leak_between_architectures():
    # build INKLING first (its overlay replaces FFN_NORM's/FFN_DOWN's HF names), then LLAMA: the LLAMA map must be what it is alone
    alone = get_tensor_name_map(MODEL_ARCH.LLAMA, 2).mapping
    get_tensor_name_map(MODEL_ARCH.INKLING, 2)
    after = get_tensor_name_map(MODEL_ARCH.LLAMA, 2).mapping
    assert after == alone
    assert "model.layers.1.mlp.w2_md" not in after, "an Inkling-only HF name leaked into the LLAMA map"
    # and the Inkling map still has its own spelling after another arch was built
    get_tensor_name_map(MODEL_ARCH.LLAMA, 2)
    assert get_tensor_name_map(MODEL_ARCH.INKLING, 2).get_name("model.layers.0.mlp.w2_md.weight", try_suffixes=(".weight",)) == "blk.0.ffn_down.weight"


def test_name_map_leak_fix_is_visible_with_an_architecture_the_parent_already_has():
    # same property with ARCTIC as the polluting arch, so this one can run against the parent commit's gguf-py (it must FAIL there)
    alone = get_tensor_name_map(MODEL_ARCH.LLAMA, 2).mapping
    get_tensor_name_map(MODEL_ARCH.ARCTIC, 2)
    after = get_tensor_name_map(MODEL_ARCH.LLAMA, 2).mapping
    assert "model.layers.1.residual_layernorm" not in after, "ARCTIC's block mapping leaked into LLAMA"
    assert after == alone


def test_converter_is_registered():
    txt = _read("conversion/__init__.py")
    assert '"InklingForConditionalGeneration": "inkling"' in txt
    assert os.path.exists(os.path.join(ROOT, "conversion", "inkling.py"))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok    ", name)
            except Exception as e:  # noqa: BLE001 -- report and keep going
                fails += 1
                print("FAIL  ", name, "-", type(e).__name__, str(e)[:300])
    print("RESULT:", "PASS" if fails == 0 else "FAIL (%d)" % fails)
    sys.exit(1 if fails else 0)
