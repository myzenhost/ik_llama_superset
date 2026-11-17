# Conversion test for BitnetModel's FFN activation key. PROGRESS-OK. REWRITE-SAFE: writes only a temp dir.
# FAILURES-SURFACED: the one caught exception (ValueError for gelu) is the EXPECTED outcome, printed with its text; not refusing fails.
# Two HF classes convert to arch "bitnet": 1bitLLM's BitnetForCausalLM (model_type llama, hidden_act silu) and Microsoft's
# BitNetForCausalLM, bitnet-b1.58-2B-4T (model_type bitnet, hidden_act relu2). The runtime picks the FFN activation from
# bitnet.hidden_activation and treats a file without it as SiLU, so the converter must write the config's value. Builds a TINY
# random checkpoint per case, converts it in-process (tokenizer stubbed out), and reads the key back:
#   Microsoft shape -> "relu2"; 1bitLLM shape -> "silu"; an unsupported value -> ValueError.
# No case omits hidden_act: transformers fills the config class default (relu2 for BitNetConfig), which is what HF itself runs.
# CONTROL: the relu2 and silu files must differ in that key (a reader that saw nothing would report both the same).
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if __name__ == "__main__" and len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import gguf  # noqa: E402
from conversion import load_all_models  # noqa: E402
from conversion.base import ModelBase, ModelType  # noqa: E402

load_all_models()

H, NH, NKV, FF, V, NL = 32, 4, 2, 48, 64, 2
KEY = "bitnet.hidden_activation"
MICROSOFT = ("bitnet", "BitNetForCausalLM")
ONEBITLLM = ("llama", "BitnetForCausalLM")


def checkpoint(dirpath: Path, hidden_act, shape):
    model_type, arch = shape
    g = torch.Generator().manual_seed(7)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
    cfg = {"architectures": [arch], "model_type": model_type, "hidden_size": H, "num_hidden_layers": NL,
           "num_attention_heads": NH, "num_key_value_heads": NKV, "intermediate_size": FF, "vocab_size": V,
           "max_position_embeddings": 128, "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "tie_word_embeddings": True,
           "bos_token_id": 1, "eos_token_id": 2, "hidden_act": hidden_act}
    hd = H // NH
    t = {"model.embed_tokens.weight": r(V, H), "model.norm.weight": r(H)}
    for i in range(NL):
        p = "model.layers.%d." % i
        t.update({p + "input_layernorm.weight": r(H), p + "post_attention_layernorm.weight": r(H),
                  p + "self_attn.q_proj.weight": r(NH * hd, H), p + "self_attn.k_proj.weight": r(NKV * hd, H),
                  p + "self_attn.v_proj.weight": r(NKV * hd, H), p + "self_attn.o_proj.weight": r(H, NH * hd),
                  p + "self_attn.attn_sub_norm.weight": r(H), p + "mlp.gate_proj.weight": r(FF, H),
                  p + "mlp.up_proj.weight": r(FF, H), p + "mlp.down_proj.weight": r(H, FF), p + "mlp.ffn_sub_norm.weight": r(FF)})
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in t.items()}, str(dirpath / "model.safetensors"))


def convert(dirpath: Path, out: Path, shape):
    cls = ModelBase.from_model_architecture(shape[1], model_type=ModelType.TEXT)
    m = cls(dirpath, gguf.LlamaFileType.ALL_F32, out, is_big_endian=False, use_temp_file=False, eager=True)
    m.set_vocab = lambda: None   # the tokenizer is out of scope; the subject is the activation key
    m.write()


def key_of(path: Path):
    f = gguf.GGUFReader(str(path)).fields.get(KEY)
    return None if f is None else f.contents()


def run() -> bool:
    ok = True
    got = {}
    with tempfile.TemporaryDirectory(prefix="bitnet_act_") as td:
        td = Path(td)
        for case, shape in (("relu2", MICROSOFT), ("silu", ONEBITLLM)):
            d = td / case
            checkpoint(d, case, shape)
            convert(d, td / ("%s.gguf" % case), shape)
            got[case] = key_of(td / ("%s.gguf" % case))
            same = got[case] == case
            print("hidden_act=%-6s (%s) -> %s = %r  %s" % (case, shape[1], KEY, got[case], "OK" if same else "WRONG (want %r)" % case))
            ok &= same
        d = td / "gelu"
        checkpoint(d, "gelu", MICROSOFT)
        refused = False
        try:
            convert(d, td / "gelu.gguf", MICROSOFT)
        except ValueError as e:
            refused = True
            print("hidden_act=gelu   -> refused: %s" % e)
        if not refused:
            print("hidden_act=gelu   -> converted (WRONG: must be refused)")
        ok &= refused
        control = got["relu2"] != got["silu"]
        print("CONTROL relu2 and silu files differ in the key:", control, "(must be True)")
        ok &= control
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def test_bitnet_hidden_act_conversion():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
