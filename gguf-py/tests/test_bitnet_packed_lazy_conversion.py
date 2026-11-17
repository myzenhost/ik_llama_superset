# Conversion test for the packed-ternary BitNet checkpoints (quantization_config.quant_method == "bitnet"). PROGRESS-OK.
# REWRITE-SAFE: writes only a temp dir. FAILURES-SURFACED: a conversion exception is caught only to be PRINTED and
# counted as a failure (the pre-fix lazy path raised RuntimeError); nothing is swallowed.
# microsoft/bitnet-b1.58-2B-4T stores each ternary projection as U8 [out/4, in] (four 2-bit lanes per byte, lane i =
# rows [i*out/4, (i+1)*out/4)) plus a BF16 weight_scale. The converter unpacks it in dequant_bitnet.
# Two defects this guards:
#   - Lazy conversion is the default, and there the weight is a meta tensor; shifting it by a CPU shift tensor raised
#     "Tensor on device cpu is not on the expected device meta!" for every projection.
#   - The scale direction depends on quantization_config.linear_class. transformers' BitLinear divides its output by
#     weight_scale; AutoBitLinear in offline mode (the 2B-4T checkpoint) multiplies by it. The converter always divided.
# Builds a TINY random packed checkpoint per linear_class and checks:
#   1. lazy conversion succeeds;
#   2. lazy and eager outputs are bit-identical (CONTROL: eager never touched meta tensors, so it is the reference path);
#   3. the ternary sign pattern of every projection equals an independent numpy unpack of the packed bytes;
#   4. the magnitude: a projection must come out as exactly t*k (k = s for autobitlinear, 1/s for bitlinear), the
#      weight transformers multiplies with. The two classes give DIFFERENT expected values (CONTROL: the magnitude
#      check can tell the directions apart, because every s differs from 1). Third defect guarded here:
#      BitnetModel.weight_quant re-ternarised the dequantized tensor and returned t*k*mean|t|.
#   5. CONTROL for 4: a full-precision checkpoint (no quantization_config, the 1bitLLM case) must still be
#      re-ternarised, round(w/mean|w|).clamp(-1,1)*mean|w|, so skipping weight_quant is scoped to packed input.
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if __name__ == "__main__" and len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import gguf  # noqa: E402
from conversion import load_all_models  # noqa: E402
from conversion.base import ModelBase, ModelType  # noqa: E402

load_all_models()

H, NH, NKV, FF, V, NL = 32, 4, 2, 48, 64, 2
ARCH = "BitNetForCausalLM"
PROJ = {"self_attn.q_proj": (H, H), "self_attn.k_proj": (NKV * H // NH, H), "self_attn.v_proj": (NKV * H // NH, H),
        "self_attn.o_proj": (H, H), "mlp.gate_proj": (FF, H), "mlp.up_proj": (FF, H), "mlp.down_proj": (H, FF)}
GGUF_NAME = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
             "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def pack(t: np.ndarray) -> np.ndarray:
    # t: int8 ternary [out, in] -> uint8 [out/4, in], lane i = rows [i*R, (i+1)*R)
    r = t.shape[0] // 4
    out = np.zeros((r, t.shape[1]), dtype=np.uint8)
    for i in range(4):
        out |= ((t[i * r:(i + 1) * r] + 1).astype(np.uint8) << (2 * i))
    return out


def checkpoint(d: Path, linear_class: str) -> dict:
    rng = np.random.default_rng(11)
    g = torch.Generator().manual_seed(11)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32).to(torch.bfloat16)  # noqa: E731
    cfg = {"architectures": [ARCH], "model_type": "bitnet", "hidden_size": H, "num_hidden_layers": NL,
           "num_attention_heads": NH, "num_key_value_heads": NKV, "intermediate_size": FF, "vocab_size": V,
           "max_position_embeddings": 128, "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "tie_word_embeddings": True,
           "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "relu2",
           "quantization_config": {"quant_method": "bitnet", "linear_class": linear_class, "quantization_mode": "offline"}}
    t = {"model.embed_tokens.weight": r(V, H), "model.norm.weight": r(H)}
    ref = {}
    for i in range(NL):
        p = "model.layers.%d." % i
        t.update({p + "input_layernorm.weight": r(H), p + "post_attention_layernorm.weight": r(H),
                  p + "self_attn.attn_sub_norm.weight": r(H), p + "mlp.ffn_sub_norm.weight": r(FF)})
        for k, (o, n) in PROJ.items():
            tern = rng.integers(-1, 2, size=(o, n), dtype=np.int8)
            s = torch.tensor([1.5 + 2 * rng.random()], dtype=torch.bfloat16)   # 1.5..3.5, never 1: directions differ
            t[p + k + ".weight"] = torch.from_numpy(pack(tern))
            t[p + k + ".weight_scale"] = s
            ref["blk.%d.%s.weight" % (i, GGUF_NAME[k])] = (tern, float(s.float()[0]))
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in t.items()}, str(d / "model.safetensors"))
    return ref


def convert(d: Path, out: Path, eager: bool) -> str | None:
    try:
        cls = ModelBase.from_model_architecture(ARCH, model_type=ModelType.TEXT)
        m = cls(d, gguf.LlamaFileType.ALL_F32, out, is_big_endian=False, use_temp_file=False, eager=eager)
        m.set_vocab = lambda: None   # the tokenizer is out of scope; the subject is the packed weights
        m.write()
        return None
    except Exception as e:  # FAILURES-SURFACED: returned, printed and counted as a failure by the caller
        return "%s: %s" % (type(e).__name__, e)


def tensors(path: Path) -> dict:
    return {t.name: np.array(t.data) for t in gguf.GGUFReader(str(path)).tensors}


def run_class(td: Path, linear_class: str) -> bool:
    ok = True
    ref = checkpoint(td / linear_class, linear_class)
    res = {}
    for mode, eager in (("lazy", False), ("eager", True)):
        err = convert(td / linear_class, td / ("%s_%s.gguf" % (linear_class, mode)), eager)
        print("[%s] %-5s conversion: %s" % (linear_class, mode, "OK" if err is None else "FAILED -- " + err))
        ok &= err is None
        res[mode] = tensors(td / ("%s_%s.gguf" % (linear_class, mode))) if err is None else None
    if res["eager"] is None:
        print("[%s] eager reference path failed, nothing to compare" % linear_class)
        return False
    n_sign = n_mag = 0
    for name, (tern, s) in ref.items():
        got = res["eager"].get(name)
        if got is None or got.shape != tern.shape or not np.array_equal(np.sign(got).astype(np.int8), tern):
            print("[%s] sign pattern WRONG: %s %s" % (linear_class, name, None if got is None else got.shape))
            ok = False
            continue
        n_sign += 1
        k = s if linear_class == "autobitlinear" else 1.0 / s
        want = tern.astype(np.float64) * k
        if np.allclose(got, want, rtol=1e-5, atol=0):
            n_mag += 1
        else:
            nz = tern != 0
            print("[%s] magnitude WRONG: %s got |w| %.6g want %.6g (s=%.6g)" % (linear_class, name, float(np.abs(got[nz]).mean()),
                                                                              float(np.abs(want[nz]).mean()), s))
            ok = False
    print("[%s] sign pattern vs independent numpy unpack: %d / %d projections exact" % (linear_class, n_sign, len(ref)))
    print("[%s] magnitude t*%s: %d / %d projections within rtol 1e-5" % (linear_class, "s" if linear_class == "autobitlinear" else "(1/s)", n_mag, len(ref)))
    ok &= n_mag == len(ref)
    if res["lazy"] is not None:
        same = res["lazy"].keys() == res["eager"].keys() and all(np.array_equal(res["lazy"][k], res["eager"][k]) for k in res["eager"])
        print("[%s] CONTROL lazy == eager, every tensor bit-identical: %s" % (linear_class, same))
        ok &= same
    return ok


def run_fullprec(td: Path) -> bool:
    # 1bitLLM-style checkpoint: float projections, no quantization_config. weight_quant must still apply.
    d = td / "fullprec"
    g = torch.Generator().manual_seed(13)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)  # noqa: E731
    cfg = {"architectures": [ARCH], "model_type": "bitnet", "hidden_size": H, "num_hidden_layers": NL,
           "num_attention_heads": NH, "num_key_value_heads": NKV, "intermediate_size": FF, "vocab_size": V,
           "max_position_embeddings": 128, "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "tie_word_embeddings": True,
           "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu"}
    t = {"model.embed_tokens.weight": r(V, H), "model.norm.weight": r(H)}
    ref = {}
    for i in range(NL):
        p = "model.layers.%d." % i
        t.update({p + "input_layernorm.weight": r(H), p + "post_attention_layernorm.weight": r(H),
                  p + "self_attn.attn_sub_norm.weight": r(H), p + "mlp.ffn_sub_norm.weight": r(FF)})
        for k, (o, n) in PROJ.items():
            w = r(o, n)
            t[p + k + ".weight"] = w
            m = w.double().abs().mean().clamp(min=1e-5)
            ref["blk.%d.%s.weight" % (i, GGUF_NAME[k])] = ((w.double() / m).round().clamp(-1, 1) * m).numpy()
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in t.items()}, str(d / "model.safetensors"))
    err = convert(d, td / "fullprec.gguf", False)
    print("[fullprec] lazy  conversion: %s" % ("OK" if err is None else "FAILED -- " + err))
    if err is not None:
        return False
    got = tensors(td / "fullprec.gguf")
    n = sum(1 for name, want in ref.items() if name in got and np.allclose(got[name], want, rtol=1e-5, atol=1e-7))
    print("[fullprec] CONTROL re-ternarised round(w/mean|w|)*mean|w|: %d / %d projections" % (n, len(ref)))
    return n == len(ref)


def run() -> bool:
    with tempfile.TemporaryDirectory(prefix="bitnet_packed_") as td:
        ok = all([run_class(Path(td), lc) for lc in ("autobitlinear", "bitlinear")] + [run_fullprec(Path(td))])
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def test_bitnet_packed_conversion():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
