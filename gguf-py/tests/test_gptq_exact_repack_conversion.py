# End-to-end: a tiny random GPTQ int4 Qwen3 checkpoint converted with and without --gptq-exact-repack.
# REWRITE-SAFE: writes only a temp dir. FAILURES-SURFACED: a conversion exception is returned, printed and counted
# as a failure by the caller; nothing is swallowed.
#   1. with the flag every dense projection is written Q4_0 (all zeros 8) or Q4_1 and dequantises to the exact
#      GPTQ value scale*(q-zero) (independent numpy reference built when the checkpoint is made);
#   2. CONTROL (the path being replaced): without the flag the projection is float -> the file type given by the
#      outtype; quantising that to Q4_1 afterwards is visibly less exact than the repack;
#   3. an act-order checkpoint (g_idx permuted) keeps the float path (no repack, conversion still succeeds);
#   4. lazy and eager conversions are bit-identical.
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

H, NH, NKV, HD, FF, V, NL, GS = 64, 4, 2, 16, 96, 64, 1, 32
ARCH = "Qwen3ForCausalLM"
PROJ = {"self_attn.q_proj": (NH * HD, H), "self_attn.k_proj": (NKV * HD, H), "self_attn.v_proj": (NKV * HD, H),
        "self_attn.o_proj": (H, NH * HD), "mlp.gate_proj": (FF, H), "mlp.up_proj": (FF, H), "mlp.down_proj": (H, FF)}
GGUF_NAME = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
             "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def pack(q, z):
    k, n = q.shape
    qw = np.zeros((k // 8, n), dtype=np.uint32)
    for j in range(8):
        qw |= q[j::8, :].astype(np.uint32) << np.uint32(4 * j)
    zs = (z.astype(np.int32) - 1).astype(np.uint32)  # checkpoint_format gptq stores zero - 1
    qz = np.zeros((z.shape[0], n // 8), dtype=np.uint32)
    for j in range(8):
        qz |= zs[:, j::8] << np.uint32(4 * j)
    return torch.from_numpy(qw.view(np.int32)), torch.from_numpy(qz.view(np.int32))


def checkpoint(d: Path, act_order: bool, sym: bool) -> dict:
    rng = np.random.default_rng(5)
    g = torch.Generator().manual_seed(5)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32).to(torch.float16)  # noqa: E731
    cfg = {"architectures": [ARCH], "model_type": "qwen3", "hidden_size": H, "num_hidden_layers": NL,
           "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD, "intermediate_size": FF,
           "vocab_size": V, "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
           "tie_word_embeddings": True, "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu",
           "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": GS, "desc_act": act_order,
                                   "sym": sym, "checkpoint_format": "gptq"}}
    t = {"model.embed_tokens.weight": r(V, H), "model.norm.weight": r(H)}
    ref = {}
    for i in range(NL):
        p = "model.layers.%d." % i
        t.update({p + "input_layernorm.weight": r(H), p + "post_attention_layernorm.weight": r(H),
                  p + "self_attn.q_norm.weight": r(HD), p + "self_attn.k_norm.weight": r(HD)})
        for k, (o, n) in PROJ.items():
            q = rng.integers(2, 14, size=(n, o)).astype(np.uint8)  # [K=n, N=o], range narrower than 0..15
            ng = n // GS
            z = np.full((ng, o), 8, dtype=np.int16) if sym else rng.integers(3, 13, size=(ng, o)).astype(np.int16)
            s = (rng.random((ng, o)) * 0.02 + 0.003).astype(np.float16)
            gi = (np.arange(n) // GS).astype(np.int32)
            if act_order:
                gi = gi[rng.permutation(n)]
            qw, qz = pack(q, z)
            t[p + k + ".qweight"], t[p + k + ".qzeros"] = qw, qz
            t[p + k + ".scales"], t[p + k + ".g_idx"] = torch.from_numpy(s), torch.from_numpy(gi)
            ref["blk.%d.%s.weight" % (i, GGUF_NAME[k])] = (s.astype(np.float32)[gi] * (q.astype(np.float32) - z[gi].astype(np.float32))).T
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in t.items()}, str(d / "model.safetensors"))
    return ref


def convert(d: Path, out: Path, eager: bool, exact: bool):
    try:
        cls = ModelBase.from_model_architecture(ARCH, model_type=ModelType.TEXT)
        m = cls(d, gguf.LlamaFileType.ALL_F32, out, is_big_endian=False, use_temp_file=False, eager=eager,
                gptq_exact_repack=exact)
        m.set_vocab = lambda: None
        m.write()
        return None
    except Exception as e:  # FAILURES-SURFACED: returned, printed and counted by the caller
        return "%s: %s" % (type(e).__name__, e)


def tensors(path: Path) -> dict:
    return {t.name: (t.tensor_type, np.array(t.data)) for t in gguf.GGUFReader(str(path)).tensors}


def deq(tt, data):
    if tt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return data.astype(np.float32)
    return gguf.quants.dequantize(data, tt)


def run() -> bool:
    ok = True
    with tempfile.TemporaryDirectory(prefix="gptq_exact_") as td:
        td = Path(td)
        for sym in (False, True):
            tag = "sym" if sym else "asym"
            ref = checkpoint(td / tag, False, sym)
            res = {}
            for mode, eager in (("lazy", False), ("eager", True)):
                err = convert(td / tag, td / ("%s_%s.gguf" % (tag, mode)), eager, True)
                print("[%s] exact %-5s conversion: %s" % (tag, mode, "OK" if err is None else "FAILED -- " + err))
                ok &= err is None
                res[mode] = tensors(td / ("%s_%s.gguf" % (tag, mode))) if err is None else None
            if res["eager"] is None:
                ok = False
                continue
            want_t = gguf.GGMLQuantizationType.Q4_0 if sym else gguf.GGMLQuantizationType.Q4_1
            n_ok = 0
            for name, w in ref.items():
                tt, data = res["eager"][name]
                err_max = np.abs(deq(tt, data) - w).max()
                good = tt == want_t and err_max <= np.abs(w).max() * 2.0 ** -10
                n_ok += good
                if not good:
                    print("[%s] %s type %s max err %.3g (want %s, tol %.3g)" % (tag, name, tt.name, err_max, want_t.name, np.abs(w).max() * 2.0 ** -10))
            print("[%s] %d / %d projections exact %s" % (tag, n_ok, len(ref), want_t.name))
            ok &= n_ok == len(ref)
            if res["lazy"] is not None:
                same = res["lazy"].keys() == res["eager"].keys() and all(
                    res["lazy"][k][0] == res["eager"][k][0] and np.array_equal(res["lazy"][k][1], res["eager"][k][1]) for k in res["eager"])
                print("[%s] CONTROL lazy == eager bit-identical: %s" % (tag, same))
                ok &= same
            if not sym:
                # CONTROL: the replaced path (float conversion, then Q4_1 quantisation of the float tensor)
                err = convert(td / tag, td / (tag + "_float.gguf"), True, False)
                ok &= err is None
                if err is None:
                    flt = tensors(td / (tag + "_float.gguf"))
                    worse = 0
                    for name, w in ref.items():
                        tt, data = flt[name]
                        again = gguf.quants.dequantize(gguf.quants.quantize(deq(tt, data).astype(np.float32), gguf.GGMLQuantizationType.Q4_1),
                                                       gguf.GGMLQuantizationType.Q4_1)
                        exact_err = np.abs(deq(*res["eager"][name]) - w).max()
                        worse += np.abs(again - w).max() > exact_err * 4
                    print("[%s] CONTROL float->Q4_1 less exact than the repack in %d / %d projections" % (tag, worse, len(ref)))
                    ok &= worse == len(ref)
        # act-order: no repack, float path still converts
        ref = checkpoint(td / "act", True, False)
        err = convert(td / "act", td / "act.gguf", True, True)
        print("[act-order] conversion: %s" % ("OK" if err is None else "FAILED -- " + err))
        ok &= err is None
        if err is None:
            t = tensors(td / "act.gguf")
            kinds = {t[n][0].name for n in ref}
            print("[act-order] projection types (float path expected): %s" % sorted(kinds))
            ok &= kinds <= {"F32", "F16"}
            ok &= all(np.abs(deq(*t[n]) - w).max() <= np.abs(w).max() * 2.0 ** -10 for n, w in ref.items())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def test_gptq_exact_repack_conversion():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
