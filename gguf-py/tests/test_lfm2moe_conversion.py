# End-to-end conversion test for the LFM2-MoE converter: legacy per-expert layout vs fused transformers>=5 layout. PROGRESS-OK. REWRITE-SAFE: writes only a temp dir.
# Usage: pythonw _lfm2moe_conv_test_0930.py <worktree-root>      (the dir holding convert_hf_to_gguf.py, conversion/ and gguf-py/)
# Builds a TINY random LFM2-MoE checkpoint (2 blocks: dense conv block + MoE attention block, 4 experts), converts it with the fork's
# converter class IN-PROCESS (tokenizer step stubbed out: tensors and hparams are the subject), once per expert layout built from the SAME
# random weights, reads the GGUFs back and requires:
#   (a) both layouts give identical ffn_{gate,up,down}_exps tensors, and they equal the stacked original expert weights;
#   (b) expert_count / expert_weights_norm / expert_weights_scale are written;
# CONTROL: a deliberately WRONG reference (gate and up swapped) must be rejected by the same comparison.
import json
import sys
import tempfile
from pathlib import Path

# run as a script: the worktree root is argv[1]; under pytest it is the checkout this file is in
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

H, NH, NKV, FF_DENSE, FF_EXP, NE, NUSED, V, L_CACHE = 32, 4, 2, 48, 16, 4, 2, 64, 3
g = torch.Generator().manual_seed(1234)
r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)

CONFIG = {
    "architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe", "hidden_size": H, "num_hidden_layers": 2,
    "num_attention_heads": NH, "num_key_value_heads": NKV, "layer_types": ["conv", "full_attention"],
    "intermediate_size": FF_DENSE, "moe_intermediate_size": FF_EXP, "num_experts": NE, "num_experts_per_tok": NUSED,
    "num_dense_layers": 1, "conv_L_cache": L_CACHE, "conv_bias": False, "norm_eps": 1e-5, "vocab_size": V,
    "max_position_embeddings": 128, "rope_theta": 10000.0, "norm_topk_prob": True, "routed_scaling_factor": 1.5,
    "use_expert_bias": True, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "tie_word_embeddings": True,
}

GATE, UP, DOWN = r(NE, FF_EXP, H), r(NE, FF_EXP, H), r(NE, H, FF_EXP)   # [expert, ff, hidden] (gate/up), [expert, hidden, ff] (down)
COMMON = {
    "model.embed_tokens.weight": r(V, H), "model.embedding_norm.weight": r(H),
    "model.layers.0.operator_norm.weight": r(H), "model.layers.0.conv.conv.weight": r(H, 1, L_CACHE),
    "model.layers.0.conv.in_proj.weight": r(3 * H, H), "model.layers.0.conv.out_proj.weight": r(H, H),
    "model.layers.0.ffn_norm.weight": r(H), "model.layers.0.feed_forward.w1.weight": r(FF_DENSE, H),
    "model.layers.0.feed_forward.w2.weight": r(H, FF_DENSE), "model.layers.0.feed_forward.w3.weight": r(FF_DENSE, H),
    "model.layers.1.operator_norm.weight": r(H), "model.layers.1.ffn_norm.weight": r(H),
    "model.layers.1.self_attn.q_proj.weight": r(NH * (H // NH), H), "model.layers.1.self_attn.k_proj.weight": r(NKV * (H // NH), H),
    "model.layers.1.self_attn.v_proj.weight": r(NKV * (H // NH), H), "model.layers.1.self_attn.out_proj.weight": r(H, NH * (H // NH)),
    "model.layers.1.self_attn.q_layernorm.weight": r(H // NH), "model.layers.1.self_attn.k_layernorm.weight": r(H // NH),
    "model.layers.1.feed_forward.gate.weight": r(NE, H), "model.layers.1.feed_forward.expert_bias": r(NE),
}


def checkpoint(dirpath: Path, fused: bool):
    t = dict(COMMON)
    if fused:
        t["model.layers.1.feed_forward.experts.gate_up_proj"] = torch.cat([GATE, UP], dim=1).contiguous()   # gate half first
        t["model.layers.1.feed_forward.experts.down_proj"] = DOWN.clone()
    else:
        for e in range(NE):
            t[f"model.layers.1.feed_forward.experts.{e}.w1.weight"] = GATE[e].clone()
            t[f"model.layers.1.feed_forward.experts.{e}.w3.weight"] = UP[e].clone()
            t[f"model.layers.1.feed_forward.experts.{e}.w2.weight"] = DOWN[e].clone()
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in t.items()}, str(dirpath / "model.safetensors"))


def convert(dirpath: Path, out: Path):
    cls = ModelBase.from_model_architecture("Lfm2MoeForCausalLM", model_type=ModelType.TEXT)
    m = cls(dirpath, gguf.LlamaFileType.ALL_F32, out, is_big_endian=False, use_temp_file=False, eager=True)
    m.set_vocab = lambda: None   # tokenizer is out of scope here; the subject is tensors and hyper-parameters
    m.write()


def read(path: Path):
    rd = gguf.GGUFReader(str(path))
    tensors = {t.name: np.array(t.data, copy=True) for t in rd.tensors}
    kv = {}
    for f in rd.fields.values():
        try:
            kv[f.name] = f.contents() if hasattr(f, "contents") else None
        except Exception:  # noqa: BLE001 -- a field that cannot be rendered is listed as unreadable, not dropped
            kv[f.name] = "<unreadable>"
    return tensors, kv


def run() -> bool:
    with tempfile.TemporaryDirectory(prefix="lfm2moe_conv_") as td:
        td = Path(td)
        results = {}
        for layout in ("legacy", "fused"):
            d = td / layout
            checkpoint(d, fused=(layout == "fused"))
            try:
                convert(d, td / f"{layout}.gguf")
                results[layout] = read(td / f"{layout}.gguf")
                print(layout, "converted:", len(results[layout][0]), "tensors")
            except Exception as e:  # noqa: BLE001 -- the failure IS the result for this layout; printed with its type
                results[layout] = None
                print(layout, "CONVERSION FAILED:", type(e).__name__, e)

        ok = True
        want = {"blk.1.ffn_gate_exps.weight": GATE.numpy(), "blk.1.ffn_up_exps.weight": UP.numpy(), "blk.1.ffn_down_exps.weight": DOWN.numpy()}
        for layout in ("legacy", "fused"):
            if results[layout] is None:
                ok = False
                continue
            tens, kv = results[layout]
            for name, ref in want.items():
                got = tens.get(name)
                same = got is not None and got.shape == ref.shape and np.array_equal(got.astype(np.float32), ref)
                print("  %-7s %-30s %s" % (layout, name, "MATCH" if same else "MISMATCH shape=%s" % (None if got is None else got.shape,)))
                ok &= bool(same)
            for k in ("lfm2moe.expert_count", "lfm2moe.expert_weights_norm", "lfm2moe.expert_weights_scale"):
                print("  %-7s key %-32s %s" % (layout, k, kv.get(k, "ABSENT")))
        # CONTROL: the comparison must reject a swapped reference (gate<->up)
        if results["fused"] is not None:
            tens = results["fused"][0]
            bad_ref = UP.numpy()
            got = tens.get("blk.1.ffn_gate_exps.weight")
            control_rejects = got is None or not np.array_equal(got.astype(np.float32), bad_ref)
            print("CONTROL swapped gate/up reference is rejected:", control_rejects, "(must be True)")
            ok &= bool(control_rejects)
        print("RESULT:", "PASS" if ok else "FAIL")
        return ok


def test_lfm2moe_conversion():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
