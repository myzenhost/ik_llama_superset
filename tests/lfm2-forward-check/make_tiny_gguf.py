# Make TINY random LFM2 (dense, control) and LFM2-MoE checkpoints, convert both to F32 GGUF with the fork's own converter, and save the raw
# HF weights + config for an independent numpy reference. PROGRESS-OK: two tiny models, seconds. REWRITE-SAFE: writes only OUT.
# Usage: python make_tiny_gguf.py <worktree-root> <out-dir>
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import gguf  # noqa: E402
from conversion import load_all_models  # noqa: E402
from conversion.base import ModelBase, ModelType  # noqa: E402

load_all_models()

H, NH, NKV, HD = 32, 4, 2, 8
FF_DENSE, FF_EXP, NE, NUSED, V, L_CACHE = 48, 16, 4, 2, 64, 3
LAYER_TYPES = ["conv", "full_attention", "conv"]
g = torch.Generator().manual_seed(2026)


def r(*s, scale=0.15):
    return torch.randn(*s, generator=g, dtype=torch.float32) * scale


def norm_w(n):
    return 1.0 + 0.1 * torch.randn(n, generator=g, dtype=torch.float32)


def common_tensors(moe: bool):
    t = {"model.embed_tokens.weight": r(V, H, scale=0.5), "model.embedding_norm.weight": norm_w(H)}
    for i, lt in enumerate(LAYER_TYPES):
        p = f"model.layers.{i}"
        t[f"{p}.operator_norm.weight"] = norm_w(H)
        t[f"{p}.ffn_norm.weight"] = norm_w(H)
        if lt == "conv":
            t[f"{p}.conv.conv.weight"] = r(H, 1, L_CACHE, scale=0.4)
            t[f"{p}.conv.in_proj.weight"] = r(3 * H, H)
            t[f"{p}.conv.out_proj.weight"] = r(H, H)
        else:
            t[f"{p}.self_attn.q_proj.weight"] = r(NH * HD, H)
            t[f"{p}.self_attn.k_proj.weight"] = r(NKV * HD, H)
            t[f"{p}.self_attn.v_proj.weight"] = r(NKV * HD, H)
            t[f"{p}.self_attn.out_proj.weight"] = r(H, NH * HD)
            t[f"{p}.self_attn.q_layernorm.weight"] = norm_w(HD)
            t[f"{p}.self_attn.k_layernorm.weight"] = norm_w(HD)
        if moe and i >= 1:     # num_dense_layers = 1: layer 0 dense, layers 1.. routed
            t[f"{p}.feed_forward.gate.weight"] = r(NE, H, scale=0.6)
            t[f"{p}.feed_forward.expert_bias"] = r(NE, scale=0.3)
            for e in range(NE):
                t[f"{p}.feed_forward.experts.{e}.w1.weight"] = r(FF_EXP, H)
                t[f"{p}.feed_forward.experts.{e}.w3.weight"] = r(FF_EXP, H)
                t[f"{p}.feed_forward.experts.{e}.w2.weight"] = r(H, FF_EXP)
        else:
            t[f"{p}.feed_forward.w1.weight"] = r(FF_DENSE, H)
            t[f"{p}.feed_forward.w3.weight"] = r(FF_DENSE, H)
            t[f"{p}.feed_forward.w2.weight"] = r(H, FF_DENSE)
    return t


BASE_CFG = {
    "hidden_size": H, "num_hidden_layers": len(LAYER_TYPES), "num_attention_heads": NH, "num_key_value_heads": NKV,
    "layer_types": LAYER_TYPES, "conv_L_cache": L_CACHE, "conv_bias": False, "norm_eps": 1e-5, "vocab_size": V,
    "max_position_embeddings": 128, "rope_theta": 10000.0, "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
    "tie_word_embeddings": True,
}
CONFIGS = {
    "dense": dict(BASE_CFG, architectures=["Lfm2ForCausalLM"], model_type="lfm2", intermediate_size=FF_DENSE,
                  block_auto_adjust_ff_dim=False, block_ffn_dim_multiplier=None, block_multiple_of=1),
    "moe": dict(BASE_CFG, architectures=["Lfm2MoeForCausalLM"], model_type="lfm2_moe", intermediate_size=FF_DENSE,
                moe_intermediate_size=FF_EXP, num_experts=NE, num_experts_per_tok=NUSED, num_dense_layers=1,
                norm_topk_prob=True, routed_scaling_factor=1.5, use_expert_bias=True),
}
ARCH = {"dense": "Lfm2ForCausalLM", "moe": "Lfm2MoeForCausalLM", "moe2": "Lfm2MoeForCausalLM"}

CONFIGS["moe2"] = dict(CONFIGS["moe"], norm_topk_prob=False, routed_scaling_factor=2.5)
for kind, cfg in CONFIGS.items():
    d = OUT / f"ckpt_{kind}"
    d.mkdir(parents=True, exist_ok=True)
    tensors = common_tensors(kind.startswith("moe"))
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(d / "model.safetensors"))
    np.savez(OUT / f"weights_{kind}.npz", **{k.replace(".", "__"): v.numpy() for k, v in tensors.items()})
    (OUT / f"config_{kind}.json").write_text(json.dumps(cfg), encoding="utf-8")
    cls = ModelBase.from_model_architecture(ARCH[kind], model_type=ModelType.TEXT)
    m = cls(d, gguf.LlamaFileType.ALL_F32, OUT / f"{kind}.gguf", is_big_endian=False, use_temp_file=False, eager=True)
    m.set_vocab = m._set_vocab_none      # tensors and graph are the subject; the model takes token ids directly
    m.write()
    print(kind, "->", OUT / f"{kind}.gguf", "tensors:", len(tensors), flush=True)
