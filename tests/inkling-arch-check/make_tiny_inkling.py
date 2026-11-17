# Make TINY random-weight Inkling checkpoints (HF layout), convert them with this tree's own converter, and write what the
# converter must have produced so check_inkling_gguf.py / test-inkling-arch-load can compare against it.
# PROGRESS-OK: four tiny conversions, seconds. REWRITE-SAFE: writes only OUT.  RANDOM WEIGHTS ONLY: the real Inkling weights are never read.
#
# Usage: python make_tiny_inkling.py <worktree-root> <out-dir>
# Writes <out-dir>/{hf_f32,ckpt_f32,hf_f16}.gguf, <out-dir>/expected.json and the HF checkpoint dirs <out-dir>/ckpt_{hf,ckpt}/.
#
# Two config spellings are converted from the SAME weights and must give the same GGUF metadata:
#   hf   -- transformers' InklingTextConfig spelling (nested text_config, intermediate_size = DENSE ff, moe_intermediate_size,
#           n_routed_experts, conv_kernel_size, mlp_layer_types, layer_types)
#   ckpt -- the original-checkpoint spelling the converter normalises from (dense_intermediate_size, intermediate_size = MoE ff,
#           num_experts, sconv_kernel_size, dense_mlp_idx, local_layer_ids)
# SOURCE-READ: upstream/hf-transformers-inkling-raw/configuration_inkling.py (Apache-2.0, config names/aliases);
#              upstream/unsloth-llama-inkling-raw/conversion__inkling.py (MIT, tensor names + layouts);
#              upstream/hf-transformers-inkling-raw/src__transformers__conversion_mapping.py:179-252 (Apache-2.0) is the
#              INDEPENDENT oracle for the ORIGINAL checkpoint names used below (model.llm.embed.weight, attn.wq_du, mlp.w13_dn, ...), and
#              src__transformers__core_model_loading.py:195-219 (Interleave) shows w13's gate rows are the EVEN rows (0::2), up the odd.
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
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

import gguf  # noqa: E402
from conversion import load_all_models  # noqa: E402
from conversion.base import ModelBase, ModelType  # noqa: E402

load_all_models()

# ---- tiny shape (all matrix input dims are multiples of 256 so IQ2_K / IQ4_K super-blocks are genuinely exercised) ----
H, NH, HD, D_REL, K = 256, 4, 64, 16, 4
NKV_GLOBAL, NKV_LOCAL = 2, 4              # different kv widths on global and local layers
N_LAYER = 4
LOCAL_IDS = [0, 1, 3]                      # layer 2 is global
DENSE_BLOCKS = 1                           # layer 0 dense, layers 1..3 MoE
FF_DENSE, FF_EXP, NE, NSH, NUSED = 512, 256, 2, 1, 1
WINDOW, REL_EXTENT = 24, 48                # sliding_window_size (also rel_extent_swa) / global rel_extent
V_PADDED, V_UNPADDED = 320, 262            # 256 bytes + 2 merges + 4 specials = 262 real tokens, 58 padding rows
EOS_ID = 261                               # <|content_model_end_sampling|>, the last real token
MUP = 8.0
ROUTE_SCALE = 2.5

g = torch.Generator().manual_seed(20261001)


def r(*s, scale=0.1):
    return torch.randn(*s, generator=g, dtype=torch.float32) * scale


def norm_w(n):
    return 1.0 + 0.1 * torch.randn(n, generator=g, dtype=torch.float32)


def is_local(il):
    return il in LOCAL_IDS


def tensors():
    p0 = "model.llm."
    t = {
        p0 + "embed.weight": r(V_PADDED, H, scale=0.5),
        p0 + "embed_norm.weight": norm_w(H),
        p0 + "norm.weight": norm_w(H),
        p0 + "unembed.weight": r(V_PADDED, H, scale=0.5),
    }
    for il in range(N_LAYER):
        p = f"{p0}layers.{il}."
        nkv = NKV_LOCAL if is_local(il) else NKV_GLOBAL
        kvw = nkv * HD
        rel = WINDOW if is_local(il) else REL_EXTENT
        t[p + "attn_norm.weight"] = norm_w(H)
        t[p + "attn.wq_du.weight"] = r(NH * HD, H)
        t[p + "attn.wk_dv.weight"] = r(kvw, H)
        t[p + "attn.wv_dv.weight"] = r(kvw, H)
        t[p + "attn.wr_du.weight"] = r(NH * D_REL, H)
        t[p + "attn.wo_ud.weight"] = r(H, NH * HD)
        t[p + "attn.q_norm.weight"] = norm_w(HD)
        t[p + "attn.k_norm.weight"] = norm_w(HD)
        t[p + "attn.rel_logits_proj.proj"] = r(D_REL, rel)            # no ".weight" suffix in the checkpoint
        t[p + "attn.k_sconv.weight"] = r(kvw, 1, K, scale=0.4)
        t[p + "attn.v_sconv.weight"] = r(kvw, 1, K, scale=0.4)
        t[p + "attn_sconv.weight"] = r(H, 1, K, scale=0.4)
        t[p + "mlp_sconv.weight"] = r(H, 1, K, scale=0.4)
        t[p + "mlp_norm.weight"] = norm_w(H)
        if il < DENSE_BLOCKS:
            t[p + "mlp.w13_dn.weight"] = r(2 * FF_DENSE, H)            # gate/up interleaved along rows
            t[p + "mlp.w2_md.weight"] = r(H, FF_DENSE)
            t[p + "mlp.global_scale"] = torch.tensor([1.0 + 0.1 * il])
        else:
            t[p + "mlp.gate.weight"] = r(NE + NSH, H, scale=0.6)      # routed rows then shared-expert sink rows
            t[p + "mlp.gate.bias"] = r(NE, scale=0.3)
            t[p + "mlp.gate.global_scale"] = torch.tensor([1.0 + 0.1 * il])
            t[p + "mlp.experts.w13_weight"] = r(NE, 2 * FF_EXP, H)
            t[p + "mlp.experts.w2_weight"] = r(NE, H, FF_EXP)
            t[p + "mlp.shared_experts.shared_w13_weight"] = r(NSH, 2 * FF_EXP, H)
            t[p + "mlp.shared_experts.shared_w2_weight"] = r(NSH, H, FF_EXP)
    return t


def make_tokenizer(d: Path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}                    # ids 0..255
    vocab["\u0120t"] = 256
    vocab["th"] = 257
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[("\u0120", "t"), ("t", "h")]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|bos_unused|>", "<|image_bos|>", "<|end_message|>", "<|content_model_end_sampling|>"]   # ids 258..261
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])
    assert tok.get_vocab_size() == V_UNPADDED, tok.get_vocab_size()
    assert tok.token_to_id("<|content_model_end_sampling|>") == EOS_ID
    tok.save(str(d / "tokenizer.json"))
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 256,
        "added_tokens_decoder": {
            str(tok.token_to_id(s)): {"content": s, "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False}
            for s in specials
        },
    }
    (d / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")


LAYER_TYPES = ["hybrid_sliding" if is_local(i) else "hybrid" for i in range(N_LAYER)]
MLP_TYPES = ["dense" if i < DENSE_BLOCKS else "sparse" for i in range(N_LAYER)]
COMMON = {
    "vocab_size": V_PADDED, "unpadded_vocab_size": V_UNPADDED, "hidden_size": H, "num_hidden_layers": N_LAYER,
    "num_attention_heads": NH, "num_key_value_heads": NKV_GLOBAL, "head_dim": HD,
    "swa_num_attention_heads": NH, "swa_num_key_value_heads": NKV_LOCAL, "swa_head_dim": HD,
    "sliding_window_size": WINDOW, "d_rel": D_REL, "rel_extent": REL_EXTENT,
    "log_scaling_n_floor": 32, "log_scaling_alpha": 0.1, "local_layer_ids": LOCAL_IDS,
    "rms_norm_eps": 1e-6, "num_experts_per_tok": NUSED, "n_shared_experts": NSH,
    "route_scale": ROUTE_SCALE, "logits_mup_width_multiplier": MUP, "bos_token_id": 1, "eos_token_id": EOS_ID,
}
CONFIGS = {
    "hf": {"architectures": ["InklingForConditionalGeneration"], "model_type": "inkling_mm_model",
           "text_config": dict(COMMON, model_type="inkling_text", layer_types=LAYER_TYPES, mlp_layer_types=MLP_TYPES,
                               max_position_embeddings=256, conv_kernel_size=K, intermediate_size=FF_DENSE,
                               moe_intermediate_size=FF_EXP, n_routed_experts=NE)},
    "ckpt": dict(COMMON, architectures=["InklingForConditionalGeneration"], model_type="inkling",
                 model_max_length=256, sconv_kernel_size=K, dense_mlp_idx=DENSE_BLOCKS,
                 dense_intermediate_size=FF_DENSE, intermediate_size=FF_EXP, num_experts=NE),
}


def convert(kind: str, ftype, out: Path):
    d = OUT / f"ckpt_{kind}"
    cls = ModelBase.from_model_architecture("InklingForConditionalGeneration", model_type=ModelType.TEXT)
    m = cls(d, ftype, out, is_big_endian=False, use_temp_file=False, eager=True)
    m.write()


W = tensors()
np.savez(OUT / "weights.npz", **{k.replace(".", "__"): v.numpy() for k, v in W.items()})
for kind, cfg in CONFIGS.items():
    d = OUT / f"ckpt_{kind}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    make_tokenizer(d)
    save_file({k: v.contiguous() for k, v in W.items()}, str(d / "model.safetensors"))

convert("hf", gguf.LlamaFileType.ALL_F32, OUT / "hf_f32.gguf")
convert("ckpt", gguf.LlamaFileType.ALL_F32, OUT / "ckpt_f32.gguf")
convert("hf", gguf.LlamaFileType.MOSTLY_F16, OUT / "hf_f16.gguf")

# ---- what the converter must have produced: {gguf tensor name: shape in numpy order (= gguf ne reversed)} ----
exp = {"token_embd.weight": [V_PADDED, H], "token_embd_norm.weight": [H], "output_norm.weight": [H], "output.weight": [V_PADDED, H]}
for il in range(N_LAYER):
    b = f"blk.{il}."
    nkv = NKV_LOCAL if is_local(il) else NKV_GLOBAL
    kvw = nkv * HD
    rel = WINDOW if is_local(il) else REL_EXTENT
    exp.update({
        b + "attn_norm.weight": [H], b + "attn_q.weight": [NH * HD, H], b + "attn_k.weight": [kvw, H], b + "attn_v.weight": [kvw, H],
        b + "attn_r.weight": [NH * D_REL, H], b + "attn_output.weight": [H, NH * HD],
        b + "attn_q_norm.weight": [HD], b + "attn_k_norm.weight": [HD], b + "attn_rel_proj.weight": [D_REL, rel],
        b + "shortconv_k.weight": [kvw, K], b + "shortconv_v.weight": [kvw, K],
        b + "shortconv_attn.weight": [H, K], b + "shortconv_mlp.weight": [H, K],
        b + "ffn_norm.weight": [H], b + "ffn_gscale.weight": [1],
    })
    if il < DENSE_BLOCKS:
        exp.update({b + "ffn_gate.weight": [FF_DENSE, H], b + "ffn_up.weight": [FF_DENSE, H], b + "ffn_down.weight": [H, FF_DENSE]})
    else:
        exp.update({
            b + "ffn_gate_inp.weight": [NE + NSH, H], b + "exp_probs_b.bias": [NE],
            b + "ffn_gate_exps.weight": [NE, FF_EXP, H], b + "ffn_up_exps.weight": [NE, FF_EXP, H], b + "ffn_down_exps.weight": [NE, H, FF_EXP],
            b + "ffn_gate_shexp.weight": [NSH, FF_EXP, H], b + "ffn_up_shexp.weight": [NSH, FF_EXP, H], b + "ffn_down_shexp.weight": [NSH, H, FF_EXP],
        })
meta = {
    "tensors": exp,
    "kv": {
        "general.architecture": "inkling", "inkling.block_count": N_LAYER, "inkling.embedding_length": H, "inkling.vocab_size": V_PADDED,
        "inkling.d_rel": D_REL, "inkling.rel_extent": REL_EXTENT, "inkling.rel_extent_swa": WINDOW, "inkling.shortconv_kernel": K,
        "inkling.dense_block_count": DENSE_BLOCKS, "inkling.logit_scale_denom": MUP, "inkling.log_scaling_n_floor": 32,
        "inkling.unpadded_vocab_size": V_UNPADDED, "inkling.expert_count": NE, "inkling.expert_used_count": NUSED,
        "inkling.expert_shared_count": NSH, "inkling.expert_feed_forward_length": FF_EXP, "inkling.feed_forward_length": FF_DENSE,
        "inkling.attention.sliding_window": WINDOW, "inkling.expert_weights_scale": ROUTE_SCALE,
        "tokenizer.ggml.pre": "inkling", "tokenizer.ggml.eos_token_id": EOS_ID,
    },
    "n_layer": N_LAYER, "local_ids": LOCAL_IDS, "n_head_kv": [NKV_LOCAL if is_local(i) else NKV_GLOBAL for i in range(N_LAYER)],
    "swa_pattern": [is_local(i) for i in range(N_LAYER)], "eos_id": EOS_ID, "v_padded": V_PADDED, "v_unpadded": V_UNPADDED,
    "dense_blocks": DENSE_BLOCKS, "n_expert": NE, "n_expert_shared": NSH, "n_embd": H, "d_rel": D_REL, "kernel": K,
    "rel_extent": REL_EXTENT, "rel_extent_swa": WINDOW, "head_dim": HD, "n_head": NH, "ff_dense": FF_DENSE, "ff_exp": FF_EXP,
}
(OUT / "expected.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("wrote", OUT, "expected tensors:", len(exp), flush=True)
