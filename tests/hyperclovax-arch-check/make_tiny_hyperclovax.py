# Tiny random-weight HyperCLOVA X text checkpoint, converted with this tree's own converter (conversion/llama.py HyperCLOVAXModel), plus
# reference logits from an independent plain-torch statement of the model. test-inkling-parity (arch-agnostic) compares the fork with them.
# Row S69-EXLLAMAV3-PICK-HYPERCLOVAX-2026-10-20-01.
# PROGRESS-OK: a 4-layer 256-wide model, seconds.  REWRITE-SAFE: writes only OUT.  RANDOM WEIGHTS ONLY.
#
# Usage: python make_tiny_hyperclovax.py <worktree-root> <out-dir> [mutation ...]
#   no mutation   -> writes the HF checkpoint, model_f32.gguf, ref_tokens.txt and ref_logits.bin (the faithful reference)
#   mutation(s)   -> ALSO writes ref_logits.mut_<name>.bin: the same weights run through a reference with ONE semantic changed. These are the
#                    negative controls: the parity test must FAIL against each of them, or "within tolerance" would not tell a right graph from
#                    a wrong one. Names: no_post_norm, logit_div, no_residual_mult, default_attn_scale, no_embed_mult
#
# The reference below is written from the HF semantics (transformers HyperCLOVAXDecoderLayer / HyperCLOVAXModel / HyperCLOVAXForCausalLM,
# modular_hyperclovax.py, Apache-2.0), not from the fork's graph: x = embed*embedding_multiplier; per layer
#   x = x + residual_multiplier * post_norm1(attn(input_layernorm(x)));  x = x + residual_multiplier * post_norm2(mlp(post_attention_layernorm(x)))
# attention scores scaled by attention_multiplier (NOT 1/sqrt(head_dim)); logits = lm_head(norm(x)) * logits_scaling (a multiply).
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
MUTATIONS = sys.argv[3:]
N_TOKENS = 40
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

H, NH, NKV, HD = 256, 8, 2, 32
N_LAYER, FF = 4, 512
V = 262
EOS = 261
EPS = 1e-6
THETA = 10000.0
EMB_MULT, RES_MULT, ATTN_MULT, LOGITS_SCALING = 3.0, 0.7, 0.1, 2.5   # deliberately none of them the identity / the default

torch.manual_seed(20261020)
W = {}


def lin(o, i):
    return torch.randn(o, i) * 0.08


def norm_w(n):
    return 1.0 + 0.1 * torch.randn(n)


W["model.embed_tokens.weight"] = torch.randn(V, H) * 0.3
for il in range(N_LAYER):
    p = f"model.layers.{il}."
    W[p + "input_layernorm.weight"] = norm_w(H)
    W[p + "self_attn.q_proj.weight"] = lin(NH * HD, H)
    W[p + "self_attn.k_proj.weight"] = lin(NKV * HD, H)
    W[p + "self_attn.v_proj.weight"] = lin(NKV * HD, H)
    W[p + "self_attn.o_proj.weight"] = lin(H, NH * HD)
    W[p + "post_norm1.weight"] = norm_w(H)
    W[p + "post_attention_layernorm.weight"] = norm_w(H)
    W[p + "mlp.gate_proj.weight"] = lin(FF, H)
    W[p + "mlp.up_proj.weight"] = lin(FF, H)
    W[p + "mlp.down_proj.weight"] = lin(H, FF)
    W[p + "post_norm2.weight"] = norm_w(H)
W["model.norm.weight"] = norm_w(H)
W["lm_head.weight"] = lin(V, H)


def rms(x, w):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS))


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def reference(ids, mut=()):
    T = ids.shape[0]
    emb_mult = 1.0 if "no_embed_mult" in mut else EMB_MULT
    res = 1.0 if "no_residual_mult" in mut else RES_MULT
    attn_scale = HD ** -0.5 if "default_attn_scale" in mut else ATTN_MULT
    x = W["model.embed_tokens.weight"][ids] * emb_mult
    pos = torch.arange(T).float()
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
    fr = pos[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    cos, sin = emb.cos()[None], emb.sin()[None]           # [1, T, HD]
    causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    for il in range(N_LAYER):
        p = f"model.layers.{il}."
        h = rms(x, W[p + "input_layernorm.weight"])
        q = (h @ W[p + "self_attn.q_proj.weight"].T).view(T, NH, HD).transpose(0, 1)
        k = (h @ W[p + "self_attn.k_proj.weight"].T).view(T, NKV, HD).transpose(0, 1)
        v = (h @ W[p + "self_attn.v_proj.weight"].T).view(T, NKV, HD).transpose(0, 1)
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        k = k.repeat_interleave(NH // NKV, dim=0)
        v = v.repeat_interleave(NH // NKV, dim=0)
        s = (q @ k.transpose(1, 2)) * attn_scale + causal
        a = (torch.softmax(s, dim=-1) @ v).transpose(0, 1).reshape(T, NH * HD)
        a = a @ W[p + "self_attn.o_proj.weight"].T
        if "no_post_norm" not in mut:
            a = rms(a, W[p + "post_norm1.weight"])
        x = x + a * res
        h = rms(x, W[p + "post_attention_layernorm.weight"])
        m = (torch.nn.functional.silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)) @ W[p + "mlp.down_proj.weight"].T
        if "no_post_norm" not in mut:
            m = rms(m, W[p + "post_norm2.weight"])
        x = x + m * res
    x = rms(x, W["model.norm.weight"])
    logits = x @ W["lm_head.weight"].T
    return logits / LOGITS_SCALING if "logit_div" in mut else logits * LOGITS_SCALING


def make_tokenizer(d: Path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["Ġt"] = 256
    vocab["th"] = 257
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[("Ġ", "t"), ("t", "h")]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|bos_unused|>", "<|image_bos|>", "<|end_message|>", "<|eos|>"]
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])
    assert tok.get_vocab_size() == V, tok.get_vocab_size()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 256,
        "added_tokens_decoder": {str(tok.token_to_id(s)): {"content": s, "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False} for s in specials},
    }), encoding="utf-8")


hf = OUT / "hf"
hf.mkdir(exist_ok=True)
save_file({k: v.contiguous() for k, v in W.items()}, str(hf / "model.safetensors"), metadata={"format": "pt"})
(hf / "config.json").write_text(json.dumps({
    "architectures": ["HyperCLOVAXForCausalLM"], "model_type": "hyperclovax", "vocab_size": V, "hidden_size": H, "intermediate_size": FF,
    "num_hidden_layers": N_LAYER, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD, "hidden_act": "silu",
    "max_position_embeddings": 256, "rms_norm_eps": EPS, "tie_word_embeddings": False, "attention_bias": False, "mlp_bias": False,
    "rope_parameters": {"rope_type": "default", "rope_theta": THETA},
    "embedding_multiplier": EMB_MULT, "residual_multiplier": RES_MULT, "attention_multiplier": ATTN_MULT, "logits_scaling": LOGITS_SCALING,
    "use_post_norm": True, "bos_token_id": EOS, "eos_token_id": EOS, "pad_token_id": 0,
}), encoding="utf-8")
make_tokenizer(hf)

# the tiny tokenizer matches no known pre-tokenizer hash; name one so the converter does not refuse
from conversion.base import TextModel  # noqa: E402
TextModel.get_vocab_base_pre = lambda self, tokenizer: "default"
cls = ModelBase.from_model_architecture("HyperCLOVAXForCausalLM", model_type=ModelType.TEXT)
cls.no_mtp = True
cls.mtp_only = False
cls.mtp_shared_embd = False
m = cls(hf, gguf.LlamaFileType.ALL_F32, OUT / "model_f32.gguf", is_big_endian=False, use_temp_file=False, eager=True)
m.write()

g = torch.Generator().manual_seed(11)
ids = torch.randint(0, 256, (N_TOKENS,), generator=g)
with torch.no_grad():
    logits = reference(ids).float().numpy()
(OUT / "ref_tokens.txt").write_text("\n".join(str(int(t)) for t in ids) + "\n", encoding="utf-8")
logits.astype(np.float32).tofile(str(OUT / "ref_logits.bin"))
print("wrote", OUT, "logits", logits.shape, "absmax %.4f" % float(np.abs(logits).max()), "finite", bool(np.isfinite(logits).all()), flush=True)

for name in MUTATIONS:
    with torch.no_grad():
        lm = reference(ids, (name,)).float().numpy()
    lm.astype(np.float32).tofile(str(OUT / f"ref_logits.mut_{name}.bin"))
    print("mutation %-20s max|d| vs faithful reference %.4e" % (name, float(np.abs(lm - logits).max())), flush=True)
