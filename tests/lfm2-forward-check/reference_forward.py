# Independent numpy (float64) reference forward pass for LFM2 / LFM2-MoE, written from the HF architecture, NOT from the engine's graph.
# PROGRESS-OK: a 3-layer tiny model, milliseconds. REWRITE-SAFE: writes only <out>.
# Usage: python reference_forward.py <weights.npz> <config.json> <T> <out.bin>     (tokens t_i = (7*i+3) % vocab, same as the C++ harness)
import json
import sys

import numpy as np

W = {k.replace("__", "."): v.astype(np.float64) for k, v in np.load(sys.argv[1]).items()}
cfg = json.load(open(sys.argv[2]))
T = int(sys.argv[3])
V = cfg["vocab_size"]
H = cfg["hidden_size"]
NH, NKV = cfg["num_attention_heads"], cfg["num_key_value_heads"]
HD = H // NH
EPS = cfg["norm_eps"]
L = cfg["conv_L_cache"]
MOE = "moe_intermediate_size" in cfg


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(x, pos):      # x [T, n, HD]; rotate_half / NEOX pairs (i, i + HD/2)
    half = HD // 2
    inv = cfg["rope_theta"] ** (-np.arange(half) * 2.0 / HD)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def conv_block(x, p):
    bcx = x @ W[f"{p}.conv.in_proj.weight"].T                # [T, 3H]
    b, c, xx = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]
    bx = b * xx
    k = W[f"{p}.conv.conv.weight"][:, 0, :]                   # [H, L]
    pad = np.concatenate([np.zeros((L - 1, H)), bx], axis=0)
    out = np.zeros_like(bx)
    for j in range(L):                                        # cross-correlation, causal: out[t] = sum_j k[:, j] * bx[t - (L-1) + j]
        out += pad[j:j + x.shape[0]] * k[:, j]
    return (c * out) @ W[f"{p}.conv.out_proj.weight"].T


def attn_block(x, p):
    n = x.shape[0]
    q = (x @ W[f"{p}.self_attn.q_proj.weight"].T).reshape(n, NH, HD)
    k = (x @ W[f"{p}.self_attn.k_proj.weight"].T).reshape(n, NKV, HD)
    v = (x @ W[f"{p}.self_attn.v_proj.weight"].T).reshape(n, NKV, HD)
    q = rms(q, W[f"{p}.self_attn.q_layernorm.weight"])
    k = rms(k, W[f"{p}.self_attn.k_layernorm.weight"])
    pos = np.arange(n, dtype=np.float64)
    q, k = rope(q, pos), rope(k, pos)
    rep = NH // NKV
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    sc = np.einsum("tnd,snd->nts", q, k) / np.sqrt(HD)
    sc = np.where(np.tril(np.ones((n, n), bool))[None], sc, -np.inf)
    sc = np.exp(sc - sc.max(-1, keepdims=True))
    sc /= sc.sum(-1, keepdims=True)
    o = np.einsum("nts,snd->tnd", sc, v).reshape(n, NH * HD)
    return o @ W[f"{p}.self_attn.out_proj.weight"].T


def dense_ffn(x, p):
    return (silu(x @ W[f"{p}.feed_forward.w1.weight"].T) * (x @ W[f"{p}.feed_forward.w3.weight"].T)) @ W[f"{p}.feed_forward.w2.weight"].T


def moe_ffn(x, p):
    ne, k = cfg["num_experts"], cfg["num_experts_per_tok"]
    score = 1.0 / (1.0 + np.exp(-(x @ W[f"{p}.feed_forward.gate.weight"].T)))      # sigmoid gating
    sel_score = score + W[f"{p}.feed_forward.expert_bias"] if cfg.get("use_expert_bias") else score
    idx = np.argsort(-sel_score, axis=-1, kind="stable")[:, :k]                      # bias steers SELECTION only
    w = np.take_along_axis(score, idx, axis=-1)
    if cfg.get("norm_topk_prob"):
        w = w / (w.sum(-1, keepdims=True) + 1e-6)
    w = w * cfg.get("routed_scaling_factor", 1.0)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(k):
            e = idx[t, j]
            h = silu(W[f"{p}.feed_forward.experts.{e}.w1.weight"] @ x[t]) * (W[f"{p}.feed_forward.experts.{e}.w3.weight"] @ x[t])
            out[t] += w[t, j] * (W[f"{p}.feed_forward.experts.{e}.w2.weight"] @ h)
    return out


toks = (7 * np.arange(T) + 3) % V
h = W["model.embed_tokens.weight"][toks]
for i, lt in enumerate(cfg["layer_types"]):
    p = f"model.layers.{i}"
    n = rms(h, W[f"{p}.operator_norm.weight"])
    h = h + (conv_block(n, p) if lt == "conv" else attn_block(n, p))
    n = rms(h, W[f"{p}.ffn_norm.weight"])
    routed = MOE and i >= cfg["num_dense_layers"]
    h = h + (moe_ffn(n, p) if routed else dense_ffn(n, p))
logits = rms(h, W["model.embedding_norm.weight"]) @ W["model.embed_tokens.weight"].T
logits.astype(np.float32).tofile(sys.argv[4])
print("wrote", logits.shape, "absmax", float(np.abs(logits).max()), "mean", float(logits.mean()))
