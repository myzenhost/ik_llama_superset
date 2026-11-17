#!/usr/bin/env python3
"""MiniCPM4 longrope runtime test: the legacy minicpm graph must apply rope_factors_long / rope_factors_short and the
rope.scaling.attn_factor, and pick long vs short by context exactly as build_rope_factors does (n_ctx > n_ctx_orig_yarn -> long).

Builds a tiny random F32 'minicpm' GGUF with rope scaling type longrope, original context 512 (train ctx 1024), non-trivial long
and short factor vectors (different from each other) and attn_factor 1.07, then runs llama-eval-callback twice:
  -c 1024 (> 512)  -> must match a numpy reference that uses the LONG factors
  -c 256  (<= 512) -> must match the reference that uses the SHORT factors
(llama.cpp pads n_ctx up to a multiple of 256, so an original context below 256 can never select the short factors -- the first
version of this test used 64 / -c 32, ran at 256 and measured the long factors twice.)
Reference math = ggml rope NORM mode with freq factors: theta_i = pos * base^(-2i/d) / f_i; q and k are each scaled by the
attn factor (ggml multiplies cos and sin by it when ext_factor == 0), i.e. HF MiniCPMLongRoPE's scaling_factor on cos/sin.
Fail-before: a binary without 470fede9e ignores the factors and fails both arms; the run also fails if the long and short
references are too close to tell apart (that would make the selection untested).
usage: test_minicpm4_longrope_runtime.py <llama-eval-callback.exe> <work dir>     exit 0 = both arms match (max|diff| < 1e-3)
"""
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gguf-py"))
import numpy as np  # noqa: E402
import gguf  # noqa: E402

H, L, NH, DH, FF, V = 64, 2, 4, 16, 128, 16
BASE, EPS, ORIG, ATTN = 10000.0, 1e-5, 512, 1.07
SCALE_EMB, SCALE_DEPTH, DIM_BASE = 12.0, 1.4, 256.0
TOKS = ["<unk>", "<s>", "</s>", "▁"] + list("abcdefghijkl")


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def rope(x, pos, ff):
    out = x.copy()
    for i in range(DH // 2):
        th = pos[:, None] * (BASE ** (-2.0 * i / DH)) / ff[i]
        c, s = np.cos(th) * ATTN, np.sin(th) * ATTN
        a, b = x[..., 2 * i], x[..., 2 * i + 1]
        out[..., 2 * i] = a * c - b * s
        out[..., 2 * i + 1] = a * s + b * c
    return out


def reference(W, toks, ff):
    T = len(toks)
    x = W["token_embd"][toks].astype(np.float64) * SCALE_EMB
    sres = SCALE_DEPTH / math.sqrt(L)
    pos = np.arange(T, dtype=np.float64)
    for l in range(L):
        h = rms(x, W[f"an{l}"])
        q = (h @ W[f"q{l}"].T).reshape(T, NH, DH)
        k = (h @ W[f"k{l}"].T).reshape(T, NH, DH)
        v = (h @ W[f"v{l}"].T).reshape(T, NH, DH)
        q, k = rope(q, pos, ff), rope(k, pos, ff)
        att = np.einsum("thd,shd->hts", q, k) / math.sqrt(DH)
        att = att + np.triu(np.full((T, T), -np.inf), 1)[None]
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hts,shd->thd", att, v).reshape(T, NH * DH) @ W[f"o{l}"].T
        x = x + o * sres
        h = rms(x, W[f"fn{l}"])
        g, u = h @ W[f"g{l}"].T, h @ W[f"u{l}"].T
        x = x + ((g / (1 + np.exp(-g)) * u) @ W[f"d{l}"].T) * sres
    h = rms(x, W["on"]) / (H / DIM_BASE)
    return (h @ W["out"].T)[-1]


def build(path, rng):
    f32 = lambda *s, sc=0.15: (rng.standard_normal(s) * sc).astype(np.float32)  # noqa: E731
    W = {"token_embd": f32(V, H), "on": 1 + f32(H, sc=0.1), "out": f32(V, H, sc=0.3)}
    for l in range(L):
        W[f"an{l}"] = 1 + f32(H, sc=0.1)
        W[f"fn{l}"] = 1 + f32(H, sc=0.1)
        for n in "qkvo":
            W[f"{n}{l}"] = f32(H, H, sc=0.4)
        W[f"g{l}"], W[f"u{l}"], W[f"d{l}"] = f32(FF, H), f32(FF, H), f32(H, FF)
    long_f = np.linspace(1.0, 8.0, DH // 2).astype(np.float32)    # strong low-frequency stretch
    short_f = np.linspace(0.6, 1.6, DH // 2).astype(np.float32)   # different, also != 1
    w = gguf.GGUFWriter(path, "minicpm")
    w.add_name("tiny-minicpm4-longrope-test")
    w.add_context_length(1024); w.add_embedding_length(H); w.add_block_count(L)
    w.add_feed_forward_length(FF); w.add_head_count(NH); w.add_head_count_kv(NH)
    w.add_layer_norm_rms_eps(EPS); w.add_rope_dimension_count(DH); w.add_rope_freq_base(BASE)
    w.add_rope_scaling_type(gguf.RopeScalingType.LONGROPE)
    w.add_rope_scaling_orig_ctx_len(ORIG)
    w.add_rope_scaling_attn_factors(ATTN)
    w.add_vocab_size(V); w.add_file_type(0)
    w.add_embedding_scale(SCALE_EMB); w.add_residual_scale(SCALE_DEPTH / L ** 0.5); w.add_logit_scale(H / DIM_BASE)
    w.add_tokenizer_model("llama"); w.add_token_list(TOKS); w.add_token_scores([0.0] * V)
    w.add_token_types([2, 3, 3, 1] + [1] * 12)
    w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0)
    w.add_add_bos_token(False); w.add_add_space_prefix(True)
    T = gguf.MODEL_TENSOR

    def put(t, arr, bid=None):
        w.add_tensor(gguf.TENSOR_NAMES[t].format(bid=bid) + ".weight", arr)
    put(T.TOKEN_EMBD, W["token_embd"]); put(T.OUTPUT_NORM, W["on"]); put(T.OUTPUT, W["out"])
    put(T.ROPE_FACTORS_LONG, long_f); put(T.ROPE_FACTORS_SHORT, short_f)
    for l in range(L):
        put(T.ATTN_NORM, W[f"an{l}"], l); put(T.FFN_NORM, W[f"fn{l}"], l)
        put(T.ATTN_Q, W[f"q{l}"], l); put(T.ATTN_K, W[f"k{l}"], l); put(T.ATTN_V, W[f"v{l}"], l)
        put(T.ATTN_OUT, W[f"o{l}"], l)
        put(T.FFN_GATE, W[f"g{l}"], l); put(T.FFN_UP, W[f"u{l}"], l); put(T.FFN_DOWN, W[f"d{l}"], l)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return W, long_f, short_f


def run(exe, g, ctx, log):
    r = subprocess.run([exe, "-m", g, "-p", "abcdefghijkl", "-ngl", "0", "-t", "2", "-c", str(ctx), "-ctk", "f32", "-ctv", "f32",
                        "-fa", "0"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    txt = r.stdout + r.stderr
    open(log, "w", encoding="utf-8").write(txt)
    m = re.search(r"result_output = .*?\n\s*\[\s*\[\s*\[(.*?)\]\s*,?\s*\]\s*,?\s*\]\s*sum = (\S+)", txt, re.S)
    if not m:
        return None, None, r.returncode
    return [float(v) for v in re.findall(r"-?\d+\.\d+", m.group(1))], float(m.group(2)), r.returncode


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    exe, wd = sys.argv[1], sys.argv[2]
    os.makedirs(wd, exist_ok=True)
    g = os.path.join(wd, "tiny_minicpm4_longrope.gguf")
    W, long_f, short_f = build(g, np.random.default_rng(1108))
    # the tokenizer splits "abcdefghijkl" into the space token + 12 letters (ids 3, 4..15)
    toks = [3] + list(range(4, 16))
    ref_long, ref_short = reference(W, toks, long_f), reference(W, toks, short_f)
    ref_plain = reference(W, toks, np.ones(DH // 2))
    sep = float(np.abs(ref_long - ref_short).max())
    print("reference separation long vs short max|diff| = %.3e (must be > 1e-2), vs plain rope %.3e" %
          (sep, float(np.abs(ref_long - ref_plain).max())))
    fails = 0 if sep > 1e-2 else 1
    for ctx, ref, name in ((1024, ref_long, "long"), (256, ref_short, "short")):
        vals, sm, rc = run(exe, g, ctx, os.path.join(wd, "run_c%d.log" % ctx))
        if vals is None:
            print("FAIL %s (-c %d): no result_output, rc %d" % (name, ctx, rc)); fails += 1; continue
        exp = list(ref[:3]) + list(ref[-3:])
        d = max(abs(a - b) for a, b in zip(vals, exp))
        ds = abs(sm - float(ref.sum()))
        ok = d < 1e-3 and ds < 1e-2
        fails += 0 if ok else 1
        print("%s %s factors (-c %d): max|diff| %.2e |dsum| %.2e" % ("ok  " if ok else "FAIL", name, ctx, d, ds))
    print("ALL OK" if fails == 0 else "FAILED %d" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
