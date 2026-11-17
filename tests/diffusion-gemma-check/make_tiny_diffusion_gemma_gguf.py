# -*- coding: utf-8 -*-
"""Tiny RANDOM-WEIGHT DiffusionGemma GGUF with a working byte-level tokenizer, a mask token, a chat template,
self-conditioning tensors, and diffusion.eb_* metadata -- so llama-diffusion-cli, llama-diffusion-gemma-visual-server
and llama-diffusion-eb-selftest can run end to end on CPU with no real checkpoint.

The OUTPUT TEXT IS MEANINGLESS (random weights). What this proves: the arch loads, the tokenizer and chat template
work, the entropy-bound loop runs, the diffusion.eb_* keys written by THIS fork's gguf-py writer are the keys the C++
readers ask for, and self-conditioning is on the graph (sc_* tensors present). It proves nothing about numerical
agreement with transformers; no DiffusionGemma checkpoint exists on this host.

Usage: python make_tiny_diffusion_gemma_gguf.py <worktree-root> <out.gguf> [--std 0.5] [--canvas 8] [--seed 0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("out")
ap.add_argument("--std", type=float, default=0.5, help="weight std; larger = more peaked logits")
ap.add_argument("--canvas", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-embd", type=int, default=64)
ap.add_argument("--no-eb-keys", action="store_true", help="omit diffusion.eb_* metadata (reader falls back to defaults)")
ap.add_argument("--force-invalid-utf8", action="store_true",
                help="write an untied output.weight whose rows for the lone UTF-8 continuation bytes 0x80/0x81 are +/-K*u, "
                     "so for any hidden state one of them wins and every decoded canvas is invalid UTF-8")
args = ap.parse_args()

sys.path.insert(0, str(Path(args.root) / "gguf-py"))
import gguf  # noqa: E402

ARCH = "diffusion-gemma"
N_EMBD = args.n_embd
N_LAYER = 2
N_HEAD, N_HEAD_KV, HEAD_DIM = 4, 2, 16
N_FF = 128
N_SWA = 8
CANVAS = args.canvas
SWA_LAYERS = [True, False]  # one sliding + one global layer: both mask variants and the rope_freqs branch


def bytes_to_unicode():
    # GPT-2 byte<->unicode table (standard), so every byte has a one-char token and any text tokenizes
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


b2u = bytes_to_unicode()
tokens = ["<pad>", "<bos>", "<eos>", "<mask>"] + [b2u[b] for b in range(256)]
# one merge so the loader accepts the BPE vocab (an empty merges list is rejected)
tokens.append(b2u[ord("a")] + b2u[ord("b")])
types = [gguf.TokenType.CONTROL] * 4 + [gguf.TokenType.NORMAL] * (len(tokens) - 4)
N_VOCAB = len(tokens)
MASK_ID = 3

rng = np.random.default_rng(args.seed)


def w(*shape, std=None):
    return (rng.standard_normal(shape) * (args.std if std is None else std)).astype(np.float32)


gw = gguf.GGUFWriter(args.out, ARCH)
gw.add_context_length(256)
gw.add_embedding_length(N_EMBD)
gw.add_block_count(N_LAYER)
gw.add_feed_forward_length(N_FF)
gw.add_head_count(N_HEAD)
gw.add_head_count_kv(N_HEAD_KV)
gw.add_key_length(HEAD_DIM)
gw.add_value_length(HEAD_DIM)
gw.add_uint32(f"{ARCH}.attention.key_length_swa", HEAD_DIM)
gw.add_uint32(f"{ARCH}.attention.value_length_swa", HEAD_DIM)
gw.add_layer_norm_rms_eps(1e-6)
gw.add_sliding_window(N_SWA)
gw.add_sliding_window_pattern(SWA_LAYERS)
gw.add_rope_dimension_count(HEAD_DIM)
gw.add_uint32(f"{ARCH}.rope.dimension_count_swa", HEAD_DIM)
gw.add_rope_freq_base(1000000.0)
gw.add_float32(f"{ARCH}.rope.freq_base_swa", 10000.0)
gw.add_file_type(gguf.LlamaFileType.ALL_F32)
gw.add_causal_attention(False)
gw.add_diffusion_canvas_length(CANVAS)

if not args.no_eb_keys:
    # deliberately NOT the reference defaults (48 / 0.4 / 0.8 / 0.1 / 1 / 0.005): a reader that fails to find
    # the key falls back to a default and the CLI log then shows the default, which the check catches.
    gw.add_diffusion_eb_max_steps(7)
    gw.add_diffusion_eb_t_min(0.25)
    gw.add_diffusion_eb_t_max(0.75)
    gw.add_diffusion_eb_entropy_bound(0.5)
    gw.add_diffusion_eb_stability_threshold(2)
    gw.add_diffusion_eb_confidence_threshold(0.0125)

gw.add_tokenizer_model("gpt2")
gw.add_tokenizer_pre("default")
gw.add_token_list(tokens)
gw.add_token_types(types)
gw.add_token_merges([b2u[ord("a")] + " " + b2u[ord("b")]])
gw.add_bos_token_id(1)
gw.add_eos_token_id(2)
gw.add_pad_token_id(0)
gw.add_mask_token_id(MASK_ID)
gw.add_chat_template(
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}model: {% endif %}")

gw.add_tensor("token_embd.weight", w(N_VOCAB, N_EMBD, std=1.0))
gw.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
if args.force_invalid_utf8:
    out_w = w(N_VOCAB, N_EMBD, std=1.0)
    u = rng.standard_normal(N_EMBD).astype(np.float32)
    u /= np.linalg.norm(u)
    out_w[4 + 0x80] = 60.0 * u      # token id 4+b is the single-byte token for byte b (see the token list above)
    out_w[4 + 0x81] = -60.0 * u
    gw.add_tensor("output.weight", out_w)
gw.add_tensor("rope_freqs.weight", np.ones(HEAD_DIM // 2, dtype=np.float32))
# self-conditioning gated MLP (model level): present so the SC subgraph is really on the graph
gw.add_tensor("self_cond_pre_norm.weight", np.ones(N_EMBD, dtype=np.float32))
gw.add_tensor("self_cond_gate.weight", w(N_FF, N_EMBD))
gw.add_tensor("self_cond_up.weight", w(N_FF, N_EMBD))
gw.add_tensor("self_cond_down.weight", w(N_EMBD, N_FF))

n_embd_gqa = HEAD_DIM * N_HEAD_KV
for i in range(N_LAYER):
    p = "blk.%d." % i
    gw.add_tensor(p + "attn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    gw.add_tensor(p + "attn_q.weight", w(HEAD_DIM * N_HEAD, N_EMBD))
    gw.add_tensor(p + "attn_k.weight", w(n_embd_gqa, N_EMBD))
    gw.add_tensor(p + "attn_v.weight", w(n_embd_gqa, N_EMBD))
    gw.add_tensor(p + "attn_output.weight", w(N_EMBD, HEAD_DIM * N_HEAD))
    gw.add_tensor(p + "attn_q_norm.weight", np.ones(HEAD_DIM, dtype=np.float32))
    gw.add_tensor(p + "attn_k_norm.weight", np.ones(HEAD_DIM, dtype=np.float32))
    gw.add_tensor(p + "post_attention_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    gw.add_tensor(p + "layer_output_scale.weight", np.array([1.0], dtype=np.float32))
    gw.add_tensor(p + "enc_layer_output_scale.weight", np.array([1.0], dtype=np.float32))
    gw.add_tensor(p + "ffn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    gw.add_tensor(p + "ffn_gate.weight", w(N_FF, N_EMBD))
    gw.add_tensor(p + "ffn_up.weight", w(N_FF, N_EMBD))
    gw.add_tensor(p + "ffn_down.weight", w(N_EMBD, N_FF))
    gw.add_tensor(p + "post_ffw_norm.weight", np.ones(N_EMBD, dtype=np.float32))

gw.write_header_to_file()
gw.write_kv_data_to_file()
gw.write_tensors_to_file()
gw.close()
print("wrote %s n_vocab=%d n_embd=%d canvas=%d eb_keys=%s" % (args.out, N_VOCAB, N_EMBD, CANVAS, not args.no_eb_keys))
