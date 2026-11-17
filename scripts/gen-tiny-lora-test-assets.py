#!/usr/bin/env python3
"""Random-weight test assets for the serving-feature tests (no real model is needed on a CPU-only box).

Writes into OUT_DIR:
  tiny-llama.gguf          a 2-layer LLaMA (n_embd 64, 4 heads, ffn 128), F32, random weights, byte-fallback SPM vocab (300 tokens)
  lora-a.gguf lora-b.gguf  rank-4 and rank-8 LoRA adapters over q/k/v/o and the three FFN matrices of both layers
                           (different random seeds, so the three configurations give three different greedy continuations)
  <name>.gguf ...          (extra adapters with --n-adapters N, named lora-a, lora-b, lora-c, ...)

Usage: PYTHONPATH=gguf-py python3 scripts/gen-tiny-lora-test-assets.py OUT_DIR [--n-adapters 3] [--lora-scale 0.6]
Needs numpy and gguf-py (in this repo).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf  # noqa: E402

N_EMBD, N_HEAD, N_FF, N_LAYER, N_CTX = 64, 4, 128, 2, 262144
PIECES = ["▁the", "▁a", "▁of", "▁and", "▁to", "▁in", "▁is", "▁it", "▁you", "▁that",
          "▁once", "▁upon", "▁time", "▁there", "▁was", "▁little", "▁girl", "▁named", "▁lily",
          "▁day", "▁she", "▁went", "▁park", "▁play", "▁with", "▁her", "▁friend", "▁happy",
          "▁ball", "▁red", "▁dog", "▁cat", "▁tree", "▁sun", "▁big", "▁small", "▁run", "▁jump", "▁fun", "."]


def vocab():
    toks, scores, types = ["<unk>", "<s>", "</s>"], [0.0, 0.0, 0.0], [2, 3, 3]
    for i in range(256):
        toks.append("<0x%02X>" % i)
        scores.append(-1000.0)   # bytes are the last resort: with a higher score they would win every merge and no word would ever be one token
        types.append(6)
    # SPM merges adjacent symbols whose concatenation is a token, so every proper prefix of a word has to be a token too, or the word is never built
    seen = set()
    prefixes = []
    for p in PIECES:
        for k in range(1, len(p)):
            if p[:k] not in seen:
                seen.add(p[:k])
                prefixes.append(p[:k])
    for i, p in enumerate(PIECES):
        toks.append(p)
        scores.append(-1.0 - 0.01 * i)
        types.append(1)
        seen.add(p)
    for p in prefixes:
        if p not in PIECES:
            toks.append(p)
            scores.append(-20.0)
            types.append(1)
    return toks, scores, types


def write_model(path, rng):
    toks, scores, types = vocab()
    n_vocab = len(toks)
    w = gguf.GGUFWriter(path, "llama")
    w.add_name("tiny-random-llama")
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(N_LAYER)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(N_EMBD // N_HEAD)
    w.add_vocab_size(n_vocab)
    w.add_file_type(0)
    w.add_tokenizer_model("llama")
    w.add_token_list(toks)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(True)

    def rnd(*shape, s=0.08):
        return rng.normal(0.0, s, size=shape).astype(np.float32)

    w.add_tensor("token_embd.weight", rnd(n_vocab, N_EMBD, s=0.3))
    w.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    w.add_tensor("output.weight", rnd(n_vocab, N_EMBD, s=0.3))
    for i in range(N_LAYER):
        w.add_tensor("blk.%d.attn_norm.weight" % i, np.ones(N_EMBD, dtype=np.float32))
        w.add_tensor("blk.%d.attn_q.weight" % i, rnd(N_EMBD, N_EMBD))
        w.add_tensor("blk.%d.attn_k.weight" % i, rnd(N_EMBD, N_EMBD))
        w.add_tensor("blk.%d.attn_v.weight" % i, rnd(N_EMBD, N_EMBD))
        w.add_tensor("blk.%d.attn_output.weight" % i, rnd(N_EMBD, N_EMBD))
        w.add_tensor("blk.%d.ffn_norm.weight" % i, np.ones(N_EMBD, dtype=np.float32))
        w.add_tensor("blk.%d.ffn_gate.weight" % i, rnd(N_FF, N_EMBD))
        w.add_tensor("blk.%d.ffn_up.weight" % i, rnd(N_FF, N_EMBD))
        w.add_tensor("blk.%d.ffn_down.weight" % i, rnd(N_EMBD, N_FF))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_lora(path, rng, rank, std):
    # alpha == rank -> llama.cpp's alpha/rank factor is 1
    w = gguf.GGUFWriter(path, "llama")
    w.add_string("general.type", "adapter")
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(rank))
    shapes = {
        "attn_q": (N_EMBD, N_EMBD), "attn_k": (N_EMBD, N_EMBD), "attn_v": (N_EMBD, N_EMBD), "attn_output": (N_EMBD, N_EMBD),
        "ffn_gate": (N_EMBD, N_FF), "ffn_up": (N_EMBD, N_FF), "ffn_down": (N_FF, N_EMBD),   # (n_in, n_out) as ggml ne0, ne1
    }
    for i in range(N_LAYER):
        for name, (n_in, n_out) in shapes.items():
            w.add_tensor("blk.%d.%s.weight.lora_a" % (i, name), rng.normal(0, std, size=(rank, n_in)).astype(np.float32))
            w.add_tensor("blk.%d.%s.weight.lora_b" % (i, name), rng.normal(0, std, size=(n_out, rank)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n-adapters", type=int, default=3)
    ap.add_argument("--lora-scale", type=float, default=0.6, help="std of the random A and B entries")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    write_model(os.path.join(a.out_dir, "tiny-llama.gguf"), np.random.default_rng(12345))
    for k in range(a.n_adapters):
        name = "lora-" + chr(ord("a") + k)
        write_lora(os.path.join(a.out_dir, name + ".gguf"), np.random.default_rng(777 + k), 4 if k % 2 == 0 else 8, a.lora_scale)
    print("wrote", sorted(os.listdir(a.out_dir)))


if __name__ == "__main__":
    main()
