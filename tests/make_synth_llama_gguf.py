#!/usr/bin/env python3
"""A random-weight Llama GGUF with a chosen vocabulary size, for tests and benchmarks that need the output projection to be a visible part of
a decode step (a real 100k+ vocabulary) without any model file.

   python tests/make_synth_llama_gguf.py OUT.gguf [--vocab 100000] [--hidden 256] [--layers 2] [--heads 4] [--ffn 512] [--seed 0] [--out-type f16]

The tokenizer is a plain "llama" (SentencePiece-style) token list "<unk>", "<s>", "</s>", "t3", "t4", ...: enough for llama-server to load the
model and for requests that pass token ids. Weights are N(0, 0.05) so logits are well spread.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402


def make(out, vocab=100000, hidden=256, layers=2, heads=4, kv_heads=None, ffn=512, seed=0, out_type="f16", ctx=2048, std=0.05):
    kv_heads = kv_heads or heads
    hd = hidden // heads
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(out), "llama")
    w.add_name("synthetic-llama")
    w.add_context_length(ctx)
    w.add_embedding_length(hidden)
    w.add_block_count(layers)
    w.add_feed_forward_length(ffn)
    w.add_head_count(heads)
    w.add_head_count_kv(kv_heads)
    w.add_rope_dimension_count(hd)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_vocab_size(vocab)
    w.add_tokenizer_model("llama")
    tokens = ["<unk>", "<s>", "</s>"] + ["t%d" % i for i in range(3, vocab)]
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * vocab)
    w.add_token_types([int(gguf.TokenType.UNKNOWN), int(gguf.TokenType.CONTROL), int(gguf.TokenType.CONTROL)] + [int(gguf.TokenType.NORMAL)] * (vocab - 3))
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    dt = {"f32": np.float32, "f16": np.float16}[out_type]

    def t(name, shape, one=False):
        a = np.ones(shape, np.float32) if one else (rng.standard_normal(shape) * std).astype(np.float32)
        w.add_tensor(name, a if one or out_type == "f32" else a.astype(dt))

    t("token_embd.weight", (vocab, hidden))
    t("output_norm.weight", (hidden,), one=True)
    t("output.weight", (vocab, hidden))
    for i in range(layers):
        p = "blk.%d." % i
        t(p + "attn_norm.weight", (hidden,), one=True)
        t(p + "attn_q.weight", (heads * hd, hidden))
        t(p + "attn_k.weight", (kv_heads * hd, hidden))
        t(p + "attn_v.weight", (kv_heads * hd, hidden))
        t(p + "attn_output.weight", (hidden, heads * hd))
        t(p + "ffn_norm.weight", (hidden,), one=True)
        t(p + "ffn_gate.weight", (ffn, hidden))
        t(p + "ffn_up.weight", (ffn, hidden))
        t(p + "ffn_down.weight", (hidden, ffn))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-type", choices=["f16", "f32"], default="f16")
    a = ap.parse_args()
    make(a.out, a.vocab, a.hidden, a.layers, a.heads, None, a.ffn, a.seed, a.out_type)
    print("wrote", a.out)
