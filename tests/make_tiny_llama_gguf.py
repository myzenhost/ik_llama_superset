#!/usr/bin/env python3
# Tiny random-weight `llama` GGUF for end-to-end server tests that need no model download (SGLang-merge picks:
# tests/test-server-sglang-picks.cpp and tests/sglang_wired_check.py).
#
#   python make_tiny_llama_gguf.py <tree-root> <out.gguf> [n_ctx] [bytes]
#
# 2 layers, n_embd 32, 4 heads, context 2048 by default. Weights are random but seeded, so the output is deterministic.
# default vocab: 64 tokens (ids 0..2 control, the rest "tN").
# "bytes" vocab: 259 tokens = 3 control + one token per byte (GPT-2 byte-to-unicode names), no merges, so every ASCII text tokenizes to one
# token per character; needed by tests that run a grammar (quotes, braces, digits) on the random-weight model.
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
n_ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
BYTES = len(sys.argv) > 4 and sys.argv[4] == "bytes"
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

V, E = (259 if BYTES else 64), 32


def bytes_to_unicode():
    # GPT-2's reversible byte -> printable unicode map
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
rng = np.random.default_rng(1357)


def rnd(*shape):
    return (rng.standard_normal(shape) * 0.1).astype(np.float32)


w = gguf.GGUFWriter(str(out), "llama")
w.add_name("tiny-random")
w.add_context_length(n_ctx)
w.add_embedding_length(E)
w.add_block_count(2)
w.add_feed_forward_length(2 * E)
w.add_head_count(4)
w.add_head_count_kv(4)
w.add_layer_norm_rms_eps(1e-5)
w.add_rope_dimension_count(E // 4)
if BYTES:
    m = bytes_to_unicode()
    toks = ["<pad>", "<s>", "</s>"] + [m[b] for b in range(256)]
    merges = [m[0] + " " + m[1]]  # an empty list is not written, and the loader insists on the key; bytes 0 and 1 never occur in ASCII text
else:
    toks = ["<pad>", "<s>", "</s>", "Ċ"] + ["t%d" % i for i in range(4, V - 1)] + ["t4t5"]
    merges = ["t4 t5"]
types = [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.NORMAL] * (V - 3)
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("default")
w.add_token_list(toks)
w.add_token_types(types)
w.add_token_merges(merges)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_pad_token_id(0)
w.add_tensor("token_embd.weight", rnd(V, E))
for i in range(2):
    w.add_tensor(f"blk.{i}.attn_norm.weight", 1.0 + rnd(E))
    for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
        w.add_tensor(f"blk.{i}.{nm}.weight", rnd(E, E))
    w.add_tensor(f"blk.{i}.ffn_norm.weight", 1.0 + rnd(E))
    w.add_tensor(f"blk.{i}.ffn_gate.weight", rnd(2 * E, E))
    w.add_tensor(f"blk.{i}.ffn_up.weight", rnd(2 * E, E))
    w.add_tensor(f"blk.{i}.ffn_down.weight", rnd(E, 2 * E))
w.add_tensor("output_norm.weight", 1.0 + rnd(E))
w.add_tensor("output.weight", rnd(V, E))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
