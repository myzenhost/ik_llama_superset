#!/usr/bin/env python3
"""Tiny random-weight `llama` GGUF (64-token gpt2 vocab) for model-free llama-audio checks.

  python make_tiny_asr_gguf.py <tree-root> <out.gguf>

Used by examples/audio/tests/run_audio_chat_checks.sh. Random weights: the output is gibberish by
design; what the checks need is a model that decodes deterministically and fast.
"""
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

V, E, L = 64, 32, 2
rng = np.random.default_rng(1357)


def rnd(*shape, scale=0.3):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


w = gguf.GGUFWriter(str(out), "llama")
w.add_name("tiny-asr")
w.add_context_length(256)
w.add_embedding_length(E)
w.add_block_count(L)
w.add_feed_forward_length(2 * E)
w.add_head_count(4)
w.add_head_count_kv(4)
w.add_layer_norm_rms_eps(1e-5)
w.add_rope_dimension_count(E // 4)
# gpt2 BPE needs the single byte-level characters of whatever the prompt contains ("w", digits, space = "Ġ"),
# else a prompt tokenizes to nothing and llama_decode sees n_tokens == 0.
toks = ["<pad>", "<s>", "</s>", "Ċ", "w", "Ġ"] + [str(d) for d in range(10)] + ["w%d" % i for i in range(4, 49)] + ["w4w5", "z0", "z1"]
assert len(toks) == V and len(set(toks)) == V
types = [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.NORMAL] * (V - 3)
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("default")
w.add_token_list(toks)
w.add_token_types(types)
w.add_token_merges(["w4 w5"])
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_pad_token_id(0)

w.add_tensor("token_embd.weight", rnd(V, E, scale=0.5))
for i in range(L):
    w.add_tensor(f"blk.{i}.attn_norm.weight", 1.0 + rnd(E, scale=0.05))
    for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
        w.add_tensor(f"blk.{i}.{nm}.weight", rnd(E, E))
    w.add_tensor(f"blk.{i}.ffn_norm.weight", 1.0 + rnd(E, scale=0.05))
    w.add_tensor(f"blk.{i}.ffn_gate.weight", rnd(2 * E, E))
    w.add_tensor(f"blk.{i}.ffn_up.weight", rnd(2 * E, E))
    w.add_tensor(f"blk.{i}.ffn_down.weight", rnd(E, 2 * E))
w.add_tensor("output_norm.weight", 1.0 + rnd(E, scale=0.05))
w.add_tensor("output.weight", rnd(V, E, scale=0.5))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
