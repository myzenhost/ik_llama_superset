#!/usr/bin/env python3
# (Was make_tiny_llama_gguf.py on cloud/wire-types-kv-gate; renamed at the merge because two other generators
# with different arguments already exist: make_tiny_llama_gguf.py (sglang) and make_tiny_wire_gguf.py.)
# Tiny random-weight `llama` GGUF (F32) for end-to-end wiring tests (MX quant types, KIVI KV cache, ...).
#   python make_tiny_mx_gguf.py <tree-root> <out.gguf> [seed]
# Dims are chosen so every weight row is a multiple of 32 (MX block) and head_dim = 64 (the fork's
# attn_rot_k Hadamard needs head_dim % 64 == 0), 2 layers, vocab 256, F32 weights, norms near 1.
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1234
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

V, E, FF, NH, NL = 256, 128, 256, 2, 2
rng = np.random.default_rng(seed)


def rnd(*shape, std=None):
    std = std if std is not None else 1.0 / np.sqrt(shape[-1])
    return (rng.standard_normal(shape) * std).astype(np.float32)


w = gguf.GGUFWriter(str(out), "llama")
w.add_name("tiny-llama-random")
w.add_context_length(512)
w.add_embedding_length(E)
w.add_block_count(NL)
w.add_feed_forward_length(FF)
w.add_head_count(NH)
w.add_head_count_kv(NH)
w.add_layer_norm_rms_eps(1e-5)
w.add_rope_dimension_count(E // NH)
toks = ["<pad>", "<s>", "</s>"] + ["t%d" % i for i in range(3, V - 1)] + ["t3t4"]
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("default")
w.add_token_list(toks)
w.add_token_types([gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.NORMAL] * (V - 3))
w.add_token_merges(["t3 t4"])  # the loader requires a merges array; one merge is enough
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_pad_token_id(0)
w.add_tensor("token_embd.weight", rnd(V, E, std=1.0))
for i in range(NL):
    w.add_tensor(f"blk.{i}.attn_norm.weight", 1.0 + rnd(E, std=0.05))
    for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
        w.add_tensor(f"blk.{i}.{nm}.weight", rnd(E, E))
    w.add_tensor(f"blk.{i}.ffn_norm.weight", 1.0 + rnd(E, std=0.05))
    w.add_tensor(f"blk.{i}.ffn_gate.weight", rnd(FF, E))
    w.add_tensor(f"blk.{i}.ffn_up.weight", rnd(FF, E))
    w.add_tensor(f"blk.{i}.ffn_down.weight", rnd(E, FF))
w.add_tensor("output_norm.weight", 1.0 + rnd(E, std=0.05))
w.add_tensor("output.weight", rnd(V, E, std=1.0 / np.sqrt(E) * 3))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
