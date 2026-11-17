#!/usr/bin/env python3
"""Tiny RANDOM-weight embeddinggemma in two layouts, same weights:
  <out>/ref.gguf     llama.cpp layout   general.architecture=gemma-embedding, gemma-embedding.* keys, dense_2/dense_3, output_norm
  <out>/ollama.gguf  Ollama layout      general.architecture=gemma3,          gemma3.* keys,          dense.0/dense.1
(layouts per ollama llama/compat/llama-ollama-compat.cpp:369-428 and compatmigrate/embeddinggemma.go:1-80).
usage: make_tiny_embeddinggemma.py <tree-root> <out-dir>"""
import sys
from pathlib import Path
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

E, NH, L, FF, HD, DF = 32, 2, 6, 64, 16, 64   # 6 layers: with the sliding-window period of 6 the last layer is global attention
toks = ["<pad>", "<eos>", "<bos>", "<unk>"] + ["<0x%02X>" % i for i in range(256)] + ["▁", "▁a", "▁the", "he", "llo"]
V = len(toks)
types = [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.UNKNOWN] + [gguf.TokenType.BYTE] * 256 + [gguf.TokenType.NORMAL] * 5
scores = [0.0] * (4 + 256) + [-1.0, -2.0, -2.0, -3.0, -3.0]
rng = np.random.default_rng(4242)
def rnd(*s): return (rng.standard_normal(s) * 0.08).astype(np.float32)

tensors = {"token_embd.weight": rnd(V, E), "output_norm.weight": 1 + rnd(E)}
dense = {"2": rnd(DF, E), "3": rnd(E, DF)}   # numpy (out, in) -> gguf ne = (in, out)
for i in range(L):
    tensors[f"blk.{i}.attn_norm.weight"] = 1 + rnd(E); tensors[f"blk.{i}.ffn_norm.weight"] = 1 + rnd(E)
    tensors[f"blk.{i}.post_attention_norm.weight"] = 1 + rnd(E); tensors[f"blk.{i}.post_ffw_norm.weight"] = 1 + rnd(E)
    tensors[f"blk.{i}.attn_q_norm.weight"] = 1 + rnd(HD); tensors[f"blk.{i}.attn_k_norm.weight"] = 1 + rnd(HD)
    tensors[f"blk.{i}.attn_q.weight"] = rnd(NH * HD, E)
    for n in ("k", "v"): tensors[f"blk.{i}.attn_{n}.weight"] = rnd(1 * HD, E)   # head_count_kv = 1
    tensors[f"blk.{i}.attn_output.weight"] = rnd(E, NH * HD)
    tensors[f"blk.{i}.ffn_gate.weight"] = rnd(FF, E); tensors[f"blk.{i}.ffn_up.weight"] = rnd(FF, E); tensors[f"blk.{i}.ffn_down.weight"] = rnd(E, FF)

def write(path, arch, ollama_layout):
    w = gguf.GGUFWriter(str(path), arch)
    w.add_name("tiny-random-embeddinggemma")
    w.add_context_length(256); w.add_embedding_length(E); w.add_block_count(L); w.add_feed_forward_length(FF)
    w.add_head_count(NH); w.add_head_count_kv(1); w.add_key_length(HD); w.add_value_length(HD)
    w.add_layer_norm_rms_eps(1e-6); w.add_sliding_window(8); w.add_pooling_type(gguf.PoolingType.MEAN)
    w.add_rope_freq_base(1000000.0)
    if ollama_layout:
        # the llama.cpp-format key the Ollama blob does NOT carry, and the ones its converter does write
        w.add_bool(f"{arch}.attention.causal", True)
        w.add_uint32(f"{arch}.attention.sliding_window_pattern", 6)
    else:
        w.add_float32(f"{arch}.rope.freq_base_swa", 10000.0)
        w.add_uint32(f"{arch}.dense_2_feat_in", E); w.add_uint32(f"{arch}.dense_2_feat_out", DF)
        w.add_uint32(f"{arch}.dense_3_feat_in", DF); w.add_uint32(f"{arch}.dense_3_feat_out", E)
    w.add_tokenizer_model("llama"); w.add_token_list(toks); w.add_token_scores(scores); w.add_token_types(types)
    w.add_bos_token_id(2); w.add_eos_token_id(1); w.add_unk_token_id(3); w.add_pad_token_id(0)
    w.add_add_bos_token(True); w.add_add_eos_token(True)
    for k, v in tensors.items(): w.add_tensor(k, v)
    names = ("dense.0.weight", "dense.1.weight") if ollama_layout else ("dense_2.weight", "dense_3.weight")
    w.add_tensor(names[0], dense["2"]); w.add_tensor(names[1], dense["3"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

write(out / "ref.gguf", "gemma-embedding", False)
write(out / "ollama.gguf", "gemma3", True)
print("wrote", out / "ref.gguf", out / "ollama.gguf")
