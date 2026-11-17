#!/usr/bin/env python3
"""Tiny RANDOM-weight llama-arch GGUF that llama-server can serve: SPM vocab with byte fallback (any text tokenises),
chatml chat template, tokenizer.ggml.add_bos_token configurable. F32, seeded. Never real weights.
usage: make_tiny_llama_server_gguf.py <tree-root> <out.gguf> [--no-bos] [--no-template]"""
import sys
from pathlib import Path
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
add_bos = "--no-bos" not in sys.argv
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

E, NH, L, FF = 32, 2, 2, 64
HD = E // NH
toks = ["<unk>", "<s>", "</s>"] + ["<0x%02X>" % i for i in range(256)] + ["▁", "▁a", "▁the", "he", "llo", "<|im_start|>", "<|im_end|>"]
V = len(toks)
types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL] + [gguf.TokenType.BYTE] * 256 + \
        [gguf.TokenType.NORMAL] * 5 + [gguf.TokenType.CONTROL] * 2
scores = [0.0] * 3 + [0.0] * 256 + [-1.0, -2.0, -2.0, -3.0, -3.0, 0.0, 0.0]
rng = np.random.default_rng(777)
def rnd(*s): return (rng.standard_normal(s) * 0.08).astype(np.float32)

w = gguf.GGUFWriter(str(out), "llama")
w.add_name("tiny-random-llama-server")
w.add_context_length(256); w.add_embedding_length(E); w.add_block_count(L); w.add_feed_forward_length(FF)
w.add_head_count(NH); w.add_head_count_kv(NH); w.add_rope_dimension_count(HD)
w.add_layer_norm_rms_eps(1e-5); w.add_vocab_size(V)
w.add_tokenizer_model("llama"); w.add_token_list(toks); w.add_token_scores(scores); w.add_token_types(types)
w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0)
w.add_add_bos_token(add_bos); w.add_add_eos_token(False)
if "--no-template" not in sys.argv: w.add_chat_template("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
w.add_tensor("token_embd.weight", rnd(V, E)); w.add_tensor("output_norm.weight", 1 + rnd(E)); w.add_tensor("output.weight", rnd(V, E))
for i in range(L):
    w.add_tensor(f"blk.{i}.attn_norm.weight", 1 + rnd(E)); w.add_tensor(f"blk.{i}.ffn_norm.weight", 1 + rnd(E))
    for n in ("q", "k", "v"): w.add_tensor(f"blk.{i}.attn_{n}.weight", rnd(E, E))
    w.add_tensor(f"blk.{i}.attn_output.weight", rnd(E, E))
    w.add_tensor(f"blk.{i}.ffn_gate.weight", rnd(FF, E)); w.add_tensor(f"blk.{i}.ffn_up.weight", rnd(FF, E)); w.add_tensor(f"blk.{i}.ffn_down.weight", rnd(E, FF))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out)
