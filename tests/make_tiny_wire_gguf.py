#!/usr/bin/env python3
# (Was make_tiny_llama_gguf.py on cloud/wire-runtime-paths; renamed at the merge because the sglang picks
# already ship a different generator under that name.)
# Tiny random-weight `llama` GGUF for the wiring tests (turbo KV on CPU, top-k prefilter, checkpoint CRAM).
# F32, seeded, toy gpt2-BPE vocab. Head size is 128 so the turbo KV types (which need 128 or 256) can be used.
#
#   python make_tiny_wire_gguf.py <tree-root> <out.gguf> [seed=1] [vocab=64] [gain=0.05]
#
# gain scales the attention weights: with a large gain the model is so sensitive that even a q8_0 KV cache moves the
# logits by tens of percent, which makes a quantised-cache comparison meaningless (measured: gain 0.3 -> q8_0 err 0.19).
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
V = int(sys.argv[4]) if len(sys.argv) > 4 else 64
GAIN = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

E, NL, NH, FF = 256, 2, 2, 512
rng = np.random.default_rng(seed)


def rnd(*shape, s=0.1):
    return (rng.standard_normal(shape) * s).astype(np.float32)


t = {"token_embd.weight": rnd(V, E, s=0.5), "output_norm.weight": 1.0 + rnd(E), "output.weight": rnd(V, E, s=0.5)}
for i in range(NL):
    t[f"blk.{i}.attn_norm.weight"] = 1.0 + rnd(E)
    for nm in ("attn_q", "attn_k", "attn_v"):
        t[f"blk.{i}.{nm}.weight"] = rnd(E, E, s=GAIN)
    t[f"blk.{i}.attn_output.weight"] = rnd(E, E, s=0.1)
    t[f"blk.{i}.ffn_norm.weight"] = 1.0 + rnd(E)
    t[f"blk.{i}.ffn_gate.weight"] = rnd(FF, E)
    t[f"blk.{i}.ffn_up.weight"] = rnd(FF, E)
    t[f"blk.{i}.ffn_down.weight"] = rnd(E, FF)

w = gguf.GGUFWriter(str(out), "llama")
w.add_name("tiny-wire")
w.add_context_length(512)
w.add_embedding_length(E)
w.add_block_count(NL)
w.add_feed_forward_length(FF)
w.add_head_count(NH)
w.add_head_count_kv(NH)
w.add_layer_norm_rms_eps(1e-5)
w.add_rope_dimension_count(E // NH)
toks = ["<pad>", "<s>", "</s>", "Ċ"] + ["t%d" % i for i in range(4, V - 1)] + ["t4t5"]
types = [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.NORMAL] * (V - 3)
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("default")
w.add_token_list(toks)
w.add_token_types(types)
w.add_token_merges(["t4 t5"])
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_pad_token_id(0)
for k, v in t.items():
    w.add_tensor(k, v)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
