#!/usr/bin/env python3
# Tiny random `llama` GGUFs for tests/test-model-shared-tensors.cpp: a target plus draft heads that
# do and do not carry their own token_embd / output_norm / output (the layout produced by
# convert_hf_to_gguf.py --mtp --mtp-shared-embd). F32, seeded, toy gpt2-BPE vocab (64 tokens).
#
#   python make_tiny_shared_pair_gguf.py <tree-root> <out-dir>
#
# writes into <out-dir>:
#   target.gguf             2 layers, n_embd 32, own embd / norm / output          (the model borrowed from)
#   draft_full.gguf         1 layer, carries COPIES of the target's three tensors   (control: must equal the shared draft)
#   draft_shared.gguf       the same layer, no token_embd / output_norm / output
#   draft_badshape.gguf     n_embd 16, no token_embd / output_norm / output          (must be refused against the 32 target)
#   target_tied.gguf        2 layers, NO output.weight (head tied to token_embd)
#   draft_full_tied.gguf    1 layer, token_embd + output_norm, no output.weight      (control for the tied case)
#   draft_shared_tied.gguf  the same layer, none of the three
#
# Row: S69-UNSLOTH-ADOPTABLE-BRANCHES-NOT-YET-PORTED-01 (unslothai/llama.cpp f36eda84, MIT; test written here).
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

V = 64


def rnd(rng, *shape):
    return (rng.standard_normal(shape) * 0.1).astype(np.float32)


def layer_tensors(rng, n_embd, n_layer, n_ff):
    t = {}
    for i in range(n_layer):
        t[f"blk.{i}.attn_norm.weight"] = 1.0 + rnd(rng, n_embd)
        for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
            t[f"blk.{i}.{nm}.weight"] = rnd(rng, n_embd, n_embd)
        t[f"blk.{i}.ffn_norm.weight"] = 1.0 + rnd(rng, n_embd)
        t[f"blk.{i}.ffn_gate.weight"] = rnd(rng, n_ff, n_embd)
        t[f"blk.{i}.ffn_up.weight"] = rnd(rng, n_ff, n_embd)
        t[f"blk.{i}.ffn_down.weight"] = rnd(rng, n_embd, n_ff)
    return t


def write(name, n_embd, n_layer, tensors):
    w = gguf.GGUFWriter(str(out / name), "llama")
    w.add_name("tiny-" + name)
    w.add_context_length(64)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(2 * n_embd)
    w.add_head_count(4)
    w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(n_embd // 4)
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
    for k, v in tensors.items():
        w.add_tensor(k, v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", name, len(tensors), "tensors")


E = 32
rng = np.random.default_rng(1234)
emb = rnd(rng, V, E)
onorm = 1.0 + rnd(rng, E)
head = rnd(rng, V, E)

target = layer_tensors(rng, E, 2, 2 * E)
target.update({"token_embd.weight": emb, "output_norm.weight": onorm, "output.weight": head})
write("target.gguf", E, 2, target)

d_layer = layer_tensors(np.random.default_rng(99), E, 1, 2 * E)
write("draft_full.gguf", E, 1, {**d_layer, "token_embd.weight": emb, "output_norm.weight": onorm, "output.weight": head})
write("draft_shared.gguf", E, 1, dict(d_layer))

bad = layer_tensors(np.random.default_rng(7), 16, 1, 32)
write("draft_badshape.gguf", 16, 1, bad)

tied = layer_tensors(np.random.default_rng(1234), E, 2, 2 * E)
tied.update({"token_embd.weight": emb, "output_norm.weight": onorm})
write("target_tied.gguf", E, 2, tied)
write("draft_full_tied.gguf", E, 1, {**d_layer, "token_embd.weight": emb, "output_norm.weight": onorm})
write("draft_shared_tied.gguf", E, 1, dict(d_layer))
