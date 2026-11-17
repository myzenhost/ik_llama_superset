#!/usr/bin/env python3
# Tiny random `llama` GGUF written as an unsplit file and as several SPLIT layouts, for
# tests/test-split-metadata-only-shard.cpp. The layout that matters is the one Unsloth's UD uploads (and
# convert_hf_to_gguf.py --no-tensor-first-split) produce: shard 00001 holds ALL the metadata and declares
# n_tensors = 0, every tensor lives in the later shards.
#
#   python make_tiny_split_gguf.py <tree-root> <out-dir>
#
# writes into <out-dir>:
#   whole.gguf                                   unsplit reference
#   meta1-00001-of-00002.gguf, ...-00002-of-...  shard 1 = metadata only (0 tensors), shard 2 = everything
#   meta3-0000{1,2,3}-of-00003.gguf              shard 1 = metadata only, tensors split over shards 2 and 3
#   bal-00001-of-00002.gguf, ...                 control: tensors spread over both shards (shard 1 is NOT empty)
#   lone-00001-of-00002.gguf                     shard 1 of a metadata-only pair whose shard 2 is MISSING
#
# Row: S69-GGUF-SHARD1-IS-METADATA-ONLY-NTENSORS-ZERO-01.
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

V = 64
E = 32
rng = np.random.default_rng(2468)


def rnd(*shape):
    return (rng.standard_normal(shape) * 0.1).astype(np.float32)


tensors = {"token_embd.weight": rnd(V, E)}
for i in range(2):
    tensors[f"blk.{i}.attn_norm.weight"] = 1.0 + rnd(E)
    for nm in ("attn_q", "attn_k", "attn_v", "attn_output"):
        tensors[f"blk.{i}.{nm}.weight"] = rnd(E, E)
    tensors[f"blk.{i}.ffn_norm.weight"] = 1.0 + rnd(E)
    tensors[f"blk.{i}.ffn_gate.weight"] = rnd(2 * E, E)
    tensors[f"blk.{i}.ffn_up.weight"] = rnd(2 * E, E)
    tensors[f"blk.{i}.ffn_down.weight"] = rnd(E, 2 * E)
tensors["output_norm.weight"] = 1.0 + rnd(E)
tensors["output.weight"] = rnd(V, E)
names = list(tensors)


def add_model_kv(w):
    w.add_name("tiny-split")
    w.add_context_length(64)
    w.add_embedding_length(E)
    w.add_block_count(2)
    w.add_feed_forward_length(2 * E)
    w.add_head_count(4)
    w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(E // 4)
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


def write_one(path, with_model_kv, split_no, split_count, shard_names):
    w = gguf.GGUFWriter(str(path), "llama")
    if with_model_kv:
        add_model_kv(w)
    if split_count > 1:
        w.add_uint16("split.no", split_no)
        w.add_uint16("split.count", split_count)
        w.add_int32("split.tensors.count", len(names))
    for n in shard_names:
        w.add_tensor(n, tensors[n])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", path.name, len(shard_names), "tensors")


def write_split(prefix, layout):
    n = len(layout)
    for i, shard_names in enumerate(layout):
        write_one(out / f"{prefix}-{i + 1:05d}-of-{n:05d}.gguf", i == 0, i, n, shard_names)


write_one(out / "whole.gguf", True, 0, 1, names)
half = len(names) // 2
write_split("meta1", [[], names])
write_split("meta3", [[], names[:half], names[half:]])
write_split("bal", [names[:half], names[half:]])
# a metadata-only first shard whose partner is missing: same bytes as meta1 shard 1, different prefix
write_one(out / "lone-00001-of-00002.gguf", True, 0, 2, [])
