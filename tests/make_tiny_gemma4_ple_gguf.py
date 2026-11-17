#!/usr/bin/env python3
# Tiny random gemma4 GGUF WITH a per-layer token-embedding table (PLE), for tests/test-mmap-row-prefetch.cpp.
# gemma4 gathers that table with ggml_get_rows(tok_embd_per_layer, inp_tokens) (src/graphs/build_gemma4.cpp);
# the table is the one --defer-ple leaves on the file, so it is the tensor whose rows the readahead hint serves.
# F32, seeded, toy gpt2-BPE vocab (64 tokens). Same toy vocab as make_tiny_shared_pair_gguf.py.
#
#   python make_tiny_gemma4_ple_gguf.py <tree-root> <out.gguf>
#
# Row: S69-UNSLOTH-ADOPTABLE-BRANCHES-NOT-YET-PORTED-01 (prefetch/lazy-tensor-row-readahead).
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

V = 64
E = 32            # n_embd
NH = 2            # heads (kv heads equal)
HD = 16           # head dim
L = 2             # layers: 0 = sliding window, 1 = full attention
NPL = 8           # per-layer embedding width
FF = 64

rng = np.random.default_rng(4321)


def rnd(*shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def nm(t, bid=None):
    s = gguf.TENSOR_NAMES[t]
    return (s.format(bid=bid) if bid is not None else s) + ".weight"


T = gguf.MODEL_TENSOR
tensors = {
    nm(T.TOKEN_EMBD): rnd(V, E),
    nm(T.OUTPUT_NORM): 1.0 + rnd(E),
    nm(T.PER_LAYER_TOKEN_EMBD): rnd(V, NPL * L),
    nm(T.PER_LAYER_MODEL_PROJ): rnd(NPL * L, E),
    nm(T.PER_LAYER_PROJ_NORM): 1.0 + rnd(NPL),
}
for i in range(L):
    tensors[nm(T.ATTN_NORM, i)] = 1.0 + rnd(E)
    tensors[nm(T.ATTN_Q, i)] = rnd(NH * HD, E)
    tensors[nm(T.ATTN_K, i)] = rnd(NH * HD, E)
    tensors[nm(T.ATTN_V, i)] = rnd(NH * HD, E)
    tensors[nm(T.ATTN_OUT, i)] = rnd(E, NH * HD)
    tensors[nm(T.ATTN_Q_NORM, i)] = 1.0 + rnd(HD)
    tensors[nm(T.ATTN_K_NORM, i)] = 1.0 + rnd(HD)
    tensors[nm(T.ATTN_POST_NORM, i)] = 1.0 + rnd(E)
    tensors["blk.%d.ffn_norm.weight" % i] = 1.0 + rnd(E)
    tensors[nm(T.FFN_GATE, i)] = rnd(FF, E)
    tensors[nm(T.FFN_UP, i)] = rnd(FF, E)
    tensors[nm(T.FFN_DOWN, i)] = rnd(E, FF)
    tensors[nm(T.FFN_POST_NORM, i)] = 1.0 + rnd(E)
    tensors[nm(T.PER_LAYER_INP_GATE, i)] = rnd(NPL, E)
    tensors[nm(T.PER_LAYER_PROJ, i)] = rnd(E, NPL)
    tensors[nm(T.PER_LAYER_POST_NORM, i)] = 1.0 + rnd(E)
    if i == 1:  # full attention layers carry rope_freqs
        tensors[nm(T.ROPE_FREQS, i)] = np.ones(HD // 2, dtype=np.float32)

w = gguf.GGUFWriter(str(out), "gemma4")
w.add_name("tiny-gemma4-ple")
w.add_context_length(64)
w.add_embedding_length(E)
w.add_block_count(L)
w.add_feed_forward_length(FF)
w.add_head_count(NH)
w.add_head_count_kv(NH)
w.add_key_length(HD)
w.add_value_length(HD)
w.add_uint32("gemma4.attention.key_length_swa", HD)
w.add_uint32("gemma4.attention.value_length_swa", HD)
w.add_uint32("gemma4.attention.sliding_window", 8)
w.add_array("gemma4.attention.sliding_window_pattern", [True, False])
w.add_uint32("gemma4.attention.shared_kv_layers", 0)
w.add_uint32("gemma4.embedding_length_per_layer_input", NPL)
w.add_layer_norm_rms_eps(1e-6)
w.add_rope_dimension_count(HD)
w.add_uint32("gemma4.rope.dimension_count_swa", HD)
w.add_float32("gemma4.rope.freq_base_swa", 10000.0)
w.add_rope_freq_base(10000.0)
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
    if v is None:
        continue
    w.add_tensor(k, v)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, len([v for v in tensors.values() if v is not None]), "tensors")
