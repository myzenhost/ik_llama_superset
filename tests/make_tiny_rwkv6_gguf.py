#!/usr/bin/env python3
# Tiny random-weight `rwkv6` GGUF (F32): drives GGML_OP_RWKV_WKV6 through the real graph builder (src/models/rwkv6*.cpp).
#   python make_tiny_rwkv6_gguf.py <tree-root> <out.gguf> [seed]
# Dims: n_embd 128, head size 64 (2 heads), 2 layers, vocab 256. Shapes below are the numpy (= reversed ggml ne) shapes of
# the tensors src/models/rwkv6.cpp load_arch_tensors() creates.
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1234
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

V, E, FF, HS, NL, XD, DD = 256, 128, 256, 64, 2, 8, 16
rng = np.random.default_rng(seed)


def rnd(*shape, std=None):
    std = std if std is not None else 1.0 / np.sqrt(shape[-1])
    return (rng.standard_normal(shape) * std).astype(np.float32)


def one(*shape, std=0.05):
    return (1.0 + rng.standard_normal(shape) * std).astype(np.float32)


w = gguf.GGUFWriter(str(out), "rwkv6")
w.add_name("tiny-rwkv6-random")
w.add_context_length(512)
w.add_embedding_length(E)
w.add_block_count(NL)
w.add_feed_forward_length(FF)
w.add_layer_norm_eps(1e-5)
w.add_layer_norm_rms_eps(1e-5)
w.add_wkv_head_size(HS)
w.add_time_mix_extra_dim(XD)
w.add_time_decay_extra_dim(DD)
w.add_token_shift_count(2)
toks = ["<pad>", "<s>", "</s>"] + ["t%d" % i for i in range(3, V - 1)] + ["t3t4"]
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("default")
w.add_token_list(toks)
w.add_token_types([gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.NORMAL] * (V - 3))
w.add_token_merges(["t3 t4"])
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_pad_token_id(0)
w.add_tensor("token_embd.weight", rnd(V, E, std=0.5))
w.add_tensor("token_embd_norm.weight", one(E))
w.add_tensor("token_embd_norm.bias", rnd(E, std=0.02))
for i in range(NL):
    p = f"blk.{i}."
    for nm in ("attn_norm", "attn_norm_2", "time_mix_ln"):
        w.add_tensor(p + nm + ".weight", one(E))
        w.add_tensor(p + nm + ".bias", rnd(E, std=0.02))
    w.add_tensor(p + "time_mix_w1.weight", rnd(XD * 5, E, std=0.05))
    w.add_tensor(p + "time_mix_w2.weight", rnd(5, E, XD, std=0.05))
    w.add_tensor(p + "time_mix_lerp_x.weight", rng.uniform(0.2, 0.8, (1, 1, E)).astype(np.float32))
    for nm in ("w", "k", "v", "r", "g"):
        w.add_tensor(p + f"time_mix_lerp_{nm}.weight", rng.uniform(0.2, 0.8, (1, 1, E)).astype(np.float32))
    w.add_tensor(p + "time_mix_first.weight", rnd(E // HS, HS, std=0.3))
    w.add_tensor(p + "time_mix_decay.weight", rng.uniform(-3.0, -1.0, (E,)).astype(np.float32))
    w.add_tensor(p + "time_mix_decay_w1.weight", rnd(DD, E, std=0.05))
    w.add_tensor(p + "time_mix_decay_w2.weight", rnd(E, DD, std=0.05))
    for nm in ("key", "value", "receptance", "gate", "output"):
        w.add_tensor(p + f"time_mix_{nm}.weight", rnd(E, E, std=0.5 / np.sqrt(E)))
    w.add_tensor(p + "channel_mix_lerp_k.weight", rng.uniform(0.2, 0.8, (1, 1, E)).astype(np.float32))
    w.add_tensor(p + "channel_mix_lerp_r.weight", rng.uniform(0.2, 0.8, (1, 1, E)).astype(np.float32))
    w.add_tensor(p + "channel_mix_key.weight", rnd(FF, E))
    w.add_tensor(p + "channel_mix_value.weight", rnd(E, FF))
    w.add_tensor(p + "channel_mix_receptance.weight", rnd(E, E))
w.add_tensor("output_norm.weight", one(E))
w.add_tensor("output_norm.bias", rnd(E, std=0.02))
w.add_tensor("output.weight", rnd(V, E, std=3.0 / np.sqrt(E)))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out)
