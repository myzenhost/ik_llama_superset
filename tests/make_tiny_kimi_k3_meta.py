#!/usr/bin/env python3
# Metadata-only Kimi-K3 GGUFs (one dummy tensor, NO weights, NO vocab) for tests/test-kimi-k3-required-keys.cpp.
# The loader reads the hyperparameters before the vocab and the tensors, so a file that is wrong in one hparam is refused
# at the hparam stage and the refusal text names that key. RANDOM/EMPTY CONTENT ONLY: nothing here is a model.
#
# Usage: python make_tiny_kimi_k3_meta.py <worktree-root> <out-dir>
# Writes <out-dir>/k3_ok.gguf (every key present), k3_no_latent.gguf (expert_latent_length absent),
#        k3_zero_latent.gguf (expert_latent_length = 0), k3_zero_block.gguf (attn_res.block_size = 0).
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "gguf-py"))

import numpy as np  # noqa: E402
import gguf  # noqa: E402

ARCH = "kimi-k3"
N_LAYER, N_EMBD, N_HEAD = 93, 64, 4


def write(name, latent, block):
    w = gguf.GGUFWriter(str(OUT / name), ARCH)
    w.add_uint32(f"{ARCH}.context_length", 128)
    w.add_uint32(f"{ARCH}.embedding_length", N_EMBD)
    w.add_uint32(f"{ARCH}.block_count", N_LAYER)
    w.add_uint32(f"{ARCH}.feed_forward_length", 128)
    w.add_uint32(f"{ARCH}.expert_feed_forward_length", 32)
    w.add_uint32(f"{ARCH}.expert_count", 4)
    w.add_uint32(f"{ARCH}.expert_used_count", 2)
    w.add_uint32(f"{ARCH}.expert_shared_count", 1)
    w.add_uint32(f"{ARCH}.leading_dense_block_count", 1)
    w.add_uint32(f"{ARCH}.expert_gating_func", 2)
    w.add_float32(f"{ARCH}.expert_weights_scale", 1.0)
    w.add_bool(f"{ARCH}.expert_weights_norm", True)
    w.add_uint32(f"{ARCH}.attention.head_count", N_HEAD)
    # n_head_kv == 0 marks a KDA (recurrent) layer
    w.add_array(f"{ARCH}.attention.head_count_kv", [0 if i % 4 else 1 for i in range(N_LAYER)])
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32(f"{ARCH}.attention.key_length_mla", 16)
    w.add_uint32(f"{ARCH}.attention.value_length_mla", 16)
    w.add_uint32(f"{ARCH}.attention.kv_lora_rank", 16)
    w.add_uint32(f"{ARCH}.attention.q_lora_rank", 16)
    w.add_uint32(f"{ARCH}.attention.key_length", 16)
    w.add_uint32(f"{ARCH}.attention.value_length", 16)
    w.add_uint32(f"{ARCH}.rope.dimension_count", 8)
    w.add_uint32(f"{ARCH}.ssm.conv_kernel", 4)
    w.add_uint32(f"{ARCH}.kda.head_dim", 16)
    w.add_float32(f"{ARCH}.activation.situ_beta", 1.0)
    w.add_float32(f"{ARCH}.activation.situ_linear_beta", 1.0)
    w.add_uint32(f"{ARCH}.attn_res.block_size", block)
    if latent is not None:
        w.add_uint32(f"{ARCH}.expert_latent_length", latent)
    # the loader reads the vocab size from the token list while loading hparams, so a tiny vocab must exist
    w.add_tokenizer_model("gpt2")
    w.add_token_list(["<unk>", "a", "b", "c"])
    w.add_tensor("token_embd.weight", np.zeros((8, N_EMBD), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", OUT / name)


write("k3_ok.gguf", 32, 4)
write("k3_no_latent.gguf", None, 4)
write("k3_zero_latent.gguf", 0, 4)
write("k3_zero_block.gguf", 32, 0)
