#!/usr/bin/env python3
"""Random-weight Ling 3.0 VL mmproj GGUF (projector "ling3vl"; tiny, CPU, seconds).

Writes the tensor set tools/mtmd/models/ling3vl.cpp and clip_model_loader read for PROJECTOR_TYPE_LING3VL, filled with seeded
random numbers: the qwen3vl-style tower (two patch convs, learned position embedding, fused qkv, GELU MLP), the norm-only
merger (mm.input_norm) and the mm.0 / mm.2 projector. It exercises the projector LOAD, the graph BUILD and the output token
count; it proves nothing about image understanding. Names are the loader's (tools/mtmd/clip-impl.h TN_* / KEY_*), which are
the ones ik_llama 719f4aa5d examples/mtmd/clip.cpp reads.

    python make_tiny_ling3vl_mmproj.py --out mmproj-tiny-ling3vl.gguf --n-out 128
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# <repo>/tools/mtmd/tests -> <repo>/gguf-py
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-out", type=int, default=128, help="LLM embedding width (must equal the text model's n_embd)")
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--n-ff", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=96, help="projector MLP hidden width")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)

    def rnd(*shape, scale=0.1, dtype=np.float32):
        return (rng.standard_normal(shape) * scale).astype(dtype)

    E, F, P, H = a.n_embd, a.n_ff, a.patch, a.hidden
    if (E // a.n_head) % 4 != 0:
        raise SystemExit("head dim must be a multiple of 4 (vision M-RoPE quarters)")
    n_pos = (a.image_size // P) ** 2

    w = gguf.GGUFWriter(a.out, "clip")
    w.add_name("tiny-ling3vl-vision-random")
    w.add_description("random-weight ling3vl mmproj for load/graph/token-count checks only")

    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "ling3vl")
    w.add_uint32("clip.vision.image_size", a.image_size)
    w.add_uint32("clip.vision.patch_size", P)
    w.add_uint32("clip.vision.embedding_length", E)
    w.add_uint32("clip.vision.feed_forward_length", F)
    w.add_uint32("clip.vision.projection_dim", a.n_out)
    w.add_uint32("clip.vision.block_count", a.n_layer)
    w.add_uint32("clip.vision.attention.head_count", a.n_head)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    w.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    w.add_uint32("clip.vision.spatial_merge_size", 2)
    w.add_bool("clip.use_gelu", True)

    # tower (numpy shapes are the reverse of ggml ne)
    w.add_tensor("v.patch_embd.weight", rnd(E, 3, P, P, dtype=np.float16))
    w.add_tensor("v.patch_embd.weight.1", rnd(E, 3, P, P, dtype=np.float16))
    w.add_tensor("v.patch_embd.bias", rnd(E))
    w.add_tensor("v.position_embd.weight", rnd(n_pos, E))
    for i in range(a.n_layer):
        b = f"v.blk.{i}."
        w.add_tensor(b + "ln1.weight", 1.0 + rnd(E, scale=0.05))
        w.add_tensor(b + "ln1.bias", rnd(E, scale=0.05))
        w.add_tensor(b + "ln2.weight", 1.0 + rnd(E, scale=0.05))
        w.add_tensor(b + "ln2.bias", rnd(E, scale=0.05))
        w.add_tensor(b + "attn_qkv.weight", rnd(3 * E, E))
        w.add_tensor(b + "attn_qkv.bias", rnd(3 * E))
        w.add_tensor(b + "attn_out.weight", rnd(E, E))
        w.add_tensor(b + "attn_out.bias", rnd(E))
        w.add_tensor(b + "ffn_up.weight", rnd(F, E))
        w.add_tensor(b + "ffn_up.bias", rnd(F))
        w.add_tensor(b + "ffn_down.weight", rnd(E, F))
        w.add_tensor(b + "ffn_down.bias", rnd(E))
    w.add_tensor("v.post_ln.weight", 1.0 + rnd(E, scale=0.05))
    w.add_tensor("v.post_ln.bias", rnd(E, scale=0.05))
    # norm-only merger (LayerNorm over n_embd, before the 2x2 merge) + projector MLP over the merged 4*E
    w.add_tensor("mm.input_norm.weight", 1.0 + rnd(E, scale=0.05))
    w.add_tensor("mm.input_norm.bias", rnd(E, scale=0.05))
    w.add_tensor("mm.0.weight", rnd(H, 4 * E))
    w.add_tensor("mm.0.bias", rnd(H))
    w.add_tensor("mm.2.weight", rnd(a.n_out, H))
    w.add_tensor("mm.2.bias", rnd(a.n_out))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
