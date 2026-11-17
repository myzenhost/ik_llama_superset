#!/usr/bin/env python3
"""Random-weight DeepSeek-V4 / V4.1 vision mmproj GGUF (tiny, CPU, seconds).

There is no trained DeepSeek-V4.x vision checkpoint on this host and ik_llama ships no converter for one, so this writes the
tensor set that tools/mtmd/models/deepseek4v.cpp and clip_model_loader expect, filled with seeded random numbers. It exists to
exercise the projector LOAD, the graph BUILD and the token-block LAYOUT -- it proves nothing about image understanding.

Tensor names / KV keys are the ones the loader reads (tools/mtmd/clip-impl.h: TN_* / KEY_*), which are the ones ik_llama's
examples/mtmd/clip.cpp reads (ik 60c74d9c4 / 248e77729).

    python make_tiny_dsv4_mmproj.py --variant 41 --out mmproj-tiny-dsv41.gguf --n-out 64
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
    ap.add_argument("--variant", choices=["41", "4v"], default="41")
    ap.add_argument("--n-out", type=int, default=64, help="LLM embedding width (must equal the text model's n_embd)")
    ap.add_argument("--n-embd", type=int, default=32)
    ap.add_argument("--n-head", type=int, default=2)
    ap.add_argument("--n-ff", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--merge", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=48, help="aligner MLP hidden width")
    ap.add_argument("--min-pixels", type=int, default=0)
    ap.add_argument("--max-n-token", type=int, default=None, help="deepseek41 only: clip.vision.dsv4_max_n_token (omit = key absent)")
    ap.add_argument("--no-pad", action="store_true", help="omit v.token_embd.img_pad (valid for 41 only)")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)

    def rnd(*shape, scale=0.1, dtype=np.float32):
        return (rng.standard_normal(shape) * scale).astype(dtype)

    E, F, P, M = a.n_embd, a.n_ff, a.patch, a.merge
    w = gguf.GGUFWriter(a.out, "clip")  # the constructor writes general.architecture
    w.add_name("tiny-dsv4-vision-random")
    w.add_description("random-weight deepseek4v/deepseek41v mmproj for load/graph/layout checks only")

    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "deepseek41v" if a.variant == "41" else "deepseek4v")  # canonical names; "deepseek41" is a read-only legacy alias
    w.add_uint32("clip.vision.image_size", 64)
    w.add_uint32("clip.vision.patch_size", P)
    w.add_uint32("clip.vision.embedding_length", E)
    w.add_uint32("clip.vision.feed_forward_length", F)
    w.add_uint32("clip.vision.projection_dim", a.n_out)
    w.add_uint32("clip.vision.block_count", a.n_layer)
    w.add_uint32("clip.vision.attention.head_count", a.n_head)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    w.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    w.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    w.add_uint32("clip.vision.projector.scale_factor", M)
    w.add_uint32("clip.vision.image_min_pixels", a.min_pixels)
    w.add_bool("clip.use_silu", True)  # SwiGLU
    if a.max_n_token is not None:
        w.add_uint32("clip.vision.dsv4_max_n_token", a.max_n_token)

    # ViT
    w.add_tensor("v.patch_embd.weight", rnd(E, 3, P, P, dtype=np.float16))
    for i in range(a.n_layer):
        b = f"v.blk.{i}."
        w.add_tensor(b + "ln1.weight", 1.0 + rnd(E, scale=0.05))
        w.add_tensor(b + "ln2.weight", 1.0 + rnd(E, scale=0.05))
        for n in ("attn_q", "attn_k", "attn_v", "attn_out"):
            w.add_tensor(b + n + ".weight", rnd(E, E))
        w.add_tensor(b + "ffn_up.weight", rnd(F, E))
        w.add_tensor(b + "ffn_gate.weight", rnd(F, E))
        w.add_tensor(b + "ffn_down.weight", rnd(E, F))
    # aligner: n_merge^2 patches concatenated -> hidden -> LLM width
    w.add_tensor("mm.1.weight", rnd(a.hidden, E * M * M))
    w.add_tensor("mm.1.bias", rnd(a.hidden))
    w.add_tensor("mm.2.weight", rnd(a.n_out, a.hidden))
    w.add_tensor("mm.2.bias", rnd(a.n_out))
    # sentinels (distinct, large enough that no aligner row can equal one by accident)
    w.add_tensor("v.image_newline", rnd(a.n_out, scale=1.0))
    w.add_tensor("v.token_embd.img_start", rnd(a.n_out, scale=1.0))
    w.add_tensor("v.token_embd.img_end", rnd(a.n_out, scale=1.0))
    if not a.no_pad:
        w.add_tensor("v.token_embd.img_pad", rnd(a.n_out, scale=1.0))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
