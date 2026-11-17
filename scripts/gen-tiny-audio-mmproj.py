#!/usr/bin/env python3
"""Random-weight audio multimodal projector for mtmd/clip (projector type qwen2a = Whisper-style encoder + one linear).

Writes OUT_DIR/mmproj-tiny-audio.gguf, to be paired with the 64-dim tiny LLM from scripts/gen-tiny-lora-test-assets.py.
qwen2a was picked because it needs no stack factor and the smallest projector (a single mm.a.fc linear): the loader
needs only conv1d x2, a.position_embd, per-layer q/k/v/o + ln1/ln2 + ffn up/down, a.post_ln and mm.a.fc.
Shapes follow clip_graph_whisper_enc: 30 s chunk = 3000 mel frames -> conv stride 2 -> 1500 positions.

Usage: PYTHONPATH=gguf-py python3 scripts/gen-tiny-audio-mmproj.py OUT_DIR [--n-embd-llm 64] [--n-mel 80]
Needs numpy and gguf-py (in this repo).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gguf-py"))
import gguf  # noqa: E402

D, N_HEAD, N_FF, N_LAYER, N_POS = 32, 4, 64, 2, 1500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n-embd-llm", type=int, default=64, help="n_embd of the LLM the projector feeds")
    ap.add_argument("--n-mel", type=int, default=80)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--projector", choices=["qwen2a", "minicpmoa"], default="qwen2a",
                    help="minicpmoa = MiniCPM-o 4.5 audio: mm.a.mlp.1/2 (linear-relu-linear) + avgpool(--pool-step)")
    ap.add_argument("--pool-step", type=int, default=5, help="minicpmoa only: clip.audio.projector.pool_step")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "mmproj-tiny-audio.gguf" if a.projector == "qwen2a" else "mmproj-tiny-audio-%s.gguf" % a.projector)
    rng = np.random.default_rng(a.seed)

    def rnd(*shape, s=0.05):
        return rng.normal(0.0, s, size=shape).astype(np.float32)

    ones = lambda n: np.ones(n, dtype=np.float32)
    zeros = lambda n: np.zeros(n, dtype=np.float32)

    w = gguf.GGUFWriter(path, "clip")
    w.add_name("tiny-random-audio-mmproj")
    w.add_bool("clip.has_audio_encoder", True)
    w.add_bool("clip.has_vision_encoder", False)
    w.add_string("clip.projector_type", a.projector)
    if a.projector == "minicpmoa":
        w.add_uint32("clip.audio.projector.pool_step", a.pool_step)
    w.add_uint32("clip.audio.embedding_length", D)
    w.add_uint32("clip.audio.feed_forward_length", N_FF)
    w.add_uint32("clip.audio.block_count", N_LAYER)
    w.add_uint32("clip.audio.attention.head_count", N_HEAD)
    w.add_uint32("clip.audio.projection_dim", a.n_embd_llm)
    w.add_float32("clip.audio.attention.layer_norm_epsilon", 1e-5)
    w.add_uint32("clip.audio.num_mel_bins", a.n_mel)

    # conv1d: numpy (out, in, k) -> ggml ne [k, in, out]; bias (out,1) -> ne [1, out] so it broadcasts over time
    w.add_tensor("a.conv1d.1.weight", rnd(D, a.n_mel, 3))
    w.add_tensor("a.conv1d.1.bias", rnd(D, 1, s=0.01))
    w.add_tensor("a.conv1d.2.weight", rnd(D, D, 3))
    w.add_tensor("a.conv1d.2.bias", rnd(D, 1, s=0.01))
    w.add_tensor("a.position_embd.weight", rnd(N_POS, D, s=0.02))
    for i in range(N_LAYER):
        p = "a.blk.%d." % i
        w.add_tensor(p + "ln1.weight", ones(D)); w.add_tensor(p + "ln1.bias", zeros(D))
        w.add_tensor(p + "ln2.weight", ones(D)); w.add_tensor(p + "ln2.bias", zeros(D))
        w.add_tensor(p + "attn_q.weight", rnd(D, D)); w.add_tensor(p + "attn_q.bias", rnd(D, s=0.01))
        w.add_tensor(p + "attn_k.weight", rnd(D, D))                                   # whisper: no k bias (asserted)
        w.add_tensor(p + "attn_v.weight", rnd(D, D)); w.add_tensor(p + "attn_v.bias", rnd(D, s=0.01))
        w.add_tensor(p + "attn_out.weight", rnd(D, D)); w.add_tensor(p + "attn_out.bias", rnd(D, s=0.01))
        w.add_tensor(p + "ffn_up.weight", rnd(N_FF, D)); w.add_tensor(p + "ffn_up.bias", rnd(N_FF, s=0.01))
        w.add_tensor(p + "ffn_down.weight", rnd(D, N_FF)); w.add_tensor(p + "ffn_down.bias", rnd(D, s=0.01))
    w.add_tensor("a.post_ln.weight", ones(D))
    w.add_tensor("a.post_ln.bias", zeros(D))
    if a.projector == "minicpmoa":
        # audio_projector.linear1 (D -> n_embd_llm), relu, linear2 (n_embd_llm -> n_embd_llm)
        w.add_tensor("mm.a.mlp.1.weight", rnd(a.n_embd_llm, D))
        w.add_tensor("mm.a.mlp.1.bias", rnd(a.n_embd_llm, s=0.01))
        w.add_tensor("mm.a.mlp.2.weight", rnd(a.n_embd_llm, a.n_embd_llm))
        w.add_tensor("mm.a.mlp.2.bias", rnd(a.n_embd_llm, s=0.01))
    else:
        w.add_tensor("mm.a.fc.weight", rnd(a.n_embd_llm, D))   # ne [D, n_embd_llm]
        w.add_tensor("mm.a.fc.bias", rnd(a.n_embd_llm, s=0.01))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", path)


if __name__ == "__main__":
    main()
