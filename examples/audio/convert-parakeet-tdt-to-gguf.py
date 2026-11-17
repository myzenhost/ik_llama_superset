#!/usr/bin/env python3
# REWRITE-SAFE: writes one NEW output file named on the command line; reads the .nemo, never modifies it
"""Convert an NVIDIA NeMo Parakeet-TDT checkpoint (.nemo) into ONE GGUF that llama-audio runs without a text model.

usage: convert-parakeet-tdt-to-gguf.py <model.nemo | extracted_nemo_dir> <out.gguf> [--outtype f16|f32]

The file holds two parts:
  1. the FastConformer encoder under the mtmd clip names for projector type "parakeet" (tools/mtmd/clip.cpp,
     tools/mtmd/models/parakeet.cpp) with NO sound_projection / mm.a.mlp tensors. clip then stops the graph at the
     encoder output (encoder-only mode) and llama-audio reads per-frame encoder states.
  2. the Token-and-Duration Transducer decoder under "tdt.*": the prediction network (blank-as-pad embedding +
     stacked LSTM), the joint network and the duration list, read by examples/audio/asr-tdt.cpp.

Weights load with torch.load(weights_only=True): tensors only, no pickled code runs.

Tested on nvidia/parakeet-tdt-0.6b-v3 (CC-BY-4.0; HF revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b,
parakeet-tdt-0.6b-v3.nemo sha256 3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d).
The output carries general.license so the licence travels with the weights.

Layout notes (each one is a fact about the graph that reads the tensor, not a choice made here):
  - depthwise conv (C, 1, K) -> (1, 1, C, K): ggml_ssm_conv takes the kernel as ne = [K, C]
    (same permute as conversion/nemotron.py for the HF form of the same encoder).
  - pointwise convs (O, I, 1) -> (O, I): they run as matrix multiplies.
  - subsampling conv biases (C,) -> (1, C, 1, 1): broadcast over the conv output.
  - batch norm: clip_graph_parakeet divides by sqrt(running_var) with no epsilon, so the converter stores
    running_var + eps (torch BatchNorm1d eps 1e-5) and the result equals torch's (x - mean) / sqrt(var + eps).
  - LSTM: torch gate order [i, f, g, o] is kept; bias_ih + bias_hh are stored summed (they are only ever added).
"""
import argparse
import io
import os
import sys
import tarfile
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf  # noqa: E402

BN_EPS = 1e-5


def load_nemo(src):
    """Return (config dict, state dict) from a .nemo tar or a directory it was extracted to."""
    if os.path.isdir(src):
        names = os.listdir(src)
        cfg_p = [n for n in names if n.endswith("model_config.yaml")]
        ck_p = [n for n in names if n.endswith("model_weights.ckpt")]
        if not cfg_p or not ck_p:
            raise SystemExit(f"{src}: no model_config.yaml / model_weights.ckpt")
        with open(os.path.join(src, cfg_p[0]), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sd = torch.load(os.path.join(src, ck_p[0]), map_location="cpu", weights_only=True)
    else:
        with tarfile.open(src, "r") as t:
            members = {m.name.lstrip("./"): m for m in t.getmembers() if m.isfile()}
            cfg_m = [m for n, m in members.items() if n.endswith("model_config.yaml")]
            ck_m = [m for n, m in members.items() if n.endswith("model_weights.ckpt")]
            if not cfg_m or not ck_m:
                raise SystemExit(f"{src}: no model_config.yaml / model_weights.ckpt inside")
            cfg = yaml.safe_load(t.extractfile(cfg_m[0]).read().decode("utf-8"))
            sd = torch.load(io.BytesIO(t.extractfile(ck_m[0]).read()), map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    return cfg, sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--outtype", choices=["f16", "f32"], default="f16",
                    help="type of the encoder's large matrices (norms, convs, biases and the decoder stay f32)")
    a = ap.parse_args()

    t0 = time.time()
    print(f"[load] {a.src}", flush=True)
    cfg, sd = load_nemo(a.src)
    print(f"[load] {len(sd)} tensors in {time.time() - t0:.1f}s", flush=True)
    enc, dec, joint, dcd, pre = cfg["encoder"], cfg["decoder"], cfg["joint"], cfg["decoding"], cfg["preprocessor"]

    # ---- what this converter and the graphs that read its output assume; refuse anything else ----
    need = {
        "encoder.self_attention_model": (enc.get("self_attention_model"), "rel_pos"),
        "encoder.subsampling": (enc.get("subsampling"), "dw_striding"),
        "encoder.subsampling_factor": (enc.get("subsampling_factor"), 8),
        "encoder.use_bias": (enc.get("use_bias"), False),
        "encoder.xscaling": (enc.get("xscaling"), False),
        "encoder.conv_norm_type": (enc.get("conv_norm_type"), "batch_norm"),
        "decoder.blank_as_pad": (dec.get("blank_as_pad"), True),
        "decoding.model_type": (dcd.get("model_type"), "tdt"),
        "joint.jointnet.activation": (joint["jointnet"].get("activation"), "relu"),
        "preprocessor.normalize": (pre.get("normalize"), "per_feature"),
        "preprocessor.n_fft": (pre.get("n_fft"), 512),
        "preprocessor.window_stride": (pre.get("window_stride"), 0.01),
        "preprocessor.window_size": (pre.get("window_size"), 0.025),
        "sample_rate": (cfg.get("sample_rate"), 16000),
    }
    bad = [f"{k}={v!r} (need {w!r})" for k, (v, w) in need.items() if v != w]
    if bad:
        raise SystemExit("unsupported checkpoint: " + "; ".join(bad))

    n_layer = int(enc["n_layers"])
    d_model = int(enc["d_model"])
    n_head = int(enc["n_heads"])
    n_ff = d_model * int(enc["ff_expansion_factor"])
    n_mel = int(enc["feat_in"])
    kconv = int(enc["conv_kernel_size"])
    vocab = list(joint["vocabulary"])
    n_vocab = int(joint["num_classes"])
    durations = [int(x) for x in dcd.get("durations") or cfg["model_defaults"]["tdt_durations"]]
    n_pred = int(dec["prednet"]["pred_hidden"])
    n_pred_layers = int(dec["prednet"]["pred_rnn_layers"])
    n_joint = int(joint["jointnet"]["joint_hidden"])
    max_symbols = int((dcd.get("greedy") or {}).get("max_symbols") or 10)
    if len(vocab) != n_vocab:
        raise SystemExit(f"vocabulary has {len(vocab)} pieces, num_classes {n_vocab}")
    if int(joint.get("num_extra_outputs", 0)) != len(durations):
        raise SystemExit(f"num_extra_outputs {joint.get('num_extra_outputs')} != {len(durations)} durations")

    emb = sd["decoder.prediction.embed.weight"]
    if emb.shape[0] != n_vocab + 1:
        raise SystemExit(f"embedding rows {emb.shape[0]} != vocab + blank {n_vocab + 1}")
    print(f"blank row (id {n_vocab}) max |x| = {float(emb[n_vocab].abs().max()):.3g} (blank_as_pad -> 0)", flush=True)
    out_rows = sd["joint.joint_net.2.weight"].shape[0]
    if out_rows != n_vocab + 1 + len(durations):
        raise SystemExit(f"joint output rows {out_rows} != vocab + blank + durations {n_vocab + 1 + len(durations)}")

    big = gguf.GGMLQuantizationType.F16 if a.outtype == "f16" else gguf.GGMLQuantizationType.F32
    w = gguf.GGUFWriter(a.out, arch="clip")
    w.add_string("general.name", cfg.get("name") or "parakeet-tdt")
    w.add_string("general.description", "Parakeet-TDT: FastConformer encoder (mtmd parakeet, encoder-only) + TDT decoder (tdt.*)")
    w.add_string("general.license", "cc-by-4.0")
    w.add_uint32("general.file_type", 1 if a.outtype == "f16" else 0)

    # encoder (clip)
    w.add_bool("clip.has_vision_encoder", False)
    w.add_bool("clip.has_audio_encoder", True)
    w.add_string("clip.audio.projector_type", "parakeet")
    w.add_uint32("clip.audio.embedding_length", d_model)
    w.add_uint32("clip.audio.feed_forward_length", n_ff)
    w.add_uint32("clip.audio.block_count", n_layer)
    w.add_uint32("clip.audio.attention.head_count", n_head)
    w.add_uint32("clip.audio.projection_dim", d_model)  # encoder-only: the output is the encoder state
    w.add_float32("clip.audio.attention.layer_norm_epsilon", 1e-5)
    w.add_uint32("clip.audio.num_mel_bins", n_mel)
    w.add_uint32("clip.audio.subsampling_factor", int(enc["subsampling_factor"]))
    w.add_uint32("clip.audio.conv_kernel_size", kconv)

    # decoder (tdt)
    w.add_uint32("tdt.vocab_size", n_vocab)
    w.add_uint32("tdt.blank_id", n_vocab)
    w.add_array("tdt.durations", durations)
    w.add_uint32("tdt.max_symbols_per_step", max_symbols)
    w.add_uint32("tdt.pred_hidden", n_pred)
    w.add_uint32("tdt.pred_layers", n_pred_layers)
    w.add_uint32("tdt.joint_hidden", n_joint)
    w.add_uint32("tdt.encoder_hidden", d_model)
    w.add_uint32("tdt.subsampling_factor", int(enc["subsampling_factor"]))
    w.add_float32("tdt.frame_seconds", float(pre["window_stride"]))
    w.add_array("tdt.vocab", vocab)

    n_written = 0

    def put(name, t, typ=gguf.GGMLQuantizationType.F32):
        nonlocal n_written
        arr = t.detach().to(torch.float32).contiguous().numpy()
        if typ == gguf.GGMLQuantizationType.F16:
            arr = arr.astype(np.float16)
        w.add_tensor(name, arr, raw_dtype=typ)
        n_written += 1

    used = set()

    def get(k):
        used.add(k)
        return sd[k]

    # front end
    put("a.mel_filters", get("preprocessor.featurizer.fb").reshape(n_mel, -1))
    put("a.window", get("preprocessor.featurizer.window"))

    # subsampling
    for i in (0, 2, 3, 5, 6):
        put(f"a.conv1d.{i}.weight", get(f"encoder.pre_encode.conv.{i}.weight"))
        put(f"a.conv1d.{i}.bias", get(f"encoder.pre_encode.conv.{i}.bias").reshape(1, -1, 1, 1))
    put("a.pre_encode.out.weight", get("encoder.pre_encode.out.weight"), big)
    put("a.pre_encode.out.bias", get("encoder.pre_encode.out.bias"))

    for il in range(n_layer):
        p = f"encoder.layers.{il}."
        b = f"a.blk.{il}."
        for src, dst in (("norm_feed_forward1", "ffn_norm"), ("norm_conv", "norm_conv"), ("norm_self_att", "ln1"),
                         ("norm_feed_forward2", "ffn_norm_1"), ("norm_out", "ln2")):
            put(b + dst + ".weight", get(p + src + ".weight"))
            put(b + dst + ".bias", get(p + src + ".bias"))
        put(b + "ffn_up.weight", get(p + "feed_forward1.linear1.weight"), big)
        put(b + "ffn_down.weight", get(p + "feed_forward1.linear2.weight"), big)
        put(b + "ffn_up_1.weight", get(p + "feed_forward2.linear1.weight"), big)
        put(b + "ffn_down_1.weight", get(p + "feed_forward2.linear2.weight"), big)
        put(b + "conv_pw1.weight", get(p + "conv.pointwise_conv1.weight").squeeze(-1), big)
        put(b + "conv_pw2.weight", get(p + "conv.pointwise_conv2.weight").squeeze(-1), big)
        put(b + "conv_dw.weight", get(p + "conv.depthwise_conv.weight").unsqueeze(-1).permute(3, 1, 0, 2))
        put(b + "conv_norm.weight", get(p + "conv.batch_norm.weight"))
        put(b + "conv_norm.bias", get(p + "conv.batch_norm.bias"))
        put(b + "conv_norm_mean", get(p + "conv.batch_norm.running_mean"))
        put(b + "conv_norm_var", get(p + "conv.batch_norm.running_var") + BN_EPS)
        used.add(p + "conv.batch_norm.num_batches_tracked")
        put(b + "attn_q.weight", get(p + "self_attn.linear_q.weight"), big)
        put(b + "attn_k.weight", get(p + "self_attn.linear_k.weight"), big)
        put(b + "attn_v.weight", get(p + "self_attn.linear_v.weight"), big)
        put(b + "attn_out.weight", get(p + "self_attn.linear_out.weight"), big)
        put(b + "linear_pos.weight", get(p + "self_attn.linear_pos.weight"), big)
        put(b + "pos_bias_u", get(p + "self_attn.pos_bias_u"))
        put(b + "pos_bias_v", get(p + "self_attn.pos_bias_v"))
        print(f"[encoder] layer {il + 1}/{n_layer} staged ({n_written} tensors, {time.time() - t0:.1f}s)", flush=True)

    # transducer decoder
    put("tdt.pred.embd.weight", get("decoder.prediction.embed.weight"))
    for l in range(n_pred_layers):
        q = "decoder.prediction.dec_rnn.lstm."
        put(f"tdt.pred.lstm.{l}.w_ih", get(f"{q}weight_ih_l{l}"))
        put(f"tdt.pred.lstm.{l}.w_hh", get(f"{q}weight_hh_l{l}"))
        put(f"tdt.pred.lstm.{l}.b", get(f"{q}bias_ih_l{l}") + get(f"{q}bias_hh_l{l}"))
    put("tdt.joint.pred.weight", get("joint.pred.weight"))
    put("tdt.joint.pred.bias", get("joint.pred.bias"))
    put("tdt.joint.enc.weight", get("joint.enc.weight"))
    put("tdt.joint.enc.bias", get("joint.enc.bias"))
    put("tdt.joint.out.weight", get("joint.joint_net.2.weight"))
    put("tdt.joint.out.bias", get("joint.joint_net.2.bias"))

    left = sorted(k for k in sd if k not in used)
    if left:
        raise SystemExit(f"{len(left)} checkpoint tensor(s) not converted, first: {left[:5]}")

    print(f"[write] {a.out}", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"wrote {a.out}: {n_written} tensors from {len(sd)} checkpoint tensors; "
          f"{n_layer} layers d={d_model}, vocab {n_vocab} + blank, durations {durations}, max_symbols {max_symbols} "
          f"({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
