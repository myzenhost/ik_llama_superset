#!/usr/bin/env python3
"""Export the audio encoder + projector of a whisper-style mmproj GGUF to an OpenVINO IR.

The IR is the model an mtmd encoder provider loads (tools/mtmd/providers/openvino, contract in
tools/mtmd/mtmd-encoder-provider.h): input f32 [1][n_mel][n_step], output the SAME tensor tools/mtmd/models/whisper-enc.cpp
would produce for that mel, i.e. [n_tokens][n_embd_out] row-major.

The graph is written from tools/mtmd/models/whisper-enc.cpp + clip.cpp build_vit(), read in full 2026-10-05:
  conv1d(k3,s1,p1)+bias -> gelu_erf -> conv1d(k3,s2,p1)+bias -> gelu_erf -> transpose -> + learned position embedding
  -> n_layer x { ln1 -> q(+bias), k(NO bias), v(+bias) -> full softmax attention, scale 1/sqrt(d_head) -> out proj(+bias)
                 -> residual -> ln2 -> up(+bias) -> gelu_erf -> down(+bias) -> residual }
  -> avg-pool(2) over frames (voxtral: audio_has_avgpool, clip.cpp:553-561) -> post layernorm -> StackAudioFrames(factor)
  (tail frames beyond a multiple of the factor dropped, as ggml's GGML_PAD-based build_stack does) -> projector.
Supported projector heads: voxtral (mm.a.mlp.1 -> gelu_erf -> mm.a.mlp.2, no biases). A head that is not listed makes the
script stop with a message instead of exporting a wrong graph.

Weights are dequantised to float32 (so the IR holds exactly the values the ggml graph dequantises on the fly; the only
numerical gap left is ggml quantising the ACTIVATIONS for a Q8_0 matmul).

usage: mtmd-export-whisper-openvino.py <mmproj.gguf> <out.xml> [--n-step 3000] [--openvino-path <dir with the openvino package>]
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gguf-py"))
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    if f.types and f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode("utf8")
    return f.parts[f.data[0]][0].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mmproj")
    ap.add_argument("out_xml")
    ap.add_argument("--n-step", type=int, default=3000, help="mel frames per chunk (whisper: 3000 = 30 s)")
    ap.add_argument("--openvino-path", default="")
    args = ap.parse_args()
    if args.openvino_path and os.path.isdir(args.openvino_path):
        sys.path.insert(0, args.openvino_path)

    import torch
    import torch.nn.functional as F
    import openvino as ov

    r = gguf.GGUFReader(args.mmproj)
    proj = field(r, "clip.projector_type")
    if proj != "voxtral":
        sys.exit("projector '%s' has no exporter head yet (supported: voxtral); add its mm.* head from whisper-enc.cpp" % proj)
    n_embd = field(r, "clip.audio.embedding_length")
    n_layer = field(r, "clip.audio.block_count")
    n_head = field(r, "clip.audio.attention.head_count")
    n_mel = field(r, "clip.audio.num_mel_bins")
    eps = field(r, "clip.audio.attention.layer_norm_epsilon")
    stack = field(r, "clip.audio.projector.stack_factor", 1)
    d_head = n_embd // n_head
    print("voxtral encoder: n_embd %d layers %d heads %d n_mel %d eps %g stack %d" % (n_embd, n_layer, n_head, n_mel, eps, stack), flush=True)

    tens = {t.name: t for t in r.tensors}

    def W(name):
        t = tens[name]
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            a = np.asarray(t.data, dtype=np.float32)
        else:
            a = dequantize(t.data, t.tensor_type).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(a.reshape(tuple(int(x) for x in reversed(t.shape)))))

    def P(name):
        return torch.nn.Parameter(W(name), requires_grad=False)

    class Block(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            p = "a.blk.%d." % i
            self.ln1_w, self.ln1_b = P(p + "ln1.weight"), P(p + "ln1.bias")
            self.ln2_w, self.ln2_b = P(p + "ln2.weight"), P(p + "ln2.bias")
            self.q_w, self.q_b = P(p + "attn_q.weight"), P(p + "attn_q.bias")
            self.k_w = P(p + "attn_k.weight")                              # no k bias (whisper-enc.cpp:32)
            self.v_w, self.v_b = P(p + "attn_v.weight"), P(p + "attn_v.bias")
            self.o_w, self.o_b = P(p + "attn_out.weight"), P(p + "attn_out.bias")
            self.up_w, self.up_b = P(p + "ffn_up.weight"), P(p + "ffn_up.bias")
            self.dn_w, self.dn_b = P(p + "ffn_down.weight"), P(p + "ffn_down.bias")

        def forward(self, x):                                              # x [1, T, C]
            h = F.layer_norm(x, (n_embd,), self.ln1_w, self.ln1_b, eps)
            q = F.linear(h, self.q_w, self.q_b)
            k = F.linear(h, self.k_w)
            v = F.linear(h, self.v_w, self.v_b)
            T = x.shape[1]
            q = q.view(1, T, n_head, d_head).transpose(1, 2)
            k = k.view(1, T, n_head, d_head).transpose(1, 2)
            v = v.view(1, T, n_head, d_head).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(d_head))
            a = a.transpose(1, 2).reshape(1, T, n_embd)
            x = x + F.linear(a, self.o_w, self.o_b)
            h = F.layer_norm(x, (n_embd,), self.ln2_w, self.ln2_b, eps)
            h = F.linear(F.gelu(F.linear(h, self.up_w, self.up_b)), self.dn_w, self.dn_b)   # gelu = erf form
            return x + h

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1_w, self.c1_b = P("a.conv1d.1.weight"), P("a.conv1d.1.bias")
            self.c2_w, self.c2_b = P("a.conv1d.2.weight"), P("a.conv1d.2.bias")
            self.c1_b.data = self.c1_b.data.reshape(-1)
            self.c2_b.data = self.c2_b.data.reshape(-1)
            self.pos = P("a.position_embd.weight")                         # [n_pos_max, C]
            self.blocks = torch.nn.ModuleList()
            t0 = time.time()
            for i in range(n_layer):
                self.blocks.append(Block(i))
                done = i + 1
                eta = (time.time() - t0) / done * (n_layer - done)
                print("[load] block %d/%d ETA=%.0fs" % (done, n_layer, eta), flush=True)
            self.pl_w, self.pl_b = P("a.post_ln.weight"), P("a.post_ln.bias")
            self.m1 = P("mm.a.mlp.1.weight")
            self.m2 = P("mm.a.mlp.2.weight")

        def forward(self, mel):                                            # [1, n_mel, n_step]
            x = F.gelu(F.conv1d(mel, self.c1_w, self.c1_b, stride=1, padding=1))
            x = F.gelu(F.conv1d(x, self.c2_w, self.c2_b, stride=2, padding=1))
            x = x.transpose(1, 2)                                          # [1, n_step/2, C]
            T = x.shape[1]
            x = x + self.pos[:T].unsqueeze(0)
            for b in self.blocks:
                x = b(x)
            # build_vit (clip.cpp:553-561): voxtral has audio_has_avgpool(), AvgPool1d(2, stride 2) over frames, BEFORE post_ln
            x = F.avg_pool1d(x.transpose(1, 2), 2, 2).transpose(1, 2)      # [1, T/2, C]
            x = F.layer_norm(x, (n_embd,), self.pl_w, self.pl_b, eps)      # post layernorm (build_vit)
            # StackAudioFrames (clip.cpp:900-918): consecutive frames concatenated. build_stack pads with GGML_PAD, a
            # power-of-two bitmask, to a multiple of stride = n_embd*stack (5120, NOT a power of two): the "padded" length
            # lands BELOW the data length, view_2d then keeps floor(T/stack) rows and the tail frames are dropped. The
            # token count the loader expects (clip_n_output_tokens = floor(n_step/stack/2/2)) is the same floor, so cut to it.
            T2 = x.shape[1]
            x = x[:, : (T2 // stack) * stack, :]
            x = x.reshape(1, T2 // stack, n_embd * stack)
            x = F.linear(F.gelu(F.linear(x, self.m1)), self.m2)            # voxtral projector, no biases
            return x

    model = Enc().eval()
    n_pos = args.n_step // 2
    assert n_pos % stack == 0, "n_step/2 must be divisible by the stack factor"
    example = torch.zeros(1, n_mel, args.n_step)
    with torch.no_grad():
        ref = model(example)
    print("torch output shape", tuple(ref.shape), flush=True)
    # clip.cpp clip_n_output_tokens (voxtral): n_step -> align to stack, / stack, / 2 (conv stride), / 2 (avg pool).
    # A mismatch here is exactly the failure the loader reports as "model output holds N floats, projector expects M".
    expect_tok = ((-(-args.n_step // stack)) // 2) // 2
    assert ref.shape[1] == expect_tok, "exported graph yields %d tokens, clip_n_output_tokens gives %d" % (ref.shape[1], expect_tok)
    ovm = ov.convert_model(model, example_input=example)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_xml)), exist_ok=True)
    ov.save_model(ovm, args.out_xml, compress_to_fp16=False)
    print("saved", args.out_xml, "output floats", int(np.prod(ref.shape)), flush=True)


if __name__ == "__main__":
    main()
