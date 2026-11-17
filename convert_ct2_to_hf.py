#!/usr/bin/env python3
"""Turn a CTranslate2 model directory (model.bin + config.json + vocabulary.json) into a Hugging Face LLaMA-style directory
(config.json + model.safetensors [+ tokenizer files]) that convert_hf_to_gguf.py reads.

Written from the behaviour of OpenNMT/CTranslate2 @ f2f54b42 (MIT; no code copied), SOURCE-READ:
  python/ctranslate2/specs/model_spec.py:364-412   model.bin layout: uint32 binary version (1..6), string spec name, uint32 spec revision,
                                                    uint32 variable count, then per variable: string name, uint8 rank, rank x uint32 dims,
                                                    uint8 dtype id, uint32 byte count, raw little-endian data; then uint32 alias count and
                                                    (alias, target) string pairs. A string is uint16 length INCLUDING a trailing NUL, then bytes.
  src/models/model.cc:50-92,595-684,790-798       the reader the C++ engine uses (same layout; binary version < 4 stores an item size instead
                                                    of a dtype id; aliases also alias `<name>_scale` / `<name>_zero`).
  include/ctranslate2/types.h:16-23                dtype ids: 0 float32, 1 int8, 2 int16, 3 int32, 4 float16, 5 bfloat16.
  python/ctranslate2/specs/model_spec.py:191-258   weight quantisation: int8 = per-ROW scale 127/amax (all-zero rows use amax 127), q = rint(w*scale);
                                                    int16 = ONE scale 2^10/amax per tensor; both stored as `<name>_scale`; float variables that are not
                                                    quantisable are stored in the "other" float type (float16/bfloat16/float32).
  src/ops/dequantize_cpu.cc:11-50                  dequantisation = q * (1/scale), the reciprocal taken in float32 first.
  src/models/model.cc:310-323                      an int16 variable without a saved scale uses the global scale 1000.
  python/ctranslate2/converters/transformers.py:1698-1840  LlamaLoader: decoder/embeddings, decoder/layer_N/{self_attention/{layer_norm,linear_0 (q;k;v fused
                                                    along the output axis),linear_1},ffn/{layer_norm,linear_0 (gate),linear_0_noact (up),linear_1 (down)}},
                                                    decoder/layer_norm, decoder/projection; rotary_interleave False (= HF layout, no permutation).
  python/ctranslate2/converters/utils.py:4-36       fuse_linear: concatenation along axis 0; a missing bias becomes zeros.

Supported: decoder-only TransformerDecoderSpec models of the Llama family (RMSNorm, gated SwiGLU FFN, rotary with the HF layout, optional GQA,
optional q/k/v biases = Qwen2 layout, RMS q/k norms = Qwen3 layout). Everything else (encoder-decoder, AWQ-quantised, interleaved rotary, other spec names) is REFUSED with the
reason; nothing is silently approximated.

   python convert_ct2_to_hf.py CT2_DIR OUT_DIR [--tokenizer-dir HF_DIR] [--outtype auto|f32|f16|bf16] [--max-position-embeddings N]
Then:  python convert_hf_to_gguf.py OUT_DIR      (or give convert_hf_to_gguf.py the CT2 directory + --ct2-tokenizer-dir directly)
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

DTYPES = {0: "float32", 1: "int8", 2: "int16", 3: "int32", 4: "float16", 5: "bfloat16"}
ITEM_SIZE = {"float32": 4, "int8": 1, "int16": 2, "int32": 4, "float16": 2, "bfloat16": 2}
GLOBAL_INT16_SCALE = 1000.0   # src/ops/quantize.cc:8
MAX_BINARY_VERSION = 6        # include/ctranslate2/models/model.h:20
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "generation_config.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja")


class CT2Error(Exception):
    pass


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise CT2Error(f"model.bin is incomplete: wanted {n} bytes at offset {self.pos}, file has {len(self.data)}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def u(self, fmt: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.u("H")
        if n == 0:
            raise CT2Error("model.bin: invalid string length 0")
        return self.take(n)[:-1].decode("utf-8")


def read_model_bin(path: Path):
    """-> (spec name, revision, {name: (dtype name, shape, raw bytes)}, {alias: target})."""
    r = _Reader(Path(path).read_bytes())
    version = r.u("I")
    if not 1 <= version <= MAX_BINARY_VERSION:
        raise CT2Error(f"model.bin: binary version {version} is not supported (1..{MAX_BINARY_VERSION})")
    spec, revision = ("", 1)
    if version >= 2:
        spec, revision = r.string(), r.u("I")
    variables = {}
    for _ in range(r.u("I")):
        name = r.string()
        shape = tuple(r.u("I") for _ in range(r.u("B")))
        if version >= 4:
            tid = r.u("B")
            if tid not in DTYPES:
                raise CT2Error(f"model.bin: variable {name!r} has unknown dtype id {tid}")
            dtype, nbytes = DTYPES[tid], r.u("I")
        else:
            item = r.u("B")
            dtype = {4: "float32", 2: "int16", 1: "int8"}.get(item)
            if dtype is None:
                raise CT2Error(f"model.bin: variable {name!r} has unknown item size {item}")
            nbytes = r.u("I") * item
        n = 1
        for d in shape:
            n *= d
        if nbytes != n * ITEM_SIZE[dtype]:
            raise CT2Error(f"model.bin: variable {name!r} has {nbytes} bytes, shape {shape} x {dtype} needs {n * ITEM_SIZE[dtype]}")
        variables[name] = (dtype, shape, r.take(nbytes))
    aliases = {}
    if version >= 3:
        for _ in range(r.u("I")):
            alias = r.string()
            aliases[alias] = r.string()
    return spec, revision, variables, aliases


def _array(entry) -> np.ndarray:
    dtype, shape, raw = entry
    if dtype == "bfloat16":
        # bfloat16 = the top 16 bits of a float32
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype={"float32": "<f4", "int8": "i1", "int16": "<i2", "int32": "<i4", "float16": "<f2"}[dtype]).reshape(shape)


class CT2Model:
    def __init__(self, path: Path):
        path = Path(path)
        self.path = path
        self.spec, self.revision, self.vars, self.aliases = read_model_bin(path / "model.bin")
        self.config = json.loads((path / "config.json").read_text()) if (path / "config.json").is_file() else {}
        voc = path / "vocabulary.json"
        self.vocab = json.loads(voc.read_text(encoding="utf-8")) if voc.is_file() else None

    def has(self, name: str) -> bool:
        return name in self.vars or name in self.aliases

    def _entry(self, name: str):
        if name in self.aliases:
            name = self.aliases[name]
        if name not in self.vars:
            raise CT2Error(f"model.bin has no variable {name!r}")
        return self.vars[name]

    def scalar(self, name: str, default=None):
        if not self.has(name):
            if default is None:
                raise CT2Error(f"model.bin has no variable {name!r}")
            return default
        return _array(self._entry(name)).reshape(-1)[0].item()

    def tensor(self, name: str) -> np.ndarray:
        """The variable as float32, dequantised exactly as the C++ engine does (dequantize_cpu.cc:11-50)."""
        entry = self._entry(name)
        dtype = entry[0]
        arr = _array(entry)
        if dtype in ("float32", "float16", "bfloat16"):
            return arr.astype(np.float32)
        if dtype not in ("int8", "int16"):
            raise CT2Error(f"variable {name!r} has type {dtype}, which is not a weight type")
        sname = name + "_scale"
        if not self.has(sname) and not self.has(self.aliases.get(name, name) + "_scale"):
            if dtype != "int16":
                raise CT2Error(f"variable {sname} not found")   # model.cc:310-323
            scale = np.float32(GLOBAL_INT16_SCALE)
        else:
            sname = sname if self.has(sname) else self.aliases.get(name, name) + "_scale"
            scale = _array(self._entry(sname)).astype(np.float32)
        recip = np.float32(1.0) / scale
        if dtype == "int16":
            if np.ndim(recip) != 0 and recip.size != 1:
                raise CT2Error("int16 quantisation scale must be a scalar")
            return arr.astype(np.float32) * np.float32(np.asarray(recip).reshape(-1)[0])
        rows = int(np.prod(arr.shape[:-1])) if arr.ndim > 1 else 1
        recip = np.asarray(recip, dtype=np.float32).reshape(-1)
        if recip.size != rows:
            raise CT2Error(f"int8 variable {name!r}: {recip.size} scales for {rows} rows")
        return (arr.astype(np.float32).reshape(rows, -1) * recip[:, None]).reshape(arr.shape)

    def source_float_dtype(self) -> str:
        """The float type the model can be written in WITHOUT loss (decides --outtype auto): a dequantised int8/int16 weight is q/scale in
        float32 and would be rounded by a 16-bit type, so any quantised weight means float32; otherwise the stored float type."""
        if any(n + "_scale" in self.vars for n in self.vars):
            return "float32"
        for n, (d, _, _) in self.vars.items():
            if n.endswith("layer_norm/gamma"):
                return d
        return "float32"


def _save_safetensors(path: Path, tensors: dict, outtype: str) -> None:
    """Minimal safetensors writer (header = u64 length + JSON, then raw data), so no safetensors/torch install is needed."""
    np_dtype, st_dtype = {"f32": (np.dtype("<f4"), "F32"), "f16": (np.dtype("<f2"), "F16"), "bf16": (None, "BF16")}[outtype]
    header, blobs, offset = {}, [], 0
    for name in sorted(tensors):
        t = np.ascontiguousarray(tensors[name].astype(np.float32))
        if outtype == "bf16":   # round to nearest even on the top 16 bits
            u = t.view(np.uint32).astype(np.uint64)
            u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
            raw = u.tobytes()
        else:
            raw = t.astype(np_dtype).tobytes()
        header[name] = {"dtype": st_dtype, "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def convert(ct2_dir: Path, out_dir: Path, tokenizer_dir: Path | None = None, outtype: str = "auto",
            max_position_embeddings: int = 4096) -> dict:
    m = CT2Model(ct2_dir)
    if m.spec != "TransformerDecoderSpec":
        raise CT2Error(f"spec {m.spec!r} is not supported: only decoder-only TransformerDecoderSpec (Llama family) is converted")
    if m.config.get("quantization_type") not in (None, 0):   # common_spec.Quantization: 0 = CT2, 1/2 = AWQ gemm/gemv
        raise CT2Error(f"quantization_type {m.config.get('quantization_type')!r} (AWQ) is not supported here; convert the original AWQ checkpoint "
                       "with convert_hf_to_gguf.py instead")
    n_layers = 0
    while m.has(f"decoder/layer_{n_layers}/self_attention/linear_0/weight"):
        n_layers += 1
    if n_layers == 0:
        raise CT2Error("no decoder/layer_0/self_attention/linear_0/weight: not a fused-QKV decoder")
    if m.has("decoder/layer_0/encoder_attention/linear_0/weight"):
        raise CT2Error("encoder attention present: encoder-decoder models are not supported")
    pre = "decoder/"
    for must, why in (("layer_norm/gamma", "final norm"),):
        if not m.has(pre + must):
            raise CT2Error(f"missing {pre + must} ({why})")
    if m.has(pre + "layer_norm/beta") or m.has(f"{pre}layer_0/self_attention/layer_norm/beta"):
        raise CT2Error("LayerNorm with beta: only RMSNorm (Llama family) is supported")
    if not m.has(f"{pre}layer_0/ffn/linear_0_noact/weight"):
        raise CT2Error("no gated FFN (linear_0_noact): only SwiGLU Llama-family FFNs are supported")
    if int(m.scalar(pre + "activation", 2)) != 2:
        raise CT2Error("activation is not SWISH: only SwiGLU Llama-family FFNs are supported")
    # Engine defaults that a real Llama export always overrides explicitly (SOURCE-READ src/layers/transformer.cc:385-398: a MISSING
    # scale_embeddings means "multiply the embeddings by sqrt(depth)"; :418,:505 pre_norm defaults to true; :475 alibi to false;
    # :514-516 sliding_window / final_logit_softcapping default to 0; :530 scale_outputs is optional; :189 layer_scalar defaults to 1).
    if not m.has(pre + "scale_embeddings") and not m.has(pre + "embeddings/multiply_by_sqrt_depth"):
        raise CT2Error("scale_embeddings is absent: the engine then multiplies the embeddings by sqrt(hidden size), which is not a Llama-family decoder")
    se = m.scalar(pre + "scale_embeddings") if m.has(pre + "scale_embeddings") else m.scalar(pre + "embeddings/multiply_by_sqrt_depth")
    se_dtype = m._entry(pre + "scale_embeddings" if m.has(pre + "scale_embeddings") else pre + "embeddings/multiply_by_sqrt_depth")[0]
    if (se_dtype == "int8" and se) or (se_dtype != "int8" and se != 1.0):
        raise CT2Error("scale_embeddings is set: not a plain Llama-family decoder")
    for flag, bad in ((pre + "pre_norm", lambda v: v == 0), (pre + "alibi", lambda v: v != 0), (pre + "sliding_window", lambda v: v != 0),
                      (pre + "final_logit_softcapping", lambda v: v != 0), (pre + "layer_0/layer_scalar", lambda v: v != 1)):
        if m.has(flag) and bad(m.scalar(flag)):
            raise CT2Error(f"{flag.split('/')[-1]} is set to a value that is not a plain Llama-family decoder")
    for absent in (pre + "scale_outputs", pre + "project_in/weight", pre + "layernorm_embedding/gamma", pre + "position_encodings/encodings"):
        if m.has(absent):
            raise CT2Error(f"{absent} present: not a plain Llama-family decoder")
    if m.has(f"{pre}layer_0/self_attention/sliding_window") and m.scalar(f"{pre}layer_0/self_attention/sliding_window") != 0:
        raise CT2Error("sliding-window attention is not supported")
    if m.has(f"{pre}layer_0/self_attention/layer_norm/layer_norm_use_residual") and m.scalar(f"{pre}layer_0/self_attention/layer_norm/layer_norm_use_residual"):
        raise CT2Error("layer_norm_use_residual is set: not a plain Llama-family decoder")
    if m.has(f"{pre}layer_0/shared_layer_norm/gamma") or m.has(f"{pre}layer_0/input_layer_norm/gamma"):
        raise CT2Error("parallel-residual / pre-post-norm layouts are not supported")
    if m.has(f"{pre}layer_0/self_attention/v_norm/gamma"):
        raise CT2Error("v_norm present: not supported")
    qk_norm = m.has(f"{pre}layer_0/self_attention/q_norm/gamma") or m.has(f"{pre}layer_0/self_attention/k_norm/gamma")
    if qk_norm and not (m.has(f"{pre}layer_0/self_attention/q_norm/gamma") and m.has(f"{pre}layer_0/self_attention/k_norm/gamma")):
        raise CT2Error("only one of q_norm / k_norm present")
    if qk_norm and (m.has(f"{pre}layer_0/self_attention/q_norm/beta") or m.has(f"{pre}layer_0/self_attention/k_norm/beta")):
        raise CT2Error("q_norm / k_norm with beta: only RMS qk-norm (Qwen3) is supported")

    sa = pre + "layer_0/self_attention/"
    n_heads = int(m.scalar(pre + "num_heads"))
    multi_query = bool(m.scalar(sa + "multi_query", 0)) if m.has(sa + "multi_query") else False   # attention_layer.cc:134
    n_kv = 1 if multi_query else int(m.scalar(sa + "num_heads_kv", n_heads))
    emb = m.tensor(pre + "embeddings/weight")
    vocab_size, hidden = emb.shape
    head_dim = int(m.scalar(sa + "head_dim", hidden // n_heads))
    if int(m.scalar(sa + "rotary_dim", -1)) < 0:
        raise CT2Error("rotary_dim is not set: this model does not use rotary embeddings")
    if int(m.scalar(sa + "rotary_dim")) not in (0, head_dim):
        raise CT2Error("partial rotary (rotary_dim != head dim) is not supported")
    if int(m.scalar(sa + "rotary_interleave", 1)) != 0:   # attention_layer.cc:75: absent means interleaved
        raise CT2Error("rotary_interleave is True or absent (GPT-J layout): the HF Llama layout needs an explicit interleave False")
    if m.has(sa + "queries_scale") and abs(m.scalar(sa + "queries_scale") * np.sqrt(head_dim) - 1.0) > 1e-5:   # attention_layer.cc:131
        raise CT2Error("queries_scale differs from 1/sqrt(head dim): not a plain Llama-family decoder")
    rope_theta = float(m.scalar(sa + "rotary_base", 10000.0))
    rope_scaling = None
    if m.has(sa + "rotary_scaling_type"):
        kind = int(m.scalar(sa + "rotary_scaling_type"))
        if kind == 0:
            rope_scaling = {"rope_type": "linear", "factor": float(m.scalar(sa + "rotary_scaling_factor", 1.0))}
        elif kind == 2:
            rope_scaling = {"rope_type": "llama3", "factor": float(m.scalar(sa + "rotary_scaling_factor", 1.0)),
                            "low_freq_factor": float(m.scalar(sa + "rotary_low_freq_factor", 1.0)),   # attention_layer.cc:92-95 defaults
                            "high_freq_factor": float(m.scalar(sa + "rotary_high_freq_factor", 4.0)),
                            "original_max_position_embeddings": int(m.scalar(sa + "original_max_position_embeddings", 0))}
            if rope_scaling["original_max_position_embeddings"] <= 0:
                raise CT2Error("llama3 rope scaling without original_max_position_embeddings")
        else:
            raise CT2Error(f"rotary scaling type {kind} (Su) is not supported")

    hf: dict[str, np.ndarray] = {"model.embed_tokens.weight": emb, "model.norm.weight": m.tensor(pre + "layer_norm/gamma")}
    tied = (pre + "projection/weight" in m.aliases and m.aliases[pre + "projection/weight"] == pre + "embeddings/weight") or \
        (m.has(pre + "projection/weight") and m._entry(pre + "projection/weight") is m._entry(pre + "embeddings/weight"))
    if not tied:
        hf["lm_head.weight"] = m.tensor(pre + "projection/weight")
    q_rows, kv_rows = n_heads * head_dim, n_kv * head_dim
    has_qkv_bias = has_o_bias = False
    inter = None
    for i in range(n_layers):
        L = f"{pre}layer_{i}/"
        p = f"model.layers.{i}."
        qkv = m.tensor(L + "self_attention/linear_0/weight")
        if qkv.shape != (q_rows + 2 * kv_rows, hidden):
            raise CT2Error(f"layer {i}: fused QKV is {qkv.shape}, expected {(q_rows + 2 * kv_rows, hidden)} "
                           f"(heads {n_heads}, kv heads {n_kv}, head dim {head_dim})")
        hf[p + "self_attn.q_proj.weight"] = qkv[:q_rows]
        hf[p + "self_attn.k_proj.weight"] = qkv[q_rows:q_rows + kv_rows]
        hf[p + "self_attn.v_proj.weight"] = qkv[q_rows + kv_rows:]
        if m.has(L + "self_attention/linear_0/bias"):
            b = m.tensor(L + "self_attention/linear_0/bias")
            hf[p + "self_attn.q_proj.bias"], hf[p + "self_attn.k_proj.bias"], hf[p + "self_attn.v_proj.bias"] = \
                b[:q_rows], b[q_rows:q_rows + kv_rows], b[q_rows + kv_rows:]
            has_qkv_bias = True
        if qk_norm:   # Qwen3: RMSNorm over the head dimension of q and of k (transformers.py:2776-2781)
            hf[p + "self_attn.q_norm.weight"] = m.tensor(L + "self_attention/q_norm/gamma")
            hf[p + "self_attn.k_norm.weight"] = m.tensor(L + "self_attention/k_norm/gamma")
        hf[p + "self_attn.o_proj.weight"] = m.tensor(L + "self_attention/linear_1/weight")
        if m.has(L + "self_attention/linear_1/bias"):
            hf[p + "self_attn.o_proj.bias"] = m.tensor(L + "self_attention/linear_1/bias")
            has_o_bias = True
        hf[p + "input_layernorm.weight"] = m.tensor(L + "self_attention/layer_norm/gamma")
        hf[p + "post_attention_layernorm.weight"] = m.tensor(L + "ffn/layer_norm/gamma")
        hf[p + "mlp.gate_proj.weight"] = m.tensor(L + "ffn/linear_0/weight")
        hf[p + "mlp.up_proj.weight"] = m.tensor(L + "ffn/linear_0_noact/weight")
        hf[p + "mlp.down_proj.weight"] = m.tensor(L + "ffn/linear_1/weight")
        for ffn_bias in ("linear_0", "linear_0_noact", "linear_1"):
            if m.has(L + f"ffn/{ffn_bias}/bias"):
                raise CT2Error("FFN biases are not supported")
        inter = hf[p + "mlp.gate_proj.weight"].shape[0]
    if has_o_bias and not has_qkv_bias:
        raise CT2Error("output-projection bias without q/k/v bias is not a known layout")
    if qk_norm and (has_qkv_bias or has_o_bias):
        raise CT2Error("qk-norm together with attention biases is not a known layout")
    arch = "Qwen3ForCausalLM" if qk_norm else ("Qwen2ForCausalLM" if (has_qkv_bias and not has_o_bias) else "LlamaForCausalLM")

    bos = m.config.get("bos_token")
    eos = m.config.get("eos_token")
    ids = {t: i for i, t in enumerate(m.vocab)} if m.vocab else {}
    config = {
        "architectures": [arch], "model_type": {"Qwen3ForCausalLM": "qwen3", "Qwen2ForCausalLM": "qwen2"}.get(arch, "llama"),
        "hidden_size": int(hidden), "intermediate_size": int(inter), "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads, "num_key_value_heads": n_kv, "head_dim": head_dim, "vocab_size": int(vocab_size),
        "rms_norm_eps": float(m.config["layer_norm_epsilon"]) if m.config.get("layer_norm_epsilon") is not None else 1e-6, "rope_theta": rope_theta,
        "max_position_embeddings": int(max_position_embeddings), "hidden_act": "silu", "tie_word_embeddings": bool(tied),
        "bos_token_id": ids.get(bos, 1) if bos else 1, "eos_token_id": ids.get(eos, 2) if eos else 2,
        "torch_dtype": "float32",
    }
    if arch == "Qwen3ForCausalLM":
        config["attention_bias"] = False
    if arch == "LlamaForCausalLM":
        config["attention_bias"] = bool(has_o_bias)
        config["mlp_bias"] = False
    if rope_scaling:
        config["rope_scaling"] = rope_scaling

    if outtype == "auto":
        src = m.source_float_dtype()
        outtype = {"float16": "f16", "bfloat16": "bf16"}.get(src, "f32")
    config["torch_dtype"] = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}[outtype]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    _save_safetensors(out_dir / "model.safetensors", hf, outtype)
    if tokenizer_dir:
        for f in TOKENIZER_FILES:
            if (Path(tokenizer_dir) / f).is_file():
                shutil.copy(Path(tokenizer_dir) / f, out_dir / f)
    return {"arch": arch, "layers": n_layers, "outtype": outtype, "tied": bool(tied), "tensors": len(hf),
            "source_dtypes": sorted({d for d, _, _ in m.vars.values()}), "vocab_in_ct2": len(m.vocab) if m.vocab else None}


def is_ct2_dir(path) -> bool:
    p = Path(path)
    return (p / "model.bin").is_file() and not list(p.glob("*.safetensors")) and not list(p.glob("pytorch_model*.bin"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ct2_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tokenizer-dir", type=Path, help="HF directory with the tokenizer files (CT2 keeps only a token list, no merges / scores)")
    ap.add_argument("--outtype", choices=["auto", "f32", "f16", "bf16"], default="auto")
    ap.add_argument("--max-position-embeddings", type=int, default=4096, help="CT2 does not store it")
    a = ap.parse_args(argv)
    try:
        info = convert(a.ct2_dir, a.out_dir, a.tokenizer_dir, a.outtype, a.max_position_embeddings)
    except CT2Error as e:
        print(f"convert_ct2_to_hf: {e}", file=sys.stderr)
        return 2
    print("convert_ct2_to_hf: " + json.dumps(info))
    if not a.tokenizer_dir:
        print("convert_ct2_to_hf: NOTE no --tokenizer-dir: convert_hf_to_gguf.py will need the tokenizer files in the output directory", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
