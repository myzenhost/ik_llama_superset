#!/usr/bin/env python3
"""convert_ct2_to_hf.py: a CTranslate2 Llama-family model directory (model.bin) -> HF directory, for every CT2 weight format.

The test does NOT use the converter's reader to make its input. An independent WRITER below serialises model.bin from the layout
python/ctranslate2/specs/model_spec.py:364-412 (OpenNMT/CTranslate2 @ f2f54b42, MIT, behaviour only) and quantises with the rules of
model_spec.py:191-258, in plain loops / float64:
  int8  : per-ROW scale 127/amax (a zero row uses amax 127), q = rint(w*scale)      -> stored `<name>_scale`, shape [rows]
  int16 : ONE scale 2^10/amax per tensor, q = clip(rint(w*scale))                   -> stored `<name>_scale`, scalar
  float16 / bfloat16 / float32 : plain casts; norms and biases follow the "other" float type (int8_float16 -> float16 ...).
The expected tensors are computed here from the ORIGINAL float weights:
  * the converter's tensor must equal q / scale of the writer (to float32 rounding), AND
  * differ from the original weight by at most half a quantisation step per element, which a wrong row/scale pairing, a transposed scale
    vector, an un-split fused QKV or a wrong scale kind cannot satisfy (controls (a)-(d) below).

   python tests/test_ct2_to_hf.py <tree-root>

BEFORE (tree without convert_ct2_to_hf.py): the script is missing -> every case fails.   AFTER: all pass.
Controls: (a) int8 rows have DIFFERENT scales (a per-tensor reading fails the half-step bound), (b) the fused QKV rows differ per part (a
swapped q/k/v fails the exact split check), (c) the GQA model has fewer kv heads than q heads (a head-count mix-up fails the shape check),
(d) the lossy formats are really lossy (otherwise "equal to the original" would be vacuous), (e) unsupported specs are REFUSED, not guessed.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np  # noqa: E402

failures = []
RNG = np.random.default_rng(1234)


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" -- " + detail) if (detail and not cond) else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- the independent WRITER
TYPE_ID = {"float32": 0, "int8": 1, "int16": 2, "int32": 3, "float16": 4, "bfloat16": 5}


def s_(string):
    b = string.encode()
    return struct.pack("<H", len(b) + 1) + b + b"\0"


def to_bf16_bytes(a):
    u = np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")).tobytes()


def bf16_value(a):
    u = np.frombuffer(to_bf16_bytes(a), dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32).reshape(np.shape(a))


def var_bytes(name, dtype, shape, raw):
    out = s_(name) + struct.pack("<B", len(shape)) + b"".join(struct.pack("<I", d) for d in shape)
    return out + struct.pack("<B", TYPE_ID[dtype]) + struct.pack("<I", len(raw)) + raw


def float_var(a, dtype):
    a = np.asarray(a, dtype=np.float32)
    if dtype == "float32":
        return "float32", a.astype("<f4").tobytes(), a
    if dtype == "float16":
        h = a.astype("<f2")
        return "float16", h.tobytes(), h.astype(np.float32)
    return "bfloat16", to_bf16_bytes(a), bf16_value(a)


class Ct2Writer:
    """Collects variables of one model and writes model.bin + config.json + vocabulary.json."""

    def __init__(self, scheme):
        self.scheme = scheme                         # float32 float16 bfloat16 int8 int8_float16 int8_bfloat16 int16
        self.vars = []                               # (name, dtype, shape, raw)
        self.expect_q = {}                           # name -> dequantised value the engine would produce (float64)
        self.step = {}                               # name -> max |w - deq| allowed (half a quantisation step)
        self.aliases = []
        self.other = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16", "int8": "float32", "int16": "float32",
                      "int8_float16": "float16", "int8_bfloat16": "bfloat16"}[scheme]

    def add_scalar(self, name, dtype, value):
        np_dt = {"int8": "i1", "int16": "<i2", "int32": "<i4", "float32": "<f4"}[dtype]
        self.vars.append((name, dtype, (), np.array(value, dtype=np_dt).tobytes()))

    def add_other(self, name, a):
        dt, raw, val = float_var(a, self.other)
        self.vars.append((name, dt, tuple(np.shape(a)), raw))
        self.expect_q[name] = val.astype(np.float64)

    def add_weight(self, name, w):
        w = np.asarray(w, dtype=np.float32)
        if self.scheme == "int16":
            scale = np.float32(2 ** 10 / np.max(np.abs(w)))
            q = np.clip(np.rint(w * scale), -32768, 32767).astype("<i2")
            self.vars.append((name, "int16", w.shape, q.tobytes()))
            self.vars.append((name + "_scale", "float32", (), np.array(scale, dtype="<f4").tobytes()))
            self.expect_q[name] = q.astype(np.float64) / float(scale)
            self.step[name] = 0.5 / float(scale) * 1.0001 + 1e-7
        elif self.scheme.startswith("int8"):
            amax = np.max(np.abs(w), axis=1)
            amax[amax == 0] = 127.0
            scale = (127.0 / amax).astype(np.float32)
            q = np.rint(w * scale[:, None]).astype("i1")
            self.vars.append((name, "int8", w.shape, q.tobytes()))
            self.vars.append((name + "_scale", "float32", (w.shape[0],), scale.astype("<f4").tobytes()))
            self.expect_q[name] = q.astype(np.float64) / scale.astype(np.float64)[:, None]
            self.step[name] = (0.5 / scale.astype(np.float64))[:, None] * 1.0001 + 1e-7
        else:
            dt, raw, val = float_var(w, self.scheme)
            self.vars.append((name, dt, w.shape, raw))
            self.expect_q[name] = val.astype(np.float64)
            self.step[name] = None

    def write(self, path, spec="TransformerDecoderSpec", config=None, vocab=None, version=6):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        body = struct.pack("<I", version) + s_(spec) + struct.pack("<I", 8) + struct.pack("<I", len(self.vars))
        for name, dtype, shape, raw in self.vars:
            body += var_bytes(name, dtype, shape, raw)
        body += struct.pack("<I", len(self.aliases))
        for a, t in self.aliases:
            body += s_(a) + s_(t)
        (path / "model.bin").write_bytes(body)
        (path / "config.json").write_text(json.dumps(config or {}))
        if vocab is not None:
            (path / "vocabulary.json").write_text(json.dumps(vocab))


# ---------------------------------------------------------------- a tiny Llama-family model, from random weights
def make_model(scheme, n_heads=4, n_kv=2, hidden=16, ffn=24, vocab=32, layers=2, tied=False, bias=False, zero_row=False,
               rotary_interleave=0, activation=2, extra=None, spec="TransformerDecoderSpec", config_extra=None, w_std=1.0, scale_emb=0, rotary_interleave_var=True):
    hd = hidden // n_heads
    wr = Ct2Writer(scheme)
    orig = {}

    def rnd(*shape, scale=1.0):
        # rows with very different magnitudes, so a per-tensor scale reading cannot pass
        a = (RNG.standard_normal(shape) * w_std).astype(np.float32)
        if len(shape) == 2:
            a *= (RNG.random((shape[0], 1)).astype(np.float32) * 3 + 0.05) * scale
        return a

    emb = rnd(vocab, hidden)
    if zero_row:
        emb[3] = 0.0
    orig["embed"] = emb
    wr.add_scalar("decoder/num_heads", "int16", n_heads)
    wr.add_scalar("decoder/activation", "int8", activation)
    if scale_emb is not None:                      # LlamaLoader.set_decoder sets spec.scale_embeddings = False -> int8 0 in model.bin
        wr.add_scalar("decoder/scale_embeddings", "int8", scale_emb)
    wr.add_scalar("decoder/pre_norm", "int8", 1)
    wr.add_weight("decoder/embeddings/weight", emb)
    g = (RNG.random(hidden).astype(np.float32) + 0.5)
    orig["norm"] = g
    wr.add_other("decoder/layer_norm/gamma", g)
    if tied:
        wr.aliases.append(("decoder/projection/weight", "decoder/embeddings/weight"))
    else:
        proj = rnd(vocab, hidden)
        orig["lm_head"] = proj
        wr.add_weight("decoder/projection/weight", proj)
    for i in range(layers):
        L = f"decoder/layer_{i}/"
        q, k, v = rnd(n_heads * hd, hidden), rnd(n_kv * hd, hidden), rnd(n_kv * hd, hidden)
        orig[f"q{i}"], orig[f"k{i}"], orig[f"v{i}"] = q, k, v
        wr.add_weight(L + "self_attention/linear_0/weight", np.concatenate([q, k, v]))
        if bias:
            b = RNG.standard_normal(q.shape[0] + 2 * k.shape[0]).astype(np.float32)
            orig[f"qkvb{i}"] = b
            wr.add_other(L + "self_attention/linear_0/bias", b)
        o = rnd(hidden, n_heads * hd)
        orig[f"o{i}"] = o
        wr.add_weight(L + "self_attention/linear_1/weight", o)
        for key, nm in (("in", "self_attention/layer_norm/gamma"), ("post", "ffn/layer_norm/gamma")):
            gg = RNG.random(hidden).astype(np.float32) + 0.5
            orig[f"{key}{i}"] = gg
            wr.add_other(L + nm, gg)
        gate, up, down = rnd(ffn, hidden), rnd(ffn, hidden), rnd(hidden, ffn)
        orig[f"gate{i}"], orig[f"up{i}"], orig[f"down{i}"] = gate, up, down
        wr.add_weight(L + "ffn/linear_0/weight", gate)
        wr.add_weight(L + "ffn/linear_0_noact/weight", up)
        wr.add_weight(L + "ffn/linear_1/weight", down)
        wr.add_scalar(L + "self_attention/rotary_dim", "int32", 0)
        if rotary_interleave_var:
            wr.add_scalar(L + "self_attention/rotary_interleave", "int8", rotary_interleave)
        wr.add_scalar(L + "self_attention/rotary_base", "float32", 500000.0)
        if n_kv != n_heads:
            wr.add_scalar(L + "self_attention/num_heads_kv", "int32", n_kv)
    for name, dtype, shape, raw in (extra or []):
        wr.vars.append((name, dtype, shape, raw))
    cfg = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "layer_norm_epsilon": 1e-5, "multi_query_attention": n_kv != n_heads}
    cfg.update(config_extra or {})
    vocab_list = ["<unk>", "<s>", "</s>"] + [f"t{i}" for i in range(vocab - 3)]
    return wr, orig, cfg, vocab_list, dict(n_heads=n_heads, n_kv=n_kv, hidden=hidden, ffn=ffn, vocab=vocab, layers=layers, hd=hd)


def read_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    out = {}
    for name, h in header.items():
        a, b = h["data_offsets"]
        data = raw[8 + n + a:8 + n + b]
        if h["dtype"] == "F32":
            arr = np.frombuffer(data, dtype="<f4")
        elif h["dtype"] == "F16":
            arr = np.frombuffer(data, dtype="<f2").astype(np.float32)
        elif h["dtype"] == "BF16":
            arr = (np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            raise AssertionError(h["dtype"])
        out[name] = arr.reshape(h["shape"]).astype(np.float64)
    return out


def run_converter(src, dst, *extra):
    p = subprocess.run([sys.executable, str(root / "convert_ct2_to_hf.py"), str(src), str(dst), *extra],
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def case(scheme, tmp, **kw):
    tag = scheme + "-" + "-".join(f"{k}{v}" for k, v in sorted(kw.items()) if not isinstance(v, (list, dict)))
    wr, orig, cfg, vocab_list, dims = make_model(scheme, **kw)
    src, dst = Path(tmp) / (tag + "-ct2"), Path(tmp) / (tag + "-hf")
    wr.write(src, config=cfg, vocab=vocab_list)
    rc, out, err = run_converter(src, dst, "--outtype", "f32")
    check(f"[{tag}] converter exits 0", rc == 0, err.strip()[-300:])
    if rc != 0:
        return None
    t = read_safetensors(dst / "model.safetensors")
    hf = json.loads((dst / "config.json").read_text())

    def vs_writer(hf_name, ct2_name, rows=None, label=None):
        """equal to the writer's dequantised value, and within half a quantisation step of the ORIGINAL float weight."""
        exp = wr.expect_q[ct2_name]
        if rows is not None:
            exp = exp[rows]
        got = t[hf_name]
        ok_shape = got.shape == exp.shape
        check(f"[{tag}] {label or hf_name}: shape", ok_shape, f"{got.shape} vs {exp.shape}")
        if not ok_shape:
            return
        tol = 1e-6 * np.maximum(1.0, np.abs(exp))
        check(f"[{tag}] {label or hf_name}: equals the engine's dequantisation", bool(np.all(np.abs(got - exp) <= tol)),
              f"max err {np.max(np.abs(got - exp)):.3g}")

    hd, nh, nk = dims["hd"], dims["n_heads"], dims["n_kv"]
    vs_writer("model.embed_tokens.weight", "decoder/embeddings/weight")
    vs_writer("model.norm.weight", "decoder/layer_norm/gamma")
    if kw.get("tied"):
        check(f"[{tag}] tied projection: no lm_head tensor, tie_word_embeddings", "lm_head.weight" not in t and hf["tie_word_embeddings"] is True)
    else:
        vs_writer("lm_head.weight", "decoder/projection/weight")
    for i in range(dims["layers"]):
        L, p = f"decoder/layer_{i}/", f"model.layers.{i}."
        qkv = L + "self_attention/linear_0/weight"
        vs_writer(p + "self_attn.q_proj.weight", qkv, slice(0, nh * hd), f"layer{i} q")
        vs_writer(p + "self_attn.k_proj.weight", qkv, slice(nh * hd, nh * hd + nk * hd), f"layer{i} k")
        vs_writer(p + "self_attn.v_proj.weight", qkv, slice(nh * hd + nk * hd, nh * hd + 2 * nk * hd), f"layer{i} v")
        vs_writer(p + "self_attn.o_proj.weight", L + "self_attention/linear_1/weight")
        vs_writer(p + "mlp.gate_proj.weight", L + "ffn/linear_0/weight")
        vs_writer(p + "mlp.up_proj.weight", L + "ffn/linear_0_noact/weight")
        vs_writer(p + "mlp.down_proj.weight", L + "ffn/linear_1/weight")
        vs_writer(p + "input_layernorm.weight", L + "self_attention/layer_norm/gamma")
        vs_writer(p + "post_attention_layernorm.weight", L + "ffn/layer_norm/gamma")
        # fidelity against the ORIGINAL weights
        for key, name in ((f"q{i}", p + "self_attn.q_proj.weight"), (f"k{i}", p + "self_attn.k_proj.weight"), (f"v{i}", p + "self_attn.v_proj.weight"),
                          (f"gate{i}", p + "mlp.gate_proj.weight"), (f"down{i}", p + "mlp.down_proj.weight")):
            ct2_name = {"q": L + "self_attention/linear_0/weight", "k": L + "self_attention/linear_0/weight", "v": L + "self_attention/linear_0/weight",
                        "g": L + "ffn/linear_0/weight", "d": L + "ffn/linear_1/weight"}[key[0]]
            step = wr.step[ct2_name]
            err = np.abs(t[name] - orig[key].astype(np.float64))
            if scheme in ("float32",):
                check(f"[{tag}] {name} equals the original weight exactly", bool(np.all(err == 0)), f"max {err.max():.3g}")
            elif step is not None and not isinstance(step, np.ndarray):
                check(f"[{tag}] {name} within half a step of the original", bool(np.all(err <= step)), f"max {err.max():.3g} > {step:.3g}")
            elif isinstance(step, np.ndarray):
                if key[0] == "q":
                    st = step[:nh * hd]
                elif key[0] == "k":
                    st = step[nh * hd:nh * hd + nk * hd]
                elif key[0] == "v":
                    st = step[nh * hd + nk * hd:]
                else:
                    st = step
                check(f"[{tag}] {name} within half a row-step of the original", bool(np.all(err <= st)), f"max {err.max():.3g}")
    if kw.get("bias"):
        for i in range(dims["layers"]):
            b = wr.expect_q[f"decoder/layer_{i}/self_attention/linear_0/bias"]
            p = f"model.layers.{i}.self_attn."
            check(f"[{tag}] layer{i} qkv bias split", bool(np.array_equal(np.concatenate([t[p + "q_proj.bias"], t[p + "k_proj.bias"], t[p + "v_proj.bias"]]), b)))
        check(f"[{tag}] bias model is Qwen2ForCausalLM", hf["architectures"] == ["Qwen2ForCausalLM"])
    else:
        check(f"[{tag}] architecture LlamaForCausalLM", hf["architectures"] == ["LlamaForCausalLM"])
    check(f"[{tag}] config: heads {nh}/{nk}, hidden, ffn, vocab, layers, rope_theta, eps, bos/eos ids",
          (hf["num_attention_heads"], hf["num_key_value_heads"], hf["hidden_size"], hf["intermediate_size"], hf["vocab_size"],
           hf["num_hidden_layers"], hf["rope_theta"], hf["rms_norm_eps"], hf["bos_token_id"], hf["eos_token_id"]) ==
          (nh, nk, dims["hidden"], dims["ffn"], dims["vocab"], dims["layers"], 500000.0, 1e-5, 1, 2), str(hf))
    return wr, t


def run_all():
  with tempfile.TemporaryDirectory() as tmp:
    # every CT2 weight format, GQA model
    for scheme in ("float32", "float16", "bfloat16", "int8", "int8_float16", "int8_bfloat16", "int16"):
        case(scheme, tmp)
    # structure variants
    case("int8", tmp, n_kv=4)                     # MHA (no num_heads_kv variable)
    case("float16", tmp, tied=True)               # projection aliased to the embeddings
    case("int8_float16", tmp, bias=True)          # q/k/v biases -> Qwen2 layout
    case("int8", tmp, zero_row=True)              # all-zero row: amax 0 -> 127 (model_spec.py:221-223)

    # control (a): the int8 rows really have different scales
    wr, _, _, _, _ = make_model("int8")
    scales = [np.frombuffer(raw, dtype="<f4") for (n, d, s, raw) in wr.vars if n == "decoder/layer_0/ffn/linear_0/weight_scale"][0]
    check("control (a): int8 row scales differ by > 2x", scales.max() / scales.min() > 2.0, str(scales))
    # control (d): the lossy formats are lossy
    wr16, orig16, _, _, _ = make_model("int8")
    deq = wr16.expect_q["decoder/layer_0/ffn/linear_0/weight"]
    check("control (d): int8 dequantisation differs from the original", bool(np.max(np.abs(deq - orig16["gate0"])) > 1e-4))

    # header details: auto outtype follows the source float type
    for scheme, want in (("float16", "F16"), ("bfloat16", "BF16"), ("int8_float16", "F32"), ("float32", "F32"), ("int8", "F32")):
        wr, orig, cfg, vocab_list, _ = make_model(scheme)
        src, dst = Path(tmp) / f"auto-{scheme}-ct2", Path(tmp) / f"auto-{scheme}-hf"
        wr.write(src, config=cfg, vocab=vocab_list)
        rc, _, err = run_converter(src, dst)
        raw = (dst / "model.safetensors").read_bytes()
        hdr = json.loads(raw[8:8 + struct.unpack("<Q", raw[:8])[0]])
        check(f"outtype auto for {scheme} -> {want}", rc == 0 and hdr["model.norm.weight"]["dtype"] == want, err[-200:])

    # no layer_norm_epsilon in config.json: the engine's RMSNorm default is 1e-6 (src/layers/common.cc:449-453)
    wr, orig, cfg, vocab_list, _ = make_model("float32")
    cfg.pop("layer_norm_epsilon")
    src, dst = Path(tmp) / "eps-ct2", Path(tmp) / "eps-hf"
    wr.write(src, config=cfg, vocab=vocab_list)
    rc, _, err = run_converter(src, dst)
    check("default RMSNorm epsilon is 1e-6 when config.json has none", rc == 0 and json.loads((dst / "config.json").read_text())["rms_norm_eps"] == 1e-6, err[-200:])

    # tokenizer files are copied through
    wr, orig, cfg, vocab_list, _ = make_model("float32")
    src, dst, tok = Path(tmp) / "tk-ct2", Path(tmp) / "tk-hf", Path(tmp) / "tk-tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")
    (tok / "tokenizer_config.json").write_text("{}")
    (tok / "unrelated.txt").write_text("x")
    wr.write(src, config=cfg, vocab=vocab_list)
    rc, _, err = run_converter(src, dst, "--tokenizer-dir", str(tok))
    check("tokenizer files copied (and only tokenizer files)", rc == 0 and (dst / "tokenizer.json").is_file() and (dst / "tokenizer_config.json").is_file()
          and not (dst / "unrelated.txt").exists(), err[-200:])

    # (e) refusals
    def refused(label, writer_args, fragment, spec="TransformerDecoderSpec", cfg_extra=None, extra=None):
        wr, orig, cfg, vocab_list, _ = make_model("float32", extra=extra, **writer_args)
        cfg.update(cfg_extra or {})
        src, dst = Path(tmp) / f"ref-{label}-ct2", Path(tmp) / f"ref-{label}-hf"
        wr.write(src, spec=spec, config=cfg, vocab=vocab_list)
        rc, _, err = run_converter(src, dst)
        check(f"refused: {label}", rc == 2 and fragment in err and not (dst / "model.safetensors").exists(), f"rc={rc} {err.strip()[-200:]}")

    refused("encoder-decoder spec name", {}, "TransformerSpec", spec="TransformerSpec")
    refused("interleaved rotary", {"rotary_interleave": 1}, "rotary_interleave")
    refused("AWQ config", {}, "AWQ", cfg_extra={"quantization_type": 1})
    # defaults of the C++ engine for ABSENT variables (src/layers/transformer.cc:385-398, attention_layer.cc:75): not what a Llama export means
    refused("scale_embeddings absent (engine scales by sqrt(depth))", {"scale_emb": None}, "scale_embeddings is absent")
    refused("scale_embeddings set", {"scale_emb": 1}, "scale_embeddings is set")
    refused("rotary_interleave absent (engine default: interleaved)", {"rotary_interleave_var": False}, "rotary_interleave")
    refused("sliding window", {}, "sliding-window", extra=[("decoder/layer_0/self_attention/sliding_window", "int32", (), np.array(8, "<i4").tobytes())])
    refused("non-default queries_scale", {}, "queries_scale", extra=[("decoder/layer_0/self_attention/queries_scale", "float32", (), np.array(0.3, "<f4").tobytes())])
    refused("non-SwiGLU activation", {"activation": 0}, "SWISH")
    refused("layer norm beta", {}, "RMSNorm", extra=[("decoder/layer_norm/beta", "float32", (16,), np.zeros(16, "<f4").tobytes())])
    # truncated model.bin is an error, not a crash with a traceback
    wr, orig, cfg, vocab_list, _ = make_model("float32")
    src, dst = Path(tmp) / "trunc-ct2", Path(tmp) / "trunc-hf"
    wr.write(src, config=cfg, vocab=vocab_list)
    data = (src / "model.bin").read_bytes()
    (src / "model.bin").write_bytes(data[:len(data) // 2])
    rc, _, err = run_converter(src, dst)
    check("refused: truncated model.bin", rc == 2 and "incomplete" in err, f"rc={rc} {err.strip()[-200:]}")



if __name__ == "__main__":
    run_all()
    print()
    print("FAILED: " + ", ".join(failures) if failures else "all checks passed")
    sys.exit(1 if failures else 0)
