#!/usr/bin/env python3
"""HF integer per-row checkpoints -- transformers quant_method "gemma" (int2/int4/int8 + weight_scale, quantized embeddings)
and bitsandbytes LLM.int8 (int8 weight + SCB) -- through the REAL ModelBase.dequant_model and through convert_hf_to_gguf.py.

   python tests/test_int_perrow_dequant.py <tree-root> [--base-tree <older tree for the unquantized byte-identity check>]

The reference quantiser / dequantiser / bit packer below are plain Python loops written from the format description (they do
not import conversion.intperrow), so a bug shared by a vectorised reader and its own test cannot hide.
BEFORE (tree without conversion/intperrow.py): quant_method "gemma" -> NotImplementedError "Quant method is not yet supported",
bitsandbytes load_in_8bit -> NotImplementedError "other than 4-bit" -> every case FAILS. AFTER: all cases pass.
Error bounds (stated, checked): symmetric per-row scale s = absmax/hi where hi = 2^(b-1)-1 for int4/int8 and 1 for int2, rounding
to nearest, so |dequant - original| <= s/2 per element (int2: only when the element is not clipped, which holds by construction
because |w| <= absmax = hi*s and lo*s <= -hi*s). bnb int8: s = SCB/127, bound SCB/254.
Controls: (a) the data discriminates lane/nibble order (reference differs from the reversed-order reading); (b) the dequantised
weight is NOT the original (lossy) so returning the packed bytes cannot pass; (c) unknown quant_method still raises; (d) legacy
bnb weight_format is refused; (e) an unquantized tiny llama converts BYTE-IDENTICAL on this tree and (with --base-tree) on the
base tree, and two runs of the same tree are identical (so identity is a real signal, not noise).
"""
import gc
import hashlib
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

args = [a for a in sys.argv[1:]]
base_tree = None
if "--base-tree" in args:
    i = args.index("--base-tree")
    base_tree = Path(args[i + 1])
    del args[i:i + 2]
root = Path(args[0]) if args else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "gguf-py"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import conversion.base as base  # noqa: E402

failures = []
LEFTOVER = ("SCB", "weight_scale", "weight_format", "embedding_quantized", "embedding_scale")


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" -- " + detail) if (detail and not cond) else ""))
    if not cond:
        failures.append(name)


rng = np.random.default_rng(1108)


# ---------------------------------------------------------------- independent reference
def lohi(bits):
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def quant_rows(w, bits):
    """w [rows, cols] float -> (codes [rows][cols] ints, scale [rows]); s = absmax/hi (hi=1 for int2)."""
    lo, hi = lohi(bits)
    codes, scales = [], []
    for r in range(w.shape[0]):
        am = float(np.abs(w[r]).max())
        s = am / max(hi, 1) if am > 0 else 1.0
        scales.append(s)
        codes.append([min(max(int(round(float(v) / s)), lo), hi) for v in w[r]])
    return codes, scales


def pack(codes, bits):
    """Pack rows of signed codes into uint8 rows. int4: first value LOW nibble, stored c+8. int2: first value LOW lane, stored c+2."""
    out = []
    for row in codes:
        if bits == 8:
            out.append([c & 0xFF for c in row])
        elif bits == 4:
            row = row + [0] * (len(row) % 2)
            out.append([((row[i] + 8) & 15) | (((row[i + 1] + 8) & 15) << 4) for i in range(0, len(row), 2)])
        else:
            row = row + [0] * (-len(row) % 4)
            out.append([sum(((row[i + k] + 2) & 3) << (2 * k) for k in range(4)) for i in range(0, len(row), 4)])
    return out


def unpack_ref(packed_rows, bits, width, reverse=False):
    out = []
    for row in packed_rows:
        vals = []
        for b in row:
            if bits == 4:
                v = [(b & 15) - 8, (b >> 4) - 8]
            else:
                v = [((b >> (2 * k)) & 3) - 2 for k in range(4)]
            vals.extend(v[::-1] if reverse else v)
        out.append(vals[:width])
    return out


def to_t(packed, bits):
    a = np.array(packed, dtype=np.int64)
    if bits == 8:
        return torch.from_numpy(a.astype(np.uint8).view(np.int8))
    return torch.from_numpy(a.astype(np.uint8))


def make_linear(n_out, n_in, bits):
    w = (rng.standard_normal((n_out, n_in)) * 0.1).astype(np.float32)
    codes, scales = quant_rows(w, bits)
    packed = pack(codes, bits)
    ref = np.array([[c * scales[r] for c in codes[r]] for r in range(n_out)], dtype=np.float32)
    return w, to_t(packed, bits), torch.tensor(scales, dtype=torch.float32).reshape(n_out, 1), ref, np.array(scales), packed


def run_dequant_model(tensors, quant_config):
    stub = types.SimpleNamespace(
        _is_nvfp4=False, hparams={"quantization_config": quant_config}, lazy=False, _fp8_as_q8=False,
        _fp8_dequantized=set(), model_tensors={k: (lambda v=v: v) for k, v in tensors.items()},
    )
    base.ModelBase.dequant_model(stub)
    return stub


# ---------------------------------------------------------------- gemma linear, per bit width
for bits, (n_out, n_in) in ((2, (12, 20)), (4, (16, 24)), (8, (9, 17)), (4, (5, 8))):
    w, q, sc, ref, scales, packed = make_linear(n_out, n_in, bits)
    p = "model.layers.0.mlp.down_proj"
    tensors = {p + ".weight": q, p + ".weight_scale": sc,
               p + ".input_activation_scale": torch.tensor(0.5), p + ".output_activation_scale": torch.tensor(0.25),
               "model.layers.0.self_attn.k_cache_scale": torch.tensor(1.0), "model.layers.0.self_attn.v_cache_scale": torch.tensor(1.0),
               "model.norm.weight": torch.ones(n_out)}
    cfg = {"quant_method": "gemma", "num_bits": bits}
    name = "gemma int%d linear %dx%d" % (bits, n_out, n_in)
    try:
        stub = run_dequant_model(dict(tensors), cfg)
        got = stub.model_tensors[p + ".weight"]()
    except Exception as e:
        check(name + ": dequant_model accepts the checkpoint", False, "%s: %s" % (type(e).__name__, e))
        continue
    check(name + ": shape", tuple(got.shape) == (n_out, n_in), str(tuple(got.shape)))
    check(name + ": dtype float32", got.dtype == torch.float32, str(got.dtype))
    check(name + ": scale / SRQ / kv-cache tensors removed",
          set(stub.model_tensors) == {p + ".weight", "model.norm.weight"}, str(sorted(stub.model_tensors)))
    check(name + ": non-quantized tensor untouched", stub.model_tensors["model.norm.weight"]().equal(torch.ones(n_out)))
    err = float(np.abs(got.numpy() - ref).max())
    check(name + ": equals independent reference dequantiser", err <= 1e-6, "max|diff|=%g" % err)
    dev = np.abs(got.numpy() - w).max(axis=1)
    check(name + ": within s/2 of the ORIGINAL weight (quantisation bound)", bool(np.all(dev <= scales / 2 * 1.001 + 1e-7)),
          "worst dev/(s/2)=%g" % float((dev / (scales / 2)).max()))
    check(name + ": CONTROL result is not the original (lossy)", float(np.abs(got.numpy() - w).max()) > 0)
    if bits in (2, 4):
        rev = np.array([[c * scales[r] for c in row] for r, row in enumerate(unpack_ref(packed, bits, n_in, reverse=True))], dtype=np.float32)
        check(name + ": CONTROL data discriminates lane/nibble order", float(np.abs(rev - ref).max()) > 1e-4)

# ---------------------------------------------------------------- module_quant_configs override (first regex match wins) + int8 dtype
mods = {"model.layers.0.mlp.down_proj": 2, "model.layers.0.self_attn.o_proj": 8, "model.layers.0.mlp.up_proj": 4}
tensors, refs = {}, {}
for p, b in mods.items():
    w, q, sc, ref, scales, _ = make_linear(8, 16, b)
    tensors[p + ".weight"], tensors[p + ".weight_scale"] = q, sc
    refs[p] = ref
cfg = {"quant_method": "gemma", "num_bits": 4, "module_quant_configs": {r"mlp\.down_proj": {"num_bits": 2}, r"self_attn\.o_proj": {"num_bits": 8}}}
try:
    stub = run_dequant_model(dict(tensors), cfg)
    worst = max(float(np.abs(stub.model_tensors[p + ".weight"]().numpy() - refs[p]).max()) for p in mods)
    check("gemma per-module num_bits (int2 override + int8 + default int4 in one checkpoint)", worst <= 1e-6, "worst=%g" % worst)
except Exception as e:
    check("gemma per-module num_bits", False, "%s: %s" % (type(e).__name__, e))

# ---------------------------------------------------------------- gemma quantized embedding (block scales)
for bits in (8, 4, 2):
    V, D, NB = 10, 32, 4
    w = (rng.standard_normal((V, D)) * 0.1).astype(np.float32)
    blk = D // NB
    lo, hi = lohi(bits)
    codes, ref_rows, scales = [], [], np.zeros((V, NB), dtype=np.float32)
    for r in range(V):
        row_codes, row_ref = [], []
        for b in range(NB):
            seg = w[r, b * blk:(b + 1) * blk]
            am = float(np.abs(seg).max())
            s = am / max(hi, 1) if am > 0 else 1.0
            scales[r, b] = s
            for v in seg:
                c = min(max(int(round(float(v) / s)), lo), hi)
                row_codes.append(c)
                row_ref.append(c * s)
        codes.append(row_codes)
        ref_rows.append(row_ref)
    ref = np.array(ref_rows, dtype=np.float32)
    p = "model.embed_tokens"
    tensors = {p + ".embedding_quantized": to_t(pack(codes, bits), bits), p + ".embedding_scale": torch.from_numpy(scales)}
    name = "gemma int%d embedding %dx%d (%d scale blocks)" % (bits, V, D, NB)
    try:
        stub = run_dequant_model(dict(tensors), {"quant_method": "gemma", "num_bits": bits, "quantize_embeddings": True})
        got = stub.model_tensors[p + ".weight"]()
    except Exception as e:
        check(name + ": accepted", False, "%s: %s" % (type(e).__name__, e))
        continue
    check(name + ": renamed to .weight, packed + scale removed", set(stub.model_tensors) == {p + ".weight"}, str(sorted(stub.model_tensors)))
    err = float(np.abs(got.numpy() - ref).max())
    check(name + ": equals independent reference", got.shape == (V, D) and err <= 1e-6, "shape=%s max|diff|=%g" % (tuple(got.shape), err))
    s_full = np.repeat(scales, blk, axis=1)
    check(name + ": within s/2 of the ORIGINAL", bool(np.all(np.abs(got.numpy() - w) <= s_full / 2 * 1.001 + 1e-7)))

# ---------------------------------------------------------------- bitsandbytes LLM.int8
for fmt in (None, 0):
    n_out, n_in = 11, 19
    w = (rng.standard_normal((n_out, n_in)) * 0.2).astype(np.float32)
    scb = np.abs(w).max(axis=1).astype(np.float32)
    cb = np.array([[int(round(float(v) / float(scb[r]) * 127.0)) for v in w[r]] for r in range(n_out)], dtype=np.int8)
    ref = np.array([[float(cb[r, c]) * float(scb[r]) / 127.0 for c in range(n_in)] for r in range(n_out)], dtype=np.float32)
    p = "model.layers.1.self_attn.q_proj"
    tensors = {p + ".weight": torch.from_numpy(cb), p + ".SCB": torch.from_numpy(scb), "model.norm.weight": torch.ones(3)}
    if fmt is not None:
        tensors[p + ".weight_format"] = torch.tensor(fmt, dtype=torch.uint8)
    name = "bnb LLM.int8 %dx%d weight_format=%s" % (n_out, n_in, fmt)
    try:
        stub = run_dequant_model(dict(tensors), {"quant_method": "bitsandbytes", "load_in_8bit": True, "load_in_4bit": False})
        got = stub.model_tensors[p + ".weight"]()
    except Exception as e:
        check(name + ": accepted", False, "%s: %s" % (type(e).__name__, e))
        continue
    check(name + ": SCB / weight_format removed", set(stub.model_tensors) == {p + ".weight", "model.norm.weight"}, str(sorted(stub.model_tensors)))
    err = float(np.abs(got.numpy() - ref).max())
    check(name + ": equals independent reference (int8*SCB/127)", got.dtype == torch.float32 and err <= 1e-6, "max|diff|=%g" % err)
    dev = np.abs(got.numpy() - w).max(axis=1)
    check(name + ": within SCB/254 of the ORIGINAL", bool(np.all(dev <= scb / 254 * 1.001 + 1e-7)))
    check(name + ": CONTROL not the raw int8 bytes", float(np.abs(got.numpy() - cb.astype(np.float32)).max()) > 1.0)

try:
    run_dequant_model({p + ".weight": torch.from_numpy(cb), p + ".SCB": torch.from_numpy(scb), p + ".weight_format": torch.tensor(3, dtype=torch.uint8)},
                      {"quant_method": "bitsandbytes", "load_in_8bit": True})
    check("CONTROL legacy bnb weight_format != row is refused", False, "no exception")
except NotImplementedError:
    check("CONTROL legacy bnb weight_format != row is refused", True)
try:
    run_dequant_model({}, {"quant_method": "bitsandbytes"})
    check("CONTROL bitsandbytes with neither 4 nor 8 bit is refused", False, "no exception")
except NotImplementedError:
    check("CONTROL bitsandbytes with neither 4 nor 8 bit is refused", True)
try:
    run_dequant_model({}, {"quant_method": "not-a-real-method"})
    check("CONTROL unknown quant_method still raises", False, "no exception")
except NotImplementedError:
    check("CONTROL unknown quant_method still raises", True)

# ---------------------------------------------------------------- end to end: tiny llama dirs -> convert_hf_to_gguf.py -> read back
try:
    import sentencepiece as spm
    import safetensors.torch as st
    import gguf
except Exception as e:  # pragma: no cover
    print("SKIP end-to-end: %s" % e)
    print("RESULT: %d failure(s)" % len(failures))
    sys.exit(1 if failures else 0)

V, H, F, L = 60, 32, 64, 2
SHAPES = {"self_attn.q_proj": (H, H), "self_attn.k_proj": (H, H), "self_attn.v_proj": (H, H), "self_attn.o_proj": (H, H),
          "mlp.gate_proj": (F, H), "mlp.up_proj": (F, H), "mlp.down_proj": (H, F)}   # (out, in)
GG = {"self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def write_dir(td, name, quant_cfg, kind):
    """kind: 'plain' | 'gemma' | 'bnb8'. Returns (dir, expected {gguf name: ref array})."""
    d = Path(td) / name
    d.mkdir()
    if not (Path(td) / "tokenizer.model").exists():
        corpus = Path(td) / "corpus.txt"
        corpus.write_text("\n".join("the quick brown fox %d jumps over the lazy dog %d" % (i, i * 7) for i in range(200)))
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(Path(td) / "tokenizer"), vocab_size=60, model_type="bpe",
                                       character_coverage=1.0, bos_id=1, eos_id=2, unk_id=0, pad_id=-1)
    (d / "tokenizer.model").write_bytes((Path(td) / "tokenizer.model").read_bytes())
    conf = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": H, "intermediate_size": F, "num_hidden_layers": L,
            "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": V, "max_position_embeddings": 128, "rms_norm_eps": 1e-5,
            "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float16"}
    if quant_cfg:
        conf["quantization_config"] = quant_cfg
    (d / "config.json").write_text(json.dumps(conf))
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "LlamaTokenizer"}))
    r = np.random.default_rng(7)   # same float weights for every dir
    t, expected = {}, {}
    emb = (r.standard_normal((V, H)) * 0.1).astype(np.float32)
    t["model.norm.weight"] = torch.ones(H, dtype=torch.float16)
    t["lm_head.weight"] = torch.from_numpy((r.standard_normal((V, H)) * 0.1).astype(np.float16))
    if kind == "gemma":
        nb, blk = 4, H // 4
        codes, sc, ref = [], np.zeros((V, nb), dtype=np.float32), np.zeros((V, H), dtype=np.float32)
        for i in range(V):
            row = []
            for b in range(nb):
                seg = emb[i, b * blk:(b + 1) * blk]
                s = float(np.abs(seg).max()) / 7.0
                sc[i, b] = s
                for k, v in enumerate(seg):
                    c = min(max(int(round(float(v) / s)), -8), 7)
                    row.append(c)
                    ref[i, b * blk + k] = c * s
            codes.append(row)
        t["model.embed_tokens.embedding_quantized"] = to_t(pack(codes, 4), 4)
        t["model.embed_tokens.embedding_scale"] = torch.from_numpy(sc)
        expected["token_embd.weight"] = ref
    else:
        t["model.embed_tokens.weight"] = torch.from_numpy(emb.astype(np.float16))
    for l in range(L):
        t["model.layers.%d.input_layernorm.weight" % l] = torch.ones(H, dtype=torch.float16)
        t["model.layers.%d.post_attention_layernorm.weight" % l] = torch.ones(H, dtype=torch.float16)
        for nm, (n_out, n_in) in SHAPES.items():
            p = "model.layers.%d.%s" % (l, nm)
            w = (r.standard_normal((n_out, n_in)) * 0.1).astype(np.float32)
            if kind == "plain":
                t[p + ".weight"] = torch.from_numpy(w.astype(np.float16))
            elif kind == "gemma":
                bits = 2 if nm == "mlp.down_proj" else (8 if nm == "self_attn.o_proj" else 4)
                codes, scales = quant_rows(w, bits)
                t[p + ".weight"] = to_t(pack(codes, bits), bits)
                t[p + ".weight_scale"] = torch.tensor(scales, dtype=torch.float32).reshape(n_out, 1)
                t[p + ".input_activation_scale"] = torch.tensor(0.5)
                exp = np.array([[c * scales[i] for c in codes[i]] for i in range(n_out)], dtype=np.float32)
                if nm in GG:
                    expected["blk.%d.%s.weight" % (l, GG[nm])] = exp
            else:
                scb = np.abs(w).max(axis=1).astype(np.float32)
                cb = np.array([[int(round(float(v) / float(scb[i]) * 127.0)) for v in w[i]] for i in range(n_out)], dtype=np.int8)
                t[p + ".weight"], t[p + ".SCB"] = torch.from_numpy(cb), torch.from_numpy(scb)
                t[p + ".weight_format"] = torch.tensor(0, dtype=torch.uint8)
                if nm in GG:
                    expected["blk.%d.%s.weight" % (l, GG[nm])] = (cb.astype(np.float32) * scb.reshape(-1, 1) / 127.0).astype(np.float32)
    st.save_file(t, str(d / "model.safetensors"))
    return d, expected


def convert(tree, d, out):
    return subprocess.run([sys.executable, str(tree / "convert_hf_to_gguf.py"), str(d), "--outfile", str(out), "--outtype", "f32"],
                          capture_output=True, text=True, cwd=str(tree))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


with tempfile.TemporaryDirectory() as td:
    cases = [
        ("gemma", {"quant_method": "gemma", "num_bits": 4, "quantize_embeddings": True,
                   "module_quant_configs": {r"mlp\.down_proj": {"num_bits": 2}, r"self_attn\.o_proj": {"num_bits": 8}}}),
        ("bnb8", {"quant_method": "bitsandbytes", "load_in_8bit": True, "load_in_4bit": False, "llm_int8_threshold": 6.0}),
    ]
    for kind, qc in cases:
        d, expected = write_dir(td, kind, qc, kind)
        out = Path(td) / (kind + ".gguf")
        r = convert(root, d, out)
        check("convert_hf_to_gguf.py converts the %s directory" % kind, r.returncode == 0 and out.exists(), (r.stderr or r.stdout)[-700:])
        if out.exists():
            rd = gguf.GGUFReader(str(out))
            names = {x.name: x for x in rd.tensors}
            # IDENTITY-CHECKED: GGUF tensor names are exact strings; a leftover HF suffix is the thing being looked for
            check("%s: no packed / scale tensor reached the GGUF" % kind, not any(n.endswith(LEFTOVER) for n in names), str(list(names)))
            worst = 0.0
            for n, exp in expected.items():
                if n not in names:
                    check("%s GGUF holds %s" % (kind, n), False)
                    continue
                got = np.array(names[n].data, dtype=np.float32).reshape(exp.shape)
                worst = max(worst, float(np.abs(got - exp).max()))
            check("%s: GGUF tensors equal the independent dequantisation (%d tensors)" % (kind, len(expected)),
                  worst < 1e-6 and len(expected) >= 10, "worst=%g n=%d" % (worst, len(expected)))
            del names, rd
            gc.collect()

    # unquantized checkpoint: byte-identical, determinism control first
    dp, _ = write_dir(td, "plain", None, "plain")
    o1, o2 = Path(td) / "plain1.gguf", Path(td) / "plain2.gguf"
    r1, r2 = convert(root, dp, o1), convert(root, dp, o2)
    check("unquantized tiny llama converts (this tree)", r1.returncode == 0 and r2.returncode == 0, (r1.stderr or r2.stderr)[-500:])
    if o1.exists() and o2.exists():
        check("CONTROL two conversions of the same tree are byte-identical (identity is a real signal)", sha(o1) == sha(o2))
    if base_tree is not None and o1.exists():
        o3 = Path(td) / "plain_base.gguf"
        r3 = convert(base_tree, dp, o3)
        check("unquantized tiny llama converts (base tree)", r3.returncode == 0 and o3.exists(), (r3.stderr or r3.stdout)[-500:])
        if o3.exists():
            check("unquantized checkpoint: GGUF byte-identical base tree vs this tree", sha(o1) == sha(o3), "%s vs %s" % (sha(o1)[:12], sha(o3)[:12]))
    else:
        print("NOTE byte-identity vs base tree not run (no --base-tree)")

print("RESULT: %d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
