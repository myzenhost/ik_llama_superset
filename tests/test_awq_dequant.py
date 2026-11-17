#!/usr/bin/env python3
"""AutoAWQ 4-bit "gemm" checkpoints through the REAL ModelBase.dequant_model, and through convert_hf_to_gguf.py end to end.

The reference PACKER below is written independently from the layout CTranslate2's CUDA kernel reads (src/ops/awq/dequantize.cuh:12-62
+ dequantize_gpu.cu:30-60): output column c of a group of eight lives in nibble NIB_OF_COL[c] of the int32 (plain loops, nothing
imported from conversion.base), so a bug shared by the vectorised unpacker and its own test cannot hide.

   python tests/test_awq_dequant.py <tree-root>

BEFORE (tree without the "awq" branch of dequant_model): the packed qweight/qzeros/scales tensors are left in the model, the
converter has no name mapping for them -> every case FAILS (unit: the tensors stay unconverted; end to end: conversion error).
AFTER: all pass. Controls: (a) the data discriminates the AWQ column order from the naive in-order reading, (b) dequantised != the
raw integers (a no-op cannot pass), (c) 8-bit and the gemv / no-zero-point variants are refused instead of mis-read.
Row: S69-ENGINE-MERGE-PROGRAMME-ALL-ENGINES-2026-09-09-01 (CTranslate2 AWQ, MIT, behaviour only).
"""
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "gguf-py"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import conversion.base as base  # noqa: E402

NIB_OF_COL = {0: 0, 1: 4, 2: 1, 3: 5, 4: 2, 5: 6, 6: 3, 7: 7}   # read off the lop3 sequence of dequantize.cuh
failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" -- " + detail) if (detail and not cond) else ""))
    if not cond:
        failures.append(name)


def pack_rows(vals):
    """vals [rows, n_out] ints 0..15 -> int32 [rows, n_out/8], AWQ nibble order (plain loops)."""
    rows, n_out = vals.shape
    out = np.zeros((rows, n_out // 8), dtype=np.int64)
    for r in range(rows):
        for p in range(n_out // 8):
            word = 0
            for c in range(8):
                word |= int(vals[r, p * 8 + c]) << (4 * NIB_OF_COL[c])
            out[r, p] = word
    out = np.where(out >= 2 ** 31, out - 2 ** 32, out)
    return out.astype(np.int32)


def make_layer(rng, n_in, n_out, group):
    q = rng.integers(0, 16, size=(n_in, n_out))
    z = rng.integers(0, 16, size=(n_in // group, n_out))
    s = (rng.random((n_in // group, n_out)) * 0.2 + 0.01).astype(np.float16)
    expect = np.zeros((n_out, n_in), dtype=np.float32)
    for i in range(n_in):                                   # independent loop: (q - z[group]) * s[group], stored [out, in]
        for o in range(n_out):
            g = i // group
            expect[o, i] = (float(q[i, o]) - float(z[g, o])) * float(s[g, o])
    return pack_rows(q), pack_rows(z), s, q, expect


def run_dequant_model(tensors, quant_config):
    stub = types.SimpleNamespace(
        _is_nvfp4=False, hparams={"quantization_config": quant_config}, lazy=False, _fp8_as_q8=False,
        _fp8_dequantized=set(), model_tensors={k: (lambda v=v: v) for k, v in tensors.items()},
    )
    base.ModelBase.dequant_model(stub)
    return stub


rng = np.random.default_rng(7)
cfg = {"quant_method": "awq", "bits": 4, "group_size": 8, "version": "gemm", "zero_point": True}

# ---- unit: dequant_model on a packed layer
qw, qz, sc, q, expect = make_layer(rng, 16, 24, 8)
tensors = {"m.qweight": torch.from_numpy(qw), "m.qzeros": torch.from_numpy(qz), "m.scales": torch.from_numpy(sc)}
try:
    stub = run_dequant_model(dict(tensors), cfg)
    got = stub.model_tensors["m.weight"]().float().numpy()
    check("dequant_model accepts an awq checkpoint", True)
    check("shape is [out, in]", got.shape == (24, 16), str(got.shape))
    err = float(np.abs(got - expect).max())
    check("equals the independent reference", err < 1e-6, "max|diff|=%g" % err)
    check("packed tensors removed", not any(k.endswith((".qweight", ".qzeros", ".scales")) for k in stub.model_tensors),
          str(list(stub.model_tensors)))
except Exception as e:
    check("dequant_model accepts an awq checkpoint", False, "%s: %s" % (type(e).__name__, e))

# controls: the data separates the AWQ column order from the naive one; the result is not the raw integers
naive = np.zeros_like(expect)
for o in range(24):
    for i in range(16):
        naive[o, i] = expect[o, i]
perm = [0, 2, 4, 6, 1, 3, 5, 7]
swapped = expect.reshape(3, 8, 16)[:, perm, :].reshape(24, 16)
check("CONTROL data discriminates the column order", float(np.abs(swapped - expect).max()) > 1e-3)
check("CONTROL result is not the raw integers", float(np.abs(expect - q.T).max()) > 1e-3)

# ---- hand-checked word, from AutoAWQ's own packer (MIT, casper-hansen/AutoAWQ@88e4c76b awq/modules/linear/gemm.py:220-227:
# order_map = [0, 2, 4, 6, 1, 3, 5, 7]; qweight[:, col] |= intweight[:, col*8 + order_map[i]] << (i*4)), i.e. nibble i holds column
# order_map[i]. Columns 0..7 = 1..8 -> nibbles 0..7 = 1,3,5,7,2,4,6,8 -> 0x86427531. Agrees with NIB_OF_COL (read from CT2) above.
check("CONTROL NIB_OF_COL equals AutoAWQ order_map", all(NIB_OF_COL[c] == [0, 2, 4, 6, 1, 3, 5, 7].index(c) for c in range(8)))
word = torch.tensor([[0x86427531]], dtype=torch.int64).to(torch.int32)          # sign bit set: 0x8... is negative as int32
hand = base.dequant_awq(word, torch.zeros((1, 1), dtype=torch.int32), torch.ones((1, 8), dtype=torch.float16), 4, -1)
check("hand-checked word 0x86427531 -> columns 1..8", hand.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
      str(hand.flatten().tolist()))

# ---- a known FLOAT weight quantised by AutoAWQ's rule, packed, dequantised: within the 4-bit error bound of the original.
# Rule (MIT, AutoAWQ@88e4c76b awq/quantize/quantizer.py:82-92, zero_point branch): per group of `group` input columns,
# s = (max - min).clamp(1e-5) / 15, z = clamp(-round(min / s), 0, 15), q = clamp(round(w / s) + z, 0, 15); scales are stored
# fp16 (gemm.py:190), layouts [in/group, out] (qweight [in, out/8], gemm.py:205). Non-square [out=64, in=256], group 128.
W = rng.standard_normal((64, 256)).astype(np.float32) * 0.05
G2 = 128
qf = np.zeros((256, 64), dtype=np.int64)
zf = np.zeros((256 // G2, 64), dtype=np.int64)
sf = np.zeros((256 // G2, 64), dtype=np.float32)
for o in range(64):
    for g in range(256 // G2):
        blk = W[o, g * G2:(g + 1) * G2]
        s_ = max(float(blk.max() - blk.min()), 1e-5) / 15.0
        z_ = int(min(max(-np.round(float(blk.min()) / s_), 0), 15))
        sf[g, o], zf[g, o] = s_, z_
        qf[g * G2:(g + 1) * G2, o] = np.clip(np.round(blk / s_) + z_, 0, 15)
s16 = sf.astype(np.float16)
deq_ref = np.zeros((64, 256), dtype=np.float64)
for o in range(64):
    for i in range(256):
        deq_ref[o, i] = (qf[i, o] - zf[i // G2, o]) * float(s16[i // G2, o])
got_f = base.dequant_awq(torch.from_numpy(pack_rows(qf)), torch.from_numpy(pack_rows(zf)), torch.from_numpy(s16), 4, G2).numpy()
check("float weight: shape [out, in] = (64, 256)", got_f.shape == (64, 256), str(got_f.shape))
err_ref = float(np.abs(got_f - deq_ref).max())
check("float weight: equals the AutoAWQ-rule dequantisation", err_ref < 1e-6, "max|diff|=%g" % err_ref)
# |w - deq| <= s/2 (rounding) + 15*|s - s16| (fp16 scale storage); measured as a fraction of each group's step s
ratio = max(float(np.abs(W[o, i] - got_f[o, i]) - 15 * abs(sf[i // G2, o] - float(s16[i // G2, o]))) / float(sf[i // G2, o])
            for o in range(64) for i in range(256))
check("float weight: within half a quantisation step of the original (worst %.4f step)" % ratio, ratio <= 0.5 + 1e-4)
print("INFO float weight: worst |w - deq| = %.3g, mean step s = %.3g" % (float(np.abs(W - got_f).max()), float(sf.mean())))
wrong = got_f.reshape(64 // 8, 8, 256)[:, [0, 2, 4, 6, 1, 3, 5, 7], :].reshape(64, 256)        # the naive (sequential) nibble read
check("CONTROL float weight: a sequential nibble read is far outside the bound",
      float(np.abs(W - wrong).max()) > 4 * float(sf.max()), "%g" % float(np.abs(W - wrong).max()))

# ---- version is case-insensitive, as both readers treat it: AutoAWQ awq/models/_config.py:24 (version.lower()) and transformers
# utils/quantization_config.py AWQLinearVersion.from_str (version.lower()); a config carrying "GEMM" is the same gemm layout.
try:
    stub = run_dequant_model(dict(tensors), dict(cfg, version="GEMM"))
    err = float(np.abs(stub.model_tensors["m.weight"]().float().numpy() - expect).max())
    check("version \"GEMM\" (upper case) is accepted as gemm", err < 1e-6, "max|diff|=%g" % err)
except Exception as e:
    check("version \"GEMM\" (upper case) is accepted as gemm", False, "%s: %s" % (type(e).__name__, e))

# ---- a quantization_config with no "bits" key is the 4-bit default (the validation above reads it with .get("bits", 4)); the tensor
# closure must read it the same way, not index the key and die with a KeyError only when the weight is finally evaluated.
try:
    nobits = {k: v for k, v in cfg.items() if k != "bits"}
    stub = run_dequant_model(dict(tensors), nobits)
    err = float(np.abs(stub.model_tensors["m.weight"]().float().numpy() - expect).max())
    check("a config without bits dequantises as the 4-bit default", err < 1e-6, "max|diff|=%g" % err)
except Exception as e:
    check("a config without bits dequantises as the 4-bit default", False, "%s: %s" % (type(e).__name__, e))

# ---- GEMV layout and llm-awq order. Independent packers, plain loops, written from AutoAWQ awq/modules/linear/gemv.py (MIT, @88e4c76b):
# gemv packs eight INPUT columns per int32 (qweight [out, in/8], :120-134, order_map identity :128), qzeros [out, zw] (:137-152, identity),
# scales [out, zw*8] zero padded (:97-106), zw = calculate_zeros_width (:12-24). llm-awq gemm keeps the gemm shapes with identity order.
ORDER_AUTOAWQ = [0, 2, 4, 6, 1, 3, 5, 7]
ORDER_ID = list(range(8))


def pack_words(vals, order):
    """vals [rows, n] ints 0..15 (n multiple of 8) -> int32 [rows, n/8]; nibble i of word p holds column p*8+order[i]."""
    rows, n = vals.shape
    out = np.zeros((rows, n // 8), dtype=np.int64)
    for r in range(rows):
        for p in range(n // 8):
            word = 0
            for i in range(8):
                word |= int(vals[r, p * 8 + order[i]]) << (4 * i)
            out[r, p] = word
    return np.where(out >= 2 ** 31, out - 2 ** 32, out).astype(np.int32)


def zeros_width(n_in, group):
    mult = 1 if group >= 128 else {64: 2, 32: 4}[group]
    w = -(-(n_in // group) // 8)
    return -(-w // mult) * mult


def gemv_layer(rng, n_in, n_out, group, order):
    q = rng.integers(0, 16, size=(n_out, n_in))
    ng = n_in // group
    z = rng.integers(0, 16, size=(n_out, ng))
    s = (rng.random((n_out, ng)) * 0.2 + 0.01).astype(np.float16)
    zw = zeros_width(n_in, group)
    zp = np.zeros((n_out, zw * 8), dtype=np.int64)
    zp[:, :ng] = z
    sp = np.zeros((n_out, zw * 8), dtype=np.float16)
    sp[:, :ng] = s
    expect = np.zeros((n_out, n_in), dtype=np.float32)
    for o in range(n_out):
        for i in range(n_in):
            expect[o, i] = (float(q[o, i]) - float(z[o, i // group])) * float(s[o, i // group])
    return pack_words(q, order), pack_words(zp, order), sp, expect


for combo, gcfg, order in (
        ("gemv group 128 (non-square 48x256)", dict(cfg, version="gemv", group_size=128), ORDER_ID),
        ("gemv group 64 (padded zero width)", dict(cfg, version="GEMV", group_size=64), ORDER_ID),
        ("gemv group 32", dict(cfg, version="gemv", group_size=32), ORDER_ID)):
    grp = gcfg["group_size"]
    qw_, qz_, sc_, exp_ = gemv_layer(rng, 256, 48, grp, order)
    try:
        stub = run_dequant_model({"m.qweight": torch.from_numpy(qw_), "m.qzeros": torch.from_numpy(qz_), "m.scales": torch.from_numpy(sc_)}, gcfg)
        got_ = stub.model_tensors["m.weight"]().float().numpy()
        check("%s: shape [out, in]" % combo, got_.shape == (48, 256), str(got_.shape))
        e_ = float(np.abs(got_ - exp_).max())
        check("%s: equals the independent reference" % combo, e_ < 1e-6, "max|diff|=%g" % e_)
    except Exception as e:
        check("%s: dequantises" % combo, False, "%s: %s" % (type(e).__name__, e))

# llm-awq gemm: same shapes as autoawq gemm, identity nibble order
qq = rng.integers(0, 16, size=(16, 24))
zz = rng.integers(0, 16, size=(2, 24))
ss = (rng.random((2, 24)) * 0.2 + 0.01).astype(np.float16)
exp_l = np.zeros((24, 16), dtype=np.float32)
for i in range(16):
    for o in range(24):
        exp_l[o, i] = (float(qq[i, o]) - float(zz[i // 8, o])) * float(ss[i // 8, o])
lt = {"m.qweight": torch.from_numpy(pack_words(qq, ORDER_ID)), "m.qzeros": torch.from_numpy(pack_words(zz, ORDER_ID)), "m.scales": torch.from_numpy(ss)}
try:
    stub = run_dequant_model(dict(lt), dict(cfg, backend="llm-awq"))
    e_ = float(np.abs(stub.model_tensors["m.weight"]().float().numpy() - exp_l).max())
    check("llm-awq gemm (identity order) equals the independent reference", e_ < 1e-6, "max|diff|=%g" % e_)
    stub = run_dequant_model(dict(lt), cfg)
    e_ = float(np.abs(stub.model_tensors["m.weight"]().float().numpy() - exp_l).max())
    check("CONTROL autoawq order on identity-packed data is wrong (test discriminates)", e_ > 1e-3, "max|diff|=%g" % e_)
except Exception as e:
    check("llm-awq gemm dequantises", False, "%s: %s" % (type(e).__name__, e))

# ---- refusals
for name, c in (("8-bit", dict(cfg, bits=8)), ("unknown version", dict(cfg, version="exllama")), ("no zero point", dict(cfg, zero_point=False))):
    try:
        run_dequant_model(dict(tensors), c)
        check("CONTROL %s awq is refused" % name, False, "no exception")
    except NotImplementedError:
        check("CONTROL %s awq is refused" % name, True)
    except Exception as e:
        check("CONTROL %s awq is refused" % name, False, "%s: %s" % (type(e).__name__, e))

# ---- end to end: a tiny llama directory -> convert_hf_to_gguf.py -> read the GGUF back
try:
    import sentencepiece as spm
    import safetensors.torch as st
    import gguf
except Exception as e:  # pragma: no cover
    print("SKIP end-to-end: %s" % e)
    print("RESULT: %d failure(s)" % len(failures))
    sys.exit(1 if failures else 0)

with tempfile.TemporaryDirectory() as td:
    d = Path(td) / "hf"
    d.mkdir()
    corpus = Path(td) / "corpus.txt"
    corpus.write_text("\n".join("the quick brown fox %d jumps over the lazy dog %d" % (i, i * 7) for i in range(200)))
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(d / "tokenizer"), vocab_size=60, model_type="bpe",
                                   character_coverage=1.0, bos_id=1, eos_id=2, unk_id=0, pad_id=-1)
    (d / "tokenizer.model").write_bytes((d / "tokenizer.model").read_bytes())
    V, H, F, L, G = 60, 32, 64, 2, 8
    (d / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": H, "intermediate_size": F, "num_hidden_layers": L,
        "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": V, "max_position_embeddings": 128, "rms_norm_eps": 1e-5,
        "bos_token_id": 1, "eos_token_id": 2, "torch_dtype": "float16", "quantization_config": cfg | {"group_size": G}}))
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "LlamaTokenizer"}))
    t, expected = {}, {}
    t["model.embed_tokens.weight"] = torch.from_numpy(rng.standard_normal((V, H)).astype(np.float16))
    t["model.norm.weight"] = torch.ones(H, dtype=torch.float16)
    t["lm_head.weight"] = torch.from_numpy(rng.standard_normal((V, H)).astype(np.float16))
    shapes = {"self_attn.q_proj": (H, H), "self_attn.k_proj": (H, H), "self_attn.v_proj": (H, H), "self_attn.o_proj": (H, H),
              "mlp.gate_proj": (H, F), "mlp.up_proj": (H, F), "mlp.down_proj": (F, H)}
    gguf_name = {"self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate",
                 "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}   # q / k are head-permuted by the converter: not compared
    for l in range(L):
        t["model.layers.%d.input_layernorm.weight" % l] = torch.ones(H, dtype=torch.float16)
        t["model.layers.%d.post_attention_layernorm.weight" % l] = torch.ones(H, dtype=torch.float16)
        for nm, (n_in, n_out) in shapes.items():
            qw, qz, sc, _, expect = make_layer(rng, n_in, n_out, G)
            p = "model.layers.%d.%s" % (l, nm)
            t[p + ".qweight"], t[p + ".qzeros"], t[p + ".scales"] = torch.from_numpy(qw), torch.from_numpy(qz), torch.from_numpy(sc)
            if nm in gguf_name:
                expected["blk.%d.%s.weight" % (l, gguf_name[nm])] = expect
    st.save_file(t, str(d / "model.safetensors"))
    out = Path(td) / "out.gguf"
    r = subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), str(d), "--outfile", str(out), "--outtype", "f32"],
                       capture_output=True, text=True, cwd=str(root))
    check("convert_hf_to_gguf.py converts the awq directory", r.returncode == 0 and out.exists(), (r.stderr or r.stdout)[-600:])
    if out.exists():
        rd = gguf.GGUFReader(str(out))
        names = {x.name: x for x in rd.tensors}
        check("no packed tensor reached the GGUF", not any(n.endswith(("qweight", "qzeros", "scales")) for n in names), str(list(names)))
        worst = 0.0
        for n, exp in expected.items():
            if n not in names:
                check("GGUF holds " + n, False)
                continue
            got = np.array(names[n].data, dtype=np.float32).reshape(exp.shape)
            worst = max(worst, float(np.abs(got - exp).max()))
        check("GGUF tensors equal the independent dequantisation (%d tensors)" % len(expected), worst < 1e-6 and len(expected) == 10,
              "worst=%g n=%d" % (worst, len(expected)))
        # GGUFReader holds out.gguf through an np.memmap; on Windows the directory cleanup then fails with WinError 32
        # (measured 2026-10-16: all checks PASS, then the test exits 1 from TemporaryDirectory.__exit__). Release it first.
        del names, rd
        import gc
        gc.collect()

print("RESULT: %d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
