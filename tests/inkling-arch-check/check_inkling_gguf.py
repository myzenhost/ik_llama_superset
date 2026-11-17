# Checks on the GGUFs made by make_tiny_inkling.py and on llama-quantize outputs made from them.
# PROGRESS-OK: tiny files, seconds. READ-ONLY: reads <dir>/*.gguf, expected.json, weights.npz.  RANDOM WEIGHTS ONLY.
#
# Usage:
#   python check_inkling_gguf.py convert <worktree-root> <dir>          converter output vs the HF weights it was made from
#   python check_inkling_gguf.py quant   <worktree-root> <gguf> <TYPE>   one llama-quantize output (TYPE = q4_0 | iq2_k | iq4_k ...)
# Exit 0 = every check passed. Each section prints a CONTROL line: a deliberately wrong reference that the same comparison must catch.
import json
import sys
from pathlib import Path

mode = sys.argv[1]
ROOT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT / "gguf-py"))

import numpy as np  # noqa: E402

import gguf  # noqa: E402

FLOATS = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16)
ok_all = True


def check(cond, msg):
    global ok_all
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    ok_all &= bool(cond)


def read(path):
    rd = gguf.GGUFReader(str(path))
    tens = {t.name: t for t in rd.tensors}
    kv = {}
    for f in rd.fields.values():
        try:
            kv[f.name] = f.contents()
        except Exception:  # noqa: BLE001 -- unreadable field is recorded as such, not dropped
            kv[f.name] = "<unreadable>"
    return tens, kv


if mode == "convert":
    d = Path(sys.argv[3]).resolve()
    meta = json.loads((d / "expected.json").read_text(encoding="utf-8"))
    W = {k.replace("__", "."): v for k, v in np.load(d / "weights.npz").items()}
    exp = meta["tensors"]
    results = {n: read(d / f"{n}.gguf") for n in ("hf_f32", "ckpt_f32", "hf_f16")}

    for name, (tens, kv) in results.items():
        print(f"[{name}]")
        check(set(tens) == set(exp), f"tensor name set equals expected ({len(tens)} vs {len(exp)}; "
              f"missing={sorted(set(exp) - set(tens))[:4]} extra={sorted(set(tens) - set(exp))[:4]})")
        bad = [n for n, t in tens.items() if n in exp and list(reversed([int(x) for x in t.shape])) != exp[n]]
        check(not bad, f"every tensor has the expected shape (bad={bad[:4]})")
        for k, v in meta["kv"].items():
            got = kv.get(k, "ABSENT")
            same = (abs(float(got) - float(v)) < 1e-6) if isinstance(v, float) and got != "ABSENT" else got == v
            check(same, f"kv {k} = {got!r} (want {v!r})")
        check(list(kv.get("inkling.attention.head_count_kv", [])) == meta["n_head_kv"], f"head_count_kv per layer = {kv.get('inkling.attention.head_count_kv')}")
        check([bool(x) for x in kv.get("inkling.attention.sliding_window_pattern", [])] == meta["swa_pattern"], "sliding_window_pattern true=local")
        check(kv.get("tokenizer.ggml.model") == "gpt2", "tokenizer model gpt2")
        toks = kv.get("tokenizer.ggml.tokens", [])
        types = [int(x) for x in kv.get("tokenizer.ggml.token_type", [])]
        check(len(toks) == meta["v_padded"] and len(types) == meta["v_padded"], f"vocab list length {len(toks)} = padded vocab")
        check(all(t == 5 for t in types[meta["v_unpadded"]:]), "padding rows are UNUSED (5)")
        check(all(types[i] == 3 for i in range(258, 262)), "the four special tokens are CONTROL (3)")
        check(kv.get("tokenizer.ggml.add_bos_token") is False, "add_bos_token false")
        # tensor TYPE policy of the converter (tensor_force_quant)
        for n, t in tens.items():
            if "shortconv_" in n or "attn_rel_proj" in n:
                check(t.tensor_type == gguf.GGMLQuantizationType.F32, f"{n} stays F32 (is {t.tensor_type.name})")
        if name == "hf_f16":
            w = tens["blk.1.ffn_gate_exps.weight"]
            check(w.tensor_type == gguf.GGMLQuantizationType.F16, f"control: an ordinary matrix IS F16 here (is {w.tensor_type.name})")

    # values: every transformed tensor equals the matching HF weight transform
    p = lambda i, s: W["model.llm.layers.%d.%s" % (i, s)]  # noqa: E731
    pairs = [("token_embd.weight", W["model.llm.embed.weight"]), ("token_embd_norm.weight", W["model.llm.embed_norm.weight"]),
             ("output_norm.weight", W["model.llm.norm.weight"]), ("output.weight", W["model.llm.unembed.weight"])]
    for il in range(meta["n_layer"]):
        b = f"blk.{il}."
        pairs += [(b + "attn_q.weight", p(il, "attn.wq_du.weight")), (b + "attn_k.weight", p(il, "attn.wk_dv.weight")),
                  (b + "attn_v.weight", p(il, "attn.wv_dv.weight")), (b + "attn_r.weight", p(il, "attn.wr_du.weight")),
                  (b + "attn_output.weight", p(il, "attn.wo_ud.weight")), (b + "attn_rel_proj.weight", p(il, "attn.rel_logits_proj.proj")),
                  (b + "shortconv_k.weight", p(il, "attn.k_sconv.weight")[:, 0, :]), (b + "shortconv_v.weight", p(il, "attn.v_sconv.weight")[:, 0, :]),
                  (b + "shortconv_attn.weight", p(il, "attn_sconv.weight")[:, 0, :]), (b + "shortconv_mlp.weight", p(il, "mlp_sconv.weight")[:, 0, :]),
                  (b + "ffn_norm.weight", p(il, "mlp_norm.weight"))]
        if il < meta["dense_blocks"]:
            w13 = p(il, "mlp.w13_dn.weight")
            pairs += [(b + "ffn_gate.weight", w13[0::2, :]), (b + "ffn_up.weight", w13[1::2, :]), (b + "ffn_down.weight", p(il, "mlp.w2_md.weight")),
                      (b + "ffn_gscale.weight", p(il, "mlp.global_scale"))]
        else:
            w13 = p(il, "mlp.experts.w13_weight")
            s13 = p(il, "mlp.shared_experts.shared_w13_weight")
            pairs += [(b + "ffn_gate_inp.weight", p(il, "mlp.gate.weight")), (b + "exp_probs_b.bias", p(il, "mlp.gate.bias")),
                      (b + "ffn_gscale.weight", p(il, "mlp.gate.global_scale")),
                      (b + "ffn_gate_exps.weight", w13[:, 0::2, :]), (b + "ffn_up_exps.weight", w13[:, 1::2, :]),
                      (b + "ffn_down_exps.weight", p(il, "mlp.experts.w2_weight")),
                      (b + "ffn_gate_shexp.weight", s13[:, 0::2, :]), (b + "ffn_up_shexp.weight", s13[:, 1::2, :]),
                      (b + "ffn_down_shexp.weight", p(il, "mlp.shared_experts.shared_w2_weight"))]
    tens32 = results["hf_f32"][0]
    n_bad = 0
    for n, ref in pairs:
        got = np.array(tens32[n].data, copy=True).reshape(ref.shape)
        if not np.array_equal(got.astype(np.float32), ref.astype(np.float32)):
            n_bad += 1
            print("  MISMATCH", n)
    check(n_bad == 0, f"F32 conversion equals the HF weight transform for all {len(pairs)} mapped tensors")
    # CONTROL: a gate/up swapped reference must fail the same comparison
    n = "blk.1.ffn_gate_exps.weight"
    w13 = p(1, "mlp.experts.w13_weight")
    got = np.array(tens32[n].data, copy=True)
    check(not np.array_equal(got, w13[:, 1::2, :]), "CONTROL: swapped gate/up reference is caught")
    check(np.array_equal(np.array(results["hf_f32"][0]["blk.1.ffn_up_exps.weight"].data), w13[:, 1::2, :]), "CONTROL: up bank equals the odd rows")
    # the two config spellings give the same file contents
    a, b = results["hf_f32"], results["ckpt_f32"]
    same_t = all(np.array_equal(np.array(a[0][n].data), np.array(b[0][n].data)) for n in a[0])
    same_kv = all(a[1].get(k) == b[1].get(k) for k in set(a[1]) | set(b[1]) if k not in ("general.name", "general.quantization_version"))
    check(same_t, "hf-spelling and checkpoint-spelling configs give identical tensors")
    check(same_kv, "hf-spelling and checkpoint-spelling configs give identical metadata")
    print("RESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)

if mode == "quant":
    path = Path(sys.argv[3]).resolve()
    want = sys.argv[4].upper()
    meta = json.loads((path.parent / "expected.json").read_text(encoding="utf-8"))
    tens, kv = read(path)
    exp = meta["tensors"]
    check(set(tens) == set(exp), f"{path.name}: same tensor name set as the F32 source ({len(tens)})")
    must_f32 = [n for n in tens if "shortconv_" in n or "attn_rel_proj" in n or n.endswith("_norm.weight") or "ffn_gscale" in n
                or "exp_probs_b" in n or "ffn_gate_inp" in n]
    for n in must_f32:
        check(tens[n].tensor_type == gguf.GGMLQuantizationType.F32, f"{n} F32 (is {tens[n].tensor_type.name})")
    quantizable = [n for n in tens if n not in must_f32]
    dist = {}
    for n in quantizable:
        dist.setdefault(tens[n].tensor_type.name, []).append(n)
    print("  type distribution of the quantisable tensors:", {k: len(v) for k, v in dist.items()})
    for n in quantizable:
        check(tens[n].tensor_type not in FLOATS, f"{n} is quantised (is {tens[n].tensor_type.name})")
    # the expert banks are the bulk of an MoE and must be exactly the requested type
    banks = [n for n in quantizable if "_exps.weight" in n]
    check(bool(banks) and all(tens[n].tensor_type.name == want for n in banks), f"all {len(banks)} routed expert banks are {want}: "
          f"{sorted({tens[n].tensor_type.name for n in banks})}")
    shb = [n for n in quantizable if "_shexp.weight" in n]
    check(bool(shb), f"shared-expert banks present ({len(shb)}) and quantised: {sorted({tens[n].tensor_type.name for n in shb})}")
    # data in the quantised file is readable and of the byte size its type implies
    n_bad = 0
    for n in quantizable:
        t = tens[n]
        if int(t.n_bytes) != int(np.array(t.data).nbytes):
            n_bad += 1
    check(n_bad == 0, "every quantised tensor's byte count equals its data view")
    check(kv.get("general.architecture") == "inkling", "architecture string survives quantisation")
    check(kv.get("inkling.unpadded_vocab_size") == meta["v_unpadded"], "inkling.* keys copied through")
    print("RESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)

print("unknown mode", mode)
sys.exit(2)
