#!/usr/bin/env python3
"""MiniCPM4 longrope attention scaling factor (rope.scaling.attn_factor) -- converter test.

DEGRADED PROXY: a tiny synthetic checkpoint (hidden 64, 2 layers, vocab-less) proves converter WIRING only (the key is written with the
HF value, and absent when the factor is 1). It says nothing about a real MiniCPM4 checkpoint's logits.

Run (CPU, no GPU, no build):
    python tests/test_minicpm4_longrope_attn_factor.py                      # unit + convert-and-read-back
    python tests/test_minicpm4_longrope_attn_factor.py --baseline-tree DIR  # also byte-compare factor==1 GGUFs against the converter
                                                                            # tree in DIR (e.g. a `git archive <old sha>` export)
Pass criterion: prints `ALL OK`, exit code 0.

Internal mode (spawned by the driver): --convert TREE CASE OUT
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

N_EMBD, N_HEAD, N_KV, N_LAYER, N_FF, N_VOCAB = 64, 4, 2, 2, 128, 32
ROPE_HALF = (N_EMBD // N_HEAD) // 2  # 8 long/short factors
KEY = "minicpm.rope.scaling.attn_factor"
ORIG_KEY = "minicpm.rope.scaling.original_context_length"

# case -> (max_position_embeddings, rope_scaling or None)
LONG = [1.0 + 0.5 * i for i in range(ROPE_HALF)]
SHORT = [1.0 + 0.1 * i for i in range(ROPE_HALF)]
CASES = {
    "longrope_4x": (131072, {"rope_type": "longrope", "long_factor": LONG, "short_factor": SHORT, "original_max_position_embeddings": 32768}),
    "longrope_eq": (32768, {"rope_type": "longrope", "long_factor": LONG, "short_factor": SHORT, "original_max_position_embeddings": 32768}),  # MiniCPM4-8B shape
    "plain": (4096, None),  # old MiniCPM-2B shape: no rope_scaling
}


def hf_formula(max_pos: int, orig: int) -> float:
    """modeling_minicpm.py:461-462 (openbmb/MiniCPM4-8B @ bb2ae14c), copied verbatim in meaning."""
    scale = max_pos / orig
    return math.sqrt(1 + math.log(scale) / math.log(orig))


def write_ckpt(d: Path, max_pos: int, rope_scaling) -> None:
    import torch
    from safetensors.torch import save_file

    cfg = {
        "architectures": ["MiniCPMForCausalLM"], "model_type": "minicpm", "hidden_size": N_EMBD, "intermediate_size": N_FF,
        "max_position_embeddings": max_pos, "num_attention_heads": N_HEAD, "num_hidden_layers": N_LAYER, "num_key_value_heads": N_KV,
        "rms_norm_eps": 1e-6, "vocab_size": N_VOCAB, "rope_theta": 10000.0, "scale_emb": 12, "scale_depth": 1.4, "dim_model_base": 256,
        "tie_word_embeddings": False, "torch_dtype": "float32",
    }
    if rope_scaling is not None:
        cfg["rope_scaling"] = rope_scaling
    (d / "config.json").write_text(json.dumps(cfg))
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    hd = N_EMBD // N_HEAD
    t = {"model.embed_tokens.weight": r(N_VOCAB, N_EMBD), "lm_head.weight": r(N_VOCAB, N_EMBD), "model.norm.weight": r(N_EMBD)}
    for i in range(N_LAYER):
        p = f"model.layers.{i}."
        t[p + "self_attn.q_proj.weight"] = r(N_HEAD * hd, N_EMBD)
        t[p + "self_attn.k_proj.weight"] = r(N_KV * hd, N_EMBD)
        t[p + "self_attn.v_proj.weight"] = r(N_KV * hd, N_EMBD)
        t[p + "self_attn.o_proj.weight"] = r(N_EMBD, N_HEAD * hd)
        t[p + "mlp.gate_proj.weight"] = r(N_FF, N_EMBD)
        t[p + "mlp.up_proj.weight"] = r(N_FF, N_EMBD)
        t[p + "mlp.down_proj.weight"] = r(N_EMBD, N_FF)
        t[p + "input_layernorm.weight"] = r(N_EMBD)
        t[p + "post_attention_layernorm.weight"] = r(N_EMBD)
    save_file(t, str(d / "model.safetensors"))


def convert_main(tree: str, case: str, out: str) -> None:
    """Run inside a subprocess against the converter tree `tree`."""
    sys.path.insert(1, os.path.join(tree, "gguf-py"))
    sys.path.insert(0, tree)
    import torch
    import gguf
    from conversion import ModelBase  # noqa: F401
    from conversion.minicpm import MiniCPMModel

    # tiny proxy has no tokenizer.model: stub only the vocab step, everything else is the real converter path
    MiniCPMModel.set_vocab = lambda self: self._set_vocab_none()
    max_pos, rs = CASES[case]
    with tempfile.TemporaryDirectory() as td, torch.inference_mode():
        d = Path(td) / "tiny-minicpm"  # fixed dir name: general.name/basename derive from it, a random tempdir name would break byte-compare
        d.mkdir()
        write_ckpt(d, max_pos, rs)
        m = MiniCPMModel(d, gguf.LlamaFileType.ALL_F32, Path(out))
        m.write()


def read_kv(path: Path) -> dict:
    import gguf
    rd = gguf.GGUFReader(str(path))
    kv = {}
    for k, f in rd.fields.items():
        if k.startswith("GGUF."):
            continue
        kv[k] = f.contents() if hasattr(f, "contents") else f.parts[f.data[0]].tolist()
    return kv


def run_convert(tree: Path, case: str, out: Path) -> None:
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--convert", str(tree), case, str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError(f"convert {case} @ {tree} failed rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-tree", default=None)
    ap.add_argument("--convert", nargs=3, metavar=("TREE", "CASE", "OUT"))
    ap.add_argument("--hf-src", default=None, help="optional: fetched modeling_minicpm.py to cross-check the formula text")
    a = ap.parse_args()
    if a.convert:
        convert_main(*a.convert)
        return 0

    sys.path.insert(1, str(REPO / "gguf-py"))
    sys.path.insert(0, str(REPO))
    from conversion.minicpm import minicpm_longrope_attn_factor as f

    # (1) unit: helper == HF formula
    assert f(32768, 32768) is None, "MiniCPM4-8B (max==orig) must give no factor"
    for mp, orig in ((131072, 32768), (65536, 32768), (262144, 32768), (8192, 4096)):
        got, want = f(mp, orig), hf_formula(mp, orig)
        assert got is not None and abs(got - want) < 1e-12, (mp, orig, got, want)
        print(f"unit ok: max={mp} orig={orig} factor={got:.12f} == HF {want:.12f}")
    assert abs(f(131072, 32768) - 1.0645) < 1e-3  # sanity on magnitude: sqrt(1+ln4/ln32768)
    assert f(None, 32768) is None and f(4096, 1) is None
    hf = Path(a.hf_src) if a.hf_src else None
    if hf and hf.is_file():
        lines = hf.read_text(encoding="utf-8").splitlines()
        assert "scale = (max_position_embeddings / self.original_max_position_embeddings)" in lines[460]
        assert "math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))" in lines[461]
        print("HF source lines 461-462 match the formula under test")

    # (2) convert tiny checkpoints, read back
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        outs = {}
        for c in CASES:
            outs[c] = td / f"{c}.gguf"
            run_convert(REPO, c, outs[c])
        kv4 = read_kv(outs["longrope_4x"])
        want = hf_formula(131072, 32768)
        assert KEY in kv4, f"{KEY} missing; keys={sorted(k for k in kv4 if 'rope' in k)}"
        got = float(kv4[KEY][0] if isinstance(kv4[KEY], (list, tuple)) else kv4[KEY])
        assert abs(got - want) < 1e-6, (got, want)  # stored as float32
        print(f"convert ok: longrope_4x writes {KEY}={got:.8f} (HF {want:.8f}); orig ctx key present: {ORIG_KEY in kv4}")
        for c in ("longrope_eq", "plain"):
            kv = read_kv(outs[c])
            assert KEY not in kv, f"{c}: {KEY} must be absent, got {kv[KEY]}"
            print(f"convert ok: {c} has no {KEY}")

        # (3) before/after: factor-1 cases byte-identical to the baseline converter
        if a.baseline_tree:
            for c in ("longrope_eq", "plain"):
                b = td / f"{c}.base.gguf"
                run_convert(Path(a.baseline_tree), c, b)
                same = b.read_bytes() == outs[c].read_bytes()
                if not same:
                    kn, kb0 = read_kv(outs[c]), read_kv(b)
                    raise AssertionError(f"{c}: GGUF differs from baseline converter; kv diff: "
                                         f"{[k for k in set(kn) | set(kb0) if kn.get(k) != kb0.get(k)]}")
                print(f"before/after ok: {c} byte-identical to baseline ({outs[c].stat().st_size} bytes)")
            # and the factor case differs ONLY by the one key
            b = td / "longrope_4x.base.gguf"
            run_convert(Path(a.baseline_tree), "longrope_4x", b)
            kb = read_kv(b)
            diff = sorted(set(kv4) ^ set(kb)) + [k for k in kb if k in kv4 and kb[k] != kv4[k]]
            assert diff == [KEY], diff
            print(f"before/after ok: longrope_4x differs from baseline only in {diff}")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
