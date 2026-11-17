#!/usr/bin/env python3
"""Baichuan2 NormHead: the HF checkpoint stores the UNNORMALISED lm_head; HF's NormHead L2-normalises every vocab row on the first
forward. The converter must therefore emit unit-norm output.weight rows for Baichuan2 (vocab 125696) and leave Baichuan-1 (vocab 64000)
alone. Drives the REAL BaichuanModel.modify_tensors (no checkpoint needed).

   python tests/test_baichuan2_normhead.py <tree-root>

BEFORE (no NormHead handling): the converted head keeps its raw row norms (0.5..2) -> the Baichuan2 cases FAIL (and logits are scaled
per vocab row). AFTER: rows have norm 1 +-1e-3, logits equal the HF reference hidden @ F.normalize(lm_head).T within 1e-3, the
vocab-64000 negative control keeps its raw head bit-for-bit, non-head tensors are untouched.
Row: S69-ENGREV-IPEXLLM-BAICHUAN2-NORMHEAD-CONVERT-2026-10-27-01. Design only from ipex-llm convert.py:912-927 (Apache-2.0, no code copied).
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "gguf-py"))

import torch  # noqa: E402

import gguf  # noqa: E402
from conversion.baichuan import BaichuanModel  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" -- " + detail) if (detail and not cond) else ""))
    if not cond:
        failures.append(name)


def make(vocab):
    m = object.__new__(BaichuanModel)
    m.hparams = {"vocab_size": vocab, "num_attention_heads": 4, "hidden_size": 64}
    m.model_arch = gguf.MODEL_ARCH.BAICHUAN
    m.fuse_gate_up_exps = False
    m.fuse_qkv = False
    m._unfusable_gate_up_layers = set()
    m.tensor_map =gguf.get_tensor_name_map(gguf.MODEL_ARCH.BAICHUAN, 2)
    return m


def convert(m, name, t):
    out = list(m.modify_tensors(t, name, None))
    return {k: v for k, v in out}


g = torch.Generator().manual_seed(11)
for vocab, expect_norm in ((125696, True), (64000, False)):
    hidden = 64
    head = torch.randn(vocab, hidden, generator=g)
    head = head / head.norm(dim=1, keepdim=True) * (0.5 + 1.5 * torch.rand(vocab, 1, generator=g))   # row norms spread 0.5..2
    raw_norms = head.norm(dim=1)
    try:
        got = convert(make(vocab), "lm_head.weight", head.clone())
    except Exception as e:
        check(f"vocab {vocab}: convert lm_head", False, f"{type(e).__name__}: {e}")
        continue
    w = got.get("output.weight")
    check(f"vocab {vocab}: output.weight emitted", w is not None, str(list(got)))
    if w is None:
        continue
    n = w.float().norm(dim=1)
    if expect_norm:
        check(f"vocab {vocab}: every row unit norm +-1e-3", float((n - 1).abs().max()) < 1e-3, f"max|n-1|={float((n-1).abs().max()):g}")
        h = torch.randn(3, hidden, generator=g)
        ref = h @ torch.nn.functional.normalize(head, p=2, dim=1).T
        lg = h @ w.float().T
        check(f"vocab {vocab}: logits match HF NormHead reference", float((lg - ref).abs().max()) < 1e-3, f"{float((lg-ref).abs().max()):g}")
        check(f"vocab {vocab}: CONTROL raw rows were NOT unit (test discriminates)", float((raw_norms - 1).abs().max()) > 0.3)
    else:
        check(f"vocab {vocab}: Baichuan-1 head kept raw (negative control)", bool(torch.equal(w.float(), head)))

# non-head tensors untouched for Baichuan2
t = torch.randn(64, 64, generator=g) * 3
got = convert(make(125696), "model.layers.0.mlp.gate_proj.weight", t.clone())
check("Baichuan2 non-head tensor unchanged", len(got) == 1 and bool(torch.equal(list(got.values())[0], t)), str(list(got)))

print("FAILED: " + ", ".join(failures) if failures else "ALL PASS")
sys.exit(1 if failures else 0)
