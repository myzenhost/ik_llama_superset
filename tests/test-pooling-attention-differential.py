#!/usr/bin/env python3
"""Differential test for `llama-embedding --pooling attention` (LLAMA_POOLING_TYPE_ATTENTION).

What it proves
  The in-graph attention pool (query = masked mean of the sequence's token states, softmax over the
  sequence's own tokens, weighted sum of the token states) equals an independent float64 numpy
  attention pool of the per-token states that the SAME binary prints under `--pooling none`.
  Several prompts are decoded together in ONE batch and each batched row must equal the same prompt
  decoded alone. Measured limit: on real BERT states the scores (q . k, |k| ~ 17) are so large that the
  softmax is one-hot on the sequence's own best token, so a cross-sequence leak changes the answer by
  < 1e-10 (22 prompts, all pairs, explore_leak.py) -- the mask cannot be told from its absence here and
  this test does not claim to. The controls that CAN fail are the mean-pool variant and a planted error.

Fail-before / pass-after
  A binary built from canonical e171b2d6e does not know the value ("invalid value", rc != 0) -> this
  script exits 1 with "FAIL-BEFORE". A binary with the port exits 0 only if every comparison is inside
  tolerance.

Usage
  python tests/test-pooling-attention-differential.py --embedding <llama-embedding[.exe]> --model <bert-style .gguf> [--threads 8]
Any BERT-style (non-causal, embeddings) GGUF works; the checked run used mxbai-embed-large-v1 f16.
CPU only: the binary under test must be a CPU build (the script passes -ngl 0 as well).
"""
import argparse
import re
import subprocess
import sys

import numpy as np

PROMPTS = [
    "The cat sat.",
    "Attention pooling weights each token by how well it agrees with the sequence average.",
    "Regional ferry operators apply the rules of safe docking when a storm warning is issued within twenty minutes of arrival.",
]
SEP = "<#sep#>"
TOL_REL = 5e-4  # relative to the largest |component| of the reference row; float32 graph vs float64 numpy


def run(exe, model, pooling, prompts, threads):
    argv = [exe, "-m", model, "--pooling", pooling, "--embd-normalize", "-1", "--embd-output-format", "json",
            "--embd-separator", SEP, "-p", SEP.join(prompts), "-c", "512", "-t", str(threads), "-ngl", "0"]
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout, p.stderr


def parse_rows(stdout):
    rows = []
    for m in re.finditer(r'"embedding":\s*\[([^\]]*)\]', stdout):
        rows.append(np.array([float(x) for x in m.group(1).split(",")], dtype=np.float64))
    return rows


def softmax_pool(states, query_from, scores_over):
    """query = mean of states[query_from]; weights = softmax(scores_over @ query); out = weights @ scores_over."""
    q = states[query_from].mean(axis=0)
    s = scores_over @ q
    s = s - s.max()
    w = np.exp(s)
    w /= w.sum()
    return w @ scores_over


def cmp(name, got, want):
    scale = float(np.abs(want).max())
    err = float(np.abs(got - want).max())
    cos = float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want)))
    print("  %-38s max|diff|=%.3e (rel %.3e)  cos=%.8f" % (name, err, err / scale, cos))
    return err / scale, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    # 1. the flag value must be known to the binary (the canonical tip does not know it)
    rc, out, err = run(a.embedding, a.model, "attention", PROMPTS[:1], a.threads)
    if rc != 0:
        tail = [ln for ln in (err + out).splitlines() if ln.strip()][-3:]
        print("FAIL-BEFORE: --pooling attention not accepted, rc=%d: %s" % (rc, " | ".join(tail)))
        return 1
    print("--pooling attention accepted, rc=0")

    # 2. per-token states from --pooling none, one prompt per run (single sequence, nothing to leak)
    states = []
    for pr in PROMPTS:
        rc, out, err = run(a.embedding, a.model, "none", [pr], a.threads)
        rows = parse_rows(out)
        if rc != 0 or not rows:
            print("FAIL: --pooling none run failed rc=%d rows=%d" % (rc, len(rows)))
            return 1
        states.append(np.stack(rows))
    print("token counts per prompt:", [s.shape[0] for s in states], " n_embd:", states[0].shape[1])

    expect = [softmax_pool(s, slice(None), s) for s in states]
    mean_pool = [s.mean(axis=0) for s in states]

    # 3. attention pooling, all prompts in ONE batch (multi-sequence)
    rc, out, err = run(a.embedding, a.model, "attention", PROMPTS, a.threads)
    got = parse_rows(out)
    if rc != 0 or len(got) != len(PROMPTS):
        print("FAIL: multi-sequence attention run rc=%d rows=%d" % (rc, len(got)))
        return 1
    ok = True
    print("multi-sequence batch (3 prompts, one decode) vs numpy attention pool of --pooling none states:")
    for i in range(len(PROMPTS)):
        rel, cos = cmp("prompt %d" % i, got[i], expect[i])
        ok &= rel < TOL_REL and cos > 0.999999
    # 4. single-sequence runs must agree with the batched rows (the mask must not change the answer)
    print("single-sequence attention runs vs the batched rows:")
    for i, pr in enumerate(PROMPTS):
        rc, out, err = run(a.embedding, a.model, "attention", [pr], a.threads)
        one = parse_rows(out)
        if rc != 0 or len(one) != 1:
            print("FAIL: single attention run rc=%d rows=%d" % (rc, len(one)))
            return 1
        rel, cos = cmp("prompt %d alone vs batched" % i, one[0], got[i])
        ok &= rel < TOL_REL and cos > 0.999999

    # 5. controls: the test must be able to fail
    print("controls (each must EXCEED tolerance, else the test cannot tell the variants apart):")
    for i in range(len(PROMPTS)):
        rel_m, _ = cmp("mean pool instead, prompt %d" % i, got[i], mean_pool[i])
        if not rel_m > 20 * TOL_REL:
            print("FAIL: control did not separate (mean rel %.2e)" % rel_m)
            ok = False
    planted = expect[0].copy()
    planted[3] += 0.01 * float(np.abs(planted).max())
    rel_p, _ = cmp("planted 1 percent perturbation", got[0], planted)
    if not rel_p > TOL_REL:
        print("FAIL: comparator missed a planted perturbation")
        ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
