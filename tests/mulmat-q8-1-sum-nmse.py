#!/usr/bin/env python3
"""Gate for the CUDA q8_1 activation-sum fix (row S69-FORK-MULMAT-Q41-CUDA-NMSE-FLAKY-2026-10-19-01).

CUDA's quantize_q8_1 stored ds.y as the sum of the FLOAT activations while the dot products pair it with
sum(qa*qb)*d built from the quantized ones. For q4_1 that made the CUDA-vs-CPU NMSE of MUL_MAT m=16 n=1 k=256 about
4x larger on average (model: mean 5.6e-5 vs 1.3e-5) and pushed ~0.25% of executions over the 5e-4 bound -- the flake.
One run is not enough to see a 0.25% event, so this gate runs the case many times with GGML_TEST_PRINT_ERR=1 (the
error of passing cases is printed too) and checks the MEAN error, which separates the two kernels cleanly.

  python tests/mulmat-q8-1-sum-nmse.py --exe build/bin/test-backend-ops --runs 100 [--max-mean 2.5e-5]

Exit 0 = mean below the bound and no failing case; 1 = above it or a failure; 2 = the run produced no error lines.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

CASE = "type_a={t},type_b=f32,m=16,n=1,k=256,bs=\\[1,1\\]"
ERR_RE = re.compile(r"\[MUL_MAT\] (?:err|ERR) = ([0-9.eE+-]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--backend", default="CUDA0")
    ap.add_argument("--type", default="q4_1")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--max-mean", type=float, default=2.5e-5)
    a = ap.parse_args()
    env = dict(os.environ, GGML_TEST_PRINT_ERR="1")
    errs, fails = [], 0
    for i in range(a.runs):
        p = subprocess.run([a.exe, "test", "-b", a.backend, "-o", "MUL_MAT", "-p", CASE.format(t=a.type)],
                           capture_output=True, text=True, env=env, errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out = p.stdout + p.stderr
        errs += [float(x) for x in ERR_RE.findall(out)]
        fails += out.count("FAIL") and 1
        if (i + 1) % 20 == 0:
            print(f"runs {i + 1}/{a.runs} errors {len(errs)} failing runs {fails}", flush=True)
    if not errs:
        print("NO ERROR LINES: the binary does not print GGML_TEST_PRINT_ERR, or the case did not run")
        return 2
    mean = statistics.fmean(errs)
    print(f"RESULT type {a.type} cases {len(errs)} mean {mean:.3e} median {statistics.median(errs):.3e} "
          f"max {max(errs):.3e} failing_runs {fails} bound {a.max_mean:.1e}")
    return 0 if mean <= a.max_mean and fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
