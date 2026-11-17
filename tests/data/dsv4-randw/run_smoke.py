#!/usr/bin/env python3
# CPU-only smoke for the tensor-bearing random-weight DSV4 fixtures: generate deepseek4 and deepseek41 GGUFs (gen_dsv4_randw.py), load each with llama-perplexity on
# the CPU (-dev none --no-op-offload) and require a finite perplexity. This is what keeps both archs from being a silent skip in test-llama-archs (their loader infers
# every shape from file metadata, so the weightless harness cannot build them). It is NOT the CUDA-vs-CPU comparison: that needs the GPU and stays in run_differential.bat.
# Exit 0 pass, 1 fail, 77 skipped (gguf/numpy or llama-perplexity missing: a skip is printed, never a pass).
# CONTROL: a model path that does not exist must make the same runner fail, so a pass here cannot come from a runner that accepts anything.
# GATE-CLAIM-OK: test driver; the skip conditions are missing Python modules or a missing binary, each printed with the path that was looked at.
# SINGLE-SHOT-OK: a two-model smoke test that finishes in seconds; there is no pipeline state to pause or resume.
# PROGRESS-OK: each child run prints its own verdict line before the next starts, and every child has a timeout.
import argparse
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHS = (("deepseek4", []), ("deepseek41", ["41"]))
CORPUS = ("the quick brown fox jumps over the lazy dog while seven small wizards quietly box a jug of liquor. " * 40)


def run_ppl(exe, model, text_path, timeout):
    cmd = [exe, "-m", model, "-f", text_path, "-c", "256", "-b", "256", "-ub", "256", "--chunks", "1", "-dev", "none", "--no-op-offload"]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode, r.stdout.decode("utf-8", "replace")


def parse_ppl(out):
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([-+0-9.eE]+|nan|inf)", out, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"\[1\]\s*([-+0-9.eE]+|nan|inf)", out, re.I)  # per-chunk fallback when only one chunk printed
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    try:
        import numpy  # noqa: F401
        import gguf  # noqa: F401
    except ImportError as e:
        print("SKIP: %s (pip install gguf numpy)" % e)
        return 77
    exe = os.path.join(a.bin_dir, "llama-perplexity.exe" if os.name == "nt" else "llama-perplexity")
    if not os.path.isfile(exe):
        print("SKIP: %s not found" % exe)
        return 77
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        text = os.path.join(td, "corpus.txt")
        with open(text, "w", encoding="utf-8") as fh:
            fh.write(CORPUS)
        rc, out = run_ppl(exe, os.path.join(td, "does_not_exist.gguf"), text, a.timeout)
        ok = rc != 0
        print("%s negative control: missing model makes the runner fail (rc=%d)" % ("ok  " if ok else "FAIL", rc))
        fails += 0 if ok else 1
        for arch, extra in ARCHS:
            model = os.path.join(td, arch + ".gguf")
            g = subprocess.run([sys.executable, os.path.join(HERE, "gen_dsv4_randw.py"), model, "1", "0.05"] + extra,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=a.timeout)
            if g.returncode != 0 or not os.path.isfile(model):
                print("FAIL %s: generator rc=%d\n%s" % (arch, g.returncode, g.stdout.decode("utf-8", "replace")[-600:]))
                fails += 1
                continue
            rc, out = run_ppl(exe, model, text, a.timeout)
            ppl = parse_ppl(out) if rc == 0 else None
            ok = rc == 0 and ppl is not None and math.isfinite(ppl) and ppl > 0
            print("%s %s loads and decodes on CPU: rc=%d ppl=%s" % ("ok  " if ok else "FAIL", arch, rc, ppl))
            if not ok:
                print(out[-800:])
                fails += 1
    print("%d failed" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
