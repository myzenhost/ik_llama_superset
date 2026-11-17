#!/usr/bin/env python3
# Generate the tiny random-weight deepseek4 GGUF (gen_dsv4_randw.py) and run test-recurrent-state-rollback on it.
# test-llama-archs cannot write a deepseek4 model (its loader infers every shape from file tensor metadata), so the plain
# "-m ${MODEL_DIR}/deepseek4-moe.gguf" registration never had a file to load. Exit 0 pass, 1 fail, 77 skipped (gguf/numpy
# or the test binary missing: a skip is printed, never a pass).
# GATE-CLAIM-OK: test driver, no model-availability claims.
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--variant", default="", help="'41' for the DeepSeek-V4.1 layout")
    a = ap.parse_args()

    exe = os.path.join(a.bin_dir, "test-recurrent-state-rollback")
    if sys.platform == "win32":
        exe += ".exe"
    if not os.path.isfile(exe):
        print("SKIP: %s not built" % exe)
        return 77
    try:
        import numpy  # noqa: F401
        import gguf  # noqa: F401
    except ImportError as e:
        print("SKIP: python module missing (%s); pip install gguf numpy" % e)
        return 77

    os.makedirs(a.work_dir, exist_ok=True)
    model = os.path.join(a.work_dir, "deepseek4-randw%s.gguf" % a.variant)
    gen = [sys.executable, os.path.join(HERE, "gen_dsv4_randw.py"), model, "1", "0.05"]
    if a.variant:
        gen.append(a.variant)
    g = subprocess.run(gen)
    if g.returncode != 0 or not os.path.isfile(model):
        print("FAIL: fixture generation rc %d" % g.returncode)
        return 1
    return subprocess.run([exe, "-m", model]).returncode


if __name__ == "__main__":
    sys.exit(main())
