#!/usr/bin/env python3
"""llama-bench must accept ik's -w / --warmup <0|1> (examples/llama-bench, unbuilt) besides --no-warmup.

Found by cortex/scripts/check_unbuilt_twin_cli_parity.py: the option literal existed only in the unbuilt
examples/llama-bench copy, so the built tool refused `-w 0` from ik-era scripts.

No model is needed: a missing -m file makes llama-bench fail at model load, which happens after every
option is parsed. Refusal at parse time prints the usage text instead. Control: --no-warmup (the built
spelling) must reach model load too, or the test measures nothing.

usage: test-llama-bench-warmup-alias.py <bin dir>
"""
import os
import subprocess
import sys

MISSING = "missing-model-for-alias-test.gguf"


def run(exe, extra):
    p = subprocess.run([exe, "-m", MISSING] + extra, capture_output=True, text=True, timeout=120,
                       creationflags=0x08000000 if os.name == "nt" else 0)
    out = p.stdout + p.stderr
    parsed = "usage:" not in out and ("failed to load model" in out or MISSING in out)
    return p.returncode, parsed, out


def main():
    exe = os.path.join(sys.argv[1], "llama-bench" + (".exe" if os.name == "nt" else ""))
    ok = True
    for extra in (["--no-warmup"], ["-w", "0"], ["--warmup", "1"]):  # PROGRESS-OK: three spellings
        rc, parsed, out = run(exe, extra)
        print("%-14s rc=%d parsed-to-model-load=%s" % (" ".join(extra), rc, parsed))
        if not parsed:
            print("  tail: " + " | ".join(out.strip().splitlines()[-3:]))
        ok = ok and parsed
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
