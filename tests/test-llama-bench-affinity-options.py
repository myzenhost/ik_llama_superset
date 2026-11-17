#!/usr/bin/env python3
"""llama-bench must accept ik's -w/--warmup <0|1>, --cpu-affinity and -cr/--cpu-range.

ik has these in examples/llama-bench, a copy this tree does not build, so the shipped llama-bench
refused them. Found by cortex/scripts/check_unbuilt_twin_cli_parity.py.

No model is needed: a missing -m file fails at model load, after every option is parsed and after the
affinity list is resolved; a parse-time refusal prints the usage text. Control: --no-warmup (the built
spelling) must reach model load too. Negative arm: a malformed range must be refused with its own
message, which shows the range parser is the ported one and not a silent accept.

usage: test-llama-bench-affinity-options.py <bin dir>
"""
import os
import subprocess
import sys

MISSING = "missing-model-for-affinity-test.gguf"


def run(exe, extra):
    p = subprocess.run([exe, "-m", MISSING] + extra, capture_output=True, text=True, timeout=120,
                       creationflags=0x08000000 if os.name == "nt" else 0)
    return p.returncode, p.stdout + p.stderr


def main():
    exe = os.path.join(sys.argv[1], "llama-bench" + (".exe" if os.name == "nt" else ""))
    ok = True
    for extra in (["--no-warmup"], ["-w", "0"], ["--warmup", "1"], ["--cpu-affinity"],
                  ["-cr", "0-3"], ["--cpu-range", "0-1"]):  # PROGRESS-OK: six spellings
        rc, out = run(exe, extra)
        parsed = "usage:" not in out and ("failed to load model" in out or MISSING in out)
        print("accept %-18s rc=%d parsed-to-model-load=%s" % (" ".join(extra), rc, parsed))
        if not parsed:
            print("  tail: " + " | ".join(out.strip().splitlines()[-3:]))
        ok = ok and parsed
    rc, out = run(exe, ["-cr", "not-a-range"])
    good = rc != 0 and "invalid CPU range" in out
    print("refuse %-18s rc=%d message-seen=%s" % ("-cr not-a-range", rc, good))
    ok = ok and good
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
