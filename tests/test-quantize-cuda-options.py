#!/usr/bin/env python3
"""llama-quantize must accept ik's --slab-size, --cuda-quantize, -dev/--device and -ts/--tensor-split.

ik landed these (#2674, #2682) in examples/quantize, a copy this tree does not build, so the shipped
llama-quantize printed usage for every one of them and the device quantize encoder was unreachable.
Found by cortex/scripts/check_unbuilt_twin_cli_parity.py.

No model and no device are needed: with every option consumed the tool prints "quantizing '<input>'"
and then fails to open the missing input, while a parse-time refusal prints the usage text. Control: --allow-requantize (a spelling the built tool always
had) must reach the load failure too, or the test measures nothing. Two negative arms check that the
ported validation runs: a tensor-split whose length differs from the device count, and a bad device
name, must each be refused with their own message.

usage: test-quantize-cuda-options.py <bin dir>
"""
import os
import subprocess
import sys

MISSING = "missing-model-for-quantize-option-test.gguf"
OUT = "never-written-quantize-option-test.gguf"


def run(exe, extra):
    p = subprocess.run([exe] + extra + [MISSING, OUT, "Q4_0"], capture_output=True, text=True, timeout=120,
                       creationflags=0x08000000 if os.name == "nt" else 0)
    return p.returncode, p.stdout + p.stderr


def main():
    exe = os.path.join(sys.argv[1], "llama-quantize" + (".exe" if os.name == "nt" else ""))
    ok = True
    accept = (["--allow-requantize"], ["--slab-size", "256"], ["--cuda-quantize"], ["-dev", "CUDA0"],
              ["--device", "0"], ["--cuda-quantize", "-dev", "CUDA0,CUDA1", "-ts", "3,1"],
              ["--tensor-split", "1", "--device", "CUDA0"])
    for extra in accept:  # PROGRESS-OK: seven spellings
        rc, out = run(exe, extra)
        # The tool names its input once every option is consumed. Matching the file name alone is not
        # enough: an unknown short option such as -dev ends the option loop and shifts the positionals,
        # so the name then shows up in "invalid ftype '<name>'" instead.
        parsed = "usage:" not in out and ("quantizing '%s'" % MISSING) in out
        print("accept %-48s rc=%d parsed-to-load=%s" % (" ".join(extra), rc, parsed))
        if not parsed:
            print("  tail: " + " | ".join(out.strip().splitlines()[-3:]))
        ok = ok and parsed
    refuse = ((["-dev", "CUDA0", "-ts", "3,1"], "must match device count"),
              (["-dev", "GPUX"], "invalid CUDA device"))
    for extra, want in refuse:  # PROGRESS-OK: two negative arms
        rc, out = run(exe, extra)
        good = rc != 0 and want in out
        print("refuse %-48s rc=%d message-seen=%s" % (" ".join(extra), rc, good))
        if not good:
            print("  tail: " + " | ".join(out.strip().splitlines()[-3:]))
        ok = ok and good
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
