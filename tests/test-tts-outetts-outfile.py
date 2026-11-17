#!/usr/bin/env python3
"""llama-tts-outetts must accept --out-file, the spelling examples/tts/tts.cpp took for the output WAV.

The built tool parses -o / --output / --output-file through common/arg.cpp; --out-file was dropped when
examples/tts/tts.cpp was ported to tools/tts/tts-outetts.cpp, so it reached common_params_parse and was
refused. Found by cortex/scripts/check_unbuilt_twin_cli_parity.py (option literal present only in the
unbuilt examples/tts copy).

No model is needed: with a prompt, a dummy -m path and no vocoder the tool parses every option and then stops at
"no vocoder provided". Before the fix it stops earlier, in common_params_parse, on --out-file itself.
Control: --output (the built spelling) must reach the vocoder check too, or the test measures nothing.

usage: test-tts-outetts-outfile.py <bin dir>
"""
import os
import subprocess
import sys


def run(exe, flag):
    # -m is required by common_params_parse for LLAMA_EXAMPLE_TTS; the file is never opened, because the
    # vocoder check (tts-outetts.cpp main) returns before any model load.
    p = subprocess.run([exe, "-m", "unused-backbone.gguf", "-p", "hello", flag, "x.wav"], capture_output=True, text=True, timeout=120,
                       creationflags=0x08000000 if os.name == "nt" else 0)
    return p.returncode, p.stdout + p.stderr


def main():
    exe = os.path.join(sys.argv[1], "llama-tts-outetts" + (".exe" if os.name == "nt" else ""))
    ok = True
    for flag in ("--output", "--out-file"):  # PROGRESS-OK: two flags
        rc, out = run(exe, flag)
        reached = "no vocoder provided" in out
        print("%-11s rc=%d reached-vocoder-check=%s" % (flag, rc, reached))
        if not reached:
            print("  tail: " + " | ".join(out.strip().splitlines()[-3:]))
        ok = ok and reached
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
