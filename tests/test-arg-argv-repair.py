#!/usr/bin/env python3
"""Drive test-arg-argv-repair: common_params_parse() must repair non-ASCII argv on Windows AND keep an
argv the caller rewrote one-for-one.

  plain   : -p "<non-ASCII text>" must come back as its exact UTF-8 bytes (the repair still works; on a
            non-UTF-8 ANSI code page the unrepaired argv would hold '?' bytes instead).
  rewrite : --prompt-alias abc, rewritten to -p abc inside the program, must parse and yield "abc".
            Before the fix the count matched, the raw command line was swapped back in, and
            --prompt-alias was refused (PARSE_FAILED / invalid argument).
  control : -p abc with no rewrite must yield "abc".

usage: test-arg-argv-repair.py <bin dir>
"""
import os
import subprocess
import sys

TEXT = "héllo 你好"


def run(exe, args, rewrite):
    env = dict(os.environ)
    env["ARGV_REPAIR_REWRITE"] = "1" if rewrite else "0"
    # -m only satisfies the "--model is required" check of LLAMA_EXAMPLE_COMMON; nothing is loaded.
    p = subprocess.run([exe, "-m", "unused-model.gguf"] + args, capture_output=True, timeout=60, env=env,
                       creationflags=0x08000000 if os.name == "nt" else 0)
    out = (p.stdout + p.stderr).decode("utf-8", "replace")
    got = None
    for line in out.splitlines():
        if line.startswith("PROMPT_HEX="):
            got = bytes.fromhex(line[len("PROMPT_HEX="):].strip())
    return p.returncode, got, out


def main():
    exe = os.path.join(sys.argv[1], "test-arg-argv-repair" + (".exe" if os.name == "nt" else ""))
    ok = True
    arms = (("control", ["-p", "abc"], False, b"abc"),
            ("plain-utf8", ["-p", TEXT], False, TEXT.encode("utf-8")),
            ("rewrite", ["--prompt-alias", "abc"], True, b"abc"))
    for name, args, rewrite, want in arms:  # PROGRESS-OK: three arms
        rc, got, out = run(exe, args, rewrite)
        good = rc == 0 and got == want
        print("%-10s rc=%d got=%r want=%r %s" % (name, rc, got, want, "ok" if good else "BAD"))
        if not good:
            print("  tail: " + " | ".join(out.strip().splitlines()[-3:]))
        ok = ok and good
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
