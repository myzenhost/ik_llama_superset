#!/usr/bin/env python3
"""llama-imatrix --split-on / --verbose-chunks (each delimited piece is its own sequence).

  run 1: 3 pieces of different lengths joined by "### CHAPTER" (plus an empty piece at the start), -c 64 -b 64
         -> exit 0, "-> 3 non-empty piece(s)", one "piece i/3" line per piece, "[3/3]", one Final estimate,
            the imatrix file written
  run 2: same text, -b 16 (pieces end in a SHORTER ubatch) with -lsim and --verbose-chunks
         -> exit 0 (the old lsim buffers exited on a ubatch size change), the three "sorted ... importances"
            reports, "===== chunk 2/3 =====" printed
  run 3: --chunk 1 --chunks 1 -> exactly one piece processed ("[1/1]")
  control: the same text WITHOUT --split-on at -c 128 (ordinary path needs 2*n_ctx = 256 tokens, the text is ~230)
           exits non-zero --
           shows that runs 1-3 passing is the split-on path, not the ordinary one.

A build without the option fails runs 1-3 (unknown argument).

usage: test-imatrix-split-on.py --bin <llama-imatrix> --model <small gguf> [--workdir PARENT]
Every run uses a NEW directory (mkdtemp); nothing outside it is written. Exit 0 pass, 1 fail.
CPU only (-ngl 0, CUDA hidden). stories260K-f32 is what was run.
"""
# REWRITE-SAFE: every file this test writes is inside a fresh mkdtemp directory; no existing store is touched
import argparse
import os
import subprocess
import sys
import tempfile

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
DELIM = "### CHAPTER"
PIECES = [
    "Once upon a time there was a little girl named Lily. She liked to play in the park with her dog. " * 2,
    "Tom had a big red ball. He threw it to his friend and they laughed. ",
    "The sun was hot and the sky was blue. A bird sang in the tree and the cat looked up. " * 3,
]


def run(argv, log):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    with open(log, "wb") as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env, creationflags=CREATE_NO_WINDOW)
    with open(log, "rb") as f:
        return p.returncode, f.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    if a.workdir:
        os.makedirs(a.workdir, exist_ok=True)
    wd = tempfile.mkdtemp(prefix="imx-spliton-", dir=a.workdir)
    print("workdir", wd)
    text = os.path.join(wd, "text.txt")
    with open(text, "w", encoding="utf-8") as f:
        f.write(DELIM + DELIM.join(PIECES))  # leading delimiter -> an empty first piece that must be dropped

    fails = []

    def check(what, ok, detail=""):
        print("  %-4s %s%s" % ("ok" if ok else "FAIL", what, ("  -- " + detail) if detail and not ok else ""))
        if not ok:
            fails.append(what)

    base = [a.bin, "-m", a.model, "-f", text, "-c", "64", "-t", "2", "-ngl", "0"]

    out1 = os.path.join(wd, "run1.gguf")
    rc, log = run(base + ["-b", "64", "--split-on", DELIM, "-o", out1], os.path.join(wd, "run1.log"))
    check("run 1: exit 0 (rc %d)" % rc, rc == 0, log[-400:])
    check("run 1: 3 non-empty pieces (empty first piece dropped)", "-> 3 non-empty piece(s)" in log)
    check("run 1: one 'piece i/3' line per piece", all(("piece %d/3:" % i) in log for i in (1, 2, 3)))
    check("run 1: progress reaches [3/3]", "[3/3]" in log)
    check("run 1: exactly one Final estimate", log.count("Final estimate: PPL =") == 1)
    check("run 1: imatrix file written", os.path.isfile(out1) and os.path.getsize(out1) > 0)

    out2 = os.path.join(wd, "run2.gguf")
    rc, log = run(base + ["-b", "16", "-lsim", "--verbose-chunks", "--split-on", DELIM, "-o", out2],
                  os.path.join(wd, "run2.log"))
    check("run 2: -lsim with short tail ubatches, exit 0 (rc %d)" % rc, rc == 0, log[-400:])
    for rep in ("sorted layer importances", "sorted attention importances", "sorted ffn importances"):
        check("run 2: lsim report '%s'" % rep, rep in log)
    check("run 2: --verbose-chunks printed piece 2", "===== chunk 2/3 =====" in log)
    check("run 2: imatrix file written", os.path.isfile(out2) and os.path.getsize(out2) > 0)

    out3 = os.path.join(wd, "run3.gguf")
    rc, log = run(base + ["-b", "64", "--chunk", "1", "--chunks", "1", "--split-on", DELIM, "-o", out3],
                  os.path.join(wd, "run3.log"))
    check("run 3: --chunk 1 --chunks 1 exit 0 (rc %d)" % rc, rc == 0, log[-400:])
    check("run 3: exactly one piece processed", "[1/1]" in log and "piece 2/" not in log)

    out4 = os.path.join(wd, "control.gguf")
    # ~230 tokens in all: every piece fits -c 128, but the ordinary path needs 2*128 = 256 tokens
    ctl = [x if x != "64" else "128" for x in base]
    rc, log = run(ctl + ["-b", "128", "-o", out4], os.path.join(wd, "control.log"))
    check("control: same text without --split-on at -c 128 is too short and fails (rc %d)" % rc,
          rc != 0 and "you need at least" in log and "non-empty piece" not in log, log[-300:])

    print("%s (%d failed)" % ("FAIL" if fails else "PASS", len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
