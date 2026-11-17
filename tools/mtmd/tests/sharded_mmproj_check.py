#!/usr/bin/env python3
"""Sharded (split) mmproj check for clip_model_loader (ik_llama #2421 port). CPU only, tiny model, seconds.

PROGRESS-OK: three short subprocess runs, no long loop.
SAFE-DELETE-OK: the only deletion is of this script's own leftover split shards (mmproj-tiny-*.gguf) in the scratch directory the caller
passes with --out; they are re-created by the split step of the same run and hold nothing else.

Splits an mmproj with llama-gguf-split, runs `llama-mtmd-debug -p encode` on the unsplit file and on the first shard, and compares the
encoder output. Needs the ggml-org tinygemma3 pair (text model + mmproj) and a build that has llama-gguf-split and llama-mtmd-debug.

    python sharded_mmproj_check.py --bin <build>/bin --text tinygemma3-Q8_0.gguf --mmproj mmproj-tinygemma3.gguf --out <scratch>

Exit 0 when the output is IDENTICAL and the comparison is SENSITIVE (a different test image changes it, so equality is not
vacuous); 1 otherwise; 2 when a step could not run.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(cmd, out_path):
    with open(out_path, "wb") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    return p.returncode


def normalised(path):
    """Log lines with the run-to-run noise removed (timestamps, millisecond figures, the mmproj file name)."""
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = re.sub(r"^\d+\.\d+\.\d+\.\d+ ", "", ln.rstrip("\n"))
            ln = re.sub(r"mmproj-tiny-\d+-of-\d+|mmproj-tinygemma3", "mmproj", ln)
            ln = re.sub(r"[0-9.]+ ms", "N ms", ln)
            lines.append(ln)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for old in glob.glob(os.path.join(a.out, "mmproj-tiny-*.gguf")):
        os.remove(old)
    exe = lambda n: os.path.join(a.bin, n + (".exe" if os.name == "nt" else ""))

    if run([exe("llama-gguf-split"), "--split-max-tensors", "12", a.mmproj, os.path.join(a.out, "mmproj-tiny")], os.path.join(a.out, "split.txt")) != 0:
        print("split failed (see split.txt)")
        return 2
    firsts = sorted(glob.glob(os.path.join(a.out, "mmproj-tiny-00001-of-*.gguf")))
    if not firsts:
        print("split produced no first shard")
        return 2
    print("first shard:", firsts[0])

    def encode(mmproj, image, tag):
        path = os.path.join(a.out, "enc_%s.txt" % tag)
        rc = run([exe("llama-mtmd-debug"), "-m", a.text, "--mmproj", mmproj, "-p", "encode", "--image", image, "-n", "64", "-ngl", "0", "-t", "2"], path)
        return rc, normalised(path)

    rc1, unsplit = encode(a.mmproj, "rainbow", "unsplit")
    rc2, split = encode(firsts[0], "rainbow", "split")
    rc3, other = encode(a.mmproj, "cb", "other")
    if rc1 or rc2 or rc3:
        print("a run failed: unsplit rc=%s split rc=%s other rc=%s" % (rc1, rc2, rc3))
        return 2
    identical = unsplit == split
    sensitive = unsplit != other
    print("IDENTICAL" if identical else "DIFFERENT", "(%d lines compared)" % len(unsplit))
    print("SENSITIVE" if sensitive else "INSENSITIVE", "(a different image changes the output)")
    return 0 if identical and sensitive else 1


if __name__ == "__main__":
    sys.exit(main())
