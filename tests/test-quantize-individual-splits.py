#!/usr/bin/env python3
"""llama-quantize --individual-tensors / --skip-first-shard (individual-shard quantize).

Builds a split model with llama-gguf-split (--no-tensor-first-split, so shard 1 is metadata only, the layout the
GGUF-Tool-Suite consumer uses), quantizes ALL shards once with --keep-split --pure as the reference, then:

  run 1: a directory holding ONLY shards 1 and 3 of the input; --individual-tensors 3
         -> exactly out-00001 and out-00003 are written, each byte-identical to the reference shard
  run 2: a directory holding ONLY shards 1, 2 and 4; --individual-tensors 2,4 --skip-first-shard
         -> exactly out-00002 and out-00004 are written, byte-identical to the reference
  guard: --individual-tensors without --keep-split is refused (exit != 0, no output)
  guard: a shard number past the model's split count is refused (exit != 0)

A build without the option rejects the flag (usage, exit 1), so runs 1 and 2 fail.

usage: test-quantize-individual-splits.py --quantize <llama-quantize> --split <llama-gguf-split> --model <small f32 gguf>
                                          [--workdir PARENT]
Every run uses a NEW directory (under PARENT, or the system temp dir). Exit 0 pass, 1 fail. CPU only.
Model: any small GGUF with >= 4 * 3 tensors (stories260K-f32 is what was run).
"""
import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def run(argv, log):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    with open(log, "wb") as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env, creationflags=CREATE_NO_WINDOW)
    return p.returncode


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def shards(d, prefix):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, prefix + "-*-of-*.gguf")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()

    if a.workdir:
        os.makedirs(a.workdir, exist_ok=True)
    wd = tempfile.mkdtemp(prefix="q-indiv-", dir=a.workdir)
    print("workdir", wd)
    fails = []

    def check(what, ok, detail=""):
        print("  %-4s %s%s" % ("ok" if ok else "FAIL", what, ("  -- " + detail) if detail and not ok else ""))
        if not ok:
            fails.append(what)

    # 1. split the input: shard 1 metadata only, 3 tensors per shard after it
    src = os.path.join(wd, "src")
    os.makedirs(src)
    rc = run([a.split, "--split", "--split-max-tensors", "3", "--no-tensor-first-split", a.model,
              os.path.join(src, "in")], os.path.join(wd, "split.log"))
    in_shards = shards(src, "in")
    check("control: gguf-split made >= 5 shards (rc %d, %d shards)" % (rc, len(in_shards)), rc == 0 and len(in_shards) >= 5)
    if fails:
        print("FAIL (%d failed)" % len(fails))
        return 1
    n = len(in_shards)
    name = lambda pre, i: "%s-%05d-of-%05d.gguf" % (pre, i, n)

    # 2. reference: every shard
    ref = os.path.join(wd, "ref")
    os.makedirs(ref)
    rc = run([a.quantize, "--keep-split", "--pure", os.path.join(src, name("in", 1)), os.path.join(ref, "out.gguf"),
              "Q8_0", "2"], os.path.join(wd, "ref.log"))
    ref_shards = shards(ref, "out")
    check("control: full --keep-split run writes all %d shards (rc %d)" % (n, rc), rc == 0 and len(ref_shards) == n,
          str(ref_shards))

    def partial(label, keep, ids, extra):
        d_in = os.path.join(wd, label + "_in")
        d_out = os.path.join(wd, label + "_out")
        os.makedirs(d_in)
        os.makedirs(d_out)
        for i in keep:
            shutil.copyfile(os.path.join(src, name("in", i)), os.path.join(d_in, name("in", i)))
        argv = [a.quantize, "--keep-split", "--pure", "--individual-tensors", ",".join(str(i) for i in ids)] + extra + [
            os.path.join(d_in, name("in", 1)), os.path.join(d_out, "out.gguf"), "Q8_0", "2"]
        rc = run(argv, os.path.join(wd, label + ".log"))
        return rc, d_out

    # 3. run 1: only shards 1 and 3 exist
    rc, d_out = partial("run1", [1, 3], [3], [])
    got = shards(d_out, "out")
    want = [name("out", 1), name("out", 3)]
    check("run 1: exit 0 with only shards 1,3 of the input on disk (rc %d)" % rc, rc == 0)
    check("run 1: exactly shards 1 and 3 written", got == want, "got %s" % got)
    for s in want:
        if s in got and s in ref_shards:
            check("run 1: %s byte-identical to the full run" % s, sha(os.path.join(d_out, s)) == sha(os.path.join(ref, s)))

    # 4. run 2: shards 1,2,4 exist; skip the metadata shard
    rc, d_out = partial("run2", [1, 2, 4], [2, 4], ["--skip-first-shard"])
    got = shards(d_out, "out")
    want = [name("out", 2), name("out", 4)]
    check("run 2: exit 0 with --skip-first-shard (rc %d)" % rc, rc == 0)
    check("run 2: exactly shards 2 and 4 written (shard 1 skipped)", got == want, "got %s" % got)
    for s in want:
        if s in got and s in ref_shards:
            check("run 2: %s byte-identical to the full run" % s, sha(os.path.join(d_out, s)) == sha(os.path.join(ref, s)))

    # 5. guards
    g = os.path.join(wd, "guard1")
    os.makedirs(g)
    rc = run([a.quantize, "--pure", "--individual-tensors", "3", os.path.join(src, name("in", 1)),
              os.path.join(g, "out.gguf"), "Q8_0", "2"], os.path.join(wd, "guard1.log"))
    check("guard: --individual-tensors without --keep-split refused (rc %d), nothing written" % rc,
          rc != 0 and not os.listdir(g))
    g = os.path.join(wd, "guard2")
    os.makedirs(g)
    rc = run([a.quantize, "--keep-split", "--pure", "--individual-tensors", str(n + 1), os.path.join(src, name("in", 1)),
              os.path.join(g, "out.gguf"), "Q8_0", "2"], os.path.join(wd, "guard2.log"))
    check("guard: shard number %d past the split count refused (rc %d)" % (n + 1, rc), rc != 0)

    print("%s (%d failed)" % ("FAIL" if fails else "PASS", len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
