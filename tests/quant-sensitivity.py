#!/usr/bin/env python3
"""quant-sensitivity end to end on a random-weight GGUF (row S69-ENGREV-FERROX-QUANT-SENSITIVITY-TOOL-2026-10-27-01).

  python quant-sensitivity.py <tree-root> <build-bin-dir> [workdir]

Checks on the 2-layer F32 tiny model (14 perturbable blk.*.weight matrices):
  - Q4_0 run: rc 0, every one of the 14 tensors measured, total KL > 0 (a quantised tensor moves the output);
  - the restore check: after every tensor was perturbed and written back the clean logits come back (leak 0.0e+00);
  - Q8_0 total KL is below Q4_0 total KL (the statistic ranks the candidates the way the bit width says it must);
  - --layers 0:1 measures exactly the 7 tensors of block 0 and none of block 1 (range is end exclusive);
  - family rollup lines name attn_v and ffn_down;
  - MUST-FAIL controls: unknown --cand, empty range 1:1, range past the model 0:9, missing model path -> nonzero rc, never a table.
Exit 0 pass, 77 skipped (numpy or binary missing).
"""
import os
import re
import subprocess
import sys
import tempfile

root, bindir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
work = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="quant-sensitivity-")
os.makedirs(work, exist_ok=True)
fails = []


def ok(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        fails.append(msg)


try:
    import numpy  # noqa: F401
except ImportError:
    print("numpy missing (SKIP)")
    sys.exit(77)
exe = os.path.join(bindir, "quant-sensitivity")
if not (os.path.exists(exe) or os.path.exists(exe + ".exe")):
    print("quant-sensitivity missing in " + bindir + " (SKIP)")
    sys.exit(77)

gg = os.path.join(work, "tiny1.gguf")
r = subprocess.run([sys.executable, os.path.join(root, "tests/make_tiny_mx_gguf.py"), root, gg, "1"], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout, r.stderr)
    sys.exit(2)


def run(args):
    p = subprocess.run([exe] + args, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def total_kl(out):
    m = re.search(r"total measured KL ([0-9.eE+-]+) nats across (\d+) tensors", out)
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)


rc, out4 = run(["-m", gg, "--cand", "q4_0", "-t", "2", "--top", "50"])
print(out4)
kl4, n4 = total_kl(out4)
ok(rc == 0, "Q4_0 run rc 0 (got rc %d)" % rc)
ok(n4 == 14, "all 14 blk.*.weight matrices measured (got %s)" % n4)
ok(kl4 is not None and kl4 > 0, "total KL at Q4_0 is positive")
ok("restore-leak KL 0.0e+00" in out4, "restore check: leak is exactly 0 after every tensor was written back")
ok("family attn_v" in out4 and "family ffn_down" in out4, "family rollup names attn_v and ffn_down")

rc, out8 = run(["-m", gg, "--cand", "q8_0", "-t", "2", "--top", "50"])
kl8, n8 = total_kl(out8)
ok(rc == 0 and n8 == 14, "Q8_0 run rc 0, 14 tensors")
ok(kl8 is not None and kl4 is not None and kl8 < kl4, "Q8_0 total KL %s below Q4_0 total KL %s" % (kl8, kl4))

rc, outl = run(["-m", gg, "--layers", "0:1", "-t", "2", "--top", "50"])
_, nl = total_kl(outl)
ok(rc == 0 and nl == 7 and "blk.1." not in outl and "blk.0.attn_v.weight" in outl, "--layers 0:1 measures the 7 tensors of block 0 only (got %s)" % nl)

for args, what in (
    (["-m", gg, "--cand", "q5_k"], "unknown candidate"),
    (["-m", gg, "--layers", "1:1"], "empty layer range"),
    (["-m", gg, "--layers", "0:9"], "layer range past the model"),
    (["-m", os.path.join(work, "nope.gguf")], "missing model"),
):
    rc, out = run(args)
    ok(rc != 0 and "total measured KL" not in out, "MUST-FAIL: %s -> rc %d, no table" % (what, rc))

print("RESULT: %s" % ("PASS" if not fails else "FAIL (%d)" % len(fails)))
sys.exit(1 if fails else 0)
