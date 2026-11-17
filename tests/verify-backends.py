#!/usr/bin/env python3
"""verify-backends end to end on a random-weight GGUF (row S69-ENGREV-FERROX-BACKEND-VERIFY-LAYER-DIVERGENCE-2026-10-27-01).

  python verify-backends.py <tree-root> <build-bin-dir> [workdir]

One child process per backend run (a backend is process-lifetime). CPU is the reference. Here both arms are CPU (no GPU in a unit
test); the same driver takes `-ngl 99` for the candidate on a CUDA build:  verify-backends run -m m.gguf -ngl 99 -o cand.vbk
Checks:
  - ref run, same-seed re-run at another thread count (the measured noise band) and the compare: verdict MATCH or DRIFT, never WRONG;
  - the run recorded K and V for every layer (a side that recorded nothing is an ERROR, not agreement);
  - MUST-FAIL CONTROL: a model with other weights (different seed) compared with the reference gives WRONG (exit 1): the verdict can
    tell two models apart;
  - MUST-FAIL CONTROL 2: an empty record file gives exit 3 (NoCounts), never a pass.
Exit 0 pass, 77 skipped (numpy or binaries missing).
"""
import os
import subprocess
import sys
import tempfile

root, bindir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
work = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="verify-backends-")
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
exe = os.path.join(bindir, "verify-backends")
if not (os.path.exists(exe) or os.path.exists(exe + ".exe")):
    print("verify-backends missing in " + bindir + " (SKIP)")
    sys.exit(77)


def run(args):
    r = subprocess.run([exe] + args, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def gguf(seed):
    p = os.path.join(work, "tiny%d.gguf" % seed)
    r = subprocess.run([sys.executable, os.path.join(root, "tests/make_tiny_mx_gguf.py"), root, p, str(seed)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit(2)
    return p


def vbk(name):
    return os.path.join(work, name)


a, b = gguf(1), gguf(2)
rc, out = run(["run", "-m", a, "-t", "1", "-o", vbk("ref.vbk")])
ok(rc == 0, "ref run rc 0: " + out.strip())
rc, out = run(["run", "-m", a, "-t", "3", "-o", vbk("ref2.vbk")])
ok(rc == 0, "ref re-run at another thread count rc 0")
rc, out = run(["run", "-m", b, "-t", "1", "-o", vbk("other.vbk")])
ok(rc == 0, "other-weights run rc 0")
ref = open(vbk("ref.vbk")).read()
ok("\nK 0 " in ref and "\nK 1 " in ref and "\nV 0 " in ref and "\nV 1 " in ref, "ref recorded K and V for both layers")
toks = ref.split("\ntok ")[1].split("\n")[0].split() if "\ntok " in ref else []
ok(len(toks) == 8, "ref recorded 8 greedy tokens")

rc, out = run(["compare", vbk("ref.vbk"), vbk("ref2.vbk"), "--ref2", vbk("ref2.vbk")])
print(out)
ok(rc == 0 and ("VERDICT: MATCH" in out or "VERDICT: DRIFT" in out), "same model, other thread count: MATCH or DRIFT (rc 0), noise band measured")
ok("MEASURED from reference-vs-reference" in out, "the noise band is reported as measured")

rc, out = run(["compare", vbk("ref.vbk"), vbk("other.vbk")])
print(out)
ok(rc == 1 and "VERDICT: WRONG" in out, "MUST-FAIL: different weights -> WRONG (rc 1)")

with open(vbk("empty.vbk"), "w") as f:
    f.write("VBK 1\nbackend none\nn_prompt 16\ntok\nmargin\n")
rc, out = run(["compare", vbk("ref.vbk"), vbk("empty.vbk")])
ok(rc == 3 and "ERROR" in out, "MUST-FAIL: a side that recorded nothing -> ERROR rc 3, not agreement")

print("RESULT: %s" % ("PASS" if not fails else "FAIL (%d)" % len(fails)))
sys.exit(1 if fails else 0)
