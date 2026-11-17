#!/usr/bin/env python3
# End-to-end check of llama-imatrix --hessian-block on a tiny random-weight llama GGUF (no download, CPU).
# Row S69-F1-PICK-IMATRIX-BLOCK-HESSIAN-OUTPUT-2026-10-06-01, test (2).
#
#   python imatrix-hessian-e2e.py --tree <fork root> --exe <llama-imatrix> --work <dir> --mode before|after
#
# --mode before (--exe = a build WITHOUT the feature): the option must be REJECTED; also writes the default imatrix
#   to <work>/imatrix-baseline.gguf for the byte-identity check of the after run. Exit 0 when the option is rejected.
# --mode after: (a) without the flag the imatrix is byte-identical to <work>/imatrix-baseline.gguf (when present);
#   (b) with --hessian-block 24 (partial blocks: K=32 -> 24+8, K=64 -> 24+24+16) and 128 the imatrix is STILL
#   byte-identical, the sidecar is written, every tensor is F32 [N, K, 1] with I64 counts equal to the imatrix
#   counts, every block is symmetric to 1e-6 and PSD (smallest eigenvalue >= -1e-4 * largest), and the block
#   diagonal equals the imatrix in_sum2 to 1e-5 relative; (c) a budget overrun is refused (--hessian-max-mib 1
#   with --hessian-block 1024) and so is combine-only mode.
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--exe", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--mode", choices=["before", "after"], required=True)
ap.add_argument("--threads", default="4")
a = ap.parse_args()

tree, work = Path(a.tree), Path(a.work)
work.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(tree / "gguf-py"))
import gguf  # noqa: E402

model = work / "tiny-llama-bytes.gguf"
text = work / "calib.txt"
if not model.exists():
    subprocess.run([sys.executable, str(tree / "tests" / "make_tiny_llama_gguf.py"), str(tree), str(model), "2048", "bytes"],
                   check=True)
if not text.exists():
    rng = np.random.default_rng(4242)
    words = ["the", "court", "held", "that", "a", "matrix", "of", "inputs", "is", "summed", "over", "tokens", "and",
             "blocks", "are", "kept", "for", "every", "weight", "quantizer", "reads", "them", "later"]
    text.write_text(" ".join(words[i] for i in rng.integers(0, len(words), 1200)) + "\n", encoding="utf-8")

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def run(out, *extra):
    cmd = [a.exe, "-m", str(model), "-f", str(text), "-o", str(out), "-c", "128", "-b", "128", "-t", a.threads,
           "--no-ppl", "--chunks", "8", *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    log = work / (Path(out).name + ".log")
    log.write_text(" ".join(cmd) + "\n--- rc=%d\n" % p.returncode + p.stdout + "\n--- stderr\n" + p.stderr, encoding="utf-8")
    return p.returncode, p.stdout + p.stderr


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


baseline = work / "imatrix-baseline.gguf"

if a.mode == "before":
    rc, out = run(baseline)
    check(rc == 0 and baseline.exists(), "default imatrix run (baseline) rc=%d" % rc)
    rc, out = run(work / "imatrix-before-flag.gguf", "--hessian-block", "24")
    check(rc != 0, "--hessian-block 24 is REJECTED by the binary without the feature (rc=%d)" % rc)
    check("hessian-block" in out, "the error names the unknown option")
    if baseline.exists():
        print("baseline sha256 %s" % sha(baseline))
    print("RESULT %s" % ("PASS (option rejected = fail-before state)" if not fails else "FAIL: " + "; ".join(fails)))
    sys.exit(1 if fails else 0)


def tensors(path):
    r = gguf.GGUFReader(str(path))
    return r, {t.name: t for t in r.tensors}


# (a) default output unchanged
off = work / "imatrix-after-off.gguf"
rc, _ = run(off)
check(rc == 0 and off.exists(), "default run rc=%d" % rc)
if baseline.exists():
    check(sha(off) == sha(baseline), "default output byte-identical to the pre-change binary (sha256 %s)" % sha(off)[:16])
else:
    print("note: no baseline from a pre-change binary in %s; byte identity checked only flag-on vs flag-off" % work)

for N in (24, 128):
    on = work / ("imatrix-after-on-%d.gguf" % N)
    side = work / ("imatrix-after-on-%d.hessian.gguf" % N)
    t0 = time.time()
    rc, out = run(on, "--hessian-block", str(N))
    check(rc == 0, "--hessian-block %d run rc=%d" % (N, rc))
    check(on.exists() and sha(on) == sha(off), "N=%d: imatrix output byte-identical with the flag on" % N)
    fresh = side.exists() and side.stat().st_mtime >= t0 - 2
    check(fresh, "N=%d: sidecar %s written by this run (default name)" % (N, side.name))
    if not fresh:
        continue
    hr, ht = tensors(side)
    ir, it = tensors(on)
    f = hr.fields
    check(bytes(f["general.type"].parts[-1]).decode() == "imatrix_hessian", "N=%d: general.type" % N)
    check(int(f["imatrix_hessian.version"].parts[-1][0]) == 1, "N=%d: version 1" % N)
    check(int(f["imatrix_hessian.block_size"].parts[-1][0]) == N, "N=%d: block_size key" % N)
    names = sorted(n[: -len(".in_sum2")] for n in it if n.endswith(".in_sum2"))
    hnames = sorted(n[: -len(".hess_blocks")] for n in ht if n.endswith(".hess_blocks"))
    check(names == hnames and len(names) > 0, "N=%d: one Hessian per imatrix entry (%d entries)" % (N, len(names)))
    worst_sym, worst_psd, worst_diag = 0.0, 0.0, 0.0
    shapes_ok, counts_ok, tails_ok = True, True, True
    for name in hnames:
        hb = ht[name + ".hess_blocks"]
        hc = ht[name + ".hess_counts"]
        s2 = it[name + ".in_sum2"]
        ic = it[name + ".counts"]
        K = int(s2.shape[0])
        # gguf-py reports shape in ggml order (ne0 first)
        if not (hb.tensor_type == gguf.GGMLQuantizationType.F32 and [int(v) for v in hb.shape[:2]] == [N, K]
                and hc.tensor_type == gguf.GGMLQuantizationType.I64):
            shapes_ok = False
            print("  shape %s: %s %s" % (name, list(hb.shape), hb.tensor_type))
            continue
        H = np.asarray(hb.data, dtype=np.float64).reshape(K, N)
        if int(np.asarray(hc.data).reshape(-1)[0]) != int(np.asarray(ic.data).reshape(-1)[0]):
            counts_ok = False
        diag_in = np.asarray(s2.data, dtype=np.float64).reshape(-1)
        for b0 in range(0, K, N):
            nb = min(N, K - b0)
            B = H[b0:b0 + nb, :nb]
            worst_sym = max(worst_sym, float(np.max(np.abs(B - B.T))))
            ev = np.linalg.eigvalsh(B)
            mx = float(np.max(np.abs(ev)))
            if mx > 0:
                worst_psd = min(worst_psd, float(ev.min()) / mx)
            if nb < N and not np.all(H[b0:b0 + nb, nb:] == 0):
                tails_ok = False
            d = np.diag(B)
            ref = diag_in[b0:b0 + nb]
            worst_diag = max(worst_diag, float(np.max(np.abs(d - ref) / np.maximum(1.0, np.abs(ref)))))
    print("  N=%d: %d tensors, shapes e.g. %s" % (N, len(hnames), [int(v) for v in ht[hnames[0] + ".hess_blocks"].shape]))
    check(shapes_ok, "N=%d: every hess_blocks is F32 [%d, K] and hess_counts is I64" % (N, N))
    check(counts_ok, "N=%d: hess_counts == imatrix counts" % N)
    check(tails_ok, "N=%d: partial-block tails are zero" % N)
    check(worst_sym <= 1e-6, "N=%d: blocks symmetric (max |B - B^T| = %.3g)" % (N, worst_sym))
    check(worst_psd >= -1e-4, "N=%d: blocks PSD (worst min/max eigenvalue %.3g)" % (N, worst_psd))
    check(worst_diag <= 1e-5, "N=%d: block diagonal == imatrix in_sum2 (worst rel diff %.3g)" % (N, worst_diag))

# (c) refusals
rc, out = run(work / "imatrix-after-overbudget.gguf", "--hessian-block", "1024", "--hessian-max-mib", "1")
check(rc != 0 and "hessian-max-mib" in out, "budget overrun refused (rc=%d)" % rc)
p = subprocess.run([a.exe, "-m", str(model), "--in-file", str(off), "-o", str(work / "combined.gguf"), "--hessian-block", "24"],
                   capture_output=True, text=True, errors="replace")
check(p.returncode != 0 and "hessian" in (p.stdout + p.stderr), "combine-only mode with --hessian-block refused (rc=%d)" % p.returncode)

print("RESULT %s" % ("PASS" if not fails else "FAIL: " + "; ".join(fails)))
sys.exit(1 if fails else 0)
