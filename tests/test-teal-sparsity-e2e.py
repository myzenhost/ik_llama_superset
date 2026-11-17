#!/usr/bin/env python3
"""End-to-end gates for TEAL training-free activation sparsity (S69-TEAL-STAGE1, arXiv:2408.14690).

Model-level counterpart of tests/test-teal-sparsity.cpp. Needs a small GGUF and a calibration /
evaluation text pair that are DISJOINT files (anti-leakage). CPU only, no server, no GPU.

  python tests/test-teal-sparsity-e2e.py --bin-dir <build>/bin --model <small.gguf> \
         --train wiki.train.raw --test wiki.test.raw [--base-bin-dir <pre-TEAL build>/bin] [--quick|--full]

Gates (each prints PASS/FAIL and the measured numbers):
  G0 flag      : `--teal-sparsity` is a recognised option. Run against a PRE-TEAL binary (--bin-dir) this
                 gate FAILS ("invalid argument") -- that is the recorded fail-before.
  G1 identity  : flag OFF => byte-identical `--save-all-logits` dump. Compares (a) this binary with no flag
                 vs `--teal-sparsity 0`, and (b) if --base-bin-dir is given, this binary's default output
                 vs the pre-TEAL baseline binary built from the parent commit (the real control).
  G2 reach     : after `cats-calibrate --teal` + `--verify-thresholds` on the DISJOINT eval text, EVERY layer
                 of EVERY family (attn_in / ffn_in / ffn_mid) reports masked activations, and the achieved
                 sparsity is within +/-ACH_TOL of the target. Run against the first TEAL build (attn_in
                 masking wired only into build_std_attention, dead for qwen2) this gate FAILS with
                 "NO masked activations observed" on all attn_in layers -- the recorded fail-before for the
                 build_qwen2.cpp fix.
  G3 effect    : sparsity ON changes the logits (dense != sparse) and perplexity degrades monotonically
                 with the target (0 < 25 < 40 < 50 %).
  G4 quality   : PPL(25%) and PPL(40%) stay within the PROPOSED relative bounds below (see PPL_BOUNDS).
                 The ledger/findings of the TEAL stage-1 work name no numeric threshold, so these are
                 PROPOSED from the measured full-corpus table, not inherited.

Exit code 0 only if every selected gate passes.
"""
import argparse
import filecmp
import json
import os
import re
import subprocess
import sys
import time

# PROPOSED quality bounds (relative PPL increase vs dense) for qwen2.5-coder-0.5b q8_0 on wikitext-2 test.
# Set from the measured 584-chunk table in temp/research/findings/S69_TEAL_RESUME_2026-10-01.md; the
# stage-1 findings name no threshold. Loosened bound for the quick (few-chunk) run because a handful of
# chunks has far higher variance than the whole corpus.
PPL_BOUNDS = {0.25: 0.10, 0.40: 0.35}
ACH_TOL = 0.03

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def run(argv, timeout=3600):
    t0 = time.time()
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                       creationflags=CREATE_NO_WINDOW)
    out = p.stdout.decode("utf-8", "replace")
    return p.returncode, out, time.time() - t0


def exe(bin_dir, name):
    p = os.path.join(bin_dir, name + (".exe" if os.name == "nt" else ""))
    if not os.path.isfile(p):
        raise SystemExit("missing binary: " + p)
    return p


def ppl_of(out):
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out)
    return float(m.group(1)) if m else None


class Ctx:
    pass


def perplexity_args(c, bin_dir, extra, chunks):
    a = [exe(bin_dir, "llama-perplexity"), "-m", c.model, "-f", c.test, "-c", "512", "-t", str(c.threads)]
    if chunks:
        a += ["--chunks", str(chunks)]
    return a + extra


results = []


def gate(name, ok, detail):
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + "  " + detail, flush=True)
    return ok


def g0_flag(c):
    rc, out, _ = run(perplexity_args(c, c.bin_dir, ["--teal-sparsity", "0"], 1), timeout=600)
    bad = ("invalid argument" in out) or ("error: unknown" in out.lower()) or rc != 0
    return gate("G0 flag --teal-sparsity recognised", not bad,
                "rc=%d%s" % (rc, " (option rejected)" if bad else ""))


def g1_identity(c):
    ok_all = True
    a = os.path.join(c.work, "logits_default.bin")
    b = os.path.join(c.work, "logits_teal0.bin")
    rc1, out1, _ = run(perplexity_args(c, c.bin_dir, ["--save-all-logits", a], 1), timeout=900)
    rc2, out2, _ = run(perplexity_args(c, c.bin_dir, ["--teal-sparsity", "0", "--save-all-logits", b], 1), timeout=900)
    same = rc1 == 0 and rc2 == 0 and os.path.isfile(a) and os.path.isfile(b) and filecmp.cmp(a, b, shallow=False)
    ok_all &= gate("G1a identity: default vs --teal-sparsity 0 (same binary)", same,
                   "ppl %s vs %s, bytes %s" % (ppl_of(out1), ppl_of(out2), os.path.getsize(a) if os.path.isfile(a) else None))
    if c.base_bin_dir:
        z = os.path.join(c.work, "logits_base.bin")
        rc3, out3, _ = run(perplexity_args(c, c.base_bin_dir, ["--save-all-logits", z], 1), timeout=900)
        same3 = rc3 == 0 and os.path.isfile(z) and os.path.isfile(a) and filecmp.cmp(a, z, shallow=False)
        ok_all &= gate("G1b identity: this binary (flag off) vs PRE-TEAL baseline binary", same3,
                       "ppl %s vs %s, bytes %s" % (ppl_of(out1), ppl_of(out3), os.path.getsize(z) if os.path.isfile(z) else None))
    return ok_all


def calibrate(c, target, tag):
    out_json = os.path.join(c.work, "thr_%s.json" % tag)
    rc, out, dt = run([exe(c.bin_dir, "cats-calibrate"), "--teal", "-m", c.model, "-f", c.train, "-o", out_json,
                       "--target-sparsity", str(target), "-n", str(c.calib_tokens), "-t", str(c.threads)], timeout=1800)
    return rc, out, out_json


def verify(c, thr_json, target):
    rc, out, _ = run([exe(c.bin_dir, "cats-calibrate"), "--teal", "-m", c.model, "-f", c.test,
                      "--verify-thresholds", thr_json, "--target-sparsity", str(target),
                      "-n", str(c.verify_tokens), "-t", str(c.threads)], timeout=1800)
    fams = {}
    cur = None
    for line in out.splitlines():
        m = re.search(r"-- family (\S+) --", line)
        if m:
            cur = m.group(1)
            fams[cur] = {"ach": {}, "none": []}
            continue
        if cur is None:
            continue
        m = re.search(r"layer\s+(\d+): achieved=([0-9.]+)", line)
        if m:
            fams[cur]["ach"][int(m.group(1))] = float(m.group(2))
            continue
        m = re.search(r"layer\s+(\d+): NO masked activations", line)
        if m:
            fams[cur]["none"].append(int(m.group(1)))
    return rc, fams, out


def g2_reach(c, targets):
    ok_all = True
    for t in targets:
        tag = "%02d" % int(round(t * 100))
        rc, out, thr = calibrate(c, t, tag)
        if rc != 0 or not os.path.isfile(thr):
            ok_all &= gate("G2 calibrate @%.2f" % t, False, "rc=%d (no threshold file)" % rc)
            continue
        c.thr[t] = thr
        rcv, fams, _ = verify(c, thr, t)
        expect = ["thresholds_attn_in", "thresholds_ffn_in", "thresholds_ffn_mid"]
        for fam in expect:
            f = fams.get(fam)
            if f is None:
                ok_all &= gate("G2 reach @%.2f %s" % (t, fam), False, "family absent from verify output (rc=%d)" % rcv)
                continue
            n_ach = len(f["ach"])
            n_none = len(f["none"])
            vals = list(f["ach"].values())
            worst = max((abs(v - t) for v in vals), default=1.0)
            ok = n_none == 0 and n_ach > 0 and worst <= ACH_TOL
            ok_all &= gate("G2 reach @%.2f %s" % (t, fam), ok,
                           "layers_masked=%d layers_NOT_masked=%d worst|achieved-target|=%.4f (tol %.2f)"
                           % (n_ach, n_none, worst, ACH_TOL))
    return ok_all


def g3_g4_effect_quality(c, targets):
    chunks = c.ppl_chunks
    rc0, out0, _ = run(perplexity_args(c, c.bin_dir, [], chunks), timeout=7200)
    base = ppl_of(out0)
    if base is None:
        return gate("G3 dense baseline PPL", False, "rc=%d no 'Final estimate'" % rc0)
    print("   dense PPL = %.4f (%s chunks)" % (base, chunks or "ALL"), flush=True)
    ppls = {0.0: base}
    ok_all = True
    # effect on raw logits: dense vs the middle target
    dense_logits = os.path.join(c.work, "logits_dense_g3.bin")
    sparse_logits = os.path.join(c.work, "logits_sparse_g3.bin")
    mid = sorted(targets)[len(targets) // 2]
    run(perplexity_args(c, c.bin_dir, ["--save-all-logits", dense_logits], 1), timeout=900)
    run(perplexity_args(c, c.bin_dir, ["--teal-sparsity", str(mid), "--teal-thresholds", c.thr[mid],
                                       "--save-all-logits", sparse_logits], 1), timeout=900)
    differs = (os.path.isfile(dense_logits) and os.path.isfile(sparse_logits)
               and not filecmp.cmp(dense_logits, sparse_logits, shallow=False))
    ok_all &= gate("G3a effect: logits differ dense vs sparse @%.2f" % mid, differs, "")
    for t in sorted(targets):
        rc, out, _ = run(perplexity_args(c, c.bin_dir, ["--teal-sparsity", str(t), "--teal-thresholds", c.thr[t]],
                                         chunks), timeout=7200)
        ppls[t] = ppl_of(out)
        rel = (ppls[t] / base - 1.0) if ppls[t] else None
        print("   TEAL %.2f PPL = %s (rel %s)" % (t, ppls[t], "%+.2f%%" % (rel * 100) if rel is not None else None), flush=True)
    seq = [ppls[k] for k in sorted(ppls)]
    mono = all(x is not None for x in seq) and all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    ok_all &= gate("G3b effect: PPL strictly increases with target", mono, "PPL " + json.dumps({str(k): v for k, v in sorted(ppls.items())}))
    for t, bound in sorted(PPL_BOUNDS.items()):
        if t in ppls and ppls[t]:
            rel = ppls[t] / base - 1.0
            b = bound if c.full else bound * 2.0
            ok_all &= gate("G4 quality @%.2f rel PPL increase <= %.0f%% (PROPOSED)" % (t, b * 100), rel <= b,
                           "measured %+.2f%%" % (rel * 100))
    c.ppls = ppls
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default=os.environ.get("TEAL_BIN_DIR"))
    ap.add_argument("--base-bin-dir", default=os.environ.get("TEAL_BASE_BIN_DIR"))
    ap.add_argument("--model", default=os.environ.get("TEAL_MODEL"))
    ap.add_argument("--train", default=os.environ.get("TEAL_TRAIN"))
    ap.add_argument("--test", default=os.environ.get("TEAL_TEST"))
    ap.add_argument("--work", default=os.environ.get("TEAL_WORKDIR", "teal_e2e_work"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gates", default="G0,G1,G2,G3", help="comma list of G0,G1,G2,G3 (G3 includes G4)")
    ap.add_argument("--full", action="store_true", help="whole eval file for the PPL gates (584 chunks of 512)")
    ap.add_argument("--ppl-chunks", type=int, default=16, help="chunks for the quick PPL gates")
    ap.add_argument("--targets", default="0.25,0.40,0.50")
    a = ap.parse_args()
    for k in ("bin_dir", "model", "train", "test"):
        if not getattr(a, k):
            ap.error("--%s required" % k.replace("_", "-"))
    if os.path.abspath(a.train) == os.path.abspath(a.test):
        ap.error("calibration and evaluation corpus must be DIFFERENT files (anti-leakage)")
    c = Ctx()
    c.bin_dir, c.base_bin_dir, c.model, c.train, c.test = a.bin_dir, a.base_bin_dir, a.model, a.train, a.test
    c.work, c.threads, c.full = a.work, a.threads, a.full
    c.ppl_chunks = None if a.full else a.ppl_chunks
    c.calib_tokens, c.verify_tokens = 4000, 2000
    c.thr, c.ppls = {}, {}
    os.makedirs(c.work, exist_ok=True)
    targets = [float(x) for x in a.targets.split(",")]
    sel = set(a.gates.split(","))
    ok = True
    if "G0" in sel:
        ok &= g0_flag(c)
    if "G1" in sel:
        ok &= g1_identity(c)
    if "G2" in sel or "G3" in sel:
        ok &= g2_reach(c, targets) if "G2" in sel else True
        if "G3" in sel:
            if not c.thr:  # G3 needs thresholds even when G2 was skipped / failed to produce them
                for t in targets:
                    rc, out, thr = calibrate(c, t, "%02d" % int(round(t * 100)))
                    if rc == 0 and os.path.isfile(thr):
                        c.thr[t] = thr
            if len(c.thr) == len(targets):
                ok &= g3_g4_effect_quality(c, targets)
            else:
                ok &= gate("G3 needs thresholds", False, "calibration produced %d/%d files" % (len(c.thr), len(targets)))
    print("\nSUMMARY: %d gates, %d failed -> %s" % (len(results), sum(1 for r in results if not r[1]), "PASS" if ok else "FAIL"))
    with open(os.path.join(c.work, "e2e_result.json"), "w") as f:
        json.dump({"ok": bool(ok), "gates": [{"name": n, "ok": bool(o), "detail": d} for n, o, d in results],
                   "ppls": {str(k): v for k, v in c.ppls.items()}}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
