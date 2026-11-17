#!/usr/bin/env python3
"""Budgeted mixed-precision allocator for llama-quantize: pick one quantization type per tensor so that the model fits a size budget while the
imatrix-weighted reconstruction error is as small as possible.

INVENT-LADDER-OK: this file IS the built capability (stage 1 of the allocator row); the words "cannot dequantize" below are a per-type runtime
skip report for ik-only types that gguf-py has no decoder for, not a block on the work.

Row S69-EXLLAMAV2-PICK-BUDGETED-MIXED-QUANT-ALLOCATOR-2026-09-29-01. The idea is the "measure, then optimise" split of ExLlamaV2's converter
(MIT): first MEASURE every candidate for every block once, then SOLVE a multiple-choice knapsack against the bit budget.
SOURCE-READ: upstream/exllamav2/exllamav2/conversion/optimize.py:1-188 and exllamav2/conversion/measure.py:100-175 (read in full this
session; MIT; no code copied, rewritten from the behaviour below).
SOURCE-NUANCE: (1) the error each candidate carries is the mean RELATIVE FROBENIUS norm of the block output difference, 1 - accuracy, floored at
1e-6 (measure.py test_error), i.e. a non-squared relative error, not an absolute one; (2) the objective is a p-NORM of those relative errors with p
swept over [1.5, 3.5] and the p kept is the one whose best solution has the smallest MAX error (optimize.py:18,115-145), so the optimiser favours a
balanced allocation over the lowest total; (3) the budget counts only the decoder-layer weights (numel * bpw, optimize.py:71-78), embeddings and
head are outside it; (4) the leftover budget is spent by a fixed-order pass upgrading each slot one step (optimize.py:149-158), here replaced by a
gain-ordered pass; (5) the candidates per block are a hand-ordered ladder, not the cartesian product.

This is stage 1 of the row's design: the measurement is a diagonal output-error PROXY from the REAL ggml quantizer (llama-quantize --pure on the
same source, so every type uses its own scale search, imatrix handling and repack rules) and the REAL stored size (n_bytes of the tensor in the
produced file):

    err(tensor, type) = sum_j  w_j * sum_rows (W[r, j] - Wq[r, j])^2        (w_j = imatrix[j], or 1 without an imatrix)
    rel(tensor, type) = sqrt(err / sum_j w_j * sum_rows W[r, j]^2)          (the weight-space analogue of EXL2's relative Frobenius error)

No forward pass is run; stage 2 (block-output measurement through the engine) is only worth building if this loses to a uniform recipe of the
same size.

    measure   run llama-quantize once per candidate type, read every produced tensor back, write measure.json
    solve     pick the types for a budget, write a --tensor-type-file for llama-quantize and a report
    selftest  pure-python checks of the solver (no model, no quantizer)

Typical use:
    quant_allocate.py measure --src model-Q8_0.gguf --quantize llama-quantize.exe --types q2_k,q3_k,q4_k,q5_k,q6_k,iq4_nl,iq3_s \\
                              --imatrix imatrix.dat --workdir work/ --out measure.json
    quant_allocate.py solve --measure measure.json --like q4_k --out types.txt
    llama-quantize --allow-requantize --imatrix imatrix.dat --tensor-type-file types.txt model-Q8_0.gguf out.gguf q4_k

Only 2-D-or-larger "blk.*" weights are allocated; everything else (embeddings, output, norms, biases) keeps what llama-quantize would give it for
the base type.
"""
# REWRITE-SAFE: the only writes are NEW output files named by the caller (--out, --out.report.json, the work dir); no existing store is read-modify-written.
import argparse
import heapq
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _import_gguf():
    # the fork's own gguf-py sits two levels up from tools/quantize
    cand = os.path.normpath(os.path.join(HERE, "..", "..", "gguf-py"))
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)
    import gguf  # noqa: F401
    return gguf


def load_imatrix(path):
    """llama-imatrix .dat (legacy binary) or imatrix .gguf -> {tensor name: float32 vector}. Returns {} for path None."""
    import numpy as np
    if not path:
        return {}
    if path.lower().endswith(".gguf"):
        gguf = _import_gguf()
        r = gguf.GGUFReader(path)
        by = {t.name: t for t in r.tensors}
        out = {}
        for name, t in by.items():
            if name.endswith(".in_sum2"):
                nm = name[: -len(".in_sum2")]
                sums = np.asarray(t.data, dtype=np.float32).reshape(-1)
                cnt = by.get(nm + ".counts")
                c = np.asarray(cnt.data, dtype=np.float32).reshape(-1) if cnt is not None else None
                out[nm] = sums / c[0] if c is not None and c.size and c[0] > 0 else sums
        return out
    out = {}
    with open(path, "rb") as f:
        n_entries = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        for _ in range(n_entries):
            ln = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            name = f.read(ln).decode("utf-8")
            ncall = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            nval = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            vals = np.frombuffer(f.read(4 * nval), dtype=np.float32).copy()
            out[name] = vals / ncall if ncall > 0 else vals
    return out


def imatrix_for(imat, name, ncols):
    """Per-column weights for one tensor (None when the imatrix has nothing usable). 3-D expert tensors keep one vector per expert."""
    v = imat.get(name)
    if v is None:
        return None
    if v.size == ncols:
        return v
    if v.size % ncols == 0:
        return v.reshape(-1, ncols)
    return None


def weighted_err(w, wq, imv):
    """sum_j imv_j * sum_rows (w - wq)^2 ; imv None -> unweighted."""
    import numpy as np
    d = (w.astype(np.float64) - wq.astype(np.float64)) ** 2
    if imv is None:
        return float(d.sum())
    cols = w.shape[-1]
    if imv.ndim == 1:
        return float((d.reshape(-1, cols) * imv.astype(np.float64)).sum())
    ne = imv.shape[0]
    return float((d.reshape(ne, -1, cols) * imv.astype(np.float64)[:, None, :]).sum())


def cmd_measure(a):
    import numpy as np
    gguf = _import_gguf()
    from gguf.quants import dequantize
    os.makedirs(a.workdir, exist_ok=True)
    imat = load_imatrix(a.imatrix)
    src = gguf.GGUFReader(a.src)
    allocate = re.compile(a.allocate)
    src_w = {t.name: t for t in src.tensors if len(t.shape) >= 2 and allocate.search(t.name)}
    print(f"[measure] {len(src_w)} allocatable tensors of {len(src.tensors)}; imatrix entries {len(imat)}", flush=True)
    result = {"src": a.src, "imatrix": a.imatrix, "allocate": a.allocate, "tensors": {}, "skipped": {}, "fixed_bytes": {}}
    for name, t in src_w.items():
        result["tensors"][name] = {"numel": int(t.n_elements), "shape": [int(x) for x in t.shape], "cand": {}}
    for ty in [x.strip().lower() for x in a.types.split(",") if x.strip()]:
        out = os.path.join(a.workdir, f"meas_{ty}.gguf")
        if not (a.reuse and os.path.exists(out)):
            cmd = [a.quantize, "--allow-requantize", "--pure"]
            if a.imatrix:
                cmd += ["--imatrix", a.imatrix]
            cmd += [a.src, out, ty, str(a.threads)]
            print("[measure] $", " ".join(cmd), flush=True)
            logp = os.path.join(a.workdir, f"meas_{ty}.log")   # the quantizer's own output streams here; tail it to watch a long run
            with open(logp, "w", encoding="utf-8") as lf:
                p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, creationflags=0x08000000 if os.name == "nt" else 0)
            if p.returncode != 0 or not os.path.exists(out):
                with open(logp, encoding="utf-8", errors="replace") as lf:
                    tail = lf.read()[-300:]
                result["skipped"][ty] = f"llama-quantize rc={p.returncode}: {tail}"
                print("[measure] SKIP", ty, result["skipped"][ty], flush=True)
                continue
        rd = gguf.GGUFReader(out)
        fixed = 0
        n_ok = 0
        unreadable = None
        for q in rd.tensors:
            if q.name not in src_w:
                fixed += int(q.n_bytes)
                continue
            if q.tensor_type.name.lower() != ty:
                # the quantizer fell back (row length not a multiple of the block): this type is not a candidate for this tensor
                result["tensors"][q.name]["cand"].setdefault("_fallback", {})[ty] = q.tensor_type.name
                continue
            s = src_w[q.name]
            try:
                wq = dequantize(q.data, q.tensor_type)
                w = np.asarray(s.data, dtype=np.float32) if s.tensor_type.name in ("F32", "F16", "BF16") else dequantize(s.data, s.tensor_type)
            except Exception as e:  # a type gguf-py has no decoder for
                unreadable = f"gguf-py cannot dequantize {q.tensor_type.name}: {type(e).__name__}: {e}"
                break
            w = w.reshape(wq.shape)
            imv = imatrix_for(imat, q.name, w.shape[-1])
            result["tensors"][q.name]["cand"][ty] = {"bytes": int(q.n_bytes), "err": weighted_err(w, wq, imv),
                                                     "energy": weighted_err(w, np.zeros_like(w), imv)}
            n_ok += 1
        if unreadable:
            for tn in result["tensors"].values():
                tn["cand"].pop(ty, None)
            result["skipped"][ty] = unreadable
            print("[measure] SKIP", ty, unreadable, flush=True)
            continue
        result["fixed_bytes"][ty] = fixed
        print(f"[measure] {ty}: {n_ok} tensors measured, non-allocated bytes in that file {fixed}", flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[measure] wrote {a.out}", flush=True)


# ---------------------------------------------------------------- solver (pure python, no numpy) ----------------------------------------------
def lower_hull(points):
    """points: [(bytes, cost, key)] -> the lower convex hull in (bytes, cost), cheapest first, strictly decreasing cost."""
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    best = []
    for p in pts:                      # drop dominated points (more bytes, no less cost)
        if best and p[1] >= best[-1][1]:
            continue
        best.append(p)
    hull = []
    for p in best:
        while len(hull) >= 2:
            (b0, e0, _), (b1, e1, _) = hull[-2], hull[-1]
            # keep hull[-1] only if it lies strictly below the segment hull[-2] -> p
            if (e1 - e0) * (p[0] - b0) >= (p[1] - e0) * (b1 - b0):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def solve(tensors, budget_bytes):
    """tensors: {name: {type: (bytes, cost)}}. Greedy marginal analysis on the convex hulls (the standard near-optimal rule for a multiple-choice
    knapsack), then passes that spend the leftover on the best single upgrade that still fits. Returns ({name: type}, total_bytes, total_cost)."""
    names = sorted(tensors)
    hulls = {n: lower_hull([(b, e, t) for t, (b, e) in tensors[n].items()]) for n in names}
    cur = {n: hulls[n][0] for n in names}              # chosen (bytes, cost, type) per tensor
    pos = {n: 0 for n in names}
    total = sum(cur[n][0] for n in names)
    if total > budget_bytes:
        raise SystemExit(f"budget {budget_bytes} B is below the cheapest allocation {total} B")
    heap = []

    def push(n):
        i = pos[n]
        if i + 1 < len(hulls[n]):
            (b0, e0, _), (b1, e1, _) = hulls[n][i], hulls[n][i + 1]
            heapq.heappush(heap, (-(e0 - e1) / max(1, b1 - b0), n, i))

    for n in names:
        push(n)
    while heap:
        _, n, i = heapq.heappop(heap)
        if pos[n] != i:
            continue
        db = hulls[n][i + 1][0] - hulls[n][i][0]
        if total + db > budget_bytes:
            continue                      # this upgrade does not fit; a cheaper one elsewhere may
        total += db
        pos[n] = i + 1
        cur[n] = hulls[n][i + 1]
        push(n)
    # leftover: the best single upgrade over ALL candidates (not only hull points) that fits, until none does
    while True:
        best = None
        for n in names:
            cb, ce, _ = cur[n]
            for t, (b, e) in tensors[n].items():
                if e < ce and b >= cb and total + (b - cb) <= budget_bytes:
                    gain = (ce - e) / max(1, b - cb)
                    if best is None or gain > best[0]:
                        best = (gain, n, t, b, e)
        if best is None:
            break
        _, n, t, b, e = best
        total += b - cur[n][0]
        cur[n] = (b, e, t)
    return {n: cur[n][2] for n in names}, total, sum(cur[n][1] for n in names)


REL_FLOOR = 1e-6      # EXL2 floors its accuracy so the error never reaches 0


def rel_error(err, energy):
    return max(REL_FLOOR, math.sqrt(err / energy)) if energy > 0 else 1.0


def build_costs(m, objective, p):
    """Per tensor/type (bytes, cost). objective 'abs': cost = err (sum of the weighted squared error, p ignored). objective 'rel': cost =
    rel^p, so minimising the sum minimises the p-norm of the relative errors (a monotone transform of the norm, hence the same optimum)."""
    out = {}
    for n, t in m["tensors"].items():
        c = {}
        for ty, v in t["cand"].items():
            if ty.startswith("_"):
                continue
            c[ty] = (v["bytes"], v["err"] if objective == "abs" else rel_error(v["err"], v["energy"]) ** p)
        if c:
            out[n] = c
    return out


def rel_stats(m, choice):
    rels = [rel_error(m["tensors"][n]["cand"][ty]["err"], m["tensors"][n]["cand"][ty]["energy"]) for n, ty in choice.items()]
    return max(rels), sum(math.log(r) for r in rels)


def solve_objective(m, budget, objective, pnorm):
    """-> (choice, bytes, p used). pnorm 'auto' sweeps p over [1.5, 3.5] and keeps the p whose solution has the smallest max relative error."""
    if objective == "abs":
        ch, tb, _ = solve(build_costs(m, "abs", 1), budget)
        return ch, tb, None
    ps = [1.5 + 0.25 * i for i in range(9)] if pnorm == "auto" else [float(pnorm)]
    best = None
    for p in ps:
        ch, tb, _ = solve(build_costs(m, "rel", p), budget)
        mx, _lg = rel_stats(m, ch)
        if best is None or mx < best[0]:
            best = (mx, ch, tb, p)
    return best[1], best[2], best[3]


def cmd_solve(a):
    with open(a.measure) as f:
        m = json.load(f)
    base = build_costs(m, "abs", 1)
    missing = [n for n in m["tensors"] if n not in base]
    if missing:
        print(f"[solve] {len(missing)} tensors have no usable candidate and are left to llama-quantize: {missing[:4]}...")
    types = sorted({ty for c in base.values() for ty in c})
    if a.like:
        like = a.like.lower()
        # what a PURE --like run stores per tensor: the type itself, or the type the quantizer fell back to (row length not a multiple of the block)
        budget = 0
        n_fb = 0
        for n, c in base.items():
            if like in c:
                budget += c[like][0]
                continue
            fb = str(m["tensors"][n]["cand"].get("_fallback", {}).get(like, "")).lower()
            if fb not in c:
                raise SystemExit(f"tensor {n}: --like {like} neither measured nor has a measured fallback type ({fb or 'none recorded'})")
            budget += c[fb][0]
            n_fb += 1
        why = f"size of a uniform {like} run over the allocated tensors ({n_fb} tensors fall back to another type there)"
    elif a.budget_bpw:
        numel = sum(m["tensors"][n]["numel"] for n in base)
        budget = int(a.budget_bpw * numel / 8)
        why = f"{a.budget_bpw} bpw over {numel} allocated weights"
    else:
        budget = int(a.budget_mib * 1048576)
        why = f"{a.budget_mib} MiB"
    sub = {"tensors": {n: m["tensors"][n] for n in base}}
    choice, tot_b, p_used = solve_objective(sub, budget, a.objective, a.pnorm)
    mx, lg = rel_stats(sub, choice)
    uni = {}
    for ty in types:
        if all(ty in c for c in base.values()):
            uch = {n: ty for n in base}
            umx, ulg = rel_stats(sub, uch)
            uni[ty] = {"bytes": sum(c[ty][0] for c in base.values()), "abs_err": sum(c[ty][1] for c in base.values()), "max_rel": umx, "sum_log_rel": ulg}
    hist = {}
    for ty in choice.values():
        hist[ty] = hist.get(ty, 0) + 1
    rep = {"budget_bytes": budget, "budget_basis": why, "objective": a.objective, "pnorm_used": p_used, "allocated_bytes": tot_b,
           "allocated_abs_err": sum(base[n][ty][1] for n, ty in choice.items()), "max_rel": mx, "sum_log_rel": lg, "uniform": uni, "type_histogram": hist}
    with open(a.out, "w") as f:
        for n in sorted(choice):
            f.write(f"{n}={choice[n]}\n")
    with open(a.out + ".report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[solve] budget {budget} B ({why}); objective {a.objective} p={p_used}; allocated {tot_b} B ({100.0 * tot_b / budget:.2f}% of budget)")
    print(f"[solve]   allocated  max_rel {mx:.5f}  sum_log_rel {lg:.3f}  abs_err {rep['allocated_abs_err']:.6g}")
    for ty, u in sorted(uni.items(), key=lambda kv: kv[1]["bytes"]):
        print(f"[solve]   uniform {ty:8s} {u['bytes']:>14d} B  max_rel {u['max_rel']:.5f}  sum_log_rel {u['sum_log_rel']:.3f}  abs_err {u['abs_err']:.6g}")
    print(f"[solve] type histogram {hist}; wrote {a.out}")


def cmd_selftest(_):
    import random
    rnd = random.Random(7)
    fails = 0

    def check(ok, msg):
        nonlocal fails
        print(("ok:   " if ok else "FAIL: ") + msg)
        fails += 0 if ok else 1

    # synthetic tensors with different sensitivity: err = s * numel * 2^(-2 bits), bytes = numel * bits / 8
    bits = {"q2": 2.6, "q3": 3.4, "q4": 4.5, "q5": 5.5, "q6": 6.6}
    tens = {}
    for i in range(40):
        numel = rnd.choice([1_000_000, 4_000_000])
        s = rnd.choice([1.0, 1.0, 5.0, 40.0])
        tens[f"t{i}"] = {ty: (int(numel * b / 8), s * numel * 2.0 ** (-2 * b)) for ty, b in bits.items()}
    uni_b = sum(c["q4"][0] for c in tens.values())
    uni_e = sum(c["q4"][1] for c in tens.values())
    choice, tb, te = solve(tens, uni_b)
    check(tb <= uni_b, f"allocation fits the budget ({tb} <= {uni_b})")
    check(te < uni_e, f"allocation beats uniform q4 at the same size (cost {te:.4g} < {uni_e:.4g})")
    check(len(set(choice.values())) > 1, f"a heterogeneous allocation is chosen ({sorted(set(choice.values()))})")
    # exact oracle: bytes in this synthetic instance are multiples of 12500, so a DP over 12500-byte bins is EXACT (no rounding); the greedy must be
    # within 2 percent of it on the full 40-tensor instance
    binsz = 12500
    nb = uni_b // binsz
    INF = float("inf")
    dp = [0.0] + [INF] * nb
    for n in sorted(tens):
        nd = [INF] * (nb + 1)
        for used in range(nb + 1):
            if dp[used] == INF:
                continue
            for ty, (bb, ee) in tens[n].items():
                assert bb % binsz == 0
                u2 = used + bb // binsz
                if u2 <= nb and dp[used] + ee < nd[u2]:
                    nd[u2] = dp[used] + ee
        dp = nd
    opt = min(dp)
    check(te <= opt * 1.02, f"greedy within 2% of the exact DP optimum (greedy {te:.5g}, optimum {opt:.5g}, gap {100 * (te / opt - 1):.2f}%)")
    check(te >= opt - 1e-9, "the DP oracle is a true lower bound (greedy never beats it)")
    # control: equal errors for every type -> never worse and never larger than the uniform recipe
    eq = {f"e{i}": {ty: (int(1000000 * b / 8), 1.0) for ty, b in bits.items()} for i in range(10)}
    ub = sum(c["q4"][0] for c in eq.values())
    _, eb, ee = solve(eq, ub)
    check(eb <= ub and ee <= 10.0 + 1e-9, "equal candidate errors: not worse and not larger than the uniform recipe")
    try:
        solve(tens, 1000)
        check(False, "an impossible budget is refused")
    except SystemExit:
        check(True, "an impossible budget is refused")
    h = lower_hull([(10, 5.0, "a"), (20, 6.0, "dominated"), (30, 1.0, "b")])
    check([p[2] for p in h] == ["a", "b"], "a dominated candidate (more bytes, more error) is dropped from the hull")
    # p-norm objective (EXL2 nuance 2): a larger p must not give a larger max relative error than p = 1, at the same budget
    meas = {"tensors": {}}
    for n, c in tens.items():
        numel = int(c["q4"][0] * 8 / 4.5)
        meas["tensors"][n] = {"numel": numel, "cand": {ty: {"bytes": b, "err": e, "energy": numel * 1.0} for ty, (b, e) in c.items()}}
    ch1, _, _ = solve_objective(meas, uni_b, "rel", 1.0)
    ch3, _, _ = solve_objective(meas, uni_b, "rel", 3.5)
    chauto, _, pu = solve_objective(meas, uni_b, "rel", "auto")
    mx1, _ = rel_stats(meas, ch1)
    mx3, _ = rel_stats(meas, ch3)
    mxa, _ = rel_stats(meas, chauto)
    check(mx3 <= mx1 + 1e-12, f"p=3.5 max relative error {mx3:.5f} <= p=1 {mx1:.5f} (the norm trades total for balance)")
    check(mxa <= mx3 + 1e-12 and 1.5 <= pu <= 3.5, f"auto p={pu} keeps the smallest max relative error ({mxa:.5f})")
    # NEGATIVE CONTROL: the fit check above must be able to fail
    biggest = sum(max(c.values())[0] for c in tens.values())
    check(biggest > uni_b, "negative control: the largest type everywhere does NOT fit, so the fit check can fail")
    print("PASS" if not fails else f"FAILED {fails}")
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("measure")
    m.add_argument("--src", required=True)
    m.add_argument("--quantize", required=True)
    m.add_argument("--types", required=True)
    m.add_argument("--imatrix")
    m.add_argument("--workdir", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--allocate", default=r"^blk\..*\.weight$")
    m.add_argument("--reuse", action="store_true", help="reuse meas_<type>.gguf files already in the workdir")
    m.set_defaults(fn=cmd_measure)
    s = sp.add_parser("solve")
    s.add_argument("--measure", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--objective", choices=["rel", "abs"], default="rel", help="rel: p-norm of relative errors (EXL2 style); abs: total weighted squared error")
    s.add_argument("--pnorm", default="auto", help="p of the p-norm for --objective rel: a number, or auto = sweep 1.5..3.5 keeping the smallest max error")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--like")
    g.add_argument("--budget-bpw", type=float)
    g.add_argument("--budget-mib", type=float)
    s.set_defaults(fn=cmd_solve)
    t = sp.add_parser("selftest")
    t.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
