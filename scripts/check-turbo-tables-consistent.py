#!/usr/bin/env python3
"""Turbo KV constants: every copy must agree, and the quantiser tables must be the Lloyd-Max optimum.

WHY THIS EXISTS. turbo KV blocks are WRITTEN on the CPU (ggml-turbo-quant.c) and READ by the GPU flash-attention kernel (ggml-cuda/turbo-quant.cuh); the
CUDA writer is checked against a host reference in tests/turbo-write-check.h. If any copy of the WHT sign arrays or the centroid/midpoint tables differs, both sides
still produce valid blocks and the decoded vectors are silently wrong on every token. The comment in ggml-turbo-quant.c used to name a verifier that did not exist
in this tree and, where it did exist, did not parse any source file. This script is that verifier.

CHECKS (all must pass, exit 1 naming the first divergence otherwise):
  1. WHT sign arrays: CPU turbo_cpu_s1/s2 == CUDA TURBO_WHT_SIGNS1/2 (128 entries each, +-1).
  2. 3-bit and 4-bit centroid tables identical across CPU, CUDA and tests/turbo-write-check.h (kCent3/kMid3, kCent4/kMid4), tol 1e-6.
  3. Midpoint tables / nearest_centroid thresholds == midpoints of the centroids, tol 2e-6.
  4. Centroids == the Lloyd-Max optimum for N(0, 1/128), computed here by iteration with math.erf (independent of any engine's tables), tol --opt-tol (default 5e-5).
  5. TURBO_WHT_SIGNBITS1/2 (packed words read by the warp-shuffle WHT kernel) == the sign bits of TURBO_WHT_SIGNS1/2 (element e = bit e&31 of word e>>5, set = -1.0f).
  7. SYCL (ggml-sycl/turbo-quant.hpp) and METAL (ggml-metal/kernels/turbo.h): C-array copies of the sign arrays and the 2/3/4-bit centroids / thresholds,
     each defined exactly once, checked against the CPU tables (check_c_backend); the selftest nudges one value of each table per backend.
  6. VULKAN: ggml-vulkan/vulkan-shaders/turbo_tables.glsl (the ONE copy every Vulkan turbo shader includes) carries the same sign arrays (128 entries), the
     2/3/4-bit centroids and their thresholds, the 32-wide weight signs and 1/sqrt(128); and no other Vulkan shader embeds a centroid literal (a second copy
     is how the first port of this feature ended up with six).
--selftest: a tampered copy of the real sources must FAIL checks 1, 2, 4, 5 and 6, and the OLD (pre-2026-10-05) 4-bit table must fail check 4. A checker that cannot fail proves nothing.
Row S69-THETOM-TURBO-CENTROID-TABLES-3BIT-4BIT-ADOPT-2026-10-05-01.
"""
import argparse
import math
import os
import re
import sys

SIGMA = 1.0 / math.sqrt(128.0)
OLD_4BIT = [-0.173926, -0.117195, -0.089527, -0.068756, -0.051262, -0.035597, -0.020989, -0.006938,
            0.006938, 0.020989, 0.035597, 0.051262, 0.068756, 0.089527, 0.117195, 0.173926]


def lloyd_max(levels, sigma=SIGMA, iters=5000):
    """Lloyd-Max centroids for N(0, sigma^2), `levels` of them, by fixed-point iteration (pure math, no engine data)."""
    phi = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    Phi = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    c = [sigma * (-2.5 + 5.0 * (i + 0.5) / levels) for i in range(levels)]
    for _ in range(iters):
        b = [-math.inf] + [(c[i] + c[i + 1]) / 2.0 for i in range(levels - 1)] + [math.inf]
        new = []
        for i in range(levels):
            lo, hi = b[i] / sigma, b[i + 1] / sigma
            plo = 0.0 if lo == -math.inf else phi(lo)
            phi_hi = 0.0 if hi == math.inf else phi(hi)
            Plo = 0.0 if lo == -math.inf else Phi(lo)
            Phi_hi = 1.0 if hi == math.inf else Phi(hi)
            new.append(sigma * (plo - phi_hi) / (Phi_hi - Plo))
        if max(abs(x - y) for x, y in zip(c, new)) < 1e-13:
            c = new
            break
        c = new
    return c


def nums(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:\.\d*)?", text.replace("f", " "))]


def array(src, name):
    """All initialiser lists for `name[...] = { ... };` in src (a name may be defined twice, e.g. CENTROIDS_4BIT in two functions)."""
    return [nums(m.group(1)) for m in re.finditer(r"\b" + re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;", src, re.S)]


def hex_words(src, name):
    """All `name[...] = { 0x...u, ... };` initialiser lists as lists of ints."""
    return [[int(x, 16) for x in re.findall(r"0[xX][0-9A-Fa-f]+", m.group(1))]
            for m in re.finditer(r"\b" + re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;", src, re.S)]


def thresholds(src, func):
    m = re.search(r"static\s+int\s+" + func + r"\s*\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
    if not m:
        return None
    return [float(x) for x in re.findall(r"val\s*<\s*(-?\d+\.\d+)f", m.group(1)) if abs(float(x)) > 0.0]


def glsl_array(src, name):
    """All `const float NAME[N] = float[N]( ... );` initialiser lists for `name` in a GLSL source."""
    return [nums(m.group(1)) for m in re.finditer(r"\b" + re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*float\s*\[[^\]]*\]\s*\((.*?)\)\s*;", src, re.S)]


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def check(cpu, cuda, tst, opt_tol, quiet=False):
    """Return a list of failure strings (empty = pass)."""
    fails = []
    log = (lambda *a: None) if quiet else print
    s1c, s2c = array(cpu, "turbo_cpu_s1"), array(cpu, "turbo_cpu_s2")
    s1g, s2g = array(cuda, "TURBO_WHT_SIGNS1"), array(cuda, "TURBO_WHT_SIGNS2")
    if not (s1c and s2c and s1g and s2g):
        fails.append("sign arrays not found (cpu s1/s2 %s/%s, cuda %s/%s): the parser reached nothing" % (len(s1c), len(s2c), len(s1g), len(s2g)))
    else:
        for tag, a, b in (("SIGNS1", s1c[0], s1g[0]), ("SIGNS2", s2c[0], s2g[0])):
            ok = len(a) == len(b) == 128 and a == b
            log("  [%s] WHT %s CPU == CUDA (%d vs %d entries)" % ("ok" if ok else "FAIL", tag, len(a), len(b)))
            if not ok:
                fails.append("WHT %s differs between ggml-turbo-quant.c and turbo-quant.cuh" % tag)
    # 5. packed sign words used by the warp-shuffle WHT kernel (turbo-wht.cu) == the sign arrays, bit for bit (element e = bit e&31 of word e>>5, set = -1.0f)
    for tag, sg, bn in (("SIGNS1", s1g, "TURBO_WHT_SIGNBITS1"), ("SIGNS2", s2g, "TURBO_WHT_SIGNBITS2")):
        words = hex_words(cuda, bn)
        if not words or not sg:
            fails.append("%s packed words not found: the parser reached nothing" % bn)
            continue
        want = [sum(1 << b for b in range(32) if sg[0][32 * w + b] < 0) for w in range(4)] if len(sg[0]) == 128 else None
        ok = want is not None and words[0] == want
        log("  [%s] WHT %s packed words == bits of the CUDA array (%s)" % ("ok" if ok else "FAIL", tag, ["%08X" % w for w in words[0]]))
        if not ok:
            fails.append("%s does not equal the sign bits of TURBO_WHT_%s (the warp-shuffle kernel would apply different signs than the shared-memory one)" % (bn, tag))
    for bits, levels, cname, gname, gmid, tname, tmid, nearest in (
            (3, 8, "CENTROIDS_3BIT", "TURBO_CENTROIDS_3BIT", "TURBO_MID_3BIT", "kCent3", "kMid3", "nearest_centroid_3bit"),
            (4, 16, "CENTROIDS_4BIT", "TURBO_CENTROIDS_4BIT", "TURBO_MID_4BIT", "kCent4", "kMid4", "nearest_centroid_4bit")):
        cpu_c, gpu_c = array(cpu, cname), array(cuda, gname)
        copies = [("CPU %s #%d" % (cname, i + 1), t) for i, t in enumerate(cpu_c)] + [("CUDA " + gname, t) for t in gpu_c]
        if tname:
            copies += [("tests " + tname, t) for t in array(tst, tname)]
        if len(cpu_c) < 1 or len(gpu_c) < 1:
            fails.append("%d-bit centroid table not found (cpu %d, cuda %d)" % (bits, len(cpu_c), len(gpu_c)))
            continue
        ref = copies[0][1]
        opt = lloyd_max(levels)
        for label, t in copies:
            same = len(t) == levels and max(abs(x - y) for x, y in zip(t, ref)) < 1e-6
            dev = max(abs(x - y) for x, y in zip(t, opt)) if len(t) == levels else float("nan")
            good = len(t) == levels and dev < opt_tol
            log("  [%s] %d-bit %-34s same-as-CPU=%s  max|dev from Lloyd-Max|=%.2e (tol %.0e)" % ("ok" if same and good else "FAIL", bits, label, same, dev, opt_tol))
            if not same:
                fails.append("%d-bit table '%s' differs from '%s'" % (bits, label, copies[0][0]))
            if not good:
                fails.append("%d-bit table '%s' is %.2e from the Lloyd-Max optimum (tol %.0e)" % (bits, label, dev, opt_tol))
        mids = [(ref[i] + ref[i + 1]) / 2.0 for i in range(levels - 1)]
        nz = [m for m in mids if abs(m) > 1e-12]
        for label, t in [("CPU " + nearest, thresholds(cpu, nearest))] + ([("CUDA " + gmid, [x for x in (array(cuda, gmid) or [[]])[0] if x != 0.0])] if gmid else []) + \
                ([("tests " + tmid, [x for x in (array(tst, tmid) or [[]])[0] if x != 0.0])] if tmid else []):
            if t is None or len(t) != len(nz):
                fails.append("%d-bit midpoints '%s': found %s, expected %d" % (bits, label, None if t is None else len(t), len(nz)))
                continue
            dev = max(abs(x - y) for x, y in zip(sorted(t), sorted(nz)))
            log("  [%s] %d-bit midpoints %-30s max|dev from centroid midpoints|=%.2e" % ("ok" if dev < 2e-6 else "FAIL", bits, label, dev))
            if dev >= 2e-6:
                fails.append("%d-bit midpoints '%s' are %.2e from the midpoints of the centroids" % (bits, label, dev))
    return fails


def check_vulkan(cpu, cuda, vk, other_vk_sources, quiet=False):
    """Check 6: the Vulkan constants (one include) against the CPU reference. other_vk_sources: {filename: text} of every other Vulkan shader."""
    fails = []
    log = (lambda *a: None) if quiet else print
    if not vk:
        return ["turbo_tables.glsl not found or empty: the Vulkan check reached nothing"]

    def one(name):
        a = glsl_array(vk, name)
        if len(a) != 1:
            fails.append("Vulkan table %s: expected exactly one definition, found %d" % (name, len(a)))
            return None
        return a[0]

    s1c, s2c = array(cpu, "turbo_cpu_s1"), array(cpu, "turbo_cpu_s2")
    for tag, vname, ref in (("SIGNS1", "TURBO_S1", s1c), ("SIGNS2", "TURBO_S2", s2c)):
        v = one(vname)
        if v is None or not ref:
            fails.append("Vulkan %s: could not compare (found %s vs CPU %s)" % (tag, None if v is None else len(v), len(ref)))
            continue
        ok = len(v) == 128 and v == ref[0]
        log("  [%s] Vulkan %s == CPU turbo_cpu_%s (128 entries)" % ("ok" if ok else "FAIL", vname, tag.lower().replace("signs", "s")))
        if not ok:
            fails.append("Vulkan %s differs from ggml-turbo-quant.c turbo_cpu_%s" % (vname, tag.lower().replace("signs", "s")))
    for bits, levels, cname, vname, vmid, nearest in ((2, 4, "CENTROIDS_2BIT", "TURBO_CENTROIDS_2BIT", "TURBO_MID_2BIT", "nearest_centroid_2bit"),
                                                     (3, 8, "CENTROIDS_3BIT", "TURBO_CENTROIDS_3BIT", "TURBO_MID_3BIT", "nearest_centroid_3bit"),
                                                     (4, 16, "CENTROIDS_4BIT", "TURBO_CENTROIDS_4BIT", "TURBO_MID_4BIT", "nearest_centroid_4bit")):
        ref = array(cpu, cname)
        v, vm = one(vname), one(vmid)
        if not ref or v is None or vm is None:
            fails.append("Vulkan %d-bit tables: missing (cpu %d, centroids %s, thresholds %s)" % (bits, len(ref), v is not None, vm is not None))
            continue
        same = len(v) == levels and max(abs(x - y) for x, y in zip(v, ref[0])) < 1e-6
        log("  [%s] Vulkan %d-bit centroids %s == CPU %s (%d entries)" % ("ok" if same else "FAIL", bits, vname, cname, len(v)))
        if not same:
            fails.append("Vulkan %s differs from ggml-turbo-quant.c %s" % (vname, cname))
        mids = [(ref[0][i] + ref[0][i + 1]) / 2.0 for i in range(levels - 1)]
        thr = thresholds(cpu, nearest)
        okm = len(vm) == levels - 1 and max(abs(x - y) for x, y in zip(vm, mids)) < 2e-6
        okt = thr is not None and sorted(x for x in vm if x != 0.0) == sorted(thr)
        log("  [%s] Vulkan %d-bit thresholds %s are the centroid midpoints and == CPU %s" % ("ok" if okm and okt else "FAIL", bits, vmid, nearest))
        if not okm:
            fails.append("Vulkan %s are not the midpoints of the centroids" % vmid)
        if not okt:
            fails.append("Vulkan %s differs from the %s thresholds in ggml-turbo-quant.c" % (vmid, nearest))
    tq = one("TURBO_TQW_SIGNS")
    tqc = array(cpu, "TQ3_0_SIGNS")
    tqg = array(cuda, "TQ_WEIGHT_SIGNS")
    if tq is None or not tqc or not tqg:
        fails.append("32-wide weight signs: not found in all of Vulkan / CPU / CUDA")
    else:
        ok = tq == tqc[0] == tqg[0] and len(tq) == 32
        log("  [%s] Vulkan TURBO_TQW_SIGNS == CPU TQ3_0_SIGNS == CUDA TQ_WEIGHT_SIGNS (32 entries)" % ("ok" if ok else "FAIL"))
        if not ok:
            fails.append("the 32-wide weight sign vector differs between Vulkan, CPU and CUDA")
    m = re.search(r"TURBO_INV_SQRT_128\s*=\s*([0-9.eE+-]+)\s*;", vk)
    ok = bool(m) and abs(float(m.group(1)) - 0.08838834764831845) < 1e-15
    log("  [%s] Vulkan TURBO_INV_SQRT_128 == 1/sqrt(128)" % ("ok" if ok else "FAIL"))
    if not ok:
        fails.append("Vulkan TURBO_INV_SQRT_128 is not 1/sqrt(128)")
    # single copy: no other Vulkan shader may embed a centroid literal
    needles = ["0.190207", "0.133462", "0.241529", "0.118786", "0.039994"]
    for fn, text in sorted(other_vk_sources.items()):
        for nd in needles:
            if nd in text:
                fails.append("Vulkan shader %s embeds the centroid literal %s: a second copy of the tables (use turbo_tables.glsl)" % (fn, nd))
                break
    log("  [%s] no Vulkan shader other than turbo_tables.glsl embeds a centroid literal (%d files scanned)" % ("ok" if not any("embeds" in f for f in fails) else "FAIL", len(other_vk_sources)))
    if not other_vk_sources:
        fails.append("the single-copy scan reached no Vulkan shader files")
    return fails


def check_c_backend(label, cpu, src, quiet=False):
    """Check 7/8: a backend whose constants are C arrays (`static constexpr float NAME[N] = { ... };`, SYCL and Metal) against the CPU reference.
    Every table must be defined exactly once in `src`."""
    fails = []
    log = (lambda *a: None) if quiet else print
    if not src:
        return ["%s tables not found or empty: the %s check reached nothing" % (label, label)]

    def one(name):
        a = array(src, name)
        if len(a) != 1:
            fails.append("%s table %s: expected exactly one definition, found %d" % (label, name, len(a)))
            return None
        return a[0]

    s1c, s2c = array(cpu, "turbo_cpu_s1"), array(cpu, "turbo_cpu_s2")
    for tag, vname, ref in (("s1", "TURBO_WHT_SIGNS1", s1c), ("s2", "TURBO_WHT_SIGNS2", s2c)):
        v = one(vname)
        if v is None or not ref:
            fails.append("%s %s: could not compare (found %s vs CPU %s)" % (label, vname, None if v is None else len(v), len(ref)))
            continue
        ok = len(v) == 128 and v == ref[0]
        log("  [%s] %s %s == CPU turbo_cpu_%s (128 entries)" % ("ok" if ok else "FAIL", label, vname, tag))
        if not ok:
            fails.append("%s %s differs from ggml-turbo-quant.c turbo_cpu_%s" % (label, vname, tag))
    for bits, levels, cname, vname, vmid, nearest in ((2, 4, "CENTROIDS_2BIT", "TURBO_CENTROIDS_2BIT", "TURBO_MID_2BIT", "nearest_centroid_2bit"),
                                                     (3, 8, "CENTROIDS_3BIT", "TURBO_CENTROIDS_3BIT", "TURBO_MID_3BIT", "nearest_centroid_3bit"),
                                                     (4, 16, "CENTROIDS_4BIT", "TURBO_CENTROIDS_4BIT", "TURBO_MID_4BIT", "nearest_centroid_4bit")):
        ref = array(cpu, cname)
        v, vm = one(vname), one(vmid)
        if not ref or v is None or vm is None:
            fails.append("%s %d-bit tables: missing (cpu %d, centroids %s, thresholds %s)" % (label, bits, len(ref), v is not None, vm is not None))
            continue
        same = len(v) == levels and max(abs(x - y) for x, y in zip(v, ref[0])) < 1e-6
        log("  [%s] %s %d-bit centroids %s == CPU %s (%d entries)" % ("ok" if same else "FAIL", label, bits, vname, cname, len(v)))
        if not same:
            fails.append("%s %s differs from ggml-turbo-quant.c %s" % (label, vname, cname))
        mids = [(ref[0][i] + ref[0][i + 1]) / 2.0 for i in range(levels - 1)]
        thr = thresholds(cpu, nearest)
        okm = len(vm) == levels - 1 and max(abs(x - y) for x, y in zip(vm, mids)) < 2e-6
        okt = thr is not None and sorted(x for x in vm if x != 0.0) == sorted(thr)
        log("  [%s] %s %d-bit thresholds %s are the centroid midpoints and == CPU %s" % ("ok" if okm and okt else "FAIL", label, bits, vmid, nearest))
        if not okm:
            fails.append("%s %s are not the midpoints of the centroids" % (label, vmid))
        if not okt:
            fails.append("%s %s differs from the %s thresholds in ggml-turbo-quant.c" % (label, vmid, nearest))
    return fails


def selftest(cpu, cuda, tst, opt_tol, vk="", other_vk=None, c_backends=None):
    ok = True

    def expect(name, fails, want_fail):
        nonlocal ok
        good = (len(fails) > 0) == want_fail
        ok &= good
        print("  [%s] selftest: %s -> %s" % ("ok" if good else "FAIL", name, "failed as required" if want_fail else "passed as required"))

    expect("the real sources", check(cpu, cuda, tst, opt_tol, quiet=True), False)
    expect("one CUDA WHT sign flipped", check(cpu, re.sub(r"(TURBO_WHT_SIGNS1\s*\[[^\]]*\]\s*=\s*\{\s*)(-?1\.0f)", lambda m: m.group(1) + ("1.0f" if m.group(2).startswith("-") else "-1.0f"), cuda, count=1), tst, opt_tol, quiet=True), True)
    expect("one packed sign bit flipped (SIGNBITS1 word 2)", check(cpu, cuda.replace("0xA4C63BBBu", "0xA4C63BBAu", 1), tst, opt_tol, quiet=True), True)
    expect("one CUDA 3-bit centroid nudged", check(cpu, cuda.replace("-0.190207f", "-0.190807f", 1), tst, opt_tol, quiet=True), True)
    old = ", ".join("%.6ff" % v for v in OLD_4BIT)
    expect("the OLD pre-2026-10-05 4-bit table in all copies", check(re.sub(r"(CENTROIDS_4BIT\s*\[16\]\s*=\s*\{).*?(\}\s*;)", lambda m: m.group(1) + old + m.group(2), cpu, flags=re.S),
                                                                       re.sub(r"(TURBO_CENTROIDS_4BIT\s*\[16\]\s*=\s*\{).*?(\}\s*;)", lambda m: m.group(1) + old + m.group(2), cuda, flags=re.S), tst, opt_tol, quiet=True), True)
    expect("a parser that reaches nothing (empty sources)", check("", "", "", opt_tol, quiet=True), True)
    if vk:
        other_vk = other_vk if other_vk is not None else {"x.comp": "// clean"}
        expect("the real Vulkan tables", check_vulkan(cpu, cuda, vk, other_vk, quiet=True), False)
        expect("one Vulkan S2 sign flipped", check_vulkan(cpu, cuda, re.sub(r"(TURBO_S2\s*\[[^\]]*\]\s*=\s*float\s*\[[^\]]*\]\s*\(\s*)(-?\d+)", lambda m: m.group(1) + ("1" if m.group(2).startswith("-") else "-1"), vk, count=1), other_vk, quiet=True), True)
        expect("one Vulkan 3-bit centroid nudged", check_vulkan(cpu, cuda, vk.replace("-0.190207", "-0.190807", 1), other_vk, quiet=True), True)
        expect("one Vulkan 4-bit threshold nudged", check_vulkan(cpu, cuda, vk.replace("-0.212203", "-0.212903", 1), other_vk, quiet=True), True)
        expect("one Vulkan weight sign flipped", check_vulkan(cpu, cuda, re.sub(r"(TURBO_TQW_SIGNS\s*\[[^\]]*\]\s*=\s*float\s*\[[^\]]*\]\s*\(\s*)\+1", lambda m: m.group(1) + "-1", vk, count=1), other_vk, quiet=True), True)
        expect("another Vulkan shader embedding a centroid literal", check_vulkan(cpu, cuda, vk, dict(other_vk, **{"rogue.comp": "const float c = -0.190207;"}), quiet=True), True)
        expect("a Vulkan parser that reaches nothing", check_vulkan(cpu, cuda, "", other_vk, quiet=True), True)
    for label, src in (c_backends or {}).items():
        if not src:
            continue
        expect("the real %s tables" % label, check_c_backend(label, cpu, src, quiet=True), False)
        expect("one %s S2 sign flipped" % label, check_c_backend(label, cpu, re.sub(r"(TURBO_WHT_SIGNS2\s*\[[^\]]*\]\s*=\s*\{\s*)(-?1\.0f)", lambda m: m.group(1) + ("1.0f" if m.group(2).startswith("-") else "-1.0f"), src, count=1), quiet=True), True)
        expect("one %s 3-bit centroid nudged" % label, check_c_backend(label, cpu, src.replace("-0.190207", "-0.190807", 1), quiet=True), True)
        expect("one %s 2-bit threshold nudged" % label, check_c_backend(label, cpu, src.replace("-0.086728", "-0.086128", 1), quiet=True), True)
        expect("one %s 4-bit centroid nudged" % label, check_c_backend(label, cpu, src.replace("0.241529", "0.241929", 1), quiet=True), True)
        expect("a %s parser that reaches nothing" % label, check_c_backend(label, cpu, "", quiet=True), True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--opt-tol", type=float, default=5e-5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    cpu = read(os.path.join(a.root, "ggml", "src", "ggml-turbo-quant.c"))
    cuda = read(os.path.join(a.root, "ggml", "src", "ggml-cuda", "turbo-quant.cuh"))
    tst = read(os.path.join(a.root, "tests", "turbo-write-check.h"))
    vkdir = os.path.join(a.root, "ggml", "src", "ggml-vulkan", "vulkan-shaders")
    vk = read(os.path.join(vkdir, "turbo_tables.glsl")) if os.path.exists(os.path.join(vkdir, "turbo_tables.glsl")) else ""
    other_vk = {}
    if os.path.isdir(vkdir):
        for fn in sorted(os.listdir(vkdir)):
            if fn != "turbo_tables.glsl" and fn.endswith((".comp", ".glsl")):
                other_vk[fn] = read(os.path.join(vkdir, fn))
    c_backends = {}
    sycl_path = os.path.join(a.root, "ggml", "src", "ggml-sycl", "turbo-quant.hpp")
    if os.path.exists(sycl_path):
        c_backends["SYCL"] = read(sycl_path)
    metal_path = os.path.join(a.root, "ggml", "src", "ggml-metal", "kernels", "turbo.h")
    if os.path.exists(metal_path):
        c_backends["Metal"] = read(metal_path)
    print("SCOPE: %s | cpu %d B, cuda %d B, tests %d B, vulkan tables %d B (+%d other shader files scanned), C-array backends %s | Lloyd-Max sigma=1/sqrt(128)=%.6f"
          % (a.root, len(cpu), len(cuda), len(tst), len(vk), len(other_vk), {k: len(v) for k, v in c_backends.items()}, SIGMA))
    fails = check(cpu, cuda, tst, a.opt_tol) + check_vulkan(cpu, cuda, vk, other_vk)
    for label, src in c_backends.items():
        fails += check_c_backend(label, cpu, src)
    if a.selftest:
        print("SELFTEST")
        if not selftest(cpu, cuda, tst, a.opt_tol, vk, other_vk, c_backends):
            print("SELFTEST FAILED")
            return 2
    for f in fails:
        print("FAIL:", f)
    print("RESULT:", "FAIL (%d)" % len(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
