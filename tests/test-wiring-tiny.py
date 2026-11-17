#!/usr/bin/env python3
"""End-to-end WIRING test on a random-weight GGUF (no model download).

  python test-wiring-tiny.py <tree-root> <build-bin-dir> [workdir]

1. MX types (row S69-MXFP6-MXFP8-INTEGRATION-FTYPE-GGUFPY-SIMD-MSVC-BUILD-2026-10-08-01):
   llama-quantize F32 -> {MXFP6_E2M3, MXFP6_E3M2, MXFP8_E4M3, MXFP8_E5M2, Q8_0} by NAME (ftype 70-73), load through the real library,
   decode 48 tokens through the real graph, compare the logits with the F32 model.
   - the load log must name the ftype ("model ftype = MXFP6_E2M3 ...") and report the tensors as that type;
   - tolerance: the logit relative RMS error must be <= 4 x the weight-level relative RMS bound of tests/test-mx-quant.cpp
     (0.045 for the 3-mantissa-bit formats E2M3/E4M3, 0.085 for the 2-mantissa-bit formats E3M2/E5M2). Rounding errors of the 15
     quantised matmuls of the 2-layer model are independent, so they add in quadrature: at most sqrt(15) = 3.9 x the weight error.
   - Q8_0 is the control: its error must be far below every MX error.
2. KIVI (--kivi-bits): tiny-logits generation with the flag ON must print the KIVI banner and a summary with quantised tokens > 0 and
   change the generated tokens; with the flag OFF (absent, and --kivi-bits 0 with the other kivi fields set) the logits are bit-identical.
   The cache-content checks are test-kivi-sim-cache (run here).
Exit 0 = all pass. Needs numpy for tests/make_tiny_mx_gguf.py (gguf-py).
"""
import os, re, subprocess, sys, tempfile

root, bindir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
work = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="wiring-tiny-")
os.makedirs(work, exist_ok=True)
V = 256
fails = []


def ok(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


try:
    import numpy as np
except ImportError:
    print("numpy missing: cannot run (SKIP)")
    sys.exit(77)

f32 = os.path.join(work, "tiny-f32.gguf")
r = run([sys.executable, os.path.join(root, "tests/make_tiny_mx_gguf.py"), root, f32])
ok(r.returncode == 0, "random-weight F32 GGUF written: " + r.stdout.strip())


def logits(model, extra=(), n=48, ub=48, gen=0, out=None):
    out = out or os.path.join(work, os.path.basename(model) + ".logits")
    r = run([os.path.join(bindir, "tiny-logits"), "-m", model, "-n", str(n), "-ub", str(ub), "-gen", str(gen), "-o", out, *extra])
    return r, out


def load(path, nvoc=V):
    return np.fromfile(path, dtype=np.float32).reshape(-1, nvoc)


r, ref_path = logits(f32)
ok(r.returncode == 0 and "LOGITS" in r.stdout, "F32 model decodes through the real graph: " + r.stdout.strip())
ref = load(ref_path)

# ---- 1. MX types
MX = {"MXFP6_E2M3": 0.045, "MXFP6_E3M2": 0.085, "MXFP8_E4M3": 0.045, "MXFP8_E5M2": 0.085}
rel = {}
for name in ["Q8_0"] + list(MX):
    q = os.path.join(work, "tiny-%s.gguf" % name)
    r = run([os.path.join(bindir, "llama-quantize"), f32, q, name])
    conv = re.findall(r"converting to (\w+)", r.stderr + r.stdout)
    ok(r.returncode == 0 and conv.count(name.lower()) >= 15, "llama-quantize %s: rc=%d, %d tensors converted to %s" % (name, r.returncode, conv.count(name.lower()), name.lower()))
    r, lp = logits(q)
    ok(r.returncode == 0 and "LOGITS" in r.stdout, "%s model decodes through the real graph" % name)
    log = r.stderr
    m = re.search(r"model ftype\s*=\s*(\S+)", log)
    ok(m is not None and m.group(1) == name, "log names the file type: 'model ftype = %s'" % (m.group(1) if m else None))
    m = re.search(r"llama_model_loader: - type\s+%s:\s+(\d+) tensors" % name.lower(), log)
    ok(m is not None and int(m.group(1)) >= 15, "log names the tensor type: '- type %s: %s tensors'" % (name.lower(), m.group(1) if m else None))
    a = load(lp)
    ok(not np.isnan(a).any(), "%s logits finite" % name)
    rel[name] = float(np.sqrt(((a - ref) ** 2).mean()) / np.sqrt((ref ** 2).mean()))
    print("       %-11s logit rel-RMS vs F32 = %.5f  top-1 agreement %.3f" % (name, rel[name], float((a.argmax(1) == ref.argmax(1)).mean())))
for name, wb in MX.items():
    ok(rel[name] <= 4 * wb, "%s logit rel-RMS %.4f <= 4 x %.3f (weight bound)" % (name, rel[name], wb))
    ok(rel[name] > rel["Q8_0"], "%s error exceeds the Q8_0 control (%.4f > %.4f): the MX weights are really in use" % (name, rel[name], rel["Q8_0"]))
ok(rel["MXFP8_E4M3"] < rel["MXFP6_E3M2"] and rel["MXFP6_E2M3"] < rel["MXFP6_E3M2"], "3-mantissa-bit formats beat 2-mantissa-bit formats (E2M3,E4M3 < E3M2)")

# ---- 2. KIVI
common = ["-n", "40", "-ub", "4", "-gen", "24", "-c", "128"]
kv = ["--kivi-group", "8", "--kivi-residual", "16"]


def gen(extra, out):
    r = run([os.path.join(bindir, "tiny-logits"), "-m", f32, *common, "-o", os.path.join(work, out), *extra])
    toks = re.search(r"GEN: (.*)", r.stdout)
    return r, (toks.group(1).split() if toks else None)


r0, g0 = gen([], "g0.bin")
r1, g1 = gen(["--kivi-bits", "0", *kv], "g1.bin")
ok(r0.returncode == 0 and r1.returncode == 0, "generation runs with the flag absent and with --kivi-bits 0")
ok(open(os.path.join(work, "g0.bin"), "rb").read() == open(os.path.join(work, "g1.bin"), "rb").read(), "OFF: flag absent vs --kivi-bits 0 -> logits bit-identical")
ok("KIVI" not in r0.stderr, "OFF: no KIVI log line")
r2, g2 = gen(["--kivi-bits", "2", *kv], "g2.bin")
ok(r2.returncode == 0 and g2 is not None, "generation runs with --kivi-bits 2")
ok("KIVI simulated KV quantisation ON: bits=2 group=8 residual=16" in r2.stderr and "K (f16)" in r2.stderr, "ON: banner names bits/group/residual and the cache type K (f16)")
m = re.search(r"K tokens quantised \(layers x heads x tokens\)=(\d+) V tokens quantised=(\d+)", r2.stderr)
ok(m is not None and int(m.group(1)) > 0 and int(m.group(2)) > 0, "ON: summary counters fired: " + (m.group(0) if m else "missing"))
ok(g2 != g0, "ON: generated tokens differ from OFF (%s... vs %s...)" % (g2[:8] if g2 else None, g0[:8] if g0 else None))
r3 = run([os.path.join(bindir, "test-kivi-sim-cache"), f32])
ok(r3.returncode == 0 and "TEST_KIVI_SIM_CACHE PASS" in r3.stdout, "test-kivi-sim-cache (cache contents, %d checks)" % r3.stdout.count("  ok  "))

print("test-wiring-tiny:", "PASS" if not fails else "FAIL (%d)" % len(fails))
sys.exit(1 if fails else 0)
