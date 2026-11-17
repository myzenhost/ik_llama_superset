#!/usr/bin/env python3
"""glm5-next TWO-IMPLEMENTATION PARITY (row S69-ENGREV-UPSTREAM-MERGE-GLM5-NEXT-TWO-IMPLEMENTATIONS-2026-11-04-01).

  python test-glm5next-two-impl-parity.py <tree-root> <build-bin-dir> [workdir]

A GGUF with general.architecture = "glm5-next" resolves to the FORK graph (LLM_ARCH_GLM5NEXT, src/graphs/build_glm5next.cpp); the merged UPSTREAM
builder (LLM_ARCH_GLM5_NEXT, src/models/glm5-next.cpp) is reachable for the SAME file only through LLAMA_GLM5NEXT_IMPL=upstream
(src/llama-model-loader.h, get_arch()). This test writes one random-weight file (tests/make_tiny_glm5next_gguf.py), decodes fixed token
prompts through tiny-logits on both paths (CPU, greedy) and requires:
  - the load log names the arch each path was meant to take ("arch = glm5next" fork / "arch = glm5-next" upstream), so a silent fallback to one
    path cannot pass as parity;
  - per prompt and per position the logits agree (max abs diff <= TOL x logit RMS) and the argmax is identical at every position;
  - greedy generation (-gen) emits the same tokens;
  - MUST-FAIL CONTROL: a model with different weights (other seed) run on the upstream path must NOT match the fork path on the first one,
    i.e. the comparison can tell two different models apart.
Exit 0 = all pass, 77 = numpy/gguf-py or tiny-logits missing (skipped).
"""
import os
import subprocess
import sys
import tempfile

root, bindir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
work = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="glm5next-parity-")
os.makedirs(work, exist_ok=True)
TOL = float(os.environ.get("GLM5NEXT_PARITY_TOL", "2e-3"))   # x logit RMS; float-noise scale, set from measurement (see findings file)
fails = []

# BOTH TOP-K MASK PATHS. The upstream builder has a fused top-k mask path and an unfused one; which one runs depends on the
# backend (the CUDA build logs that it disabled the fused path) or on LLAMA_FUSED_TKM=0. This test used to run only the build's
# default, so a CPU build (fused) passed while the CUDA gate (unfused) asserted at glm5-next.cpp:1036 and, once that was fixed,
# at :1104 (gate b1109i, 2026-11-09). Measured with LLAMA_FUSED_TKM=0 on a CPU build before the fixes: FAIL (3), 3 GGML_ASSERT.
# The driver below re-runs this file once per path in its own workdir and fails if either arm fails, so every build exercises
# both. GLM5NEXT_PARITY_TKM_ARM set = run one arm only (the inner invocation).
if "GLM5NEXT_PARITY_TKM_ARM" not in os.environ:
    arm_rc = {}
    for arm, tkm in (("default", None), ("unfused", "0")):
        env = dict(os.environ, GLM5NEXT_PARITY_TKM_ARM=arm)
        env.pop("LLAMA_FUSED_TKM", None)
        if tkm is not None:
            env["LLAMA_FUSED_TKM"] = tkm
        print("=== top-k mask arm: %s (LLAMA_FUSED_TKM=%s)" % (arm, tkm if tkm is not None else "unset"), flush=True)
        arm_rc[arm] = subprocess.run([sys.executable, os.path.abspath(__file__), root, bindir, os.path.join(work, arm)],
                                     env=env).returncode
    print("arms:", ", ".join("%s rc %d" % kv for kv in arm_rc.items()))
    if all(rc == 77 for rc in arm_rc.values()):
        sys.exit(77)
    bad = [a for a, rc in arm_rc.items() if rc != 0]
    print("RESULT: %s" % ("PASS" if not bad else "FAIL (arm(s) %s)" % ", ".join(bad)))
    sys.exit(1 if bad else 0)


def ok(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


try:
    import numpy as np
except ImportError:
    print("numpy missing: cannot run (SKIP)")
    sys.exit(77)

exe = os.path.join(bindir, "tiny-logits")
if not (os.path.exists(exe) or os.path.exists(exe + ".exe")):
    print("tiny-logits missing in " + bindir + " (SKIP)")
    sys.exit(77)

V = 64


def make(seed, name):
    path = os.path.join(work, name)
    r = subprocess.run([sys.executable, os.path.join(root, "tests", "make_tiny_glm5next_gguf.py"), root, path, str(seed)],
                       capture_output=True, text=True)
    ok(r.returncode == 0, "random-weight glm5-next GGUF seed %d written: %s" % (seed, r.stdout.strip() or r.stderr.strip()[-200:]))
    return path


def run(model, impl, tokens, tag, ub, gen=0, fa=0):
    env = dict(os.environ)
    env.pop("LLAMA_GLM5NEXT_IMPL", None)
    if impl == "upstream":
        env["LLAMA_GLM5NEXT_IMPL"] = "upstream"
    out = os.path.join(work, "%s.%s.%s.bin" % (os.path.basename(model), impl, tag))
    # -dsa 1: the fork's k-pool indexer is OPT-IN (cparams.dsa, llama-cparams.h:96). Without it the fork path runs dense MLA and is bit-identical for
    # every indexer.top_k (measured 2026-11-09), while the upstream builder always runs the indexer -- a "divergence" that is only a flag.
    r = subprocess.run([exe, "-m", model, "-t", ",".join(map(str, tokens)), "-ub", str(ub), "-gen", str(gen), "-fa", str(fa), "-dsa", "1", "-o", out],
                       capture_output=True, text=True, env=env)
    return r, out


def device_cache_buffers(r):
    """-> [(device, kind, MiB)] for every non-CPU K/V or recurrent-state buffer the run allocated with a non-zero size.
    tiny-logits loads with n_gpu_layers 0, so in a CUDA build every cache must stay on the host. MEASURED 2026-11-10 (gate b1109j):
    the upstream path put 0.12 + 0.16 MiB of KV and 0.01 MiB of recurrent state on CUDA0 (llama_model::dev_layer read the pinned
    host buffer type's CUDA0 device), the scheduler then ran those ops on the GPU and the upstream logits moved by 0.128 x rms."""
    import re
    out = []
    for ln in (r.stderr + r.stdout).splitlines():
        m = re.search(r"(\S+) (KV|RS) buffer size =\s+([0-9.]+) MiB", ln)
        # CPU* and *_Host (pinned host memory, e.g. CUDA_Host: the fork path's cache) are host buffers; anything else is a device
        if m and not m.group(1).upper().startswith("CPU") and not m.group(1).upper().endswith("_HOST") and float(m.group(3)) > 0:
            out.append((m.group(1), m.group(2), float(m.group(3))))
    return out


def arch_line(r):
    for ln in (r.stderr + r.stdout).splitlines():
        if "arch" in ln and "=" in ln and ("print_meta" in ln or "llm_load_print_meta" in ln):
            if ln.split("arch")[1].lstrip().startswith("="):
                return ln.strip()
    return ""


m1 = make(2468, "tiny-glm5next-a.gguf")
m2 = make(1357, "tiny-glm5next-b.gguf")

# prompts: short (dense: fewer cells than one pool x top-k), medium (indexer picks 2 of 5 pools), long (chunked into ubatches of 8 = state carried across ubatches)
PROMPTS = {
    "short5":  ([5, 9, 12, 33, 7], 8),
    "mid21":   ([3 + (i * 7 + 2) % (V - 3) for i in range(21)], 21),
    "long40u8": ([3 + (i * 37 + 5) % (V - 3) for i in range(40)], 8),
}
worst = 0.0
fork_first = None
for name, (toks, ub) in PROMPTS.items():
    rf, pf = run(m1, "fork", toks, name, ub, gen=6)
    ru, pu = run(m1, "upstream", toks, name, ub, gen=6)
    ok(rf.returncode == 0 and "LOGITS" in rf.stdout, "%s fork path decodes: %s" % (name, (rf.stdout.strip() or rf.stderr.strip()[-300:])))
    ok(ru.returncode == 0 and "LOGITS" in ru.stdout, "%s upstream path decodes: %s" % (name, (ru.stdout.strip() or ru.stderr.strip()[-300:])))
    if not (rf.returncode == 0 and ru.returncode == 0):
        continue
    ok(not device_cache_buffers(rf) and not device_cache_buffers(ru),
       "%s n_gpu_layers 0 keeps every cache on the host: fork %s / upstream %s" % (name, device_cache_buffers(rf) or "none on a device",
                                                                               device_cache_buffers(ru) or "none on a device"))
    af, au = arch_line(rf), arch_line(ru)
    ok(af.endswith("glm5next") and au.endswith("glm5-next"), "%s paths differ: fork '%s' / upstream '%s'" % (name, af, au))
    lf = np.fromfile(pf, dtype=np.float32).reshape(-1, V)
    lu = np.fromfile(pu, dtype=np.float32).reshape(-1, V)
    ok(lf.shape == lu.shape and np.isfinite(lf).all() and np.isfinite(lu).all(), "%s logits finite, shape %s" % (name, lf.shape))
    if lf.shape != lu.shape:
        continue
    rms = float(np.sqrt((lf.astype(np.float64) ** 2).mean()))
    d = float(np.abs(lf - lu).max())
    worst = max(worst, d / max(rms, 1e-12))
    am = bool((lf.argmax(1) == lu.argmax(1)).all())
    ok(d <= TOL * rms, "%s max|diff|=%.3e = %.2e x rms(%.4f) (tol %.1e x)" % (name, d, d / max(rms, 1e-12), rms, TOL))
    ok(am, "%s argmax identical at all %d positions" % (name, lf.shape[0]))
    gf = [ln for ln in rf.stdout.splitlines() if ln.startswith("GEN:")]
    gu = [ln for ln in ru.stdout.splitlines() if ln.startswith("GEN:")]
    ok(gf == gu and len(gf) == 1, "%s greedy GEN identical: %s | %s" % (name, gf, gu))
    if fork_first is None:
        fork_first = lf
print("worst relative diff over prompts: %.3e" % worst)

# must-fail control: a different model on the upstream path must not match the first model's fork logits
if fork_first is not None:
    toks, ub = PROMPTS["short5"]
    rc_, pc = run(m2, "upstream", toks, "short5ctl", ub)
    rf0, pf0 = run(m1, "fork", toks, "short5ctl", ub)
    if rc_.returncode == 0 and rf0.returncode == 0:
        a = np.fromfile(pc, dtype=np.float32).reshape(-1, V)
        b = np.fromfile(pf0, dtype=np.float32).reshape(-1, V)
        rms = float(np.sqrt((b.astype(np.float64) ** 2).mean()))
        ok(float(np.abs(a - b).max()) > 10 * TOL * rms, "control: different weights differ by %.2e x rms (must exceed %.1e x)" %
           (float(np.abs(a - b).max()) / rms, 10 * TOL))
    else:
        ok(False, "control runs failed: %s %s" % (rc_.stderr[-200:], rf0.stderr[-200:]))

# selection-active control: the same weights with indexer.top_k 64 (no pool is ever dropped for these prompts) must give DIFFERENT logits from top_k 8 on
# the mid21 prompt, otherwise the indexer was never exercised and the parity above would be dense-vs-dense.
if "mid21" in PROMPTS:
    r = subprocess.run([sys.executable, os.path.join(root, "tests", "make_tiny_glm5next_gguf.py"), root, os.path.join(work, "tiny-glm5next-dense.gguf"), "2468", "64"],
                       capture_output=True, text=True)
    toks, ub = PROMPTS["mid21"]
    rd, pd_ = run(os.path.join(work, "tiny-glm5next-dense.gguf"), "fork", toks, "mid21dense", ub)
    rs, ps = run(m1, "fork", toks, "mid21sel", ub)
    if rd.returncode == 0 and rs.returncode == 0:
        a = np.fromfile(pd_, dtype=np.float32).reshape(-1, V)
        b = np.fromfile(ps, dtype=np.float32).reshape(-1, V)
        rms = float(np.sqrt((b.astype(np.float64) ** 2).mean()))
        ok(float(np.abs(a - b).max()) > 100 * TOL * rms, "control: indexer selection changes the logits (top_k 8 vs 64): %.2e x rms (must exceed %.1e x)" %
           (float(np.abs(a - b).max()) / rms, 100 * TOL))
    else:
        ok(False, "selection control runs failed: %s" % (rd.stderr[-200:] + rs.stderr[-200:]))

print("RESULT: %s" % ("PASS" if not fails else "FAIL (%d)" % len(fails)))
sys.exit(1 if fails else 0)
