#!/usr/bin/env python3
"""Model-level test of the recurrent ops through the real graph (row S69-GGML-OP-3-RWKV-WKV6-WKV7 wiring).

  python test-wiring-rwkv6.py <tree-root> <build-bin-dir> [workdir]

Builds a random-weight rwkv6 GGUF (tests/make_tiny_rwkv6_gguf.py, needs numpy), loads it through the real library
(src/models/rwkv6.cpp -> build_rwkv6_time_mix -> GGML_OP_RWKV_WKV6) and decodes 48 tokens twice with tiny-logits:
  A) one ubatch of 48 tokens   B) 48 ubatches of 1 token (the recurrent state is carried between decodes by the real
     llama_memory_recurrent code).
Checks: arch line names rwkv6; logits finite and non-trivial; A and B agree (max abs diff <= 1e-3, i.e. the state
carry is right); the reference digest REF_SUM matches (it changes if the WKV6 kernel's state selection is mutated,
which is how the test was mutation-checked: `t % (T / n_seqs)` -> `(t + 1) % (T / n_seqs)` in ggml_compute_forward_rwkv_wkv6_f32).
"""
import array, math, os, re, subprocess, sys, tempfile

root, bindir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
work = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="wiring-rwkv6-")
os.makedirs(work, exist_ok=True)
REF_RMS, REF_TOL = float(os.environ.get("RWKV6_REF_RMS", "3.003432")), 1e-3
try:
    import numpy  # noqa: F401
except ImportError:
    print("numpy missing: cannot run (SKIP)"); sys.exit(0)
fails = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


gg = os.path.join(work, "rw6.gguf")
subprocess.run([sys.executable, os.path.join(root, "tests", "make_tiny_rwkv6_gguf.py"), root, gg], check=True, stdout=subprocess.DEVNULL)
res = {}
for ub in (48, 1):
    o = os.path.join(work, "l%d.bin" % ub)
    p = subprocess.run([os.path.join(bindir, "tiny-logits"), "-m", gg, "-n", "48", "-ub", str(ub), "-fa", "0", "-o", o],
                       capture_output=True, text=True)
    check(p.returncode == 0 and "LOGITS" in p.stdout, "ub=%d decodes through the real rwkv6 graph (rc=%d)" % (ub, p.returncode))
    if p.returncode:
        print(p.stdout, p.stderr[-800:]); continue
    check(re.search(r"arch\s*=\s*rwkv6", p.stderr) is not None, "ub=%d log names arch rwkv6" % ub)
    a = array.array("f"); a.frombytes(open(o, "rb").read())
    res[ub] = (a, float(re.search(r"rms=([\d.]+)", p.stdout).group(1)))
if len(res) == 2:
    a, b = res[48][0], res[1][0]
    check(all(math.isfinite(x) for x in a), "logits finite")
    check(res[48][1] > 0.1, "logits non-trivial (rms %.4f)" % res[48][1])
    d = max(abs(x - y) for x, y in zip(a, b))
    check(d <= 1e-3, "batched vs token-by-token logits agree (max abs diff %.2e <= 1e-3): recurrent state carried" % d)
    check(abs(res[48][1] - REF_RMS) <= REF_TOL, "logit rms %.6f == reference %.6f (+-%g): WKV6 kernel digest" % (res[48][1], REF_RMS, REF_TOL))
print("test-wiring-rwkv6:", "PASS" if not fails else "FAIL")
sys.exit(1 if fails else 0)
