# Compare engine logits with the numpy reference: per (model, chunk) max abs diff, diff relative to logit absmax, per-position argmax agreement.
# CONTROLS: the dense engine output vs the MOE reference (wrong on purpose) must disagree widely; chunked vs unchunked engine output must agree.
# PROGRESS-OK: 6 small arrays. REWRITE-SAFE: read-only.
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])   # the directory make_tiny_gguf.py wrote into
T, V = 12, 64


def load(p):
    return np.fromfile(p, dtype=np.float32).reshape(T, V).astype(np.float64)


def cmp(a, b):
    d = np.abs(a - b)
    return d.max(), d.max() / np.abs(b).max(), int((a.argmax(1) == b.argmax(1)).sum())


ok = True
for kind in ("dense", "moe", "moe2"):
    ref = load(D / f"ref_{kind}.bin")
    outs = {}
    for c in (12, 5):
        eng = load(D / f"out_{kind}_c{c}.bin")
        outs[c] = eng
        mx, rel, am = cmp(eng, ref)
        good = rel < 1e-3 and am == T
        ok &= good
        print("%-5s chunk %-2d vs reference: max|diff| %.3e  rel-to-absmax %.3e  argmax agree %d/%d  %s" % (kind, c, mx, rel, am, T, "PASS" if good else "FAIL"))
    mx, rel, am = cmp(outs[5], outs[12])
    good = rel < 1e-4
    ok &= good
    print("%-5s chunk 5 vs chunk 12 (engine only):    max|diff| %.3e  rel %.3e  %s" % (kind, mx, rel, "PASS" if good else "FAIL"))

# CONTROLS
mx, rel, am = cmp(load(D / "out_dense_c12.bin"), load(D / "ref_moe.bin"))
sens = rel > 1e-2
ok &= sens
print("CONTROL dense engine vs MOE reference (wrong on purpose): rel %.3e argmax agree %d/%d  -> %s (must be large)" % (rel, am, T, "sensitive" if sens else "INSENSITIVE"))
mx, rel, am = cmp(load(D / "out_moe_c12.bin"), load(D / "ref_dense.bin"))
sens2 = rel > 1e-2
ok &= sens2
print("CONTROL moe engine vs DENSE reference (wrong on purpose): rel %.3e argmax agree %d/%d  -> %s (must be large)" % (rel, am, T, "sensitive" if sens2 else "INSENSITIVE"))
# the written norm_topk_prob / routed_scaling_factor must reach the graph: moe2 (norm off, scale 2.5) against a reference built with the
# DEFAULT-looking values (norm on, scale 1.5) on the SAME weights must disagree
mx, rel, am = cmp(load(D / "out_moe2_c12.bin"), load(D / "ref_moe2_wrongcfg.bin"))
sens3 = rel > 1e-2
ok &= sens3
print("CONTROL moe2 engine vs reference with norm on/scale 1.5 (wrong config): rel %.3e argmax agree %d/%d -> %s (must be large: the keys reach the graph)" % (rel, am, T, "sensitive" if sens3 else "INSENSITIVE"))
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
