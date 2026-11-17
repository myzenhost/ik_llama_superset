#!/usr/bin/env python3
"""Source scan: the MMA flash-attention combine loop must barrier after every k00 pass.

process_tile in fattn-new-mma.cu writes tile_Q at the top of each k00 pass and reads it in the
middle.  With np == 1 and nbatch_combine != DV/2 there are several passes, so a trailing barrier
that only runs for np > 1 lets a fast warp overwrite tile_Q while a slow warp still reads it.
Usage: test-fattn-combine-sync-scan.py [path-to-fattn-new-mma.cu]
"""
import re
import sys
from pathlib import Path

default = Path(__file__).resolve().parent.parent / "ggml/src/ggml-cuda/fattn-new-mma.cu"
src = Path(sys.argv[1] if len(sys.argv) > 1 else default).read_text(encoding="utf-8").replace("\r\n", "\n")

# the trailing barrier of the k00 loop: last `if (...) { __syncthreads(); }` before the #else
conds = re.findall(r"if \(([^\n]*)\) \{\n\s*__syncthreads\(\);\n\s*\}\n    \}\n#else\n    GGML_UNUSED\(Q_f2\)", src)
if not conds:
    print("FAIL: trailing barrier of the combine loop not found")
    sys.exit(2)
cond = conds[-1]
ok = "np > 1" in cond and "nbatch_combine != DV/2" in cond
print(("PASS" if ok else "FAIL") + ": trailing combine barrier condition = " + cond)
sys.exit(0 if ok else 1)
