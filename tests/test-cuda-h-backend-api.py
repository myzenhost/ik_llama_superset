#!/usr/bin/env python3
"""ggml-cuda.h declarations must use GGML_BACKEND_API, not GGML_API.

The CUDA backend is its own DLL (GGML_BACKEND_BUILD, no GGML_BUILD), so GGML_API there
expands to dllimport on the defining TU. Row
S69-GGML-CUDA-H-13-OF-14-DECLS-CARRY-THE-WRONG-EXPORT-MACRO-2026-09-15-01.
Source scan, no GPU. usage: test-cuda-h-backend-api.py [source_root]
"""
import re, sys, os
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
path = os.path.join(root, "ggml", "include", "ggml-cuda.h")
bad, good = [], 0
for n, line in enumerate(open(path, encoding="utf-8"), 1):
    s = line.lstrip()
    if s.startswith("//"):
        continue
    if re.match(r"GGML_API\b", s):
        bad.append((n, line.strip()))
    elif re.match(r"GGML_BACKEND_API\b", s):
        good += 1
if good == 0:
    print("FAIL: no GGML_BACKEND_API declaration found (scan reach broken)")
    sys.exit(1)
for n, l in bad:
    print(f"FAIL ggml-cuda.h:{n}: {l[:90]}")
print(f"{good} GGML_BACKEND_API decls, {len(bad)} wrong GGML_API decls")
sys.exit(1 if bad else 0)
