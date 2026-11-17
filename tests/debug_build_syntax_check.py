#!/usr/bin/env python3
"""Debug-build guard: every src/ and common/ translation unit that carries an NDEBUG block must compile with NDEBUG UNDEFINED.

Why: release gates define NDEBUG, so code inside `#ifndef NDEBUG` is never compiled by them. On 2026-11-08 src/llama.cpp's
debug-only buffer report referenced a lambda from a closed scope and treated a vector as one buffer; every Debug build failed
and no gate saw it (row S69-FORK-DEBUG-BUILD-LLAMA-CPP-NDEBUG-BLOCK-DOES-NOT-COMPILE-2026-11-08-01).

Each selected file is checked twice with clang++ -fsyntax-only: -DNDEBUG (the release view, must pass -- if it fails, the
check itself is broken, so the run fails as a harness error) and -UNDEBUG (the debug view, the thing under test).
usage: debug_build_syntax_check.py <build-dir with src/llama-version.h> [clang++]
exit 0 = every file passes both views; 1 = a debug-view failure; 2 = harness/setup error.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2:
    sys.exit("usage: debug_build_syntax_check.py <build-dir> [clang++]")
BUILD = sys.argv[1]
CXX = sys.argv[2] if len(sys.argv) > 2 else "clang++"
gen = os.path.join(BUILD, "src")
if not os.path.isfile(os.path.join(gen, "llama-version.h")):
    print("SETUP: %s has no src/llama-version.h (configure the build first)" % BUILD)
    sys.exit(2)
INC = ["include", "src", "ggml/include", "ggml/src", "common", "vendor"]


def cxx_standard(root):
    """The C++ standard the BUILD uses, read from the top CMakeLists.txt `set(CMAKE_CXX_STANDARD N)`.
    MEASURED 2026-11-11 gate b1111b: a hard-coded -std=c++17 here made src/llama.cpp:2716 (std::map::contains, C++20,
    from the srvckpt port) fail the RELEASE view -> rc 2 setup failure, although the fork builds with
    CMAKE_CXX_STANDARD 20 (CMakeLists.txt:29) and the real build compiles it. The check must use the build's standard."""
    try:
        with open(os.path.join(root, "CMakeLists.txt"), encoding="utf-8", errors="replace") as fh:
            m = re.search(r"^\s*set\s*\(\s*CMAKE_CXX_STANDARD\s+(\d+)\s*\)", fh.read(), re.M)
    except OSError as e:
        print("SETUP: cannot read %s/CMakeLists.txt for CMAKE_CXX_STANDARD: %r" % (root, e))
        sys.exit(2)
    if not m:
        print("SETUP: no set(CMAKE_CXX_STANDARD N) in %s/CMakeLists.txt -- refusing to guess a standard" % root)
        sys.exit(2)
    return m.group(1)


STD = cxx_standard(ROOT)
FLAGS = ["-std=c++%s" % STD, "-fsyntax-only"] + ["-I" + os.path.join(ROOT, d) for d in INC] + ["-I" + gen]
print("standard: c++%s (from CMakeLists.txt)" % STD)
PAT = re.compile(r"^\s*#\s*(ifndef\s+NDEBUG|ifdef\s+NDEBUG|if\s+!\s*defined\s*\(\s*NDEBUG\s*\))", re.M)

files = []
for d in ("src", "common"):
    for dirpath, _dirs, names in os.walk(os.path.join(ROOT, d)):
        for n in sorted(names):
            if n.endswith(".cpp"):
                p = os.path.join(dirpath, n)
                with open(p, encoding="utf-8", errors="replace") as fh:
                    if PAT.search(fh.read()):
                        files.append(p)
if not files:   # a scan that finds nothing has measured nothing
    print("SETUP: no translation unit with an NDEBUG block found under src/ common/ (pattern or root wrong)")
    sys.exit(2)

debug_fail = setup_fail = 0
for p in files:
    rel = os.path.relpath(p, ROOT).replace("\\", "/")
    res = {}
    for view, d in (("release", "-DNDEBUG"), ("debug", "-UNDEBUG")):
        r = subprocess.run([CXX] + FLAGS + [d, p], capture_output=True, text=True)
        errs = [ln for ln in r.stderr.splitlines() if "error:" in ln]
        res[view] = (r.returncode, errs)
    rr, dr = res["release"], res["debug"]
    if rr[0] != 0:
        setup_fail += 1
        print("SETUP  %s: release view fails too (rc %d) -- include paths/flags wrong, not a debug defect: %s" %
              (rel, rr[0], rr[1][:2]))
    elif dr[0] != 0:
        debug_fail += 1
        print("FAIL   %s: compiles with NDEBUG, not without (%d error lines)" % (rel, len(dr[1])))
        for ln in dr[1][:6]:
            print("         " + ln[:240])
    else:
        print("ok     %s" % rel)
print("checked %d file(s) with an NDEBUG block: %d debug-view failure(s), %d setup failure(s)" %
      (len(files), debug_fail, setup_fail))
sys.exit(2 if setup_fail else (1 if debug_fail else 0))
