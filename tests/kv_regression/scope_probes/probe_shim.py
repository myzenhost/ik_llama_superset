# REWRITE-SAFE: writes only NEW files under temp/_agF_kvregress/shim_out/; reads no store.
# PROGRESS-OK: one cl.exe run per TU (seconds each), flushed result line + JSON rewritten
# after every probe. The 300 s timeout is a hang cap, not the expected duration.
"""Rung 3 of the invent ladder (rule 1.35): ADAPT, then measure.

The symbol-gap probe said the KV-cache/graph layer needs 17 identifiers our tree does
not provide.  Concluding "blocked" from that alone would be an impression.  Seventeen
is small, so the obvious adaptation is a compatibility shim -- and the honest thing is
to BUILD it and see what happens rather than assert what would happen.

agF_compat_shim.h declares all 17.  This probe force-includes it (/FI) ahead of each
upstream TU and recompiles.  The result is falsifiable either way:

  * errors go to zero            -> the graft is viable through a shim; verdict flips
  * a NEW, deeper error set      -> evidenced negative: the 17 were the surface, and
                                    the real gap is structural

Reported as a delta against the no-shim run, so "did the shim help at all" is a number.
"""
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000   # rule 1.45

CL = Path(__import__("os").environ.get("CL_EXE", "cl.exe"))
FORK = Path(".")
UP = Path("upstream/unsloth-llama.cpp/src")
BASE = Path("tmp/_agF_kvregress")
SHIM = BASE / "agF_compat_shim.h"
OUT = BASE / "shim_out"
PRIOR = BASE / "gap_out" / "gap_results.json"

TUS = ["llama-graph.cpp", "llama-kv-cache.cpp", "llama-kv-cache-iswa.cpp",
       "llama-memory-hybrid-iswa.cpp", "llama-memory-recurrent.cpp",
       "llama-batch.cpp", "llama-memory.cpp", "llama-adapter.cpp",
       "llama-kv-cache-kpool.cpp"]

DEFINES = ["/DGGML_USE_CUDA", "/DLLAMA_BUILD", "/DLLAMA_SHARED",
           "/D_CRT_SECURE_NO_WARNINGS", "/Dllama_EXPORTS"]
FLAGS = ["/DWIN32", "/D_WINDOWS", "/W3", "/GR", "/EHsc", "/MD", "/O2", "/Ob2",
         "/DNDEBUG", "-std:c++20", "/utf-8", "/bigobj", "/c", "/errorReport:none"]
OURS = [f"-I{FORK}/src", f"-I{FORK}/include", f"-I{FORK}/ggml/src", f"-I{FORK}/ggml/include"]
UPS = [f"-I{UP}", f"-I{UP.parent}/include", f"-I{UP.parent}/ggml/src",
       f"-I{UP.parent}/ggml/include"]

PATTERNS = [
    re.compile(r"error C2065: '([^']+)': undeclared identifier"),
    re.compile(r"error C3861: '([^']+)': identifier not found"),
    re.compile(r"error C2039: '([^']+)' *: is not a member"),
    re.compile(r"error C2027: use of undefined type '([^']+)'"),
    re.compile(r"error C2061: syntax error: identifier '([^']+)'"),
    re.compile(r"error C3646: '([^']+)': unknown override specifier"),
]
SITE = re.compile(r"^(.*?)\((\d+)\)")


def run(tu):
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [str(CL)] + DEFINES + FLAGS + [f"/FI{SHIM}"] + OURS + UPS + \
          [f"/Fo{OUT}\\{Path(tu).stem}_shim.obj", str(UP / tu)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           cwd=str(OUT), timeout=300, creationflags=CREATE_NO_WINDOW)
        rc, log = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        rc, log = -9, "<TIMEOUT>"
    (OUT / f"{Path(tu).stem}.errlog").write_text(log, encoding="utf-8", errors="replace")
    errs = [l.strip() for l in log.splitlines() if ": error " in l]
    syms, sites = set(), Counter()
    for line in errs:
        for pat in PATTERNS:
            m = pat.search(line)
            if m:
                syms.add(m.group(1))
        s = SITE.match(line)
        if s:
            sites[Path(s.group(1)).name] += 1
    return {"tu": tu, "rc": rc, "ok": rc == 0, "n_errors": len(errs),
            "symbols": sorted(syms), "sites": dict(sites),
            "sample": errs[:10]}


def main():
    prior = {r["tu"]: r for r in json.load(open(PRIOR, encoding="utf-8"))} \
        if PRIOR.exists() else {}
    results, allsyms, allsites = [], set(), Counter()
    for tu in TUS:
        r = run(tu)
        before = prior.get(tu, {}).get("n_errors")
        r["n_errors_before_shim"] = before
        r["delta"] = None if before is None else r["n_errors"] - before
        results.append(r)
        allsyms |= set(r["symbols"])
        allsites.update(r["sites"])
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {tu:30s} "
              f"errs {before} -> {r['n_errors']} (delta {r['delta']}) "
              f"syms={len(r['symbols'])}", flush=True)
        with open(OUT / "shim_results.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    tot_before = sum(r["n_errors_before_shim"] or 0 for r in results)
    tot_after = sum(r["n_errors"] for r in results)
    summary = {
        "verdict": "SHIM SUFFICIENT" if all(r["ok"] for r in results)
                   else "SHIM INSUFFICIENT",
        "tus_passing": [r["tu"] for r in results if r["ok"]],
        "total_errors_before_shim": tot_before,
        "total_errors_with_shim": tot_after,
        "errors_removed_by_shim": tot_before - tot_after,
        "remaining_distinct_symbols": len(allsyms),
        "remaining_symbols": sorted(allsyms),
        "remaining_sites": dict(allsites.most_common()),
    }
    with open(OUT / "shim_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
