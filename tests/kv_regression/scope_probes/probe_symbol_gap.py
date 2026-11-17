# REWRITE-SAFE: writes only NEW files under temp/_agF_kvregress/gap_out/; reads no store.
# PROGRESS-OK: one cl.exe run per TU (seconds each), each printing a flushed line and
# rewriting gap_out/gap_results.json immediately. The 300 s timeout is a hang cap.
"""Stage-2 scope probe, part 3: what is the COMPLETE symbol bill?

probe_impl_layer.py showed 9 of 10 upstream TUs fail against our tree, and that the
error sites are in ggml.h / llama.h / llama-impl.h -- the shared foundations, not the
KV-cache files themselves.  That is suggestive but not yet a decision, because it was
measured from only the first 15 error lines per TU.

"It bottoms out in ggml" and "it needs ten shim symbols" are very different answers and
they look identical in a truncated log.  So this run captures EVERY error line and
extracts the full set of identifiers our tree does not provide.  If the bill is a
handful of enum values and a hint function, the graft is viable and should proceed; if
it is broad, it is not, and the number says so rather than an impression.
"""
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000   # rule 1.45

CL = Path(__import__("os").environ.get("CL_EXE", "cl.exe"))
FORK = Path(".")
UP = Path("upstream/unsloth-llama.cpp/src")
OUT = Path("tmp/_agF_kvregress/gap_out")

TUS = ["llama-graph.cpp", "llama-kv-cache.cpp", "llama-kv-cache-iswa.cpp",
       "llama-memory-hybrid-iswa.cpp", "llama-memory-recurrent.cpp",
       "llama-batch.cpp", "llama-memory.cpp", "llama-adapter.cpp",
       "llama-kv-cache-kpool.cpp"]

DEFINES = ["/DGGML_USE_CUDA", "/DLLAMA_BUILD", "/DLLAMA_SHARED",
           "/D_CRT_SECURE_NO_WARNINGS", "/Dllama_EXPORTS"]
# /W3 kept, but raise the error cap far above the default 100 so nothing is truncated
FLAGS = ["/DWIN32", "/D_WINDOWS", "/W3", "/GR", "/EHsc", "/MD", "/O2", "/Ob2",
         "/DNDEBUG", "-std:c++20", "/utf-8", "/bigobj", "/c", "/errorReport:none"]
OURS = [f"-I{FORK}/src", f"-I{FORK}/include", f"-I{FORK}/ggml/src", f"-I{FORK}/ggml/include"]
UPS = [f"-I{UP}", f"-I{UP.parent}/include", f"-I{UP.parent}/ggml/src",
       f"-I{UP.parent}/ggml/include"]

# MSVC diagnostics that name a symbol our tree does not provide
PATTERNS = [
    re.compile(r"error C2065: '([^']+)': undeclared identifier"),
    re.compile(r"error C3861: '([^']+)': identifier not found"),
    re.compile(r"error C2039: '([^']+)' *: is not a member"),
    re.compile(r"error C2027: use of undefined type '([^']+)'"),
    re.compile(r"error C2061: syntax error: identifier '([^']+)'"),
    re.compile(r"error C3646: '([^']+)': unknown override specifier"),
    re.compile(r"error C2143: syntax error: missing '[^']*' before '([^']+)'"),
]
SITE = re.compile(r"^(.*?)\((\d+)\)")


def run(tu):
    OUT.mkdir(parents=True, exist_ok=True)
    src = UP / tu
    cmd = [str(CL)] + DEFINES + FLAGS + OURS + UPS + \
          [f"/Fo{OUT}\\{Path(tu).stem}_gap.obj", str(src)]
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
            "symbols": sorted(syms), "sites": dict(sites)}


def main():
    results, allsyms, allsites = [], set(), Counter()
    bysym = defaultdict(set)
    for tu in TUS:
        r = run(tu)
        results.append(r)
        allsyms |= set(r["symbols"])
        allsites.update(r["sites"])
        for s in r["symbols"]:
            bysym[s].add(tu)
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] {tu:30s} errs={r['n_errors']:<4} "
              f"syms={len(r['symbols'])}", flush=True)
        with open(OUT / "gap_results.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    summary = {
        "total_distinct_missing_symbols": len(allsyms),
        "missing_symbols": sorted(allsyms),
        "error_sites_by_header": dict(allsites.most_common()),
        "tus_failing": [r["tu"] for r in results if not r["ok"]],
    }
    with open(OUT / "gap_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
