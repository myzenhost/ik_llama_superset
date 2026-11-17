# REWRITE-SAFE: writes only NEW files under temp/_agF_kvregress/probe_impl_out/; reads no store.
# PROGRESS-OK: one cl.exe run per translation unit (seconds each), each printing a flushed
# result line and rewriting probe_impl_results.json immediately, so a watcher sees progress
# continuously. The 300 s timeout is a hang cap, not the expected duration.
"""Stage-2 scope probe: can the KV-cache / graph layer be COMPILED, not just parsed?

The header probe (probe_graph_layer.py) already showed all five named headers fail
against our tree and pass only with the whole upstream src/ ahead on the include path.
But a header that parses is not a port.  The classes are pure-virtual interfaces plus
declarations; nothing runs until the .cpp files compile and link.

So this probe compiles the IMPLEMENTATION files, in the only configuration that has any
chance of working -- upstream's own src/ first -- and then, for each one, asks the
question that actually decides scope:

    when it is compiled with OUR tree first (the realistic graft layout), what breaks?

It also records, per TU, whether it PASSES with upstream-first.  A file that passes
upstream-first and fails ours-first is a file whose dependencies we would have to bring
across; a file that fails even upstream-first is one whose dependencies are not in the
five-file list at all.
"""
import json
import subprocess
import sys
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000   # rule 1.45: windows stay hidden

CL = Path(__import__("os").environ.get("CL_EXE", "cl.exe"))
FORK = Path(".")
UP = Path("upstream/unsloth-llama.cpp/src")
OUT = Path("tmp/_agF_kvregress/probe_impl_out")

# the implementations behind the five named headers, plus the files they pull in
TUS = [
    "llama-graph.cpp",
    "llama-kv-cache.cpp",
    "llama-kv-cache-iswa.cpp",
    "llama-memory-hybrid-iswa.cpp",
    "llama-memory-recurrent.cpp",
    # transitive: named by the headers' own #includes, NOT by the port rows
    "llama-batch.cpp",
    "llama-memory.cpp",
    "llama-adapter.cpp",
    "llama-io.cpp",
    # the consumer the whole layer exists to serve
    "llama-kv-cache-kpool.cpp",
]

DEFINES = ["/DGGML_USE_CUDA", "/DLLAMA_BUILD", "/DLLAMA_SHARED",
           "/D_CRT_SECURE_NO_WARNINGS", "/Dllama_EXPORTS"]
FLAGS = ["/DWIN32", "/D_WINDOWS", "/W3", "/GR", "/EHsc", "/MD", "/O2", "/Ob2",
         "/DNDEBUG", "-std:c++20", "/utf-8", "/bigobj", "/c"]
OURS = [f"-I{FORK}/src", f"-I{FORK}/include", f"-I{FORK}/ggml/src", f"-I{FORK}/ggml/include"]
UPS = [f"-I{UP}", f"-I{UP.parent}/include", f"-I{UP.parent}/ggml/src",
       f"-I{UP.parent}/ggml/include"]

VARIANTS = {
    "upstream_first": UPS + OURS,   # best case: upstream's own world
    "our_first": OURS + UPS,        # the realistic graft layout
}


def probe(tu_name, variant):
    OUT.mkdir(parents=True, exist_ok=True)
    src = UP / tu_name
    if not src.exists():
        return {"tu": tu_name, "variant": variant, "ok": False, "rc": -1,
                "note": "source missing", "n_error_lines": 0, "errors": []}
    cmd = [str(CL)] + DEFINES + FLAGS + VARIANTS[variant] + \
          [f"/Fo{OUT}\\{Path(tu_name).stem}_{variant}.obj", str(src)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           cwd=str(OUT), timeout=300, creationflags=CREATE_NO_WINDOW)
        rc, log = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        rc, log = -9, "<TIMEOUT>"
    errlines = [l.strip() for l in log.splitlines() if ": error " in l]
    return {"tu": tu_name, "variant": variant, "rc": rc, "ok": rc == 0,
            "n_error_lines": len(errlines),
            "errors": errlines[:15]}


def main():
    if not CL.exists():
        print(json.dumps({"fatal": f"compiler missing: {CL}"}))
        return 2
    results = []
    for tu in TUS:
        for v in VARIANTS:
            r = probe(tu, v)
            results.append(r)
            print(f"[{'PASS' if r['ok'] else 'FAIL'}] {tu:30s} {v:15s} "
                  f"errs={r['n_error_lines']}", flush=True)
            with open(OUT / "probe_impl_results.json", "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1)
    up_ok = [r["tu"] for r in results if r["variant"] == "upstream_first" and r["ok"]]
    our_ok = [r["tu"] for r in results if r["variant"] == "our_first" and r["ok"]]
    summary = {
        "tus": len(TUS),
        "compile_in_upstream_world": sorted(up_ok),
        "compile_against_our_tree": sorted(our_ok),
        "graftable_without_bringing_deps": sorted(our_ok),
    }
    with open(OUT / "probe_impl_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
