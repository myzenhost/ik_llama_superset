# REWRITE-SAFE: writes only to NEW files under temp/_agF_kvregress/probe_out/; reads no store.
# PROGRESS-OK: each subprocess here is ONE cl.exe run over ONE header (measured: a few
# seconds each), and the loop prints a flushed PASS/FAIL line plus writes
# probe_out/probe_results.json after EVERY probe, so a watcher sees a new line every few
# seconds. The 180 s timeout is a hang cap, not the expected duration -- it is not a
# blind wait, and if a probe ever did approach it the preceding lines already say which
# header is stuck.
"""Measured scope probe for the upstream KV-cache / graph-layer graft.

Instead of ASSERTING that "llama-graph.h is the least entangled piece", compile it --
and each of the other four named headers -- as a standalone translation unit against
OUR fork's include paths, with the exact flags build.ninja uses for the llama target.

Three variants per header, so the result distinguishes three different situations:
  bare        : #include "<header>"                       against our src/ only
  with_deps   : same, but upstream's src/ FIRST on the include path (so the header's
                own siblings resolve).  If this passes and `bare` fails, the header is
                self-consistent and the gap is purely the files it pulls in.
  our_first   : our src/ first, upstream second -- the realistic graft layout, where
                a header of the same NAME in our tree would win.

A header that only compiles with the whole upstream src/ ahead of ours is not a file
we can adopt; it is a tree we would be adopting.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000   # rule 1.45: windows stay hidden

CL = Path(__import__("os").environ.get("CL_EXE", "cl.exe"))
FORK = Path(".")
UP = Path("upstream/unsloth-llama.cpp/src")
OUT = Path("tmp/_agF_kvregress/probe_out")

HEADERS = [
    "llama-graph.h",
    "llama-kv-cache.h",
    "llama-kv-cache-iswa.h",
    "llama-memory-hybrid-iswa.h",
    "llama-memory-recurrent.h",
]

DEFINES = ["/DGGML_USE_CUDA", "/DLLAMA_BUILD", "/DLLAMA_SHARED",
           "/D_CRT_SECURE_NO_WARNINGS", "/Dllama_EXPORTS"]
FLAGS = ["/DWIN32", "/D_WINDOWS", "/W3", "/GR", "/EHsc", "/MD", "/O2", "/Ob2",
         "/DNDEBUG", "-std:c++20", "/utf-8", "/bigobj", "/c"]
OURS = [f"-I{FORK}/src", f"-I{FORK}/include", f"-I{FORK}/ggml/src", f"-I{FORK}/ggml/include"]
UPS = [f"-I{UP}", f"-I{UP.parent}/include", f"-I{UP.parent}/ggml/src",
       f"-I{UP.parent}/ggml/include"]

VARIANTS = {
    "bare": lambda: OURS,
    "with_deps": lambda: UPS + OURS,
    "our_first": lambda: OURS + UPS,
}

MISSING = re.compile(r"Cannot open include file: '([^']+)'")
UNDECL = re.compile(r"error C(\d+):")


def probe(header, variant):
    OUT.mkdir(parents=True, exist_ok=True)
    tu = OUT / f"tu_{header.replace('.', '_')}_{variant}.cpp"
    src = UP / header
    tu.write_text(f'#include "{src.as_posix()}"\nint main_probe_{variant}() {{ return 0; }}\n',
                  encoding="utf-8")
    cmd = [str(CL)] + DEFINES + FLAGS + VARIANTS[variant]() + \
          [f"/Fo{OUT}\\{tu.stem}.obj", str(tu)]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                       cwd=str(OUT), timeout=180, creationflags=CREATE_NO_WINDOW)
    log = (p.stdout or "") + (p.stderr or "")
    missing = sorted(set(MISSING.findall(log)))
    errs = sorted(set(UNDECL.findall(log)))
    return {
        "header": header, "variant": variant, "rc": p.returncode,
        "ok": p.returncode == 0,
        "missing_includes": missing,
        "error_codes": errs,
        "n_error_lines": len([l for l in log.splitlines() if ": error " in l]),
        "log_head": "\n".join([l for l in log.splitlines() if ": error " in l][:12]),
    }


def main():
    if not CL.exists():
        print(json.dumps({"fatal": f"compiler missing: {CL}"}))
        return 2
    results = []
    for h in HEADERS:
        for v in VARIANTS:
            r = probe(h, v)
            results.append(r)
            print(f"[{'PASS' if r['ok'] else 'FAIL'}] {h:32s} {v:10s} "
                  f"errs={r['n_error_lines']:<4} missing={r['missing_includes']}",
                  flush=True)
            with open(OUT / "probe_results.json", "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1)
    with open(OUT / "probe_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"[done] {OUT/'probe_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
