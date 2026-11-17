#!/usr/bin/env python3
"""The dynamic n-gram lookup cache (-lcd) must be written back at exit (upstream 83c2d012a).

Before the port the fork loaded -lcd once and never wrote it, and a missing -lcd file aborted the run ("Couldn't read dynamic
lookup cache"), so the dynamic cache could never be created by generation. Arms (llama-speculative-simple --spec-type
ngram_cache, CPU, greedy):
  NEW      -lcd <missing file>        -> exit 0, the file now exists, the log says it started empty and saved it
  RELOAD   the same -lcd again        -> exit 0, no "does not exist" warning, the file is loaded with > 0 n-grams and saved again
  BADPATH  -lcd <an existing directory> -> refused (non-zero exit), the directory is untouched (never overwritten)
  CONTROL  no -lcd                    -> no cache file is written
Binary: first argument (bin dir) or LLAMA_SERVER_BIN's directory. Model: LLAMA_TEST_NGC_MODEL (default: the Qwen3-8B Q4_K_M
gguf in the ollama store). Exit 77 = skipped (no binary or no model).
Row: S69-ENGREV-UPSTREAM-MERGE-SPECULATIVE-PORT-LIST-2026-11-04-01.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP = 77
# portability-ok: default fixture path on this host; LLAMA_TEST_NGC_MODEL overrides it
DEFAULT_MODEL = "models/ollama/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
PROMPT = ("def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n\n"
          "def fibonacci_memo(n, memo={}):\n    if n < 2:\n        return n\n")


def find_bin():
    cands = []
    if len(sys.argv) > 1:
        cands.append(Path(sys.argv[1]))
    if os.environ.get("LLAMA_SERVER_BIN"):
        cands.append(Path(os.environ["LLAMA_SERVER_BIN"]).parent)
    cands.append(Path("bin"))
    for c in cands:
        for name in ("llama-speculative-simple.exe", "llama-speculative-simple"):
            if (c / name).is_file():
                return c / name
    return None


def run(exe, model, extra, log_path, threads):
    argv = [str(exe), "-m", str(model), "--spec-type", "ngram_cache", "--spec-draft-n-max", "8", "-t", str(threads),
            "-c", "1024", "--temp", "0", "-n", "32", "-ngl", "0", "-p", PROMPT] + extra
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, timeout=1800).returncode
    return rc, Path(log_path).read_text(encoding="utf-8", errors="replace")


def main():
    exe = find_bin()
    model = Path(os.environ.get("LLAMA_TEST_NGC_MODEL", DEFAULT_MODEL))
    if exe is None or not model.is_file():
        print("SKIP: exe=%s model=%s" % (exe, model))
        return SKIP
    threads = int(os.environ.get("LLAMA_TEST_THREADS", "8"))
    work = Path(tempfile.mkdtemp(prefix="ngc_dyn_"))
    print("exe  : %s\nmodel: %s\nwork : %s" % (exe, model, work))
    fails = 0

    def check(ok, label):
        nonlocal fails
        print("%s %s" % ("PASS" if ok else "FAIL", label))
        fails += 0 if ok else 1

    dyn = work / "dynamic.bin"
    rc, log = run(exe, model, ["-lcd", str(dyn)], work / "new.log", threads)
    check(rc == 0, "NEW: exits 0 with a missing -lcd file (rc %d)" % rc)
    check(dyn.is_file() and dyn.stat().st_size > 0, "NEW: the dynamic cache file was written at exit (%s)" %
          (dyn.stat().st_size if dyn.is_file() else "absent"))
    check("does not exist" in log, "NEW: the log says the cache started empty")
    check("saved dynamic lookup cache" in log, "NEW: the log reports the save")
    size1 = dyn.stat().st_size if dyn.is_file() else 0

    rc, log = run(exe, model, ["-lcd", str(dyn)], work / "reload.log", threads)
    m = re.search(r"dynamic cache \S+ \((\d+) n-grams\)", log)
    loaded = int(m.group(1)) if m else -1
    check(rc == 0, "RELOAD: exits 0 (rc %d)" % rc)
    check("does not exist" not in log, "RELOAD: no 'does not exist' warning")
    check(loaded > 0, "RELOAD: the saved cache is loaded with > 0 n-grams (%d)" % loaded)
    check(dyn.is_file() and dyn.stat().st_size >= size1, "RELOAD: saved again, not smaller (%d -> %d)" %
          (size1, dyn.stat().st_size if dyn.is_file() else 0))

    baddir = work / "a_directory"
    baddir.mkdir()
    (baddir / "keep.txt").write_text("x")
    rc, log = run(exe, model, ["-lcd", str(baddir)], work / "badpath.log", threads)
    check(rc != 0, "BADPATH: an existing but unreadable -lcd is refused (rc %d)" % rc)
    check(baddir.is_dir() and (baddir / "keep.txt").is_file(), "BADPATH: the existing path is untouched")

    before = sorted(p.name for p in work.iterdir())
    rc, log = run(exe, model, [], work / "control.log", threads)
    after = sorted(p.name for p in work.iterdir() if p.suffix != ".log")
    check(rc == 0 and "saved dynamic lookup cache" not in log, "CONTROL: no -lcd -> nothing saved (rc %d)" % rc)
    check(set(after) <= set(before), "CONTROL: no new cache file in the work dir")

    print("RESULT: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
