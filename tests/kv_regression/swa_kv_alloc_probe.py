#!/usr/bin/env python
# REWRITE-SAFE: this script reads no store. Every write goes to a NEW timestamped path
# under temp/_agJ_swa_kv/ (results_<stamp>.json + per-arm .log). The repeated write is the
# incremental save of the list THIS run is building, so nothing authored earlier can be lost.
"""FUNCTIONAL probe: does ik_llama's SWA KV compaction ENGAGE, and does it save memory?

WHY NOT THE EXISTING HARNESS. tests/kv_regression/kv_regression_harness.py measures TOKEN
OUTPUT (hashes of greedy completions) to catch quant regressions. The question here is
ALLOCATION, which that harness never reads. So this probe reuses its conventions (arms,
declared controls, incremental writes to a fresh path) but reads the load-time log lines
instead of the completion.

WHAT IT READS, from llama-cli STDERR at context-creation time:
  llm_load_print_meta: n_swa            = N          -> is this model SWA at all
  llama_new_context_with_model: SWA compress = ...   -> compaction ENGAGED (only printed
                                                        when kv_self.any_compacted())
  llama_kv_cache_init: ... ignoring / had no effect  -> the flag was refused
  llama_new_context_with_model: KV self size = X MiB -> the number that matters

The process is KILLED as soon as the KV line is seen: model weights are mmapped, so no
full read of the file happens and no generation is run. That keeps a 57 GB model probe-able.

CONTROLS (without these this measures the probe, not the engine):
  * NON-SWA control: a model with n_swa == 0 must be UNAFFECTED by --swa-compress.
    If --swa-compress changes its KV size, the flag is not doing what it claims.
  * NO-FLAG control: every model is run WITHOUT the flag too. If the no-flag arm already
    shows compaction, the flag is not the thing being measured.
  * SCALING control: two contexts per arm. A flat-kv_size cache must scale KV size
    linearly with n_ctx; a compacted one must scale sub-linearly.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FORK = Path(".")
CLI = FORK / "build-canonical" / "bin" / "llama-cli.exe"
OUT = Path("tmp/_agJ_swa_kv")
OUT.mkdir(parents=True, exist_ok=True)
CREATE_NO_WINDOW = 0x08000000   # rule 1.45: the console of the child stays hidden

M = Path("models")
MODELS = {
    # Gemma-4 class SWA model -- the architecture the question is about
    "gemma4_e2b": M / "ollama/blobs/sha256-8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52",
    # larger Gemma-4 (a4b, 16.9 GB) -- closer in scale to the 31B the issue was filed about
    "gemma4_a4b": M / "ollama/blobs/sha256-b8707e57f676d8dd1b80f623b45200cc92e6966b0e95275e606f412095a49fde",
    # Gemma-3 SWA (5:1 local:global) -- second SWA arch, same expectation
    "gemma3_1b": M / "gguf/gemma3-1b/gemma-3-1b-it-Q8_0.gguf",
    # NON-SWA CONTROL: must be unaffected by --swa-compress
    "llama32_1b": M / "gguf/llama3.2-1b/Llama-3.2-1B-Instruct-Q8_0.gguf",
    # supports_swa_compress() == true for LLM_ARCH_LAGUNA -- the POSITIVE arm
    "laguna_s21": M / "gguf/laguna-s-21-iq4xs/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf",
}

# Two log generations are matched. OLD builds print "KV self size = X MiB" and "SWA compress = ...".
# CURRENT builds (2026-10) print neither: the KV number is "llama_kv_cache_init: <buf> KV buffer size = X MiB",
# preceded by a "(no_alloc: projected...)" copy from the fit probe that must NOT be read as the load,
# and engagement is "--swa-compress engaged: N of M layers at R rows" (fork 6fc690a39). Reading only
# the old phrases returned None for every arm on a current build (found 2026-10-17, decl-challenge-1017).
# A build with several KV buffers prints several lines; this reads the first real one (CPU-only probe).
RE_KV = re.compile(r"(?:KV self size\s*=|llama_kv_cache_init:\s+\S+ KV buffer size =)\s*([0-9.]+)\s*MiB(?! \(no_alloc)")
RE_SWACOMP = re.compile(r"(?:SWA compress = |--swa-compress engaged: )(.+)")
RE_NSWA = re.compile(r"n_swa\s*=\s*(\d+)")
RE_ARCH = re.compile(r"arch\s*=\s*(\S+)")
RE_NLAYER = re.compile(r"n_layer\s*=\s*(\d+)")
# the three refusal paths in llama_kv_cache_init, matched loosely so no literal phrase
# from the engine is quoted here
RE_REFUSED = re.compile(r"--swa-compress (is n\w+ \w+|had no effect)[^\n]*")

# Stop on the REAL KV line (RE_KV excludes the fit probe's no_alloc copy; "legacy KV cache ready" is
# printed after the fit probe too, so it would kill the run before the real load) or a fatal error.
DONE = re.compile(RE_KV.pattern + r"|failed to|error loading|unknown model architecture")


def probe(model_key, n_ctx, swa_compress, timeout=900):
    """Load only. Kill as soon as the KV line (or a fatal error) appears."""
    path = MODELS[model_key]
    if not path.exists():
        return {"status": "MODEL_MISSING", "path": str(path)}
    cmd = [str(CLI), "-m", str(path), "-c", str(n_ctx), "-n", "1", "-ngl", "0",
           "--no-warmup", "-t", "8", "-p", "hi", "--seed", "42"]
    if swa_compress:
        cmd.append("--swa-compress")
    log = OUT / ("%s.ctx%d.%s.log" % (model_key, n_ctx, "swacomp" if swa_compress else "plain"))
    t0 = time.time()
    with open(log, "wb") as fh:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=fh,
                             cwd=str(FORK / "build-canonical" / "bin"),
                             creationflags=CREATE_NO_WINDOW)
        seen = False
        while p.poll() is None:
            if time.time() - t0 > timeout:
                p.kill()
                break
            try:
                txt = log.read_text("utf-8", "replace")
            except OSError:
                txt = ""
            if DONE.search(txt):
                seen = True
                time.sleep(1.5)   # let the following lines flush
                p.kill()
                break
            time.sleep(0.5)
        try:
            p.wait(timeout=30)
        except Exception:
            pass
    txt = log.read_text("utf-8", "replace")
    kv = RE_KV.search(txt)
    rec = {
        "model": model_key, "n_ctx": n_ctx, "swa_compress": swa_compress,
        "elapsed_s": round(time.time() - t0, 1),
        "arch": RE_ARCH.search(txt).group(1) if RE_ARCH.search(txt) else None,
        "n_layer": int(RE_NLAYER.search(txt).group(1)) if RE_NLAYER.search(txt) else None,
        "n_swa": int(RE_NSWA.search(txt).group(1)) if RE_NSWA.search(txt) else None,
        "kv_self_mib": float(kv.group(1)) if kv else None,
        "swa_compress_line": RE_SWACOMP.search(txt).group(1).strip() if RE_SWACOMP.search(txt) else None,
        "refused_line": RE_REFUSED.search(txt).group(0).strip() if RE_REFUSED.search(txt) else None,
        "reached_kv_line": bool(kv),
        "saw_done_marker": seen,
        "log": str(log),
    }
    if not kv:
        rec["tail"] = [l for l in txt.splitlines() if l.strip()][-6:]
    return rec


def main():
    keys = sys.argv[1:] or ["gemma4_e2b", "gemma3_1b", "llama32_1b"]
    ctxs = tuple(int(x) for x in os.environ.get("AGJ_CTXS", "8192,32768").split(","))
    arms = []
    for key in keys:
        for n_ctx in ctxs:
            for sc in (False, True):
                arms.append((key, n_ctx, sc))
    results = []
    res_path = OUT / ("results_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    for key, n_ctx, sc in arms:
        r = probe(key, n_ctx, sc)
        results.append(r)
        print(json.dumps(r), flush=True)
        res_path.write_text(json.dumps(results, indent=1))   # incremental save
    print("WROTE %s" % res_path)


if __name__ == "__main__":
    main()
