#!/usr/bin/env python3
"""--swa-compress on gemma3: the flag must ENGAGE, SHRINK the KV cache, and leave greedy output unchanged.

Why this exists: gemma3 was left off llama_model::supports_swa_compress(), so --swa-compress
printed "--swa-compress is not implemented for this model; ignoring" and the KV cache stayed at
n_ctx rows on every layer (measured 2026-09-02: gemma3-1b 208.0 -> 208.0 MiB at 8192). gemma4
was added upstream in ik #2378 with a mask + store-head change in its graph builder; gemma3 needs
the same two changes in build_gemma3.cpp.

Each model is run twice with llama-completion, greedy, identical arguments except the flag:
  arm OFF : no --swa-compress
  arm ON  : --swa-compress
and the test asserts, per SWA model:
  1. both arms exit 0
  2. arm ON logs "--swa-compress engaged: ..." and NOT "not implemented for this model"
  3. arm OFF does NOT log "--swa-compress engaged"        (control: the flag is the cause)
  4. llama_kv_cache_init KV buffer size ON < OFF (real load, not the no_alloc fit probe)
  5. generated text ON == generated text OFF              (the prompt is longer than n_swa and is
     fed in ubatches smaller than n_swa, so the compacted ring wraps several times before decode)
NON-SWA control (optional, used when the model file exists): KV size and text identical in both
arms -- if the flag changed a dense model, the flag would not be doing what it claims.

Usage: test_swa_compress_gemma3.py --bin <dir with llama-completion(.exe)> [--model PATH ...]
       [--dense PATH] [--threads N]
Exit: 0 pass, 1 fail, 77 skip (no gemma3 model on this host).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# llama_kv_cache_init prints one buffer line per backend buffer; the fit probe prints the same
# line with "(no_alloc: projected...)" first, so only lines WITHOUT that suffix are the real load.
RE_KV = re.compile(r"llama_kv_cache_init:\s+\S+ KV buffer size =\s*([0-9.]+) MiB(?! \(no_alloc)")
RE_ENGAGED = re.compile(r"--swa-compress engaged: (\d+) of (\d+) layers")
RE_REFUSED = re.compile(r"--swa-compress (is not implemented for this model|had no effect)")
RE_NSWA = re.compile(r"n_swa\s*=\s*(\d+)")
RE_PROMPT_TOK = re.compile(r"prompt eval time =.*?/\s*(\d+) tokens")
RE_ENG_ROWS = re.compile(r"--swa-compress engaged: \d+ of \d+ layers at (\d+) rows")

DEFAULT_MODELS = [
    "models/gguf/gemma3-270m/gemma-3-270m-Q8_0.gguf",  # portability-ok: optional local test model, skipped when absent
    "models/gguf/gemma3-1b/gemma-3-1b-it-Q8_0.gguf",   # portability-ok: optional local test model, skipped when absent
]
DEFAULT_DENSE = "models/gguf/llama3.2-1b/Llama-3.2-1B-Instruct-Q8_0.gguf"  # portability-ok: optional non-SWA control


def make_prompt() -> str:
    # deterministic, ~1700 tokens for a gemma tokenizer: well past n_swa=512
    lines = [f"Entry {i}: the river at mile {i * 7} carries {i % 13} boats and {i % 5} barges." for i in range(110)]
    return "\n".join(lines) + "\nSummary of the entries above:"


def run_arm(exe: Path, model: str, prompt_file: Path, flag: bool, threads: int):
    argv = [str(exe), "-m", model, "-f", str(prompt_file), "-n", "48", "--temp", "0", "--seed", "1",
            "-c", "4096", "-b", "256", "-ub", "256", "-t", str(threads), "-ngl", "0",
            "--no-display-prompt", "-no-cnv"]
    if flag:
        argv.append("--swa-compress")
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    p = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900, creationflags=flags)
    kv = RE_KV.findall(p.stderr)
    nswa = RE_NSWA.findall(p.stderr)
    return {
        "rc": p.returncode,
        "kv": sum(float(x) for x in kv) if kv else None,
        "engaged": bool(RE_ENGAGED.search(p.stderr)),
        "refused": bool(RE_REFUSED.search(p.stderr)),
        "n_swa": int(nswa[0]) if nswa else None,
        "prompt_tok": int(RE_PROMPT_TOK.findall(p.stderr)[-1]) if RE_PROMPT_TOK.search(p.stderr) else None,
        "rows": int(RE_ENG_ROWS.findall(p.stderr)[-1]) if RE_ENG_ROWS.search(p.stderr) else None,
        "text": p.stdout,
        "stderr_tail": p.stderr[-1500:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", action="append")
    ap.add_argument("--dense", default=DEFAULT_DENSE)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    exe = Path(a.bin) / ("llama-completion.exe" if os.name == "nt" else "llama-completion")
    if not exe.exists():
        print(f"SKIP: {exe} not built")
        return 77
    env_models = [m for m in os.environ.get("LLAMA_TEST_GEMMA3_MODELS", "").split(os.pathsep) if m]
    models = [m for m in (a.model or env_models or DEFAULT_MODELS) if Path(m).exists()]
    if not models:
        print("SKIP: no gemma3 model on this host")
        return 77

    prompt_file = Path(os.environ.get("TEMP", "/tmp")) / "test_swa_compress_gemma3_prompt.txt"
    prompt_file.write_text(make_prompt(), encoding="utf-8")

    failures = 0

    def check(cond: bool, what: str) -> None:
        nonlocal failures
        print(("  ok   " if cond else "  FAIL ") + what)
        if not cond:
            failures += 1

    for m in models:
        print(f"== SWA model {Path(m).name}")
        off = run_arm(exe, m, prompt_file, False, a.threads)
        on = run_arm(exe, m, prompt_file, True, a.threads)
        print(f"     n_swa={off['n_swa']}  KV off={off['kv']} MiB  on={on['kv']} MiB  "
              f"engaged off/on={off['engaged']}/{on['engaged']}  refused on={on['refused']}")
        check(off["rc"] == 0 and on["rc"] == 0, f"both arms exit 0 (off={off['rc']}, on={on['rc']})")
        if off["rc"] != 0 or on["rc"] != 0:
            print(off["stderr_tail"] if off["rc"] else on["stderr_tail"])
            continue
        check(off["n_swa"] is not None and off["n_swa"] > 0, "model is SWA (n_swa > 0)")
        check(on["engaged"] and not on["refused"], "flag ENGAGES compaction (no 'not implemented' warning)")
        check(not off["engaged"], "control: no compaction without the flag")
        check(on["kv"] is not None and off["kv"] is not None and on["kv"] < off["kv"],
              "KV self size shrinks with the flag")
        check(len(off["text"].strip()) > 0, "control: the OFF arm generated text")
        # identity only means something if the compacted ring had to wrap: prompt longer than it
        print(f"     prompt tokens={on['prompt_tok']}  compacted rows={on['rows']}")
        check(on["prompt_tok"] is not None and on["rows"] is not None and on["prompt_tok"] > on["rows"],
              "control: prompt is longer than the compacted SWA rows (the ring wraps before decode)")
        check(on["text"] == off["text"], "greedy output identical with and without the flag")
        if on["text"] != off["text"]:
            print("     OFF:", repr(off["text"][:300]))
            print("     ON :", repr(on["text"][:300]))

    if a.dense and Path(a.dense).exists():
        print(f"== NON-SWA control {Path(a.dense).name}")
        off = run_arm(exe, a.dense, prompt_file, False, a.threads)
        on = run_arm(exe, a.dense, prompt_file, True, a.threads)
        print(f"     KV off={off['kv']} on={on['kv']}  engaged on={on['engaged']}")
        check(off["rc"] == 0 and on["rc"] == 0, "both arms exit 0")
        check(off["kv"] is not None, "control: the KV size line was parsed (a None == None pass would be vacuous)")
        check(off["kv"] == on["kv"] and not on["engaged"], "dense model: flag changes nothing in the cache")
        check(on["text"] == off["text"], "dense model: output identical")

    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} check(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
