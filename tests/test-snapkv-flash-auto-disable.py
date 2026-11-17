#!/usr/bin/env python3
"""Runtime test: --snapkv-cap / --h2o-heavy must take effect without the user also turning flash attention off.

Background
  SnapKV and H2O score KV positions from the post-softmax tensor kq_soft_max_ext-<il>. Flash attention
  fuses the softmax inside the kernel and never builds it, so the capture stays empty and the post-prefill
  eviction block (src/llama.cpp, "SnapKV dropped N/M prompt KV positions") is a silent no-op.
  common_context_params_to_llama now forces non-flash attention and logs why when either flag is set.

What it runs (llama-completion, CPU, 96+ token prompt, -n 4)
  A  --snapkv-cap 64                       (flash attention left at its default)
  B  --snapkv-cap 64 -fa on                (flash attention explicitly requested)
  C  --h2o-heavy 32 --h2o-recent 16        (flash attention left at its default)
  D  --h2o-heavy 32 --h2o-recent 16 -fa on
  E  no eviction flag, -fa on              (control: the new code must stay out of the way)
  F  --snapkv-cap 64 -fa off               (control: the pre-existing working configuration)
A binary without the port passes F and E and fails A..D in the cases where flash attention is on;
a binary with the port passes all six. Evidence is the log: a "dropped N/M" line from the eviction
block (the KV-cell count that actually went away) and the "disabling flash attention" line.

Usage: python tests/test-snapkv-flash-auto-disable.py --completion <llama-completion[.exe]> --model <small .gguf> [--threads 8]
Exit 0 = every expectation below holds; the printed table is the record of what each arm did.
"""
import argparse
import re
import subprocess
import sys

PROMPT = ("The history of the canal system in the region begins with the early surveyors,"
          "who borrowed older methods of mapping and adapted them to a frontier economy."
          "Over the following decades the regional council refined rules of freight, scheduling,"
          "inspection, and repair, and the engineers codified many of them."
          "Today a crew that plans a repair must record the condition of the canal, the lock gate,"
          "and resulting delays, and must do so with enough detail to pass a safety review.") * 2


def run(exe, model, extra, threads):
    argv = [exe, "-m", model, "-p", PROMPT, "-n", "4", "-c", "512", "-t", str(threads), "-ngl", "0",
            "--temp", "0", "-no-cnv", "--no-warmup"] + extra
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = p.stdout + "\n" + p.stderr
    m = re.search(r"(SnapKV|H2O) dropped (\d+)/(\d+) prompt KV positions", log)
    dropped = (m.group(1), int(m.group(2)), int(m.group(3))) if m else None
    forced = "disabling flash attention for this context" in log
    n_prompt = re.search(r"prompt eval time.*?/\s*(\d+) tokens", log)
    return p.returncode, dropped, forced, (int(n_prompt.group(1)) if n_prompt else None), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completion", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    arms = [
        # name, extra args, expect_evict (with the port), expect_forced_line (with the port)
        ("A snapkv default-fa", ["--snapkv-cap", "64"], True, "maybe"),
        ("B snapkv -fa on", ["--snapkv-cap", "64", "-fa", "on"], True, True),
        ("C h2o default-fa", ["--h2o-heavy", "32", "--h2o-recent", "16"], True, "maybe"),
        ("D h2o -fa on", ["--h2o-heavy", "32", "--h2o-recent", "16", "-fa", "on"], True, True),
        ("E no-flag -fa on", ["-fa", "on"], False, False),
        ("F snapkv -fa off", ["--snapkv-cap", "64", "-fa", "off"], True, False),
    ]
    ok = True
    print("%-22s %-3s %-22s %-8s %s" % ("arm", "rc", "eviction line", "forced", "prompt tokens"))
    for name, extra, want_evict, want_forced in arms:
        rc, dropped, forced, ntok, log = run(a.completion, a.model, extra, a.threads)
        ev = "%s dropped %d/%d" % dropped if dropped else "none"
        print("%-22s %-3d %-22s %-8s %s" % (name, rc, ev, "yes" if forced else "no", ntok))
        if rc != 0:
            tail = [ln for ln in log.splitlines() if ln.strip()][-3:]
            print("   FAIL: run failed: " + " | ".join(tail))
            ok = False
            continue
        if ntok is not None and ntok < 90:
            print("   FAIL: prompt only %d tokens; the arms need > 64 + window" % ntok)
            ok = False
        if want_evict and not dropped:
            print("   FAIL: eviction flag set but no 'dropped N/M' line -> eviction was a silent no-op")
            ok = False
        if (not want_evict) and dropped:
            print("   FAIL: eviction happened without an eviction flag")
            ok = False
        if want_forced is True and not forced:
            print("   FAIL: expected the 'disabling flash attention' line")
            ok = False
        if want_forced is False and forced:
            print("   FAIL: 'disabling flash attention' printed where it must not be")
            ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
