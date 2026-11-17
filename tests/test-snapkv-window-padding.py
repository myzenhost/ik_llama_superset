#!/usr/bin/env python3
"""Runtime test: SnapKV / H2O prompt prune must count the prompt's real KV cells, not the padded n_kv.

Background
  snapkv_eval_callback sizes lctx.snapkv_obs to n_kv = ne[0] of kq_soft_max_ext. On the non-flash path
  (which SnapKV / H2O force) n_kv is padded to a multiple of 32, so a prompt of L tokens with L % 32 != 0
  leaves cells [L, n_kv) empty in the capture. snapkv_keep_plan sizes its plan from obs.size(), and the
  H2O path was handed n_obs = obs.size(), so the "recent window" [n_obs - W, n_obs) protected up to 31
  empty cells instead of the last W real tokens, and the budget counted those empty cells as kept.

What it runs (llama-completion, CPU, two prompts whose token counts are not multiples of 32, -n 2)
  for each prompt: --snapkv-cap 64 --snapkv-window 16, and --h2o-heavy 32 --h2o-recent 16
Expectation: every arm logs "<algo> dropped N/M prompt KV positions" with M == the prompt token count
reported by "prompt eval time ... / L tokens", and M - N == the budget (SnapKV 64, H2O 48).
A binary without the fix reports M == L rounded up to 32 for the prompts with L % 32 != 0.

Usage: python tests/test-snapkv-window-padding.py --completion <llama-completion[.exe]> --model <small .gguf> [--threads 4]
Exit 0 = every expectation holds; the printed table is the record of what each arm did.
"""
import argparse
import re
import subprocess
import sys

BASE = ("The history of the common law of contracts in Indiana begins with the territorial courts, "
        "which borrowed English forms of action and adapted them to a frontier economy. "
        "Over the following decades the Supreme Court of Indiana refined doctrines of offer, acceptance, "
        "consideration, and remedy, and the legislature codified many of them. ")
PROMPTS = [BASE * 2 + "Breach must be pleaded.", BASE * 2 + "A litigant must plead the contract, the breach, and damages."]
ARMS = [("SnapKV", ["--snapkv-cap", "64", "--snapkv-window", "16"], 64),
        ("H2O", ["--h2o-heavy", "32", "--h2o-recent", "16"], 48)]


def run(exe, model, prompt, extra, threads):
    argv = [exe, "-m", model, "-p", prompt, "-n", "2", "-c", "512", "-t", str(threads), "-ngl", "0",
            "--temp", "0", "-no-cnv", "--no-warmup", "-fa", "off"] + extra
    p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=600)  # PROGRESS-OK: a 135M-model prefill of ~100 tokens on CPU takes seconds
    log = p.stdout + "\n" + p.stderr
    m = re.search(r"(SnapKV|H2O) dropped (\d+)/(\d+) prompt KV positions", log)
    n = re.search(r"prompt eval time.*?/\s*(\d+) tokens", log)
    return p.returncode, (m.group(1), int(m.group(2)), int(m.group(3))) if m else None, int(n.group(1)) if n else None, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--completion", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    fails, unpadded = [], 0
    print(f"{'prompt':>6} {'algo':>6} {'rc':>3} {'L':>5} {'dropped':>8} {'n_obs':>6} {'kept':>5} verdict")
    for pi, prompt in enumerate(PROMPTS):
        for algo, extra, budget in ARMS:
            rc, dropped, L, log = run(a.completion, a.model, prompt, extra, a.threads)
            if L is not None and L % 32 != 0:
                unpadded += 1
            ok = rc == 0 and dropped is not None and L is not None and dropped[0] == algo \
                and dropped[2] == L and dropped[2] - dropped[1] == budget
            d = dropped[1] if dropped else "-"
            o = dropped[2] if dropped else "-"
            k = (dropped[2] - dropped[1]) if dropped else "-"
            print(f"{pi:>6} {algo:>6} {rc:>3} {str(L):>5} {str(d):>8} {str(o):>6} {str(k):>5} {'PASS' if ok else 'FAIL'}")
            if not ok:
                fails.append((pi, algo))
                if dropped is None:
                    print("  --- log tail ---\n" + "\n".join(log.splitlines()[-25:]))
    if unpadded == 0:
        print("CONTROL FAIL: no prompt had a token count off a multiple of 32, so the test cannot see the defect")
        return 2
    print(f"arms with L % 32 != 0: {unpadded}; failures: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
