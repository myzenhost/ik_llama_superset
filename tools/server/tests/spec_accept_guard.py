#!/usr/bin/env python3
"""Speculative-decoding ACCEPTANCE-RATE regression guard (row S69-MNN-PICK-DFLASH-ACCEPTANCE-RATE-REFERENCE-GUARD-2026-10-05-01).

Why: verification is lossless, so a draft-side bug (a draft that lost most of its context, a wrong mask, a wrong position) still produces CORRECT
output and only shows up as a lower acceptance rate. MNN found exactly that: acceptance fell 34.38 percent -> 6.25 percent while the text looked fine
(MNN PR #4898, read by the row's reviewers). So the guard compares accepted/proposed on a fixed greedy prompt with a RECORDED reference and fails on a
drop, and it carries a MUTATION CONTROL: a deliberately crippled draft configuration must collapse acceptance by at least 2x, otherwise the guard
cannot fail and proves nothing.

  record  start llama-server with --server-args + --spec-args, run the prompt set greedily, store {accepted, proposed, rate} in --reference
  check   same run; fail if the rate is more than --tolerance points below the reference; then run --mutation-args and require
          rate_mutated <= rate_reference * --control-ratio (default 0.5, the control)

Everything about the draft is the caller's: pass the speculative flags verbatim (e.g. --spec-args "--spec-type dflash -md draft.gguf"). The reference must
come from a run that is known good (for DFlash: the upstream PyTorch implementation at F16, per the row) and quantised targets get their own references.
Example (self-speculative suffix automaton, the form this repo can run on any model):
  spec_accept_guard.py record --bin llama-server.exe --model m.gguf --spec-args="--spec-type suffix_automaton" --reference ref.json
  spec_accept_guard.py check  --bin llama-server.exe --model m.gguf --spec-args="--spec-type suffix_automaton" --reference ref.json \\
                              --mutation-args="--spec-type suffix_automaton:suffix_min_match_len=4000"
(use the --flag=value form: argparse reads a value that starts with "--" as a new option.)
Exit code 0 = guard passed (and the control collapsed), 1 = regression or the control did not collapse, 2 = usage / server error.
"""
# REWRITE-SAFE: writes only the NEW reference file named by --reference (record) and reads it in check; no existing store is modified.
import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. The cat",
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n\ndef div(a, b):\n    return",
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,",
]


def run_server(a, extra, port):
    log = open(a.log, "ab")
    cmd = [a.bin, "-m", a.model, "--port", str(port), "-t", str(a.threads), "-ngl", str(a.ngl), "-c", "4096", "--parallel", "1"] + shlex.split(a.server_args) + shlex.split(extra)
    log.write(("\n$ " + " ".join(cmd) + "\n").encode())
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000 if sys.platform == "win32" else 0)
    t0 = time.time()
    while time.time() - t0 < a.start_timeout:
        time.sleep(1)
        if p.poll() is not None:
            raise SystemExit(f"server exited rc={p.returncode} during start; see {a.log}")
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200:
                return p
        except Exception:
            pass
    p.terminate()
    raise SystemExit(f"server not healthy after {a.start_timeout}s; see {a.log}")


def measure(a, extra, port):
    p = run_server(a, extra, port)
    try:
        acc = prop = out = 0
        texts = []
        for i, pr in enumerate(PROMPTS):
            body = {"prompt": pr, "n_predict": a.n_predict, "temperature": 0.0, "seed": 1 + i, "cache_prompt": False}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=600))
            t = r["timings"]
            acc += int(t.get("draft_n_accepted", 0))
            prop += int(t.get("draft_n", 0))
            out += int(t.get("predicted_n", 0))
            texts.append(r["content"])
            print(f"[guard] prompt {i}: predicted {t.get('predicted_n')} draft_n {t.get('draft_n')} accepted {t.get('draft_n_accepted')}", flush=True)
        return {"accepted": acc, "proposed": prop, "predicted": out, "rate": (100.0 * acc / prop) if prop else 0.0, "texts": texts}
    finally:
        p.terminate()
        try:
            p.wait(30)
        except Exception:
            p.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec-args", required=True, help="speculative flags exactly as for llama-server")
    ap.add_argument("--server-args", default="")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--mutation-args", default="", help="a crippled draft configuration; required for check")
    ap.add_argument("--tolerance", type=float, default=3.0, help="allowed drop in acceptance, percentage points")
    ap.add_argument("--control-ratio", dest="control_ratio", type=float, default=0.5,
                    help="the mutated draft must score <= ratio * reference (default 0.5 = a 2x collapse; a milder mutation, such as a 2-bit twin of a 4-bit self-draft, needs a larger ratio and then only detects regressions at least that large)")
    ap.add_argument("--n-predict", dest="n_predict", type=int, default=96)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--port", type=int, default=18137)
    ap.add_argument("--log", default="spec_accept_guard.log")
    ap.add_argument("--start-timeout", dest="start_timeout", type=int, default=240)
    a = ap.parse_args()
    cur = measure(a, a.spec_args, a.port)
    print(f"[guard] acceptance {cur['accepted']}/{cur['proposed']} = {cur['rate']:.2f}%", flush=True)
    if cur["proposed"] == 0:
        print("[guard] FAIL: the draft proposed nothing, nothing to compare")
        sys.exit(1)
    if a.mode == "record":
        with open(a.reference, "w") as f:
            json.dump({"spec_args": a.spec_args, "model": a.model, "n_predict": a.n_predict, "rate": cur["rate"], "accepted": cur["accepted"], "proposed": cur["proposed"], "texts": cur["texts"]}, f, indent=1)
        print(f"[guard] recorded {a.reference}")
        sys.exit(0)
    ref = json.load(open(a.reference))
    ok = True
    drop = ref["rate"] - cur["rate"]
    print(f"[guard] reference {ref['rate']:.2f}%, drop {drop:+.2f} points, tolerance {a.tolerance}")
    if drop > a.tolerance:
        print("[guard] FAIL: acceptance fell below the reference by more than the tolerance")
        ok = False
    if cur["texts"] != ref["texts"]:
        print("[guard] NOTE: greedy output differs from the recorded text (lossless verification should keep it identical for the same target)")
    if not a.mutation_args:
        print("[guard] FAIL: --mutation-args is required in check mode (a guard without a control proves nothing)")
        sys.exit(2)
    mut = measure(a, a.mutation_args, a.port + 1)
    print(f"[guard] mutation control acceptance {mut['accepted']}/{mut['proposed']} = {mut['rate']:.2f}% (must be <= {ref['rate'] * a.control_ratio:.2f}%, ratio {a.control_ratio})")
    if mut["rate"] > ref["rate"] * a.control_ratio:
        print("[guard] FAIL: the mutated draft did not drop acceptance below the control ratio, so this guard cannot detect a draft bug of that size")
        ok = False
    print("[guard] PASS" if ok else "[guard] FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
