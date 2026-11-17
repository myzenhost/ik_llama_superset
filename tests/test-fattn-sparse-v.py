#!/usr/bin/env python3
# Test for the opt-in CUDA sparse-V skip in the vec flash-attention kernels (GGML_CUDA_FA_SPARSE_V).
#
# usage: test-fattn-sparse-v.py <bin-dir> <model.gguf> <out-dir>
#
# Checks (exit 1 if any FAILS):
#   1. env unset: test-backend-ops -o FLASH_ATTN_EXT -b CUDA0 passes and prints no "sparse V enabled" line
#      (the kernels must be unchanged when the switch is off).
#   2. env "1": the threshold line is printed with 0.001 (fails on a build without the feature).
#   3. env "2" (out of range): a warning is printed and the feature stays off.
# Reported, not gated:
#   4. test-backend-ops FLASH_ATTN_EXT pass/fail counts with the skip on. NOTE the reach: test-backend-ops has no
#      turbo cases (the CPU reference has no turbo attention) and the vec kernels serve only turbo K/V decode on an
#      mma-capable card, so this mostly shows the switch does not disturb the other kernels. The skip's accuracy on
#      turbo is measured by test-turbo-fa CUDA0 with the env set, and by a KLD run against f16-KV logits.
#   5. llama-bench token generation at depth 0 and 32768, q8_0 and turbo3 V, off vs on.
# Each child writes straight into its own log file, so a watcher can tail it while it runs.
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BIN = Path(sys.argv[1])
MODEL = sys.argv[2]
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
EXE = ".exe" if os.name == "nt" else ""
NOWIN = 0x08000000 if os.name == "nt" else 0

LINE_ON = re.compile(r"sparse V enabled in the CUDA vec flash-attention kernels, threshold ([0-9.e+-]+)")
LINE_WARN = "is not a threshold in (0,1); sparse V stays off"


def run(name, argv, sparse, timeout):
    env = dict(os.environ)
    env.pop("GGML_CUDA_FA_SPARSE_V", None)
    if sparse is not None:
        env["GGML_CUDA_FA_SPARSE_V"] = sparse
    log = OUT / f"{name}.log"
    print(f"start {name} -> {log}", flush=True)
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout, creationflags=NOWIN)
    return p.returncode, log.read_text(encoding="utf-8", errors="replace")


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def tbo_counts(text):
    # test-backend-ops colours each verdict ("\x1b[1;32mOK\x1b[0m"); a \bOK\b search finds nothing there because
    # "mOK" has no word boundary, which made the first version of this test count 0 OK and 0 FAIL on a 2371-case run.
    plain = ANSI.sub("", text)
    ok = len(re.findall(r": OK$", plain, re.M))
    fail = len(re.findall(r": FAIL$", plain, re.M))
    return ok, fail


# The counter must see a coloured verdict, or a "fail == 0" gate passes on a run it cannot read.
assert tbo_counts("  FLASH_ATTN_EXT(hsk=64): \x1b[1;32mOK\x1b[0m\n  X: \x1b[1;31mFAIL\x1b[0m\n") == (1, 1)


res = {}
bad = 0
tbo = [str(BIN / f"test-backend-ops{EXE}"), "-o", "FLASH_ATTN_EXT", "-b", "CUDA0"]

rc, text = run("tbo_off", tbo, None, 3600)
ok, fail = tbo_counts(text)
c1 = rc == 0 and ok > 0 and fail == 0 and LINE_ON.search(text) is None
res["1_off_unchanged"] = {"rc": rc, "ok": ok, "fail": fail, "line": LINE_ON.search(text) is not None, "pass": c1}
bad |= not c1

rc, text = run("tbo_on", tbo, "1", 3600)
m = LINE_ON.search(text)
c2 = m is not None and abs(float(m.group(1)) - 1e-3) < 1e-9
ok, fail = tbo_counts(text)
res["2_on_line"] = {"rc": rc, "threshold": m.group(1) if m else None, "pass": c2}
res["4_on_tbo_report"] = {"rc": rc, "ok": ok, "fail": fail}
bad |= not c2

rc, text = run("tbo_bad", tbo + ["-p", "hsk=64"], "2", 3600)
c3 = LINE_WARN in text and LINE_ON.search(text) is None
res["3_bad_value_off"] = {"rc": rc, "warn": LINE_WARN in text, "pass": c3}
bad |= not c3

bench = {}
for vt in ("q8_0", "turbo3"):
    for sp in (None, "1"):
        name = f"bench_{vt}_{'on' if sp else 'off'}"
        argv = [str(BIN / f"llama-bench{EXE}"), "-m", MODEL, "-ngl", "99", "-fa", "1", "-ctk", "q8_0", "-ctv", vt,
                "-p", "0", "-n", "64", "-d", "0,32768", "-r", "3", "-o", "json"]
        rc, text = run(name, argv, sp, 3600)
        rows = []
        try:
            js = json.loads(text[text.index("["):text.rindex("]") + 1])
            rows = [{"depth": r.get("n_depth"), "tg_ts": r.get("avg_ts"), "tg_sd": r.get("stddev_ts")} for r in js]
        except ValueError:
            pass
        bench[name] = {"rc": rc, "rows": rows}
        print(name, bench[name], flush=True)
res["5_bench_report"] = bench

for k, v in res.items():
    print(k, v, flush=True)
(OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("PASS" if not bad else "FAIL", flush=True)
sys.exit(1 if bad else 0)
