#!/usr/bin/env python3
"""llama-embedding --embd-sort-by-length: longest-first packing (CTranslate2 rebatch_input, src/batch_reader.cc:171-228).

   python tests/test-embd-sort-by-length.py <llama-embedding> <model.gguf> [<baseline llama-embedding built before the change>]

Prompts alternate ~150 and ~110 tokens, context/batch 256: a long and a short prompt never fit one batch (~150+~110 > 256) but two short ones
do (2*~110 <= 256). In the caller's order every prompt closes the batch of the one before it: 12 decode batches. Longest first the
shorts pair up: 9. The test checks, from the program's own log line and its stdout:
  1. flag OFF: 12 decode batches, "as given"; and, with a baseline binary, stdout byte-identical to the build without the change;
  2. flag ON: 9 decode batches, "longest first" (the feature FIRED), and fewer batches than OFF;
  3. flag ON: every embedding equals the flag-OFF embedding of the SAME prompt within 1e-4 (order restored, values unchanged);
  4. CONTROL: the 12 embeddings are pairwise different, so a wrong order cannot pass (3);
  5. per-token output (--pooling none): the rows come back in the caller's order too.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

exe, model = sys.argv[1], sys.argv[2]
baseline = sys.argv[3] if len(sys.argv) > 3 else None
failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ((" -- " + detail) if (detail and not cond) else ""))
    if not cond:
        failures.append(name)


words = "the quick brown fox jumps over lazy dog and runs to big red house cat sun".split()
td = tempfile.TemporaryDirectory()
pf = Path(td.name) / "prompts.txt"
BASE = ["-m", model, "--pooling", "mean", "-c", "256", "-b", "256", "-ub", "256", "--embd-output-format", "raw", "--verbose-prompt"]


def token_counts(texts):
    pf.write_text("\n".join(texts))
    r = subprocess.run([exe, *BASE, "-f", str(pf)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    return [int(x) for x in re.findall(r"number of tokens in prompt = (\d+)", r.stderr)]


# a word is a varying number of tokens: measure each word's slope and offset, then size every prompt to an exact token count
t10, t20 = token_counts([" ".join([w] * 10) for w in words[:12]]), token_counts([" ".join([w] * 20) for w in words[:12]])
prompts = []
for i in range(12):
    slope, target = (t20[i] - t10[i]) / 10.0, (150 if i % 2 == 0 else 110)
    off = t10[i] - 10 * slope
    prompts.append(" ".join([words[i]] * max(1, round((target - off) / slope))))
lens = token_counts(prompts)
check("precondition: long + short does not fit 256, two shorts do", all(140 <= lens[i] <= 160 for i in range(0, 12, 2)) and
      all(100 <= lens[i] <= 120 for i in range(1, 12, 2)) and min(lens[0::2]) + min(lens[1::2]) > 256 and 2 * max(lens[1::2]) <= 256,
      str(lens))
pf.write_text("\n".join(prompts))


def run(binary, extra=(), pooling="mean"):
    r = subprocess.run([binary, "-m", model, "--pooling", pooling, "-f", str(pf), "-c", "256", "-b", "256", "-ub", "256", "--embd-output-format", "raw",
                        "--embd-normalize", "2", *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    m = re.search(r"(\d+) prompt\(s\) in (\d+) decode batch\(es\), packing order: ([a-z ]+)", r.stderr)
    rows = [[float(x) for x in ln.split()] for ln in r.stdout.strip().splitlines() if ln.strip()]
    return r.stdout, rows, (int(m.group(2)), m.group(3).strip()) if m else None


out_off, rows_off, info_off = run(exe)
out_on, rows_on, info_on = run(exe, ["--embd-sort-by-length"])
drops = sum(len(run(exe)[1]) != 12 for _ in range(12))
check("no output row is lost at exit (12 runs)", drops == 0, "%d of 12 runs lost rows" % drops)
check("flag OFF: 12 decode batches, packed as given", info_off == (12, "as given"), str(info_off))
check("flag ON: 9 decode batches, longest first (feature fired)", info_on == (9, "longest first"), str(info_on))
check("flag ON uses fewer decode batches than OFF", info_on is not None and info_off is not None and info_on[0] < info_off[0])
check("12 embeddings in both runs", len(rows_off) == 12 and len(rows_on) == 12, "%d %d" % (len(rows_off), len(rows_on)))
worst = max((abs(a - b) for ro, rn in zip(rows_off, rows_on) for a, b in zip(ro, rn)), default=9.0)
check("ON == OFF per prompt within 1e-4 (order restored)", worst < 1e-4, "worst=%g" % worst)
dist = min((sum((a - b) ** 2 for a, b in zip(rows_off[i], rows_off[j])) ** 0.5 for i in range(12) for j in range(i + 1, 12)), default=0)
check("CONTROL embeddings are pairwise different (a wrong order would show)", dist > 1e-3, "min dist=%g" % dist)
if baseline:
    # the unmodified binary sometimes exits with log lines still queued (rows missing): take its first complete answer
    for _ in range(10):
        out_base, rows_base, _ = run(baseline)
        if len(rows_base) == 12:
            break
    check("flag OFF output is byte-identical to the build without the change", out_base == out_off)
n_off = run(exe, pooling="none")
n_on = run(exe, ["--embd-sort-by-length"], pooling="none")
worst = max((abs(a - b) for ro, rn in zip(n_off[1], n_on[1]) for a, b in zip(ro, rn)), default=9.0)
check("pooling none: same rows in the caller's order (%d rows)" % len(n_on[1]), len(n_on[1]) == len(n_off[1]) > 100 and worst < 1e-4,
      "rows %d/%d worst=%g" % (len(n_on[1]), len(n_off[1]), worst))

print("RESULT: %d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
