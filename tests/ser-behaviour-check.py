"""-ser / --smart-expert-reduction must CHANGE the output (row S69-FLAG-WIRING-SER-NO-READER-2026-10-01-01).

Before 2026-10-01 the graph never read cparams.min_experts / thresh_experts (the ggml_top_k_thresh call in llm_build_moe_ffn was
commented out), so `-ser 1,0.99999` printed "ser = 1, 0.99999" in the load log and produced output BYTE-IDENTICAL to a run without
the flag. This check runs the tiny random LFM2-MoE (4 experts, 2 used per token, layers 1-2 routed; built by
tests/lfm2-forward-check/make_tiny_moe_vocab_gguf.py) through llama-completion on CPU, sampled (temp 1, seed 1, 60 tokens) so a small
change in the logits shows up, and requires:
  keep : -ser 4,1         == base   (min_experts 4 == n_expert, nothing is dropped -> the thresh path is bit-identical to plain top-k)
  drop : -ser 1,0.99999   != base   (the second expert of every MoE token is skipped -> the output MUST move)
Why "keep" is 4,1 and not "2,0.99" (n_expert_used is 2): llama-build-context.cpp treats a context's first eval of a lone BOS as a
warmup pass and routes it to ALL experts (n_expert_used = n_expert, :83). The tiny model samples <s> (the BOS) as its first token,
so the 2nd decode IS a lone-BOS "warmup" graph, 4 experts wide; there SER's -1 ids (positions 2,3) land inside the view and drop
real experts, so -ser 2,0.99 / 3,0.99 legitimately differ from base (MEASURED 2026-10-01 with llama-eval-callback: the prompt pass is
identical, the 4-wide decode graph zeroes the dropped expert's weight and renormalises). That is SER working, not a bug.
  and every run exits 0 with no GGML_ASSERT (CPU get_rows must accept the -1 ids SER produces).
Usage: python tests/ser-behaviour-check.py <llama-completion.exe> <moe.gguf>      exit 0 PASS / 1 FAIL
CPU only (-ngl 0), a 128 KB model, four runs of a second each; run it under the host slot like any binary.
GATE-CLAIM-OK: test driver. SINGLE-SHOT-OK: four short runs.
# PROGRESS-OK: four runs of a 128 KB model; each prints its own result line as it finishes
# REWRITE-SAFE: writes nothing.
"""
import subprocess
import sys

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)
exe, model = sys.argv[1], sys.argv[2]
flags = 0x08000000 if sys.platform == "win32" else 0     # CREATE_NO_WINDOW
kw = {}
if sys.platform == "win32":
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0
    kw = dict(creationflags=flags, startupinfo=si)
common = ["-m", model, "-ngl", "0", "-t", "4", "--temp", "1.0", "--top-k", "0", "--top-p", "1.0", "--min-p", "0", "--ignore-eos",
          "--seed", "1", "-n", "60", "-c", "128", "-b", "32", "-ub", "32", "-no-cnv", "--no-warmup", "--no-display-prompt",
          "-p", "<s><s><s><s><s><s>"]
runs = {"base": [], "keep": ["-ser", "4,1"], "drop": ["-ser", "1,0.99999"]}
res = {}
for name, extra in runs.items():
    p = subprocess.run([exe] + common + extra, capture_output=True, timeout=300, **kw)
    out, err = p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")
    res[name] = dict(rc=p.returncode, out=out, assert_=[l for l in err.splitlines() if "ASSERT" in l][:2],
                     ser=[l.strip() for l in err.splitlines() if "ser = " in l][:1])
    print("%-5s rc=%d len=%d %s %s" % (name, p.returncode, len(out), res[name]["ser"], res[name]["assert_"]), flush=True)
    print("      %r" % out[:120], flush=True)

ok = all(r["rc"] == 0 and not r["assert_"] and len(r["out"]) > 20 for r in res.values())
keep_same = res["keep"]["out"] == res["base"]["out"]
drop_moved = res["drop"]["out"] != res["base"]["out"]
print("runs ok: %s | -ser 4,1 == base: %s (want True) | -ser 1,0.99999 != base: %s (want True)" % (ok, keep_same, drop_moved))
verdict = ok and keep_same and drop_moved
print("SER-CHECK %s" % ("PASS" if verdict else "FAIL"))
sys.exit(0 if verdict else 1)
