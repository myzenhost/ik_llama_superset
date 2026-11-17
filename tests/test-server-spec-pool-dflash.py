#!/usr/bin/env python3
"""DFlash must draft in the per-sequence speculator that llama-server (n_seq = n_parallel) and llama-speculative-simple (n_seq = 1) build.

Row S69-ENGREV-UPSTREAM-MERGE-SPECULATIVE-PORT-LIST-2026-11-04-01 (approved 2026-11-05). Before the pooled impl
(common_speculative_impl_dflash_pool, common/speculative.cpp) the pool logged "speculative stage 'dflash' requested but not built
by the per-sequence speculator" and the server ran without speculation, and the server's draft context was created with no
dflash_query_capacity, so even a built impl had no cross-context. This test fails on that binary (no drafts, the warning present).

Arms (all CPU, temperature 0, top_k 1):
  PLAIN    llama-server, no speculation, -np 1                     -> reference greedy outputs for P1, P2 (+ tie-check controls)
  DF       llama-server, -np 2, --spec-type dflash -md <draft>,     -> the log names the per-sequence dflash impl, no "not built";
           P1 and P2 concurrently, then P1 again on a reused slot      every request drafts and accepts; outputs == PLAIN
  SIMPLE   llama-speculative-simple --spec-type dflash -md <draft>  -> n_drafted > 0 and n_accept > 0

Fixtures: LLAMA_TEST_DF_TARGET (default: the Qwen3-8B Q4_K_M gguf in the ollama store) and LLAMA_TEST_DF_DRAFT (default:
z-lab/Qwen3-8B-DFlash-b16 @9b41424b converted by this tree to f16). Binary: LLAMA_SERVER_BIN or the first argument (bin dir or
exe). Ports: LLAMA_TEST_PORT (default 18960) and the next one. LLAMA_TEST_ARMS = comma list of plain,np2,df,simple (default all).
Exit 77 = skipped (no binary or no model).
"""

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mtp_serving", HERE / "test-server-mtp-serving.py")
mtp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mtp)

SKIP = 77
# portability-ok: default fixture paths on this host; LLAMA_TEST_DF_TARGET / LLAMA_TEST_DF_DRAFT override them
DEFAULT_TARGET = "models/ollama/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
DEFAULT_DRAFT = "models/gguf/qwen3-8b-dflash-b16-f16.gguf"  # portability-ok: see above
N_PREDICT = 48
# LLAMA_TEST_DF_TYPE: the stage to test (dflash, or dspark for a draft with a DSpark Markov head). Appending a second
# --spec-type through LLAMA_TEST_DF_EXTRA builds BOTH stages, and the dflash one rightly refuses a DSpark draft.
DF_TYPE = os.environ.get("LLAMA_TEST_DF_TYPE", "dflash")
DF_ARGS = ["--spec-type", DF_TYPE, "--spec-draft-n-max", os.environ.get("LLAMA_TEST_DF_NMAX", "8")] + os.environ.get("LLAMA_TEST_DF_EXTRA", "").split()
LOAD_ERRORS = ("failed to load draft model", "DFlash draft is incompatible", "failed to set DFlash target features",
               "DFlash draft decode failed", "exposes no cross-context", "failed to process speculative batch", "GGML_ASSERT")


def main():
    exe = mtp.find_server()
    target = Path(os.environ.get("LLAMA_TEST_DF_TARGET", DEFAULT_TARGET))
    draft = Path(os.environ.get("LLAMA_TEST_DF_DRAFT", DEFAULT_DRAFT))
    arms = set((os.environ.get("LLAMA_TEST_ARMS") or "plain,np2,df,simple").split(","))
    if exe is None or not target.is_file() or not draft.is_file():
        print("SKIP: server=%s target=%s draft=%s" % (exe, target, draft))
        return SKIP
    simple = exe.parent / ("llama-speculative-simple" + exe.suffix)
    port0 = int(os.environ.get("LLAMA_TEST_PORT", "18960"))
    threads = int(os.environ.get("LLAMA_TEST_THREADS", "4"))
    log_dir = Path(os.environ.get("LLAMA_TEST_LOG_DIR", tempfile.gettempdir()))
    log_dir.mkdir(parents=True, exist_ok=True)
    mtp.N_PREDICT = N_PREDICT
    print("server: %s\ntarget: %s\ndraft : %s\nlogs  : %s\narms  : %s" % (exe, target, draft, log_dir, sorted(arms)))

    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails += 1

    def run_arm(name, port, extra, body):
        print("\n== %s: %s" % (name, " ".join(extra)), flush=True)
        s = mtp.Server(exe, target, port, extra, log_dir / ("server_df_%s.log" % name), threads)
        try:
            ready = s.wait_ready()
            check(ready, "%s: server ready (rc %s)" % (name, s.proc.poll()))
            if ready:
                body(s)
        finally:
            s.stop()
        text = s.log_text()
        for ln in mtp.key_lines(text):
            print("   log| " + ln)
        return text

    prompts = {"P1": mtp.P1, "P2": mtp.P2}
    ref = {}

    def check_same(s, arm, k, got):
        ok, note = mtp.same_greedy(s.port, prompts[k], got, ref[k])
        check(ok, "%s %s: output == PLAIN (%s; %s)" % (arm, k, note, got.stats()))
        if not ok:
            print("   %-5s: %r\n   PLAIN: %r" % (arm[:5], got.text[:160], ref[k].text[:160]))

    if arms & {"plain", "df"}:
        def plain(s):
            for k, p in prompts.items():
                ref[k] = mtp.complete(s.port, p)
                check(len(ref[k].tokens) > 8, "PLAIN %s: generated %d tokens (%s)" % (k, len(ref[k].tokens), ref[k].stats()))
                print("   PLAIN %s: %r" % (k, ref[k].text[:120]))
            if len(ref["P1"].tokens) > 8 and len(ref["P2"].tokens) > 8:
                ok, note = mtp.same_greedy(s.port, mtp.P1, ref["P1"], ref["P1"])
                check(ok, "CONTROL tie check passes PLAIN against itself (%s)" % note)
                fake = mtp.Out()
                fake.tokens = list(ref["P1"].tokens)
                fake.tokens[5] = ref["P2"].tokens[3] if ref["P2"].tokens[3] != fake.tokens[5] else ref["P2"].tokens[4]
                fake.text = "<P1 with token 5 swapped>"
                ok, note = mtp.same_greedy(s.port, mtp.P1, fake, ref["P1"])
                check(not ok, "CONTROL tie check fails PLAIN with token 5 swapped (%s)" % note)

        run_arm("plain", port0, ["-np", "1"], plain)
    have_ref = len(ref) == 2 and all(len(r.tokens) > 8 for r in ref.values())

    if "np2" in arms and have_ref:
        # CONTROL for the DF arm: the same two concurrent requests on -np 2 WITHOUT speculation. If these already differ from
        # PLAIN, a DF mismatch says nothing about the DFlash pool (concurrent batching changes the target's numerics or state).
        def np2(s):
            outs = mtp.complete_concurrent(s.port, [mtp.P1, mtp.P2])
            for k, got in zip(prompts, outs):
                check_same(s, "NP2", k, got)

        run_arm("np2", port0 + 2, ["-np", "2"], np2)

    if "self" in arms and have_ref:
        # CONTROL for the DF arm: multi-token drafts from a model-free stage on -np 2. If these break the outputs too, the
        # fault is in the pooled verify / rollback path shared by every stage, not in the DFlash impl.
        def selfspec(s):
            outs = mtp.complete_concurrent(s.port, [mtp.P1, mtp.P2])
            for k, got in zip(prompts, outs):
                print("   SELF %s: %s" % (k, got.stats()))
                check_same(s, "SELF", k, got)

        run_arm("self", port0 + 3, ["-np", "2", "--spec-type", "suffix_automaton", "--spec-draft-n-max", "8"], selfspec)

    dm_draft = Path(os.environ.get("LLAMA_TEST_DM_DRAFT", "models/gguf/Qwen3-0.6B-Q8_0.gguf"))  # portability-ok: host fixture, env overrides
    if "dm" in arms and have_ref and dm_draft.is_file():
        # CONTROL for the DF arm: long (n_max 8) drafts from an ordinary draft model in BOTH slots at once. The suffix control
        # rarely drafts; this one drafts every round, which is the shape that broke the DF arm.
        def dm(s):
            outs = mtp.complete_concurrent(s.port, [mtp.P1, mtp.P2])
            for k, got in zip(prompts, outs):
                print("   DM %s: %s" % (k, got.stats()))
                check_same(s, "DM", k, got)

        run_arm("dm", port0 + 4, ["-np", "2", "-md", str(dm_draft), "--spec-type", "draft", "--spec-draft-n-max", "8"]
                + os.environ.get("LLAMA_TEST_DM_EXTRA", "").split(), dm)
    elif "dm" in arms:
        print("SKIP dm: no draft model at %s" % dm_draft)

    # The pool's mitigation for row S69-SPEC-POOL-DFLASH-CONCURRENT-VERIFY-CORRUPTS-TARGET-2026-11-05-01: with two sequences
    # drafting, DFlash stays out of the round, so concurrent requests are only required to stay CORRECT. With
    # LLAMA_DFLASH_POOL_CONCURRENT=1 (the defect reproduced) they must also draft.
    concurrent_drafting = os.environ.get("LLAMA_DFLASH_POOL_CONCURRENT", "") == "1"
    if "df" in arms and have_ref:
        def df(s):
            outs = mtp.complete_concurrent(s.port, [mtp.P1, mtp.P2])
            for k, got in zip(prompts, outs):
                if concurrent_drafting:
                    check(got.draft_n > 0 and got.accepted > 0, "DF %s (concurrent): drafts verified and accepted (%s)" % (k, got.stats()))
                else:
                    print("   DF %s (concurrent, mitigation on): %s" % (k, got.stats()))
                check_same(s, "DF", k, got)
            again = mtp.complete(s.port, mtp.P1)  # a reused slot: the feature window must follow the new request
            check(again.draft_n > 0 and again.accepted > 0, "DF P1 (repeat, reused slot): drafts verified and accepted (%s)" % again.stats())
            check_same(s, "DF-repeat", "P1", again)
            tot = sum(o.draft_n for o in outs + [again])
            acc = sum(o.accepted for o in outs + [again])
            print("   DF acceptance: %d of %d drafted (%.1f%%)" % (acc, tot, 100.0 * acc / max(1, tot)))

        log = run_arm("df", port0 + 1, ["-np", "2", "-md", str(draft)] + DF_ARGS, df)
        check(mtp.STAGE_NOT_BUILT.format(DF_TYPE) not in log, "DF: no 'not built' warning for %s" % DF_TYPE)
        check("adding speculative implementation '%s' (per-sequence" % DF_TYPE in log, "DF: the per-sequence speculator builds %s" % DF_TYPE)
        bad = [e for e in LOAD_ERRORS if e in log]
        check(not bad, "DF: no draft-load / feature errors in the log (%s)" % (bad or "none"))
        if not concurrent_drafting:
            check("DFlash drafts only while a single sequence is active" in log,
                  "DF: the concurrent-drafting mitigation engaged (warning present)")
    elif "df" in arms:
        check(False, "DF arm: no PLAIN reference output")

    if "simple" in arms:
        if not simple.is_file():
            check(False, "SIMPLE: %s not found" % simple)
        else:
            log_path = log_dir / "speculative_simple_df.log"
            argv = [str(simple), "-m", str(target), "-md", str(draft)] + DF_ARGS + [
                "-t", str(threads), "-c", "2048", "--temp", "0", "-n", str(N_PREDICT), "-p", mtp.P2]
            print("\n== simple: %s" % " ".join(argv[1:]), flush=True)
            with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
                rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, timeout=1800,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
            text = log_path.read_text(encoding="utf-8", errors="replace")
            m_d = re.search(r"n_drafted\s*=\s*(\d+)", text)
            m_a = re.search(r"n_accept\s*=\s*(\d+)", text)
            n_d = int(m_d.group(1)) if m_d else -1
            n_a = int(m_a.group(1)) if m_a else -1
            for ln in mtp.key_lines(text):
                print("   log| " + ln)
            check(rc == 0, "SIMPLE: llama-speculative-simple exits 0 (rc %d)" % rc)
            check(n_d > 0 and n_a > 0, "SIMPLE: n_drafted %d > 0 and n_accept %d > 0" % (n_d, n_a))
            bad = [e for e in LOAD_ERRORS if e in text]
            check(not bad, "SIMPLE: no draft-load / feature errors in the log (%s)" % (bad or "none"))

    print("\nRESULT: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
