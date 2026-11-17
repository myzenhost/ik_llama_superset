#!/usr/bin/env python3
"""llama-server's per-sequence speculator must build the n-gram stages, and --spec-draft-policy must be reachable from the environment.

Rows: S69-FORK-ARG-SPEC-DRAFT-POLICY-ENV-2026-10-17-01 (LLAMA_ARG_SPEC_DRAFT_POLICY) and
S69-FORK-SERVER-POOL-NGRAM-MOD-SIMPLE-CACHE-DSPARK-IMPLS-2026-10-17-01 (ngram_simple / ngram_mod / ngram_cache in the pool that
llama-server builds with common_speculative_init(params, n_seq); before, each of them was warned "not built" and dropped).

Arms (all CPU, /completion, temperature 0, top_k 1, fixed seed):
  PLAIN          no speculation, -np 1                                -> reference greedy outputs for P1, P2
  <stage>        --spec-type <stage>, -np 2, P1 and P2 concurrently   -> the log names the stage as built (no "not built" warning),
                 for ngram_simple, ngram_mod, ngram_cache                 drafts are verified and accepted, outputs == PLAIN
  MOD_SHARED     --spec-type ngram_mod, -np 2: T2 once, then T2 again while the other slot is busy with P1
                                                                      -> the repeat drafts from the table the first request filled
  CACHE_REUSE    --spec-type ngram_cache, -np 1: a long request, then a SHORTER unrelated one on the same slot
                                                                      -> the second request still drafts and accepts (its context
                                                                         cache was rebuilt for the new text, see upstream #27852)
  POLICY_ENV     --spec-type suffix,mtp, -np 2, LLAMA_ARG_SPEC_DRAFT_POLICY=1 in the environment, no flag
                                                                      -> log "draft policy on"
  POLICY_OFF     the same with no env and no flag (control)           -> no "draft policy on"

The n-gram stages draft from text the model has already seen, so P1 and P2 are COPY prompts: a passage, the passage again, and its
first words a third time. A model that copies continues with tokens the n-gram tables predict. stories15M, tried first on
2026-10-18, writes a new story instead (ngram_mod: 48 drafted, 0 accepted) and its slots are capped at its 128-token training
context, so the arms use the MTP test's Qwen3.5-0.8B.

"output == PLAIN" is the check of tests/test-server-mtp-serving.py (identical tokens, or a first difference that is a near-tie for the
target with speculation off); it is imported from there, with its controls.

Fixture: LLAMA_TEST_MTP_GGUF (default Qwen3.5-0.8B-Q8_0, as the MTP test). Binary: LLAMA_SERVER_BIN, or the first argument (a bin dir
or the exe). Ports: LLAMA_TEST_PORT (default 18930) and the next 7. Arms: LLAMA_TEST_ARMS = comma list of plain,ngram,shared,reuse,
policy (default all). Exit 77 = skipped (no binary or no model).
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mtp_serving", HERE / "test-server-mtp-serving.py")
mtp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mtp)

SKIP = 77
NGRAM_STAGES = ["ngram_simple", "ngram_mod", "ngram_cache"]


def copy_prompt(passage, n_lead_words=6):
    return passage + "\n\n" + passage + "\n\n" + " ".join(passage.split()[:n_lead_words])


T1 = ("The river ran past the old mill every morning, carrying leaves and small branches down to the sea. The miller's daughter "
      "counted the boats that passed and wrote their names in a blue notebook that she kept under her bed.")
T2 = ("To build the project, first install the compiler, then run the configure script with the release option, and finally "
      "call make with four jobs. The binaries are written to the bin directory next to the source tree.")
T3 = ("A quiet market opened at dawn in the square, where farmers sold apples, bread and honey from wooden carts. By noon the "
      "carts were empty and the square was quiet again, apart from the pigeons looking for crumbs.")
P1 = copy_prompt(T1)
P2 = copy_prompt(T2)
# CACHE_REUSE: a long first text, then a SHORTER one that shares none of its sentences
REUSE_A = T1 + "\n\n" + T1 + "\n\n" + T1 + "\n\n" + " ".join(T1.split()[:6])
REUSE_B = copy_prompt(T3.split(". ")[0] + ".", 4)


class Server(mtp.Server):
    """mtp.Server with an explicit environment for the child."""

    def __init__(self, exe, model, port, extra, log_path, threads, env=None):
        self.port = port
        self.log_path = log_path
        argv = [str(exe), "-m", str(model), "--port", str(port), "--host", "127.0.0.1", "-t", str(threads),
                "-c", "4096", "-ngl", "0", "--no-webui"] + extra
        self.argv = argv
        self.log = open(log_path, "w", encoding="utf-8", errors="replace")
        child_env = dict(os.environ)
        for k in [k for k in child_env if k.startswith("LLAMA_ARG_")]:
            del child_env[k]  # only what an arm sets reaches the server
        child_env.update(env or {})
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, creationflags=flags, env=child_env)


def main():
    exe = mtp.find_server()
    model = Path(os.environ.get("LLAMA_TEST_MTP_GGUF", mtp.DEFAULT_MODEL))
    arms = set((os.environ.get("LLAMA_TEST_ARMS") or "plain,ngram,shared,reuse,policy").split(","))
    if exe is None or not model.is_file():
        print("SKIP: server=%s model=%s" % (exe, model))
        return SKIP
    port0 = int(os.environ.get("LLAMA_TEST_PORT", "18930"))
    threads = int(os.environ.get("LLAMA_TEST_THREADS", "4"))
    log_dir = Path(os.environ.get("LLAMA_TEST_LOG_DIR", tempfile.gettempdir()))
    log_dir.mkdir(parents=True, exist_ok=True)
    print("server: %s\nmodel : %s\nlogs  : %s\narms  : %s" % (exe, model, log_dir, sorted(arms)))

    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails += 1

    def run_arm(name, port, extra, body, env=None):
        print("\n== %s: %s%s" % (name, " ".join(extra), (" env %s" % env) if env else ""), flush=True)
        s = Server(exe, model, port, extra, log_dir / ("server_%s.log" % name), threads, env)
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

    prompts = {"P1": P1, "P2": P2}
    ref = {}

    def check_same(s, arm, k, got):
        ok, note = mtp.same_greedy(s.port, prompts[k], got, ref[k])
        check(ok, "%s %s: output == PLAIN (%s; %s)" % (arm, k, note, got.stats()))
        if not ok:
            print("   %-5s: %r\n   PLAIN: %r" % (arm[:5], got.text[:160], ref[k].text[:160]))

    if arms & {"plain", "ngram", "shared"}:
        def plain(s):
            for k, p in prompts.items():
                ref[k] = mtp.complete(s.port, p)
                check(len(ref[k].tokens) > 8, "PLAIN %s: generated %d tokens (%s)" % (k, len(ref[k].tokens), ref[k].stats()))
                print("   PLAIN %s: %r" % (k, ref[k].text[:120]))
            if len(ref["P1"].tokens) > 8 and len(ref["P2"].tokens) > 8:
                ok, note = mtp.same_greedy(s.port, P1, ref["P1"], ref["P1"])
                check(ok, "CONTROL tie check passes PLAIN against itself (%s)" % note)
                fake = mtp.Out()
                fake.tokens = list(ref["P1"].tokens)
                fake.tokens[5] = ref["P2"].tokens[3] if ref["P2"].tokens[3] != fake.tokens[5] else ref["P2"].tokens[4]
                fake.text = "<P1 with token 5 swapped>"
                ok, note = mtp.same_greedy(s.port, P1, fake, ref["P1"])
                check(not ok, "CONTROL tie check fails PLAIN with token 5 swapped (%s)" % note)

        run_arm("plain", port0, ["-np", "1"], plain)
    have_ref = len(ref) == 2 and all(len(r.tokens) > 8 for r in ref.values())

    if "ngram" in arms and have_ref:
        for i, stage in enumerate(NGRAM_STAGES):
            def body(s, stage=stage):
                outs = mtp.complete_concurrent(s.port, [P1, P2])
                tot = sum(o.draft_n for o in outs)
                acc = sum(o.accepted for o in outs)
                check(tot > 0 and acc > 0, "%s: drafts verified and accepted (draft_n %d, accepted %d)" % (stage, tot, acc))
                for k, got in zip(prompts, outs):
                    check_same(s, stage, k, got)

            log = run_arm(stage, port0 + 1 + i, ["-np", "2", "--spec-type", stage], body)
            check(mtp.STAGE_NOT_BUILT.format(stage) not in log, "%s: no 'not built' warning for the stage" % stage)
            check(("stages in draft order: %s" % stage) in log, "%s: the per-sequence speculator lists the stage as built" % stage)
    elif "ngram" in arms:
        check(False, "ngram arms: no PLAIN reference output")

    if "shared" in arms:
        def shared(s):
            first = mtp.complete(s.port, T2)  # fills the shared table with the passage and its continuation
            res = {}
            t = threading.Thread(target=lambda: res.setdefault("busy", mtp.complete(s.port, P1)))
            t.start()
            again = mtp.complete(s.port, T2)  # served while the other slot is busy with P1
            t.join()
            print("   first : %s\n   repeat: %s\n   other : %s" % (first.stats(), again.stats(), res["busy"].stats()))
            check(again.draft_n > 0 and again.accepted > first.accepted,
                  "MOD_SHARED: the repeat drafts from the shared table and accepts more than the first (%d > %d)" % (
                      again.accepted, first.accepted))
            ok, note = mtp.same_greedy(s.port, T2, again, first)
            check(ok, "MOD_SHARED: the repeat's text == the first's (%s)" % note)

        run_arm("mod_shared", port0 + 4, ["-np", "2", "--spec-type", "ngram_mod"], shared)

    if "reuse" in arms:
        def reuse(s):
            a = mtp.complete(s.port, REUSE_A)
            b = mtp.complete(s.port, REUSE_B)
            print("   first (long) : %s\n   second (short): %s" % (a.stats(), b.stats()))
            check(a.draft_n > 0 and a.accepted > 0, "CACHE_REUSE: the first request drafts and accepts (%s)" % a.stats())
            check(b.draft_n > 0 and b.accepted > 0, "CACHE_REUSE: the shorter second request on the same slot drafts and accepts (%s)" % b.stats())

        run_arm("cache_reuse", port0 + 5, ["-np", "1", "--spec-type", "ngram_cache"], reuse)

    if "policy" in arms:
        chain = ["-np", "2", "--spec-type", "suffix," + mtp.MTP_STAGE]

        def one(s):
            got = mtp.complete(s.port, mtp.P1)
            check(len(got.tokens) > 8, "request served (%d tokens, %s)" % (len(got.tokens), got.stats()))

        log = run_arm("policy_env", port0 + 6, chain, one, env={"LLAMA_ARG_SPEC_DRAFT_POLICY": "1"})
        check("draft policy on" in log, "POLICY_ENV: LLAMA_ARG_SPEC_DRAFT_POLICY=1 turns the draft policy on (log 'draft policy on')")
        log = run_arm("policy_off", port0 + 7, chain, one)
        check("draft policy on" not in log, "POLICY_OFF: control, no env and no flag -> no 'draft policy on'")

    print("\nRESULT: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
