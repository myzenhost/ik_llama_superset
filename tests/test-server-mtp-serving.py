#!/usr/bin/env python3
"""llama-server must serve MTP speculative decoding, name every stage it cannot build, and run --spec-draft-policy per slot.

llama-server builds its speculator with the per-sequence pool (common/speculative.cpp common_speculative_init(params, n_seq)),
not the single-context engine llama-cli / llama-speculative-simple use. Until 2026-10-17 that pool had no MTP implementation:
`--spec-type mtp` served with no speculation at all, `--spec-type suffix,mtp` silently dropped the MTP stage, and
`--spec-draft-policy` was never built (row S69-FORK-SERVER-PER-SEQ-SPECULATOR-NO-MTP-NO-DRAFT-POLICY-2026-10-17-01).

Arms (all CPU, /completion, temperature 0, top_k 1, fixed seed; prompts P1 prose, P2 code):
  PLAIN      no speculation, -np 1                          -> the reference greedy outputs; the tie check's controls
  PLAIN_NP2  no speculation, -np 2, P1 and P2 CONCURRENTLY  -> control: what batching alone does to the greedy text
  MTP        --spec-type mtp, -np 1                         -> timings.draft_n > 0 on both prompts, output == PLAIN
  MTP_NP2    --spec-type mtp, -np 2, concurrent             -> both draft, both outputs == PLAIN (per-slot MTP state)
  POLICY     --spec-type suffix,mtp --spec-draft-policy -np 2, concurrent
                                                            -> log "draft policy on", drafts, outputs == PLAIN
  WARN_E3    --spec-type eagle3 (no draft model)           -> a WARN naming the stage 'eagle3' as not built; output == PLAIN
  WARN_CHAIN --spec-type ngram_simple,mtp                   -> both stages built (ngram_simple is pooled since 2026-10-18), drafts

"output == PLAIN": speculative decoding with greedy verification never commits a token the target would not pick greedily, but
it is NOT bitwise batch-invariant: a verify batch of 1+k tokens, or two slots decoding in one batch, runs other kernels
(multi-row GEMM, the chunked gated-delta-net path of qwen35) than one token alone, and the logits move by float noise.
MEASURED 2026-10-17 on the db8a01003 binary: two slots decoding concurrently with NO speculation already changed the greedy text
of both prompts. So each output is compared to PLAIN token by token: identical passes; otherwise the FIRST differing position is
re-scored on the arm's own server with speculation off for that request (speculative.n_max 0, prompt + the common prefix, top 8
logprobs), and it passes only if the two tokens are a near-tie there (|logprob gap| <= TIE_NATS). A state or rollback bug commits
a token the target does not rank level with its argmax, which fails. Every gap is printed. The check is itself controlled in the
PLAIN arm: PLAIN against itself passes, and PLAIN with one token swapped for an unrelated one fails.

Fixture: a small GGUF with an MTP (NextN) head. Default: Qwen3.5-0.8B-Q8_0 (qwen35, 1 NextN layer), ggml-org/Qwen3.5-0.8B-GGUF.
Set LLAMA_TEST_MTP_GGUF to another one. Binary: LLAMA_SERVER_BIN, or the first argument (a bin dir or the exe), or ./bin.
Ports: LLAMA_TEST_PORT (default 18960) and the next 6. Logs: LLAMA_TEST_LOG_DIR (default: the system temp dir).
Exit 77 = skipped (no binary or no model).
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

SKIP = 77
DEFAULT_MODEL = "models/gguf/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf"  # portability-ok: default fixture path, LLAMA_TEST_MTP_GGUF overrides

P1 = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined several "
      "existing technologies into a single system. In this short essay we explain")
P2 = ("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n"
      "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n\ndef factorial(n):\n")
N_PREDICT = 64
MTP_STAGE = "mtp:n_max=3"

STAGE_NOT_BUILT = "speculative stage '{}' requested but not built"
TIE_NATS = 0.25


def find_server():
    cands = []
    if os.environ.get("LLAMA_SERVER_BIN"):
        cands.append(Path(os.environ["LLAMA_SERVER_BIN"]))
    if len(sys.argv) > 1:
        a = Path(sys.argv[1])
        cands += [a, a / "llama-server.exe", a / "llama-server"]
    cands += [Path("bin") / "llama-server.exe", Path("bin") / "llama-server"]
    for c in cands:
        if c.is_file():
            return c
    return None


def post(port, path, body, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health(port):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


class Server:
    def __init__(self, exe, model, port, extra, log_path, threads):
        self.port = port
        self.log_path = log_path
        argv = [str(exe), "-m", str(model), "--port", str(port), "--host", "127.0.0.1", "-t", str(threads),
                "-c", "4096", "-ngl", "0", "--no-webui"] + extra
        self.argv = argv
        self.log = open(log_path, "w", encoding="utf-8", errors="replace")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, creationflags=flags)

    def wait_ready(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            if health(self.port):
                return True
            time.sleep(0.5)
        return False

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        self.log.close()

    def log_text(self):
        return Path(self.log_path).read_text(encoding="utf-8", errors="replace")


class Out:
    def __init__(self, r=None, error=None):
        r = r or {}
        t = r.get("timings", {})
        self.text = r.get("content", "") if error is None else "<error %s>" % error
        self.tokens = list(r.get("tokens") or [])
        self.draft_n = int(t.get("draft_n", 0) or 0)
        self.accepted = int(t.get("draft_n_accepted", 0) or 0)

    def stats(self):
        return "draft_n %d, accepted %d" % (self.draft_n, self.accepted)


def complete(port, prompt):
    try:
        return Out(post(port, "/completion", {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0.0, "top_k": 1,
                                              "seed": 1, "cache_prompt": False, "return_tokens": True}))
    except Exception as e:  # reported as a failed check by the caller
        return Out(error=e)


def complete_concurrent(port, prompts):
    out = [None] * len(prompts)

    def one(i):
        out[i] = complete(port, prompts[i])

    th = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def same_greedy(port, prompt, got, ref):
    """(ok, note): got == ref, or their first differing token is a near-tie for the target, scored with speculation off."""
    if got.tokens == ref.tokens and got.text == ref.text:
        return True, "identical"
    if not got.tokens or not ref.tokens:
        return False, "no tokens returned (%s)" % got.text[:80]
    d = 0
    while d < min(len(got.tokens), len(ref.tokens)) and got.tokens[d] == ref.tokens[d]:
        d += 1
    if d == min(len(got.tokens), len(ref.tokens)):
        return False, "one output is a strict prefix of the other (%d vs %d tokens)" % (len(got.tokens), len(ref.tokens))
    r = post(port, "/completion", {"prompt": [prompt] + got.tokens[:d], "n_predict": 1, "n_probs": 8, "temperature": 0.0,
                                   "top_k": 1, "seed": 1, "cache_prompt": False, "speculative.n_max": 0})
    probs = (r.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
    lp = {int(p["id"]): float(p["logprob"]) for p in probs if "id" in p and "logprob" in p}
    a, b = got.tokens[d], ref.tokens[d]
    if a not in lp or b not in lp:
        top = sorted(lp.items(), key=lambda x: -x[1])[:3]
        return False, "first difference at token %d: %d vs %d, not both in the target's top 8 (top 3 %s)" % (d, a, b, top)
    gap = abs(lp[a] - lp[b])
    return gap <= TIE_NATS, "first difference at token %d of %d: logprob %.4f vs %.4f, gap %.4f nats (tie <= %.2f)" % (
        d, len(ref.tokens), lp[a], lp[b], gap, TIE_NATS)


def key_lines(text):
    keys = ("draft policy", "not built", "per-sequence speculator", "WITHOUT speculative", "adding speculative implementation",
            "speculative decoding context", "failed")
    return [ln for ln in text.splitlines() if any(k in ln for k in keys)][:14]


def main():
    exe = find_server()
    model = Path(os.environ.get("LLAMA_TEST_MTP_GGUF", DEFAULT_MODEL))
    if exe is None or not model.is_file():
        print("SKIP: server=%s model=%s" % (exe, model))
        return SKIP
    port0 = int(os.environ.get("LLAMA_TEST_PORT", "18960"))
    threads = int(os.environ.get("LLAMA_TEST_THREADS", "4"))
    log_dir = Path(os.environ.get("LLAMA_TEST_LOG_DIR", tempfile.gettempdir()))
    log_dir.mkdir(parents=True, exist_ok=True)
    print("server: %s\nmodel : %s\nlogs  : %s" % (exe, model, log_dir))

    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + what, flush=True)
        if not ok:
            fails += 1

    def run_arm(name, port, extra, body):
        print("\n== %s: %s" % (name, " ".join(extra)), flush=True)
        s = Server(exe, model, port, extra, log_dir / ("server_%s.log" % name), threads)
        try:
            ready = s.wait_ready()
            check(ready, "%s: server ready (rc %s)" % (name, s.proc.poll()))
            if ready:
                body(s)
        finally:
            s.stop()
        text = s.log_text()
        for ln in key_lines(text):
            print("   log| " + ln)
        return text

    prompts = {"P1": P1, "P2": P2}
    ref = {}

    def check_same(s, arm, k, got):
        ok, note = same_greedy(s.port, prompts[k], got, ref[k])
        check(ok, "%s %s: output == PLAIN (%s; %s)" % (arm, k, note, got.stats()))
        if not ok:
            print("   %-5s: %r\n   PLAIN: %r" % (arm[:5], got.text[:160], ref[k].text[:160]))

    def plain(s):
        for k, p in prompts.items():
            ref[k] = complete(s.port, p)
            check(len(ref[k].text) > 20 and len(ref[k].tokens) > 8,
                  "PLAIN %s: generated %d chars / %d tokens (%s)" % (k, len(ref[k].text), len(ref[k].tokens), ref[k].stats()))
        if len(ref.get("P1", Out()).tokens) > 8:
            # controls for the tie check: it must pass PLAIN against itself and fail a swapped token
            ok, note = same_greedy(s.port, P1, ref["P1"], ref["P1"])
            check(ok, "CONTROL tie check passes PLAIN against itself (%s)" % note)
            fake = Out()
            fake.tokens = list(ref["P1"].tokens)
            fake.tokens[5] = ref["P2"].tokens[3] if ref["P2"].tokens[3] != fake.tokens[5] else ref["P2"].tokens[4]
            fake.text = "<P1 with token 5 swapped>"
            ok, note = same_greedy(s.port, P1, fake, ref["P1"])
            check(not ok, "CONTROL tie check fails PLAIN with token 5 swapped (%s)" % note)

    run_arm("plain", port0, ["-np", "1"], plain)
    if len(ref) != 2 or any(len(o.tokens) <= 8 for o in ref.values()):
        print("RESULT: FAIL (no reference output)")
        return 1

    def plain2(s):
        for k, got in zip(prompts, complete_concurrent(s.port, [P1, P2])):
            check_same(s, "PLAIN_NP2", k, got)

    run_arm("plain_np2", port0 + 6, ["-np", "2"], plain2)

    def mtp1(s):
        for k, p in prompts.items():
            got = complete(s.port, p)
            check(got.draft_n > 0, "MTP %s: timings.draft_n > 0 (%s)" % (k, got.stats()))
            check_same(s, "MTP", k, got)

    log = run_arm("mtp", port0 + 1, ["-np", "1", "--spec-type", MTP_STAGE], mtp1)
    check("WITHOUT speculative decoding" not in log, "MTP: server does not fall back to plain decoding")

    def mtp2(s):
        for k, got in zip(prompts, complete_concurrent(s.port, [P1, P2])):
            check(got.draft_n > 0, "MTP_NP2 %s: timings.draft_n > 0 (%s)" % (k, got.stats()))
            check_same(s, "MTP_NP2", k, got)

    run_arm("mtp_np2", port0 + 2, ["-np", "2", "--spec-type", MTP_STAGE], mtp2)

    def policy(s):
        tot = 0
        for k, got in zip(prompts, complete_concurrent(s.port, [P1, P2])):
            tot += got.draft_n
            check_same(s, "POLICY", k, got)
        check(tot > 0, "POLICY: drafts were verified (draft_n total %d)" % tot)

    log = run_arm("policy", port0 + 3, ["-np", "2", "--spec-type", "suffix," + MTP_STAGE, "--spec-draft-policy"], policy)
    check("draft policy on" in log, "POLICY: server log says 'draft policy on'")

    def warn_e3(s):
        check_same(s, "WARN_E3", "P1", complete(s.port, P1))

    log = run_arm("warn_eagle3", port0 + 4, ["-np", "1", "--spec-type", "eagle3"], warn_e3)
    check(STAGE_NOT_BUILT.format("eagle3") in log, "WARN_E3: a WARN names stage 'eagle3' as not built")

    def warn_chain(s):
        got = complete(s.port, P1)
        check(got.draft_n > 0, "WARN_CHAIN: the mtp stage still drafts (%s)" % got.stats())
        check_same(s, "WARN_CHAIN", "P1", got)

    # ngram_simple is built by the pool since 2026-10-18 (row S69-FORK-SERVER-POOL-NGRAM-MOD-SIMPLE-CACHE-DSPARK-IMPLS-2026-10-17-01;
    # until then this arm asserted its 'not built' warning): now both stages of the chain must be built, in chain order
    log = run_arm("warn_chain", port0 + 5, ["-np", "1", "--spec-type", "ngram_simple," + MTP_STAGE], warn_chain)
    check(STAGE_NOT_BUILT.format("ngram_simple") not in log, "WARN_CHAIN: no WARN names stage 'ngram_simple' as not built")
    check("stages in draft order: ngram_simple,mtp" in log, "WARN_CHAIN: the per-sequence speculator builds ngram_simple,mtp")

    print("\nRESULT: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
