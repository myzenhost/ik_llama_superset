#!/usr/bin/env python3
"""End-to-end proof that the SGLang picks are WIRED: a real llama-server on a random-weight GGUF, each feature switched on through its
real flag / request field, and the always-on log line or the response counter that shows it FIRED. Also the flag-off controls.

    python3 sglang_wired_check.py <llama-server> <tiny.gguf> [pick ...]

picks: namespace  radix  sched  logprob  promptids  skipspecial  spf  identity   (default: all; identity needs env BASE_BIN).   tiny.gguf: tests/make_tiny_llama_gguf.py
Row S69-ENGINE-MERGE-PROGRAMME-ALL-ENGINES-2026-09-09-01. Exit 0 = every check passed.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

BIN, MODEL = sys.argv[1], sys.argv[2]
WANT = set(sys.argv[3:]) or {"namespace", "radix", "sched", "logprob", "promptids", "skipspecial", "spf", "identity"}
PORT = 18900
fails = []


def check(cond, msg):
    print(("  ok:   " if cond else "  FAIL: ") + msg)
    if not cond:
        fails.append(msg)


class Server:
    def __init__(self, *args):
        global PORT
        PORT += 1
        self.port = PORT
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
        cmd = [BIN, "-m", MODEL, "--port", str(self.port), "--host", "127.0.0.1", "-c", "8192", "--threads", "2", "--no-webui"] + list(args)
        # the libraries next to the binary win over the build's RUNPATH, so a copied baseline directory really runs the baseline code
        env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(os.path.abspath(BIN)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError("server died:\n" + self.text())
                time.sleep(0.2)
        raise RuntimeError("server did not start:\n" + self.text())

    def text(self):
        self.log.flush()
        return open(self.log.name).read()

    def post(self, path, body, timeout=120):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def complete(self, prompt, **kw):
        body = {"prompt": prompt, "n_predict": 1, "temperature": 0.0, "seed": 1}
        body.update(kw)
        st, js = self.post("/completion", body)
        assert st == 200, (st, js)
        return js

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except Exception:
            self.proc.kill()


def toks(start, n):
    return [4 + (start + i) % 58 for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------------------------
def run_namespace():
    print("namespace: request field cache_salt / extra_key")
    s = Server("-np", "1", "--no-slots")  # one slot: a hit can only come from the slot's own cache
    try:
        P = toks(0, 120)
        a = s.complete(P)["timings"]["cache_n"]
        b = s.complete(P)["timings"]["cache_n"]
        check(a == 0 and b >= 100, f"control (no salt): the second identical request reuses the first (cache_n {a} -> {b})")
        c1 = s.complete(P, cache_salt="tenant-a")["timings"]["cache_n"]
        c2 = s.complete(P, cache_salt="tenant-a")["timings"]["cache_n"]
        check(c1 == 0, f"first request of a new salt does not see the unsalted cache (cache_n {c1})")
        check(c2 >= 100, f"same salt reuses its own cache (cache_n {c2})")
        d = s.complete(P, cache_salt="tenant-b")["timings"]["cache_n"]
        check(d == 0, f"another salt does not reuse tenant-a's cache (cache_n {d})")
        e = s.complete(P, cache_salt="tenant-b", extra_key="v2")["timings"]["cache_n"]
        check(e == 0, f"same salt with another extra_key is another namespace (cache_n {e})")
        f = s.complete(P)["timings"]["cache_n"]
        check(f == 0, f"an unsalted request does not reuse a salted cache (cache_n {f})")
        txt = s.text()
        check("cache namespace active: cache_salt 8 byte(s), extra_key 0 byte(s)" in txt, "FIRED: log line 'cache namespace active' with the salt length")
        check("tenant-a" not in txt and "tenant-b" not in txt, "the salt itself is never logged")
    finally:
        s.stop()
    # prompt cache (host RAM): a salted prompt evicted from the slot must not be restored for another salt
    s = Server("-np", "1", "--cache-ram", "64", "--no-slots")
    try:
        P, Q = toks(0, 120), toks(30, 120)
        s.complete(P, cache_salt="x")
        s.complete(Q, cache_salt="x")                 # P leaves the slot, goes to the host prompt cache under salt x
        same = s.complete(P, cache_salt="x")["timings"]["cache_n"]
        s.complete(Q, cache_salt="x")
        other = s.complete(P, cache_salt="y")["timings"]["cache_n"]
        check(same >= 100, f"host prompt cache restores a prompt under its own salt (cache_n {same})")
        check(other == 0, f"host prompt cache does not restore it under another salt (cache_n {other})")
    finally:
        s.stop()


# ---------------------------------------------------------------------------------------------------------------------------------
def run_radix():
    print("radix: --radix-evict-policy")
    # Two reserved ids: the third distinct prompt (C) forces one eviction. A is sent twice (its node gets 2 hits and the newest use), B once.
    # Victim by policy: lru -> B (hits 1), lfu -> B, slru -> B, fifo -> A (hits 2, created first), filo -> B (created last), mru -> A,
    # priority -> the node of the request with the largest priority value (A is sent with priority 9).
    A, B, C = toks(0, 80), toks(20, 80), toks(40, 80)
    want_victim = {None: "hits 1,", "lru": "hits 1,", "lfu": "hits 1,", "slru": "hits 1,", "fifo": "hits 2,", "filo": "hits 1,", "mru": "hits 2,",
                   "priority": "priority 9,"}
    for policy in (None, "lru", "lfu", "slru", "fifo", "filo", "mru", "priority"):
        args = ["-np", "2", "--kv-unified", "--radix-cache", "--radix-cache-slots", "2", "--cache-ram", "0", "--no-slots"]
        if policy:
            args += ["--radix-evict-policy", policy]
        s = Server(*args)
        try:
            for p, prio in ((A, 9), (B, 0), (A, 9), (C, 0)):
                s.complete(p, n_predict=1, priority=prio)
            txt = s.text()
            name = policy or "lru"
            check(f"eviction policy {name}" in txt, f"[{name}] startup line names the policy")
            ev = [l for l in txt.splitlines() if "radix cache: evicted idle leaf by policy" in l]
            check(len(ev) == 1 and f"by policy {name} (" in ev[0], f"[{name}] FIRED: exactly one eviction line, naming the policy: {ev[0].split('radix cache: ')[-1] if ev else '-'}")
            check(len(ev) == 1 and want_victim[policy] in ev[0], f"[{name}] the leaf it chose has '{want_victim[policy]}'")
        finally:
            s.stop()
    # flag validation
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(os.path.abspath(BIN)) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    r = subprocess.run([BIN, "-m", MODEL, "--radix-evict-policy", "bogus", "--port", "1"], capture_output=True, text=True, env=env)
    check(r.returncode != 0 and "radix-evict-policy" in (r.stdout + r.stderr), "an unknown policy name is refused at parse time")


# ---------------------------------------------------------------------------------------------------------------------------------
def run_sched():
    print("sched: --sched-policy")
    S = toks(0, 70)           # warmed first: ends up in the host prompt cache
    order = {}
    for policy in ("fcfs", "lpm"):
        s = Server("-np", "1", "-c", "16384", "--cache-ram", "64", "--sched-policy", policy, "--no-slots", "-b", "64", "-ub", "64")
        try:
            check(f"scheduling policy {policy}" in s.text(), f"[{policy}] startup line names the policy")
            s.complete(S)                                       # S is now in the slot
            done = []
            lock = threading.Lock()

            def go(name, prompt, **kw):
                s.complete(prompt, **kw)
                with lock:
                    done.append(name)

            blocker = threading.Thread(target=go, args=("blocker", toks(50, 30)), kwargs={"n_predict": 6000, "ignore_eos": True})
            blocker.start()                                     # takes the slot; S moves to the host prompt cache
            time.sleep(0.5)
            t1 = threading.Thread(target=go, args=("r_unrelated", toks(5, 60)), kwargs={"n_predict": 1})
            t1.start()
            time.sleep(0.3)
            t2 = threading.Thread(target=go, args=("r_cached", S[:60] + [9, 9, 9]), kwargs={"n_predict": 1})
            t2.start()
            for t in (blocker, t1, t2):
                t.join()
            order[policy] = done
            txt = s.text()
            lines = [l for l in txt.splitlines() if "sched-policy lpm:" in l]
            if policy == "lpm":
                check(len(lines) >= 1, f"[lpm] FIRED: {len(lines)} 'sched-policy lpm:' decision line(s)")
                check(any("(REORDERED)" in l for l in lines), "[lpm] a decision line says the order differs from fcfs")
            else:
                check(len(lines) == 0, "[fcfs] no lpm decision lines (flag off = old path)")
        finally:
            s.stop()
    check(order["fcfs"].index("r_unrelated") < order["fcfs"].index("r_cached"), f"fcfs starts the earlier request first: {order['fcfs']}")
    check(order["lpm"].index("r_cached") < order["lpm"].index("r_unrelated"), f"lpm starts the request with the cached prefix first: {order['lpm']}")


# ---------------------------------------------------------------------------------------------------------------------------------
def run_logprob():
    print("logprob: logprob_start_len")
    s = Server("-np", "1", "--cache-ram", "0", "--no-slots", "-b", "64", "-ub", "64")
    try:
        P = toks(3, 160)
        choices = [[7, 8, 9, 10], [20, 21], [30, 31, 32]]

        def score(prompt, **kw):
            return s.complete(prompt, prompt_logprobs=True, n_predict=1, cache_prompt=True, **kw)

        # reference: the whole prompt scored, no reuse (also the flag-off behaviour)
        base = [score(P + c) for c in choices]
        check(all(b["timings"]["cache_n"] == 0 for b in base), "flag off: the whole prompt is evaluated, no cache reuse (cache_n 0)")
        check(all(len(b["prompt_logprobs"]["logprobs"]) == len(P) + len(c) for b, c in zip(base, choices)), "flag off: one entry per prompt token")
        s.complete(P, n_predict=1)                               # prime the shared prefix, like SGLang's select()
        for c, b in zip(choices, base):
            r = score(P + c, logprob_start_len=len(P))
            lp, ref = r["prompt_logprobs"]["logprobs"], b["prompt_logprobs"]["logprobs"]
            check(r["timings"]["cache_n"] == len(P) - 1, f"choice {c}: cache_n {r['timings']['cache_n']} == {len(P) - 1} (everything before the start but one token reused)")
            check(all(v is None for v in lp[:len(P)]), "entries before logprob_start_len are null")
            diff = max(abs(lp[k] - ref[k]) for k in range(len(P), len(P) + len(c)))
            check(diff < 1e-3, f"scored entries equal the full-prompt scores (max diff {diff:.2e})")
        txt = s.text()
        check("prompt_logprobs from position 160: 159 cached token(s) reused" in txt, "FIRED: log line 'prompt_logprobs from position ... cached token(s) reused'")
        st, js = s.post("/completion", {"prompt": P, "n_predict": 1, "prompt_logprobs": True, "logprob_start_len": len(P)})
        check(st == 400, f"a start at or past the prompt end is refused with 400 (got {st})")
    finally:
        s.stop()


def run_promptids():
    print("promptids: return_prompt_token_ids")
    s = Server("-np", "1", "--no-slots")
    try:
        P = toks(0, 25)
        r = s.complete(P)
        check("prompt_token_ids" not in r, "flag off: no prompt_token_ids in the reply (unchanged)")
        r = s.complete(P, return_prompt_token_ids=True)
        check(r.get("prompt_token_ids") == P, f"native /completion: prompt_token_ids == the prompt ids ({len(r.get('prompt_token_ids', []))} ids)")
        # OpenAI chat route: the templated prompt as ids, the same ones /apply-template + /tokenize give
        msgs = [{"role": "user", "content": "t10 t11"}]
        st, ch = s.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 1, "temperature": 0, "return_prompt_token_ids": True})
        check(st == 200 and isinstance(ch.get("prompt_token_ids"), list) and len(ch["prompt_token_ids"]) >= 1, f"chat route carries prompt_token_ids ({len(ch.get('prompt_token_ids', []))} ids)")
        st2, tpl = s.post("/apply-template", {"messages": msgs})
        st3, tk2 = s.post("/tokenize", {"content": tpl.get("prompt", ""), "add_special": True, "parse_special": True})
        check(st2 == 200 and st3 == 200 and ch.get("prompt_token_ids") == tk2["tokens"], "chat ids equal tokenize(apply-template(messages))")
        st, err = s.post("/completion", {"prompt": P, "n_predict": 1, "stream": True, "return_prompt_token_ids": True})
        check(st == 400, f"stream + return_prompt_token_ids is refused with 400 (got {st})")
    finally:
        s.stop()


def run_skipspecial():
    print("skipspecial: request field skip_special_tokens")
    s = Server("-np", "1", "--no-slots")  # no --special: control tokens are normally not rendered
    try:
        # logit_bias pushes the control token 0 ("<pad>", not end-of-generation) to be sampled every step
        kw = dict(n_predict=3, logit_bias=[[0, 100.0]], prompt=toks(0, 8))
        base = s.post("/completion", dict(kw, temperature=0.0, seed=1))[1]
        check(base["tokens_predicted"] == 3 and "<pad>" not in base["content"], f"control: default hides the special token (content {base['content']!r})")
        keep = s.post("/completion", dict(kw, temperature=0.0, seed=1, skip_special_tokens=False))
        check(keep[0] == 200 and keep[1]["content"].count("<pad>") == 3, f"skip_special_tokens=false renders it (content {keep[1].get('content')!r})")
        check(keep[1]["generation_settings"].get("skip_special_tokens") is False, "FIRED: generation_settings echoes skip_special_tokens=false")
        on = s.post("/completion", dict(kw, temperature=0.0, seed=1, skip_special_tokens=True))[1]
        check(on["content"] == base["content"], "explicit true == default (unchanged)")
    finally:
        s.stop()


def run_spf():
    print("spf: --sched-policy spf")
    import concurrent.futures as cf
    s = Server("-np", "1", "--no-slots", "--sched-policy", "spf")
    try:
        check("scheduling policy spf" in s.text(), "FIRED: startup log line 'scheduling policy spf'")
        # one slot busy with a long generation; three requests wait: long prompt first, short prompt last
        order = []
        def go(tag, n_prompt, delay, n_pred=1):
            time.sleep(delay)
            r = s.complete(toks(7 * len(tag), n_prompt), n_predict=n_pred, cache_prompt=False, ignore_eos=True)
            order.append(tag)
        with cf.ThreadPoolExecutor(4) as ex:
            fs = [ex.submit(go, "busy", 16, 0.0, 3000), ex.submit(go, "long", 900, 0.5), ex.submit(go, "mid", 400, 0.7), ex.submit(go, "short", 30, 0.9)]
            for f in fs:
                f.result()
        txt = s.text()
        check("sched-policy spf: started task" in txt, "FIRED: log line 'sched-policy spf: started task ...' per decision")
        check("(REORDERED)" in txt, "spf reordered at least one decision relative to fcfs")
        check(order.index("short") < order.index("mid") < order.index("long"), f"completion order short < mid < long under spf (got {order})")
    finally:
        s.stop()


def run_identity():
    """Flag-off identity: the same requests, with none of the new flags / fields, answer the same on the baseline build and on this one.
    Needs the baseline server in env BASE_BIN (a build of the commit before the picks); skipped without it."""
    print("identity: flags off == baseline build")
    base = os.environ.get("BASE_BIN")
    if not base:
        print("  SKIP (set BASE_BIN to the baseline llama-server)")
        return
    global BIN
    reqs = [
        ("/completion", {"prompt": toks(0, 40), "n_predict": 24, "temperature": 0.0, "seed": 3}),
        ("/completion", {"prompt": toks(7, 60), "n_predict": 24, "temperature": 0.8, "seed": 11, "top_k": 20}),
        ("/completion", {"prompt": toks(5, 30), "n_predict": 1, "temperature": 0.0, "prompt_logprobs": True}),
        ("/completion", {"prompt": toks(9, 30), "n_predict": 8, "temperature": 0.0, "n_probs": 3, "priority": 2}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "t10 t11"}], "max_tokens": 12, "temperature": 0.0, "seed": 5}),
    ]
    answers = {}
    for label, binary in (("base", base), ("picks", BIN)):
        BIN_saved, BIN = BIN, binary
        s = Server("-np", "2", "--no-slots", "--kv-unified", "--radix-cache", "--radix-cache-slots", "4")
        BIN = BIN_saved
        try:
            out = []
            for path, body in reqs:
                st, js = s.post(path, body)
                js.pop("timings", None); js.pop("id", None); js.pop("created", None); js.pop("system_fingerprint", None); js.pop("id_slot", None)
                js.pop("generation_settings", None)  # lists the new fields on the new build
                out.append((st, json.dumps(js, sort_keys=True)))
            answers[label] = out
        finally:
            s.stop()
    for i, (a, b) in enumerate(zip(answers["base"], answers["picks"])):
        check(a == b, f"request {i} ({reqs[i][0]}): identical answer on the baseline build and this build")


for name, fn in (("namespace", run_namespace), ("radix", run_radix), ("sched", run_sched), ("logprob", run_logprob), ("promptids", run_promptids), ("skipspecial", run_skipspecial), ("spf", run_spf), ("identity", run_identity)):
    if name in WANT:
        try:
            fn()
        except Exception as e:  # a server that dies or lacks the field is a failed check, not a crash of the harness
            check(False, f"{name}: {type(e).__name__}: {str(e).strip().splitlines()[-1][:200] if str(e).strip() else ''}")
print(("PASS" if not fails else "FAIL") + f" -- {len(fails)} failure(s)")
sys.exit(1 if fails else 0)
