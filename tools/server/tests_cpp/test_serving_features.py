#!/usr/bin/env python3
"""Server-level tests of the wired serving features (docs/agent-reports/wire-serving-features.md).

Every test starts a REAL llama-server on a random-weight model (scripts/gen-tiny-lora-test-assets.py) and sends real HTTP requests.
Each feature is checked three ways: flag OFF is bit-identical to a server that never heard of the feature, flag ON shows the feature
FIRED (an always-on log line / counter in the server log), and the result changes where the feature is meant to change it.

Usage:  python3.13 tools/server/tests_cpp/test_serving_features.py --server build/bin/llama-server [--only lora,kobold] [--keep DIR]
Needs numpy + gguf-py (to write the assets). Exit 77 = skipped (no numpy / no server binary), 1 = a check failed.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

FAILS = []


def check(cond, what):
    print(("PASS  " if cond else "FAIL  ") + what, flush=True)
    if not cond:
        FAILS.append(what)
    return cond


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Server:
    def __init__(self, exe, model, workdir, name, extra=(), n_parallel=4, ctx=4096):
        self.port = free_port()
        self.log_path = os.path.join(workdir, name + ".log")
        self.log_f = open(self.log_path, "w")
        cmd = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(self.port), "-c", str(ctx), "-np", str(n_parallel),
               "-t", "2", "--no-warmup", "--no-webui", "-fa", "off", "--cache-ram", "0"] + list(extra)
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=self.log_f, stderr=subprocess.STDOUT)
        self.base = "http://127.0.0.1:%d" % self.port
        t0 = time.time()
        while time.time() - t0 < 90:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited early:\n" + self.log())
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.3)
        raise RuntimeError("server did not become healthy:\n" + self.log())

    def log(self):
        self.log_f.flush()
        with open(self.log_path) as f:
            return f.read()

    def post(self, path, body, timeout=120):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            except Exception:
                return e.code, {}

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log_f.close()


BASE_REQ = {"prompt": "once upon a time", "n_predict": 24, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
            "return_tokens": True, "n_probs": 1}


def complete(srv, extra=None, base=None):
    body = dict(base or BASE_REQ)
    body.update(extra or {})
    st, r = srv.post("/completion", body)
    return st, r


def toks(r):
    return r.get("tokens")


def logprobs(r):
    return [c.get("logprob", c.get("probs", [{}])[0].get("logprob")) for c in r.get("completion_probabilities", [])]


def maxdiff(a, b):
    if not a or not b:
        return 1e9
    return max([abs(x - y) for x, y in zip(a, b)] + [0.0])   # over the common prefix of the two lists


def cache_n(r):
    return r.get("timings", {}).get("cache_n", -1)


# ------------------------------------------------------------------------------------------------------------------------------
# LoRA pool + heterogeneous batching
# ------------------------------------------------------------------------------------------------------------------------------
def test_lora(exe, model, assets, work):
    print("\n=== LoRA adapter pool (--lora-pool-dir) and heterogeneous batching (--lora-hetero-batch) ===")
    pool = os.path.join(work, "pool")
    os.makedirs(pool, exist_ok=True)
    for n in ("lora-a", "lora-b", "lora-c"):
        shutil.copy(os.path.join(assets, n + ".gguf"), os.path.join(pool, n + ".gguf"))

    # --- reference: a server with NO pool flag at all ---------------------------------------------------------------------------
    off = Server(exe, model, work, "lora-off")
    st, base = complete(off)
    check(st == 200 and toks(base), "baseline completion works (%d tokens)" % len(toks(base) or []))
    st, off_named = complete(off, {"lora_adapter": "lora-a"})
    check(st == 200 and toks(off_named) == toks(base) and logprobs(off_named) == logprobs(base),
          "flag OFF: a request naming lora_adapter is IGNORED -- tokens and logprobs bit-identical to the baseline")
    check("lora-pool" not in off.log(), "flag OFF: no lora-pool log line at all")
    off.stop()

    # --- pool ON, max 2 adapters ---------------------------------------------------------------------------------------------
    srv = Server(exe, model, work, "lora-pool", ["--lora-pool-dir", pool, "--lora-pool-max", "2"], ctx=400000)
    check("lora-pool: dir=" in srv.log(), "flag ON: server log announces the pool")
    st, plain = complete(srv)
    check(st == 200 and toks(plain) == toks(base) and logprobs(plain) == logprobs(base),
          "flag ON, request without lora_adapter: bit-identical to the baseline (the pool is not touched)")
    check("acquire" not in srv.log(), "flag ON, no adapter named: pool not consulted")

    st, a1 = complete(srv, {"lora_adapter": "lora-a"})
    check(st == 200, "lora_adapter=lora-a answered 200")
    check(toks(a1) != toks(base), "the adapter changes the continuation (a: %s vs base: %s)" % (toks(a1)[:6], toks(base)[:6]))
    check(maxdiff(logprobs(a1), logprobs(base)) > 1e-3, "the adapter changes the logprobs (max diff %.3g)" % maxdiff(logprobs(a1), logprobs(base)))
    check(re.search(r"lora-pool: acquire 'lora-a' -> ok loaded .*loads 1", srv.log()) is not None, "log: acquire 'lora-a' -> ok loaded, loads 1")
    st, a2 = complete(srv, {"lora_adapter": "lora-a"})
    check(toks(a2) == toks(a1) and logprobs(a2) == logprobs(a1), "same adapter twice -> identical result")
    check(re.search(r"lora-pool: acquire 'lora-a' -> ok hit .*hits 1", srv.log()) is not None, "log: second acquire is a hit, hits 1")

    st, a_half = complete(srv, {"lora_adapter": "lora-a", "lora_adapter_scale": 0.5})
    check(st == 200 and toks(a_half) != toks(a1) or maxdiff(logprobs(a_half), logprobs(a1)) > 1e-3, "lora_adapter_scale changes the result")

    st, b1 = complete(srv, {"lora_adapter": "lora-b"})
    check(st == 200 and toks(b1) != toks(a1), "a different adapter gives a different continuation")
    # a third adapter with max 2: the least recently used (lora-a, last used before lora-b) is evicted
    st, c1 = complete(srv, {"lora_adapter": "lora-c"})
    check(st == 200 and re.search(r"evictions 1 \(\+1\)", srv.log()) is not None, "third adapter evicts one (log: evictions 1 (+1))")
    st, a3 = complete(srv, {"lora_adapter": "lora-a"})
    check(st == 200 and toks(a3) == toks(a1) and logprobs(a3) == logprobs(a1),
          "adapter evicted and loaded again gives the SAME result as before eviction")
    check(len(re.findall(r"acquire 'lora-a' -> ok loaded", srv.log())) == 2, "lora-a was loaded twice (1st load + reload after eviction)")

    # missing / invalid names
    st, r = complete(srv, {"lora_adapter": "nope"})
    check(st == 404, "unknown adapter -> HTTP 404 (got %s)" % st)
    st, r = complete(srv, {"lora_adapter": "../etc/passwd"})
    check(st == 400, "path-traversal name refused -> HTTP 400 (got %s)" % st)
    st, r = complete(srv, {"lora_adapter": "a/b"})
    check(st == 400, "name with a slash refused -> HTTP 400 (got %s)" % st)

    # prompt-cache identity: cached KV of an adapter must not be served for the reloaded (different serial) adapter
    cached = dict(BASE_REQ, cache_prompt=True, prompt="once upon a time there was a little girl named lily", n_predict=4)
    st, k1 = complete(srv, {"lora_adapter": "lora-a"}, cached)
    st, k2 = complete(srv, {"lora_adapter": "lora-a"}, cached)
    check(cache_n(k2) > 0, "same adapter, cache_prompt: the second request reuses the cache (timings.cache_n=%s)" % cache_n(k2))
    check(toks(k2) == toks(k1), "cache reuse keeps the result")
    complete(srv, {"lora_adapter": "lora-b"}, cached)
    complete(srv, {"lora_adapter": "lora-c"}, cached)  # evicts lora-a (LRU)
    st, k3 = complete(srv, {"lora_adapter": "lora-a"}, cached)
    check(toks(k3) == toks(k1) and cache_n(k3) == 0,
          "adapter reloaded under a new serial: KV computed before the eviction is NOT reused (timings.cache_n=%s) and the result is unchanged" % cache_n(k3))

    # refusal when every resident adapter is pinned by a running request
    long_req = dict(BASE_REQ, n_predict=6000, ignore_eos=True, stream=True, n_probs=0, return_tokens=False)
    conns = []

    def hold(name):
        body = dict(long_req, lora_adapter=name)
        req = urllib.request.Request(srv.base + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        conns.append(urllib.request.urlopen(req, timeout=300))

    n_before = len(re.findall(r"acquire 'lora-[ab]'", srv.log()))
    h = [threading.Thread(target=hold, args=(n,), daemon=True) for n in ("lora-a", "lora-b")]
    for t in h:
        t.start()
    t0 = time.time()
    while time.time() - t0 < 20 and len(re.findall(r"acquire 'lora-[ab]'", srv.log())) < n_before + 2:
        time.sleep(0.02)
    time.sleep(0.3)   # both holders are now generating 6000 tokens (several seconds) and hold a pin each
    st, r = complete(srv, {"lora_adapter": "lora-c"})
    check(st == 503, "both resident adapters pinned by running requests: the third gets HTTP 503 (got %s)" % st)
    check("REFUSED(no_room)" in srv.log(), "log: acquire 'lora-c' -> REFUSED(no_room)")
    for c in conns:
        c.close()
    time.sleep(1.5)
    st, r = complete(srv, {"lora_adapter": "lora-c"})
    check(st == 200, "after the holders disconnect, the same request is served (got %s)" % st)
    srv.stop()

    # --- heterogeneous batch ---------------------------------------------------------------------------------------------------
    het = Server(exe, model, work, "lora-het", ["--lora-pool-dir", pool, "--lora-pool-max", "4", "--lora-hetero-batch"])
    check("lora-hetero-batch: slots with different LoRA adapters share one decode batch" in het.log(), "flag ON: server log announces hetero batching")
    long_cfg = dict(BASE_REQ, n_predict=160, ignore_eos=True)
    names = ["lora-a", "lora-b", None, "lora-c"]
    alone = {}
    for n in names:
        st, r = complete(het, {"lora_adapter": n} if n else {}, long_cfg)
        alone[n] = r
    check(len(het.log().split("lora-hetero: decode batch")) == 1, "sequential single requests never produce a heterogeneous batch")
    res = {}

    def run(n):
        st, r = complete(het, {"lora_adapter": n} if n else {}, long_cfg)
        res[n] = r

    th = [threading.Thread(target=run, args=(n,)) for n in names]
    for t in th:
        t.start()
    for t in th:
        t.join()
    mixed = het.log().count("lora-hetero: decode batch")
    check(mixed > 0, "concurrent requests with different adapters shared decode batches (%d heterogeneous batches in the log)" % mixed)
    for n in names:
        same = toks(res[n]) == toks(alone[n])
        d = maxdiff(logprobs(res[n]), logprobs(alone[n]))
        check(same and d < 1e-3, "batched together == run alone for %s (tokens equal=%s, max |logprob diff|=%.2e)" % (n or "no adapter", same, d))
    het.stop()

    # --- hetero batching OFF, same concurrent traffic: still correct, never a mixed batch ---------------------------------------
    nh = Server(exe, model, work, "lora-nohet", ["--lora-pool-dir", pool, "--lora-pool-max", "4"])
    res2 = {}

    def run2(n):
        st, r = complete(nh, {"lora_adapter": n} if n else {}, long_cfg)
        res2[n] = r

    th = [threading.Thread(target=run2, args=(n,)) for n in names]
    for t in th:
        t.start()
    for t in th:
        t.join()
    check("lora-hetero: decode batch" not in nh.log(), "flag OFF: no heterogeneous batch is ever formed")
    for n in names:
        check(toks(res2[n]) == toks(alone[n]), "flag OFF, concurrent: %s equals the run-alone result" % (n or "no adapter"))
    nh.stop()


# ------------------------------------------------------------------------------------------------------------------------------
# koboldcpp port: banned_tokens (vocab-substring ban) and the prompt-reuse planner
# ------------------------------------------------------------------------------------------------------------------------------
def gen_vocab():
    spec = importlib.util.spec_from_file_location("gen_assets", os.path.join(ROOT, "scripts", "gen-tiny-lora-test-assets.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g.vocab()[0]


def piece_of(srv, tok):
    st, r = srv.post("/detokenize", {"tokens": [tok]})
    return r.get("content", "")


def test_kobold(exe, model, work):
    print("\n=== koboldcpp port: --token-ban-substr and --prompt-reuse-plan ===")
    off = Server(exe, model, work, "kcpp-off")
    # the random model emits mostly raw bytes; bias two real words up so the greedy continuation is ASCII text we can ban
    st, tk = off.post("/tokenize", {"content": " the a", "add_special": False})
    the_id, a_id = tk["tokens"][0], tk["tokens"][1]
    biased = dict(BASE_REQ, logit_bias=[[the_id, 60.0], [a_id, 59.0]], n_predict=24)
    st, base = complete(off, None, biased)
    ids = toks(base)
    text = base.get("content", "")
    print("   baseline text:", repr(text[:60]))
    check(st == 200 and ids and "the" in text, "baseline completion (%d tokens) contains the word 'the'" % len(ids))
    ban = "the"
    st, off_req = complete(off, {"banned_tokens": [ban]}, biased)
    check(st == 200 and toks(off_req) == ids and logprobs(off_req) == logprobs(base),
          "flag OFF: banned_tokens is ignored -- tokens and logprobs bit-identical to the baseline")
    off.stop()

    on = Server(exe, model, work, "kcpp-on", ["--token-ban-substr"])
    st, plain = complete(on, None, biased)
    check(toks(plain) == ids and logprobs(plain) == logprobs(base), "flag ON, field absent: bit-identical to the baseline")
    st, banned = complete(on, {"banned_tokens": [ban]}, biased)
    check(st == 200 and toks(banned) != ids, "flag ON: banned_tokens=[%r] changes the continuation" % ban)
    bpieces = [piece_of(on, t) for t in toks(banned)]
    pieces = [piece_of(on, t) for t in ids]
    check(all(ban not in p.lower() for p in bpieces), "no emitted token contains %r (case-insensitive substring), %d tokens checked" % (ban, len(bpieces)))
    check(any(ban in p.lower() for p in pieces), "(control) the baseline DID emit a token containing %r" % ban)
    n_hits = sum(1 for t in range(0, len(gen_vocab())) if ban in piece_of(on, t).lower())
    check(n_hits >= 1, "the substring matches %d vocabulary entries (the ban is a substring ban, not an equality ban)" % n_hits)
    # substring semantics: banning only "he" must still remove the token " the"
    st, sub = complete(on, {"banned_tokens": ["he"]}, biased)
    sp = [piece_of(on, t) for t in toks(sub)]
    check(st == 200 and all("he" not in p.lower() for p in sp) and any("he" in p.lower() for p in pieces),
          "banning the substring 'he' removes the token ' the' (no emitted token contains 'he')")
    # phrase path: a string of more than one token goes through banned_strings (rewind and resample)
    phrase = "the  the"
    st, ph = complete(on, {"banned_tokens": [phrase]}, biased)
    check(st == 200 and phrase in text.lower() and phrase not in ph.get("content", "").lower(),
          "phrase %r (2 tokens) is banned like banned_strings: absent from the output (%r), present in the baseline" % (phrase, ph.get("content", "")[:40]))
    on.stop()

    # --- prompt reuse planner -----------------------------------------------------------------------------------------------------
    p1 = "once upon a time there was a little girl named lily and she went to the park"
    p2 = p1 + " to play with her friend"
    req1 = dict(BASE_REQ, prompt=p1, n_predict=4, cache_prompt=True, n_probs=0, return_tokens=False)
    req2 = dict(req1, prompt=p2)
    plain_srv = Server(exe, model, work, "reuse-off", n_parallel=1)
    plain_srv.post("/completion", req1)
    st, r_plain = plain_srv.post("/completion", req2)
    cached_plain = cache_n(r_plain)
    keep = r_plain.get("timings", {}).get("prompt_n", 0) + 5   # leaves prompt_n + 5 tokens to evaluate => reuse is capped 5 tokens below the common prefix
    check(st == 200 and cached_plain > 4, "baseline server reuses the shared prefix (timings.cache_n=%s)" % cached_plain)
    plain_srv.stop()

    for name, flags, expect in (
        ("plan-default", ["--prompt-reuse-plan"], lambda c: 0 < c <= cached_plain),
        ("plan-min", ["--prompt-reuse-plan", "--prompt-reuse-min", "1000"], lambda c: c == 0),
        ("plan-keep", ["--prompt-reuse-plan", "--prompt-reuse-keep-input", str(keep)], lambda c: 0 < c < cached_plain),
    ):
        s = Server(exe, model, work, "reuse-" + name, flags, n_parallel=1)
        s.post("/completion", req1)
        st, r = s.post("/completion", req2)
        c = cache_n(r)
        check(st == 200 and expect(c), "%s: timings.cache_n=%s (baseline %s)" % (" ".join(flags), c, cached_plain))
        check(re.search(r"prompt-reuse-plan: common prefix \d+ -> planner reuses \d+", s.log()) is not None, "%s: log line 'prompt-reuse-plan: common prefix N -> planner reuses M'" % name)
        # greedy decoding: how much of the prompt was recomputed does not change the text
        check(r.get("content") == r_plain.get("content"), "%s: the generated text equals the baseline's" % name)
        s.stop()


# ------------------------------------------------------------------------------------------------------------------------------
# ASR decode quality on /v1/audio/transcriptions (synthetic 16 kHz audio, random-weight audio projector + LLM)
# ------------------------------------------------------------------------------------------------------------------------------
def write_tone(path, seconds=3):
    import math
    import random
    import struct
    import wave
    rng = random.Random(1)
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000) + rng.gauss(0, 800))) for i in range(16000 * seconds)))
    w.close()


def transcribe(srv, wav, **fields):
    import uuid
    form = {"temperature": "0", "max_tokens": "60", "response_format": "json"}
    form.update({k: str(v) for k, v in fields.items()})
    b = uuid.uuid4().hex
    body = b""
    for k, v in form.items():
        body += ('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n' % (b, k, v)).encode()
    body += ('--%s\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n' % b).encode()
    body += open(wav, "rb").read() + ("\r\n--%s--\r\n" % b).encode()
    req = urllib.request.Request(srv.base + "/v1/audio/transcriptions", data=body, headers={"Content-Type": "multipart/form-data; boundary=" + b})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        print("   HTTP", e.code, e.read().decode()[:300])
        return e.code, {}


def test_asr(exe, model, assets, work):
    print("\n=== ASR decode quality (--asr-quality) on /v1/audio/transcriptions ===")
    mm = os.path.join(assets, "mmproj-tiny-audio.gguf")
    if not os.path.exists(mm):
        spec = importlib.util.spec_from_file_location("gen_mm", os.path.join(ROOT, "scripts", "gen-tiny-audio-mmproj.py"))
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        sys.argv = ["gen", assets]
        g.main()
    wav = os.path.join(work, "tone.wav")
    write_tone(wav)
    common = ["--mmproj", mm, "--chat-template", "chatml"]
    off = Server(exe, model, work, "asr-off", common, n_parallel=1, ctx=4096)
    st, base = transcribe(off, wav)
    check(st == 200 and base.get("text"), "flag OFF: a transcription of the synthetic audio works (%d output tokens)" % base.get("usage", {}).get("output_tokens", -1))
    check("asr_quality" not in base and "asr_stats" not in base, "flag OFF: the reply carries no asr_quality / asr_stats")
    st, base2 = transcribe(off, wav)
    check(base2.get("text") == base.get("text"), "flag OFF: the same request twice is deterministic (the reference for the checks below)")
    check("asr-quality" not in off.log(), "flag OFF: no asr-quality log line")
    off.stop()

    # gates that always pass: ON must be bit-identical to OFF (one attempt, the same temperature and seed)
    easy = Server(exe, model, work, "asr-easy", common + ["--asr-quality", "--asr-compression-ratio-thold", "1000", "--asr-logprob-thold", "-1000", "--asr-entropy-thold", "0"],
                  n_parallel=1, ctx=4096)
    st, r = transcribe(easy, wav)
    q = r.get("asr_quality", {})
    check(st == 200 and r.get("text") == base.get("text"), "flag ON, gates that pass: the transcript equals the flag-OFF transcript")
    check(len(q.get("attempts", [])) == 1 and q.get("chosen") == 0, "flag ON, gates that pass: exactly one attempt (%s)" % [a.get("temperature") for a in q.get("attempts", [])])
    check("asr_stats" not in r, "internal asr_stats is not leaked to the client")
    easy.stop()

    # default gates: the greedy transcript of a random model loops, so its compression ratio is far above 2.4 and the server decodes again
    on = Server(exe, model, work, "asr-on", common + ["--asr-quality", "--asr-entropy-thold", "0", "--asr-logprob-thold", "-1000"], n_parallel=1, ctx=4096)
    st, r = transcribe(on, wav)
    q = r.get("asr_quality", {})
    att = q.get("attempts", [])
    print("   attempts:", [(a["temperature"], round(a["compression_ratio"], 2), a["reasons"], a["retried"]) for a in att], "chosen", q.get("chosen"))
    check(st == 200 and len(att) >= 2, "flag ON: the looping greedy transcript is rejected and decoded again (%d attempts)" % len(att))
    check(att and att[0]["temperature"] == 0.0 and att[0]["retried"] and (att[0]["reasons"] & 1), "first attempt: T=0, rejected for the compression ratio (%.1f > 2.4)" % (att[0]["compression_ratio"] if att else -1))
    check(len(att) >= 2 and abs(att[1]["temperature"] - 0.2) < 1e-6, "second attempt runs at the next temperature of the ladder (0.2)")
    check(len(re.findall(r"asr-quality: attempt \d+ T=", on.log())) == len(att) and "asr-quality: %d attempt(s)" % len(att) in on.log(), "log: one 'asr-quality: attempt N T=...' line per attempt and a summary line")
    chosen = att[q["chosen"]] if att else {}
    check(r.get("text") != base.get("text") or chosen.get("temperature") == 0.0,
          "the returned transcript is the chosen attempt's (T=%s)%s" % (chosen.get("temperature"), "" if r.get("text") != base.get("text") else " [same text as OFF]"))
    check(r.get("text") != base.get("text"), "ON visibly changes the decode: transcript differs from the flag-OFF one (%r vs %r)" % (r.get("text", "")[:24], base.get("text", "")[:24]))
    on.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--only", default="lora,kobold,asr")
    ap.add_argument("--keep", default=None, help="work directory to keep (default: a temp dir that is removed)")
    a = ap.parse_args()
    if not os.path.exists(a.server):
        print("server binary not found: SKIPPED")
        sys.exit(77)
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("numpy missing: SKIPPED")
        sys.exit(77)
    work = a.keep or tempfile.mkdtemp(prefix="serving-features-")
    os.makedirs(work, exist_ok=True)
    spec = importlib.util.spec_from_file_location("gen_assets", os.path.join(ROOT, "scripts", "gen-tiny-lora-test-assets.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assets = os.path.join(work, "assets")
    sys.argv = ["gen", assets]
    gen.main()
    model = os.path.join(assets, "tiny-llama.gguf")
    only = a.only.split(",")
    try:
        if "lora" in only:
            test_lora(a.server, model, assets, work)
        if "kobold" in only:
            test_kobold(a.server, model, work)
        if "asr" in only:
            test_asr(a.server, model, assets, work)
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    print("\n%d check(s) failed" % len(FAILS))
    for f in FAILS:
        print("  FAILED:", f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
