# REWRITE-SAFE: this test writes only NEW PNG bytes in memory and server logs under temp/logs; it reads no data store.
"""E2E served checks for --mmproj-embd-cache (byte-budgeted LRU of vision-encoder outputs).

Usage: pythonw e2e_mmproj_embd_cache_served.py <llama-server.exe> <model.gguf> <mmproj.gguf> [base-port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Row: S69-tabbyapi-PICK-MTMD-IMAGE-EMBEDDING-LRU-CACHE-2026-09-29-01
Model: ggml-org/tinygemma3 (random weights, CPU). The encode counter (llamacpp:mmproj_encodes_total) counts image chunks that really ran
through the vision tower; the cache counters are hits / misses / evictions / not_stored.
"""
import base64
import json
import os
import random
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

EXE, MODEL, MMPROJ = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = int(sys.argv[4]) if len(sys.argv) > 4 else 18210
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def http(port, method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def jb(text):
    try:
        return json.loads(text)
    except ValueError:
        return {}


def png(seed, w=64, h=64):
    rnd = random.Random(seed)
    raw = b"".join(b"\x00" + bytes(rnd.randrange(256) for _ in range(w * 3)) for _ in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def data_uri(seed):
    return "data:image/png;base64," + base64.b64encode(png(seed)).decode()


class Srv:
    def __init__(self, port, args, tag):
        self.port = port
        self.log = os.path.join(LOGDIR, "mmcache_%s_%d.log" % (tag, port))
        self.f = open(self.log, "wb")
        self.p = subprocess.Popen([EXE, "--host", "127.0.0.1", "--port", str(port), "-m", MODEL, "--mmproj", MMPROJ, "-c", "4096", "-np", "1",
                                   "-t", "4", "--metrics", "--no-warmup", "--no-cache-prompt"] + args, stdout=self.f,
                                  stderr=subprocess.STDOUT, creationflags=0x08000000)

    def wait_ready(self, timeout=240):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                raise RuntimeError("server exited rc=%s, log %s" % (self.p.returncode, self.log))
            try:
                st, _ = http(self.port, "GET", "/health", timeout=3)
                if st == 200:
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError("not ready, log %s" % self.log)

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(15)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.f.close()

    def logtext(self):
        self.f.flush()
        return open(self.log, "rb").read().decode("utf-8", "replace")


def metrics(port):
    st, tx = http(port, "GET", "/metrics")
    out = {}
    for line in tx.splitlines():
        if line.startswith("llamacpp:mmproj_"):
            k, v = line.split(" ", 1)
            out[k[len("llamacpp:mmproj_"):]] = float(v)
    return out


def ask(port, system, images, question="What is in the picture?", n=24):
    content = [{"type": "image_url", "image_url": {"url": u}} for u in images] + [{"type": "text", "text": question}]
    st, tx = http(port, "POST", "/v1/chat/completions", {"messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                                                         "max_tokens": n, "temperature": 0, "seed": 1, "logprobs": True, "top_logprobs": 3})
    j = jb(tx)
    if st != 200:
        return None, tx[:200]
    lp = [(t["token"], t["logprob"], [(c["token"], c["logprob"]) for c in t.get("top_logprobs", [])]) for t in j["choices"][0]["logprobs"]["content"]]
    return j["choices"][0]["message"]["content"], lp


I1, I2, I3 = data_uri(1), data_uri(2), data_uri(3)

# ------------------------------------------------------------------------------------------ E0: cache off (today's behaviour)
s = Srv(BASE, [], "off")
try:
    s.wait_ready()
    m0 = metrics(BASE)
    a, _ = ask(BASE, "You are terse.", [I1])
    b_off, lp_off = ask(BASE, "You are verbose.", [I1])
    m1 = metrics(BASE)
    check("E0 counters exist and the encode counter is exported", "encodes_total" in m1, json.dumps(m1))
    check("E0 cache off: the same image under a different system prompt is encoded AGAIN (fail-before)", m1.get("encodes_total", 0) - m0.get("encodes_total", 0) == 2,
          "encodes=%s" % (m1.get("encodes_total", 0) - m0.get("encodes_total", 0)))
    check("E0 cache off: no hit is ever counted", m1.get("embd_cache_hits_total", 0) == 0)
finally:
    s.stop()

# ------------------------------------------------------------------------------------------ E1: cache on
s = Srv(BASE + 1, ["--mmproj-embd-cache", "64"], "on")
try:
    s.wait_ready()
    m0 = metrics(BASE + 1)
    a, _ = ask(BASE + 1, "You are terse.", [I1])
    m1 = metrics(BASE + 1)
    b_on, lp_on = ask(BASE + 1, "You are verbose.", [I1])
    m2 = metrics(BASE + 1)
    check("E1 first sight of the image: encoded once, cached", m1["encodes_total"] - m0["encodes_total"] == 1 and m1["embd_cache_entries"] == 1,
          json.dumps({k: m1[k] for k in ("encodes_total", "embd_cache_entries", "embd_cache_bytes")}))
    check("E1 same image under a different prefix: NOT encoded again, one hit", m2["encodes_total"] == m1["encodes_total"] and m2["embd_cache_hits_total"] - m1["embd_cache_hits_total"] == 1,
          "encodes %s -> %s, hits %s -> %s" % (m1["encodes_total"], m2["encodes_total"], m1["embd_cache_hits_total"], m2["embd_cache_hits_total"]))
    check("E1 the cached request's greedy output is identical to the uncached server's", b_on == b_off and b_on is not None, "%r vs %r" % (b_on, b_off))
    check("E1 and so are the per-token logprobs (a degenerate answer alone would prove little): identical to the last float",
          lp_on == lp_off and len(lp_on) > 0, "n=%d" % len(lp_on or []))
    c, _ = ask(BASE + 1, "You are terse.", [I2])
    m3 = metrics(BASE + 1)
    check("E1 a different image of the same size is NOT served from the cache", m3["encodes_total"] - m2["encodes_total"] == 1 and m3["embd_cache_entries"] == 2,
          "encodes %s -> %s" % (m2["encodes_total"], m3["encodes_total"]))
    d, _ = ask(BASE + 1, "Another prefix entirely.", [I1, I2])
    m4 = metrics(BASE + 1)
    check("E1 two images, both cached: neither is encoded", m4["encodes_total"] == m3["encodes_total"] and m4["embd_cache_hits_total"] - m3["embd_cache_hits_total"] == 2,
          "encodes %s -> %s" % (m3["encodes_total"], m4["encodes_total"]))
    d2, _ = ask(BASE + 1, "Another prefix entirely.", [I1, I2])
    check("E1 the multi-image answer is the same on a repeat", d == d2)
    e1, _ = ask(BASE + 1, "Third prefix.", [I1, I3])
    m5 = metrics(BASE + 1)
    check("E1 one cached image and one new image in the same request: one hit, one encode", m5["embd_cache_hits_total"] - m4["embd_cache_hits_total"] - 2 == 1 and
          m5["encodes_total"] - m4["encodes_total"] == 1, json.dumps({k: m5[k] for k in ("encodes_total", "embd_cache_hits_total")}))
finally:
    s.stop()

# ------------------------------------------------------------------------------------------ E2: a tiny budget
s = Srv(BASE + 2, ["--mmproj-embd-cache", "1"], "tiny")
try:
    s.wait_ready()
    ask(BASE + 2, "p1", [I1])
    ask(BASE + 2, "p2", [I1])
    m2 = metrics(BASE + 2)
    # tinygemma3 emits a fixed number of tokens x n_embd floats; with a 1 MiB budget it may or may not fit, so assert the invariants
    check("E2 invariants: cache bytes never exceed the budget", m2["embd_cache_bytes"] <= 1024 * 1024, "bytes=%s" % m2["embd_cache_bytes"])
    check("E2 every image seen is either a hit or an encode", m2["embd_cache_hits_total"] + m2["encodes_total"] == 2,
          json.dumps({k: m2[k] for k in ("encodes_total", "embd_cache_hits_total", "embd_cache_not_stored_total")}))
finally:
    s.stop()

print("\n%d failure(s)" % len(fails) if fails else "\nALL PASS", flush=True)
sys.exit(1 if fails else 0)
