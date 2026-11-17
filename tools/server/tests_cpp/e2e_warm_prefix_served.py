# REWRITE-SAFE: this test writes only server logs and a small JSON file under temp/logs; it reads no data store.
"""E2E served checks for --warm-prefix-file (startup-warmed, eviction-pinned prefixes of the host prompt cache), on a real llama-server.

Usage: pythonw e2e_warm_prefix_served.py <llama-server.exe> <small.gguf> [base-port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Row: S69-rtp-llm-PICK-STARTUP-WARMED-PINNED-PREFIX-CACHE-2026-09-29-01.
Checks:
  W1  the warm-up runs after the model loads and the log says a prefix was pinned
  W2  a request that starts with the warmed prefix re-uses it: timings.cache_n covers the prefix (control: the same request against a server
      started WITHOUT the file has cache_n 0 on its first request)
  W3  after unrelated requests that exceed the cache budget, the warmed prefix is still re-used (control: the same churn on a server without pins
      loses it, cache_n 0)
  E1  --cache-ram 0 with the file stops start-up with an error (no silent cold start)
  E2  an unreadable / malformed file stops start-up with an error
  E3  a prefix that cannot fit in --warm-prefix-share stops start-up with an error
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXE, MODEL = sys.argv[1], sys.argv[2]
BASE = int(sys.argv[3]) if len(sys.argv) > 3 else 18380
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def post(port, path, body, timeout=180):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as e:
        return e.code, {}


def start(port, extra, tag):
    log = os.path.join(LOGDIR, "warm_%s_%d.log" % (tag, port))
    f = open(log, "wb")
    p = subprocess.Popen([EXE, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "-c", "2048", "-np", "1", "-t", "4", "--no-warmup"] + extra,
                         stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000)
    return p, log


def wait_ready(p, port, log, limit=240):
    t0 = time.time()
    while time.time() - t0 < limit:
        if p.poll() is not None:
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def read(log):
    try:
        return open(log, "rb").read().decode("utf-8", "replace")
    except OSError:
        return ""


PREFIX = ("You are a careful assistant for a legal research tool. " * 12).strip()     # about 130 tokens, a stable shared start
Q = lambda i: PREFIX + " Question %d: summarise the rule in one line." % i                # noqa: E731
spec = os.path.join(LOGDIR, "warm_prefix_spec.json")
with open(spec, "w", encoding="utf8") as fh:
    json.dump([{"id": "legal-system", "prompt": PREFIX}], fh)


def run_case(port, extra, tag, churn):
    p, log = start(port, extra, tag)
    out = {}
    try:
        ok = wait_ready(p, port, log)
        out["ready"] = ok
        out["log0"] = read(log)
        if not ok:
            return out
        time.sleep(1.5)                         # the warm-up thread pins after its task ends
        st, r = post(port, "/completion", {"prompt": Q(1), "n_predict": 2, "temperature": 0, "cache_prompt": True})
        out["first"] = (r.get("timings") or {}).get("cache_n")
        for i in range(churn):                  # unrelated prompts: one fits in --cache-ram (a state larger than the whole cache is never stored, so it
            # would evict nothing), two do not fit beside the warmed prefix -> every later one evicts something
            post(port, "/completion", {"prompt": ("unrelated topic %d " % i) * 40, "n_predict": 1, "temperature": 0, "cache_prompt": True})
        st, r = post(port, "/completion", {"prompt": Q(2), "n_predict": 2, "temperature": 0, "cache_prompt": True})
        out["after_churn"] = (r.get("timings") or {}).get("cache_n")
        out["log1"] = read(log)
    finally:
        p.kill()
        p.wait()
    return out


# --cache-ram 2, not 8: MEASURED 2026-10-17 on stories260K -c 2048 a 421-token state is 0.3 MiB ("warm prefix pinned ... 0.3 MiB"), so the 12
# churn states (~482 tokens each) total ~4 MiB. Under 8 MiB nothing was ever evicted: the control kept the prefix (cache_n 429) and W3 passed
# without the pin doing anything. 2 MiB holds the 0.3 MiB pin (share 50% = 1 MiB) plus ~5 churn states, so the churn must evict.
CACHE_RAM = "2"
warm = run_case(BASE, ["--cache-ram", CACHE_RAM, "--warm-prefix-file", spec], "pinned", churn=12)
check("W1 warm-up ran and a prefix was pinned", warm.get("ready") and "warm prefix pinned" in warm.get("log1", ""), "log excerpt=%r" % warm.get("log1", "")[-300:])
check("W2 the first request re-uses the warmed prefix (cache_n > 100)", (warm.get("first") or 0) > 100, "cache_n=%s" % warm.get("first"))
check("W3 the prefix survives churn that overflows --cache-ram", (warm.get("after_churn") or 0) > 100, "cache_n=%s" % warm.get("after_churn"))

cold = run_case(BASE + 1, ["--cache-ram", CACHE_RAM], "control", churn=12)
check("W2c control: without the file the first request has cache_n 0", cold.get("ready") and (cold.get("first") or 0) == 0, "cache_n=%s" % cold.get("first"))
check("W3c control: without pins the same churn loses the prefix", cold.get("ready") and (cold.get("after_churn") or 0) <= 1, "cache_n=%s" % cold.get("after_churn"))


def expect_exit(port, extra, tag, needle):
    p, log = start(port, extra, tag)
    t0 = time.time()
    while p.poll() is None and time.time() - t0 < 120:
        time.sleep(0.5)
    rc = p.poll()
    if rc is None:
        p.kill()
        p.wait()
    txt = read(log)
    return rc, needle in txt, txt[-250:]


rc, found, tail = expect_exit(BASE + 2, ["--cache-ram", "0", "--warm-prefix-file", spec], "ramoff", "--cache-ram 0")
check("E1 --cache-ram 0 with the file exits non-zero with a reason", rc not in (None, 0) and found, "rc=%s tail=%r" % (rc, tail))
bad = os.path.join(LOGDIR, "warm_prefix_bad.json")
with open(bad, "w", encoding="utf8") as fh:
    fh.write("{not json")
rc, found, tail = expect_exit(BASE + 3, ["--cache-ram", "8", "--warm-prefix-file", bad], "badfile", "expected a non-empty JSON list")
check("E2 a malformed file exits non-zero with a reason", rc not in (None, 0) and found, "rc=%s tail=%r" % (rc, tail))
big = os.path.join(LOGDIR, "warm_prefix_big.json")
with open(big, "w", encoding="utf8") as fh:
    json.dump([{"id": "huge", "prompt": "legal research context. " * 80}], fh)
rc, found, tail = expect_exit(BASE + 4, ["--cache-ram", "1", "--warm-prefix-share", "0.05", "--warm-prefix-file", big], "toobig", "do not fit")
check("E3 a prefix larger than the pin share exits non-zero with a reason", rc not in (None, 0) and found, "rc=%s tail=%r" % (rc, tail))

print("RESULT %d failed" % len(fails) if fails else "RESULT all passed")
sys.exit(1 if fails else 0)
