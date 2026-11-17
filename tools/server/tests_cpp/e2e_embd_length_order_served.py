# REWRITE-SAFE: this test writes only server logs under temp/logs; it reads no data store.
"""Served check of the length-ordered embedding admission (flag --embd-length-order, default on), on a real llama-server and a real embedding model.

Usage: pythonw e2e_embd_length_order_served.py <llama-server.exe> <embedding.gguf> [base-port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Row: S69-infinity-PICK-LENGTH-BUCKET-BATCH-2026-09-28-01.
Two servers, same model and flags except --no-embd-length-order, the same /v1/embeddings request (a burst of inputs of mixed length that does not fit one
micro-batch). Checks:
  O1  every response lists its inputs in REQUEST order (data[i].index == i) on both servers, and for the non-OAI /embedding shape too
  O2  the vectors are the same on both servers (cosine >= 0.99999 per input): the order of admission must not change any embedding
  O3  llamacpp:n_decode_total (the number of llama_decode calls the burst needed) with ordering is <= without; both numbers are printed, a strict
      improvement is reported but not required (it depends on the slot count and the length mix)
"""
import json
import math
import os
import subprocess
import sys
import time
import urllib.request

EXE, MODEL = sys.argv[1], sys.argv[2]
BASE = int(sys.argv[3]) if len(sys.argv) > 3 else 18420
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def http(port, path, body=None, timeout=300):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def metric(port, name):
    for ln in http(port, "/metrics").splitlines():
        if ln.startswith(name + " "):
            return float(ln.split()[1])
    return None


def run_server(port, extra, tag):
    log = os.path.join(LOGDIR, "embdorder_%s_%d.log" % (tag, port))
    f = open(log, "wb")
    p = subprocess.Popen([EXE, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port), "--embeddings", "--metrics", "-np", "6", "-c", "6144",
                          "-b", "512", "-ub", "512", "-t", "4", "--no-warmup"] + extra, stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000)
    t0 = time.time()
    while time.time() - t0 < 240:
        if p.poll() is not None:
            raise RuntimeError("server exited rc=%s, log %s" % (p.returncode, log))
        try:
            if json.loads(http(port, "/health", timeout=3)).get("status") == "ok":
                return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    raise RuntimeError("not ready, log " + log)


def text(n_words, seed):
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "sigma"]
    return " ".join(words[(i * 7 + seed) % len(words)] for i in range(n_words))


# a burst that does not fit one 512-token micro-batch and whose arrival order interleaves long and short inputs
LENGTHS = [300, 12, 280, 18, 260, 22, 200, 9, 150, 30, 120, 6]
INPUTS = [text(n, i) for i, n in enumerate(LENGTHS)]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


results = {}
for tag, extra, port in (("unordered", ["--no-embd-length-order"], BASE), ("ordered", [], BASE + 1)):
    p = run_server(port, extra, tag)
    try:
        d0 = metric(port, "llamacpp:n_decode_total") or 0.0
        oai = json.loads(http(port, "/v1/embeddings", {"input": INPUTS}))
        d1 = metric(port, "llamacpp:n_decode_total") or 0.0
        native = json.loads(http(port, "/embedding", {"content": INPUTS}))
        results[tag] = {"vec": [e["embedding"] for e in oai["data"]], "idx": [e["index"] for e in oai["data"]], "decodes": d1 - d0,
                        "native_idx": [e["index"] for e in native], "tokens": oai["usage"]["prompt_tokens"]}
    finally:
        p.terminate()
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()

for tag in ("unordered", "ordered"):
    r = results[tag]
    check("O1 %s: OAI indices are request order" % tag, r["idx"] == list(range(len(INPUTS))), "idx=%s" % r["idx"])
    check("O1 %s: native /embedding indices are request order" % tag, r["native_idx"] == list(range(len(INPUTS))), "idx=%s" % r["native_idx"])
sims = [cos(a, b) for a, b in zip(results["unordered"]["vec"], results["ordered"]["vec"])]
check("O2 the vectors are the same with and without ordering", min(sims) >= 0.99999, "min cosine=%.7f over %d inputs, %d prompt tokens" % (min(sims), len(sims), results["ordered"]["tokens"]))
du, do = results["unordered"]["decodes"], results["ordered"]["decodes"]
check("O3 ordering does not need MORE llama_decode calls", do <= du, "n_decode: unordered=%d ordered=%d%s" % (du, do, "  (strictly fewer)" if do < du else "  (no change)"))
print("FAILED: %d checks: %s" % (len(fails), fails) if fails else "ALL PASS", flush=True)
sys.exit(1 if fails else 0)
