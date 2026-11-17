# REWRITE-SAFE: this test only writes NEW override JSON files under temp/logs; it reads no data store.
"""E2E served checks for the operator sampler-override layers (--sampler-overrides, POST /sampler-overrides).

Usage: pythonw e2e_sampler_overrides_served.py <llama-server.exe> <small.gguf> [base-port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Row: S69-tabbyapi-PICK-SAMPLER-FORCE-ADDITIVE-OVERRIDE-LAYERS-2026-09-29-01
Fail-before: S0 runs the same requests on a server WITHOUT overrides: max_tokens=64 yields 64 tokens, which is what a forced 16 must change.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXE, MODEL = sys.argv[1], sys.argv[2]
BASE = int(sys.argv[3]) if len(sys.argv) > 3 else 18170
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def http(port, method, path, body=None, key=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data, method=method, headers=headers)
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


class Srv:
    def __init__(self, port, args, tag):
        self.port = port
        self.log = os.path.join(LOGDIR, "overrides_%s_%d.log" % (tag, port))
        self.f = open(self.log, "wb")
        self.p = subprocess.Popen([EXE, "--host", "127.0.0.1", "--port", str(port), "-m", MODEL, "-c", "1024", "-np", "1", "-t", "4"] + args,
                                  stdout=self.f, stderr=subprocess.STDOUT, creationflags=0x08000000)

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


def completion(port, extra, key=None):
    body = {"prompt": "Count: one two three four five six seven eight nine ten", "ignore_eos": True, "cache_prompt": False}
    body.update(extra)
    st, tx = http(port, "POST", "/completion", body, key=key)
    return st, jb(tx)


def chat(port, extra, key=None):
    body = {"messages": [{"role": "user", "content": "Write a long story about a fox."}], "ignore_eos": True}
    body.update(extra)
    st, tx = http(port, "POST", "/v1/chat/completions", body, key=key)
    return st, jb(tx)


def settings(resp):
    return resp.get("generation_settings", {})


OV = os.path.join(LOGDIR, "overrides_global_%d.json" % BASE)
MV = os.path.join(LOGDIR, "overrides_model_%d.json" % BASE)
with open(OV, "w") as f:
    json.dump({
        "n_predict": {"override": 16, "force": True},
        "stop": {"override": ["ZZZSTOP"], "additive": True},
        "temperature": {"override": 0.0},
        "top_k": {"override": 1, "force": True},
        "bogus_field": {"override": 1},
    }, f)
with open(MV, "w") as f:
    json.dump({"top_k": {"override": 7, "force": True}}, f)  # the model layer replaces the global top_k

# ------------------------------------------------------------------------------------------ S0: control, no overrides
s0 = Srv(BASE, [], "s0")
try:
    s0.wait_ready()
    st, r = completion(BASE, {"n_predict": 64})
    check("S0 control: no overrides, n_predict=64 yields 64 tokens", st == 200 and r.get("tokens_predicted") == 64, "tokens=%s" % r.get("tokens_predicted"))
    st, r = chat(BASE, {"max_tokens": 64})
    check("S0 control: chat max_tokens=64 yields 64 tokens", st == 200 and r.get("usage", {}).get("completion_tokens") == 64,
          "usage=%s" % r.get("usage"))
    st, tx = http(BASE, "GET", "/sampler-overrides")
    check("S0 GET /sampler-overrides on a server with none: empty layers", st == 200 and jb(tx).get("effective") == {}, tx[:120])
finally:
    s0.stop()

# ------------------------------------------------------------------------------------------ S1: files at start
P = BASE + 1
s1 = Srv(P, ["--sampler-overrides", OV, "--sampler-overrides-model", MV], "s1")
try:
    s1.wait_ready()
    log = s1.logtext()
    check("S1 unknown key warned at load", "bogus_field" in log, "")
    check("S1 load line logged for both layers", "global layer" in log and "model layer" in log)

    st, r = completion(P, {"n_predict": 64})
    check("S1a force: n_predict=64 is cut to 16", st == 200 and r.get("tokens_predicted") == 16, "tokens=%s" % r.get("tokens_predicted"))
    st, r = chat(P, {"max_tokens": 64})
    check("S1a force through an alias: chat max_tokens=64 is cut to 16, finish_reason length",
          st == 200 and r.get("usage", {}).get("completion_tokens") == 16 and r["choices"][0].get("finish_reason") == "length",
          "usage=%s finish=%s" % (r.get("usage"), r.get("choices", [{}])[0].get("finish_reason")))
    st, r = chat(P, {"max_completion_tokens": 4, "max_tokens": 50})
    check("S1a force beats every alias at once", r.get("usage", {}).get("completion_tokens") == 16, "usage=%s" % r.get("usage"))

    st, r = completion(P, {"n_predict": 4})
    check("S1a force raises a smaller request too (force is force)", r.get("tokens_predicted") == 16, "tokens=%s" % r.get("tokens_predicted"))

    # default: temperature override 0.0 (a falsy default still fills in), request value wins when sent
    st, r = completion(P, {})
    check("S1b default fills in when the request omits the field (temperature 0.0)", settings(r).get("temperature") == 0.0,
          "temperature=%s" % settings(r).get("temperature"))
    st, r = completion(P, {"temperature": 0.2})
    check("S1b request value wins over an un-forced override", abs(settings(r).get("temperature", -1) - 0.2) < 1e-6,
          "temperature=%s" % settings(r).get("temperature"))
    st, r = completion(P, {"temperature": None})
    check("S1b null counts as omitted", settings(r).get("temperature") == 0.0, "temperature=%s" % settings(r).get("temperature"))

    # layers: the model layer's forced top_k 7 replaces the global forced top_k 1
    st, r = completion(P, {"top_k": 40})
    check("S1c model layer wins over global (top_k forced 7, not 1, not the request's 40)", settings(r).get("top_k") == 7,
          "top_k=%s" % settings(r).get("top_k"))

    # additive stop
    st, r = completion(P, {"stop": ["Y"]})
    stop = settings(r).get("stop", [])
    check("S1d additive: operator stop list in front of the request's", stop[:2] == ["ZZZSTOP", "Y"], "stop=%s" % stop)
    st, r = completion(P, {})
    check("S1d additive with no request list: operator list alone", settings(r).get("stop") == ["ZZZSTOP"], "stop=%s" % settings(r).get("stop"))

    # GET shows the layers
    st, tx = http(P, "GET", "/sampler-overrides")
    j = jb(tx)
    check("S1e GET shows global, model and effective", st == 200 and "n_predict" in j.get("global", {}) and "top_k" in j.get("model", {})
          and j.get("effective", {}).get("top_k", {}).get("override") == 7, tx[:200])
    check("S1e unknown key is not in the layer", "bogus_field" not in j.get("global", {}))

    # POST replaces the global layer at run time
    st, tx = http(P, "POST", "/sampler-overrides", {"global": {"n_predict": {"override": 8, "force": True}}})
    check("S1f POST replaces the global layer", st == 200 and jb(tx).get("global", {}).get("n_predict", {}).get("override") == 8, "status=%s %s" % (st, tx[:160]))
    st, r = completion(P, {"n_predict": 64})
    check("S1f the new value takes effect on the next request", r.get("tokens_predicted") == 8, "tokens=%s" % r.get("tokens_predicted"))
    # the model layer was not named in that POST: it stays
    st, r = completion(P, {"top_k": 40})
    check("S1f the model layer was left alone", settings(r).get("top_k") == 7, "top_k=%s" % settings(r).get("top_k"))

    # a layer that fails validation changes nothing
    st, tx = http(P, "POST", "/sampler-overrides", {"global": {"n_predict": {"override": 3, "force": True}, "temperature": {"override": None}}})
    check("S1g malformed layer is refused with 400", st == 400, "status=%s %s" % (st, tx[:160]))
    st, r = completion(P, {"n_predict": 64})
    check("S1g a refused layer changed nothing (still 8)", r.get("tokens_predicted") == 8, "tokens=%s" % r.get("tokens_predicted"))
    st, tx = http(P, "POST", "/sampler-overrides", {"global": {"stream": {"override": True, "force": True}}})
    check("S1g a field the handler reads before the schema is refused", st == 400 and "stream" in tx, "status=%s" % st)
    st, tx = http(P, "POST", "/sampler-overrides", {"nothing": 1})
    check("S1g body without a layer is refused", st == 400)

    # null empties the layers
    st, tx = http(P, "POST", "/sampler-overrides", {"global": None, "model": None})
    check("S1h null empties both layers", st == 200 and jb(tx).get("effective") == {}, tx[:160])
    st, r = completion(P, {"n_predict": 64})
    check("S1h with the layers gone the request is untouched again", r.get("tokens_predicted") == 64, "tokens=%s" % r.get("tokens_predicted"))
finally:
    s1.stop()

# ------------------------------------------------------------------------------------------ S2: admin tier
P2 = BASE + 2
s2 = Srv(P2, ["--api-key", "regular-key", "--api-key-admin", "admin-key", "--sampler-overrides", OV], "s2")
try:
    s2.wait_ready()
    st, tx = http(P2, "GET", "/sampler-overrides", key="regular-key")
    check("S2 a regular key can read the layers", st == 200 and "n_predict" in jb(tx).get("global", {}), "status=%s" % st)
    st, tx = http(P2, "POST", "/sampler-overrides", {"global": None}, key="regular-key")
    check("S2 a regular key cannot replace them (403)", st == 403, "status=%s" % st)
    st, r = completion(P2, {"n_predict": 64}, key="regular-key")
    check("S2 and nothing changed (still forced 16)", r.get("tokens_predicted") == 16, "tokens=%s" % r.get("tokens_predicted"))
    st, tx = http(P2, "POST", "/sampler-overrides", {"global": None}, key="admin-key")
    check("S2 the admin key can", st == 200, "status=%s" % st)
finally:
    s2.stop()

# ------------------------------------------------------------------------------------------ S3: management switch + bad files
P3 = BASE + 3
s3 = Srv(P3, ["--sampler-overrides", OV, "--disable-management-routes"], "s3")
try:
    s3.wait_ready()
    st, tx = http(P3, "POST", "/sampler-overrides", {"global": None})
    check("S3 --disable-management-routes covers the POST (403)", st == 403, "status=%s" % st)
    st, tx = http(P3, "GET", "/sampler-overrides")
    check("S3 the GET stays readable", st == 200)
finally:
    s3.stop()

bad = os.path.join(LOGDIR, "overrides_bad_%d.json" % BASE)
with open(bad, "w") as f:
    f.write('{"temperature": {"override": null}}')
s4 = Srv(BASE + 4, ["--sampler-overrides", bad], "s4")
try:
    s4.p.wait(120)
    check("S4 a malformed file stops start-up (non-zero exit)", s4.p.returncode not in (None, 0), "rc=%s" % s4.p.returncode)
    check("S4 and the log names the file and the cause", "overrides" in s4.logtext() and "null" in s4.logtext())
except subprocess.TimeoutExpired:
    check("S4 a malformed file stops start-up", False, "server kept running")
finally:
    s4.stop()

print("\n%d failure(s)" % len(fails) if fails else "\nALL PASS", flush=True)
sys.exit(1 if fails else 0)
