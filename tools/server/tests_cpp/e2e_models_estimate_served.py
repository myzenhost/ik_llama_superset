"""E2E served checks: GGUF header info in the model listing + pre-load memory estimate, on a real router and a real small model.

Usage: pythonw e2e_models_estimate_served.py <llama-server.exe> <small.gguf> [base-port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Rows:
  S69-gpt4all-PICK-GGUF-LIGHT-METADATA-PROBE-2026-09-28-01          GET /models?info=1
  S69-lmstudio-PICK-PRELOAD-MEMORY-ESTIMATE-ENDPOINT-2026-09-28-01  POST /models/estimate
Fail-before: run against a build without these changes; every check about the new surface fails there (404 / no "gguf" key).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXE, MODEL = sys.argv[1], sys.argv[2]
BASE = int(sys.argv[3]) if len(sys.argv) > 3 else 18340
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def http(port, method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def jbody(text):
    try:
        return json.loads(text)
    except ValueError:
        return {}


class Srv:
    def __init__(self, port, args, tag, env=None):
        self.port = port
        self.log = os.path.join(LOGDIR, "estimate_%s_%d.log" % (tag, port))
        self.f = open(self.log, "wb")
        e = dict(os.environ)
        e.update(env or {})
        self.p = subprocess.Popen([EXE, "--host", "127.0.0.1", "--port", str(port)] + args, stdout=self.f, stderr=subprocess.STDOUT,
                                  creationflags=0x08000000, env=e)

    def wait_ready(self, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                raise RuntimeError("server exited rc=%s, log %s" % (self.p.returncode, self.log))
            try:
                st, _ = http(self.port, "GET", "/props", timeout=3)
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


def gguf_truth(path):
    """Independent read of the header with gguf-py: architecture, <arch>.block_count, <arch>.context_length, chat template present."""
    sys.path.insert(0, "gguf-py")
    from gguf import GGUFReader
    r = GGUFReader(path)

    def val(name):
        f = r.fields.get(name)
        if f is None:
            return None
        v = f.parts[f.data[0]]
        return bytes(v).decode("utf-8", "replace") if f.types[0].name == "STRING" else int(v[0])

    arch = val("general.architecture")
    return {"arch": arch, "blocks": val(arch + ".block_count"), "ctx": val(arch + ".context_length"),
            "template": val("tokenizer.chat_template") is not None}


mdir = "tmp/_e1005_est_models_%d" % int(time.time())
os.makedirs(mdir, exist_ok=True)
try:
    os.link(MODEL, os.path.join(mdir, "tiny.gguf"))
except OSError:
    shutil.copyfile(MODEL, os.path.join(mdir, "tiny.gguf"))
file_bytes = os.path.getsize(MODEL)
truth = gguf_truth(MODEL)
print("truth:", truth, "file_bytes:", file_bytes, flush=True)

s = Srv(BASE, ["--models-dir", mdir, "--models-max", "1", "-np", "1", "-t", "2", "--no-models-autoload"], "main",
        env={"LLAMA_SERVER_HOST_FREE_MB": "100000"})
try:
    s.wait_ready()

    # ---- the listing without the parameter is unchanged
    st, tx = http(BASE, "GET", "/models")
    items = jbody(tx).get("data", [])
    check("listing works and has the model", st == 200 and any(i.get("id") == "tiny" for i in items), "status=%s" % st)
    check("no gguf key without ?info", all("gguf" not in i for i in items))

    # ---- GGUF header info
    st, tx = http(BASE, "GET", "/models?info=1")
    item = next((i for i in jbody(tx).get("data", []) if i.get("id") == "tiny"), {})
    g = item.get("gguf")
    check("?info=1 adds a gguf object", isinstance(g, dict), "gguf=%r" % (g,))
    if isinstance(g, dict):
        check("architecture matches an independent read", g.get("architecture") == truth["arch"], "%r vs %r" % (g.get("architecture"), truth["arch"]))
        check("block_count matches", g.get("block_count") == truth["blocks"], "%r vs %r" % (g.get("block_count"), truth["blocks"]))
        check("context_length matches", g.get("context_length") == truth["ctx"], "%r vs %r" % (g.get("context_length"), truth["ctx"]))
        check("has_chat_template matches", g.get("has_chat_template") == truth["template"], "%r vs %r" % (g.get("has_chat_template"), truth["template"]))
        check("file_bytes matches the file", g.get("file_bytes") == file_bytes)
        check("template text is NOT in ?info=1", "chat_template" not in g)
    st, tx = http(BASE, "GET", "/models?info=full")
    g2 = next((i for i in jbody(tx).get("data", []) if i.get("id") == "tiny"), {}).get("gguf", {})
    if truth["template"]:
        check("?info=full carries the template text", isinstance(g2.get("chat_template"), str) and len(g2["chat_template"]) == g2.get("chat_template_bytes"))
    check("the probe loaded nothing: model is still unloaded", item.get("status", {}).get("value") == "unloaded",
          "status=%r" % item.get("status", {}).get("value"))

    # ---- estimate
    st, tx = http(BASE, "POST", "/models/estimate", {"model": "tiny"})
    e = jbody(tx)
    check("estimate answers 200", st == 200, "status=%s body=%s" % (st, tx[:200]))
    host = e.get("host", {})
    check("fit tag is a known tag", e.get("fit") in ("cpu_only", "full_gpu_offload", "partial_gpu_offload", "will_not_fit"), "fit=%r" % e.get("fit"))
    check("context assumed 4096 and flagged when the preset leaves it unset", e.get("n_ctx") == 4096 and e.get("n_ctx_assumed") is True,
          "n_ctx=%r assumed=%r" % (e.get("n_ctx"), e.get("n_ctx_assumed")))
    total_model = host.get("model_bytes", 0) + sum(d.get("model_bytes", 0) for d in e.get("devices", []))
    check("weights bytes are within 40%% of the file size", 0.6 * file_bytes <= total_model <= 1.4 * file_bytes,
          "model_bytes=%d file=%d" % (total_model, file_bytes))
    check("confidence says the free memory is simulated (test hook)", "simulated" in str(e.get("confidence")), "confidence=%r" % e.get("confidence"))
    check("a roomy host fits", e.get("fit") != "will_not_fit" and host.get("fits") is True)

    st, tx = http(BASE, "POST", "/models/estimate", {"model": "tiny", "n_ctx": 256})
    e2 = jbody(tx)
    check("n_ctx override is applied and not flagged as assumed", e2.get("n_ctx") == 256 and e2.get("n_ctx_assumed") is False,
          "n_ctx=%r assumed=%r" % (e2.get("n_ctx"), e2.get("n_ctx_assumed")))
    ctx_small = e2.get("host", {}).get("context_bytes", 0) + sum(d.get("context_bytes", 0) for d in e2.get("devices", []))
    ctx_big = host.get("context_bytes", 0) + sum(d.get("context_bytes", 0) for d in e.get("devices", []))
    check("a smaller context needs less KV memory", 0 < ctx_small < ctx_big, "256 -> %d, 4096 -> %d" % (ctx_small, ctx_big))

    st, tx = http(BASE, "POST", "/models/estimate", {"model": "no-such-model"})
    check("unknown model is 404", st == 404, "status=%s" % st)
    st, tx = http(BASE, "POST", "/models/estimate", {"model": "tiny", "n_ctx": -5})
    check("negative n_ctx is 400", st == 400, "status=%s" % st)

    st, tx = http(BASE, "GET", "/models")
    stat = next((i for i in jbody(tx).get("data", []) if i.get("id") == "tiny"), {}).get("status", {}).get("value")
    check("no child was spawned by any of this", "spawning server instance" not in s.logtext() and stat == "unloaded", "status=%r" % stat)
finally:
    s.stop()

# ---- a starved host: the same request must come back will_not_fit
s2 = Srv(BASE + 1, ["--models-dir", mdir, "--models-max", "1", "-np", "1", "-t", "2", "--no-models-autoload"], "starved",
         env={"LLAMA_SERVER_HOST_FREE_MB": "1"})
try:
    s2.wait_ready()
    st, tx = http(BASE + 1, "POST", "/models/estimate", {"model": "tiny", "n_ctx": 256})
    e = jbody(tx)
    check("1 MiB free host: will_not_fit", st == 200 and e.get("fit") == "will_not_fit" and e.get("host", {}).get("fits") is False,
          "status=%s fit=%r" % (st, e.get("fit")))
finally:
    s2.stop()

print("FAILED: %s" % fails if fails else "ALL PASS", flush=True)
sys.exit(1 if fails else 0)
