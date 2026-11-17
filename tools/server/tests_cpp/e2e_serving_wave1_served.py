"""E2E served checks for the 2026-10-05 serving wave 1, on a real llama-server and real (small) models.

Usage: pythonw e2e_serving_wave1_served.py <llama-server.exe> <small.gguf> <tiny-but-not-instant.gguf> [base-port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Rows:
  S69-NEXA-PICK-OPENAI-CONTEXT-LENGTH-EXCEEDED-CODE-2026-09-29-01          S1a
  S69-OLLAMA-PICK-EXPOSE-MEMORY-BREAKDOWN-JSON-2026-09-29-01               S1b
  S69-LOCALAI-PICK-RETRY-AFTER-HEADERS-2026-09-29-01                       S1c (busy), S3 (loading)
  S69-Q9-PICK-MANAGEMENT-PLANE-DEFAULT-OFF-CONTROL-VECTOR-PATH-2026-09-30-01   S2
  S69-LOCALAI-PICK-MODEL-LOAD-FAILURE-COOLDOWN-2026-09-29-01               S4 (router)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

EXE, MODEL, MODEL_B = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = int(sys.argv[4]) if len(sys.argv) > 4 else 18130
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
            return r.status, dict(r.headers), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", "replace")


def jbody(text):
    try:
        return json.loads(text)
    except ValueError:
        return {}


class Srv:
    def __init__(self, port, args, tag):
        self.port, self.tag = port, tag
        self.log = os.path.join(LOGDIR, "wave1_%s_%d.log" % (tag, port))
        self.f = open(self.log, "wb")
        self.p = subprocess.Popen([EXE, "--host", "127.0.0.1", "--port", str(port)] + args, stdout=self.f,
                                  stderr=subprocess.STDOUT, creationflags=0x08000000)

    def wait_ready(self, timeout=240):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                raise RuntimeError("server exited rc=%s, log %s" % (self.p.returncode, self.log))
            try:
                st, _, _ = http(self.port, "GET", "/props", timeout=3)
                if st == 200:
                    return time.time() - t0
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


def hdr(headers, name):
    return {k.lower(): v for k, v in headers.items()}.get(name.lower())


# ------------------------------------------------------------------------------------------------ S1: plain server
s1 = Srv(BASE, ["-m", MODEL, "-c", "512", "-np", "1", "-t", "4", "--metrics"], "s1")
try:
    s1.wait_ready()

    # S1a: context_length_exceeded
    long_msgs = [{"role": "user", "content": "What is the best book " * 300}]
    st, hd, tx = http(BASE, "POST", "/v1/chat/completions", {"messages": long_msgs})
    err = jbody(tx).get("error", {})
    check("S1a HTTP status stays 400", st == 400, "status=%s" % st)
    check("S1a error.code is the OpenAI string", err.get("code") == "context_length_exceeded", "code=%r" % err.get("code"))
    check("S1a error.status carries 400", err.get("status") == 400, "status=%r" % err.get("status"))
    check("S1a error.type unchanged", err.get("type") == "exceed_context_size_error")
    check("S1a n_prompt_tokens / n_ctx still present", err.get("n_prompt_tokens", 0) > 0 and err.get("n_ctx", 0) > 0,
          "n_prompt_tokens=%s n_ctx=%s" % (err.get("n_prompt_tokens"), err.get("n_ctx")))
    # control: another error keeps the numeric code
    st, hd, tx = http(BASE, "POST", "/v1/chat/completions", {"messages": "not a list"})
    e2 = jbody(tx).get("error", {})
    check("S1a control: other 400 errors keep the numeric code", st == 400 and e2.get("code") == 400, "status=%s code=%r" % (st, e2.get("code")))

    # S1b: /props memory
    st, hd, tx = http(BASE, "GET", "/props")
    mem = jbody(tx).get("memory")
    check("S1b /props has a memory object", isinstance(mem, dict) and {"devices", "host", "other"} <= set(mem), json.dumps(mem)[:200] if mem else "absent")
    if isinstance(mem, dict):
        host = mem.get("host", {})
        size = os.path.getsize(MODEL)
        # the model weights live in host memory on a CPU build; mmap/repack can shift a little
        check("S1b host model_bytes within 0.5x..1.6x of the GGUF size", 0.5 * size <= host.get("model_bytes", 0) <= 1.6 * size,
              "model_bytes=%s gguf=%s" % (host.get("model_bytes"), size))
        check("S1b host context_bytes > 0 (KV)", host.get("context_bytes", 0) > 0, "context_bytes=%s" % host.get("context_bytes"))
        check("S1b self == model + context + compute", host.get("self_bytes") ==
              host.get("model_bytes", 0) + host.get("context_bytes", 0) + host.get("compute_bytes", 0))
        check("S1b *_mib is bytes // 1048576", host.get("model_mib") == host.get("model_bytes", 0) // 1048576 and
              host.get("context_mib") == host.get("context_bytes", 0) // 1048576)
        for d in mem.get("devices", []):
            check("S1b device row complete (%s)" % d.get("name"), all(k in d for k in ("total_bytes", "free_bytes_at_load", "unaccounted_bytes")))

    # S1c: busy 503 carries a computed Retry-After
    res = {}

    def hold():
        res["A"] = http(BASE, "POST", "/completion", {"prompt": "Count slowly:", "n_predict": 200, "ignore_eos": True, "temperature": 0,
                                                       "cache_prompt": False}, timeout=300)

    # feed the estimator first so the header is a computed value rather than the 1 s floor
    http(BASE, "POST", "/completion", {"prompt": "Hi", "n_predict": 60, "ignore_eos": True, "temperature": 0, "cache_prompt": False})
    ta = threading.Thread(target=hold)
    ta.start()
    time.sleep(0.5)
    st, hd, tx = http(BASE, "POST", "/completion", {"prompt": "Hello", "n_predict": 4, "max_queue_wait_ms": 150})
    ra = hdr(hd, "retry-after")
    check("S1c second request refused 503 when the only slot is busy", st == 503, "status=%s body=%s" % (st, tx[:120]))
    check("S1c Retry-After header present, positive integer", ra is not None and ra.isdigit() and int(ra) >= 1, "Retry-After=%r" % ra)
    ta.join()
    st, hd, tx = http(BASE, "POST", "/completion", {"prompt": "Hello", "n_predict": 4})
    check("S1c control: a 200 answer carries no Retry-After", st == 200 and hdr(hd, "retry-after") is None)

    # S2 control: without the switch the management routes are reachable
    st, _, tx = http(BASE, "POST", "/lora-adapters", [])
    check("S2 control: POST /lora-adapters reachable by default (not 403)", st != 403, "status=%s" % st)
finally:
    s1.stop()

# ------------------------------------------------------------------------------------------------ S2: management routes
s2 = Srv(BASE + 1, ["-m", MODEL, "-c", "512", "-np", "1", "-t", "4", "--disable-management-routes", "--metrics"], "s2")
try:
    s2.wait_ready()
    for method, path, body in [("POST", "/lora-adapters", []), ("POST", "/props", {}), ("POST", "/slots/0?action=erase", {}),
                               ("POST", "/control-vectors/load", {"path": "/etc/passwd"}), ("POST", "/control-vectors/apply", []),
                               ("POST", "/save", {}), ("POST", "/delete", {})]:
        st, hd, tx = http(BASE + 1, method, path, body)
        e = jbody(tx).get("error", {})
        check("S2 %s %s -> 403 feature_disabled" % (method, path), st == 403 and e.get("type") == "feature_disabled", "status=%s type=%r" % (st, e.get("type")))
    for method, path in [("GET", "/lora-adapters"), ("GET", "/slots"), ("GET", "/props"), ("GET", "/metrics"), ("GET", "/health"),
                         ("GET", "/control-vectors")]:
        st, hd, tx = http(BASE + 1, method, path)
        check("S2 read-only %s %s still works" % (method, path), st == 200, "status=%s" % st)
    st, _, tx = http(BASE + 1, "POST", "/completion", {"prompt": "Hi", "n_predict": 2})
    check("S2 inference unaffected", st == 200, "status=%s" % st)
finally:
    s2.stop()

# ------------------------------------------------------------------------------------------------ S3: loading-model Retry-After
s3 = Srv(BASE + 2, ["-m", MODEL_B, "-c", "2048", "-np", "1", "-t", "2"], "s3")
try:
    seen = None
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            st, hd, tx = http(BASE + 2, "GET", "/props", timeout=2)
        except Exception:
            time.sleep(0.01)
            continue
        if st == 503:
            seen = hd
            break
        if st == 200:
            break
        time.sleep(0.01)
    if seen is None:
        check("S3 loading 503 observed", False, "model loaded before the first 503 was seen")
    else:
        ra = hdr(seen, "retry-after")
        check("S3 loading 503 carries Retry-After 1..10", ra is not None and ra.isdigit() and 1 <= int(ra) <= 10, "Retry-After=%r" % ra)
finally:
    s3.stop()

# ------------------------------------------------------------------------------------------------ S4: router load-failure cooldown
mdir = "tmp/_e1005_models_%d" % int(time.time())
os.makedirs(mdir, exist_ok=True)
with open(MODEL, "rb") as src, open(os.path.join(mdir, "bad.gguf"), "wb") as dst:
    dst.write(src.read(4 * 1024 * 1024))            # a truncated GGUF: the header parses, the tensors do not
try:
    os.link(MODEL, os.path.join(mdir, "good.gguf"))
except OSError:
    shutil.copyfile(MODEL, os.path.join(mdir, "good.gguf"))
s4 = Srv(BASE + 3, ["--models-dir", mdir, "--models-max", "2", "-c", "512", "-np", "1", "-t", "2",
                    "--model-load-cooldown", "4s", "--model-load-cooldown-max", "16s"], "s4")
try:
    s4.wait_ready(60)

    def spawns(name):
        return len(re.findall(r"spawning server instance with name=%s " % re.escape(name), s4.logtext()))

    def chat(model):
        return http(BASE + 3, "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}, timeout=120)

    st1, hd1, tx1 = chat("bad")
    check("S4 first request for a corrupt model fails", st1 >= 400, "status=%s body=%s" % (st1, tx1[:140]))
    n1 = spawns("bad")
    codes = []
    for _ in range(4):
        st, hd, tx = chat("bad")
        codes.append((st, hdr(hd, "retry-after")))
    n2 = spawns("bad")
    check("S4 one spawn for the first request", n1 == 1, "spawns=%d" % n1)
    check("S4 the next 4 requests spawn nothing (cooldown)", n2 == n1, "spawns %d -> %d" % (n1, n2))
    check("S4 and answer 503 with Retry-After in 1..4", all(c == 503 and ra and ra.isdigit() and 1 <= int(ra) <= 4 for c, ra in codes), str(codes))
    # control: a healthy model in the same router is untouched
    st, hd, tx = chat("good")
    check("S4 control: a healthy model still serves", st == 200, "status=%s %s" % (st, tx[:100]))
    time.sleep(4.5)                                  # past the first 4 s cooldown
    n3 = spawns("bad")
    chat("bad")
    n4 = spawns("bad")
    check("S4 after the cooldown the model is tried once more", n4 == n3 + 1, "spawns %d -> %d" % (n3, n4))
    st, hd, tx = chat("bad")
    ra2 = hdr(hd, "retry-after")
    check("S4 second consecutive failure doubles the cooldown (Retry-After 5..8)", st == 503 and ra2 and ra2.isdigit() and 5 <= int(ra2) <= 8,
          "status=%s Retry-After=%r" % (st, ra2))
    # explicit operator action bypasses
    n5 = spawns("bad")
    http(BASE + 3, "POST", "/models/load", {"model": "bad"})
    time.sleep(2)
    n6 = spawns("bad")
    check("S4 POST /models/load bypasses the cooldown", n6 == n5 + 1, "spawns %d -> %d" % (n5, n6))
finally:
    s4.stop()

# ------------------------------------------------------------------------------------------------ S5: admin / API key tiers + hot reload
# S69-tabbyapi-PICK-ADMIN-APIKEY-TIER-2026-09-28-01
keyfile = "tmp/_e1005_keys_%d.txt" % int(time.time())
with open(keyfile, "w") as kf:
    kf.write("# regular keys\nfile-key-one\n")
s5 = Srv(BASE + 4, ["-m", MODEL, "-c", "512", "-np", "1", "-t", "2", "--api-key", "user1", "--api-key-admin", "root",
                    "--api-key-file", keyfile], "s5")


def auth_req(key_headers, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (BASE + 4, path), data=data, method=method,
                                 headers=dict({"Content-Type": "application/json"}, **key_headers))
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def bearer(k):
    return {"Authorization": "Bearer " + k}


try:
    t0 = time.time()
    while time.time() - t0 < 120:                       # /health is public even with keys on
        try:
            if auth_req({}, "GET", "/health")[0] == 200:
                break
        except Exception:
            pass
        time.sleep(0.3)
    check("S5 /health is public", auth_req({}, "GET", "/health")[0] == 200)
    check("S5 no key -> 401", auth_req({}, "POST", "/completion", {"prompt": "Hi", "n_predict": 1})[0] == 401)
    check("S5 wrong key -> 401", auth_req(bearer("nope"), "POST", "/completion", {"prompt": "Hi", "n_predict": 1})[0] == 401)
    check("S5 regular key serves inference", auth_req(bearer("user1"), "POST", "/completion", {"prompt": "Hi", "n_predict": 1})[0] == 200)
    check("S5 regular key (X-Api-Key style) serves inference", auth_req({"X-Api-Key": "user1"}, "POST", "/completion", {"prompt": "Hi", "n_predict": 1})[0] == 200)
    st, tx = auth_req(bearer("user1"), "POST", "/lora-adapters", [])
    check("S5 regular key on an admin route -> 403 permission_error", st == 403 and jbody(tx).get("error", {}).get("type") == "permission_error", "status=%s" % st)
    st, tx = auth_req(bearer("root"), "POST", "/lora-adapters", [])
    check("S5 admin key on an admin route is admitted (not 401/403)", st not in (401, 403), "status=%s" % st)
    st, tx = auth_req(dict(bearer("user1"), **{"X-Admin-Key": "root"}), "POST", "/lora-adapters", [])
    check("S5 X-Admin-Key beside a regular Authorization is admitted", st not in (401, 403), "status=%s" % st)
    check("S5 admin key also serves inference", auth_req(bearer("root"), "POST", "/completion", {"prompt": "Hi", "n_predict": 1})[0] == 200)
    check("S5 key from the key file works", auth_req(bearer("file-key-one"), "POST", "/completion", {"prompt": "Hi", "n_predict": 1})[0] == 200)

    # hot reload: swap the file's key, no restart
    with open(keyfile, "w") as kf:
        kf.write("# rotated\nfile-key-two-rotated\n")
    ok_new = ok_old_gone = False
    for _ in range(30):
        time.sleep(0.5)
        if auth_req(bearer("file-key-two-rotated"), "GET", "/props")[0] == 200:
            ok_new = True
            ok_old_gone = auth_req(bearer("file-key-one"), "GET", "/props")[0] == 401
            break
    check("S5 hot reload: the new file key starts working without a restart", ok_new)
    check("S5 hot reload: the pulled file key stops working", ok_old_gone)
    check("S5 hot reload: the command-line key is untouched", auth_req(bearer("user1"), "GET", "/props")[0] == 200)
    # an emptied file must not turn authentication off: but the command-line keys stay, so the set is not empty;
    # empty the file and check the rotated key is gone while user1 still works and a stranger still gets 401
    with open(keyfile, "w") as kf:
        kf.write("")
    time.sleep(5)
    check("S5 emptied file: command-line keys keep working", auth_req(bearer("user1"), "GET", "/props")[0] == 200)
    check("S5 emptied file: authentication is still on (stranger gets 401)", auth_req(bearer("stranger"), "GET", "/props")[0] == 401)
finally:
    s5.stop()

print("RESULT:", "FAIL" if fails else "PASS", fails)
sys.exit(1 if fails else 0)
