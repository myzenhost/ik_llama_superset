# REWRITE-SAFE: this test only writes NEW files (a GGUF copy with one extra key) under temp/logs; it reads no data store.
# SAFE-DELETE-OK: the only file removed is the 144 MB GGUF copy this script itself created a few lines earlier under temp/logs.
"""E2E served checks for --draft-max-by-slots and --sampling-family-defaults.

Usage: pythonw e2e_draft_family_served.py <llama-server.exe> <smollm2-135m.gguf> [base-port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Rows: S69-aphrodite-PICK-DYNAMIC-SPEC-DRAFT-LEN-BY-BATCH-SIZE-2026-09-29-01   (stage 1: the table)   D0..D2
      S69-LOCALAI-PICK-FAMILY-SAMPLING-DEFAULTS-FALLBACK-2026-09-29-01                                F0..F4
The draft model is the target itself (self-speculation): greedy output is identical with and without drafting, which is the identity
check; the drafted-token counter is read from /metrics.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

EXE, MODEL = sys.argv[1], sys.argv[2]
BASE = int(sys.argv[3]) if len(sys.argv) > 3 else 18190
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def http(port, method, path, body=None, timeout=180):
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


class Srv:
    def __init__(self, port, args, tag):
        self.port = port
        self.log = os.path.join(LOGDIR, "draftfam_%s_%d.log" % (tag, port))
        self.f = open(self.log, "wb")
        self.p = subprocess.Popen([EXE, "--host", "127.0.0.1", "--port", str(port), "-t", "4"] + args, stdout=self.f,
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


def metric(port, name):
    st, tx = http(port, "GET", "/metrics")
    for line in tx.splitlines():
        if line.startswith("llamacpp:" + name + " ") or line.startswith("llamacpp:" + name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return None


def slots_processing(port):
    st, tx = http(port, "GET", "/slots")
    return sum(1 for s in jb(tx) if s.get("is_processing")) if st == 200 else -1


PROMPT = "Count: one two three four five six seven eight nine ten"


def gen(port, n, results, idx, temp=0.0):
    st, tx = http(port, "POST", "/completion", {"prompt": PROMPT, "n_predict": n, "ignore_eos": True, "temperature": temp, "cache_prompt": False})
    results[idx] = jb(tx)


def overlap_run(port, n_each=300):
    """Two long requests started together. The server adds a request's drafted tokens to /metrics only when the request ends, so the
    per-request timings.draft_n of the final responses is the observable: with both slots generating for almost the whole run, a
    schedule that says '2 slots: no drafting' must leave draft_n near zero."""
    res = [None, None]
    ts = [threading.Thread(target=gen, args=(port, n_each, res, i)) for i in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return [((r or {}).get("timings") or {}).get("draft_n", 0) for r in res], res


COMMON = ["-m", MODEL, "-md", MODEL, "--spec-type", "draft", "-c", "4096", "-np", "4", "--metrics", "--spec-draft-n-max", "4", "--no-warmup"]

# ------------------------------------------------------------------------------------------ D0: control, no schedule
P = BASE
s = Srv(P, COMMON, "d0")
try:
    s.wait_ready()
    drafted, res = overlap_run(P)
    D0_DRAFTED = drafted
    check("D0 control: two concurrent slots, no schedule: both draft most of their tokens", all(d > 100 for d in drafted),
          "draft_n=%s tokens=%s" % (drafted, [r and r.get("tokens_predicted") for r in res]))
    solo = [None]
    gen(P, 64, solo, 0)
    ref_text = solo[0].get("content")
    check("D0 solo request produced text", bool(ref_text) and solo[0].get("tokens_predicted") == 64)
finally:
    s.stop()

# ------------------------------------------------------------------------------------------ D1: schedule '1:4,2:0'
P = BASE + 1
s = Srv(P, COMMON + ["--draft-max-by-slots", "1:4,2:0"], "d1")
try:
    s.wait_ready()
    check("D1 start-up logs the schedule", "draft length by generating slots" in s.logtext())
    solo = [None]
    gen(P, 64, solo, 0)
    solo_draft_n = ((solo[0] or {}).get("timings") or {}).get("draft_n", 0)
    check("D1 one slot still drafts (schedule says 4)", solo_draft_n > 20, "draft_n=%s of 64" % solo_draft_n)
    check("D1 greedy output with drafting == output of the unscheduled server", solo[0].get("content") == ref_text,
          "len %s vs %s" % (len(solo[0].get("content") or ""), len(ref_text or "")))
    drafted, res = overlap_run(P)
    check("D1 two concurrent slots: drafting stops while both generate (draft_n under a tenth of the control's)",
          all(d <= max(10, min(D0_DRAFTED) // 10) for d in drafted), "draft_n=%s control=%s" % (drafted, D0_DRAFTED))
    check("D1 both long requests completed", all(r and r.get("tokens_predicted") == 300 for r in res), "tokens=%s" % [r and r.get("tokens_predicted") for r in res])
finally:
    s.stop()

# ------------------------------------------------------------------------------------------ D2: refusals
for tag, spec in (("d2a", "3:2,1:5"), ("d2b", "2:2"), ("d2c", "1-4:5,3-6:2"), ("d2d", "1:x")):
    s = Srv(BASE + 2, COMMON + ["--draft-max-by-slots", spec], tag)
    try:
        s.p.wait(120)
        txt = s.logtext()
        check("D2 '%s' stops start-up (non-zero exit) and the log names the flag" % spec, s.p.returncode not in (None, 0) and "--draft-max-by-slots" in txt,
              "rc=%s" % s.p.returncode)
    except subprocess.TimeoutExpired:
        check("D2 '%s' stops start-up" % spec, False, "server kept running")
    finally:
        s.stop()


# ------------------------------------------------------------------------------------------ F: family defaults
def gen_settings(port):
    st, tx = http(port, "GET", "/props")
    j = jb(tx)
    d = j.get("default_generation_settings", {})
    return d.get("params", d)


FARGS = ["-m", MODEL, "-c", "1024", "-np", "1", "--no-warmup"]
s = Srv(BASE + 3, FARGS, "f0")
try:
    s.wait_ready()
    g = gen_settings(BASE + 3)
    check("F0 control: no flag, the built-in defaults stand (temperature 0.8, min_p 0.05, top_k 40)",
          abs(g.get("temperature", -1) - 0.8) < 1e-6 and abs(g.get("min_p", -1) - 0.05) < 1e-6 and g.get("top_k") == 40,
          json.dumps({k: g.get(k) for k in ("temperature", "min_p", "top_k", "top_p")}))
finally:
    s.stop()

s = Srv(BASE + 4, FARGS + ["--sampling-family-defaults"], "f1")
try:
    s.wait_ready()
    g = gen_settings(BASE + 4)
    check("F1 flag on: SmolLM2 matches the smollm family (temperature 0.7, top_p 0.95, top_k 0, min_p 0.01)",
          abs(g.get("temperature", -1) - 0.7) < 1e-6 and abs(g.get("top_p", -1) - 0.95) < 1e-6 and g.get("top_k") == 0 and abs(g.get("min_p", -1) - 0.01) < 1e-6,
          json.dumps({k: g.get(k) for k in ("temperature", "min_p", "top_k", "top_p")}))
    log = s.logtext()
    check("F1 every value is logged with its source", "family 'smollm'" in log and "temp" in log)
finally:
    s.stop()

s = Srv(BASE + 5, FARGS + ["--sampling-family-defaults", "--temp", "0.1"], "f2")
try:
    s.wait_ready()
    g = gen_settings(BASE + 5)
    check("F2 an explicit --temp beats the table, the rest still comes from it", abs(g.get("temperature", -1) - 0.1) < 1e-6 and abs(g.get("min_p", -1) - 0.01) < 1e-6,
          json.dumps({k: g.get(k) for k in ("temperature", "min_p")}))
    check("F2 the kept flag is logged", "kept from --temp" in s.logtext())
finally:
    s.stop()

# F3: a GGUF that carries its own general.sampling.temp: the model's value wins and the table is not used at all
sys.path.insert(0, "gguf-py")
have_kv_file = False
WITH_KV = os.path.join(LOGDIR, "smollm2_with_sampling_kv_%d.gguf" % BASE)
try:
    import gguf
    from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata
    reader = gguf.GGUFReader(MODEL, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(WITH_KV, arch=arch, endianess=reader.endianess)
    copy_with_new_metadata(reader, writer, {"general.sampling.temp": MetadataDetails(gguf.GGUFValueType.FLOAT32, 0.33)}, [])
    have_kv_file = True
except Exception as e:  # the python gguf package is a test dependency; say so instead of skipping silently
    check("F3 could build a GGUF with general.sampling.temp (needs numpy, tqdm and gguf-py)", False, repr(e))

if have_kv_file:
    s = Srv(BASE + 6, ["-m", WITH_KV, "-c", "1024", "-np", "1", "--no-warmup", "--sampling-family-defaults"], "f3")
    try:
        s.wait_ready()
        g = gen_settings(BASE + 6)
        check("F3 the model's own general.sampling.temp (0.33) wins over the family table", abs(g.get("temperature", -1) - 0.33) < 1e-4, "temperature=%s" % g.get("temperature"))
        check("F3 and the table is not applied to the other values (min_p stays at the built-in 0.05)", abs(g.get("min_p", -1) - 0.05) < 1e-6, "min_p=%s" % g.get("min_p"))
        check("F3 the log says why", "family defaults not used" in s.logtext())
    finally:
        s.stop()
    try:
        os.remove(WITH_KV)
    except OSError:
        pass

# F4: an alias that names another family is the first candidate and wins over the file name
s = Srv(BASE + 7, FARGS + ["--sampling-family-defaults", "--alias", "my-qwen3-coder-box"], "f4")
try:
    s.wait_ready()
    g = gen_settings(BASE + 7)
    check("F4 candidate order: the alias (qwen3-coder: 0.7 / 20 / 0.8) is tried before the file name", abs(g.get("temperature", -1) - 0.7) < 1e-6 and g.get("top_k") == 20,
          json.dumps({k: g.get(k) for k in ("temperature", "top_k", "top_p")}))
finally:
    s.stop()

print("\n%d failure(s)" % len(fails) if fails else "\nALL PASS", flush=True)
sys.exit(1 if fails else 0)
