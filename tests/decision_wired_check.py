#!/usr/bin/env python3
"""End-to-end proof that /v1/decision is WIRED: a real llama-server on a random-weight GGUF, the endpoint switched on through the
real flag (--decision-seqs), the response and the always-on log line that show it FIRED, and the flag-off control.

    python3 decision_wired_check.py <llama-server> <tiny.gguf>

tiny.gguf: tests/make_tiny_llama_gguf.py <tree> out.gguf 2048 bytes. Exit 0 = every check passed.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

BIN, MODEL = sys.argv[1], sys.argv[2]
PORT = 18950
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

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


SCHEMA = {
    "urgent": {"type": "boolean", "description": "Is it urgent?"},
    "kind": {"type": "enum", "choices": ["billing", "technical", "other"], "description": "Request type."},
    "prio": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Priority.", "aggregate": "median"},
}
BODY = {"instructions": "Answer from the request.", "schema": SCHEMA, "contexts": ["charged twice", "app crashes on start"]}

print("flag OFF (default): endpoint refuses, nothing reserved")
s = Server()
try:
    code, body = s.post("/v1/decision", BODY)
    check(code == 400 and "decisions are disabled" in json.dumps(body), f"flag off -> 400 'decisions are disabled' (got {code}: {json.dumps(body)[:120]})")
    check("decision: " not in s.text().replace("decisions are disabled: ", ""), "flag off: no decision log line")
    code, off_c = s.post("/completion", {"prompt": "abc", "n_predict": 8, "temperature": 0, "seed": 1})
    off_text = off_c.get("content")
    check(code == 200 and off_text is not None, "flag off: /completion works")
finally:
    s.stop()

print("flag too small is refused at parse time")
bad = subprocess.run([BIN, "-m", MODEL, "--decision-seqs", "2"], capture_output=True, text=True, timeout=60,
                     env=dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(os.path.abspath(BIN))))
check(bad.returncode != 0 and "at least 3" in (bad.stdout + bad.stderr), "--decision-seqs 2 -> error mentioning 'at least 3'")

print("flag ON: --decision-seqs 24")
s = Server("--decision-seqs", "24")
try:
    code, r1 = s.post("/v1/decision", BODY)
    check(code == 200 and r1.get("object") == "decision", f"200 and object=decision (got {code})")
    res = r1.get("results", [])
    check(len(res) == 2, "one result per context")
    for i, it in enumerate(res):
        d = it["decision"]
        check(isinstance(d["urgent"], bool) and d["kind"] in SCHEMA["kind"]["choices"] and 1 <= d["prio"] <= 5, f"context {i}: values come from the allowed sets {d}")
        for name, f in it["fields"].items():
            check(0.0 <= f["probability"] <= 1.0 + 1e-6, f"context {i} {name}: probability {f['probability']:.4f} in [0,1]")
        check("interval_p10_p90" in it["fields"]["prio"], f"context {i}: numeric field carries the p10-p90 interval")
    check(r1["usage"]["scored_rows"] > 0 and r1["timings"]["rounds"] >= 1, f"scored_rows {r1['usage']['scored_rows']}, rounds {r1['timings']['rounds']}")
    check(r1["usage"]["cached_tokens"] == 0, "first call: prefix cache miss")
    code, r2 = s.post("/v1/decision", BODY)
    check(code == 200 and r2["usage"]["cached_tokens"] > 0, f"second call reuses the cached prefix (cached_tokens {r2['usage'].get('cached_tokens')})")
    check([x["decision"] for x in r2["results"]] == [x["decision"] for x in res], "cached call gives the same decisions")
    code, r3 = s.post("/v1/decision", dict(BODY, cache_prompt=False))
    check(code == 200 and r3["usage"]["cached_tokens"] == 0, "cache_prompt=false rebuilds the prefix")
    code, r4 = s.post("/decision", dict(BODY, mode="greedy"))
    check(code == 200 and all(not f["tree"] for it in r4["results"] for f in it["fields"].values()), "/decision alias; mode=greedy walks the trie")
    for bad_body, what in (({"schema": SCHEMA}, "missing contexts"), ({"contexts": ["x"]}, "missing schema"),
                           ({"contexts": ["x"], "schema": {"a": {"type": "string", "description": "d"}}}, "free-text field")):
        code, _ = s.post("/v1/decision", bad_body)
        check(code == 400, f"{what} -> 400 (got {code})")
    # an ordinary completion still works next to the decision sequences (slots are untouched)
    code, c = s.post("/completion", {"prompt": "abc", "n_predict": 8, "temperature": 0, "seed": 1})
    check(code == 200 and c.get("content") == off_text, "normal /completion identical to the flag-off server (%r)" % c.get("content"))
    log = s.text()
    check("decision: 2 context(s), 3 field(s)" in log, "always-on log line 'decision: 2 context(s), 3 field(s) ...' fired")
    check("enabling the unified KV cache" in log or "kv_unified = 'true'" in log, "unified KV cache switched on by the flag")
finally:
    s.stop()

print("FAILED" if fails else "all checks passed", len(fails))
sys.exit(1 if fails else 0)
