# REWRITE-SAFE: this test writes only server logs under temp/logs; it reads no data store.
"""Served check of the OLMo-3 <function_calls> tool-call format with a REAL OLMo-3 model.

Usage: pythonw e2e_olmo3_tools_served.py <llama-server.exe> <Olmo-3-7B-Instruct-Q4_K_M.gguf> [port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Row: S69-tabbyapi-PICK-OLMO3-FUNCTION-CALLS-TOOL-FORMAT-2026-09-29-01 (the parser is unit-tested in tests/test-chat.cpp; what was missing is a served
request that goes template -> grammar -> model -> parser). The tiny random test model cannot do it: the required-tools grammar leaves free text before the
block, which a random model never closes.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXE, MODEL = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 18250
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def post(body, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:300]}


TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the web.",
                                      "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "deep": {"type": "boolean"}}, "required": ["q"]}}},
]
MSGS = [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool, with units in celsius."}]

log = os.path.join(LOGDIR, "olmo3_tools_%d.log" % PORT)
f = open(log, "wb")
p = subprocess.Popen([EXE, "--host", "127.0.0.1", "--port", str(PORT), "-m", MODEL, "--jinja", "-c", "4096", "-np", "1", "-t", "8", "--no-warmup"],
                     stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000)
try:
    t0 = time.time()
    while True:
        if p.poll() is not None:
            raise RuntimeError("server exited rc=%s, log %s" % (p.returncode, log))
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        if time.time() - t0 > 900:
            raise RuntimeError("not ready, log " + log)
        time.sleep(1)

    # required: the grammar must make the model produce a block; the parser must turn it into a tool call
    st, j = post({"messages": MSGS, "tools": TOOLS, "tool_choice": "required", "temperature": 0, "max_tokens": 120})
    msg = (j.get("choices") or [{}])[0].get("message", {})
    calls = msg.get("tool_calls") or []
    check("R1 tool_choice=required: HTTP 200 and at least one tool call", st == 200 and len(calls) >= 1, "status=%s msg=%s" % (st, json.dumps(msg)[:300]))
    if calls:
        c0 = calls[0]["function"]
        try:
            args = json.loads(c0["arguments"])
        except ValueError:
            args = None
        check("R1 the call names a declared tool and its arguments are a JSON object", c0["name"] in ("get_weather", "search") and isinstance(args, dict), json.dumps(c0)[:200])
        check("R1 finish_reason is tool_calls", j["choices"][0].get("finish_reason") == "tool_calls", "finish=%s" % j["choices"][0].get("finish_reason"))
        check("R1 the model chose get_weather for Paris (a 7B instruct model on this prompt)", c0["name"] == "get_weather" and "paris" in json.dumps(args).lower(), json.dumps(c0)[:200])
        check("R1 the <function_calls> block is NOT left in the content", "<function_calls>" not in (msg.get("content") or ""), "content=%r" % (msg.get("content") or "")[:120])

    # control: tool_choice none must not produce a tool call
    st, j = post({"messages": MSGS, "tools": TOOLS, "tool_choice": "none", "temperature": 0, "max_tokens": 60})
    msg = (j.get("choices") or [{}])[0].get("message", {})
    check("R2 control tool_choice=none: no tool call", st == 200 and not (msg.get("tool_calls") or []), "status=%s msg=%s" % (st, json.dumps(msg)[:200]))

    # auto: the model decides; a call, when made, must parse (the lazy grammar's trigger is the opening tag)
    st, j = post({"messages": MSGS, "tools": TOOLS, "tool_choice": "auto", "temperature": 0, "max_tokens": 120})
    msg = (j.get("choices") or [{}])[0].get("message", {})
    calls = msg.get("tool_calls") or []
    check("R3 tool_choice=auto: either a parsed call or plain content, never a raw block in content", st == 200 and (calls or "<function_calls>" not in (msg.get("content") or "")),
          "status=%s msg=%s" % (st, json.dumps(msg)[:300]))
finally:
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(20)
        except subprocess.TimeoutExpired:
            p.kill()
    f.close()

print("\n%d failure(s)" % len(fails) if fails else "\nALL PASS", flush=True)
sys.exit(1 if fails else 0)
