"""E2E served checks for request field token_healing, on a real llama-server and a real small model.

Usage: pythonw e2e_token_healing_served.py <llama-server.exe> <small.gguf> [port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Row: S69-tabbyapi-PICK-TOKEN-HEALING-2026-09-28-01.

What is checked (all deterministic, none depends on the model being "good"):
  * the request evaluates ONE token fewer (the last prompt token is removed and re-sampled)
  * the first generated token starts with the text of the removed token, at any temperature and seed
  * the returned text has that prefix stripped: content == detokenize(generated tokens)[len(prefix):]
  * the same request without the flag is unchanged (control): nothing stripped, no constraint
  * refusals: grammar / json_schema, and a one-token prompt is left alone
Fail-before: on a build without the field the flag is ignored, so the token count and the stripping checks fail.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXE, MODEL = sys.argv[1], sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 18360
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def post(path, body, timeout=120):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "replace") or "{}")
        except ValueError:
            return e.code, {}


log = os.path.join(LOGDIR, "healing_%d.log" % PORT)
f = open(log, "wb")
p = subprocess.Popen([EXE, "-m", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "-c", "1024", "-np", "1", "-t", "4"],
                     stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000)
try:
    t0 = time.time()
    while time.time() - t0 < 180:
        if p.poll() is not None:
            raise RuntimeError("server exited rc=%s, log %s" % (p.returncode, log))
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/props" % PORT, timeout=3) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.4)
    else:
        raise RuntimeError("not ready, log " + log)

    prompts = [
        'The documentation lives at http:',
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fib",
        "The capital of France is Par",
        "import numpy as n",
        "San Fran",
    ]

    def toks(text):
        st, r = post("/tokenize", {"content": text, "add_special": True, "parse_special": True})
        return r.get("tokens", [])

    def detok(ids):
        st, r = post("/detokenize", {"tokens": ids})
        return r.get("content", "")

    def gen(prompt, healing, seed, temp=0.8, **extra):
        body = {"prompt": prompt, "n_predict": 6, "temperature": temp, "seed": seed, "cache_prompt": False, "return_tokens": True}
        if healing:
            body["token_healing"] = True
        body.update(extra)
        return post("/completion", body)

    for prompt in prompts:
        ids = toks(prompt)
        last = ids[-1]
        pre = detok([last])
        label = repr(prompt[-18:])
        st0, base = gen(prompt, False, 1)
        n_base = base.get("timings", {}).get("prompt_n")
        for seed in (1, 2, 3, 4):
            for temp in (0.0, 1.0):
                st, r = gen(prompt, True, seed, temp)
                gen_ids = r.get("tokens", [])
                if r.get("stop_type") == "eos" and gen_ids:
                    # the end-of-generation token is in `tokens` but its text is never part of `content`; at t=1.0 a 6-token request can end on it
                    # (measured: content 'p.\n```' vs detok ' np.\n```<|im_end|>'), so it must not be detokenized into the expected text
                    gen_ids = gen_ids[:-1]
                first_piece = detok(gen_ids[:1]) if gen_ids else ""
                n_heal = r.get("timings", {}).get("prompt_n")
                tag = "%s seed=%d t=%.1f" % (label, seed, temp)
                check("%s: answered" % tag, st == 200 and gen_ids, "status=%s" % st)
                check("%s: one prompt token fewer evaluated" % tag, n_base is not None and n_heal == n_base - 1, "base=%s healed=%s" % (n_base, n_heal))
                check("%s: first token starts with the removed token's text %r" % (tag, pre), first_piece.startswith(pre), "first=%r" % first_piece)
                check("%s: output has the prefix stripped" % tag, r.get("content") == detok(gen_ids)[len(pre):],
                      "content=%r detok=%r" % (r.get("content"), detok(gen_ids)))
        # control: same request without the flag is unchanged
        st, c = gen(prompt, False, 1, 0.0)
        c_ids = c.get("tokens", [])
        check("%s: control keeps the whole prompt" % label, c.get("timings", {}).get("prompt_n") == n_base)
        check("%s: control output is not stripped" % label, c.get("content") == detok(c_ids), "content=%r" % c.get("content"))

    # one-token prompt (BOS alone is not text; use a prompt that tokenizes to a single token without special)
    st, one = post("/completion", {"prompt": "a", "n_predict": 3, "token_healing": True, "cache_prompt": False, "temperature": 0.0, "add_special": False})
    n_one = one.get("timings", {}).get("prompt_n")
    st, one_b = post("/completion", {"prompt": "a", "n_predict": 3, "cache_prompt": False, "temperature": 0.0, "add_special": False})
    n_plain = one_b.get("timings", {}).get("prompt_n")
    if len(toks("a")) == 1:      # a model that adds no BOS: a single token must be left alone
        check("one-token prompt is left alone", st == 200 and n_one == n_plain, "healed=%s plain=%s" % (n_one, n_plain))
    else:                        # BOS + token: two tokens, healing applies as usual
        check("two-token prompt (BOS + one) is healed", st == 200 and n_one == n_plain - 1, "healed=%s plain=%s" % (n_one, n_plain))

    # refusals
    st, r = post("/completion", {"prompt": "x = ", "n_predict": 4, "token_healing": True, "grammar": 'root ::= "a" | "b"'})
    check("grammar + token_healing is refused with 400", st == 400, "status=%s %s" % (st, json.dumps(r)[:160]))
    st, r = post("/completion", {"prompt": "x = ", "n_predict": 4, "token_healing": True, "json_schema": {"type": "object"}})
    check("json_schema + token_healing is refused with 400", st == 400, "status=%s" % st)
    st, r = post("/completion", {"prompt": "x = ", "n_predict": 4, "token_healing": "yes"})
    check("a non-boolean token_healing is refused", st == 400, "status=%s" % st)

    # informational: what healing does to a boundary-sensitive prompt (not asserted)
    for prompt in prompts[:2]:
        st, a = gen(prompt, False, 7, 0.0)
        st, b = gen(prompt, True, 7, 0.0)
        print("INFO %r\n  plain:  %r\n  healed: %r" % (prompt[-30:], a.get("content"), b.get("content")), flush=True)
finally:
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(15)
        except subprocess.TimeoutExpired:
            p.kill()
    f.close()

print("FAILED: %d checks: %s" % (len(fails), fails[:6]) if fails else "ALL PASS", flush=True)
sys.exit(1 if fails else 0)
