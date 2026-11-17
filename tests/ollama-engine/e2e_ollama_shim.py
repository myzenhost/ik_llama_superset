#!/usr/bin/env python3
"""End-to-end check of the Ollama-native shim on a tiny RANDOM-weight model, through the real llama-server.
usage: e2e_ollama_shim.py <llama-server> <tiny.gguf> <workdir> [<tree-root> (enables the Go-template-from-store checks)]
Every assertion names the feature it proves FIRED (a changed observable), with a control where the feature is off."""
import hashlib, json, os, shutil, subprocess, sys, time, urllib.request, urllib.error

SERVER, MODEL, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(WORK, exist_ok=True)
fails = 0
def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok: fails += 1

def call(port, path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def j(port, path, body=None):
    st, tx = call(port, path, body)
    try: return st, json.loads(tx)
    except Exception: return st, tx

class Srv:
    def __init__(self, port, *extra, model=True, env=None):
        self.port = port
        self.log = open(os.path.join(WORK, f"srv{port}.log"), "w")
        e = dict(os.environ); e.update(env or {})
        self.p = subprocess.Popen([SERVER, *(["-m", MODEL] if model else []), "--port", str(port), "-c", "512", "-np", "1", "--jinja", "--alias", "tiny", "-t", "2", *extra],
                                  stdout=self.log, stderr=subprocess.STDOUT, env=e)
        for _ in range(240):
            time.sleep(0.5)
            try:
                if call(port, "/health")[0] == 200: return
            except Exception: pass
            if self.p.poll() is not None: break
        raise SystemExit(f"server {port} did not start; see {self.log.name}")
    def stop(self):
        self.p.terminate(); self.p.wait(30); self.log.close()
    def logtext(self): return open(self.log.name, errors="replace").read()

def gen(port, **kw):
    b = {"model": "tiny", "stream": False}; b.update(kw)
    return j(port, "/api/generate", b)

# ------------------------------------------------------------ plain server (all new flags OFF)
s = Srv(18101)
try:
    # random weights may sample EOS before num_predict (seed-dependent, changes with unrelated base merges), so scan seeds:
    # done_reason must equal "length" exactly when eval_count hit num_predict, and at least one seed must hit it
    rs = [gen(18101, prompt="hello", options={"num_predict": 3, "seed": sd}) for sd in range(1, 9)]
    consistent = all(st == 200 and (r["done_reason"] == "length" and r["eval_count"] == 3 or r["done_reason"] == "stop" and r["eval_count"] <= 3) for st, r in rs)
    nlen = sum(1 for st, r in rs if r["done_reason"] == "length")
    check(consistent and nlen >= 1, f"done_reason is length only when num_predict hit (EOS may end earlier or on the last token) ({nlen}/8 seeds hit length, consistent={consistent} {[(r["eval_count"], r["done_reason"]) for st, r in rs]})")
    st, r = rs[0]
    check(r.get("total_duration", 0) > 0, "non-stream total_duration present")
    # options: min_p=1.0 keeps only the top token -> seed-independent output; control without min_p differs by seed
    outs = [gen(18101, prompt="hello", options={"num_predict": 12, "seed": sd, "min_p": 1.0})[1]["response"] for sd in (1, 2, 3)]
    check(len(set(outs)) == 1, f"options.min_p mapped: 3 seeds give identical output ({len(set(outs))} distinct)")
    ctrl = [gen(18101, prompt="hello", options={"num_predict": 12, "seed": sd})[1]["response"] for sd in (1, 2, 3, 4)]
    check(len(set(ctrl)) > 1, f"control (no min_p): seeds differ ({len(set(ctrl))} distinct)")
    # presence_penalty / num_keep / repeat_last_n accepted
    st, r = gen(18101, prompt="hello", options={"num_predict": 4, "presence_penalty": 1.5, "frequency_penalty": 0.5, "repeat_last_n": 8, "num_keep": 2, "typical_p": 0.9})
    check(st == 200, "penalty options accepted")
    # raw
    st, r = gen(18101, prompt="abc", raw=True, options={"num_predict": 2, "seed": 1})
    st2, tk = j(18101, "/tokenize", {"content": "abc", "add_special": True})
    check(st == 200 and r.get("eval_count") == 2 and r.get("done_reason") == "length", f"raw:true served via /completion ({st}, {str(r)[:120]})")
    check(st == 200 and r.get("prompt_eval_count") == len(tk["tokens"]), f"raw prompt not chat-templated: prompt_eval_count {r.get('prompt_eval_count')} == tokenize('abc') {len(tk['tokens'])}")
    st, _ = gen(18101, prompt="abc", raw=True, system="x")
    check(st == 400, "raw + system -> 400")
    st, r = gen(18101, prompt="hi", format="xml")
    check(st == 400 and "invalid format" in json.dumps(r), "invalid format string -> 400")
    st, r = gen(18101, prompt="hi", format={"type": "object"}, options={"num_predict": 2})
    st_in, _ = j(18101, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
                 "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}}})
    check(st == st_in, f"format schema object reaches the grammar path exactly like response_format ({st} == {st_in})")
    # keep_alive accepted in single-model mode
    st, r = gen(18101, prompt="hi", keep_alive="5m", options={"num_predict": 1})
    check(st == 200, "keep_alive passthrough does not break a single-model server")
    # show completeness
    st, r = j(18101, "/api/show", {"model": "tiny"})
    check(st == 200 and "completion" in r["capabilities"], "show.capabilities has completion")
    check(r["model_info"].get("llama.context_length", 0) > 0 and r["model_info"].get("llama.embedding_length") == 32, f"show.model_info arch-prefixed keys: {r['model_info']}")
    check("im_start" in r["template"], "show.template is the chat template")
    # /api/ps size / size_vram come from the backend's allocation record (weights + KV + compute), not the GGUF file size
    st, ps = j(18101, "/api/ps")
    st, tg = j(18101, "/api/tags")
    pm, tm = ps["models"][0], tg["models"][0]
    check(pm["size"] > tm["size"] and pm.get("size_vram") == 0, f"ps.size {pm['size']} > weights-only tags.size {tm['size']} (KV + compute buffers counted); size_vram {pm.get('size_vram')} == 0 on a CPU-only box")
    st, mi = j(18101, "/v1/models")
    check(st == 200, "unchanged /v1/models still answers")
    # any name accepted when strict is OFF
    st, _ = j(18101, "/api/show", {"model": "whatever"})
    check(st == 200, "strict names OFF: unknown name accepted (bit-identical default)")
    # think values (api/types.go ValidateLegacyThinking): levels pass, anything else is a 400 with ollama's message
    for lvl in ("low", "medium", "high", "max", True, False):
        st, r = gen(18101, prompt="hi", think=lvl, options={"num_predict": 1})
        check(st == 200, f"think={lvl!r} accepted")
    st, r = gen(18101, prompt="hi", think="banana")
    check(st == 400 and 'invalid think value: "banana" (must be "high", "medium", "low", "max", true, or false)' in r.get("error", ""), f"think='banana' -> 400 with ollama's message ({st} {str(r)[:120]})")
    st, r = gen(18101, prompt="hi", think=5)
    check(st == 400 and "think must be a boolean or string" in json.dumps(r), f"think=5 -> 400 ({st})")
    st, r = j(18101, "/api/chat", {"model": "tiny", "stream": False, "messages": [{"role": "user", "content": "hi"}], "think": "x"})
    check(st == 400, "/api/chat validates think too")
    # tool-call history reaches the template (template below renders tool_calls)
finally:
    s.stop()

# ------------------------------------------------------------ tool history through a template that prints tool_calls
TPL = ("{% for m in messages %}[{{ m.role }}]{{ m.content }}"
       "{% if m.tool_calls %}{% for tc in m.tool_calls %} CALL {{ tc.function.name }} {{ tc.function.arguments | tojson }}{% endfor %}{% endif %}"
       "{% if m.reasoning_content %} THINK {{ m.reasoning_content }}{% endif %}\n{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}")
tf = os.path.join(WORK, "tpl.jinja"); open(tf, "w").write(TPL)
s = Srv(18102, "--chat-template-file", tf)
try:
    base = [{"role": "user", "content": "weather?"}, {"role": "assistant", "content": ""}, {"role": "tool", "tool_name": "get_weather", "content": "sunny"}]
    withc = [dict(m) for m in base]
    withc[1] = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]}
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "w", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    def chat(msgs): return j(18102, "/api/chat", {"model": "tiny", "stream": False, "messages": msgs, "options": {"num_predict": 1}})
    st1, r1 = chat(base); st2, r2 = chat(withc)
    check(st1 == 200 and st2 == 200, f"tool history requests ok ({st1},{st2}: {str(r2)[:150]})")
    check(st2 == 200 and r2["prompt_eval_count"] > r1["prompt_eval_count"] + 6, f"assistant tool_calls reach the prompt: {r1.get('prompt_eval_count')} -> {r2.get('prompt_eval_count')}")
    up = [dict(m) for m in base]; up[0]["role"] = "USER"
    st3, r3 = chat(up)
    check(st3 == 200 and r3["prompt_eval_count"] == r1["prompt_eval_count"], "role is lower-cased (USER == user)")
    # thinking in history
    th = [dict(m) for m in base]; th[1] = {"role": "assistant", "content": "", "thinking": "pondering the weather deeply"}
    st4, r4 = chat(th)
    check(st4 == 200 and r4["prompt_eval_count"] > r1["prompt_eval_count"] + 3, "assistant thinking reaches the prompt as reasoning_content")
finally:
    s.stop()

# ------------------------------------------------------------ Modelfile + strict names
mf = os.path.join(WORK, "Modelfile")
open(mf, "w").write('FROM tiny\nPARAMETER temperature 0\nPARAMETER top_k 1\nPARAMETER stop "ZZZ"\nSYSTEM """You are a very verbose assistant."""\nMESSAGE user ping\nMESSAGE assistant pong\n')
s = Srv(18103, "--modelfile", mf, "--ollama-strict-model-names")
try:
    check("modelfile: loaded" in s.logtext(), "always-on log line shows the Modelfile reached the shim")
    outs = [gen(18103, prompt="hello", options={"num_predict": 10, "seed": sd})[1]["response"] for sd in (1, 2, 3)]
    check(len(set(outs)) == 1, "Modelfile PARAMETER temperature 0 / top_k 1 makes output seed-independent")
    st, r = gen(18103, prompt="hello", options={"num_predict": 10, "seed": 5, "temperature": 2.0, "top_k": 0})
    # request options win: with temperature 2 and top_k disabled the output differs from the greedy output at some seed
    alt = {gen(18103, prompt="hello", options={"num_predict": 10, "seed": sd, "temperature": 2.0, "top_k": 0})[1]["response"] for sd in (1, 2, 3, 4)}
    check(len(alt) > 1, "request options override Modelfile defaults")
    st, a = chat = j(18103, "/api/chat", {"model": "tiny", "stream": False, "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 1}})
    st, b = j(18103, "/api/chat", {"model": "tiny", "stream": False, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}], "options": {"num_predict": 1}})
    check(a["prompt_eval_count"] > b["prompt_eval_count"] + 8, f"Modelfile SYSTEM + MESSAGE seeds are in the prompt ({a['prompt_eval_count']} vs explicit-system {b['prompt_eval_count']})")
    st, r = j(18103, "/api/show", {"model": "tiny"})
    check("temperature" in r["parameters"] and "verbose" in r["modelfile"] and r["system"].startswith("You are"), "show reports parameters / modelfile / system")
    for nm in ("tiny", "tiny:latest", "library/tiny", "registry.ollama.ai/library/TINY:latest"):
        st, _ = gen(18103, prompt="x", options={"num_predict": 1}) if False else j(18103, "/api/generate", {"model": nm, "prompt": "x", "stream": False, "options": {"num_predict": 1}})
        check(st == 200, f"strict names: '{nm}' resolves to the loaded model")
    st, r = j(18103, "/api/generate", {"model": "other", "prompt": "x", "stream": False})
    check(st == 404, f"strict names: 'other' -> 404 ({st})")
    st, r = j(18103, "/api/show", {"model": "other"})
    check(st == 404, "strict names: show 'other' -> 404")
finally:
    s.stop()


# ------------------------------------------------------------ Ollama model store: --ollama-model / --ollama-go-template / copy / delete / env
def sha(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
def put_blob(store, data):
    h = hashlib.sha256(data).hexdigest()
    open(os.path.join(store, "blobs", "sha256-" + h), "wb").write(data)
    return "sha256:" + h, len(data)
def layer(mt, dg_sz): return {"mediaType": mt, "digest": dg_sz[0], "size": dg_sz[1]}
def write_manifest(store, name_path, layers):
    d = os.path.join(store, "manifests", "registry.ollama.ai", *name_path[:-1]); os.makedirs(d, exist_ok=True)
    cfg = put_blob(store, b"{}")
    open(os.path.join(d, name_path[-1]), "w").write(json.dumps({"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
                            "config": layer("application/vnd.docker.container.image.v1+json", cfg), "layers": layers}))

store = os.path.join(WORK, "store"); shutil.rmtree(store, ignore_errors=True); os.makedirs(os.path.join(store, "blobs"))
gguf_with = open(MODEL, "rb").read()
GOTPL = b'<<GO>>{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n'
lyr_model = put_blob(store, gguf_with)
lyr_tpl = put_blob(store, GOTPL)
lyr_sys = put_blob(store, b"You are a store-layer system prompt.")
lyr_par = put_blob(store, json.dumps({"temperature": 0, "top_k": 1, "stop": ["ZZZ"]}).encode())
lyr_lic = put_blob(store, b"TEST-LICENSE-TEXT")
write_manifest(store, ["library", "mytiny", "7b"], [layer("application/vnd.ollama.image.model", lyr_model), layer("application/vnd.ollama.image.template", lyr_tpl),
               layer("application/vnd.ollama.image.system", lyr_sys), layer("application/vnd.ollama.image.params", lyr_par), layer("application/vnd.ollama.image.license", lyr_lic)])
write_manifest(store, ["acme", "other", "latest"], [layer("application/vnd.ollama.image.model", lyr_model)])

s = Srv(18110, "--ollama-model", "mytiny:7b", "--ollama-models-dir", store, model=False)
try:
    lg = s.logtext()
    check("--ollama-model: mytiny:7b -> " + os.path.join(store, "blobs", "sha256-" + sha(MODEL)) in lg, "store name resolved to its GGUF blob (always-on log line)")
    check("ollama store: mytiny:7b -> system=yes, 3 params" in lg and "1 licenses" in lg, "store layers reached the shim state (log line)")
    outs = [gen(18110, prompt="hello", options={"num_predict": 10, "seed": sd})[1]["response"] for sd in (1, 2, 3)]
    check(len(set(outs)) == 1, "store params layer (temperature 0, top_k 1) makes output seed-independent")
    s2 = Srv(18111, model=True); 
    ctrl = [gen(18111, prompt="hello", options={"num_predict": 10, "seed": sd})[1]["response"] for sd in (1, 2, 3, 4)]
    s2.stop()
    check(len(set(ctrl)) > 1, f"control (-m, no store): seeds differ ({len(set(ctrl))} distinct)")
    st, r = j(18110, "/api/show", {"model": "mytiny:7b"})
    check(r.get("system") == "You are a store-layer system prompt." and "TEST-LICENSE-TEXT" in r.get("license", "") and "temperature" in r.get("parameters", ""), "show reports store system / license / parameters")
    st, r = j(18110, "/api/tags")
    names = sorted(m["name"] for m in r["models"])
    check("acme/other:latest" in names, f"tags lists the store's other model: {names}")
    ent = [m for m in r["models"] if m["name"] == "acme/other:latest"][0]
    check(ent["size"] == lyr_model[1] + 2 and ent["digest"].startswith("sha256:") and len(ent["digest"]) == 71, f"store entry size = layer sizes + config, digest = manifest sha: {ent['size']} {ent['digest'][:20]}")
    # copy / delete are refused (403) while the store is read-only: control for the write-enabled run below
    st, r = j(18110, "/api/copy", {"source": "mytiny:7b", "destination": "copy:v1"})
    check(st == 403, f"copy refused without --ollama-store-write ({st})")
    st, r = call(18110, "/api/delete", {"model": "acme/other"}, "DELETE")
    check(st == 403 and os.path.exists(os.path.join(store, "manifests", "registry.ollama.ai", "acme", "other", "latest")), f"delete refused, manifest still on disk ({st})")
finally:
    s.stop()

s = Srv(18112, "--ollama-model", "mytiny:7b", "--ollama-models-dir", store, "--ollama-store-write", model=False)
try:
    st, r = j(18112, "/api/copy", {"source": "mytiny:7b", "destination": "copy:v1"})
    check(st == 200 and r.get("status") == "success", f"copy ok ({st} {r})")
    st, r = j(18112, "/api/tags")
    check("copy:v1" in [m["name"] for m in r["models"]], "copied name appears in tags")
    st, r = j(18112, "/api/copy", {"source": "nope", "destination": "x:1"})
    check(st == 404, f"copy of a missing source -> 404 ({st})")
    st, r = j(18112, "/api/copy", {"source": "mytiny:7b", "destination": "a/b/c/d:e"})
    check(st == 400, f"copy to an invalid destination -> 400 ({st})")
    st, tx = call(18112, "/api/delete", {"model": "copy:v1"}, "DELETE")
    check(st == 200, f"delete copy ok ({st})")
    check(os.path.exists(os.path.join(store, "blobs", "sha256-" + lyr_model[0][7:])), "blob shared with the original survives the delete")
    st, tx = call(18112, "/api/delete", {"model": "acme/other"}, "DELETE")
    check(st == 200 and not os.path.exists(os.path.join(store, "manifests", "registry.ollama.ai", "acme")), "delete removes the manifest and prunes its empty dirs")
    check(os.path.exists(os.path.join(store, "blobs", "sha256-" + lyr_model[0][7:])), "model blob still referenced by mytiny:7b is kept")
    st, tx = call(18112, "/api/delete", {"model": "acme/other"}, "DELETE")
    check(st == 404, f"second delete -> 404 ({st})")
finally:
    s.stop()

# a name that is not in the store stops the server at startup, naming the store
p = subprocess.run([SERVER, "--ollama-model", "ghost:1", "--ollama-models-dir", store, "--port", "18113"], capture_output=True, text=True, timeout=120)
check(p.returncode != 0 and "not found" in (p.stdout + p.stderr), "unknown store model refused at startup")

# Go TEMPLATE layer -> Jinja, used only when the GGUF has no chat template
if len(sys.argv) > 4:
    nt = os.path.join(WORK, "tiny_notpl.gguf")
    subprocess.run(["/usr/bin/python3", os.path.join(sys.argv[4], "tests/ollama-engine/make_tiny_llama_server_gguf.py"), sys.argv[4], nt, "--no-template"], check=True, capture_output=True)
    store2 = os.path.join(WORK, "store2"); shutil.rmtree(store2, ignore_errors=True); os.makedirs(os.path.join(store2, "blobs"))
    m2 = put_blob(store2, open(nt, "rb").read()); t2 = put_blob(store2, GOTPL)
    write_manifest(store2, ["library", "notpl", "latest"], [layer("application/vnd.ollama.image.model", m2), layer("application/vnd.ollama.image.template", t2)])
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    want = "<<GO>><|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    s = Srv(18114, "--ollama-model", "notpl", "--ollama-models-dir", store2, "--ollama-go-template", model=False)
    try:
        check("translated the Go TEMPLATE layer" in s.logtext(), "Go TEMPLATE translated at startup (log line)")
        st, r = j(18114, "/apply-template", {"messages": msgs})
        check(st == 200 and r.get("prompt") == want, f"translated template renders the ollama prompt: {str(r)[:200]}")
    finally:
        s.stop()
    s = Srv(18115, "--ollama-model", "notpl", "--ollama-models-dir", store2, model=False)
    try:
        st, r = j(18115, "/apply-template", {"messages": msgs})
        check(st != 200 or r.get("prompt") != want, "control (flag OFF): the GGUF has no template and the Go layer is not used")
    finally:
        s.stop()
    # a GGUF that has its own template keeps it
    s = Srv(18116, "--ollama-model", "mytiny:7b", "--ollama-models-dir", store, "--ollama-go-template", model=False)
    try:
        check("the GGUF has its own chat template" in s.logtext(), "GGUF template wins over the Go layer (server/images.go PreferChatTemplate)")
    finally:
        s.stop()
else:
    print("SKIP Go-template checks (pass the tree root as argv[4])")

# --ollama-env: OLLAMA_* variables reach the context parameters; an explicit flag wins; flag OFF ignores them
ENV = {"OLLAMA_KV_CACHE_TYPE": "f32", "OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_NUM_PARALLEL": "2"}
s = Srv(18117, "--ollama-env", env=ENV)
try:
    lg = s.logtext()
    check("--ollama-env: kv=f32 flash_attn=1 n_ctx=512 n_parallel=1 models_keep_alive_ms=300000" in lg, "env mapping log line (explicit -c 512 -np 1 beat OLLAMA_NUM_PARALLEL=2; ollama default keep_alive 5m)")
    check("K (f32)" in lg, "KV cache type f32 reached the context (llama.cpp log; q8_0 needs head dim % 32 == 0, the tiny model has 16)")
finally:
    s.stop()
s = Srv(18118, env=ENV)
try:
    lg = s.logtext()
    check("--ollama-env:" not in lg and "K (f32)" not in lg, "control (flag OFF): OLLAMA_KV_CACHE_TYPE ignored (K stays f16)")
finally:
    s.stop()
s = Srv(18119, "--ollama-env", "-ctk", "f16", "-ctv", "f16", env=ENV)
try:
    lg = s.logtext()
    check("K (f32)" not in lg, "explicit -ctk wins over OLLAMA_KV_CACHE_TYPE")
finally:
    s.stop()

s = Srv(18120, "--ollama-env", env={"OLLAMA_FLASH_ATTENTION": "0"})
try:
    check("flash_attn            = disabled" in s.logtext(), "OLLAMA_FLASH_ATTENTION=0 + --ollama-env disables flash attention (llama.cpp log)")
finally:
    s.stop()
s = Srv(18121, env={"OLLAMA_FLASH_ATTENTION": "0"})
try:
    check("flash_attn            = enabled" in s.logtext(), "control (flag OFF): OLLAMA_FLASH_ATTENTION=0 ignored, flash attention stays on")
finally:
    s.stop()

# ------------------------------------------------------------ /api/embed truncate (ollama EmbedRequest.Truncate, default true)
s = Srv(18122, "--embeddings", "--pooling", "mean")
try:
    LONG = "hello " * 300
    st, r = j(18122, "/api/embed", {"model": "tiny", "input": LONG})
    check(st == 200 and len(r["embeddings"]) == 1 and len(r["embeddings"][0]) == 32, f"over-long input is truncated and embedded ({st} {str(r)[:80]})")
    import re
    mt = re.search(r"ollama embed: input of (\d+) tokens truncated to (\d+)", s.logtext())
    check(bool(mt) and int(mt.group(2)) <= 255, f"truncation log line ({mt.group(0) if mt else None}); limit is the 256 train context minus the BOS")
    NT = int(mt.group(2)) if mt else 255
    st, e = j(18122, "/api/embed", {"model": "tiny", "input": LONG, "truncate": False})
    check(st == 400 and "exceeds the context length" in json.dumps(e), f"truncate:false -> 400 ({st} {str(e)[:100]})")
    st, e = j(18122, "/v1/embeddings", {"model": "tiny", "input": LONG})
    check(st == 500 and "too large" in json.dumps(e), f"control: the native endpoint keeps refusing over-long input ({st})")
    st, tk = j(18122, "/tokenize", {"content": LONG, "add_special": False})
    st, dt = j(18122, "/detokenize", {"tokens": tk["tokens"][:NT]})
    st, r1 = j(18122, "/api/embed", {"model": "tiny", "input": dt["content"]})
    st, r2 = j(18122, "/api/embed", {"model": "tiny", "input": LONG})
    d = max(abs(a - b) for a, b in zip(r1["embeddings"][0], r2["embeddings"][0]))
    check(d < 1e-6, f"truncated embedding == embedding of the first NT tokens (max diff {d:.2e})")
    st, r3 = j(18122, "/api/embed", {"model": "tiny", "input": ["hello", LONG, "abc"]})
    check(st == 200 and len(r3["embeddings"]) == 3, "a batch with one over-long item still answers every item")
    st, r4 = j(18122, "/api/embed", {"model": "tiny", "input": "hello", "truncate": False})
    check(st == 200, "truncate:false with a short input is fine")
finally:
    s.stop()


# FunctionGemma tool-call format reaches the real chat path (peg-functiongemma), selected from the template alone
if len(sys.argv) > 4:
    fgt = os.path.join(sys.argv[4], "models/templates/unsloth-functiongemma-270m-it.jinja")
    TOOLS = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}}]
    s = Srv(18123, "--chat-template-file", fgt, "-c", "2048", "-lv", "5")
    try:
        j(18123, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "tools": TOOLS})
        check("chat format: peg-functiongemma" in s.logtext(), "FunctionGemma template selects the peg-functiongemma parser on the real request path")
    finally:
        s.stop()
    s = Srv(18124, "-c", "2048", "-lv", "5")
    try:
        j(18124, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "tools": TOOLS})
        check("chat format: " in s.logtext() and "peg-functiongemma" not in s.logtext(), "control: the default (chatml) template does not")
    finally:
        s.stop()

# a malformed Modelfile stops the server at startup with the parser message
open(mf, "w").write("PARAMETER temperature 0\n")
p = subprocess.run([SERVER, "-m", MODEL, "--port", "18104", "--modelfile", mf], capture_output=True, text=True, timeout=120)
check(p.returncode != 0 and "no FROM line" in (p.stdout + p.stderr), "malformed Modelfile refused at startup with parser message")

print("FAILED" if fails else "ALL PASS", fails)
sys.exit(1 if fails else 0)
