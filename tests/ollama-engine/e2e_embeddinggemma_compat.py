#!/usr/bin/env python3
"""Ollama-layout embeddinggemma (arch gemma3 + dense.0/dense.1) must load through the real llama-server and embed IDENTICALLY to the same
weights in llama.cpp layout (arch gemma-embedding + dense_2/dense_3). usage: e2e_embeddinggemma_compat.py <llama-server> <tree-root> <workdir>"""
import json, os, subprocess, sys, time, urllib.request
SERVER, ROOT, WORK = sys.argv[1:4]
os.makedirs(WORK, exist_ok=True)
subprocess.run(["/usr/bin/python3", os.path.join(ROOT, "tests/ollama-engine/make_tiny_embeddinggemma.py"), ROOT, WORK], check=True, capture_output=True)
fails = 0
def check(ok, msg):
    global fails
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok: fails += 1
def embed(gguf, port):
    log = open(os.path.join(WORK, f"eg{port}.log"), "w")
    p = subprocess.Popen([SERVER, "-m", gguf, "--embeddings", "--port", str(port), "-c", "128", "-np", "1", "-t", "2", "-ngl", "0"], stdout=log, stderr=subprocess.STDOUT)
    vec = None
    try:
        for _ in range(120):
            time.sleep(0.5)
            if p.poll() is not None: break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
            except Exception: pass
        if p.poll() is None:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings", data=json.dumps({"input": "the hello the a"}).encode(), headers={"Content-Type": "application/json"})
            vec = json.loads(urllib.request.urlopen(req, timeout=60).read())["data"][0]["embedding"]
    except Exception as e:
        print("  request failed:", e)
    finally:
        if p.poll() is None: p.terminate()
        p.wait(30); log.close()
    return vec, open(log.name).read()
ref, rlog = embed(os.path.join(WORK, "ref.gguf"), 18301)
check(ref is not None and len(ref) == 32, f"reference (llama.cpp layout) embeds: dim {len(ref) if ref else None}")
check("Ollama-format embeddinggemma" not in rlog, "no compat message on a llama.cpp-layout file (bit-identical default)")
oll, olog = embed(os.path.join(WORK, "ollama.gguf"), 18302)
check(oll is not None and "detected Ollama-format embeddinggemma" in olog, "Ollama-layout file loads and the compat log line fired")
if ref and oll:
    d = max(abs(a - b) for a, b in zip(ref, oll))
    check(d < 1e-6, f"Ollama-layout embedding == llama.cpp-layout embedding (max diff {d:.2e})")
    check(max(abs(x) for x in ref) > 1e-3, "embedding is not degenerate")
print("FAILED" if fails else "ALL PASS", fails)
sys.exit(1 if fails else 0)
