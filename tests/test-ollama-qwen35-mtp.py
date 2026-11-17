#!/usr/bin/env python3
"""Ollama qwen35 blob: the embedded mtp.* head must load as the NextN/MTP layer and draft tokens that are accepted.

llama_model_loader::apply_ollama_qwen35_mtp_compat translates an Ollama-format qwen35 blob (MTP head stored as mtp.*,
block_count without it) to the llama.cpp layout when MTP is requested, and adds +1.0 to the three raw norms
(enorm, hnorm, shared_head_norm) at read time.

Arms, all CPU, greedy, same prompt:
  ON       llama-speculative-simple --spec-type mtp                 -> head translated, drafts accepted
  CONTROL  the same with LLAMA_OLLAMA_COMPAT=0 (the opt-out; the loader then behaves exactly as before the change)
           -> no translation, no accepted draft
  GREEDY   llama-completion, no speculation                         -> the ON arm must produce the same text

Fixture: the ollama qwen3.6:27b blob (64 trunk blocks + 15 mtp.* tensors + vision). It is a 17 GB model, so this test
is labelled "model" and is slow on CPU (minutes). Set LLAMA_TEST_OLLAMA_QWEN35_BLOB to a blob path, or OLLAMA_MODELS to
an ollama store. Exit 77 = skipped (no binary, no blob).
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SKIP = 77
PROMPT = ("Here is a Python function that returns the n-th Fibonacci number:\n\n"
          "def fib(n):\n")
N_PREDICT = 48

RE_TRANSLATED = re.compile(r"Ollama qwen35 MTP head translated: (\d+) mtp\.\* tensors -> blk\.(\d+), block_count (\d+) -> (\d+)")
RE_DRAFTED = re.compile(r"n_drafted\s*=\s*(\d+)")
RE_ACCEPT_N = re.compile(r"n_accept\s*=\s*(\d+)")
RE_ACCEPT = re.compile(r"accept\s*=\s*([0-9.]+)%")


def find_blob():
    p = os.environ.get("LLAMA_TEST_OLLAMA_QWEN35_BLOB")
    if p:
        return Path(p)
    store = Path(os.environ.get("OLLAMA_MODELS", "models/ollama"))  # portability-ok: default store, env override above
    man = store / "manifests" / "registry.ollama.ai" / "library" / "qwen3.6" / "27b"
    if not man.is_file():
        return None
    m = json.loads(man.read_text())
    digest = [l["digest"] for l in m["layers"] if l["mediaType"].endswith(".model")][0]
    return store / "blobs" / digest.replace(":", "-")


def find_bin(name):
    for d in [os.environ.get("LLAMA_BIN_DIR", ""), str(Path(sys.argv[0]).resolve().parent), "bin", "../bin"]:
        if not d:
            continue
        for cand in (Path(d) / (name + ".exe"), Path(d) / name):
            if cand.is_file():
                return cand
    return None


def run(argv, env_extra, log_path):
    env = dict(os.environ)
    env.update(env_extra)
    p = subprocess.run([str(a) for a in argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=3600)
    out = p.stdout.decode("utf-8", errors="replace")
    err = p.stderr.decode("utf-8", errors="replace")
    if log_path:
        Path(log_path).write_text(out + "\n----- stderr -----\n" + err, encoding="utf-8")
    return p.returncode, out, err


def spec_generated(text):
    # speculative-simple LOGs the prompt then streams the generated pieces; the stats block starts with "encoded"
    i = text.find(PROMPT.strip().splitlines()[-1])
    if i < 0:
        return ""
    tail = text[i + len(PROMPT.strip().splitlines()[-1]):]
    j = tail.find("\nencoded")
    return tail[:j] if j >= 0 else tail


def main():
    blob = find_blob()
    spec = find_bin("llama-speculative-simple")
    comp = find_bin("llama-completion")
    if blob is None or not blob.is_file() or spec is None or comp is None:
        print("SKIP: blob=%s spec=%s completion=%s" % (blob, spec, comp))
        return SKIP
    threads = os.environ.get("LLAMA_TEST_THREADS", "8")
    log_dir = os.environ.get("LLAMA_TEST_LOG_DIR", "")
    lp = (lambda n: os.path.join(log_dir, n)) if log_dir else (lambda n: None)

    common = ["-m", blob, "-t", threads, "-c", "1024", "--temp", "0", "-n", str(N_PREDICT), "-p", PROMPT, "-ngl", "0"]
    spec_args = [spec] + common + ["--spec-type", "mtp:n_max=1,p_min=0.0"]

    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            fails += 1

    rc_on, on_out, on_err = run(spec_args, {}, lp("mtp_on.log"))
    rc_ct, ct_out, ct_err = run(spec_args, {"LLAMA_OLLAMA_COMPAT": "0"}, lp("mtp_control.log"))
    rc_gr, gr, _ = run([comp] + common + ["-no-cnv", "--no-display-prompt"], {}, lp("greedy.log"))
    on = on_out + on_err
    ct = ct_out + ct_err

    m = RE_TRANSLATED.search(on)
    check(rc_on == 0, "ON arm exits 0 (rc %d)" % rc_on)
    check(m is not None and m.group(1) == "15", "ON arm: head translated, 15 mtp.* tensors (%s)" % (m.group(0) if m else None))
    d = RE_DRAFTED.search(on)
    a = RE_ACCEPT_N.search(on)
    r = RE_ACCEPT.search(on)
    n_d = int(d.group(1)) if d else -1
    n_a = int(a.group(1)) if a else -1
    rate = float(r.group(1)) if r else -1.0
    check(n_d > 0, "ON arm drafted tokens (n_drafted %d)" % n_d)
    # an MTP head with wrong norms or wrong tensor mapping drafts noise and is accepted ~never
    check(rate >= 40.0, "ON arm acceptance >= 40%% (%.1f%%, %d of %d)" % (rate, n_a, n_d))

    check(RE_TRANSLATED.search(ct) is None, "CONTROL (LLAMA_OLLAMA_COMPAT=0): no translation")
    a_ct = RE_ACCEPT_N.search(ct)
    check(rc_ct != 0 or a_ct is None or int(a_ct.group(1)) == 0,
          "CONTROL: no accepted draft without the translation (rc %d, n_accept %s)" % (rc_ct, a_ct.group(1) if a_ct else None))

    check(rc_gr == 0, "GREEDY arm exits 0 (rc %d)" % rc_gr)
    g_on = spec_generated(on_out).strip()
    g_gr = gr.strip()
    check(len(g_on) > 0, "ON arm generated text (%d chars)" % len(g_on))
    # speculative decoding with greedy verification must not change the greedy text; compare the common prefix of the two
    # renderings (the tools stop at slightly different points)
    n = min(len(g_on), len(g_gr))
    check(n >= 40 and g_on[:n] == g_gr[:n], "ON arm text == plain greedy text over %d chars" % n)
    if n and g_on[:n] != g_gr[:n]:
        print("  ON    : %r" % g_on[:200])
        print("  GREEDY: %r" % g_gr[:200])

    print("RESULT: %s (%d failed)" % ("PASS" if fails == 0 else "FAIL", fails))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
