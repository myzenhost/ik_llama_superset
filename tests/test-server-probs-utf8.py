#!/usr/bin/env python3
"""Per-token probabilities stay one entry per generated token when a character spans several tokens.

usage: test-server-probs-utf8.py [<llama-server(.exe) or bin dir>] [<model.gguf>] [<port>] [server log]
       (defaults: ./bin, LLAMA_TEST_PROBS_GGUF or Qwen3.5-0.8B-Q8_0, port 18996; exit 77 without a binary or model)

A token that ends inside a multi-byte UTF-8 character carries no complete text of its own; the server holds the
bytes until the character is complete. Until 2026-10-18 that token was also left out of completion_probabilities
(and, when streaming, never sent, so its id and probs were lost). Every split character then removed one entry and
shifted every later entry by one token, and a losslessness check that mapped text positions to probability entries
read the wrong token's top-2 gap.

A grammar forces output made of rare characters (byte-level BPE splits them into several tokens). Then:
  1. non-stream, n_probs 2 + return_tokens: one probability entry per returned token, ids in the same order;
  2. stream, n_probs 1 + return_tokens: the streamed token ids equal the non-stream ids, one probability entry each,
     and the concatenated content equals the non-stream content;
  3. control, stream without n_probs / return_tokens: same content, and no chunk is sent for a held token (the
     default stream is unchanged).
Exit 77 (skipped) if the model's vocabulary encodes every forced character as one token: nothing was exercised.
Before the fix check 1 FAILS (fewer entries than tokens) and check 2 FAILS (fewer streamed ids).
"""
import http.client
import json
import os
import subprocess
import sys
import time

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
# portability-ok: default fixture path; LLAMA_TEST_PROBS_GGUF or the second argument overrides it
DEFAULT_MODEL = 'models/gguf/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf'
arg0 = sys.argv[1] if len(sys.argv) > 1 else 'bin'
server_exe = next((c for c in (arg0, os.path.join(arg0, 'llama-server.exe'), os.path.join(arg0, 'llama-server'))
                   if os.path.isfile(c)), None)
model = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('LLAMA_TEST_PROBS_GGUF', DEFAULT_MODEL)
port = int(sys.argv[3]) if len(sys.argv) > 3 else 18996
if server_exe is None or not os.path.isfile(model):
    print('SKIP: no llama-server under %r or no model at %r' % (arg0, model))
    sys.exit(77)
fails, passes = [], 0
FORCED = '\U0001F9A9\U00013000\U0001F9CC\U00013001'  # flamingo, two hieroglyphs, troll: rare, so split into bytes


def check(cond, msg):
    global passes
    if cond:
        passes += 1
        print('PASS', msg, flush=True)
    else:
        fails.append(msg)
        print('FAIL', msg, flush=True)


def post(body, timeout=300):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    c.request('POST', '/completion', body=json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def stream_chunks(body, timeout=300):
    st, data = post(dict(body, stream=True), timeout)
    chunks = []
    for line in data.decode('utf-8').splitlines():
        if line.startswith('data: ') and line[6:].strip() != '[DONE]':
            chunks.append(json.loads(line[6:]))
    return st, chunks


log = open(sys.argv[4], 'wb') if len(sys.argv) > 4 else subprocess.DEVNULL  # optional server log path
srv = subprocess.Popen([server_exe, '-m', model, '-c', '2048', '-np', '1', '-t', '2', '--host', '127.0.0.1', '--port', str(port)],
                       stdout=log, stderr=log, creationflags=CREATE_NO_WINDOW, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
try:
    deadline = time.time() + 120
    while True:
        try:
            c = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
            c.request('GET', '/health')
            if c.getresponse().status == 200:
                break
        except OSError:
            pass
        if time.time() > deadline or srv.poll() is not None:
            print('FAIL server did not come up (rc=%s)' % srv.poll())
            sys.exit(1)
        time.sleep(0.5)

    base = {'prompt': 'Write four symbols:', 'temperature': 0, 'n_predict': 64, 'cache_prompt': False,
            'grammar': 'root ::= "%s"' % FORCED}

    # 1. non-stream
    st, data = post(dict(base, n_probs=2, return_tokens=True))
    check(st == 200, 'non-stream status 200 (got %d)' % st)
    j = json.loads(data)
    toks = j.get('tokens') or []
    probs = j.get('completion_probabilities') or []
    content = j.get('content', '')
    print('content %r, %d tokens %s, %d probability entries' % (content, len(toks), toks, len(probs)), flush=True)
    check(content == FORCED, 'grammar forced the expected text')
    if len(toks) <= len(FORCED):
        print('SKIP: every forced character is one token in this vocabulary; nothing split, nothing exercised')
        sys.exit(77)
    check(len(probs) == len(toks), 'one probability entry per generated token (%d entries, %d tokens)' % (len(probs), len(toks)))
    check([p.get('id') for p in probs] == toks, 'probability entry ids equal the generated ids, in order')

    # 2. stream with per-token ids + probs
    st, id_chunks = stream_chunks(dict(base, n_probs=1, return_tokens=True))
    check(st == 200, 'stream status 200 (got %d)' % st)
    s_toks = [t for ch in id_chunks for t in (ch.get('tokens') or [])]
    s_probs = [p for ch in id_chunks for p in (ch.get('completion_probabilities') or [])]
    s_text = ''.join(ch.get('content', '') for ch in id_chunks)
    check(s_toks == toks, 'streamed ids equal the non-stream ids (%s vs %s)' % (s_toks, toks))
    check(len(s_probs) == len(toks), 'one streamed probability entry per token (%d vs %d)' % (len(s_probs), len(toks)))
    check(s_text == content, 'streamed content equals the non-stream content')

    # 3. control: the default stream is unchanged -- it carries no chunk for a held token. A held-token chunk is one
    # with empty content that is not final and is not the last generated token (the end-of-generation token's own
    # chunk has empty content too, and was always sent). The control must have exactly the other chunks.
    n_held = sum(1 for ch in id_chunks if ch.get('content', '') == '' and not ch.get('stop')
                 and (ch.get('tokens') or [None])[-1] != toks[-1])
    st, chunks = stream_chunks(base)
    check(st == 200, 'control stream status 200 (got %d)' % st)
    c_text = ''.join(ch.get('content', '') for ch in chunks)
    check(c_text == content, 'control stream content equals the non-stream content')
    check(len(chunks) == len(id_chunks) - n_held,
          'control stream sends no chunk for a held token (%d chunks; id stream %d minus %d held)' % (len(chunks), len(id_chunks), n_held))
finally:
    srv.terminate()
    try:
        srv.wait(30)
    except subprocess.TimeoutExpired:
        srv.kill()

print('%d passed, %d failed' % (passes, len(fails)))
sys.exit(1 if fails else 0)
