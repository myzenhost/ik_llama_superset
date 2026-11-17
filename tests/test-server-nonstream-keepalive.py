#!/usr/bin/env python3
"""Non-stream keepalive (X-KoboldCpp-Keepalive) against a real llama-server on the CPU.

usage: test-server-nonstream-keepalive.py <llama-server(.exe)> <model.gguf> <port> [server log]

Starts the server (CPU, 1 slot), then:
  0. calibrates n_predict so a plain non-stream /completion takes >= 3 s (temperature 0, ignore_eos);
  1. the same request WITH "X-KoboldCpp-Keepalive: 1": status 200, header X-Keepalive: whitespace, the first body byte
     arrives well before the end, the body is whitespace + one JSON object whose content equals the plain answer;
  1b. the slow request with "X-KoboldCpp-Keepalive: false": off (KoboldCpp enables it only on "true"), the plain form;
  2. a short request WITH the header (done inside 1 s): answered exactly as without it -- no leading whitespace,
     a Content-Length, no X-Keepalive header;
  3. a fast failure WITH the header (a prompt longer than the slot context is refused by the slot): its real 4xx/5xx
     status, not 200;
  4. /v1/chat/completions with the header: the keepalive form too, and the body parses as an OAI chat object.
Before the feature the header is ignored: check 1 FAILS (no whitespace, no header, first byte only at the end).
"""
import http.client
import json
import os
import subprocess
import sys
import time

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
server_exe, model, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
fails, passes = [], 0


def check(cond, msg):
    global passes
    if cond:
        passes += 1
        print('PASS', msg, flush=True)
    else:
        fails.append(msg)
        print('FAIL', msg, flush=True)


def request(path, body, headers=None, timeout=300):
    """POST; returns (status, headers dict lower-cased, body bytes, seconds to first body byte, total seconds)."""
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    h = {'Content-Type': 'application/json'}
    h.update(headers or {})
    t0 = time.time()
    c.request('POST', path, body=json.dumps(body), headers=h)
    r = c.getresponse()
    first = r.read(1)
    t_first = time.time() - t0
    rest = r.read()
    t_total = time.time() - t0
    hdrs = {k.lower(): v for k, v in r.getheaders()}
    c.close()
    return r.status, hdrs, first + rest, t_first, t_total


log = open(sys.argv[4], 'wb') if len(sys.argv) > 4 else subprocess.DEVNULL  # optional server log path
srv = subprocess.Popen([server_exe, '-m', model, '-c', '4096', '-np', '1', '-t', '2', '--host', '127.0.0.1', '--port', str(port)],
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

    base = {'prompt': 'Once upon a time', 'temperature': 0, 'ignore_eos': True, 'seed': 1, 'cache_prompt': False}
    # 0. calibrate: >= 3 s plain
    n = 256
    while True:
        st, _, body, _, t = request('/completion', dict(base, n_predict=n))
        if st != 200:
            print('FAIL calibration request status', st, body[:200])
            sys.exit(1)
        if t >= 3.0 or n >= 3800:
            break
        n = min(3800, int(n * max(2.0, 3.5 / max(t, 0.05))))
    print(f'calibrated n_predict={n}: plain request {t:.2f} s', flush=True)
    plain = json.loads(body)

    # 1. slow request with the header
    st, h, body, t_first, t_total = request('/completion', dict(base, n_predict=n), {'X-KoboldCpp-Keepalive': '1'})
    text = body.decode('utf-8')
    lead = len(text) - len(text.lstrip())
    check(st == 200, f'keepalive status {st}')
    check(h.get('x-keepalive') == 'whitespace', f'X-Keepalive header {h.get("x-keepalive")!r}')
    check(lead >= 2048, f'leading whitespace before the JSON: {lead} bytes (total {t_total:.2f} s)')
    check(t_first < t_total - 1.0, f'first body byte at {t_first:.2f} s, body done at {t_total:.2f} s')
    try:
        obj = json.loads(text)
    except ValueError as e:
        obj = None
        check(False, f'keepalive body is not JSON: {e}')
    if obj is not None:
        check(obj.get('content') == plain.get('content') and obj.get('tokens_predicted') == plain.get('tokens_predicted'),
              f'keepalive content == plain content ({obj.get("tokens_predicted")} vs {plain.get("tokens_predicted")} tokens)')

    # 1b. slow request with "X-KoboldCpp-Keepalive: false": off (KoboldCpp acts only on "true"), the plain answer
    st, h, body, _, t = request('/completion', dict(base, n_predict=n), {'X-KoboldCpp-Keepalive': 'false'})
    check(st == 200 and body[:1] == b'{' and 'content-length' in h and 'x-keepalive' not in h,
          f'slow request with header "false": status {st}, starts {body[:1]!r}, x-keepalive {h.get("x-keepalive")} ({t:.2f} s)')

    # 2. short request with the header: unchanged
    st, h, body, _, t = request('/completion', dict(base, n_predict=4), {'X-KoboldCpp-Keepalive': '1'})
    check(st == 200 and body[:1] == b'{' and 'content-length' in h and 'x-keepalive' not in h,
          f'fast request with header: status {st}, starts {body[:1]!r}, content-length {h.get("content-length")}, x-keepalive {h.get("x-keepalive")} ({t:.2f} s)')

    # 3. fast failure with the header keeps its real status
    #    (a prompt longer than the 4096-token slot context: the slot refuses it with ERROR_TYPE_EXCEED_CONTEXT_SIZE, through the
    #    same result reader the keepalive path waits on)
    st, h, body, _, _ = request('/completion', dict(base, n_predict=4, prompt='Once upon a time ' * 2000), {'X-KoboldCpp-Keepalive': '1'})
    check(st != 200 and 'x-keepalive' not in h, f'fast failure with header: status {st} (want non-200), body {body[:120]!r}')

    # 4. OAI chat route, slow, with the header
    chat = {'messages': [{'role': 'user', 'content': 'Tell me a story.'}], 'temperature': 0, 'ignore_eos': True,
            'max_tokens': n, 'seed': 1}
    st, h, body, t_first, t_total = request('/v1/chat/completions', chat, {'X-KoboldCpp-Keepalive': '1'})
    text = body.decode('utf-8')
    lead = len(text) - len(text.lstrip())
    try:
        obj = json.loads(text)
        ok = obj.get('object') == 'chat.completion' and len(obj.get('choices', [])) == 1
    except ValueError:
        ok = False
    check(st == 200 and lead >= 2048 and ok, f'chat keepalive: status {st}, {lead} B whitespace, OAI object {ok} ({t_total:.2f} s)')
finally:
    srv.terminate()
    try:
        srv.wait(20)
    except subprocess.TimeoutExpired:
        srv.kill()

print(f'{passes} passed, {len(fails)} failed')
sys.exit(1 if fails else 0)
