#!/usr/bin/env python3
"""Tensor hot-swap in the built llama-server (LLAMA_HOTSWAP_*), on the CPU.

usage: test-server-hotswap.py <llama-server(.exe)> <small F32 model.gguf> <port> <workdir>

Copies the model to <workdir>/hs.gguf, starts the server on it (--no-mmap: a same-dtype swap is refreshed in place) with
LLAMA_HOTSWAP_ENABLED + CONTROL / STATUS files, and drives the protocol of docs/development/on-demand-tensor-reload.md:
   1. status "done 1 0" once the model is loaded;
   2. "reload 1" with nothing changed -> "reload_failed 1 1", the control file consumed, the answer unchanged;
   3. blk.0.attn_norm.weight scaled by 0.5 on disk (feeds layer-0 K/V), "reload 2" -> "done 2 2", "reloaded tensor" in
      the log; the answer changes, and a cache_prompt request equals a cache_prompt=false one (no KV or cached prompt
      from the old weights is reused);
   4. the original file put back, GET /health (the /health hook) -> "hotswap": {"reloaded": true}; the answer is the
      original one again, exactly;
   5. "compute 3" -> "done 3 3"; a stale "compute 2" is ignored; "exit 4" -> "exiting 3 4" and the process ends;
   6. LLAMA_HOTSWAP_CONTROL_FILE without LLAMA_HOTSWAP_STATUS_FILE is refused at start (non-zero exit).
Before the feature the server ignores every LLAMA_HOTSWAP_* variable: no status file, no reload, no /health hook.
"""
import http.client
import json
import os
import shutil
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_raw_util as gr  # noqa: E402

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
server_exe, model, port, work = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(work, exist_ok=True)
fails, passes = [], 0


def check(cond, msg):
    global passes
    if cond:
        passes += 1
        print('PASS', msg, flush=True)
    else:
        fails.append(msg)
        print('FAIL', msg, flush=True)


def call(method, path, body=None, timeout=120):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    c.request(method, path, body=None if body is None else json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def complete(cache_prompt=True):
    st, data = call('POST', '/completion', {'prompt': 'Once upon a time, there was a little', 'n_predict': 16,
                                           'temperature': 0, 'n_probs': 3, 'cache_prompt': cache_prompt, 'seed': 1})
    if st != 200:
        return ('HTTP', st, data[:200])
    j = json.loads(data)
    return (j.get('content'), json.dumps(j.get('completion_probabilities'), sort_keys=True))


ctrl = os.path.join(work, 'hs.ctrl')
stat = os.path.join(work, 'hs.status')
hs = os.path.join(work, 'hs.gguf')
orig = os.path.join(work, 'hs.orig.gguf')
log_path = os.path.join(work, 'hs_server.log')


def unlink_own(p):
    # SAFE-DELETE-OK: only this test's own protocol files (control / status) inside its own workdir
    if os.path.exists(p):
        os.unlink(p)


for p in (ctrl, stat, stat + '.tmp'):
    unlink_own(p)
shutil.copyfile(model, orig)
shutil.copyfile(model, hs)


def read_status():
    try:
        with open(stat, 'rb') as f:
            return f.read().decode('utf-8').strip()
    except OSError:
        return None


def wait_status(want, seconds=10.0):
    t_end = time.time() + seconds
    s = None
    while time.time() < t_end:
        s = read_status()
        if s == want:
            return s
        time.sleep(0.1)
    return s


def send(cmd):
    with open(ctrl + '.w', 'wb') as f:
        f.write((cmd + '\n').encode())
    os.replace(ctrl + '.w', ctrl)  # atomic, as the document recommends


def bump_mtime(path, plus):
    # the reload detects a change by mtime (whole seconds on Windows): move it clearly forward
    t = time.time() + plus
    os.utime(path, (t, t))


def log_text():
    with open(log_path, 'rb') as f:
        return f.read().decode('utf-8', 'replace')


env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', LLAMA_HOTSWAP_ENABLED='1', LLAMA_HOTSWAP_CONTROL_FILE=ctrl,
           LLAMA_HOTSWAP_STATUS_FILE=stat)
args = [server_exe, '-m', hs, '--no-mmap', '-c', '512', '-np', '1', '-t', '2', '--host', '127.0.0.1', '--port', str(port)]
log = open(log_path, 'wb')
srv = subprocess.Popen(args, stdout=log, stderr=log, creationflags=CREATE_NO_WINDOW, env=env)
try:
    deadline = time.time() + 120
    while True:
        try:
            if call('GET', '/health', timeout=2)[0] == 200:
                break
        except OSError:
            pass
        if time.time() > deadline or srv.poll() is not None:
            print('FAIL server did not come up (rc=%s)' % srv.poll())
            sys.exit(1)
        time.sleep(0.5)

    # 1
    s = wait_status('done 1 0')
    check(s == 'done 1 0', f'status after load: {s!r} (want "done 1 0")')
    a = complete()
    check(a[0] != 'HTTP', f'baseline completion: {a[:1]!r}')

    # 2
    send('reload 1')
    s = wait_status('reload_failed 1 1')
    check(s == 'reload_failed 1 1', f'"reload 1", nothing changed: status {s!r}')
    check(not os.path.exists(ctrl), 'control file consumed')
    check(complete() == a, 'answer unchanged after a reload that changed nothing')

    # 3
    g = gr.load(orig)
    t = next(x for x in g.tensors if x['name'] == 'blk.0.attn_norm.weight')
    check(t['type'] == 0, f'blk.0.attn_norm.weight is F32 (type {t["type"]})')
    n = len(t['data']) // 4
    vals = struct.unpack(f'<{n}f', t['data'])
    t['data'] = struct.pack(f'<{n}f', *[v * 0.5 for v in vals])
    gr.dump(g, hs)
    bump_mtime(hs, 10)
    n_reloaded_before = log_text().count("reloaded tensor 'blk.0.attn_norm.weight'")
    send('reload 2')
    s = wait_status('done 2 2', 60)
    check(s == 'done 2 2', f'"reload 2" after the on-disk change: status {s!r}')
    lt = log_text()
    check(lt.count("reloaded tensor 'blk.0.attn_norm.weight'") == n_reloaded_before + 1,
          '"reloaded tensor \'blk.0.attn_norm.weight\'" in the log')
    check('hotswap: tensors reloaded; KV cache and cached prompts cleared' in lt, 'log: KV cache and cached prompts cleared')
    c1 = complete(cache_prompt=True)
    c2 = complete(cache_prompt=False)
    check(c1[0] != 'HTTP' and c1 != a, 'answer changed with the changed weights')
    check(c1 == c2, 'cache_prompt answer == cache_prompt=false answer (no stale KV / cached prompt reused)')

    # 4
    shutil.copyfile(orig, hs)
    bump_mtime(hs, 20)
    st, body = call('GET', '/health')
    try:
        hj = json.loads(body).get('hotswap', {})
    except ValueError:
        hj = {}
    check(st == 200 and hj.get('reloaded') is True, f'/health hook reloaded the restored file: {st} {body[:200]!r}')
    d = complete()
    check(d == a, 'restored weights give the original answer exactly')
    st, body = call('GET', '/health')
    try:
        hj = json.loads(body).get('hotswap', {})
    except ValueError:
        hj = {}
    check(st == 200 and hj.get('reloaded') is False, f'/health with nothing changed: reloaded false ({body[:200]!r})')

    # 5
    send('compute 3')
    s = wait_status('done 3 3')
    check(s == 'done 3 3', f'"compute 3": status {s!r}')
    send('compute 2')
    time.sleep(1.5)
    check(read_status() == 'done 3 3' and not os.path.exists(ctrl), f'stale "compute 2" ignored (status {read_status()!r})')
    send('exit 4')
    s = wait_status('exiting 3 4')
    check(s == 'exiting 3 4', f'"exit 4": status {s!r}')
    try:
        rc = srv.wait(60)
    except subprocess.TimeoutExpired:
        rc = None
    check(rc is not None, f'server ended after "exit" (rc {rc})')
finally:
    if srv.poll() is None:
        srv.terminate()
        try:
            srv.wait(20)
        except subprocess.TimeoutExpired:
            srv.kill()
    log.close()

# 6
env2 = dict(env)
del env2['LLAMA_HOTSWAP_STATUS_FILE']
log2_path = os.path.join(work, 'hs_server_nostatus.log')
with open(log2_path, 'wb') as log2:
    p2 = subprocess.Popen(args[:-1] + [str(port + 1)], stdout=log2, stderr=log2, creationflags=CREATE_NO_WINDOW, env=env2)
    try:
        rc2 = p2.wait(60)
    except subprocess.TimeoutExpired:
        p2.kill()
        rc2 = None
with open(log2_path, 'rb') as f:
    t2 = f.read().decode('utf-8', 'replace')
check(rc2 not in (None, 0) and 'requires LLAMA_HOTSWAP_STATUS_FILE' in t2, f'CONTROL without STATUS refused at start (rc {rc2})')

print(f'{passes} passed, {len(fails)} failed')
sys.exit(1 if fails else 0)
