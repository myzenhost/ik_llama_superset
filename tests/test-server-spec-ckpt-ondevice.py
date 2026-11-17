#!/usr/bin/env python3
"""llama-server: the hybrid target's speculative checkpoint stays on the device (llama_spec_ckpt_*).

usage: test-server-spec-ckpt-ondevice.py <llama-server> <hybrid model with an MTP head .gguf> <port> <workdir> [extra server args...]

Arms, each a fresh server, greedy, the same prompt:
  nospec  no speculation (reference text, reported only)
  host    --spec-type mtp:n_max=3 --spec-ckpt-mode cpu  (the host copy, common_prompt_checkpoint)
  step    ... --spec-ckpt-mode per-step                 (on-device, restore in place, no replay)
  gpufb   ... --spec-ckpt-mode gpu-fallback             (on-device shadow, replay of the accepted tokens)
  np2     ... -np 2 (auto)                              (two slots: must keep the host copy)

Checks: each on-device arm says so at startup and actually restored at least once (LLAMA_TRACE=1 round lines), the host
arm restored at least once (so the rollback path is exercised at all), gpu-fallback's text equals the host copy's exactly
(both restore the same bytes and replay), per-step's text equals the host copy's, and np2 reports the host copy.
Row S69-HYPERQWEN-CHALLENGE-SPEC-CKPT-HOST-COPY-2026-10-18-01.
"""
import http.client
import json
import os
import subprocess
import sys
import time

server_exe, model, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
work = os.path.join(sys.argv[4], time.strftime('run-%Y%m%d_%H%M%S'))
extra = sys.argv[5:]
os.makedirs(work, exist_ok=True)

PROMPT = ('List the numbers from one to forty in words, one per line, then list the days of the week twice, '
          'then the months of the year.\n\none\ntwo\nthree\n')
N_PREDICT = 160
SPEC = ['--spec-type', 'mtp:n_max=3']

fails, passes = [], 0


def check(cond, msg):
    global passes
    if cond:
        passes += 1
        print('PASS', msg, flush=True)
    else:
        fails.append(msg)
        print('FAIL', msg, flush=True)


def call(method, path, body=None, timeout=600):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    c.request(method, path, body=None if body is None else json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def run_arm(tag, args):
    log = os.path.join(work, f'server_{tag}.log')
    env = dict(os.environ, LLAMA_TRACE='1')
    argv = [server_exe, '-m', model, '--port', str(port), '--host', '127.0.0.1', '-c', '2048', '--no-webui'] + extra + args
    with open(log, 'w', encoding='utf-8') as lf:
        p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, env=env,
                             creationflags=0x08000000 if os.name == 'nt' else 0)
        text, err = None, None
        try:
            up = False
            for _ in range(600):  # PROGRESS-OK: bounded health poll (300 s), reported on timeout
                if p.poll() is not None:
                    break
                try:
                    if call('GET', '/health', timeout=2)[0] == 200:
                        up = True
                        break
                except OSError:
                    pass
                time.sleep(0.5)
            if not up:
                err = f'server {tag} not up (exit {p.poll()})'
            else:
                st, data = call('POST', '/completion', {'prompt': PROMPT, 'n_predict': N_PREDICT, 'temperature': 0.0,
                                                        'top_k': 1, 'cache_prompt': False, 'seed': 1})
                if st != 200:
                    err = f'completion {tag} http {st}: {data[:300]!r}'
                else:
                    j = json.loads(data)
                    text = j.get('content')
                    t = j.get('timings', {})
                    print(f'  {tag}: {len(text or "")} chars, predicted {t.get("predicted_n")} at '
                          f'{t.get("predicted_per_second")} t/s, draft_n {t.get("draft_n")} accepted {t.get("draft_n_accepted")}',
                          flush=True)
        finally:
            p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
    with open(log, encoding='utf-8', errors='replace') as f:
        lt = f.read()
    with open(os.path.join(work, f'text_{tag}.txt'), 'w', encoding='utf-8') as f:
        f.write(text or '')
    return text, lt, err


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


print('workdir', work, flush=True)
res = {}
for tag, args in [('nospec', []),
                  ('host', SPEC + ['--spec-ckpt-mode', 'cpu', '-np', '1']),
                  ('step', SPEC + ['--spec-ckpt-mode', 'per-step', '-np', '1']),
                  ('gpufb', SPEC + ['--spec-ckpt-mode', 'gpu-fallback', '-np', '1']),
                  ('np2', SPEC + ['-np', '2'])]:
    print(f'== {tag}', flush=True)
    text, lt, err = run_arm(tag, args)
    check(err is None and text, f'{tag}: server up and completed' + (f' ({err})' if err else ''))
    res[tag] = (text or '', lt)

host_t, host_l = res['host']
step_t, step_l = res['step']
gpufb_t, gpufb_l = res['gpufb']
np2_t, np2_l = res['np2']

check('speculative checkpoint: host copy (--spec-ckpt-mode cpu)' in host_l, 'host: startup names the host copy')
n_host = host_l.count('(restore checkpoint)')
check(n_host >= 1, f'host: the rollback path ran ({n_host} restore rounds; positive control)')

check('speculative checkpoint: on-device (per-step' in step_l, 'step: startup names the on-device per-step checkpoint')
n_step = step_l.count('(on-device checkpoint, direct)')
check(n_step >= 1, f'step: restored in place on the device ({n_step} rounds)')
check('(restore checkpoint)' not in step_l, 'step: no host restore round')

check('speculative checkpoint: on-device (gpu-fallback' in gpufb_l, 'gpufb: startup names the on-device gpu-fallback checkpoint')
n_gpufb = gpufb_l.count('on-device, replay')
check(n_gpufb >= 1, f'gpufb: restored from the device shadow and replayed ({n_gpufb} rounds)')

check('speculative checkpoint: host copy (n_slots = 2' in np2_l, 'np2: two slots keep the host copy')

check(gpufb_t == host_t, f'gpufb text == host text (common prefix {common_prefix(gpufb_t, host_t)} of {len(host_t)} chars)')
check(step_t == host_t, f'step text == host text (common prefix {common_prefix(step_t, host_t)} of {len(host_t)} chars)')
nospec_t = res['nospec'][0]
print(f'  reference only: host vs nospec common prefix {common_prefix(host_t, nospec_t)} of {len(nospec_t)} chars', flush=True)

print(f'{passes} passed, {len(fails)} failed', flush=True)
for f_ in fails:
    print('  FAILED:', f_, flush=True)
sys.exit(1 if fails else 0)
