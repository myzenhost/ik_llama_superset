#!/usr/bin/env python3
"""Per-request "expiring_logit_bias" field (ik_llama.cpp #1731 API) against a real llama-server on the CPU.

usage: test-server-expiring-logit-bias.py <llama-server(.exe)> <model.gguf> <port> [server log]

The field is an array of strings, one rule per element, in the --expiring-logit-bias-file format
(common/common.cpp argparse_expiring_logit_bias). A rule (D : "phrase" : B) adds B to the phrase's first token while
fewer than D tokens have been generated. The test needs a real tokenizer (BPE pieces with a leading space), which is
why it complements tools/server/tests/unit/test_expiring_logit_bias.py (stories260K).

Greedy decoding throughout, so every output is reproducible and each check is compared with a control:
  0. reference: no field; the target tokens are picked FROM the reference (a token whose piece tokenizes back to
     itself, does not occur earlier, and can be written inside the rule's quotes);
  1. ban inside the window: (N : "<piece of the first target>" : -100), N = n_predict, so the window covers the whole
     generation: the target token id is absent from every generated token, and generation_settings reports 1 stage;
  2. expiry: for a target at index k >= 1, a window of k tokens has expired when the target is due (output == reference),
     a window of k+1 has not (the token at index k changes, the k tokens before it are unchanged);
  3. no leak: the next request without the field is byte-identical to the reference (the target is back);
  4. a rule that does not parse is refused with 400, not silently dropped.
Before the field existed the server ignored it: checks 1, 2 (window k+1) and 4 FAIL.
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

PROMPT = 'The capital of France is'
N_PREDICT = 24


def check(cond, msg):
    global passes
    if cond:
        passes += 1
        print('PASS', msg, flush=True)
    else:
        fails.append(msg)
        print('FAIL', msg, flush=True)


def post(path, body, timeout=300):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    c.request('POST', path, body=json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, {'_raw': data[:300].decode('utf-8', 'replace')}


def complete(**extra):
    body = {'prompt': PROMPT, 'n_predict': N_PREDICT, 'temperature': 0, 'top_k': 1, 'seed': 1,
            'cache_prompt': False, 'return_tokens': True}
    body.update(extra)
    return post('/completion', body)


def piece_of(tok):
    st, j = post('/detokenize', {'tokens': [tok]})
    return j.get('content') if st == 200 else None


def round_trips(piece, tok):
    st, j = post('/tokenize', {'content': piece, 'add_special': False})
    return st == 200 and j.get('tokens') == [tok]


log = open(sys.argv[4], 'wb') if len(sys.argv) > 4 else subprocess.DEVNULL
srv = subprocess.Popen([server_exe, '-m', model, '-c', '2048', '-np', '1', '-t', '4', '--host', '127.0.0.1',
                        '--port', str(port)],
                       stdout=log, stderr=log, creationflags=CREATE_NO_WINDOW, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
try:
    deadline = time.time() + 180
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

    # 0. reference and targets
    st, ref = complete()
    if st != 200 or not ref.get('tokens'):
        print('FAIL reference request: status %s, body %s' % (st, str(ref)[:300]))
        sys.exit(1)
    ref_toks = ref['tokens']
    print('reference: %r' % ref['content'], flush=True)
    print('reference tokens: %s' % ref_toks, flush=True)
    cands = []
    for i, t in enumerate(ref_toks):
        if t in ref_toks[:i]:
            continue
        p = piece_of(t)
        if not p or not p.strip() or any(ch in p for ch in '"#\n\r'):
            continue
        if round_trips(p, t):
            cands.append((i, t, p))
    print('candidate targets (index, id, piece): %s' % cands, flush=True)
    if not cands or not [c for c in cands if c[0] >= 1]:
        print('FAIL no usable target token in the reference (need one at index 0.. and one at index >= 1)')
        sys.exit(1)
    a_idx, a_tok, a_piece = cands[0]
    b_idx, b_tok, b_piece = [c for c in cands if c[0] >= 1][0]

    # 1. ban inside the window
    rule = '(%d : "%s" : -100)' % (N_PREDICT, a_piece)
    st, got = complete(expiring_logit_bias=[rule])
    toks = got.get('tokens', []) if st == 200 else []
    print('ban %s -> %r' % (rule, got.get('content')), flush=True)
    check(st == 200 and len(toks) > a_idx, 'ban request status %s, %d tokens' % (st, len(toks)))
    check(a_tok not in toks, 'target %d %r (reference index %d) absent from all %d generated tokens' % (
        a_tok, a_piece, a_idx, len(toks)))
    stages = got.get('generation_settings', {}).get('expiring_logit_bias_stages')
    check(stages == 1, 'generation_settings.expiring_logit_bias_stages = %s (want 1)' % stages)

    # 2. expiry: window k has expired at index k, window k+1 has not
    rule_exp = '(%d : "%s" : -100)' % (b_idx, b_piece)
    rule_act = '(%d : "%s" : -100)' % (b_idx + 1, b_piece)
    st_e, exp = complete(expiring_logit_bias=[rule_exp])
    st_a, act = complete(expiring_logit_bias=[rule_act])
    exp_t = exp.get('tokens', []) if st_e == 200 else []
    act_t = act.get('tokens', []) if st_a == 200 else []
    print('expired %s -> %r' % (rule_exp, exp.get('content')), flush=True)
    print('active  %s -> %r' % (rule_act, act.get('content')), flush=True)
    check(exp_t == ref_toks, 'window %d on %r (due at index %d) has expired: output == reference' % (b_idx, b_piece, b_idx))
    check(act_t[:b_idx] == ref_toks[:b_idx] and len(act_t) > b_idx and act_t[b_idx] != b_tok,
          'window %d on %r is live at index %d: first %d tokens unchanged, token %d is %s (reference %d)' % (
              b_idx + 1, b_piece, b_idx, b_idx, b_idx, act_t[b_idx] if len(act_t) > b_idx else None, b_tok))

    # 3. the next request without the field is the reference again
    st, again = complete()
    check(st == 200 and again.get('tokens') == ref_toks and a_tok in again.get('tokens', []),
          'request without the field after the ban: byte-identical to the reference, target %d present' % a_tok)

    # 4. a malformed rule is refused
    st, bad = complete(expiring_logit_bias=['(x : "%s" : 5)' % a_piece])
    check(st == 400, 'malformed rule -> status %s (want 400)' % st)
    st, bad = complete(expiring_logit_bias='not a list')
    check(st == 400, 'non-array field -> status %s (want 400)' % st)
finally:
    srv.terminate()
    try:
        srv.wait(20)
    except subprocess.TimeoutExpired:
        srv.kill()

print(f'{passes} passed, {len(fails)} failed')
sys.exit(1 if fails else 0)
