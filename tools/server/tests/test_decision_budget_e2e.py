"""End-to-end test of --decision-budget (auto / model / N) of /v1/systemone on a laya decision model (Julia-1 GGUF).
Starts its own llama-server on a free port. No pytest needed:
    python -I tools/server/tests/test_decision_budget_e2e.py <llama-server> <Julia-1.gguf> [port]
Pass criteria (exit 0, "DECISION-BUDGET E2E PASS"):
  1 short case: auto == model byte for byte in "answers"; usage has no windows / truncated
  2 long head (12 options x ~100-token descriptions): model reports usage.truncated + truncated_tokens, auto does not, and
    auto input_tokens == model input_tokens + truncated_tokens (nothing cut, the full prompt was read)
  3 tiny context (-c 512, one pass takes at most 512 tokens): a state of ~2500 tokens is read in usage.windows > 1, nothing truncated, each
    window carries the head (input_tokens > the state alone)
  6 each aggregation (max, mean, lse, maxconf, medsum) of the server == the same aggregate computed in the test from the raw per-window logits
  4 state above the trained 8192 positions (default context): usage.windows > 1, no truncated, window_aggregation reported;
    --decision-window-agg first says truncated
  5 a head that alone does not fit the context is an error that names the rope flags, not a silent cut
SCOUT-NA: test of the feature in this commit."""
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request

exe, gguf = sys.argv[1], sys.argv[2]
port = int(sys.argv[3]) if len(sys.argv) > 3 else 19110

FILLER = [
    'The meeting notes from the quarterly planning session were circulated to the whole department on Monday morning.',
    'Please remember to switch off the lights in the conference room when you leave the building in the evening.',
    'The cafeteria menu this week includes vegetable soup, roasted potatoes, a green salad and fresh fruit every day.',
    'A reminder that the parking garage will be closed for cleaning on the second weekend of the month.',
    'The annual calendar of public holidays has been updated and is available on the internal website.',
    'The recycling bins are emptied twice a week, please sort paper, plastic and glass into the correct containers.',
]
NEEDLE = 'The mobile app crashes with an error every time I try to log in after the latest update.'
CRIT3 = {'billing': 'Only duplicate charges, invoices and refunds of payments.',
         'shipping': 'Only late, lost or damaged parcels and courier tracking.',
         'technical': 'Only crashes, error messages and login problems of the app.'}
BOILER = ('This queue is staffed by trained agents who work through tickets in the order they arrive, follow the written escalation guide, '
          'answer in the language of the customer, keep a record of every contact and hand over to a senior colleague when a case is not solved on the first reply. ')
TOPICS = ['billing', 'shipping', 'technical', 'returns', 'accounts', 'security', 'sales', 'legal', 'press', 'hr', 'partnerships', 'feedback']
CRIT12 = {t: BOILER + 'Only ' + t + ' matters of customers.' for t in TOPICS}


def filler(n_sentences):
    return ' '.join(FILLER[i % len(FILLER)] for i in range(n_sentences))


def post(base, body):
    req = urllib.request.Request(base + '/v1/systemone', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=1800).read())
    except urllib.error.HTTPError as e:
        return {'error': json.loads(e.read())}


class Server(object):
    def __init__(self, extra):
        self.log = open('_decision_budget_e2e_%d.log' % port, 'w')
        self.p = subprocess.Popen([exe, '-m', gguf, '--port', str(port), '--host', '127.0.0.1', '-ngl', '0', '-np', '1', '--no-webui', '-t', '16'] + extra,
                                  stdout=self.log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.base = 'http://127.0.0.1:%d' % port
        for _ in range(180):
            try:
                if urllib.request.urlopen(self.base + '/health', timeout=2).status == 200:
                    return
            except Exception:
                time.sleep(1)
        self.stop()
        raise RuntimeError('server did not start')

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(20)
        except Exception:
            self.p.kill()
        self.log.close()


fails = []


def check(name, ok, info=''):
    print('%-4s %s %s' % ('ok' if ok else 'FAIL', name, info), flush=True)
    if not ok:
        fails.append(name)


Q3 = {'route': {'type': 'choice', 'instructions': 'Which team should handle this request?', 'criteria': CRIT3}}
Q12 = {'route': {'type': 'choice', 'instructions': 'Which queue should handle this request?', 'criteria': CRIT12}}

s = Server([])
try:
    # 1
    a = post(s.base, {'state': NEEDLE, 'questions': Q3, 'decision_budget': 'auto'})
    m = post(s.base, {'state': NEEDLE, 'questions': Q3, 'decision_budget': 'model'})
    check('1 short: auto == model byte for byte', json.dumps(a['answers'], sort_keys=True) == json.dumps(m['answers'], sort_keys=True) and a['usage'] == m['usage'], str(a['usage']))
    check('1 short: no windows / truncated', all(k not in a['usage'] for k in ('windows', 'truncated')))
    # 2
    m = post(s.base, {'state': NEEDLE, 'questions': Q12, 'decision_budget': 'model'})
    a = post(s.base, {'state': NEEDLE, 'questions': Q12, 'decision_budget': 'auto'})
    check('2 long head: model reports the cut', m['usage'].get('truncated') is True and m['usage'].get('truncated_tokens', 0) > 0, str(m['usage']))
    check('2 long head: auto does not cut', 'truncated' not in a['usage'], str(a['usage']))
    check('2 long head: auto read the whole prompt (model tokens + cut tokens)', a['usage']['input_tokens'] == m['usage']['input_tokens'] + m['usage'].get('truncated_tokens', 0),
          '%s vs %s + %s' % (a['usage']['input_tokens'], m['usage']['input_tokens'], m['usage'].get('truncated_tokens')))
    # 4
    big = filler(520) + ' ' + NEEDLE
    w = post(s.base, {'state': big, 'questions': Q3})
    u = w['usage']
    check('4 above 8192: windows > 1, nothing truncated', u.get('windows', 1) > 1 and 'truncated' not in u, str(u))
    check('4 above 8192: aggregation reported', u.get('window_aggregation') == 'max', str(u))
    f = post(s.base, {'state': big, 'questions': Q3, 'decision_window_agg': 'first'})
    check('4 first: says truncated', f['usage'].get('truncated') is True, str(f['usage']))
    # 6 the aggregations of the server == the same aggregations computed here from the raw logits of the windows (argmax of the aggregate = the choice)
    raw = post(s.base, {'state': big, 'questions': Q3, 'decision_window_logits': True})
    wl = raw['usage']['window_logits']['route'][0]
    keys = list(CRIT3)

    def lp(w):
        top = max(w)
        z = top + math.log(sum(math.exp(x - top) for x in w))
        return [x - z for x in w]

    def aggregate(ws, how):
        n = len(ws[0])
        if how == 'first':
            return ws[0]
        if how == 'mean':
            return [sum(w[i] for w in ws) / len(ws) for i in range(n)]
        if how == 'lse':
            return [max(w[i] for w in ws) + math.log(sum(math.exp(w[i] - max(x[i] for x in ws)) for w in ws)) for i in range(n)]
        if how == 'maxconf':
            return max((lp(w) for w in ws), key=max)
        if how == 'medsum':
            lps = [lp(w) for w in ws]
            med = [sorted(l[i] for l in lps)[len(lps) // 2] for i in range(n)]
            return [sum(l[i] - med[i] for l in lps) for i in range(n)]
        return [max(w[i] for w in ws) for i in range(n)]
    for how in ('max', 'mean', 'lse', 'maxconf', 'medsum'):
        r = post(s.base, {'state': big, 'questions': Q3, 'decision_window_agg': how})
        agg = aggregate(wl, how)
        want = keys[max(range(len(keys)), key=lambda i: agg[i])]
        check('6 aggregation %s: server choice == offline aggregate of the raw window logits' % how, r['answers']['route']['choice'] == want,
              '%s vs %s, %d windows' % (r['answers']['route']['choice'], want, len(wl)))
    # 5
    e = post(s.base, {'state': NEEDLE, 'questions': {'q': {'type': 'choice', 'instructions': 'x ' * 4, 'criteria': {t: BOILER * 90 for t in TOPICS}}}})
    check('5 head larger than the context: error names rope scaling', 'error' in e and 'rope' in json.dumps(e['error']), json.dumps(e)[:200])
finally:
    s.stop()

s = Server(['-c', '512'])
try:
    # 3
    state = filler(120) + ' ' + NEEDLE
    r = post(s.base, {'state': state, 'questions': Q3})
    if 'error' in r:
        check('3 tiny context: request served', False, json.dumps(r)[:300])
    else:
        u = r['usage']
        check('3 tiny context: windows > 1, nothing truncated', u.get('windows', 1) > 1 and 'truncated' not in u, str(u))
        check('3 tiny context: each window carries the head', u['input_tokens'] > 2400, str(u))
        # the answer is printed, not asserted: with windows this small the filler windows vote too (measured, see the findings)
        print('info 3 tiny context answer:', json.dumps(r['answers']['route']['probabilities']), flush=True)
finally:
    s.stop()

print('DECISION-BUDGET E2E %s' % ('PASS' if not fails else 'FAIL: ' + ', '.join(fails)))
sys.exit(1 if fails else 0)
