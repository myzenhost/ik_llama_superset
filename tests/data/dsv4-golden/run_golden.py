"""LIMITS (measured 2026-10-04): --common is the real server path and now honours the case's thinking / reasoning_effort (test-chat-template.cpp format_using_common maps them to enable_thinking + chat_template_kwargs): 15/15 byte-identical on a build from that source. Needs an exe built AFTER that change, else chat-mode/effort cases DIFFER by harness limit (9/15). --no-common lacks the from_json filter, so every string-argument tool-call case errors (rc 3); use --common.
Run tests/data/dsv4-golden/golden.json through the fork's own jinja engine
(prebuilt build-static/bin/test-chat-template.exe, --no-common = raw minja) and
byte-compare with the OFFICIAL-encoder expected strings. Read-only measurement."""
# PROGRESS-OK: each render is one sub-second child process (15 cases, 120 s cap each); no long blind wait
# REWRITE-SAFE: reads golden.json only; writes only NEW files under temp/ (result json + per-case tmp dir)
import json, os, subprocess, sys, tempfile

FORK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
EXE = FORK + '/build-static/bin/test-chat-template.exe'
TPL = FORK + '/models/templates/deepseek-ai-DeepSeek-V4-Flash-0731.jinja'
GOLD = FORK + '/tests/data/dsv4-golden/golden.json'
BOS = '<\uff5cbegin\u2581of\u2581sentence\uff5c>'
mode = sys.argv[1] if len(sys.argv) > 1 else '--no-common'
only = sys.argv[2] if len(sys.argv) > 2 else None

# STALE-EXE GUARD: a prebuilt exe older than the template-handling sources reports defects
# the source already fixed (2026-10-04: golden case 14 rc=3 came from a Sep-26 exe).
_SRC = [FORK + '/common/chat.cpp', FORK + '/common/jinja/value.cpp', TPL]
_stale = [s for s in _SRC if os.path.exists(s) and os.path.getmtime(s) > os.path.getmtime(EXE)]
if _stale:
    print('WARNING: exe older than %s -- rebuild test-chat-template before trusting DIFF/ERROR rows' % ', '.join(os.path.basename(s) for s in _stale), file=sys.stderr)
    if os.environ.get('DSV4_GOLDEN_STRICT'):
        sys.exit(4)

gold = json.load(open(GOLD, encoding='utf8'))
res = []
tmp = tempfile.mkdtemp(prefix='dsv4tct_1004_')
for i, c in enumerate(gold['cases']):
    if only is not None and str(i) != only:
        continue
    ctx = {'messages': c['messages'], 'bos_token': BOS, 'eos_token': '<\uff5cend\u2581of\u2581sentence\uff5c>',
           'add_generation_prompt': True, 'thinking': bool(c['thinking'])}
    if c.get('tools'):
        ctx['tools'] = c['tools']
    if c.get('reasoning_effort'):
        ctx['reasoning_effort'] = c['reasoning_effort']
    jp = os.path.join(tmp, f'c{i}.json'); op = os.path.join(tmp, f'c{i}.out')
    json.dump(ctx, open(jp, 'w', encoding='utf8'), ensure_ascii=False)
    args = [EXE, TPL, '--json', jp, '--output', op]
    if mode == '--no-common':
        args.insert(2, '--no-common')
    p = subprocess.run(args, capture_output=True, creationflags=0x08000000, timeout=120)
    got = open(op, encoding='utf8', newline='').read().replace('\r\n', '\n') if os.path.exists(op) else None
    exp = c['expected']
    res.append({'i': i, 'name': c['name'], 'rc': p.returncode, 'match': got == exp,
                'got_len': None if got is None else len(got), 'exp_len': len(exp),
                'err': (p.stderr or b'').decode('utf8', 'replace')[-300:]})
    if got is not None and got != exp:
        k = next((j for j in range(min(len(got), len(exp))) if got[j] != exp[j]), min(len(got), len(exp)))
        res[-1]['first_diff'] = k
        res[-1]['got_ctx'] = got[max(0, k - 40):k + 60]
        res[-1]['exp_ctx'] = exp[max(0, k - 40):k + 60]
out = os.path.join(tmp, 'result_%s.json') % ('raw' if mode == '--no-common' else 'common')
json.dump(res, open(out, 'w', encoding='utf8'), ensure_ascii=False, indent=1)
print('cases', len(res), 'match', sum(r['match'] for r in res), 'nonzero_rc', sum(1 for r in res if r['rc']))
for r in res:
    print(r['i'], 'OK ' if r['match'] else 'DIFF', r['name'], '' if r['match'] else ('rc=%s got_len=%s exp_len=%s' % (r['rc'], r['got_len'], r['exp_len'])))
