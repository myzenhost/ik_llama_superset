#!/usr/bin/env python3
"""/v1/audio/transcriptions always sends a system message, so a chat template's default persona cannot ride along.

usage: test-asr-system-prompt.py <llama-server> <audio model GGUF> <mmproj GGUF> <speech WAV> <reference .txt> <long clip> <port> [extra server args...]

Port check for fork commit ec6520161 (2026-09-02), whose fix lived in examples/server and was lost when the built server
became tools/server. Voxtral-Mini-3B-2507's GGUF carries an Unsloth Devstral template whose default_system_message (a
~1.2k-token coding persona) is substituted whenever the request has no system message. ec6520161 measured 1429 prompt
tokens without a system message vs 209 with one, a conversational reply register, and a long clip whose opening was dropped.

Checks (temperature 0 via the route default, no prompt field):
  1. short clip: HTTP 200 with a non-empty text;
  2. short clip: usage.input_tokens < 700 (the persona alone is ~1.2k tokens);
  3. short clip: at least 80% of the reference words appear (bag of words);
  4. long clip (test-2.mp3, 17.4 s): the OPENING sentence ("...New York Times ... 1969") is in the transcript, i.e.
     before the word "newsprint" (the paper's name recurs later in the clip, so a whole-text search cannot see a drop);
  5. long clip: the transcript is not wrapped in a balanced pair of quotes;
  6. --log-file (ik #2313, carried by merge 6f68667f5): the log file holds a llama model-load line AND a server line.
Exit code = number of failed checks.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import uuid

server, model, mmproj, speech_wav, ref_txt, long_clip, port = sys.argv[1:8]
extra = sys.argv[8:]
BASE = f'http://127.0.0.1:{port}'


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())  # digits too: the opening check looks for "1969"


def post(audio, filename, ctype):
    b = uuid.uuid4().hex
    parts = [
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\nx\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'.encode() + audio + b'\r\n',
        f'--{b}--\r\n'.encode(),
    ]
    req = urllib.request.Request(BASE + '/v1/audio/transcriptions', data=b''.join(parts), method='POST',
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read().decode('utf-8', 'replace'))
    except urllib.error.HTTPError as e:
        return e.code, {'error': e.read().decode('utf-8', 'replace')[:500]}


failed = 0
N_CHECKS = 6


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


ref = words(open(ref_txt, encoding='utf-8').read())
log_file = os.path.abspath(f'_asr_sysprompt_logfile_{port}.log')  # in the caller's cwd
if os.path.exists(log_file):
    os.replace(log_file, log_file + '.prev')
log = open(f'_asr_sysprompt_server_{port}.log', 'wb')
flags = 0x08000000 if os.name == 'nt' else 0
srv = subprocess.Popen([server, '-m', model, '--mmproj', mmproj, '--port', port, '-c', '4096', '--no-webui',
                        '--log-file', log_file, *extra],
                       stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
try:
    up = False
    for _ in range(600):
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=2) as r:
                if r.status == 200:
                    up = True
                    break
        except Exception:  # noqa: BLE001
            pass
        if srv.poll() is not None:
            break
        time.sleep(1)
    print(('PASS' if up else 'FAIL') + f' server up -- rc {srv.poll()}', flush=True)
    if not up:
        sys.exit(N_CHECKS)

    st, js = post(open(speech_wav, 'rb').read(), 'speech.wav', 'audio/wav')
    text = js.get('text') or ''
    ptok = (js.get('usage') or {}).get('input_tokens', -1)
    print('short text:', repr(text[:400]), 'input_tokens:', ptok, flush=True)
    check('short: 200 with text', st == 200 and bool(text.strip()), f'status {st}')
    check('short: input_tokens < 700 (no template default persona)', 0 < ptok < 700, f'input_tokens {ptok}')
    got = set(words(text))
    hit = sum(1 for w in ref if w in got)
    check('short: >= 80% of reference words present', bool(ref) and hit >= 0.8 * len(ref), f'{hit}/{len(ref)}')

    st2, js2 = post(open(long_clip, 'rb').read(), os.path.basename(long_clip), 'audio/mpeg')
    text2 = js2.get('text') or ''
    print('long text:', repr(text2[:600]), 'input_tokens:', (js2.get('usage') or {}).get('input_tokens', -1), flush=True)
    # The clip's OPENING sentence names the paper and the date; "New York Times" recurs later ("the New York Times
    # documented..."), so a whole-text search passes even when the opening is dropped. Look only before "newsprint",
    # the first word of the second sentence.
    lw = words(text2)
    head = ' '.join(lw[:lw.index('newsprint')] if 'newsprint' in lw else lw)
    check('long: opening "new york times ... 1969" present (before "newsprint")',
          st2 == 200 and 'new york times' in head and ('1969' in head or 'sixty' in head), f'status {st2}, head {head[:80]!r}')
    t2 = text2.strip()
    wrapped = len(t2) >= 2 and t2[0] in '"\'' and t2[-1] == t2[0] and t2.find(t2[0], 1) == len(t2) - 1
    check('long: not wrapped in a balanced quote pair', st2 == 200 and bool(t2) and not wrapped, repr(t2[:1] + '...' + t2[-1:]) if t2 else 'empty')
finally:
    srv.terminate()
    try:
        srv.wait(timeout=30)
    except subprocess.TimeoutExpired:
        srv.kill()

lf = open(log_file, encoding='utf-8', errors='replace').read() if os.path.exists(log_file) else ''
has_model = ('llama_model_loader' in lf) or ('load_tensors' in lf) or ('print_info' in lf)
has_srv = ('srv ' in lf) or ('main: server is listening' in lf) or ('server is listening' in lf)
check('--log-file captures llama load + server lines', has_model and has_srv,
      f'{len(lf)} bytes, model-load line {has_model}, server line {has_srv}')
print(f'{N_CHECKS - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
