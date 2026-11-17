#!/usr/bin/env python3
"""/v1/audio/transcriptions with NO prompt field returns the transcript, not a conversational refusal.

usage: test-asr-default-prompt.py <llama-server> <audio model GGUF> <mmproj GGUF> <speech WAV> <reference .txt> <port> [extra server args...]

The default ASR prompt comes from common_chat_get_asr_prompt. With upstream's "Transcribe audio to text" an
instruction-tuned audio model (Voxtral-Mini-3B-2507) answers "I'm unable to transcribe audio directly ..." with HTTP 200.

Checks (one request, response_format json, temperature 0, no prompt):
  1. HTTP 200 with a non-empty text;
  2. the text is not a refusal ("unable to transcribe" / "can't transcribe" / "cannot transcribe" absent);
  3. at least 80% of the reference words appear in the text (case- and punctuation-insensitive bag of words).
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

server, model, mmproj, speech_wav, ref_txt, port = sys.argv[1:7]
extra = sys.argv[7:]
BASE = f'http://127.0.0.1:{port}'


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def post(audio):
    b = uuid.uuid4().hex
    parts = [
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\nx\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + audio + b'\r\n',
        f'--{b}--\r\n'.encode(),
    ]
    req = urllib.request.Request(BASE + '/v1/audio/transcriptions', data=b''.join(parts), method='POST',
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read().decode('utf-8', 'replace'))
    except urllib.error.HTTPError as e:
        return e.code, {'error': e.read().decode('utf-8', 'replace')[:500]}


failed = 0


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


ref = words(open(ref_txt, encoding='utf-8').read())
log = open(f'_asr_default_prompt_server_{port}.log', 'wb')  # in the caller's cwd
flags = 0x08000000 if os.name == 'nt' else 0
srv = subprocess.Popen([server, '-m', model, '--mmproj', mmproj, '--port', port, '-c', '4096', '--no-webui', *extra],
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
    check('server up', up, f'rc {srv.poll()}')
    if not up:
        sys.exit(failed + 3)
    st, js = post(open(speech_wav, 'rb').read())
    text = js.get('text') or ''
    print('text:', repr(text[:300]), flush=True)
    check('200 with text', st == 200 and bool(text.strip()), f'status {st}')
    low = text.lower()
    check('not a refusal', not any(p in low for p in ('unable to transcribe', "can't transcribe", 'cannot transcribe')))
    got = set(words(text))
    hit = sum(1 for w in ref if w in got)
    check('>= 80% of reference words present', bool(ref) and hit >= 0.8 * len(ref), f'{hit}/{len(ref)} ({" ".join(ref)})')
finally:
    srv.terminate()
    try:
        srv.wait(timeout=30)
    except subprocess.TimeoutExpired:
        srv.kill()
print(f'{4 - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
