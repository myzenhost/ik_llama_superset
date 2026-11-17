#!/usr/bin/env python3
# REWRITE-SAFE: writes only per-run server logs (new files) in the caller's cwd; reads no store.
"""/v1/audio/transcriptions --asr-native-prompt on: the model's own transcription prompt when the vocabulary has [TRANSCRIBE].

usage: test-asr-native-prompt.py <llama-server> <Voxtral GGUF> <mmproj GGUF> <speech WAV> <reference .txt> <port> [extra server args...]

Voxtral's transcription request is [INST][BEGIN_AUDIO]<audio>[/INST]lang:<code>[TRANSCRIBE] (mistral-common), not a chat turn.
The switch is OPT-IN (default off). MEASURED 2026-10-17 (temp/arex1014/_voxtx_1017/summary.json): the native prompt gives the
bare transcript on speech, but on silence it writes invented text, so silence is NOT tested as empty here -- that belongs to
voice-activity detection, not the prompt.

Three servers, same model:
  A. --asr-native-prompt on:
     1. server up;
     2. speech clip, no prompt: >= 80% of the reference words, not a refusal;
     3. the transcript is bare: no leading quote, asterisk or 'Transcription' label;
     4. speech clip WITH a request prompt still transcribes (the chat path is kept for an explicit prompt).
  B. default flags (off, the chat template):
     5. server up and the speech clip still transcribes (>= 80% of the reference words).
  C. --asr-native-prompt bogus:
     6. the server refuses to start (argument validation).
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
REFUSAL = ('unable to transcribe', "can't transcribe", 'cannot transcribe', "don't have the capability", 'i cannot process')

failed = 0


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def post(audio, name, fields):
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + audio + b'\r\n')
    parts.append(f'--{b}--\r\n'.encode())
    req = urllib.request.Request(BASE + '/v1/audio/transcriptions', data=b''.join(parts), method='POST',
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read().decode('utf-8', 'replace'))
    except urllib.error.HTTPError as e:
        return e.code, {'error': e.read().decode('utf-8', 'replace')[:500]}


def start(tag, args, wait_s=600):
    log = open(f'_asr_native_prompt_server_{tag}_{port}.log', 'wb')  # in the caller's cwd
    srv = subprocess.Popen([server, '-m', model, '--mmproj', mmproj, '--port', port, '-c', '4096', '--no-webui', *args, *extra],
                           stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000 if os.name == 'nt' else 0)
    for _ in range(wait_s):
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=2) as r:
                if r.status == 200:
                    return srv, True
        except Exception:  # noqa: BLE001 -- still loading; the outcome is the returned flag, checked by the caller
            pass
        if srv.poll() is not None:
            return srv, False
        time.sleep(1)
    return srv, False


def stop(srv):
    if srv.poll() is not None:
        return
    srv.terminate()
    try:
        srv.wait(timeout=30)
    except subprocess.TimeoutExpired:
        srv.kill()


ref = words(open(ref_txt, encoding='utf-8').read())
speech = open(speech_wav, 'rb').read()


def ref_hit(text):
    got = set(words(text))
    return sum(1 for w in ref if w in got)


srv, up = start('on', ['--asr-native-prompt', 'on'])
try:
    check('A server up (--asr-native-prompt on)', up, f'rc {srv.poll()}')
    if up:
        st_s, js_s = post(speech, 'speech.wav', {'model': 'x', 'temperature': '0'})
        ts = (js_s.get('text') or '').strip()
        print('A speech text:', repr(ts), flush=True)
        hit = ref_hit(ts)
        check('A speech transcribes, not a refusal', st_s == 200 and hit >= 0.8 * len(ref) and not any(p in ts.lower() for p in REFUSAL),
              f'{st_s} {hit}/{len(ref)}')
        bare = bool(ts) and ts[0] not in '"*“' and not ts.lower().startswith('transcription')
        check('A transcript is bare (no quote / markdown / label wrapping)', st_s == 200 and bare, repr(ts[:40]))
        st_p, js_p = post(speech, 'speech.wav', {'model': 'x', 'temperature': '0',
                                                  'prompt': 'Transcribe the speech in this audio verbatim. Output only the words spoken.'})
        tp = js_p.get('text') or ''
        check('A explicit request prompt still transcribes (chat path)', st_p == 200 and ref_hit(tp) >= 0.8 * len(ref), f'{st_p} {ref_hit(tp)}/{len(ref)} {tp[:80]!r}')
finally:
    stop(srv)

srv, up = start('default', [])
try:
    ok = False
    detail = f'rc {srv.poll()}'
    if up:
        st, js = post(speech, 'speech.wav', {'model': 'x', 'temperature': '0'})
        t = js.get('text') or ''
        ok = st == 200 and ref_hit(t) >= 0.8 * len(ref)
        detail = f'{st} {ref_hit(t)}/{len(ref)} {t[:80]!r}'
    check('B default (off): server up and speech transcribes', up and ok, detail)
finally:
    stop(srv)

srv, up = start('bogus', ['--asr-native-prompt', 'bogus'], wait_s=120)
try:
    check('C --asr-native-prompt bogus: server refuses to start', not up and srv.poll() not in (None, 0), f'up {up} rc {srv.poll()}')
finally:
    stop(srv)

print(f'{6 - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
