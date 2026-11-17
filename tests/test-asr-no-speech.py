#!/usr/bin/env python3
"""/v1/audio/transcriptions verbose_json carries a no_speech_prob proxy.

usage: test-asr-no-speech.py <llama-server> <audio model GGUF> <mmproj GGUF> <speech WAV> <port> [extra server args...]

An LLM audio decoder has no no-speech head, so verbose_json's no_speech_prob used to be null on every segment. It is
now the summed probability of the end-of-generation tokens at the FIRST generated position (the model's own chance of
ending the reply before writing anything), written on every segment like avg_logprob.

Checks (one server, two requests):
  1. both requests return 200 with at least one segment;
  2. every segment of both replies has a numeric no_speech_prob in [0, 1];
  3. the value is the same on every segment of one reply (one decode covers the whole audio);
  4. the silent clip's no_speech_prob is higher than the speech clip's.
Exit code = number of failed checks.
"""
import io
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
import uuid
import wave

server, model, mmproj, speech_wav, port = sys.argv[1:6]
extra = sys.argv[6:]
BASE = f'http://127.0.0.1:{port}'


def silence_wav(seconds=3.0, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack('<h', 0) * int(seconds * sr))
    return buf.getvalue()


def post(audio, name):
    b = uuid.uuid4().hex
    parts = [
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\nx\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\nverbose_json\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + audio + b'\r\n',
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


log = open(f'_asr_no_speech_server_{port}.log', 'wb')  # in the caller's cwd
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
    st_s, js_s = post(open(speech_wav, 'rb').read(), 'speech.wav')
    st_z, js_z = post(silence_wav(), 'silence.wav')
    seg_s, seg_z = js_s.get('segments') or [], js_z.get('segments') or []
    print('speech  text:', repr(js_s.get('text')), ' silence text:', repr(js_z.get('text')), flush=True)
    check('both 200 with segments', st_s == 200 and st_z == 200 and seg_s and seg_z, f'{st_s} {len(seg_s)} / {st_z} {len(seg_z)}')
    vals_s = [s.get('no_speech_prob') for s in seg_s]
    vals_z = [s.get('no_speech_prob') for s in seg_z]
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0  # noqa: E731
    check('no_speech_prob numeric in [0,1] on every segment', bool(vals_s and vals_z) and all(map(num, vals_s + vals_z)),
          f'speech {vals_s} silence {vals_z}')
    check('one value per reply', len(set(vals_s)) == 1 and len(set(vals_z)) == 1)
    check('silence no_speech_prob > speech no_speech_prob', bool(vals_s and vals_z) and all(map(num, vals_s + vals_z)) and vals_z[0] > vals_s[0],
          f'silence {vals_z[:1]} speech {vals_s[:1]}')
finally:
    srv.terminate()
    try:
        srv.wait(timeout=30)
    except subprocess.TimeoutExpired:
        srv.kill()
print(f'{5 - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
