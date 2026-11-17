#!/usr/bin/env python3
"""/v1/audio/transcriptions decodes at the requested temperature, OpenAI's default 0 when omitted, and reports it.

usage: test-asr-temperature.py <llama-server> <audio model GGUF> <mmproj GGUF> <speech WAV> <port> [extra server args...]

Two defects, one route:
  - a request with no temperature fell through to the server's general sampling default instead of OpenAI's
    transcription default of 0, while verbose_json reported 0;
  - verbose_json read the temperature from the raw form body, where it is a string, so it reported 0.0 whatever was sent.
The decode temperature is read back from GET /slots (params.temperature of the slot that ran the request).

Checks (one server, two verbose_json requests, one at a time so the slot that ran each is unambiguous):
  1. both return 200;
  2. request A (no temperature field): the slot decoded at temperature 0;
  3. request A: verbose_json reports temperature 0 on its segments;
  4. request B (temperature "0.7" as a form field): the slot decoded at 0.7;
  5. request B: verbose_json reports temperature 0.7 on its segments.
Exit code = number of failed checks.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid

server, model, mmproj, speech_wav, port = sys.argv[1:6]
extra = sys.argv[6:]
BASE = f'http://127.0.0.1:{port}'


def post(audio, temperature):
    b = uuid.uuid4().hex
    parts = [
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\nx\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\nverbose_json\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="max_tokens"\r\n\r\n64\r\n'.encode(),
    ]
    if temperature is not None:
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n{temperature}\r\n'.encode())
    parts += [
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


def last_slot_temperature():
    with urllib.request.urlopen(BASE + '/slots', timeout=10) as r:
        slots = json.loads(r.read().decode('utf-8', 'replace'))
    ran = [s for s in slots if isinstance(s, dict) and s.get('id_task', -1) is not None and s.get('id_task', -1) >= 0]
    if not ran:
        return None, slots
    s = max(ran, key=lambda s: s.get('id_task', -1))
    return (s.get('params') or {}).get('temperature'), s.get('id_task')


failed = 0


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


def close(a, b):
    return isinstance(a, (int, float)) and abs(a - b) < 1e-4


log = open(f'_asr_temperature_server_{port}.log', 'wb')  # in the caller's cwd
flags = 0x08000000 if os.name == 'nt' else 0
srv = subprocess.Popen([server, '-m', model, '--mmproj', mmproj, '--port', port, '-c', '4096', '--no-webui', '-np', '1', *extra],
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
        sys.exit(failed + 5)
    audio = open(speech_wav, 'rb').read()
    st_a, js_a = post(audio, None)
    slot_a, task_a = last_slot_temperature()
    st_b, js_b = post(audio, '0.7')
    slot_b, task_b = last_slot_temperature()
    rep_a = sorted({s.get('temperature') for s in js_a.get('segments') or []}, key=str)
    rep_b = sorted({s.get('temperature') for s in js_b.get('segments') or []}, key=str)
    print('A slot', slot_a, 'task', task_a, 'reported', rep_a, '| B slot', slot_b, 'task', task_b, 'reported', rep_b, flush=True)
    check('both 200', st_a == 200 and st_b == 200, f'{st_a} {st_b}')
    check('A (omitted) decoded at 0', close(slot_a, 0.0), f'slot {slot_a}')
    check('A (omitted) reports 0', len(rep_a) == 1 and close(rep_a[0], 0.0), f'{rep_a}')
    check('B ("0.7") decoded at 0.7', close(slot_b, 0.7), f'slot {slot_b}')
    check('B ("0.7") reports 0.7', len(rep_b) == 1 and close(rep_b[0], 0.7), f'{rep_b}')
finally:
    srv.terminate()
    try:
        srv.wait(timeout=30)
    except subprocess.TimeoutExpired:
        srv.kill()
print(f'{6 - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
