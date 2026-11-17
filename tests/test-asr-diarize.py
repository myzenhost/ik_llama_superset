#!/usr/bin/env python3
"""/v1/audio/transcriptions diarize=true labels verbose_json segments by stereo channel energy.

usage: test-asr-diarize.py <llama-server> <audio model GGUF> <mmproj GGUF> <mono 16-bit speech WAV> <port> [extra server args...]

The request field `diarize` (whisper.cpp server's name for it) asks for a per-segment "speaker" on verbose_json: "0" when
the left channel carries over 10% more absolute energy across the segment's [start, end], "1" for the right channel, "?"
otherwise. It is meaningful only for channel-separated recordings (one speaker per channel), and the file must be 2-channel;
a mono upload gets no "speaker" key rather than an invented one.

Four requests against one server, built from the one mono speech WAV:
  A. stereo, speech LEFT, silence right, diarize=true
  B. stereo, silence left, speech RIGHT, diarize=true
  C. stereo, speech left, diarize omitted
  D. the mono WAV itself, diarize=true
Checks:
  1. all four return 200 with at least one segment;
  2. every segment of A has speaker "0";
  3. every segment of B has speaker "1";
  4. no segment of C has a "speaker" key (off unless asked);
  5. no segment of D has a "speaker" key (mono: nothing to attribute).
Exit code = number of failed checks.
"""
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
import wave

server, model, mmproj, speech_wav, port = sys.argv[1:6]
extra = sys.argv[6:]
BASE = f'http://127.0.0.1:{port}'


def read_mono16(path):
    with wave.open(path, 'rb') as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f'{path}: need a mono 16-bit WAV (got {w.getnchannels()} ch, {8 * w.getsampwidth()} bit)')
        return w.getframerate(), w.readframes(w.getnframes())


def stereo_wav(sr, mono, speech_left):
    n = len(mono) // 2
    zero = b'\x00\x00'
    out = bytearray()
    for i in range(n):
        s = mono[2 * i:2 * i + 2]
        out += (s + zero) if speech_left else (zero + s)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(out))
    return buf.getvalue()


def post(audio, name, diarize):
    b = uuid.uuid4().hex
    parts = [
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\nx\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\nverbose_json\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0\r\n'.encode(),
    ]
    if diarize:
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="diarize"\r\n\r\ntrue\r\n'.encode())
    parts += [
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


sr, mono = read_mono16(speech_wav)
log = open(f'_asr_diarize_server_{port}.log', 'wb')  # in the caller's cwd
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
        sys.exit(failed + 5)
    res = {
        'A': post(stereo_wav(sr, mono, True), 'left.wav', True),
        'B': post(stereo_wav(sr, mono, False), 'right.wav', True),
        'C': post(stereo_wav(sr, mono, True), 'left_nodiar.wav', False),
        'D': post(open(speech_wav, 'rb').read(), 'mono.wav', True),
    }
    segs = {k: (v[1].get('segments') or []) for k, v in res.items()}
    for k, (st, js) in res.items():
        print(k, st, 'text:', repr(js.get('text')), 'speakers:', [s.get('speaker', '<absent>') for s in segs[k]], flush=True)
    check('all four 200 with segments', all(st == 200 and segs[k] for k, (st, _) in res.items()),
          ' '.join(f'{k}:{st}/{len(segs[k])}' for k, (st, _) in res.items()))
    check('A (speech left) every segment speaker "0"', bool(segs['A']) and all(s.get('speaker') == '0' for s in segs['A']),
          str([s.get('speaker') for s in segs['A']]))
    check('B (speech right) every segment speaker "1"', bool(segs['B']) and all(s.get('speaker') == '1' for s in segs['B']),
          str([s.get('speaker') for s in segs['B']]))
    check('C (diarize omitted) no speaker key', bool(segs['C']) and all('speaker' not in s for s in segs['C']))
    check('D (mono, diarize=true) no speaker key', bool(segs['D']) and all('speaker' not in s for s in segs['D']))
finally:
    srv.terminate()
    try:
        srv.wait(timeout=30)
    except subprocess.TimeoutExpired:
        srv.kill()
print(f'{6 - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
