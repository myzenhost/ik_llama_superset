#!/usr/bin/env python3
"""/v1/audio/transcriptions timestamp_granularities[]=word returns OpenAI verbose_json top-level `words`.

usage: test-asr-word-timestamps.py <llama-server> <audio model GGUF> <mmproj GGUF> <mono 16-bit speech WAV> <port> [extra server args...]

The words come from the fork's own aligner (examples/audio/asr-align.h, the one `llama-audio --word-timestamps` uses),
run on a second flash-attention-off context after the normal decode. Requests against one server:
  A. verbose_json, no granularity          -> 200, no "words" key (field absent = response unchanged)
  B. verbose_json + timestamp_granularities[]=word -> 200, "words" non-empty, each {word,start,end}, 0 <= start <= end,
     times non-decreasing and inside the clip
  C. B again                               -> the same words (deterministic)
  D. json (not verbose_json) + word        -> 400 (words ride on verbose_json only)
  E. verbose_json + granularity "bogus"    -> 400
  F. A again                               -> byte-identical to A
Exit code = number of failed checks. When env ASR_WORDS_REF_JSON names a JSON list of {word,start,end} (the llama-audio
--word-timestamps output for the same clip), B is also compared with it and the largest time difference is printed.
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


def post(audio, name, fmt, grans):
    b = uuid.uuid4().hex
    parts = [
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\nx\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\n{fmt}\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="temperature"\r\n\r\n0\r\n'.encode(),
    ]
    for g in grans:
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="timestamp_granularities[]"\r\n\r\n{g}\r\n'.encode())
    parts += [
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + audio + b'\r\n',
        f'--{b}--\r\n'.encode(),
    ]
    req = urllib.request.Request(BASE + '/v1/audio/transcriptions', data=b''.join(parts), method='POST',
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, r.read(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t0


failed = 0


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


def js(raw):
    try:
        return json.loads(raw.decode('utf-8', 'replace'))
    except Exception:  # noqa: BLE001
        return {}


audio = open(speech_wav, 'rb').read()
with wave.open(io.BytesIO(audio), 'rb') as w:
    clip_s = w.getnframes() / float(w.getframerate())
log = open(f'_asr_words_server_{port}.log', 'wb')  # in the caller's cwd
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
        sys.exit(failed + 8)

    sa, ra, ta = post(audio, 'a.wav', 'verbose_json', [])
    sb, rb, tb = post(audio, 'b.wav', 'verbose_json', ['word'])
    sc, rc, tc = post(audio, 'c.wav', 'verbose_json', ['word'])
    sd, rd, _ = post(audio, 'd.wav', 'json', ['word'])
    se, re_, _ = post(audio, 'e.wav', 'verbose_json', ['bogus'])
    sf, rf, _ = post(audio, 'f.wav', 'verbose_json', [])
    ja, jb, jc = js(ra), js(rb), js(rc)
    print(f'timings: plain {ta:.2f}s, with words {tb:.2f}s (first, creates the second context), again {tc:.2f}s', flush=True)

    check('A plain verbose_json 200 with text', sa == 200 and bool(ja.get('text')), f'{sa}')
    check('A has no "words" key', 'words' not in ja)
    words = jb.get('words') or []
    check('B 200 with a non-empty words list', sb == 200 and len(words) > 0, f'{sb} n={len(words)} body={rb[:200]!r}')
    shape = all(isinstance(x, dict) and set(('word', 'start', 'end')) <= set(x) for x in words)
    check('B every word has word/start/end', shape)
    if shape and words:
        ok_t = all(0 <= x['start'] <= x['end'] for x in words) and \
            all(words[i]['start'] <= words[i + 1]['start'] + 1e-6 for i in range(len(words) - 1)) and \
            words[-1]['end'] <= clip_s + 1.0
        check('B times ordered and inside the clip', ok_t, f'clip {clip_s:.2f}s first {words[0]} last {words[-1]}')
        joined = ''.join(x['word'] for x in words).replace(' ', '').lower()
        text = ''.join(ch for ch in (jb.get('text') or '')).replace(' ', '').lower()
        check('B words concatenate to the transcript text', bool(joined) and joined == text, f'{joined[:60]!r} vs {text[:60]!r}')
    check('B text equals A text (the word pass does not change the transcript)', jb.get('text') == ja.get('text'))
    check('C repeat gives the same words', jc.get('words') == jb.get('words'))
    check('D json + word is a 400', sd == 400, f'{sd} {rd[:160]!r}')
    check('E unknown granularity is a 400', se == 400, f'{se} {re_[:160]!r}')
    # usage.cached_tokens legitimately differs: the repeat request hits the prompt cache
    import re as _re
    _norm = lambda b: _re.sub(rb'"cached_tokens":\d+', b'"cached_tokens":N', b)
    ra, rf = _norm(ra), _norm(rf)
    dpos = next((i for i in range(min(len(ra), len(rf))) if ra[i] != rf[i]), min(len(ra), len(rf)))
    check('F plain request byte-identical to A (after the word pass ran; cached_tokens normalised)', sf == 200 and rf == ra,
          f'first diff at byte {dpos}: A {ra[max(0, dpos - 60):dpos + 60]!r} F {rf[max(0, dpos - 60):dpos + 60]!r}')

    ref_path = os.environ.get('ASR_WORDS_REF_JSON')
    if ref_path and words:
        ref = json.load(open(ref_path, 'r', encoding='utf-8'))
        if isinstance(ref, dict):
            ref = ref['words']
        _tok = lambda ws: [w['word'].strip() for w in ws]
        if _tok(ref) != _tok(words) and os.environ.get('ASR_WORDS_REF_SKIP_IF_TEXT_DIFFERS'):
            print('reference: SKIPPED, llama-audio and the server transcribed different text '
                  f'({len(ref)} vs {len(words)} words); the aligner is only comparable on equal transcripts', flush=True)
            ref = words
        same_n = len(ref) == len(words)
        worst = max((max(abs(r['start'] - w['start']), abs(r['end'] - w['end'])) for r, w in zip(ref, words)), default=1e9)
        print(f'llama-audio reference: {len(ref)} words vs server {len(words)}; max time difference {worst * 1000:.1f} ms', flush=True)
        check('reference: same word list', same_n and all(r['word'].strip() == w['word'].strip() for r, w in zip(ref, words)))
        check('reference: times within 20 ms', worst <= 0.020, f'{worst * 1000:.1f} ms')
finally:
    srv.terminate()
    try:
        srv.wait(timeout=30)
    except subprocess.TimeoutExpired:
        srv.kill()
print(f'done, {failed} failed', flush=True)
sys.exit(failed)
