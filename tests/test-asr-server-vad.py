#!/usr/bin/env python3
# REWRITE-SAFE: writes only per-run server logs (new files) in the caller's cwd; reads no store.
"""/v1/audio/transcriptions --asr-vad: the native Silero VAD gate returns an empty transcript for an upload with no speech.

usage: test-asr-server-vad.py <llama-server> <audio GGUF> <mmproj GGUF> <speech WAV> <reference .txt> <silero file> <port> [extra server args...]

MEASURED 2026-10-17 on Voxtral-Mini-3B: a silent or -50 dBFS clip gets a refusal (chat prompt) or invented text (native prompt), and the
first-position EOG probability is 0.0 on silence and speech alike, so the model cannot detect silence. --asr-vad is OPT-IN (default off).
Silence and noise are generated here: 16 kHz mono int16, 3 s; the noise is Gaussian at -50 dBFS RMS (seeded).

Three servers, same model:
  A. --asr-vad <silero file>:
     1. server up;
     2. silence -> 200, text == "";
     3. -50 dBFS noise -> 200, text == "";
     4. speech -> >= 80% of the reference words;
     5. silence, response_format=verbose_json -> 200, text "", segments [];
     6. the server log shows the gate fired and the model was not run for the silent uploads.
  B. default flags (no --asr-vad):
     7. server up and the speech clip still transcribes (>= 80% of the reference words);
        (informational, not a check: what the ungated model says on silence)
  C. --asr-vad <nonexistent file>:
     8. the server refuses to start.
  D. --asr-vad <silero file> --asr-vad-trim, upload = speech + 30 s digital silence + speech:
     9. server up;
     10. verbose_json: asr_vad.trimmed, the model heard at least 25 s less than the upload, "duration" is the upload's;
     11. both copies of the speech are transcribed;
     12. segment times are ordered, none lands inside the removed silence, the last one ends in the second copy;
     13. the plain json reply carries the asr_vad object too;
     14. the server log shows the trim ran for both uploads.
Exit code = number of failed checks.
"""
import io
import json
import os
import random
import re
import struct
import subprocess
import sys
import time
import urllib.request
import uuid
import wave

server, model, mmproj, speech_wav, ref_txt, silero, port = sys.argv[1:8]
extra = sys.argv[8:]
BASE = f'http://127.0.0.1:{port}'
N_CHECKS = 14

failed = 0


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def wav_bytes(samples_i16, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack('<%dh' % len(samples_i16), *samples_i16))
    return buf.getvalue()


def make_silence(seconds=3.0, sr=16000):
    return wav_bytes([0] * int(seconds * sr), sr)


def make_noise(dbfs=-50.0, seconds=3.0, sr=16000, seed=1017):
    rng = random.Random(seed)
    sigma = 32767.0 * (10.0 ** (dbfs / 20.0))   # RMS of the Gaussian = sigma
    return wav_bytes([max(-32768, min(32767, int(round(rng.gauss(0.0, sigma))))) for _ in range(int(seconds * sr))], sr)


def post(audio, name, fields):
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode() + audio + b'\r\n')
    parts.append(f'--{b}--\r\n'.encode())
    req = urllib.request.Request(BASE + '/v1/audio/transcriptions', data=b''.join(parts), method='POST',
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            body = r.read().decode('utf-8', 'replace')
            try:
                return r.status, json.loads(body), time.time() - t0
            except json.JSONDecodeError:
                return r.status, {'raw': body}, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {'error': e.read().decode('utf-8', 'replace')[:500]}, time.time() - t0


def log_path(tag):
    return f'_asr_server_vad_{tag}_{port}.log'   # in the caller's cwd


def start(tag, args, wait_s=600):
    log = open(log_path(tag), 'wb')
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
silence = make_silence()
noise = make_noise()


def ref_hit(text):
    got = set(words(text))
    return sum(1 for w in ref if w in got)


# the upload of section D (speech + 30 s digital silence + speech), built up front so section A can transcribe it
# UNTRIMMED as the baseline D is judged against
GAP_S = 30.0


def compose_with_gap(wav, gap_s):
    """speech + gap_s of digital silence + speech, in the speech file's own format. -> (bytes, speech seconds) or (None, why)."""
    try:
        with wave.open(io.BytesIO(wav), 'rb') as w:
            nch, sw, sr, nfr = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            frames = w.readframes(nfr)
    except (wave.Error, EOFError) as e:
        return None, f'speech WAV not readable as PCM: {e}'
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as o:
        o.setnchannels(nch)
        o.setsampwidth(sw)
        o.setframerate(sr)
        o.writeframes(frames + b'\x00' * (int(gap_s * sr) * nch * sw) + frames)
    return buf.getvalue(), nfr / float(sr)


gap_clip, speech_s = compose_with_gap(speech, GAP_S)
base_gap_text = None   # what the gate-only server (no trim) writes for gap_clip

# ---- A. gate on ----
srv, up = start('vad', ['--asr-vad', silero])
try:
    check('1 A server up (--asr-vad)', up, f'rc {srv.poll()}')
    if up:
        st, js, dt = post(silence, 'silence.wav', {'model': 'x', 'temperature': '0'})
        print('A silence reply:', json.dumps(js)[:300], f'({dt:.2f} s)', flush=True)
        check('2 A silence -> empty text', st == 200 and js.get('text') == '', f'{st} {js.get("text")!r}')

        st, js, dt = post(noise, 'noise.wav', {'model': 'x', 'temperature': '0'})
        print('A noise reply:', json.dumps(js)[:300], f'({dt:.2f} s)', flush=True)
        check('3 A -50 dBFS noise -> empty text', st == 200 and js.get('text') == '', f'{st} {js.get("text")!r}')

        st, js, dt = post(speech, 'speech.wav', {'model': 'x', 'temperature': '0'})
        t = js.get('text') or ''
        print('A speech text:', repr(t), f'({dt:.2f} s)', flush=True)
        check('4 A speech transcribes (>= 80% reference words)', st == 200 and ref_hit(t) >= 0.8 * len(ref), f'{st} {ref_hit(t)}/{len(ref)}')

        st, js, dt = post(silence, 'silence.wav', {'model': 'x', 'temperature': '0', 'response_format': 'verbose_json'})
        print('A silence verbose_json:', json.dumps(js)[:300], flush=True)
        check('5 A silence verbose_json -> text "" and no segments',
              st == 200 and js.get('text') == '' and js.get('segments') == [], f'{st} {js.get("text")!r} {js.get("segments")!r}')
        if gap_clip is not None:
            # baseline for section D (not a check): the same speech + silence + speech upload, gated but NOT trimmed
            st, js, dt = post(gap_clip, 'speech-gap-speech.wav', {'model': 'x', 'temperature': '0'})
            base_gap_text = (js.get('text') or '') if st == 200 else None
            print('A (baseline for D) untrimmed speech+silence+speech:', st, repr((base_gap_text or '')[:200]), f'({dt:.2f} s)', flush=True)
    else:
        for n in ('2 A silence -> empty text', '3 A -50 dBFS noise -> empty text', '4 A speech transcribes (>= 80% reference words)',
                  '5 A silence verbose_json -> text "" and no segments'):
            check(n, False, 'server not up')
finally:
    stop(srv)
try:
    lg = open(log_path('vad'), 'rb').read().decode('utf-8', 'replace')
except OSError:
    lg = ''
n_gated = len(re.findall(r'asr-vad: .* -> NO SPEECH: empty transcript, model not run', lg))
n_speech = len(re.findall(r'asr-vad: .* -> speech: transcribing', lg))
n_speech_want = 1 + (1 if gap_clip is not None else 0)   # the speech clip, plus the baseline upload for section D
check('6 A log: gate fired (3 silent uploads not run, %d speech upload(s) transcribed)' % n_speech_want,
      n_gated == 3 and n_speech == n_speech_want, f'gated {n_gated} speech {n_speech}')

# ---- B. default (gate off) ----
srv, up = start('default', [])
try:
    ok = False
    detail = f'rc {srv.poll()}'
    if up:
        st, js, dt = post(speech, 'speech.wav', {'model': 'x', 'temperature': '0'})
        t = js.get('text') or ''
        ok = st == 200 and ref_hit(t) >= 0.8 * len(ref)
        detail = f'{st} {ref_hit(t)}/{len(ref)} {t[:80]!r}'
        st, js, dt = post(silence, 'silence.wav', {'model': 'x', 'temperature': '0'})
        print('B (informational) ungated silence reply:', st, repr((js.get('text') or js.get('error') or '')[:200]), f'({dt:.2f} s)', flush=True)
    check('7 B default (no --asr-vad): server up and speech transcribes', up and ok, detail)
finally:
    stop(srv)

# ---- C. bad file ----
srv, up = start('badfile', ['--asr-vad', os.path.join(os.getcwd(), 'no-such-silero-file.bin')], wait_s=300)
try:
    check('8 C --asr-vad <nonexistent>: server refuses to start', not up and srv.poll() not in (None, 0), f'up {up} rc {srv.poll()}')
finally:
    stop(srv)
try:
    lg = open(log_path('badfile'), 'rb').read().decode('utf-8', 'replace')
    print('C log tail:', ' | '.join(lg.strip().splitlines()[-3:])[:400], flush=True)
except OSError:
    pass

# ---- D. --asr-vad-trim: the silence is cut out before the model runs, the times come back on the upload ----
D_NAMES = ('9 D server up (--asr-vad --asr-vad-trim)',
           '10 D verbose_json: asr_vad.trimmed, the model heard >= 25 s less than the upload, duration is the upload',
           '11 D trimmed transcript keeps the reference words of the untrimmed one (A baseline, one word slack) and >= 80% of them',
           '12 D segment times: ordered, none inside the removed silence, the last one ends in the second copy',
           '13 D json reply carries the asr_vad object (trimmed)',
           '14 D log: the trim ran for both uploads')
srv, up = start('trim', ['--asr-vad', silero, '--asr-vad-trim'])
try:
    check(D_NAMES[0], up, f'rc {srv.poll()}')
    if up and gap_clip is not None:
        total_s = 2 * speech_s + GAP_S
        st, js, dt = post(gap_clip, 'speech-gap-speech.wav', {'model': 'x', 'temperature': '0', 'response_format': 'verbose_json'})
        print('D verbose_json:', json.dumps(js)[:900], f'({dt:.2f} s)', flush=True)
        av = js.get('asr_vad') or {}
        check(D_NAMES[1], st == 200 and av.get('trimmed') is True and isinstance(av.get('transcribed_s'), (int, float))
              and av['transcribed_s'] <= total_s - 25.0 and abs(float(js.get('duration', -1)) - total_s) <= 0.05,
              f'{st} asr_vad {av} duration {js.get("duration")} upload {total_s:.3f} s')
        t = js.get('text') or ''
        # MEASURED in gate b1114e (2026-11-14): with the same sentence twice joined by 100 ms the model wrote it ONCE, so a
        # "both copies" check judged the model, not the trim. The trim's own claim is that it loses nothing the untrimmed
        # upload gives: every reference word the gate-only server (section A) wrote for the same upload is still there.
        # MEASURED in the b1114e rerun (2026-11-15): trimmed 'Concord returned its place amidst the tents.' (7/8) vs untrimmed
        # 8/8 -- the joined copies (100 ms apart instead of 28 s) came out once and without the short word "to". That is
        # the model on a different input, not lost audio (check 10: both speech pieces kept, check 12: times outside the
        # silence). So the trimmed text may trail the untrimmed one by ONE reference word, and must still reach 80%.
        # The pad length that keeps sentences apart is row S69-FORK-ASR-VAD-TRIM-DEFAULT-PAD-MERGES-SENTENCES-2026-11-15-01.
        base_hit = ref_hit(base_gap_text) if base_gap_text is not None else None
        check(D_NAMES[2], st == 200 and base_hit is not None and ref_hit(t) >= base_hit - 1 and ref_hit(t) >= 0.8 * len(ref),
              f'{st} trimmed hit {ref_hit(t)}/{len(ref)} ({len(words(t))} words) vs untrimmed hit {base_hit} '
              f'({len(words(base_gap_text or ""))} words); trimmed {t[:100]!r} untrimmed {(base_gap_text or "")[:100]!r}')
        segs = js.get('segments') or []
        lo, hi = speech_s + 1.0, speech_s + GAP_S - 1.0     # the inside of the silence; no time may land there
        ordered = all(s['end'] >= s['start'] for s in segs) and all(segs[i + 1]['start'] >= segs[i]['start'] for i in range(len(segs) - 1))
        outside = all(not (lo < s['start'] < hi) and not (lo < s['end'] < hi) for s in segs)
        reaches = bool(segs) and segs[0]['start'] < speech_s and segs[-1]['end'] >= speech_s + GAP_S + 0.5 * speech_s and segs[-1]['end'] <= total_s + 0.05
        check(D_NAMES[3], st == 200 and ordered and outside and reaches,
              f'ordered {ordered} outside {outside} reaches {reaches} silence ({lo:.2f}, {hi:.2f}) segs {[(s["start"], s["end"]) for s in segs]}')
        st, js, dt = post(gap_clip, 'speech-gap-speech.wav', {'model': 'x', 'temperature': '0'})
        print('D json:', json.dumps(js)[:500], f'({dt:.2f} s)', flush=True)
        check(D_NAMES[4], st == 200 and (js.get('asr_vad') or {}).get('trimmed') is True and ref_hit(js.get('text') or '') >= 0.8 * len(ref),
              f'{st} asr_vad {js.get("asr_vad")}')
    else:
        for n in D_NAMES[1:5]:
            check(n, False, 'server not up' if not up else str(speech_s))
finally:
    stop(srv)
try:
    lg = open(log_path('trim'), 'rb').read().decode('utf-8', 'replace')
except OSError:
    lg = ''
n_trim = len(re.findall(r'asr-vad-trim: [0-9.]+ s of [0-9.]+ s kept in \d+ piece', lg))
check(D_NAMES[5], n_trim == 2, f'trim lines {n_trim}')

print(f'{N_CHECKS - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
