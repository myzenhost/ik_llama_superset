#!/usr/bin/env bash
# llama-audio --vad-model on the real CLI path (no LLM needed: --vad-chunks-out serves before model load).
# usage: test-asr-silero-cli.sh <llama-audio> <fixture-dir>
set -u
BIN="$1"; F="$2"; fail=0
chk() { if ! eval "$2"; then echo "FAIL: $1"; fail=1; fi; }

on=$("$BIN" --vad-chunks-out --audio "$F/jfk.wav" --vad-model "$F/silero-v6.2.0-for-tests.ggml" 2>&1)
echo "$on"
chk "silero path FIRED (always-on log line)"      'grep -q "vad: silero v6.2.0 fired -- 344 probabilities -> 4 speech segment" <<<"$on"'
chk "segment count 4 (whisper.cpp: 4)"            'grep -q "^segments: 4" <<<"$on"'
chk "first segment starts at 0.322 s (whisper.cpp 0.32)" 'grep -qE "^  \[00\] +0\.32[0-9]" <<<"$on"'

off=$("$BIN" --vad-chunks-out --audio "$F/jfk.wav" 2>&1)
chk "flag OFF: energy VAD, no silero line"        '! grep -q "vad: silero" <<<"$off" && grep -q "^vad: energy" <<<"$off"'
chk "flag OFF: energy segments unchanged (3)"     'grep -q "^segments: 3" <<<"$off"'
chk "flag OFF: first segment 0.150 s"             'grep -qE "^  \[00\] +0\.150" <<<"$off"'

bad=$("$BIN" --vad-chunks-out --audio "$F/jfk.wav" --vad-model "$F/jfk.wav" 2>&1); rc=$?
chk "wrong file refused with rc!=0"               '[ $rc -ne 0 ] && grep -q "bad magic" <<<"$bad"'
# --diarize on a synthetic stereo file: jfk.wav's first half only on the LEFT channel, second half only on the RIGHT
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
# $PYTHON first: on Windows "python3" is often the Microsoft Store alias, which prints an error and runs nothing
PY=""; for c in ${PYTHON:-} python3 python; do if "$c" -c 'import wave' >/dev/null 2>&1; then PY=$c; break; fi; done
[ -n "$PY" ] || { echo "FAIL: no working python (set PYTHON)"; exit 1; }
"$PY" - "$F/jfk.wav" "$tmp/stereo.wav" <<'PY'
import sys, wave, array
w = wave.open(sys.argv[1]); n = w.getnframes(); a = array.array('h'); a.frombytes(w.readframes(n))
half = n // 2
out = array.array('h', [0]) * (2 * n)
for i in range(n):
    out[2 * i + (0 if i < half else 1)] = a[i]
o = wave.open(sys.argv[2], 'wb'); o.setnchannels(2); o.setsampwidth(2); o.setframerate(16000); o.writeframes(out.tobytes()); o.close()
PY
d=$("$BIN" --vad-chunks-out --audio "$tmp/stereo.wav" --vad-model "$F/silero-v6.2.0-for-tests.ggml" --diarize 2>&1)
echo "$d"
chk "diarize FIRED (always-on log line)"          'grep -q "diarize: stereo-energy fired" <<<"$d"'
chk "first segment (left half) is speaker 0"      'grep -qE "^  \[00\].*speaker 0$" <<<"$d"'
chk "last segment (right half) is speaker 1"      'grep -qE "^  \[03\].*speaker 1$" <<<"$d"'
nod=$("$BIN" --vad-chunks-out --audio "$tmp/stereo.wav" --vad-model "$F/silero-v6.2.0-for-tests.ggml" 2>&1)
chk "flag OFF: no speaker column"                 '! grep -q "speaker" <<<"$nod"'
mono=$("$BIN" --vad-chunks-out --audio "$F/jfk.wav" --diarize 2>&1); rc=$?
chk "mono file with --diarize refused"            '[ $rc -ne 0 ] && grep -q "needs a 2-channel" <<<"$mono"'
[ $fail -eq 0 ] && echo OK
exit $fail
