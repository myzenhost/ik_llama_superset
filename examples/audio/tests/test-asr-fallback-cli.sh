#!/usr/bin/env bash
# llama-audio on the tiny random-weight model (examples/audio/tests/make_tiny_asr_gguf.py): no mmproj, no audio.
# Covers (1) the exit-time double free, (2) --asr-fallback OFF is unchanged, (3) it FIRES and changes the outcome when
# the gates trip, (4) a control with permissive gates accepts attempt 0 and equals the OFF output.
# usage: test-asr-fallback-cli.sh <llama-audio> <tiny-asr.gguf>
set -u
BIN="$1"; M="$2"; fail=0
chk() { if ! eval "$2"; then echo "FAIL: $1"; fail=1; fi; }
run() { "$BIN" --task chat -m "$M" -p "w5 w6 w7" -n 12 --temp 0 -c 256 -s 1 "$@" 2>&1; echo "RC=$?"; }

off=$(run)
echo "$off" | tail -3
chk "exit code 0 (was: double free or corruption / SIGSEGV)"  'grep -q "^RC=0$" <<<"$off"'
chk "no glibc abort"                                            '! grep -q "double free" <<<"$off"'
chk "flag OFF prints no asr-fallback line"                      '! grep -q "asr-fallback:" <<<"$off"'
off_text=$(grep -v "^RC=" <<<"$off" | grep -E "^w[0-9 ]|^[a-z]+[0-9]+" | tail -1)
chk "flag OFF generated something"                              '[ -n "$off_text" ]'

perm=$(run --asr-fallback --asr-fallback-logprob -1000 --asr-fallback-entropy 0 --asr-fallback-compression 1000)
echo "$perm" | grep "asr-fallback:"
chk "permissive gates: FIRED, single attempt, accepted"        'grep -q "asr-fallback: attempt 0 T=0.00" <<<"$perm" && ! grep -q "attempt 1 " <<<"$perm" && grep -q "chose attempt 0 of 1" <<<"$perm"'
perm_text=$(grep -v "^RC=" <<<"$perm" | grep -E "^w[0-9 ]|^[a-z]+[0-9]+" | tail -1)
chk "permissive gates: same text as OFF"                       '[ "$perm_text" = "$off_text" ]'

on=$(run --asr-fallback --asr-fallback-logprob -0.001)
echo "$on" | grep "asr-fallback:"
chk "tight logprob gate: FIRED with retries"                    'grep -q "attempt 1 T=0.20" <<<"$on" && grep -q "RETRY" <<<"$on"'
chk "ladder ran to T=1.00 and was exhausted"                   'grep -q "attempt 5 T=1.00" <<<"$on" && grep -q "ladder exhausted" <<<"$on"'
chk "still exit 0"                                              'grep -q "^RC=0$" <<<"$on"'
[ $fail -eq 0 ] && echo OK
exit $fail
