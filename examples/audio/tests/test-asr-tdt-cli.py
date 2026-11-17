#!/usr/bin/env python3
# REWRITE-SAFE: writes only fresh files (pid-suffixed WAVs, logs, word JSON) into the workdir named on the command line
"""llama-audio transducer ASR (Parakeet-TDT, no text model): known-answer checks on a real model.

usage: test-asr-tdt-cli.py <llama-audio> <parakeet-tdt.gguf> <fixtures_dir> <workdir> [<jfk.wav> <jfk_expected.txt>]

The GGUF comes from examples/audio/convert-parakeet-tdt-to-gguf.py (nvidia/parakeet-tdt-0.6b-v3, CC-BY-4.0).
fixtures_dir holds librispeech_test_clean_6930-75918-000{0,1}.{wav,txt} (examples/audio/tests/fixtures).

Checks (each one prints PASS/FAIL; any FAIL -> exit 1):
  1. jfk.wav (optional): the transcript equals whisper.cpp's tests/parakeet-expected-jfk-output.txt EXACTLY
     (case and punctuation included).
  2. the two LibriSpeech clips: word error rate against the references (lower-cased, punctuation dropped)
     at or below 2.0% over both clips (the Voxtral route's figure on the same 51 words, examples/audio/README.md).
  3. --words-out: one word per transcript word, every word inside [0, duration], start <= end, starts non-decreasing.
  4. clip A + 12 s of digital silence + clip B: the last word of A ends before dur_A + 0.5 s and the first word of
     B starts after dur_A + 12 - 0.5 s (the duration head has to skip the silence, not stretch a word over it).
  5. 20 s of -50 dBFS Gaussian noise (seed 1234) and 10 s of digital silence: the transcript is EMPTY. On the same
     noise the Voxtral LLM route wrote 383 words (examples/audio/tests/test-asr-halluc-cli.py positive control).
Negative controls: the WER of a hypothesis with two words dropped must exceed 0, and the boundary check on a copy of
the words shifted by +6 s must fail -- so neither check can pass everything.
"""
import array
import json
import os
import random
import re
import subprocess
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_wer  # noqa: E402  shared WER + English normaliser

SR = 16000
GAP = 12.0
TOL = 0.5


def read_pcm16(path):
    w = wave.open(path)
    assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == SR, path
    a = array.array('h')
    a.frombytes(w.readframes(w.getnframes()))
    return a


def write_pcm16(path, a):
    o = wave.open(path, "wb")
    o.setnchannels(1); o.setsampwidth(2); o.setframerate(SR); o.writeframes(a.tobytes()); o.close()


def noise(seconds, dbfs, seed=1234):
    rng = random.Random(seed)
    amp = 32767 * 10 ** (dbfs / 20)
    return array.array('h', [int(max(-32768, min(32767, rng.gauss(0, amp)))) for _ in range(int(seconds * SR))])


def wer(ref, hyp):
    return asr_wer.wer_counts(ref, hyp)     # (errors, reference words), Whisper-style English normaliser


def run(exe, model, wav, work, tag):
    """-> (rc, transcript, words or None)"""
    words_path = os.path.join(work, f"{tag}_words_{os.getpid()}.json")   # fresh name: a stale file can never pass
    cmd = [exe, "--task", "asr", "--mmproj", model, "--audio", wav, "--words-out", words_path, "--metrics", "-t", "4"]
    print(f"RUN [{tag}]:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log = os.path.join(work, f"{tag}_{os.getpid()}.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n----- stderr -----\n" + p.stderr)
    text = None
    for line in (p.stdout + p.stderr).splitlines():
        m = re.search(r"run_asr_tdt: transcript: ?(.*)$", line)
        if m:
            text = m.group(1).strip()
        if "metrics:" in line or "error" in line.lower() or "failed" in line.lower():
            print("  |", line[:300])
    print(f"  rc={p.returncode} transcript={text!r} log={log}", flush=True)
    if p.returncode != 0 or text is None or not os.path.exists(words_path):
        return p.returncode, text, None
    return p.returncode, text, json.load(open(words_path, encoding="utf-8"))["words"]


def words_ok(words, text, dur):
    if len(words) != len(text.split()):
        return f"{len(words)} words vs {len(text.split())} in the transcript"
    prev = -1.0
    for w in words:
        if not (0.0 <= w["start"] <= w["end"] <= dur + 1e-6):
            return f"word {w} outside [0,{dur:.3f}] or start>end"
        if w["start"] < prev - 1e-6:
            return f"starts decrease at {w}"
        prev = w["start"]
    return None


def boundary(words, n_a, dur_a):
    """the first n_a words are clip A's"""
    if len(words) <= n_a:
        return f"only {len(words)} words, clip A alone has {n_a}"
    a_end, b_start = words[n_a - 1]["end"], words[n_a]["start"]
    if a_end > dur_a + TOL:
        return f"clip A's last word '{words[n_a - 1]['word']}' ends {a_end:.3f} > {dur_a + TOL:.3f}"
    if b_start < dur_a + GAP - TOL:
        return f"clip B's first word '{words[n_a]['word']}' starts {b_start:.3f} < {dur_a + GAP - TOL:.3f}"
    return None


def main():
    if len(sys.argv) not in (5, 7):
        print(__doc__)
        return 2
    exe, model, fx, work = sys.argv[1:5]
    os.makedirs(work, exist_ok=True)
    fails = []

    def verdict(name, err):
        print(("PASS " if err is None else "FAIL ") + name + ("" if err is None else ": " + err), flush=True)
        if err is not None:
            fails.append(name)

    # 1. jfk exact
    if len(sys.argv) == 7:
        expected = open(sys.argv[6], encoding="utf-8").read().strip()
        rc, text, _ = run(exe, model, sys.argv[5], work, "jfk")
        verdict("jfk exact", None if rc == 0 and text == expected else f"rc={rc} got {text!r}, expected {expected!r}")

    # 2./3. LibriSpeech
    clips = []
    for i in (0, 1):
        base = os.path.join(fx, f"librispeech_test_clean_6930-75918-000{i}")
        ref = open(base + ".txt", encoding="utf-8").read().strip()
        a = read_pcm16(base + ".wav")
        clips.append((base, ref, a))
    errs, n_ref = 0, 0
    for i, (base, ref, a) in enumerate(clips):
        rc, text, words = run(exe, model, base + ".wav", work, f"ls{i}")
        if rc != 0 or text is None:
            verdict(f"librispeech {i} ran", f"rc={rc}")
            continue
        e, n = wer(ref, text)
        errs += e
        n_ref += n
        print(f"  ls{i}: {e} error(s) / {n} words  hyp={text!r}")
        verdict(f"librispeech {i} words", "no words file" if words is None else words_ok(words, text, len(a) / SR))
    if n_ref:
        w = 100.0 * errs / n_ref
        print(f"  WER {errs}/{n_ref} = {w:.2f}%")
        verdict("librispeech WER <= 2.0%", None if w <= 2.0 else f"WER {w:.2f}%")
    e_ctrl, _ = wer(clips[1][1], " ".join(clips[1][1].split()[2:]))
    verdict("negative control: WER sees two dropped words", None if e_ctrl == 2 else f"WER errors {e_ctrl}")

    # 4. A + 12 s silence + B
    (_, ref_a, a), (_, ref_b, b) = clips
    dur_a = len(a) / SR
    stim = array.array('h', a) + array.array('h', [0] * int(GAP * SR)) + array.array('h', b)
    p = os.path.join(work, f"A_sil12_B_{os.getpid()}.wav")
    write_pcm16(p, stim)
    rc, text, words = run(exe, model, p, work, "gap")
    n_a = len(ref_a.split())   # raw words: the --words-out list is one entry per transcript word
    if words is None:
        verdict("gap boundary", f"rc={rc}")
    else:
        verdict("gap words", words_ok(words, text, len(stim) / SR))
        verdict("gap boundary", boundary(words, n_a, dur_a))
        shifted = [dict(w, start=w["start"] + GAP / 2, end=w["end"] + GAP / 2) for w in words]
        verdict("negative control: shifted words fail the boundary", None if boundary(shifted, n_a, dur_a) else "it passed")

    # 5. nothing spoken
    for tag, sig in (("noise20", noise(20.0, -50.0)), ("silence10", array.array('h', [0] * int(10 * SR)))):
        p = os.path.join(work, f"{tag}_{os.getpid()}.wav")
        write_pcm16(p, sig)
        rc, text, words = run(exe, model, p, work, tag)
        verdict(f"{tag} empty", None if rc == 0 and text == "" and words == [] else f"rc={rc} transcript={text!r}")

    print(("FAILED: " + ", ".join(fails)) if fails else "OK", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
