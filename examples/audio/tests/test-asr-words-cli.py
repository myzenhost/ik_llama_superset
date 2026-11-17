#!/usr/bin/env python3
"""llama-audio --word-timestamps on a real audio LLM: word times must follow the audio.

usage: test-asr-words-cli.py <llama-audio> <model.gguf> <mmproj.gguf> <clipA.wav> <clipB.wav> <workdir> [--gap=S] [extra llama-audio args...]

Stimulus with a KNOWN answer: clip A, then GAP seconds (default 3, --gap=S) of digital silence, then clip B (16 kHz mono
PCM16). A long gap (8-12 s) is the hard case for an audio LLM: plain attention DTW hands the silence to a word in the
middle of clip A (MEASURED 2026-10-19 on Voxtral; llama-audio weights the DTW by the audio's energy for it).
Nothing is spoken during [dur_A, dur_A + GAP], so a correct aligner puts every word of A before the gap and every
word of B after it. Checks:
  1. llama-audio accepts --words-out and writes the file (fail-before: the flag does not exist)
  2. at least 80% as many words as the two reference transcripts have
  3. every word inside [0, duration], start <= end, starts non-decreasing
  4. the boundary: the LAST word of clip A ends before dur_A + TOL, the word after it starts after dur_A + GAP - TOL
  5. no word lies wholly inside the silent gap
Negative control: the same checks against a SHIFTED copy of the words (every time + GAP/2) must fail check 4,
so a check that passes anything is caught.
Reference transcripts are the .txt files next to the clips.
"""
import array
import json
import os
import re
import subprocess
import sys
import wave

GAP = 3.0
TOL = 0.5


def read_pcm16(path):
    w = wave.open(path)
    assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000, path
    a = array.array('h')
    a.frombytes(w.readframes(w.getnframes()))
    return a


def norm(s):
    return re.sub(r"[^a-z0-9']", "", s.lower())


def check(words, dur_a, dur, ref_a, ref_b, label):
    fails = []
    n_ref = len(ref_a.split()) + len(ref_b.split())
    if len(words) < 0.8 * n_ref:
        fails.append(f"{label}: {len(words)} words < 80% of {n_ref}")
    prev = -1.0
    for w in words:
        if not (0.0 <= w["start"] <= w["end"] <= dur + 1e-6):
            fails.append(f"{label}: word {w} outside [0,{dur:.3f}] or start>end")
            break
        if w["start"] < prev - 1e-6:
            fails.append(f"{label}: starts decrease at {w}")
            break
        prev = w["start"]
    last_a = norm(ref_a.split()[-1])
    idx = [i for i, w in enumerate(words) if norm(w["word"]) == last_a]
    if not idx:
        fails.append(f"{label}: last word of clip A '{last_a}' not in the hypothesis")
    else:
        i = idx[0]
        ea = words[i]["end"]
        sb = words[i + 1]["start"] if i + 1 < len(words) else None
        print(f"{label}: '{words[i]['word']}' ends {ea:.3f} (clip A ends {dur_a:.3f}); next word "
              f"{(words[i + 1]['word'] if sb is not None else None)!r} starts {sb} (clip B starts {dur_a + GAP:.3f})")
        if ea > dur_a + TOL:
            fails.append(f"{label}: last word of A ends {ea:.3f} > {dur_a + TOL:.3f}")
        if sb is None or sb < dur_a + GAP - TOL:
            fails.append(f"{label}: first word of B starts {sb} < {dur_a + GAP - TOL:.3f}")
    for w in words:
        if w["start"] >= dur_a + TOL and w["end"] <= dur_a + GAP - TOL:
            fails.append(f"{label}: word {w} lies inside the silent gap")
            break
    return fails


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        return 2
    global GAP
    exe, model, mmproj, wa, wb, work = sys.argv[1:7]
    extra = sys.argv[7:]
    for x in list(extra):
        if x.startswith("--gap="):
            GAP = float(x.split("=", 1)[1])
            extra.remove(x)
    print(f"gap {GAP:.1f} s")
    os.makedirs(work, exist_ok=True)
    a, b = read_pcm16(wa), read_pcm16(wb)
    stim = array.array('h', a) + array.array('h', [0]) * int(GAP * 16000) + array.array('h', b)
    stim_path = os.path.join(work, "stim_a_gap_b.wav")
    o = wave.open(stim_path, "wb")
    o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000); o.writeframes(stim.tobytes()); o.close()
    dur_a, dur = len(a) / 16000.0, len(stim) / 16000.0
    ref_a = open(os.path.splitext(wa)[0] + ".txt").read().strip()
    ref_b = open(os.path.splitext(wb)[0] + ".txt").read().strip()

    # a fresh name per run, so a file left by an earlier run can never satisfy check 1
    words_path = os.path.join(work, f"words_{os.getpid()}.json")
    cmd = [exe, "--task", "asr", "-m", model, "--mmproj", mmproj, "--audio", stim_path,
           "--audio-chunk-mode", "none", "--words-out", words_path, "--temp", "0", "-c", "4096"] + extra
    print("RUN:", " ".join(cmd), flush=True)
    # no console window on Windows (0 elsewhere)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log = os.path.join(work, f"llama-audio_{os.getpid()}.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n----- stderr -----\n" + p.stderr)
    for line in (p.stdout + p.stderr).splitlines():
        if "word-timestamps" in line or "transcript:" in line or "error" in line.lower():
            print("  |", line)
    print(f"rc={p.returncode} log={log}")
    if p.returncode != 0 or not os.path.exists(words_path):
        print("FAIL: llama-audio did not write --words-out (rc %d)" % p.returncode)
        print("FAILED")
        return 1
    words = json.load(open(words_path, encoding="utf-8"))["words"]
    for w in words:
        print(f"  [{w['start']:7.3f} -> {w['end']:7.3f}] p={w['probability']:.3f} {w['word']}")
    fails = check(words, dur_a, dur, ref_a, ref_b, "words")
    shifted = [dict(w, start=min(dur, w["start"] + GAP / 2), end=min(dur, w["end"] + GAP / 2)) for w in words]
    neg = check(shifted, dur_a, dur, ref_a, ref_b, "negative-control")
    if not neg:
        fails.append("negative control PASSED: the checks cannot tell shifted times from real ones")
    else:
        print("negative control rejected as expected:", neg[0])
    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else "FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
