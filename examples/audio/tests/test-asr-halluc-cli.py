#!/usr/bin/env python3
"""llama-audio --hallucination-silence-threshold on a real audio LLM: text written over noise is dropped, speech is kept.

usage: test-asr-halluc-cli.py <llama-audio> <model.gguf> <mmproj.gguf> <clipA.wav> <clipB.wav> <workdir> [extra llama-audio args...]

Stimuli (16 kHz mono PCM16, built here, deterministic):
  noise20     20 s of Gaussian noise at -50 dBFS, nothing spoken. MEASURED 2026-10-19 on Voxtral-Mini-3B: the model
              writes a fabricated sentence over it, over and over ("Yeah, I think it's a great idea." ~40 times), every
              word piled onto the last audio position.
  A_noise12_B clip A, 12 s of the same noise, clip B: real speech on both sides of a long noise stretch.
Checks:
  0. POSITIVE CONTROL, flag off: noise20 must produce at least MIN_HALLUC words. If the model does not hallucinate
     here the test proves nothing, so that is a failure, not a pass.
  1. llama-audio accepts --hallucination-silence-threshold (fail-before: the flag does not exist)
  2. flag on, noise20: no word survives
  3. flag on, A_noise12_B: at least 80% as many words as the two reference transcripts, the last word of A ends
     before dur_A + TOL and the first word of B starts after dur_A + 12 - TOL (real speech is not dropped)
Negative control: check 2 applied to the flag-off noise20 words must fail (a check that passes anything is caught).
Reference transcripts are the .txt files next to the clips.
"""
# REWRITE-SAFE: reads only the two input clips; writes only fresh stimulus WAVs, logs and per-pid word files in <workdir>
import array
import json
import os
import random
import re
import subprocess
import sys
import wave

SR = 16000
GAP = 12.0
TOL = 0.5
MIN_HALLUC = 5
THRESH = "2"


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


def norm(s):
    return re.sub(r"[^a-z0-9']", "", s.lower())


def run(exe, model, mmproj, wav, work, tag, extra):
    """-> (rc, words or None, log path)"""
    words_path = os.path.join(work, f"{tag}_words_{os.getpid()}.json")   # fresh name: a stale file can never pass
    cmd = [exe, "--task", "asr", "-m", model, "--mmproj", mmproj, "--audio", wav, "--audio-chunk-mode", "none",
           "--words-out", words_path, "--temp", "0", "-c", "4096"] + extra
    print(f"RUN [{tag}]:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log = os.path.join(work, f"{tag}_{os.getpid()}.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n----- stderr -----\n" + p.stderr)
    for line in (p.stdout + p.stderr).splitlines():
        if "hallucination-skip" in line or "transcript:" in line or "invalid argument" in line or "error" in line.lower():
            print("  |", line[:300])
    print(f"  rc={p.returncode} log={log}")
    if p.returncode != 0 or not os.path.exists(words_path):
        return p.returncode, None, log
    return p.returncode, json.load(open(words_path, encoding="utf-8"))["words"], log


def check_dropped(words, label):
    if words:
        return [f"{label}: {len(words)} word(s) survive over pure noise, first {words[0]}"]
    return []


def check_kept(words, dur_a, dur, ref_a, ref_b, label):
    fails = []
    n_ref = len(ref_a.split()) + len(ref_b.split())
    if len(words) < 0.8 * n_ref:
        fails.append(f"{label}: {len(words)} words < 80% of {n_ref}")
    for w in words:
        if not (0.0 <= w["start"] <= w["end"] <= dur + 1e-6):
            fails.append(f"{label}: word {w} outside [0,{dur:.3f}] or start>end")
            break
    last_a = norm(ref_a.split()[-1])
    idx = [i for i, w in enumerate(words) if norm(w["word"]) == last_a]
    if not idx:
        fails.append(f"{label}: last word of clip A '{last_a}' not in the hypothesis")
    else:
        i = idx[0]
        ea = words[i]["end"]
        sb = words[i + 1]["start"] if i + 1 < len(words) else None
        print(f"{label}: '{words[i]['word']}' ends {ea:.3f} (clip A ends {dur_a:.3f}); next word starts {sb} "
              f"(clip B starts {dur_a + GAP:.3f})")
        if ea > dur_a + TOL:
            fails.append(f"{label}: last word of A ends {ea:.3f} > {dur_a + TOL:.3f}")
        if sb is None or sb < dur_a + GAP - TOL:
            fails.append(f"{label}: first word of B starts {sb} < {dur_a + GAP - TOL:.3f}")
    return fails


def main():
    if len(sys.argv) < 7:
        print(__doc__)
        return 2
    exe, model, mmproj, wa, wb, work = sys.argv[1:7]
    extra = sys.argv[7:]
    os.makedirs(work, exist_ok=True)
    a, b = read_pcm16(wa), read_pcm16(wb)
    ref_a = open(os.path.splitext(wa)[0] + ".txt").read().strip()
    ref_b = open(os.path.splitext(wb)[0] + ".txt").read().strip()
    n20 = noise(20, -50)
    anb = array.array('h', a) + noise(GAP, -50) + array.array('h', b)
    p_n20 = os.path.join(work, "noise20.wav"); write_pcm16(p_n20, n20)
    p_anb = os.path.join(work, "A_noise12_B.wav"); write_pcm16(p_anb, anb)
    dur_a, dur = len(a) / SR, len(anb) / SR
    fails = []

    # 0. positive control: the stimulus really makes this model hallucinate
    rc, w_off, _ = run(exe, model, mmproj, p_n20, work, "noise20_off", extra)
    if w_off is None:
        print("FAIL: flag-off run wrote no words (rc %d)" % rc)
        print("FAILED")
        return 1
    print(f"positive control: flag off, noise20 -> {len(w_off)} word(s): "
          f"{' '.join(w['word'] for w in w_off[:12])}{' ...' if len(w_off) > 12 else ''}")
    if len(w_off) < MIN_HALLUC:
        fails.append(f"positive control: flag off, noise20 gave only {len(w_off)} word(s) < {MIN_HALLUC}; "
                     "the stimulus does not trigger a hallucination, so checks 2-3 would prove nothing")
    neg = check_dropped(w_off, "negative-control")
    if not neg:
        fails.append("negative control PASSED: the drop check cannot tell a hallucination from silence")
    else:
        print("negative control caught as expected:", neg[0][:200])

    # 1+2. flag on, pure noise
    flag = ["--hallucination-silence-threshold", THRESH]
    rc, w_on, _ = run(exe, model, mmproj, p_n20, work, "noise20_on", flag + extra)
    if w_on is None:
        fails.append(f"llama-audio refused or failed with {' '.join(flag)} (rc {rc})")
    else:
        fails += check_dropped(w_on, "noise20")

    # 3. flag on, speech around noise: nothing real is dropped
    if w_on is not None:
        rc, w_anb, _ = run(exe, model, mmproj, p_anb, work, "A_noise12_B_on", flag + extra)
        if w_anb is None:
            fails.append(f"A_noise12_B: run failed (rc {rc})")
        else:
            for w in w_anb:
                print(f"  [{w['start']:7.3f} -> {w['end']:7.3f}] p={w['probability']:.3f} {w['word']}")
            fails += check_kept(w_anb, dur_a, dur, ref_a, ref_b, "A_noise12_B")

    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else "FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
