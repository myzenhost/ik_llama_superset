#!/usr/bin/env python3
"""llama-audio --word-timestamps --words-out on a random-weight audio LLM: the capture path runs and the file is sane.

usage: test-asr-words-tiny-cli.py <llama-audio> <speech.wav (mono 16 kHz PCM16)> <tree-root> <workdir>

examples/audio/audio.cpp captures the audio-position attention rows of the LLM decode, aligns them (asr-align.h) and
writes the words as JSON. No gate step ran that binary, so a break in the capture (a callback that never fires, a
flash-attention default that hides the rows, a window offset error) could merge unseen. The random-weight model
(scripts/gen-tiny-lora-test-assets.py 64-dim LLM + scripts/gen-tiny-audio-mmproj.py qwen2a projector) decodes
gibberish by design; this test does not judge the TEXT, it judges the SHAPE of what the capture produced:
  1. rc 0 and --words-out wrote a JSON file with a "words" list (the log line "word-timestamps: window" is printed)
  2. at least one word (an empty list = the capture or the aligner saw nothing)
  3. every word has word/start/end, 0 <= start <= end <= clip length (+ rounding), starts non-decreasing
Negative control: the same checks on a copy whose last word is moved past the clip end must fail, so a check that
passes anything is caught.
Exit 77 (ctest SKIPPED) when numpy, the clip or a generator is missing.
"""
import json
import os
import subprocess
import sys
import wave

SKIP = 77


def check(words, dur, label):
    fails = []
    if not words:
        fails.append(f"{label}: no words")
        return fails
    prev = -1.0
    for w in words:
        if not all(k in w for k in ("word", "start", "end")):
            fails.append(f"{label}: word without word/start/end: {w}")
            break
        if not (0.0 <= w["start"] <= w["end"] <= dur + 1e-3):
            fails.append(f"{label}: word {w} outside [0,{dur:.3f}] or start>end")
            break
        if w["start"] < prev - 1e-6:
            fails.append(f"{label}: starts decrease at {w}")
            break
        prev = w["start"]
    return fails


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **kw)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    exe, wav, root, work = sys.argv[1:5]
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("numpy missing: SKIPPED")
        return SKIP
    gen_llm = os.path.join(root, "scripts", "gen-tiny-lora-test-assets.py")
    gen_mm = os.path.join(root, "scripts", "gen-tiny-audio-mmproj.py")
    if not (os.path.exists(wav) and os.path.exists(gen_llm) and os.path.exists(gen_mm)):
        print("fixture or generator missing: SKIPPED")
        return SKIP
    os.makedirs(work, exist_ok=True)
    for g in (gen_llm, gen_mm):
        p = run([sys.executable, g, work])
        if p.returncode != 0:
            print(p.stdout, p.stderr)
            print("FAIL: asset generator failed:", g)
            return 1
    model, mmproj = os.path.join(work, "tiny-llama.gguf"), os.path.join(work, "mmproj-tiny-audio.gguf")
    with wave.open(wav, "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
    words_path = os.path.join(work, f"words_{os.getpid()}.json")  # fresh name: an old file cannot satisfy check 1
    cmd = [exe, "--task", "asr", "-m", model, "--mmproj", mmproj, "--audio", wav, "--audio-chunk-mode", "none",
           "--words-out", words_path, "--temp", "0", "-c", "4096", "-n", "24"]
    print("RUN:", " ".join(cmd), flush=True)
    p = run(cmd)
    with open(os.path.join(work, "llama-audio.log"), "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n----- stderr -----\n" + p.stderr)
    for line in (p.stdout + p.stderr).splitlines():
        if "word-timestamps" in line or "error" in line.lower():
            print("  |", line)
    print(f"rc={p.returncode} clip={dur:.3f}s")
    if p.returncode != 0 or not os.path.exists(words_path):
        print("FAIL: llama-audio rc %d or --words-out not written" % p.returncode)
        return 1
    # random-weight tokens are arbitrary bytes and audio.cpp writes them raw (MEASURED 2026-11-16: byte 0x95 at offset 211
    # made a strict utf-8 json.load raise), so decode with replacement; the times are what is judged here
    words = json.loads(open(words_path, "rb").read().decode("utf-8", "replace")).get("words", [])
    print(f"{len(words)} word(s)")
    for w in words[:12]:
        print(f"  [{w['start']:7.3f} -> {w['end']:7.3f}] {w['word']!r}")
    fails = check(words, dur, "words")
    bad = [dict(w) for w in words]
    if bad:
        bad[-1]["end"] = dur + 5.0
    if not check(bad, dur, "negative-control"):
        fails.append("negative control PASSED: the checks cannot tell a word past the clip end from a real one")
    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else "FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
