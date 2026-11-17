#!/usr/bin/env python3
# REWRITE-SAFE: writes only fresh pid-suffixed logs into the workdir named on the command line
"""llama-audio --task stream (live sliding-window transcription, whisper.cpp examples/stream): checks on a real model.

usage: test-asr-stream-cli.py <llama-audio> <parakeet-tdt.gguf> <fixtures_dir> <workdir> <jfk.wav> <jfk_expected.txt>

The GGUF comes from examples/audio/convert-parakeet-tdt-to-gguf.py. fixtures_dir holds
librispeech_test_clean_6930-75918-0001.{wav,txt} (examples/audio/tests/fixtures).

Checks (each prints PASS/FAIL; any FAIL -> exit 1):
  1. jfk.wav, file source, lockstep (one 3 s step at a time, 10 s windows, 200 ms keep): at least 2 committed lines
     with increasing times, and the joined lines within 2 word errors of the reference (22 words). Lines join
     windows that overlap by the keep, so a word on a line boundary may be cut or doubled; 2 errors allows one.
  2. the 14.2 s LibriSpeech clip, same settings: WER of the joined lines <= 10%.
  3. jfk.wav played into the capture ring at 4x real time (the path a microphone uses): exits 0 and the audio it
     took plus the audio it reports as dropped add up to the whole file (11.0 s, within one 10 ms block).
  4. capture device, miniaudio's null backend (a real capture device with no hardware): 2 s requested -> exits 0,
     the callback delivered at least 1.8 s of samples, and the transcript of that silence is empty.
  5. --list-capture-devices --stream-capture null: exits 0 and lists one device.
Negative controls: one window covering the whole file (--stream-step 12000 --stream-length 12000) commits ONE
line (so the line count of check 1 comes from the sliding window, not from the file), and the WER of a hypothesis
with two words dropped is 2.
"""
import os
import re
import subprocess
import sys


def norm_words(s):
    return [w for w in (re.sub(r"[^a-z0-9']", "", t.lower()) for t in s.split()) if w]


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def run(exe, args, work, tag):
    """-> dict(rc, transcript, lines=[(t0, t1, text)], stats line fields)"""
    cmd = [exe] + args
    print(f"RUN [{tag}]:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log = os.path.join(work, f"{tag}_{os.getpid()}.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(p.stdout + "\n----- stderr -----\n" + p.stderr)
    r = {"rc": p.returncode, "transcript": None, "lines": [], "taken": None, "dropped": None, "captured": None,
         "stdout": p.stdout, "log": log}
    for line in p.stdout.splitlines():
        m = re.match(r"\[\s*([\d.]+) -->\s*([\d.]+)\] ?(.*)$", line)
        if m:
            r["lines"].append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
    for line in (p.stdout + "\n" + p.stderr).splitlines():
        m = re.search(r"run_stream_tdt: transcript: ?(.*)$", line)
        if m:
            r["transcript"] = m.group(1).strip()
        m = re.search(r"([\d.]+) s of audio taken, ([\d.]+) s dropped, captured (\d+) sample", line)
        if m:
            r["taken"], r["dropped"], r["captured"] = float(m.group(1)), float(m.group(2)), int(m.group(3))
        if "error" in line.lower() or "failed" in line.lower() or "cannot keep up" in line:
            print("  |", line[:300])
    print(f"  rc={r['rc']} lines={len(r['lines'])} transcript={r['transcript']!r} log={log}", flush=True)
    return r


def main():
    if len(sys.argv) != 7:
        print(__doc__)
        return 2
    exe, model, fx, work, jfk, jfk_txt = sys.argv[1:7]
    os.makedirs(work, exist_ok=True)
    fails = []

    def verdict(name, err):
        print(("PASS " if err is None else "FAIL ") + name + ("" if err is None else ": " + err), flush=True)
        if err is not None:
            fails.append(name)

    base = [exe, "--task", "stream", "--mmproj", model, "-t", "4"][1:]
    jfk_ref = open(jfk_txt, encoding="utf-8").read().strip()

    # 1. jfk lockstep
    r = run(exe, base + ["--audio", jfk], work, "jfk_lockstep")
    if r["rc"] != 0 or r["transcript"] is None:
        verdict("jfk lockstep ran", f"rc={r['rc']}")
    else:
        t0s = [l[0] for l in r["lines"]]
        verdict("jfk lockstep: >= 2 committed lines, times increasing",
                None if len(r["lines"]) >= 2 and t0s == sorted(t0s) and all(l[1] > l[0] for l in r["lines"])
                else f"lines {r['lines']}")
        e, n = wer(jfk_ref, r["transcript"])
        print(f"  jfk: {e} error(s) / {n} words")
        verdict("jfk lockstep: <= 2 word errors", None if e <= 2 else f"{e} errors: {r['transcript']!r}")
    one = run(exe, base + ["--audio", jfk, "--stream-step", "12000", "--stream-length", "12000"], work, "jfk_one_window")
    verdict("negative control: one 12 s window commits one line",
            None if one["rc"] == 0 and len(one["lines"]) == 1 else f"rc={one['rc']} lines={one['lines']}")
    e_ctrl, _ = wer(jfk_ref, " ".join(jfk_ref.split()[2:]))
    verdict("negative control: WER sees two dropped words", None if e_ctrl == 2 else f"WER errors {e_ctrl}")

    # 2. LibriSpeech B lockstep
    b = os.path.join(fx, "librispeech_test_clean_6930-75918-0001")
    ref_b = open(b + ".txt", encoding="utf-8").read().strip()
    r = run(exe, base + ["--audio", b + ".wav"], work, "ls1_lockstep")
    if r["rc"] != 0 or r["transcript"] is None:
        verdict("librispeech lockstep ran", f"rc={r['rc']}")
    else:
        e, n = wer(ref_b, r["transcript"])
        print(f"  ls1: {e} error(s) / {n} words, {len(r['lines'])} line(s)")
        verdict("librispeech lockstep WER <= 10%", None if 100.0 * e / n <= 10.0 else f"{e}/{n}: {r['transcript']!r}")

    # 3. paced through the capture ring
    r = run(exe, base + ["--audio", jfk, "--stream-speed", "4"], work, "jfk_paced")
    if r["rc"] != 0 or r["taken"] is None:
        verdict("jfk paced ran", f"rc={r['rc']}")
    else:
        tot = r["taken"] + r["dropped"]
        print(f"  paced: taken {r['taken']:.3f} s + dropped {r['dropped']:.3f} s = {tot:.3f} s, captured {r['captured']}")
        verdict("jfk paced: taken + dropped = the whole file", None if abs(tot - 11.0) <= 0.011 else f"{tot:.3f} s")

    # 4. capture device, null backend
    r = run(exe, base + ["--stream-capture", "null", "--stream-duration", "2"], work, "null_capture")
    if r["rc"] != 0 or r["captured"] is None:
        verdict("null capture ran", f"rc={r['rc']}")
    else:
        print(f"  null: captured {r['captured']} samples, transcript {r['transcript']!r}")
        verdict("null capture: >= 1.8 s delivered by the device callback", None if r["captured"] >= int(1.8 * 16000) else f"{r['captured']}")
        verdict("null capture: silence transcribes to nothing", None if r["transcript"] == "" else repr(r["transcript"]))

    # 5. device listing
    r = run(exe, ["--list-capture-devices", "--stream-capture", "null"], work, "list_null")
    verdict("list null capture devices", None if r["rc"] == 0 and "capture devices (null): 1" in r["stdout"] else f"rc={r['rc']}")

    print(("FAILED: " + ", ".join(fails)) if fails else "OK", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
