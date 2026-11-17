#!/usr/bin/env python3
# REWRITE-SAFE: this test never rewrites a data store. It READS two read-only
# fixture files (fixtures/*.wav, fixtures/*.txt) and writes ONLY into a fresh
# tempfile.mkdtemp() directory it creates itself. No existing path is opened for
# writing, no record is filtered, nothing is deleted.
"""Regression test for the unified-audio loop: TTS -> ASR -> assert the words came back.

WHY THIS EXISTS
---------------
On 2026-08-31 this fork's TTS produced audio that passed EVERY signal-shape check
we had -- spectrum 85% below 4 kHz, a 94 Hz voiced peak, 1% above 8 kHz, a clean
VAD segment, and hparams cross-checked against the GGUF metadata -- while speaking
ENTIRELY THE WRONG WORDS. It was real speech, well formed and correctly voiced,
just not the requested text. Three separate defects hid behind those passing checks:

  1. a hardcoded speaker preamble ported from llama.cpp b4404, which the model
     spoke INSTEAD of the user's text;
  2. a 0.25 s lead-silence blank that ate the first word of short clips;
  3. (on the ASR side) a missing chat template, which returned an empty transcript
     that presented exactly like a projector fault.

Shape evidence is blind to semantics. The only instrument that can read what audio
SAYS is a speech recogniser. This test wires our verified ASR back onto our own TTS
so that class of defect cannot return silently.

WHAT IT ASSERTS
---------------
  T5     NEGATIVE CONTROL (pure): the comparison itself must FAIL on the b4404
         preamble text. Runs FIRST -- a test that can only pass proves nothing.
  T1/T2  ASR pinned against LibriSpeech ground truth, so a regression in the ASR
         cannot mask a regression in the TTS. If the instrument drifts we learn it
         here, rather than from a confusing TTS failure.
  T3     TTS round trip: synthesise a known string, transcribe it, compare by WER.
  T4     the FIRST WORD survives -- defect 2 above passed every other check, and a
         WER budget alone can swallow one word of four.
  T6     NEGATIVE CONTROL (live, optional): the rejected OuteTTS 0.3 pairing must
         still produce text that does NOT match the request. This is the actual
         historical fixture -- 0.3 through identical code says unrelated words.

USAGE
-----
  python test_audio_roundtrip.py --bin-dir <build>/bin --models-dir <dir> [--quick]

Exit 0 = every executed check passed. Non-zero = at least one failed, or nothing ran.
Missing MODELS -> SKIP (reported, never silent). Missing BINARIES -> FAIL.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_wer  # noqa: E402  shared WER + English normaliser

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# The phrase the TTS is asked to say. Short on purpose: the first-word defect only
# appeared on a clip short enough for 0.25 s to matter.
TTS_TEXT = "The fork can speak."

# Verbatim from llama.cpp b4404 examples/tts/tts.cpp. This is what the model used to
# say instead of the request, and it is the negative-control fixture.
B4404_PREAMBLE = (
    "the overall package from just two people is pretty remarkable sure i have some "
    "critiques about some of the gameplay aspects but its still really enjoyable and "
    "it looks lovely"
)

VOXTRAL_LLM = "Voxtral-Mini-3B-2507-Q4_K_M.gguf"
VOXTRAL_MMPROJ = "mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf"
VOCODER = "WavTokenizer-Large-75-F16.gguf"
OUTETTS_GOOD = "OuteTTS-0.2-500M-Q8_0.gguf"
OUTETTS_BAD = "OuteTTS-0.3-500M-Q4_K_M.gguf"


# --------------------------------------------------------------------------- WER

def normalise(s):
    return asr_wer.normalise(s)     # shared Whisper-style English normaliser (examples/audio/tests/asr_wer.py)


def wer(ref, hyp):
    """Word error rate plus substitution / deletion / insertion counts."""
    r, h = normalise(ref), normalise(hyp)
    if not r:
        return (0.0, 0, 0, 0) if not h else (1.0, 0, 0, len(h))
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    i, j, S, D, I = len(r), len(h), 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if r[i - 1] == h[j - 1] else 1):
            if r[i - 1] != h[j - 1]:
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return d[len(r)][len(h)] / float(len(r)), S, D, I


# ------------------------------------------------------------------------ runner

class Ctx(object):
    def __init__(self, bin_dir, models_dir, threads, verbose):
        self.bin_dir = bin_dir
        self.models_dir = models_dir
        self.threads = str(threads)
        self.verbose = verbose
        self.tmp = tempfile.mkdtemp(prefix="audio_roundtrip_")

    def exe(self, name):
        p = os.path.join(self.bin_dir, name + (".exe" if os.name == "nt" else ""))
        return p if os.path.exists(p) else None

    def model(self, name):
        p = os.path.join(self.models_dir, name)
        return p if os.path.exists(p) else None

    def run(self, argv, log_name):
        log_path = os.path.join(self.tmp, log_name)
        t0 = time.time()
        with open(log_path, "w") as log:
            rc = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=CREATE_NO_WINDOW)
        dt = time.time() - t0
        with open(log_path, "r", errors="replace") as f:
            out = f.read()
        if self.verbose:
            sys.stderr.write("  [%s] rc=%d %.1fs\n" % (log_name, rc, dt))
        return rc, out, dt


def asr(ctx, wav, tag, n_predict=160):
    """Transcribe wav with Voxtral. Returns the transcript string, or None."""
    argv = [
        ctx.exe("llama-audio"), "--task", "asr",
        "-m", ctx.model(VOXTRAL_LLM),
        "--mmproj", ctx.model(VOXTRAL_MMPROJ),
        "--audio", wav,
        "--asr-prompt", "Transcribe this audio verbatim.",
        "-t", ctx.threads, "-n", str(n_predict), "--temp", "0",
    ]
    rc, out, _ = ctx.run(argv, "asr_%s.log" % tag)
    if rc != 0:
        return None
    # NOT anchored with ^: the sampled text streams to stdout while the log line
    # goes elsewhere, so this marker can land mid-line. Anchoring it is how this
    # test first reported "no transcript" against a transcript that was present.
    m = re.search(r"main: transcript:\s*(.*)$", out, re.M)
    return m.group(1).strip() if m else None


def tts(ctx, text, out_wav, backbone, tag, n_predict=768):
    """Synthesise text. Returns (ok, seconds_of_audio)."""
    argv = [
        ctx.exe("llama-tts"),
        "-m", ctx.model(backbone),
        "--vocoder-model", ctx.model(VOCODER),
        "-p", text, "-o", out_wav,
        "-t", ctx.threads, "-n", str(n_predict),
        "--tts-normalize", "--seed", "42",
    ]
    rc, out, _ = ctx.run(argv, "tts_%s.log" % tag)
    if rc != 0:
        return False, 0.0
    m = re.search(r"-- ([0-9.]+) s of audio", out)
    return True, (float(m.group(1)) if m else 0.0)


# ------------------------------------------------------------------------- tests

class Results(object):
    def __init__(self):
        self.rows = []

    def add(self, name, status, detail, exercised=False):
        """exercised=True means this row is backed by an ACTUAL binary invocation.

        The distinction is load-bearing, not bookkeeping. MEASURED 2026-09-02 with every model
        absent and both binaries present, this file printed "1 checks ran, 0 failed" and exited
        0 -- a PASS -- having synthesised nothing and transcribed nothing. T5 is a pure string
        comparison that needs no model, so it always runs and always counted toward ran(),
        which made the exit-2 branch below ("nothing ran ... that is a SKIP, not a pass")
        UNREACHABLE. The docstring promised the behaviour; the accounting defeated it. That
        matters now that this harness is a merge gate in _build_merge_verify.bat: on a host
        without the GGUFs the gate would have reported PASS.
        """
        self.rows.append((name, status, detail, exercised))
        print("[%s] %-32s %s" % (status, name, detail))
        sys.stdout.flush()

    def failed(self):
        return [r for r in self.rows if r[1] == "FAIL"]

    def ran(self):
        return [r for r in self.rows if r[1] != "SKIP"]

    def exercised_ran(self):
        """Rows that actually drove a binary. Zero of these means nothing was PROVEN."""
        return [r for r in self.rows if r[1] != "SKIP" and r[3]]


def read_fixture(stem):
    wav = os.path.join(FIXTURES, stem + ".wav")
    txt = os.path.join(FIXTURES, stem + ".txt")
    if not (os.path.exists(wav) and os.path.exists(txt)):
        return None, None
    with open(txt, "r") as f:
        return wav, f.read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--quick", action="store_true",
                    help="skip the long LibriSpeech clip and the live 0.3 negative control")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    ctx = Ctx(args.bin_dir, args.models_dir, args.threads, args.verbose)
    res = Results()

    print("audio round-trip regression")
    print("  bin    : %s" % ctx.bin_dir)
    print("  models : %s" % ctx.models_dir)
    print("  logs   : %s" % ctx.tmp)
    print("")

    # binaries are a hard requirement -- absence is a failure, not a skip
    for b in ("llama-audio", "llama-tts"):
        if ctx.exe(b) is None:
            res.add("binary:" + b, "FAIL", "not found in --bin-dir")
    if res.failed():
        print("")
        print("FAIL: binaries missing; build llama-audio and llama-tts first.")
        return 1

    have_asr = bool(ctx.model(VOXTRAL_LLM) and ctx.model(VOXTRAL_MMPROJ))
    have_voc = bool(ctx.model(VOCODER))
    have_good = bool(ctx.model(OUTETTS_GOOD))
    have_bad = bool(ctx.model(OUTETTS_BAD))

    # ---- T5 runs FIRST: a pure control needing no models. If the comparison
    #      cannot FAIL, every other result in this file is worthless.
    rate, _, _, _ = wer(TTS_TEXT, B4404_PREAMBLE)
    res.add("T5 matcher-negative-control", "PASS" if rate > 0.5 else "FAIL",
            "preamble vs request: WER %.0f%% (want >50%%), so the comparison can fail"
            % (100 * rate))

    # ---- T1/T2: pin the instrument against ground truth
    # Whether the INSTRUMENT is trustworthy this run. T6 must not draw a
    # conclusion from an ASR that is not working -- see its comment below.
    asr_pin_ok = [True]

    pin_cases = (
        ("T1 asr-pin-short", "librispeech_test_clean_6930-75918-0000", 0.00, False),
        ("T2 asr-pin-long", "librispeech_test_clean_6930-75918-0001", 0.10, True),
    )
    for tag, stem, budget, quick_skip in pin_cases:
        if not have_asr:
            res.add(tag, "SKIP", "Voxtral model or mmproj not in --models-dir")
            continue
        if quick_skip and args.quick:
            res.add(tag, "SKIP", "--quick")
            continue
        wav, truth = read_fixture(stem)
        if wav is None:
            res.add(tag, "SKIP", "fixture missing: %s" % stem)
            continue
        got = asr(ctx, wav, stem[-4:])
        if got is None:
            asr_pin_ok[0] = False
            res.add(tag, "FAIL", "ASR produced no transcript line", exercised=True)
            continue
        rate, S, D, I = wer(truth, got)
        ok = rate <= budget
        if not ok:
            asr_pin_ok[0] = False
        res.add(tag, "PASS" if ok else "FAIL",
                "WER %.1f%% (budget %.0f%%) s=%d d=%d i=%d | %r"
                % (100 * rate, 100 * budget, S, D, I, got[:64]), exercised=True)

    # ---- T3/T4: the round trip itself
    if not (have_asr and have_voc and have_good):
        why = "need Voxtral + WavTokenizer + OuteTTS 0.2"
        res.add("T3 tts-roundtrip", "SKIP", why)
        res.add("T4 first-word-survives", "SKIP", why)
    else:
        wav = os.path.join(ctx.tmp, "roundtrip.wav")
        ok, secs = tts(ctx, TTS_TEXT, wav, OUTETTS_GOOD, "rt")
        if not ok:
            res.add("T3 tts-roundtrip", "FAIL", "llama-tts failed", exercised=True)
            res.add("T4 first-word-survives", "FAIL", "llama-tts failed", exercised=True)
        else:
            got = asr(ctx, wav, "rt", n_predict=64)
            if got is None:
                res.add("T3 tts-roundtrip", "FAIL", "ASR produced no transcript line",
                        exercised=True)
                res.add("T4 first-word-survives", "FAIL", "no transcript", exercised=True)
            else:
                rate, S, D, I = wer(TTS_TEXT, got)
                budget = 0.25   # tolerates sampling drift, not wrong words
                res.add("T3 tts-roundtrip", "PASS" if rate <= budget else "FAIL",
                        "%.2fs audio, WER %.1f%% (budget %.0f%%) | %r"
                        % (secs, 100 * rate, 100 * budget, got[:64]), exercised=True)

                # T4 is separate ON PURPOSE: losing the leading article is only one
                # word of four, which a WER budget alone can swallow.
                want_first = normalise(TTS_TEXT)[0]
                got_words = normalise(got)
                got_first = got_words[0] if got_words else "<empty>"
                res.add("T4 first-word-survives",
                        "PASS" if got_first == want_first else "FAIL",
                        "expected first word %r, got %r" % (want_first, got_first),
                        exercised=True)

    # ---- T6: the live negative control, using the historical fixture
    if args.quick:
        res.add("T6 live-negative-control", "SKIP", "--quick")
    elif not (have_asr and have_voc and have_bad):
        res.add("T6 live-negative-control", "SKIP",
                "OuteTTS 0.3 (the rejected pairing) not present")
    else:
        wav = os.path.join(ctx.tmp, "neg03.wav")
        ok, _ = tts(ctx, TTS_TEXT, wav, OUTETTS_BAD, "neg03")
        if not ok:
            res.add("T6 live-negative-control", "SKIP", "0.3 synthesis did not run")
        else:
            got = asr(ctx, wav, "neg03", n_predict=96)
            if got is None:
                # An empty transcript is NOT evidence of a non-match unless the
                # instrument is known good this run: "0.3 said the wrong words" and
                # "the ASR is not working" produce the identical empty string. This
                # test PASSED for that wrong reason once; it does not any more.
                if asr_pin_ok[0]:
                    res.add("T6 live-negative-control", "PASS",
                            "0.3 pairing produced no recoverable text, and the ASR pin "
                            "passed this run, so the non-match is real", exercised=True)
                else:
                    res.add("T6 live-negative-control", "FAIL",
                            "no transcript AND the ASR pin failed -- cannot tell a bad "
                            "pairing from a bad instrument", exercised=True)
            else:
                rate, _, _, _ = wer(TTS_TEXT, got)
                # The point is that a KNOWN-BAD pairing is still DETECTED as bad.
                # If this ever "matches", either 0.3 changed or the check stopped
                # checking -- either way, look before trusting T3.
                res.add("T6 live-negative-control", "PASS" if rate > 0.5 else "FAIL",
                        "0.3 pairing WER %.0f%% (want >50%%) | %r" % (100 * rate, got[:56]),
                        exercised=True)

    print("")
    n_fail = len(res.failed())
    n_ran = len(res.ran())
    n_real = len(res.exercised_ran())
    print("%d checks ran, %d failed, %d exercised a binary" % (n_ran, n_fail, n_real))
    if n_fail:
        print("logs kept at %s" % ctx.tmp)
        return 1
    # The count that decides a PASS is n_real, NOT n_ran. n_ran includes T5, a pure string
    # comparison that runs with no models and no binaries -- so gating on it meant a run that
    # synthesised nothing and transcribed nothing exited 0. Measured 2026-09-02: models absent
    # gave "1 checks ran, 0 failed" and rc=0. This branch was unreachable until now.
    if n_real == 0:
        print("nothing but pure controls ran -- no audio was synthesised and none transcribed.")
        print("That is a SKIP, not a pass. (models absent from --models-dir?)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
