#!/usr/bin/env python3
# REWRITE-SAFE: writes only into a fresh directory given with --work (created here, must not exist yet); no existing path is opened for writing.
"""End-to-end check of `llama-audio --speak-stream`: the first audible sample must arrive after ONE sentence, not after the whole reply,
and the words that come out must still be the words of the reply.

How "first audio" is measured: the process is started and the output WAV is polled every 20 ms from the outside; the first audio is the
first poll at which the file exists, its data-chunk size field is non-zero and it holds at least 0.1 s of PCM. That is what a player tailing
the file would see, and it is the same instrument for both arms: the plain `--speak` path writes the file once, at the very end, so there the
first audio coincides with the end of the process.

  Both arms run the same two commands: plain --speak (reference) and --speak-stream. The binary from before the change does not know
  --speak-stream, so its run FAILS here; the binary with the change must PASS every criterion below.

Criteria (after): first_audio < 0.6 x total wall time of the streamed run, >= 2 chunks logged, the WAV is complete and valid, and the Voxtral
read-back of the streamed audio has WER vs the generated reply <= 0.30 and <= (WER of the plain --speak audio) + 0.10.
Negative controls: the WER matcher must fail on unrelated text; the TTFA criterion must fail on the plain --speak arm.

usage: test-speak-stream-cli.py --arm before|after --bin llama-audio[.exe] --models-dir DIR --work NEWDIR [--threads N]
"""
import argparse, os, re, struct, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_wer  # noqa: E402  shared WER + English normaliser

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
LLM = "qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf"
TTS = "audio/OuteTTS-0.2-500M-Q8_0.gguf"
VOC = "audio/WavTokenizer-Large-75-F16.gguf"
ASR_LLM = "audio/Voxtral-Mini-3B-2507-Q4_K_M.gguf"
ASR_PROJ = "audio/mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf"
PROMPT = ("<|im_start|>user\nWrite exactly three short sentences about the sea.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def norm(s):
    return asr_wer.normalise(s)


def wer(ref, hyp):
    return asr_wer.wer(ref, hyp)


def wav_state(path):
    """(valid, seconds of PCM) of a possibly still-growing WAV: header present, sizes consistent with the bytes on disk"""
    try:
        with open(path, "rb") as f:
            b = f.read()
    except OSError:
        return False, 0.0
    if len(b) < 44 or b[0:4] != b"RIFF" or b[8:12] != b"WAVE":
        return False, 0.0
    riff, data = struct.unpack("<I", b[4:8])[0], struct.unpack("<I", b[40:44])[0]
    sr = struct.unpack("<I", b[24:28])[0]
    ok = data > 0 and len(b) - 44 >= data and riff == 36 + data
    return ok, (data / 2.0 / sr) if (ok and sr) else 0.0


def run_timed(argv, out_wav, log_dir, tag):
    """start, poll the output file, return (rc, total_s, first_audio_s or None, stdout text, stderr text)"""
    so, se = os.path.join(log_dir, tag + ".out"), os.path.join(log_dir, tag + ".err")
    t0 = time.time()
    first = None
    with open(so, "w", encoding="utf-8", errors="replace") as fo, open(se, "w", encoding="utf-8", errors="replace") as fe:
        p = subprocess.Popen(argv, stdout=fo, stderr=fe, creationflags=CREATE_NO_WINDOW)
        while p.poll() is None:
            if first is None:
                ok, secs = wav_state(out_wav)
                if ok and secs >= 0.1:
                    first = time.time() - t0
            time.sleep(0.02)
        total = time.time() - t0
    if first is None:
        ok, secs = wav_state(out_wav)
        if ok and secs >= 0.1:
            first = total
    with open(so, errors="replace") as f:
        out = f.read()
    with open(se, errors="replace") as f:
        err = f.read()
    return p.returncode, total, first, out, err


def transcribe(binp, models, wav, log_dir, tag, threads):
    argv = [binp, "--task", "asr", "-m", os.path.join(models, ASR_LLM), "--mmproj", os.path.join(models, ASR_PROJ), "--audio", wav,
            "--asr-prompt", "Transcribe this audio verbatim.", "-t", str(threads), "-n", "200", "--temp", "0"]
    rc, total, _, out, err = run_timed(argv, os.path.join(log_dir, "none.wav"), log_dir, tag)
    m = re.search(r"transcript:\s*(.*)$", out + "\n" + err, re.M)
    return (m.group(1).strip() if (rc == 0 and m) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["before", "after"])
    ap.add_argument("--bin", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--asr-bin", default=None, help="binary used for the read-back (default: --bin)")
    a = ap.parse_args()
    os.makedirs(a.work)   # must be new
    fails = []
    rows = []

    def row(name, ok, detail):
        rows.append((name, ok, detail))
        print("[%s] %-34s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
        if not ok:
            fails.append(name)

    # negative control of the matcher: unrelated words must score badly
    row("matcher-negative-control", wer("the sea is wide and blue", "stock prices fell on tuesday") > 0.8, "unrelated text WER > 0.8")

    base = [a.bin, "--task", "chat", "-m", os.path.join(a.models_dir, LLM), "--tts-model", os.path.join(a.models_dir, TTS),
            "--vocoder-model", os.path.join(a.models_dir, VOC), "-p", PROMPT, "-t", str(a.threads), "-n", "768", "--temp", "0",
            "--seed", "42", "--tts-normalize", "--metrics"]
    # reference: the existing behaviour, one WAV written at the end
    ref_wav = os.path.join(a.work, "plain.wav")
    rc, total_p, first_p, out_p, err_p = run_timed(base + ["--speak", "-o", ref_wav], ref_wav, a.work, "plain")
    row("plain --speak runs", rc == 0 and first_p is not None, "rc %d total %.1fs first-audio %s" % (rc, total_p, "%.1fs" % first_p if first_p else "none"))
    reply = out_p.strip()
    row("reply was generated", len(norm(reply)) >= 12, "%d words: %r" % (len(norm(reply)), reply[:110]))
    row("TTFA criterion FAILS on plain --speak (control)", first_p is not None and not (first_p < 0.6 * total_p),
        "first %.1fs vs 0.6 x total %.1fs" % (first_p or -1, 0.6 * total_p))

    if True:   # both arms drive the SAME streamed command: the before binary does not know --speak-stream and must fail it
        st_wav = os.path.join(a.work, "stream.wav")
        rc, total_s, first_s, out_s, err_s = run_timed(base + ["--speak-stream", "-o", st_wav], st_wav, a.work, "stream")
        chunks = re.findall(r"speak-stream: chunk (\d+)", err_s + out_s)
        row("--speak-stream runs", rc == 0, "rc %d total %.1fs" % (rc, total_s))
        row("same reply text (temp 0)", out_s.strip() == reply, "stream reply == plain reply: %s" % (out_s.strip() == reply))
        row(">= 2 chunks spoken", len(chunks) >= 2, "%d chunk log lines" % len(chunks))
        ok, secs = wav_state(st_wav)
        row("final WAV valid", ok and secs > 1.0, "%.2f s of PCM" % secs)
        row("TTFA < 0.6 x total", first_s is not None and first_s < 0.6 * total_s,
            "first audio %.1fs, total %.1fs (plain --speak: first %.1fs, total %.1fs)" % (first_s or -1, total_s, first_p or -1, total_p))
        if rc != 0:
            print("RESULT: FAILED (%d failed)" % len(fails))
            return 1
        asr_bin = a.asr_bin or a.bin
        t_plain = transcribe(asr_bin, a.models_dir, ref_wav, a.work, "asr_plain", a.threads)
        t_stream = transcribe(asr_bin, a.models_dir, st_wav, a.work, "asr_stream", a.threads)
        row("read-back transcripts exist", t_plain is not None and t_stream is not None, "plain %r | stream %r" % (t_plain and t_plain[:70], t_stream and t_stream[:70]))
        if t_plain is not None and t_stream is not None:
            wp, ws = wer(reply, t_plain), wer(reply, t_stream)
            row("stream words == reply words", ws <= 0.30 and ws <= wp + 0.10, "WER vs reply: stream %.1f%%, plain %.1f%% (budget 30%% and plain+10)" % (100 * ws, 100 * wp))

    print("RESULT: %s (%d failed)" % ("OK" if not fails else "FAILED", len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
