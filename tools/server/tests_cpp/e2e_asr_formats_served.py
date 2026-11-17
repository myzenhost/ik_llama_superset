# REWRITE-SAFE: this test writes only an in-memory WAV and server logs under temp/logs; it reads no data store.
"""E2E served checks for /v1/audio/transcriptions response formats (text, srt, vtt, verbose_json).

Usage: pythonw e2e_asr_formats_served.py <new llama-server.exe> <baseline llama-server.exe | -> <model.gguf> <mmproj.gguf> [base-port]
       (the caller redirects stdout to a file; exit 0 = every check passed)
The baseline binary is a build WITHOUT the form-field fix of convert_transcriptions_to_chatcmpl (canonical copies every form field into the chat
body, so response_format=srt reaches the chat parser, which wants an object): fail-before is that 400 "response_format must be an object" on
every non-json format. Pass "-" to skip the baseline run.

Row: S69-A-PICK-TRANSCRIPTIONS-ENDPOINT-TEXT-SRT-VTT-VERBOSE-JSON-SEGMENTS-AND-STEREO-DIARIZE-2026-10-05-01
Model: Voxtral-Mini-3B (CPU). The audio is a synthetic 3 s tone, so the text is whatever the model makes of it; the checks are about the
shape of each format, not about recognition quality.
"""
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXE, EXE_OLD, MODEL, MMPROJ = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BASE = int(sys.argv[5]) if len(sys.argv) > 5 else 18230
LOGDIR = "tmp/logs"
os.makedirs(LOGDIR, exist_ok=True)
fails = []


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def wav_bytes(seconds=3.0, rate=16000):
    n = int(seconds * rate)
    pcm = b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 220 * i / rate) + 3000 * math.sin(2 * math.pi * 1300 * i / rate))) for i in range(n))
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm))
    return hdr + pcm


def post_form(port, fields, wav, timeout=900):
    boundary = "----e2eboundary7d3"
    parts = []
    for k, v in fields.items():
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (boundary, k, v)).encode())
    parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n" % boundary).encode() + wav + b"\r\n")
    parts.append(("--%s--\r\n" % boundary).encode())
    req = urllib.request.Request("http://127.0.0.1:%d/v1/audio/transcriptions" % port, data=b"".join(parts), method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode("utf-8", "replace")


class Srv:
    def __init__(self, exe, port, tag):
        self.port = port
        self.log = os.path.join(LOGDIR, "asrfmt_%s_%d.log" % (tag, port))
        self.f = open(self.log, "wb")
        self.p = subprocess.Popen([exe, "--host", "127.0.0.1", "--port", str(port), "-m", MODEL, "--mmproj", MMPROJ, "-c", "4096", "-np", "1", "-t", "8",
                                   "--no-warmup", "--no-mmproj-offload"], stdout=self.f, stderr=subprocess.STDOUT, creationflags=0x08000000)

    def wait_ready(self, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                raise RuntimeError("server exited rc=%s, log %s" % (self.p.returncode, self.log))
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % self.port, timeout=3) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("not ready, log %s" % self.log)

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(20)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self.f.close()


WAV = wav_bytes()
F = {"model": "x", "max_tokens": "24", "temperature": "0"}

# ------------------------------------------------------------------------------------------ B: the baseline build
if EXE_OLD != "-":
    s = Srv(EXE_OLD, BASE, "old")
    try:
        s.wait_ready()
        st, hd, tx = post_form(BASE, dict(F, response_format="srt"), WAV)
        check("B fail-before: srt on the baseline reaches the chat parser and is refused ('response_format must be an object')", st == 400 and "must be an object" in tx, "status=%s %s" % (st, tx[:140]))
    finally:
        s.stop()

# ------------------------------------------------------------------------------------------ N: this build
s = Srv(EXE, BASE + 1, "new")
P = BASE + 1
try:
    s.wait_ready()
    st, hd, tx = post_form(P, dict(F, response_format="json"), WAV)
    j = json.loads(tx) if st == 200 else {}
    check("N0 json: shape unchanged (type transcript.text.done, text; no avg_logprob unless verbose_json)", st == 200 and j.get("type") == "transcript.text.done" and "text" in j and "avg_logprob" not in j, "status=%s keys=%s" % (st, sorted(j)))
    text_json = j.get("text", "")

    st, hd, tx = post_form(P, dict(F, response_format="text"), WAV)
    check("N1 text: plain text body, text/plain", st == 200 and hd.get("Content-Type", "").startswith("text/plain") and not tx.lstrip().startswith("{"), "status=%s ct=%s" % (st, hd.get("Content-Type")))

    st, hd, tx = post_form(P, dict(F, response_format="srt"), WAV)
    ok_srt = st == 200 and hd.get("Content-Type", "").startswith("application/x-subrip")
    cue = re.compile(r"^(\d+)\n(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)\n(.+)\n\n", re.M)
    cues = cue.findall(tx)
    check("N2 srt: numbered cue(s) in SRT syntax, times in 'HH:MM:SS,mmm'", ok_srt and (len(cues) >= 1 or tx == ""), "status=%s cues=%s" % (st, cues[:2]))
    if cues:
        check("N2 srt: the cues span the audio (first starts at 0, last ends at the 3 s of the decoded audio)", cues[0][1] == "00:00:00,000" and cues[-1][2] == "00:00:03,000", "%s --> %s" % (cues[0][1], cues[-1][2]))

    st, hd, tx = post_form(P, dict(F, response_format="vtt"), WAV)
    check("N3 vtt: WEBVTT header, '.' millisecond separator, text/vtt", st == 200 and tx.startswith("WEBVTT\n\n") and hd.get("Content-Type", "").startswith("text/vtt") and
          (tx == "WEBVTT\n\n" or (re.search(r"^00:00:00\.000 --> ", tx, re.M) and re.search(r" --> 00:00:03\.000$", tx, re.M))), "status=%s head=%r" % (st, tx[:60]))

    st, hd, tx = post_form(P, dict(F, response_format="verbose_json", language="en"), WAV)
    j = json.loads(tx) if st == 200 else {}
    need = {"task", "language", "duration", "text", "segments"}
    check("N4 verbose_json: OpenAI field set (task, language, duration, text, segments)", st == 200 and need <= set(j), "status=%s keys=%s" % (st, sorted(j)))
    check("N4 verbose_json: language echoes the request, duration comes from the decoded audio", j.get("language") == "en" and abs((j.get("duration") or 0) - 3.0) < 1e-6, "lang=%s dur=%s" % (j.get("language"), j.get("duration")))
    segs = j.get("segments", [])
    if segs:
        sg = segs[0]
        check("N4 verbose_json: segment fields; the first starts at 0, the last ends at 3 s, avg_logprob is a negative-or-zero number (mean log-probability)",
              {"id", "seek", "start", "end", "text", "tokens", "temperature", "avg_logprob", "compression_ratio", "no_speech_prob"} <= set(sg) and sg["start"] == 0.0 and abs(segs[-1]["end"] - 3.0) < 1e-6 and
              isinstance(sg["avg_logprob"], float) and sg["avg_logprob"] <= 0.0, json.dumps(sg)[:200])
        check("N4 verbose_json: says how the times were made ('interpolated' or 'model')", j.get("timestamps") in ("interpolated", "model"), "timestamps=%s" % j.get("timestamps"))
    else:
        check("N4 verbose_json: an empty transcript gives an empty segment list", j.get("text") == "", "text=%r" % j.get("text"))

    st, hd, tx = post_form(P, dict(F, response_format="ass"), WAV)
    check("N5 an unknown format is a 400 naming the supported ones", st == 400 and "srt" in tx and "verbose_json" in tx, "status=%s" % st)
    st, hd, tx = post_form(P, dict(F, response_format="srt", stream="true"), WAV)
    check("N6 srt with stream=true is a 400", st == 400, "status=%s %s" % (st, tx[:100]))
finally:
    s.stop()

print("\n%d failure(s)" % len(fails) if fails else "\nALL PASS", flush=True)
sys.exit(1 if fails else 0)
