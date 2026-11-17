# -*- coding: utf-8 -*-
# DISABLE-REVERSIBLE: nothing is disabled by this file; "--diffusion-eb off" below is a per-invocation CLI test flag
# (RESEARCH-DONE: upstream/unsloth-llama.cpp/examples/diffusion/diffusion-cli.cpp, the flag's own definition)
"""End-to-end CPU check of the llama-diffusion subsystem + DiffusionGemma visual server on a random-weight tiny model.

Usage: python run_check.py <worktree-root> <build-dir> <work-dir>
  <build-dir>/bin must hold llama-diffusion-eb-selftest, llama-diffusion-cli, llama-diffusion-gemma-visual-server.
Run it inside the host slot (it starts model processes). Exit 0 = every check passed.

Checks (each is a FAIL line if violated; the control for each is named):
  selftest      examples/diffusion/diffusion-eb-selftest.cpp: determinism, serial-reference equality (+ other-seed
                control), kv-cache == unified, device-SC path, adaptive stop, suppress_mask_token (+ control)
  eb-keys       the diffusion.eb_* values the fork's gguf-py wrote reach the CLI's effective parameters
                (control: a model written with --no-eb-keys shows the reference defaults instead)
  visual-server stdin/stdout protocol: READY, F frames, C commit, STATS, DONE; determinism across two identical
                requests (control: another seed differs); ERR toolong / ERR badreq paths; 20 back-to-back seeds;
                clean QUIT; and a model whose lm_head forces lone UTF-8 continuation bytes (control: the F frames
                really contain U+FFFD, i.e. the invalid-UTF-8 case was exercised)
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

root, build, work = (Path(sys.argv[i]) for i in (1, 2, 3))
bin_dir = build / "bin"
work.mkdir(parents=True, exist_ok=True)
EXE = ".exe" if os.name == "nt" else ""
fails = []
passes = 0
CNW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def check(cond, msg):
    global passes
    if cond:
        passes += 1
    else:
        fails.append(msg)
        print("FAIL:", msg, flush=True)


def run(cmd, timeout=600, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([str(c) for c in cmd], capture_output=True, timeout=timeout, env=e, creationflags=CNW)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def make_model(name, *extra):
    out = work / name
    rc, so, se = run([sys.executable, root / "tests/diffusion-gemma-check/make_tiny_diffusion_gemma_gguf.py", root, out, *extra])
    check(rc == 0 and out.exists(), "generator failed for %s: %s %s" % (name, so, se))
    return out


model = make_model("tiny_dg.gguf")
model_nokeys = make_model("tiny_dg_nokeys.gguf", "--no-eb-keys")
model_badutf8 = make_model("tiny_dg_badutf8.gguf", "--force-invalid-utf8")

# ---- selftest ----------------------------------------------------------------------------------------------
selftest = bin_dir / ("llama-diffusion-eb-selftest" + EXE)
rc, so, se = run([selftest, model], timeout=900)
print(so, flush=True)
check(rc == 0 and "RESULT pass=" in so and " fail=0" in so, "selftest rc=%d\n%s\n%s" % (rc, so[-2000:], se[-2000:]))
if rc != 0 and "RESULT pass=" not in so:
    # the process died (assert/abort) before finishing: rerun without the device-SC section so the other
    # sections still report; the abort above stays recorded as a failure.
    rc2, so2, se2 = run([selftest, model, "--skip-devsc"], timeout=900)
    print("--- rerun with --skip-devsc ---\n" + so2, flush=True)
    check(rc2 == 0 and " fail=0" in so2, "selftest --skip-devsc rc=%d\n%s\n%s" % (rc2, so2[-2000:], se2[-2000:]))

# ---- eb-keys via the CLI -------------------------------------------------------------------------------------
cli = bin_dir / ("llama-diffusion-cli" + EXE)
common = ["-p", "hello", "-n", "8", "-t", "4", "-ngl", "0", "--seed", "3", "-no-cnv"]
want = "diffusion_eb: max_steps=7 t=[0.250,0.750] entropy_bound=0.5000 stability=2 confidence=0.0125"
rc, so, se = run([cli, "-m", model, *common], timeout=600)
log = so + se
check(rc == 0, "cli rc=%d: %s" % (rc, log[-1500:]))
check(want in log, "eb keys from the GGUF did not reach the CLI parameters; wanted '%s' in log:\n%s" % (want, log[-2500:]))
check("total time:" in log and "entropy-bound" in log,
      "the CLI's reply/timing summary (printed through the modern output macro) is missing from its output:\n" + log[-1500:])
rc, so, se = run([cli, "-m", model_nokeys, *common], timeout=600)
log2 = so + se
check("diffusion_eb: max_steps=48 t=[0.400,0.800] entropy_bound=0.1000 stability=1 confidence=0.0050" in log2,
      "control (no eb keys) must show the reference defaults:\n" + log2[-2500:])
# CLI override beats the GGUF value
rc, so, se = run([cli, "-m", model, *common, "--diffusion-eb-max-steps", "3", "--diffusion-eb-t-min", "0.1"], timeout=600)
check("diffusion_eb: max_steps=3 t=[0.100,0.750]" in (so + se), "CLI override of eb parameters not honoured:\n" + (so + se)[-2000:])
# the masked path (forced with --diffusion-eb off) must also run to completion on a canvas model
rc, so, se = run([cli, "-m", model, *common, "--diffusion-eb", "off", "--diffusion-steps", "4"], timeout=600)
check(rc == 0, "cli --diffusion-eb off (masked path) rc=%d:\n%s" % (rc, (so + se)[-1500:]))

# ---- visual server --------------------------------------------------------------------------------------------
vs = bin_dir / ("llama-diffusion-gemma-visual-server" + EXE)


def visual_session(mdl, bad_utf8):
    """One visual-server process on `mdl`."""
    env = dict(os.environ, NGL="0", MAXTOK="512")
    proc = subprocess.Popen([str(vs), str(mdl)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, creationflags=CNW)
    lines, errs = [], []
    cv = threading.Condition()

    def pump(stream, sink):
        for raw in iter(stream.readline, b""):
            with cv:
                sink.append(raw.decode("utf-8", "replace").rstrip("\r\n"))
                cv.notify_all()

    threading.Thread(target=pump, args=(proc.stdout, lines), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, errs), daemon=True).start()

    def read_until(stop, start, timeout=300):
        deadline = time.time() + timeout
        i = start
        while time.time() < deadline:
            with cv:
                while i < len(lines):
                    if stop(lines[i]):
                        return lines[start:i + 1], i + 1
                    i += 1
                if proc.poll() is not None and i >= len(lines):
                    break
                cv.wait(timeout=1.0)
        return lines[start:i], i

    def request(obj):
        path = work / ("req_%d.json" % int(time.time() * 1000000))
        path.write_text(json.dumps(obj), encoding="utf-8")
        start = len(lines)
        try:
            proc.stdin.write((str(path) + "\n").encode())
            proc.stdin.flush()
        except OSError:
            return lines[start:]
        r, _ = read_until(lambda l: l == "DONE", start)
        return r

    tag = "[badutf8] " if bad_utf8 else ""
    recs, _ = read_until(lambda l: l.startswith("READY"), 0)
    check(recs and recs[-1].split()[0] == "READY" and recs[-1].split()[2] == "512",
          tag + "bad READY line: %r | stderr: %s" % (recs, errs[-5:]))

    strip = lambda rs: [l for l in rs if not l.startswith("STATS")]  # STATS carries wall-clock ms
    req = {"seed": 7, "n_blocks": 2, "messages": [{"role": "user", "content": "hello there"}]}
    r1 = request(req)
    F = [l for l in r1 if l.startswith("F ")]
    C = [l for l in r1 if l.startswith("C ")]
    S = [l for l in r1 if l.startswith("STATS")]
    check(r1 and r1[-1] == "DONE", tag + "no DONE (server alive=%s): %r | stderr: %s" % (proc.poll() is None, r1[-5:], errs[-8:]))
    check(len(F) >= 1 and all(len(f.split(" ", 4)) == 5 for f in F), tag + "F records malformed: %r" % F[:3])
    check(len(C) >= 1 and C[0].startswith("C 0 "), tag + "no C record for block 0: %r" % C)
    check(len(S) == 1 and "predicted_n=" in S[0] and "canvas=8" in S[0] and "n_ctx=512" in S[0], tag + "STATS missing/odd: %r" % S)
    payloads = []
    for f in F:
        try:
            payloads.append(json.loads(f.split(" ", 4)[4]))
        except Exception as ex:  # noqa: BLE001
            check(False, tag + "F payload is not a JSON string: %r (%s)" % (f, ex))
            break
    if bad_utf8:
        check(any("\ufffd" in p for p in payloads),
              tag + "control: the forced-invalid-UTF-8 model should show U+FFFD in its frames: %r" % payloads[:3])
    else:
        r2 = request(req)
        check(strip(r1) == strip(r2), "same request twice must give the identical F/C stream (seeded); first diff:\n%r\n%r"
              % (strip(r1)[:3], strip(r2)[:3]))
        r3 = request({"seed": 8, "n_blocks": 2, "messages": [{"role": "user", "content": "hello there"}]})
        check(strip(r1) != strip(r3), "control: a different seed must change the frames")
        big = request({"seed": 1, "n_blocks": 1, "messages": [{"role": "user", "content": "x" * 900}]})
        check(any(l.startswith("ERR toolong ") for l in big), "over-budget prompt must answer 'ERR toolong <needed> <budget>': %r" % big[-3:])
        start = len(lines)
        proc.stdin.write(b"Z:/definitely/not/a/request.json\n")
        proc.stdin.flush()
        bad, _ = read_until(lambda l: l == "DONE" or l.startswith("ERR"), start, timeout=30)
        check(any(l.startswith("ERR badreq") for l in bad), "missing request file must answer ERR badreq: %r" % bad)
        ok = 0
        for sd in range(100, 120):
            rr = request({"seed": sd, "n_blocks": 1, "messages": [{"role": "user", "content": "q%d" % sd}]})
            ok += bool(rr) and rr[-1] == "DONE" and proc.poll() is None
        check(ok == 20, "visual server answered DONE for %d of 20 back-to-back seeds (alive=%s)" % (ok, proc.poll() is None))

    try:
        proc.stdin.write(b"QUIT\n")
        proc.stdin.flush()
    except OSError:
        pass
    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = -9
    check(rc == 0, tag + "visual server exit code %r after QUIT" % rc)


visual_session(model, False)
visual_session(model_badutf8, True)

print("RESULT checks_passed=%d failed=%d" % (passes, len(fails)), flush=True)
sys.exit(1 if fails else 0)
