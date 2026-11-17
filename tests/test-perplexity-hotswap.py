#!/usr/bin/env python3
"""llama-perplexity hot-swap signal mode (docs/development/on-demand-tensor-reload.md, "signal mode").

Drives a real llama-perplexity through the control/status file protocol and checks every state it must publish:

    start                      -> status "computing 1 0", then "done 1 0"
    "compute 1"                -> "computing 2 1", then "done 2 1"      (recompute without reload)
    "reload 2" (nothing moved) -> "reload_failed 2 2"                    (llama_reload_changed_tensors found no change)
    "compute 2" (stale seq)    -> ignored: status stays "reload_failed 2 2", the command file is consumed
    "exit 3"                   -> "exiting 2 3", process exits 0

Plus the guard: LLAMA_HOTSWAP_CONTROL_FILE without LLAMA_HOTSWAP_STATUS_FILE must refuse with exit 1.

A build without the mode runs one computation and exits without ever writing the status file, so the first wait fails.

usage: test-perplexity-hotswap.py --bin <llama-perplexity> --model <small gguf> [--workdir PARENT]
Every run uses a NEW directory (under PARENT, or the system temp dir), so no stale control or status file can exist.
Exit 0 pass, 1 fail. Model: any small GGUF (stories260K-f32 is what was run). CPU only (-ngl 0, CUDA hidden).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

TEXT = ("Once upon a time there was a little girl named Lily. She liked to play in the park with her dog. "
        "One day the sun was hot and the sky was blue, and Lily found a red ball under a big tree. ") * 40


def read_status(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace").strip()
    except OSError:
        return None


def wait_status(path, want, proc, timeout=120.0):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        last = read_status(path)
        if last == want:
            return True, last
        if proc.poll() is not None and read_status(path) != want:
            return False, "%s (process exited rc=%s)" % (last, proc.returncode)
        time.sleep(0.05)
    return False, "%s (timeout %.0fs)" % (last, timeout)


def send(path, line):
    # written whole and then renamed into place, as a driver should
    tmp = path + ".w"
    with open(tmp, "wb") as f:
        f.write((line + "\n").encode())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workdir", default=None, help="parent directory for the run's new directory")
    a = ap.parse_args()

    if a.workdir:
        os.makedirs(a.workdir, exist_ok=True)
    wd = tempfile.mkdtemp(prefix="ppl-hotswap-", dir=a.workdir)
    text = os.path.join(wd, "text.txt")
    ctrl = os.path.join(wd, "ctrl.txt")
    stat = os.path.join(wd, "status.txt")
    log = os.path.join(wd, "perplexity.log")
    with open(text, "w", encoding="utf-8") as f:
        f.write(TEXT)

    fails = []

    def check(what, ok, detail=""):
        print("  %-4s %s%s" % ("ok" if ok else "FAIL", what, ("  -- " + detail) if detail and not ok else ""))
        if not ok:
            fails.append(what)

    base = [a.bin, "-m", a.model, "-f", text, "-c", "64", "-b", "64", "--chunks", "2", "-t", "2", "-ngl", "0"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", LLAMA_HOTSWAP_ENABLED="1")
    env.pop("LLAMA_HOTSWAP_STATUS_FILE", None)
    env.pop("LLAMA_HOTSWAP_CONTROL_FILE", None)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    # guard: a control file without a status file is refused
    env_bad = dict(env, LLAMA_HOTSWAP_CONTROL_FILE=ctrl)
    with open(log + ".guard", "wb") as lf:
        rc = subprocess.call(base, env=env_bad, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags, timeout=120)
    check("CONTROL_FILE without STATUS_FILE exits 1", rc == 1, "rc=%s" % rc)

    env_ok = dict(env, LLAMA_HOTSWAP_CONTROL_FILE=ctrl, LLAMA_HOTSWAP_STATUS_FILE=stat)
    with open(log, "wb") as lf:
        proc = subprocess.Popen(base, env=env_ok, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            ok, got = wait_status(stat, "done 1 0", proc)
            check("first computation publishes 'done 1 0'", ok, got)
            if ok:
                check("process stays resident after the first computation", proc.poll() is None)
                send(ctrl, "compute 1")
                ok, got = wait_status(stat, "done 2 1", proc)
                check("'compute 1' -> recompute, 'done 2 1'", ok, got)
                send(ctrl, "reload 2")
                ok, got = wait_status(stat, "reload_failed 2 2", proc)
                check("'reload 2' with nothing changed on disk -> 'reload_failed 2 2'", ok, got)
                send(ctrl, "compute 2")
                time.sleep(1.5)
                check("stale 'compute 2' is ignored (status unchanged)", read_status(stat) == "reload_failed 2 2",
                      str(read_status(stat)))
                check("stale command file was consumed", not os.path.exists(ctrl))
                send(ctrl, "exit 3")
                ok, got = wait_status(stat, "exiting 2 3", proc)
                check("'exit 3' -> 'exiting 2 3'", ok, got)
                try:
                    rc = proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    rc = None
                check("process exits 0 after 'exit'", rc == 0, "rc=%s" % rc)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    with open(log, "rb") as f:
        out = f.read().decode("utf-8", "replace")
    check("log names signal mode", "hot-swap signal mode enabled" in out)
    check("log shows two perplexity runs", out.count("Final estimate: PPL") >= 2, "count=%d" % out.count("Final estimate: PPL"))

    print("%s (%d failed)  workdir=%s" % ("FAIL" if fails else "PASS", len(fails), wd))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
