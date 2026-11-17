#!/usr/bin/env python3
"""Two-device loopback harness: the multi-device logic of the fork on a host that has one accelerator.

Two ggml-rpc-server processes on 127.0.0.1 each expose a CPU backend; a client started with
`--rpc A,B -ngl 99 --split-mode M` therefore sees TWO extra ggml devices, and the scheduler, tensor split, per-device
memory fit and every split mode run for real (only interconnect speed is not representative).

For every split mode the harness records: did the client start, did it exit cleanly, the final perplexity, and
whether that perplexity equals the single-device run (the parity test). A negative control then kills the second
server in the middle of a run and requires a clean, bounded failure (non-zero exit within a deadline, no hang).

usage: run_rpc_2dev.py --bin-dir DIR --model GGUF --text FILE [--modes layer,row,graph,attn,tensor]
                       [--ports 50061,50062] [--threads 4] [--chunks 3] [--ctx 128] [--timeout 600] [--out FILE]
exit code: 0 when every mode that started has parity AND no rpc-server process died in any mode AND the kill control
failed cleanly; 1 otherwise; 2 on usage errors. A mode that is REFUSED (client exits non-zero) is a legitimate answer;
a server process that exits is not.
Windows and POSIX. Only loopback addresses are used.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def exe(bin_dir, name):
    p = os.path.join(bin_dir, name + (".exe" if os.name == "nt" else ""))
    return p


def wait_port(port, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def start_servers(bin_dir, ports, threads, log_dir, tag="start"):
    procs = []
    for p in ports:
        # one log per (tag, port): a restart must not overwrite the log of the server that just died
        log = open(os.path.join(log_dir, "rpc-server-%d-%s.log" % (p, tag)), "w")
        pr = subprocess.Popen([exe(bin_dir, "ggml-rpc-server"), "-H", "127.0.0.1", "-p", str(p), "-d", "CPU", "-t", str(threads)],
                              stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        procs.append((pr, log))
    for (pr, _), p in zip(procs, ports):
        if not wait_port(p, 30):
            stop_servers(procs)
            raise RuntimeError("rpc-server on port %d did not open (see rpc-server-%d-%s.log)" % (p, p, tag))
    return procs


def stop_servers(procs):
    for pr, log in procs:
        if pr.poll() is None:
            pr.kill()
        try:
            pr.wait(timeout=10)
        except Exception:
            pass
        log.close()


PPL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+)")


def run_ppl(bin_dir, model, text, ctx, chunks, threads, extra, timeout, log_path):
    cmd = [exe(bin_dir, "llama-perplexity"), "-m", model, "-f", text, "-c", str(ctx), "--chunks", str(chunks), "-t", str(threads)] + extra
    t0 = time.time()
    with open(log_path, "w") as log:
        try:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, creationflags=NO_WINDOW)
            rc, timed_out = r.returncode, False
        except subprocess.TimeoutExpired:
            rc, timed_out = None, True
    body = open(log_path, errors="replace").read()
    m = PPL_RE.search(body)
    return {"rc": rc, "timed_out": timed_out, "ppl": float(m.group(1)) if m else None, "secs": round(time.time() - t0, 1), "log": log_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--modes", default="layer,row,graph,attn,tensor")
    ap.add_argument("--ports", default="50061,50062")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="rpc-2dev-result.json")
    ap.add_argument("--log-dir", default=".")
    a = ap.parse_args()
    ports = [int(x) for x in a.ports.split(",")]
    if len(ports) != 2 or any(p in (8088, 8090) for p in ports):
        print("need exactly two ports, and 8088/8090 are claimed", file=sys.stderr)
        return 2
    for need in ("ggml-rpc-server", "llama-perplexity"):
        if not os.path.exists(exe(a.bin_dir, need)):
            print("missing binary: " + exe(a.bin_dir, need), file=sys.stderr)
            return 2
    os.makedirs(a.log_dir, exist_ok=True)
    result = {"model": a.model, "modes": {}, "ports": ports}

    # reference: single device, no rpc
    ref = run_ppl(a.bin_dir, a.model, a.text, a.ctx, a.chunks, a.threads, [], a.timeout, os.path.join(a.log_dir, "ref.log"))
    result["reference"] = ref
    print("reference: rc=%s ppl=%s (%.1fs)" % (ref["rc"], ref["ppl"], ref["secs"]), flush=True)
    if ref["ppl"] is None:
        json.dump(result, open(a.out, "w"), indent=1)
        print("reference run produced no perplexity; harness cannot continue", file=sys.stderr)
        return 1

    rpc = ",".join("127.0.0.1:%d" % p for p in ports)
    all_ok = True
    procs = start_servers(a.bin_dir, ports, a.threads, a.log_dir)
    try:
        for mode in a.modes.split(","):
            r = run_ppl(a.bin_dir, a.model, a.text, a.ctx, a.chunks, a.threads,
                        ["--rpc", rpc, "-ngl", "99", "--split-mode", mode], a.timeout, os.path.join(a.log_dir, "mode-%s.log" % mode))
            started = r["ppl"] is not None
            r["started"] = started
            r["parity"] = bool(started and abs(r["ppl"] - ref["ppl"]) <= 1e-3 * ref["ppl"])
            r["verdict"] = "PARITY" if r["parity"] else ("RAN-BUT-DIFFERENT" if started else ("HANG" if r["timed_out"] else "REFUSED-OR-ABORTED"))
            result["modes"][mode] = r
            print("mode %-7s rc=%s ppl=%s %s (%.1fs)" % (mode, r["rc"], r["ppl"], r["verdict"], r["secs"]), flush=True)
            if started and not r["parity"]:
                all_ok = False
            # a server that died during the mode would poison the next one
            if any(pr.poll() is not None for pr, _ in procs):
                r["server_exit_codes"] = [pr.poll() for pr, _ in procs]
                print("an rpc-server exited during mode %s (exit codes %s); restarting both" % (mode, r["server_exit_codes"]), flush=True)
                # A server must survive a client whose graph it cannot run: refusing one graph is fine, the whole process
                # aborting (measured 2026-10-11: GGML_OP_REDUCE reaching ggml_get_n_tasks, exit code 3) is a defect.
                r["server_died"] = True
                all_ok = False
                stop_servers(procs)
                procs = start_servers(a.bin_dir, ports, a.threads, a.log_dir, tag="after-" + mode)
    finally:
        stop_servers(procs)

    # negative control: kill server #2 mid-run. The run must end (non-zero) before the deadline.
    procs = start_servers(a.bin_dir, ports, a.threads, a.log_dir, tag="killctl")
    try:
        cmd = [exe(a.bin_dir, "llama-perplexity"), "-m", a.model, "-f", a.text, "-c", str(a.ctx), "--chunks", "400", "-t", str(a.threads),
               "--rpc", rpc, "-ngl", "99", "--split-mode", "layer"]
        klog = open(os.path.join(a.log_dir, "kill-control.log"), "w")
        client = subprocess.Popen(cmd, stdout=klog, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        time.sleep(15)
        alive_before = client.poll() is None
        procs[1][0].kill()
        t_kill = time.time()
        try:
            rc = client.wait(timeout=120)
            hung = False
        except subprocess.TimeoutExpired:
            client.kill()
            rc, hung = None, True
        klog.close()
        ctl = {"client_alive_before_kill": alive_before, "rc": rc, "hung": hung, "secs_to_exit": round(time.time() - t_kill, 1)}
        ctl["clean_failure"] = bool(alive_before and not hung and rc not in (0, None))
        result["kill_control"] = ctl
        print("kill control: %s" % ctl, flush=True)
        if not ctl["clean_failure"]:
            all_ok = False
    finally:
        stop_servers(procs)

    json.dump(result, open(a.out, "w"), indent=1)
    print("RESULT: %s" % ("PASS" if all_ok else "FAIL"), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
