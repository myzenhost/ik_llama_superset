"""Connections the fork RPC client opens per run, as a function of generated tokens.

MODEL-CHOICE-OK: 135M test fixture for an engine measurement; nothing is selected for serving.
Row S69-FORK-RPC-CLIENT-371-RECONNECTS-PER-RUN-2026-09-26-01. If the count does not move with -n, the connections
come from load/setup, not decode. CONTROL: every run must actually offload (RPC model buffer present).
Usage: pythonw rpc_conn_count.py <fork_bin_dir> <model.gguf> <out_dir> [n_predict ...]
"""
import os
import re
import socket
import subprocess
import sys
import time

bin_dir, model, out_dir = sys.argv[1:4]
ns = [int(x) for x in sys.argv[4:]] or [1, 64]
os.makedirs(out_dir, exist_ok=True)
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
port = 50252
rows = []
for n in ns:
    port += 1
    slog = os.path.join(out_dir, "count_server_n%d.log" % n)
    with open(slog, "w", encoding="utf-8") as sl:
        srv = subprocess.Popen([os.path.join(bin_dir, "ggml-rpc-server.exe"), "-cpu", "-p", str(port)],
                               stdout=sl, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NOWIN)
        try:
            for _ in range(120):
                with socket.socket() as s:
                    s.settimeout(1)
                    if s.connect_ex(("127.0.0.1", port)) == 0:
                        break
                time.sleep(0.5)
            else:
                sys.exit("server did not listen")
            # the readiness probe above is itself one connection; subtract it below
            cerr = os.path.join(out_dir, "count_client_n%d.err" % n)
            print("  n=%d: running client" % n, flush=True)
            with open(cerr, "w", encoding="utf-8") as ce, open(os.devnull, "w") as dn:
                rc = subprocess.call([os.path.join(bin_dir, "llama-completion.exe"), "-m", model, "-p", "The capital of France is",
                                      "-n", str(n), "--temp", "0", "-t", "8", "-c", "512", "--no-warmup",
                                      "--rpc", "127.0.0.1:%d" % port, "-ngl", "99"],
                                     stdout=dn, stderr=ce, stdin=subprocess.DEVNULL, creationflags=NOWIN, timeout=300)
            time.sleep(1.0)
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=15)
            except subprocess.TimeoutExpired:
                srv.kill()
    t = open(slog, encoding="utf-8", errors="replace").read()
    acc = len(re.findall(r"^Accepted client connection$", t, re.M)) - 1  # minus the readiness probe
    err = open(cerr, encoding="utf-8", errors="replace").read()
    offl = re.search(r"offloaded (\d+)/(\d+) layers", err)
    rpc_buf = bool(re.search(r"RPC0\[[^\]]+\] buffer size\s*=\s*[1-9]", err))
    rows.append((n, rc, acc, offl.group(0) if offl else "?", rpc_buf))
    print("n_predict %3d  rc %d  client connections %4d  %s  rpc weights present %s" % rows[-1], flush=True)
ok = all(r[1] == 0 and r[4] for r in rows)
print("CONTROL (every run offloaded to RPC): %s" % ok)
sys.exit(0 if ok else 1)
