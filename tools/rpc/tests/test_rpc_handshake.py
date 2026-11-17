"""Fork RPC behavioural check -- wired into _build_merge_verify.bat as its RPC stage.

MODEL-CHOICE-OK: a 135M test model for an engine protocol check; nothing is selected for serving.

1. fork-to-fork on localhost: greedy output with every layer offloaded to the RPC server (-ngl 99) must equal
   the local-only greedy output, for server device selections -cpu, -d CPU and -dev CPU.
   CONTROL: the client log must show model buffers allocated on the RPC device. Identical text from a run
   that never used RPC proves nothing -- and that is exactly what --rpc did after the upstream #11262 merge
   (registry-only add_rpc_devices, params.rpc_servers never set, 0/31 layers offloaded, output "identical").
2. NEGATIVE control: fork client -> upstream rpc-server (protocol major 4/5 vs the fork's 133) must be REFUSED
   and must not generate. Before the fix the mismatch was logged and generation silently ran locally.
Timings are not reported; any CPU number from this host would be RAM@2133 (DOCP off).

EXIT CODES, as the audio round-trip harness: 0 = every check passed, 1 = a check FAILED,
2 = NOTHING RAN (binaries, model or upstream server absent). 2 is not a pass.

Usage: pythonw test_rpc_handshake.py <fork_bin_dir> <upstream_rpc_server.exe> <model.gguf> <out_dir>
"""
import os
import re
import socket
import subprocess
import sys
import time

if len(sys.argv) < 5:
    print(__doc__)
    sys.exit(2)
bin_dir, up_server, model, out_dir = sys.argv[1:5]
missing = [p for p in (os.path.join(bin_dir, "llama-completion.exe"), os.path.join(bin_dir, "ggml-rpc-server.exe"),
                       up_server, model) if not os.path.isfile(p)]
if missing:
    print("NOT RUN -- missing: %s" % ", ".join(missing))
    sys.exit(2)
os.makedirs(out_dir, exist_ok=True)
CLIENT = os.path.join(bin_dir, "llama-completion.exe")
SERVER = os.path.join(bin_dir, "ggml-rpc-server.exe")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
PROMPT = "The capital of France is"
COMMON = ["-m", model, "-p", PROMPT, "-n", "32", "--temp", "0", "--seed", "1", "-t", "8", "-c", "512", "--no-warmup"]


def run_client(tag, extra):
    op, ep = os.path.join(out_dir, "client_%s.out" % tag), os.path.join(out_dir, "client_%s.err" % tag)
    print("  client %s: running -> %s" % (tag, op), flush=True)
    with open(op, "w", encoding="utf-8") as fo, open(ep, "w", encoding="utf-8") as fe:
        rc = subprocess.call([CLIENT] + COMMON + extra, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL,
                             creationflags=NOWIN, timeout=300)
    return rc, open(op, encoding="utf-8", errors="replace").read(), open(ep, encoding="utf-8", errors="replace").read()


def start_server(exe, args, tag, port):
    log = open(os.path.join(out_dir, "server_%s.log" % tag), "w", encoding="utf-8")
    p = subprocess.Popen([exe] + args + ["-p", str(port)], stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=NOWIN)
    deadline = time.time() + 60
    while time.time() < deadline:
        if p.poll() is not None:
            log.close()
            raise RuntimeError("server %s exited %s before listening" % (tag, p.returncode))
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return p, log
        time.sleep(0.5)
    p.kill()
    log.close()
    raise RuntimeError("server %s not listening after 60 s" % tag)


def stop(p, log):
    p.terminate()
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
    log.close()


results = []
rc, local_out, local_err = run_client("local", ["-ngl", "0"])
if rc != 0 or not local_out.strip():
    sys.exit("local run failed rc=%d or produced no text -- see client_local.err" % rc)
print("local text: %r" % local_out.strip()[:160], flush=True)

port = 50152
for tag, sargs in (("cpu", ["-cpu"]), ("d_CPU", ["-d", "CPU"]), ("dev_CPU", ["-dev", "CPU"])):
    port += 1
    p, log = start_server(SERVER, sargs, tag, port)
    try:
        rc, out, err = run_client("rpc_" + tag, ["--rpc", "127.0.0.1:%d" % port, "-ngl", "99"])
    finally:
        stop(p, log)
    # CONTROL: model weights must actually live on the RPC device
    rpc_buf = re.findall(r"RPC[^\n]*(?:model buffer size|buffer size)\s*=\s*([\d.]+) MiB", err)
    used_rpc = any(float(x) > 1.0 for x in rpc_buf)
    same = out == local_out
    ok = rc == 0 and same and used_rpc
    results.append(ok)
    print("%-8s rc %d  rpc model buffers %s MiB  output identical to local: %s  -> %s"
          % (tag, rc, rpc_buf or "NONE", same, "OK" if ok else "FAIL"), flush=True)
    if not same:
        print("    rpc text: %r" % out.strip()[:160])

# NEGATIVE control: upstream server speaks a different protocol major -> the fork client must refuse
port += 1
p, log = start_server(up_server, [], "upstream", port)
try:
    rc, out, err = run_client("vs_upstream", ["--rpc", "127.0.0.1:%d" % port, "-ngl", "99"])
finally:
    stop(p, log)
refused = bool(re.search(r"version mismatch", err, re.I))
generated = out.strip() != "" and out == local_out
neg_ok = refused and not generated
results.append(neg_ok)
line = next((l for l in err.splitlines() if re.search(r"version mismatch", l, re.I)), "(no mismatch line)")
print("upstream  rc %d  refused (version mismatch): %s  generated anyway: %s  -> %s"
      % (rc, refused, generated, "OK" if neg_ok else "FAIL"))
print("    %s" % line.strip()[:200])
print("VERDICT %s (%d of %d checks passed)" % ("PASS" if all(results) else "FAIL", sum(results), len(results)))
sys.exit(0 if all(results) else 1)
