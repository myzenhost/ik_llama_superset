#!/usr/bin/env python3
"""llama-cli on the random-weight tiny model: the opt-in flags must reach the code (log lines), and with the flags OFF the
generated text must be unchanged.   usage: test-wire-cli.py <llama-cli> <model.gguf>"""
import subprocess, sys

cli, model = sys.argv[1], sys.argv[2]
base = [cli, "-m", model, "-fa", "on", "-c", "128", "-n", "12", "-p", "t5 t6 t7", "--temp", "0", "-st", "-ngl", "0"]


def run(extra):
    r = subprocess.run(base + extra, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout, r.stderr


fail = 0


def check(c, msg):
    global fail
    if not c:
        print("FAIL", msg)
        fail += 1


rc0, out0, err0 = run([])
check(rc0 == 0, "default run rc")
check("CRAM probe" not in out0 + err0, "probe line must be absent without --cram-probe")
check("TurboQuant KV cache enabled" not in out0 + err0, "turbo line must be absent with the default cache")

rc1, out1, err1 = run(["--cram-probe", "--sampler-topk-prefilter"])
check(rc1 == 0, "flag run rc")
check("CRAM probe (--cram-probe): available=" in out1 + err1, "--cram-probe log line")
check("GUESSES" in out1 + err1, "probe says plainly its sysfs names are guesses")
# greedy sampling is exact with and without the prefilter, so the visible text must not change
gen0 = out0.split("t5 t6 t7")[-1].strip().splitlines()
gen1 = out1.split("t5 t6 t7")[-1].strip().splitlines()
check(bool(gen0) and bool(gen1) and gen0[0] == gen1[0], "generated text identical with --sampler-topk-prefilter (greedy): %r vs %r" % (gen0[:1], gen1[:1]))

for t in ("turbo4", "turbo3", "turbo2"):
    rc, out, err = run(["-ctk", t, "-ctv", t])
    check(rc == 0, t + " run rc")
    check("TurboQuant KV cache enabled: K=%s V=%s" % (t, t) in out + err, t + " log line naming the cache type")
    g = out.split("t5 t6 t7")[-1].strip()
    check(len(g) > 0, t + " generated text")

rc, out, err = run(["-ctk", "turbo3", "-ctv", "turbo3", "-fa", "off"])
check(rc != 0 or "failed" in (out + err).lower() or "TurboQuant KV cache enabled" not in out + err, "turbo without flash attention must not silently run")

print("test-wire-cli:", "FAILED (%d)" % fail if fail else "PASSED")
sys.exit(1 if fail else 0)
