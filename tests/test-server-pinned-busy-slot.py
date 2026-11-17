#!/usr/bin/env python3
"""End-to-end check: a request pinned (id_slot) to a BUSY slot must not touch that slot.

SOURCE-READ: upstream llama.cpp 23b0202a "server: leave a busy slot untouched when a request pins it"
(#30295), tools/server/server-context.cpp:1686-1694 and tools/server/tests/unit/test_completion.py
test_pinned_request_on_busy_slot. The fork had the same defect: get_available_slot() returned the pinned
slot even while it was generating, then, because the similarity loop skips busy slots (f_sim_best = 0 ->
f_keep = 0 < --cache-ram-similarity), ran prompt_save() + prompt_load() on it. When the RAM prompt cache
held a better match for the new request, that state was loaded into the slot under the running generation,
which then continued on the wrong context.

Method (same as upstream): run a long greedy generation on slot 0 twice; in the second run, while it is
streaming, send another request pinned to slot 0 whose prompt is in the RAM cache. The streamed text of
the two runs must be identical. CONTROL: the second request must have arrived while the first was still
generating (checked by counting stream chunks after it was sent), otherwise the run proves nothing.

usage: test-server-pinned-busy-slot.py <llama-server exe> <model.gguf> [threads]
exit 0 = pass, 1 = fail, 2 = inconclusive (control not met / setup error)
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

STORY = "Once upon a time a dragon named Ember guarded a golden key in a deep cave. " * 8
BREAD = "To bake bread, mix flour, water and salt, then"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post(port, path, body, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def stream(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/completion", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError("stream status %d: %r" % (r.status, r.read()[:300]))
    buf = b""
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        obj = json.loads(payload)
        yield obj
        if obj.get("stop"):
            break
    c.close()


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    exe, model = sys.argv[1], sys.argv[2]
    threads = sys.argv[3] if len(sys.argv) > 3 else "4"
    port = free_port()
    args = [exe, "-m", model, "--port", str(port), "-c", "8192", "-np", "2", "-t", threads, "--cache-ram", "256"]
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".." , "test-server-pinned-busy-slot.server.log"), "wb") \
        if os.environ.get("PINNED_TEST_KEEP_LOG") else subprocess.DEVNULL
    srv = subprocess.Popen(args, stdout=log, stderr=log, **kw)
    try:
        t0 = time.time()
        while True:
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                pass
            if srv.poll() is not None or time.time() - t0 > 120:
                print("SETUP: server did not become healthy (rc=%s)" % srv.poll())
                return 2
            time.sleep(0.2)

        def run(pin_busy):
            st, _ = post(port, "/completion", {"prompt": STORY, "id_slot": 0, "n_predict": 4, "temperature": 0.0})
            if st != 200:
                raise RuntimeError("story request status %d" % st)
            gen = stream(port, {"prompt": BREAD, "id_slot": 0, "n_predict": 3000, "ignore_eos": True,
                                "temperature": 0.0, "stream": True})
            first = next(gen)
            parts = [first.get("content", "")]
            after = None
            th = None
            res = {}
            if pin_busy:
                def pinned():
                    res["status"], _ = post(port, "/completion", {"prompt": STORY + "The dragon", "id_slot": 0,
                                                                  "n_predict": 4, "temperature": 0.0})
                th = threading.Thread(target=pinned)
                th.start()
                after = 0
            for ch in gen:
                parts.append(ch.get("content", ""))
                if after is not None:
                    after += 1
            if th is not None:
                th.join(600)
            return "".join(parts), after, res.get("status")

        base, _, _ = run(False)
        pinned, chunks_after, st_pinned = run(True)
        print("baseline chars=%d pinned-run chars=%d chunks streamed after the pinned request was sent=%s pinned status=%s"
              % (len(base), len(pinned), chunks_after, st_pinned))
        if not chunks_after or chunks_after < 50:
            print("INCONCLUSIVE: the pinned request did not overlap the generation (control not met)")
            return 2
        if st_pinned != 200:
            print("FAIL: pinned request status %s" % st_pinned)
            return 1
        if pinned != base:
            i = next((k for k in range(min(len(base), len(pinned))) if base[k] != pinned[k]), min(len(base), len(pinned)))
            print("FAIL: the generation running on the pinned slot changed at char %d" % i)
            print("  baseline: %r" % base[max(0, i - 40):i + 80])
            print("  pinned  : %r" % pinned[max(0, i - 40):i + 80])
            return 1
        print("PASS: generation on the busy slot is unchanged")
        return 0
    finally:
        srv.terminate()
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
