"""SCOUT-NA: driver for the probe-schema-depth executable (same fork tree, no new tool).

Row S69-ENGREV-UPSTREAM-MERGE-SCHEMA-DEPTH-CAP-MEASURE-2026-11-04-01. For each (kind, where, stage) it finds the deepest
nest the code survives with the cap lifted: a stack overflow kills the child (rc 0xC00000FD), so every probe point is its own
process. Run through temp/_run_hidden.py (CREATE_NO_WINDOW is also set here).

    python probe-schema-depth.py <probe.exe> <cwd-with-models/> [out.json]
"""
import json
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000
CRASH = {0xC00000FD: "STACK_OVERFLOW", 0xC0000005: "ACCESS_VIOLATION", 0xC0000409: "FAILFAST"}


def run(exe, cwd, kind, depth, where, stage):
    try:
        p = subprocess.run([exe, kind, str(depth), where, stage], cwd=cwd, capture_output=True, text=True,
                           creationflags=CREATE_NO_WINDOW, timeout=90)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    rc = p.returncode & 0xFFFFFFFF
    out = (p.stdout or "").strip().splitlines()
    last = out[-1] if out else ""
    if rc == 0:
        return ("OK" if last.startswith("OK") else "REFUSED"), last[:120]
    return CRASH.get(rc, "rc=%#x" % rc), last[:120]


def max_ok(exe, cwd, kind, where, stage, hi=4096):
    """Largest depth in [1, hi] that completes with OK; also the first depth that fails and how."""
    lo = 1
    st, _ = run(exe, cwd, kind, lo, where, stage)
    if st != "OK":
        return {"max_ok": 0, "first_fail": 1, "fail": st}
    st, info = run(exe, cwd, kind, hi, where, stage)
    if st == "OK":
        return {"max_ok": hi, "first_fail": None, "fail": None, "note": "no failure up to hi"}
    fail_kind = st
    while hi - lo > 1:
        mid = (lo + hi) // 2
        st, _ = run(exe, cwd, kind, mid, where, stage)
        if st == "OK":
            lo = mid
        else:
            hi = mid
            fail_kind = st
    return {"max_ok": lo, "first_fail": hi, "fail": fail_kind}


def main():
    exe, cwd = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    res = {}
    for stage in ("schema", "tmpl"):
        for where in ("main", "thread"):
            for kind in ("array", "object", "anyof"):
                r = max_ok(exe, cwd, kind, where, stage)
                # the points the row names
                r["at_127"] = run(exe, cwd, kind, 127, where, stage)[0]
                r["at_255"] = run(exe, cwd, kind, 255, where, stage)[0]
                r["at_256"] = run(exe, cwd, kind, 256, where, stage)[0]
                res["%s/%s/%s" % (stage, where, kind)] = r
                print(stage, where, kind, json.dumps(r), flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
