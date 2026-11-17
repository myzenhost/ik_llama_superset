#!/usr/bin/env python3
"""Lint: a test that asserts byte-identical generated text across concurrent or
cache-reusing requests must state its determinism setup.

Why: measured 2026-11-08 (gate b1107g2): concurrent greedy streams differ run to
run with the feature under test never used, and a request that reuses a cached
prompt changes greedy text on GPU. Continuous batching changes the batch
composition (matmul numerics); prompt-cache reuse computes the prompt another way.

Rule: a python test file that BOTH runs requests concurrently (threads, executor,
asyncio.gather) AND asserts text equality must carry two marker comments:
    # DETERMINISM-SETUP: <how identity is made safe: n_slots, cache_prompt false, ...>
    # NULL-RUN-CONTROL: <test name whose feature-off arms must PASS first> | n/a: <reason>
Usage: check-text-identity-determinism.py [--root DIR] [--selftest]
Exit 0 clean, 1 findings or unreadable files, 2 selftest failure.
"""
import argparse
import re
import sys
from pathlib import Path

CONC = re.compile(r"threading\.Thread|ThreadPoolExecutor|asyncio\.gather|concurrent\.futures|ThreadPool\(")
TEXTISH = re.compile(r"\b(got|ref|text|texts|content|answer|answers|reply|sha|digest|solo\w*)\b")
LITERAL = re.compile(r"""^\s*(["'f]|-?\d|None\b|True\b|False\b|\[\s*\]|\{\s*\})""")
ASSERT_EQ = re.compile(r"^\s*assert\s+(?P<l>[^=!<>]+?)\s*==\s*(?P<r>[^=\n]+?)\s*(,\s*f?[\"'].*)?$")


def has_text_identity(text: str) -> bool:
    """True when some assert compares two non-literal expressions, one text-ish."""
    for line in text.splitlines():
        m = ASSERT_EQ.match(line.split("#", 1)[0])
        if not m:
            continue
        l, r = m.group("l"), m.group("r")
        if "status_code" in l or "status_code" in r or "len(" in l or "len(" in r:
            continue
        if LITERAL.match(l) or LITERAL.match(r):
            continue
        if TEXTISH.search(l) or TEXTISH.search(r):
            return True
    return False
SETUP = re.compile(r"#\s*DETERMINISM-SETUP:\s*\S")
NULLRUN = re.compile(r"#\s*NULL-RUN-CONTROL:\s*\S")


def check_text(text: str) -> list[str]:
    if not CONC.search(text) or not has_text_identity(text):
        return []
    missing = []
    if not SETUP.search(text):
        missing.append("DETERMINISM-SETUP")
    if not NULLRUN.search(text):
        missing.append("NULL-RUN-CONTROL")
    return missing


def scan(root: Path):
    pats = ["tools/server/tests/**/*.py", "tests/*.py", "scripts/server-test-*.py"]
    flagged = []
    unreadable = []
    seen = set()
    for pat in pats:
        for p in sorted(root.glob(pat)):
            if p in seen or not p.is_file():
                continue
            seen.add(p)
            try:
                t = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                # FAILURES-SURFACED: collected into `unreadable`, printed by main() and makes it exit 1
                unreadable.append((p.as_posix(), str(e)))
                continue
            m = check_text(t)
            if m:
                flagged.append((p.relative_to(root).as_posix(), m))
    return flagged, unreadable, len(seen)


def selftest() -> int:
    bad = "import threading\nthreads=[threading.Thread(target=f)]\nassert got == ref\n"
    good = "# DETERMINISM-SETUP: cache_prompt false, temp 0\n# NULL-RUN-CONTROL: n/a: ref is fresh server\n" + bad
    seq = "assert got == ref\n"  # no concurrency
    conc_only = "import threading\nassert res.status_code == 200\n"
    fails = []
    if check_text(bad) != ["DETERMINISM-SETUP", "NULL-RUN-CONTROL"]:
        fails.append("positive control: unmarked concurrent identity not flagged")
    if check_text(good):
        fails.append("marked file flagged")
    if check_text(seq):
        fails.append("sequential identity flagged")
    if check_text(conc_only):
        fails.append("status-only assert flagged")
    if check_text('threading.Thread(target=f)\nassert content == ""\nassert res.body["role"] == "router"\n'):
        fails.append("literal comparison flagged")
    if check_text("threading.Thread(target=f)\nassert body[\"content\"] == _solo_short(srv), \"msg\"\n") != ["DETERMINISM-SETUP", "NULL-RUN-CONTROL"]:
        fails.append("content vs call comparison not flagged")
    if check_text("# DETERMINISM-SETUP: x\n" + bad) != ["NULL-RUN-CONTROL"]:
        fails.append("half-marked file not flagged for the missing marker")
    for f in fails:
        print("SELFTEST FAIL:", f)
    print("SELFTEST", "FAIL" if fails else "PASS")
    return 2 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rc = selftest()
    if rc:
        return rc
    flagged, unreadable, n = scan(Path(a.root))
    for f, m in flagged:
        print(f"{f}: concurrent text-identity assert lacks marker(s): {', '.join(m)}")
    for f, e in unreadable:
        print(f"UNREADABLE {f}: {e}")
    print(f"scanned {n} file(s); {len(flagged)} flagged; {len(unreadable)} unreadable")
    return 1 if (flagged or unreadable) else 0


if __name__ == "__main__":
    sys.exit(main())
