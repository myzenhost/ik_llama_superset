"""docs/parameters.md table gate: every flag row is complete and no flag is documented twice in one table.

WHY: a second, truncated `--cache-disk-write-budget` row (no closing cells, no blank line before the next heading) sat in
the table for days; nothing read the file, so the duplicate and the broken markdown went unnoticed.

CHECKS, per markdown table (a run of lines starting with '|'):
  1. every row has the header's cell count and ends with '|';
  2. the first cell's flag spelling (text inside the first `backticks`) appears at most once in that table.
A flag may legitimately appear in two different tables (e.g. a server table and a common table), so uniqueness is per table.
EXIT: 0 ok | 1 finding(s) | 2 the file could not be read
--selftest runs the must-fail and must-pass controls on in-memory text first.
Usage: python scripts/check-docs-parameters-table.py [--root <fork>] [--selftest]
"""
import argparse, os, re, sys

FLAG = re.compile(r"`([^`]+)`")


def cells(line):
    # GitHub tables split on every '|' not escaped as '\|' -- including one inside a `code span`, so that counts as a break
    s = line.strip()
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return re.split(r"(?<!\\)\|", s[1:]) if s.startswith("|") else []


def check(text):
    findings, table, start = [], [], 0
    lines = text.splitlines()
    for i, line in enumerate(lines + [""]):
        if line.lstrip().startswith("|"):
            if not table:
                start = i
            table.append((i + 1, line))
            continue
        if table:
            findings += check_table(table)
            table = []
    return findings


def check_table(rows):
    out = []
    if len(rows) < 2:
        return out
    ncol = len(cells(rows[0][1]))
    seen = {}
    for lineno, line in rows[2:]:  # skip header and separator
        n = len(cells(line))
        if n != ncol or not line.rstrip().endswith("|"):
            out.append("line %d: %d cell(s), header has %d%s" % (lineno, n, ncol, "" if line.rstrip().endswith("|") else ", row not closed by '|'"))
        m = FLAG.search(cells(line)[0] if cells(line) else "")
        if m:
            key = m.group(1)  # the whole spelling: `-cuda fa-offset=value` and `-cuda fa-fabsum=1` are two rows, not one flag twice
            if key in seen:
                out.append("line %d: %s already documented at line %d of the same table" % (lineno, key, seen[key]))
            else:
                seen[key] = lineno
    return out


def selftest():
    hdr = "| Parameter | Description | Default | Notes |\n| - | - | - | - |\n"
    good = hdr + "| `--a N` | x | 1 | y |\n| `--b` | x | 0 | y |\n"
    dup = hdr + "| `--a N` | x | 1 | y |\n| `--a N` | x | 1 | y |\n"
    trunc = hdr + "| `--a N` | x | 1 | y |\n| `--b MiB` | x \n## Next\n"
    two_tables = hdr + "| `--a N` | x | 1 | y |\n\ntext\n\n" + hdr + "| `--a N` | x | 1 | y |\n"
    escaped = hdr + "| `--m {x\\|y}` | x | 1 | y |\n"
    raw_pipe = hdr + "| `--m` | x | 1 | uses `(a|b)` |\n"
    sub_opts = hdr + "| `-cuda a=1` | x | 1 | y |\n| `-cuda b=1` | x | 1 | y |\n"
    ok = True
    for name, txt, want in (("good", good, 0), ("duplicate", dup, 1), ("truncated", trunc, 1), ("same flag in two tables", two_tables, 0),
                            ("escaped pipe", escaped, 0), ("raw pipe in code span", raw_pipe, 1), ("sub-options of one flag", sub_opts, 0)):
        got = len(check(txt))
        res = (got == 0) == (want == 0)
        ok &= res
        print("selftest %-24s findings=%d expected %s -> %s" % (name, got, "none" if want == 0 else "some", "ok" if res else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest and not selftest():
        print("SELFTEST FAILED: the gate cannot be trusted")
        return 2
    path = os.path.join(a.root, "docs", "parameters.md")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        print("cannot read %s: %s" % (path, e))
        return 2
    rows = sum(1 for l in text.splitlines() if l.lstrip().startswith("| `"))
    f = check(text)
    print("SCOPE: %s, %d flag row(s) checked" % (path, rows))
    for x in f:
        print("FINDING", x)
    print("%d finding(s)" % len(f))
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
