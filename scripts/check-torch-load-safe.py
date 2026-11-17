#!/usr/bin/env python3
"""Every torch.load() in the tree's Python must pass weights_only=True (or carry an explicit, reasoned exemption).

WHY. torch.load unpickles. On torch < 2.6 the default is weights_only=False, so loading an untrusted checkpoint (a downloaded .bin/.pt/.clip projector, a
LoRA file) executes arbitrary code. The converters that read real model files already pass weights_only=True (convert_lora_to_gguf.py, conversion/base.py);
the legacy mmproj converters under {examples,tools}/mtmd/legacy-models did not (14 calls, found 2026-10-05 via the inferflow review and re-verified on this tree).

CHECK. Every `torch.load(` call in every *.py under --root (skipping .git and build* directories): its argument text, found by balanced-parenthesis scan, must contain
`weights_only=True`. A call with no weights_only, or with weights_only=False, is reported file:line. A trusted-local-file exemption is a `# torch-load-ok: <why>`
comment on the call's first line.
--selftest: synthetic sources the checker MUST flag (no weights_only; weights_only=False; nested call args) and MUST pass (weights_only=True; exempted; the word
torch.load inside a string or comment). A checker that cannot fail proves nothing.
Row S69-INFERFLOW-PICK-LEGACY-MMPROJ-CONVERTER-TORCH-LOAD-2026-10-05-01.
"""
import argparse
import os
import re
import sys

CALL = re.compile(r"\btorch\.load\s*\(")


def _strip_noise(src):
    """Blank out comments and string literals (keeping newlines/length) so a `torch.load(` inside them is not a call."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "#":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c in "\"'":
            q = src[i:i + 3] if src[i:i + 3] in ('"""', "'''") else c
            j = src.find(q, i + len(q))
            while j > 0 and q == c and src[j - 1] == "\\":
                j = src.find(q, j + 1)
            j = n if j < 0 else j + len(q)
            seg = src[i:j]
            out.append("".join(ch if ch == "\n" else " " for ch in seg))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def findings(src, name="<src>"):
    """Return [(line, reason)] for unsafe torch.load calls in one source text."""
    clean = _strip_noise(src)
    lines = src.splitlines()
    res = []
    for m in CALL.finditer(clean):
        depth, j = 1, m.end()
        while j < len(clean) and depth:
            depth += (clean[j] == "(") - (clean[j] == ")")
            j += 1
        args = clean[m.end():j - 1]
        line = clean.count("\n", 0, m.start()) + 1
        if "torch-load-ok:" in lines[line - 1]:
            continue
        if re.search(r"weights_only\s*=\s*False", args):
            res.append((line, "weights_only=False"))
        elif not re.search(r"weights_only\s*=\s*True", args):
            res.append((line, "no weights_only=True"))
    return res


def scan(root):
    bad, files = [], 0
    for d, dirs, fns in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".git" and not x.startswith("build")]
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(d, fn)
            try:
                src = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            files += 1
            for line, why in findings(src):
                bad.append((os.path.relpath(p, root).replace(os.sep, "/"), line, why))
    return files, bad


def selftest():
    cases = [
        ("flag: no weights_only", "import torch\nx = torch.load(path)\n", 1),
        ("flag: only map_location", "y = torch.load(p, map_location='cpu')\n", 1),
        ("flag: weights_only=False", "z = torch.load(p, weights_only=False)\n", 1),
        ("flag: nested call in args", "w = torch.load(os.path.join(a, 'b'))\n", 1),
        ("pass: weights_only=True", "a = torch.load(p, map_location='cpu', weights_only=True)\n", 0),
        ("pass: nested args with weights_only=True", "a = torch.load(os.path.join(a, 'b'), weights_only=True)\n", 0),
        ("pass: exempted with a reason", "a = torch.load(p)  # torch-load-ok: our own just-written file\n", 0),
        ("pass: only in a comment", "# torch.load(p) is unsafe\n", 0),
        ("pass: only in a string", "s = 'call torch.load(p) here'\n", 0),
        ("flag: two calls, one unsafe", "a = torch.load(p, weights_only=True)\nb = torch.load(q)\n", 1),
    ]
    ok = True
    for name, src, want in cases:
        got = len(findings(src))
        good = got == want
        ok &= good
        print("  [%s] selftest: %-42s expected %d finding(s), got %d" % ("ok" if good else "FAIL", name, want, got))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest and not selftest():
        print("SELFTEST FAILED")
        return 2
    files, bad = scan(a.root)
    print("SCOPE: %s | %d .py file(s) scanned (skipping .git and build*)" % (a.root, files))
    for f, line, why in bad:
        print("FAIL: %s:%d  torch.load: %s" % (f, line, why))
    print("RESULT:", "FAIL (%d)" % len(bad) if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
