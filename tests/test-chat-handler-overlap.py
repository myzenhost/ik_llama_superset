#!/usr/bin/env python3
"""Chat-handler overlap report: which specialized handlers does each bundled template match?

Row S69-ENGREV-TABBYAPI-CHAT-HANDLER-LOADLOG-2026-10-27-01, part 2 (part 1 is the load-time INFO line, common/chat.cpp).

common_chat_template_detect_params_init() in common/chat.cpp is an ordered ladder: the first matching `if` wins, and the
ladder logs and returns, so a template that satisfies two predicates silently loses the later one. This test reads the
ladder from the C++ source (no build, no model, CPU only), turns every predicate into a Python expression, and for each
models/templates/*.jinja prints the handler that WINS plus every other entry it ALSO matches (the overlap).

Predicate forms handled: src.find("lit") != / == std::string::npos joined by && || ! and parentheses, plus
is_lfm2_template(src) (common/parsers/lfm2.cpp:5, expanded here). Any other construct in a condition fails the test, so a new
predicate shape cannot be skipped silently.

Exit 0 = pass (overlaps are reported, not failures), 1 = the ladder could not be parsed or a control failed.
Usage: test-chat-handler-overlap.py [repo-root] [--quiet]
"""
import os
import re
import sys

LFM2_EXPANSION = '(("<|tool_list_start|>" in src) and ("<|tool_list_end|>" in src))'
STR = r'"(?:[^"\\]|\\.)*"'
FIND = re.compile(r"src\s*\.\s*find\s*\(\s*(" + STR + r")\s*\)\s*(!=|==)\s*std::string::npos")


def decode(lit):
    body = lit[1:-1]
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            n = body[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(n, n))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def strip_line_comments(text):
    """Drop // comments, leaving string literals alone."""
    res = []
    for line in text.split("\n"):
        in_str, i, cut = False, 0, None
        while i < len(line):
            c = line[i]
            if in_str:
                if c == "\\":
                    i += 2
                    continue
                if c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "/" and line[i:i + 2] == "//":
                cut = i
                break
            i += 1
        res.append(line if cut is None else line[:cut])
    return "\n".join(res)


def balanced(text, start, open_ch, close_ch):
    """Index of the bracket matching text[start] (== open_ch), skipping string literals."""
    depth, i, in_str = 0, start, False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced " + open_ch)


def to_python(cond):
    cond = cond.replace("is_lfm2_template(src)", "@LFM2@")

    def sub(m):
        return "(%r %s src)" % (decode(m.group(1)), "in" if m.group(2) == "!=" else "not in")

    py = FIND.sub(sub, cond)
    py = py.replace("&&", " and ").replace("||", " or ").replace("@LFM2@", LFM2_EXPANSION)
    py = re.sub(r"!(?!=)", " not ", py)
    # Whatever is left must be only boolean glue; a leftover identifier means an unhandled predicate shape.
    leftover = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "", py)
    leftover = re.sub(r"\b(and|or|not|in|src)\b", "", leftover)
    if re.search(r"[A-Za-z_]", leftover):
        raise ValueError("unhandled predicate text: " + leftover.strip()[:120])
    return py


def parse_ladder(source):
    m = re.search(r"common_chat_template_detect_params_init\s*\(const std::string & src\)\s*\{", source)
    if not m:
        raise ValueError("ladder function not found")
    body_start = m.end() - 1
    body_end = balanced(source, body_start, "{", "}")
    body = strip_line_comments(source[body_start + 1:body_end])
    entries, pos = [], 0
    while True:
        im = re.compile(r"(?m)^    if\s*\(").search(body, pos)
        if not im:
            break
        p_open = im.end() - 1
        p_close = balanced(body, p_open, "(", ")")
        cond = " ".join(body[p_open + 1:p_close].split())  # one line: Python eval rejects the ladder's line breaks
        b_open = body.index("{", p_close)
        b_close = balanced(body, b_open, "{", "}")
        block = body[b_open:b_close + 1]
        name = re.search(r'Using specialized template:\s*([^\\"]+)', block)
        entries.append({"name": name.group(1).strip() if name else "?",
                        "expr": to_python(cond), "cond": " ".join(cond.split())})
        pos = b_close + 1
    return entries


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv
    root = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    root = os.path.abspath(root)
    source = open(os.path.join(root, "common", "chat.cpp"), encoding="utf-8").read()
    try:
        ladder = parse_ladder(source)
    except ValueError as e:
        print("FAIL: ladder parse: %s" % e)
        return 1
    if len(ladder) < 20:
        print("FAIL: only %d ladder entries parsed (expected 20+); the parser lost the ladder" % len(ladder))
        return 1
    bad = [e["name"] for e in ladder if e["name"] == "?"]
    if bad:
        print("FAIL: %d ladder entries without a 'Using specialized template' name" % len(bad))
        return 1

    def matches(src):
        return [e["name"] for e in ladder if eval(e["expr"], {"src": src})]  # noqa: S307 (expressions built above)

    # Controls: an empty template matches nothing; a template made of one predicate's literals matches that entry.
    if matches(""):
        print("FAIL: control: empty source matched %s" % matches(""))
        return 1
    qwen = next(e for e in ladder if e["name"] == "Qwen3-Coder")
    probe = "<tool_call> <function= <parameter="
    if "Qwen3-Coder" not in matches(probe) or not eval(qwen["expr"], {"src": probe}):
        print("FAIL: control: synthetic Qwen3-Coder source did not match its own entry")
        return 1

    tdir = os.path.join(root, "models", "templates")
    names = sorted(f for f in os.listdir(tdir) if f.endswith(".jinja"))
    overlaps, auto, total = 0, 0, 0
    print("ladder entries parsed: %d (order = priority, first match wins)" % len(ladder))
    for i, e in enumerate(ladder, 1):
        print("  %2d. %s" % (i, e["name"]))
    print()
    for f in names:
        src = open(os.path.join(tdir, f), encoding="utf-8", errors="replace").read()
        hit = matches(src)
        total += 1
        if not hit:
            auto += 1
            if not quiet:
                print("%-56s -> autoparser" % f)
            continue
        extra = hit[1:]
        if extra:
            overlaps += 1
        if extra or not quiet:
            print("%-56s -> %s%s" % (f, hit[0], ("   ALSO MATCHES (shadowed): " + ", ".join(extra)) if extra else ""))
    print()
    print("templates %d: specialized %d, autoparser %d, with shadowed overlap %d" % (total, total - auto, auto, overlaps))
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
