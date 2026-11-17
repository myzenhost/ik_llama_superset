"""Static check that the BUILT server's BDD suite (tools/server/tests/features) can resolve its steps.
behave is not installed on this host, so this does what `behave --dry-run` would report, from the source:
  A. every `import steps` in features/steps/*.py resolves to a steps.py in that same directory, and every
     `steps.<name>` the modules use is defined at top level of that steps.py;
  B. every Given/When/Then/And/But line of every features/*.feature matches at least one @step/@given/@when/@then
     decorator pattern in features/steps/*.py (behave "parse" placeholders {x} / {x:d} -> a non-greedy wildcard).
     Decorators are read by AST, because one spans two lines with implicitly concatenated strings.
     Step text is normalised as behave does: ONE trailing ':' stripped (behave/parser.py:457-459
     _normalize_step_name, STRIP_STEPS_WITH_TRAILING_COLON), Scenario Outline <x> parameters filled in.
CONTROLS: a step built from a responses_steps.py pattern must match (positive); an invented step line must match
nothing (negative). If either control fails, the run measured this checker, not the suite.
Usage: python tests/test-server-bdd-steps.py [<fork tree, default: this checkout>] [<out file, default: stdout>]   exit 0 = PASS
PROGRESS-OK: parses ~10 files, instant."""
import ast, glob, os, re, sys

tree = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[2] if len(sys.argv) > 2 else None
feat = os.path.join(tree, "tools", "server", "tests", "features")
sdir = os.path.join(feat, "steps")
lines, fails = [], []
PLACEHOLDER = re.compile(r"(\{[^}]*\})")
OUTLINE_PARAM = re.compile(r"<[^>]+>")
KW = re.compile(r"^\s*(Given|When|Then|And|But)\s+(.*?)\s*$")
STEP_DECOS = {"step", "given", "when", "then"}


def say(s):
    lines.append(s)


def parse(path):
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read())


# ---- A. imports
mods = sorted(glob.glob(os.path.join(sdir, "*.py")))
say("step modules: %s" % ", ".join(os.path.basename(m) for m in mods))
steps_py = os.path.join(sdir, "steps.py")
defined = set()
if os.path.isfile(steps_py):
    for n in parse(steps_py).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Assign):
            defined |= {x.id for x in n.targets if isinstance(x, ast.Name)}
for m in mods:
    t = parse(m)
    imports_steps = any(isinstance(n, ast.Import) and any(a.name == "steps" for a in n.names) for n in ast.walk(t))
    if not imports_steps:
        continue
    used = {n.attr for n in ast.walk(t)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "steps"}
    if not os.path.isfile(steps_py):
        fails.append("A %s: `import steps` but %s does not exist" % (os.path.basename(m), steps_py))
        continue
    missing = sorted(used - defined)
    say("A %s: import steps OK, uses %d names, missing %d %s" % (os.path.basename(m), len(used), len(missing), missing))
    if missing:
        fails.append("A %s: steps.py lacks %s" % (os.path.basename(m), missing))

# ---- B. step patterns
pats = []
for m in mods:
    for n in ast.walk(parse(m)):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in n.decorator_list:
            if (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in STEP_DECOS and d.args
                    and isinstance(d.args[0], ast.Constant) and isinstance(d.args[0].value, str)):
                p = d.args[0].value
                rx = "".join(".+?" if part.startswith("{") else re.escape(part)
                             for part in PLACEHOLDER.split(p) if part != "")
                pats.append((os.path.basename(m), p, re.compile("^" + rx + "$")))
say("step patterns: %d" % len(pats))


def matches(text):
    return [p for p in pats if p[2].match(text)]


pos = None
for mname, p, _ in pats:
    if mname == "responses_steps.py":
        pos = PLACEHOLDER.sub("1", p)
        break
neg = "a zz invented step that nothing defines 12345"
pos_ok, neg_ok = bool(pos and matches(pos)), not matches(neg)
say("CONTROL positive %r -> %s | negative -> %s" % (pos, pos_ok, neg_ok))
for f in sorted(glob.glob(os.path.join(feat, "*.feature"))):
    n = und = 0
    with open(f, encoding="utf-8") as fh:
        flines = fh.read().splitlines()
    for i, ln in enumerate(flines, 1):
        k = KW.match(ln)
        if not k:
            continue
        n += 1
        text = k.group(2)
        if text.endswith(":"):
            text = text[:-1]
        text = OUTLINE_PARAM.sub("1", text)
        if not matches(text):
            und += 1
            if und <= 5:
                fails.append("B %s:%d undefined step: %s" % (os.path.basename(f), i, k.group(2)))
    say("B %s: %d step lines, %d undefined" % (os.path.basename(f), n, und))
    if und > 5:
        fails.append("B %s: ... %d undefined in all" % (os.path.basename(f), und))

ok = pos_ok and neg_ok and not fails
report = "\n".join(lines + ["FAIL: " + x for x in fails] + ["RESULT: %s" % ("PASS" if ok else "FAIL")]) + "\n"
if out:
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)
else:
    sys.stdout.write(report)
sys.exit(0 if ok else 1)
