"""Fork CLI-flag WIRING gate: every REGISTERED flag must reach a behaviour (or say, at parse time, that it does not).

WHY (rule 1.59 -- prevention). check_fork_cli_flags_registered.py proves a flag is ACCEPTED by the parser. That is not
the same as the flag DOING anything: the --spec-replace failure (S69-SPEC-REPLACE-BUILT-BUT-UNREACHABLE-2026-09-14-01) and
the -ser failure found 2026-10-01 (row S69-FLAG-WIRING-SER-NO-READER-2026-10-01-01) are flags whose params field is
written by the parser, copied into cparams, printed in a log line -- and READ BY NOTHING. check_params_parsed_but_unconsumed.py
cannot see this class any more: it scans common.cpp/arg.cpp only for `params.X =` and counts ANY mention outside the parser as
consumption, and the restored flags live in common/arg-fork-restored.inc (not scanned), and a log line or a cparams copy
is a "mention". This gate replaces "mentioned" with "read by something that is not a copy or a print".

METHOD (static, textual -- stated limits below)
  1. Parse every `add_opt(common_arg({flags...}, ... [](common_params & params ...) {handler}))` in common/arg.cpp and
     common/*.inc. Record the dotted `params.<path>` fields the handler touches.
  2. Index every member access `.leaf` / `->leaf` in the COMPILED source dirs (common, src, include, app, and the
     tools/ + examples/ subdirectories that CMake add_subdirectory()s). Comments and string literals are blanked first.
     Option handler bodies inside the parser are masked, so only the parser's OTHER code counts as a reader.
  3. A field is LIVE when some line reads it that is not (a) a print/log line (incl. multi-line LOG calls), (b) a plain
     `x.dest = ...field...;` copy -- a copy is followed: dest is LIVE if dest is read, depth <= 4 -- or (c) a file named
     like a wiring check / selftest. A call inside the handler that passes `params` (or the field) to a non-trivial
     function (e.g. common_log_set_json_format(..., params.log_json)) also counts.
  4. Verdict per REGISTERED baseline flag:
       WIRED       some written field is LIVE, or the handler calls something with params
       REFUSE      the handler throws / arg_removed(): the user is told at parse time
       DECLARED    nothing reads the field BUT the handler warns "no effect / deprecated / ignored / no longer"
       NO-READER   a field is written and nothing reads it, no warning      -> FAIL
       NO-EFFECT   the handler touches nothing and says nothing             -> FAIL
BASELINE: <root>/.fork-cli-flags-baseline.txt. The gate is ACTIVE only when that file contains the line
    # WIRING-GATE: on
(a tree without the marker has not adopted the gate: reported loudly, rc 0 -- not a pass, not a fail).
LIMITS (the gate points at a suspect, it does not deliver a verdict): matching is by member NAME, not type, so a generic
leaf (n_ctx) read anywhere makes a flag look live; a field consumed through a macro / reflection table / a bare-name
member function reads as dead; positional function arguments count as reads. Every finding prints the option group
and the fields that were examined. COVERAGE IS PRINTED (options parsed, flags checked, files indexed).
EXIT: 0 ok | 1 flag(s) with no reader | 2 untrustworthy (a control failed / source unreadable)
CONTROLS: --ctx-size must be WIRED (positive); a planted reader-less flag must be caught (--selftest and --plant).
Usage: python scripts/check-fork-cli-flags-wired.py [--root <fork>] [--table out.md] [--plant file.inc] [--selftest]
Also run as ctest `test-cli-flags-wired` / `test-cli-flags-wired-selftest` (tests/CMakeLists.txt) and by the fork pre-commit
hook ("cli-flags-wired") through cortex/scripts/check_fork_cli_flags_wired.py.
GATE-CLAIM-OK: static analysis, nothing declared unavailable.
SINGLE-SHOT-OK: one pass over ~700 source files, seconds.
PROGRESS-OK: instant (index built once).
# REWRITE-SAFE: reads source files only; writes only the --table output path given on the command line.
"""
import argparse, collections, glob, os, re, sys, tempfile

# This file lives in <fork>/scripts/, so the default tree is its parent: the gate travels with the tree it checks.
# cortex/scripts/check_fork_cli_flags_wired.py is a thin wrapper the fork pre-commit hook calls; it runs THIS file from
# the tree being committed (a tree without this file has not adopted the gate).
DEFAULT_FORK = os.environ.get("IK_FORK_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BASELINE_NAME = ".fork-cli-flags-baseline.txt"
GATE_MARKER = "# WIRING-GATE: on"
POS_CONTROL = "--ctx-size"
EXT = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".m", ".mm")
SKIP_TOP = ("ggml", "vendor", "docs", "scripts", "models", "pocs", "benches", "ci", "build", ".git", "tests")

# ---------------------------------------------------------------- source cleaning
_STRIP_RE = re.compile(
    r'//[^\n]*|/\*.*?\*/|R"([^(\s]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)


def _blank(m):
    t = m.group(0)
    if t.startswith('//') or t.startswith('/*') or t[0] == 'R':
        return ''.join('\n' if ch == '\n' else ' ' for ch in t)
    return t[0] + '_' * (len(t) - 2) + t[0]


def strip_src(text):
    """blank comments, raw strings, string/char literal CONTENT; offsets and newlines preserved."""
    return _STRIP_RE.sub(_blank, text)


def match_paren(s, i):
    d = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == '(':
            d += 1
        elif c == ')':
            d -= 1
            if d == 0:
                return j
    return -1


# ---------------------------------------------------------------- option parsing
def parser_files(root, extra=()):
    fs = [os.path.join(root, "common", "arg.cpp")] + sorted(glob.glob(os.path.join(root, "common", "*.inc")))
    return fs + list(extra)


def parse_options(root, extra=()):
    """-> (options, masks). option: flags, file, line, handler (stripped), raw_handler. masks: file -> [(a,b)]"""
    opts, masks = [], {}
    for p in parser_files(root, extra):
        try:
            raw = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        clean = strip_src(raw)
        rel = os.path.relpath(p, root).replace(os.sep, "/") if p.startswith(root) else os.path.basename(p)
        for m in re.finditer(r'\badd_opt\s*\(', clean):
            o = m.end() - 1
            c = match_paren(clean, o)
            if c < 0:
                continue
            blk_c, blk_r = clean[o:c + 1], raw[o:c + 1]
            fm = re.match(r'\(\s*common_arg\s*\(\s*((?:\{[^{}]*\}\s*,\s*)+)', blk_c)
            if not fm:
                continue
            flags = re.findall(r'"((?:[^"\\]|\\.)*)"', blk_r[fm.start(1):fm.end(1)])
            lm = re.search(r'\[[^\]]*\]\s*\(\s*common_params', blk_c)
            hc = blk_c[lm.start():] if lm else ""
            hr = blk_r[lm.start():] if lm else ""
            opts.append(dict(flags=flags, file=rel, line=raw.count('\n', 0, o) + 1, handler=hc, raw_handler=hr))
            masks.setdefault(rel, []).append((o, c + 1))
    return opts, masks


_METHODS = {"push_back", "emplace_back", "insert", "resize", "empty", "size", "c_str", "back", "front", "assign",
            "clear", "pop_back", "begin", "end", "append", "data", "reserve", "find", "erase", "substr", "at",
            "swap", "emplace", "count", "reset", "get", "has_value", "value", "length", "str", "compare"}


def fields_written(handler):
    paths = []
    for m in re.finditer(r'\bparams\b((?:\s*\.\s*[A-Za-z_]\w*)+)', handler):
        parts = [x.strip() for x in m.group(1).split('.') if x.strip()]
        popped = False
        while len(parts) > 1 and parts[-1] in _METHODS:
            parts.pop()
            popped = True
        if handler[m.end():m.end() + 3].lstrip().startswith('(') and parts and not popped:
            parts.pop()
        if parts:
            paths.append('.'.join(parts))
    out, seen = [], set()
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


_TRIVIAL_CALLS = {"std::stoi", "std::stof", "std::stoul", "std::stol", "std::stoll", "std::stod", "string_format",
                  "string_split", "string_split_pairs", "read_file", "fprintf", "printf", "string_lower",
                  "kv_cache_type_from_str", "std::string", "std::make_tuple", "std::min", "std::max", "std::all_of",
                  "static_cast", "if", "while", "for", "switch", "return", "sizeof", "std::move", "params"}
_WARN_RE = re.compile(r'(no effect|deprecated|ignored|no longer|not supported|REMOVED)', re.I)


def handler_effect(clean_handler, paths):
    """True when the handler itself acts: passes params (or a written field) to a non-trivial call."""
    if re.search(r'\b[A-Za-z_][\w:]*\s*\(\s*params\b', clean_handler):
        return True
    for p in paths:
        for m in re.finditer(r'\b([A-Za-z_][\w:]*)\s*\([^;]*\bparams\.%s\b' % re.escape(p), clean_handler):
            if m.group(1) not in _TRIVIAL_CALLS and not m.group(1).startswith("LOG"):
                return True
    return False


_PRINT_TOK = re.compile(r'\b(?:fprintf|printf|LOG_[A-Z]+|LLAMA_LOG\w*|arg_removed)\b')


def _print_stmts(clean_handler):
    """[(start, end)] of every print/log statement; end found on the STRING-BLANKED text so a ';' inside a message is harmless."""
    spans = []
    for m in _PRINT_TOK.finditer(clean_handler):
        e = clean_handler.find(';', m.end())
        spans.append((m.start(), len(clean_handler) if e < 0 else e + 1))
    return spans


def handler_declares_inert(clean_handler, raw_handler):
    """the handler (or the text it prints) tells the user the flag does nothing."""
    for (a, b) in _print_stmts(clean_handler):
        seg = raw_handler[a:b]
        if _WARN_RE.search(seg) or seg.startswith("arg_removed"):
            return True
    return False


_NOT_CALLS = {"if", "while", "for", "return", "switch", "sizeof", "static_cast", "catch", "GGML_UNUSED"}


def handler_body_acts(clean_handler):
    """no-field handlers (--version, -h ...): True when, print statements removed, the body still calls something."""
    if '{' not in clean_handler:
        return False
    base = clean_handler.index('{') + 1
    body = list(clean_handler[base:])
    for (a, b) in _print_stmts(clean_handler):
        for k in range(max(a - base, 0), min(b - base, len(body))):
            body[k] = ' '
    text = "".join(body)
    return any(m.group(1) not in _NOT_CALLS for m in re.finditer(r'\b([A-Za-z_][\w:]*)\s*\(', text))


# ---------------------------------------------------------------- source index
def built_dirs(root):
    ok = {"common", "src", "include", "app"}
    for base in ("tools", "examples"):
        cm = os.path.join(root, base, "CMakeLists.txt")
        if os.path.exists(cm):
            t = re.sub(r'(?m)^\s*#.*$', '', open(cm, encoding="utf-8", errors="replace").read())
            for m in re.finditer(r'add_subdirectory\(\s*([\w\-./]+)\s*\)', t):
                ok.add(base + "/" + m.group(1))
    return ok


CHECK_FILE_RE = re.compile(r'(wiring-check|selftest|/test-|-test/|-check/|probe-|offload-attn-test)')
PRINT_RE = re.compile(r'\b(printf|fprintf|snprintf|LOG_[A-Z]+|LLAMA_LOG\w*|LOG|COM_\w+|SRV_\w+|string_format|GGML_LOG\w*|GGML_UNUSED|get_env)\b')
IF_RE = re.compile(r'\b(if|while|for|switch|return|assert|GGML_ASSERT)\b')
_ACC = re.compile(r'(?:\.|->)\s*([A-Za-z_]\w*)')


class Index:
    def __init__(self, root, masks):
        self.root, self.lines, self.acc, self.n_files = root, {}, collections.defaultdict(list), 0
        ok = built_dirs(root)
        for dp, dns, fns in os.walk(root):
            rel = os.path.relpath(dp, root).replace(os.sep, "/")
            dns[:] = [d for d in dns if not d.startswith("build") and d not in (".git", "CMakeFiles", "node_modules")]
            if rel == ".":
                dns[:] = [d for d in dns if d in ("common", "src", "include", "app", "tools", "examples")]
                continue
            top = rel.split("/")[0]
            if top in SKIP_TOP:
                continue
            if top in ("tools", "examples"):
                parts = rel.split("/")
                if len(parts) >= 2 and "/".join(parts[:2]) not in ok:
                    dns[:] = []
                    continue
            for fn in fns:
                if not fn.endswith(EXT):
                    continue
                if rel == "common" and fn == "arg.cpp":
                    continue      # indexed below with its option handlers masked out
                f = rel + "/" + fn
                try:
                    raw = open(os.path.join(dp, fn), encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                self._add(f, strip_src(raw))
        # the parser itself: only the code OUTSIDE option blocks counts as a reader
        fp = os.path.join(root, "common", "arg.cpp")
        if os.path.exists(fp):
            raw = open(fp, encoding="utf-8", errors="replace").read()
            clean = list(strip_src(raw))
            for (a, b) in masks.get("common/arg.cpp", []):
                for k in range(a, min(b, len(clean))):
                    if clean[k] != '\n':
                        clean[k] = ' '
            self._add("common/arg.cpp#parser-outside-options", "".join(clean))

    def _add(self, f, clean):
        ls = clean.split('\n')
        self.lines[f] = ls
        self.n_files += 1
        for i, ln in enumerate(ls, 1):
            if '.' in ln or '->' in ln:
                for m in _ACC.finditer(ln):
                    self.acc[m.group(1)].append((f, i, m.end()))

    def reads(self, leaf):
        """[(file, lineno, text)] member-access occurrences of leaf that are not plain assignments TO it."""
        out = []
        for (f, i, end) in self.acc.get(leaf, ()):
            ln = self.lines[f][i - 1]
            after = ln[end:].lstrip()
            if re.match(r'(=(?!=)|\+=|-=|\|=|&=|\*=|/=)', after) and not after.startswith("=="):
                continue
            out.append((f, i, ln.strip()[:150]))
        return out


def classify_line(idx, f, i, leaf):
    ln = idx.lines[f][i - 1]
    acc = r'(?:\.|->)\s*%s\b' % re.escape(leaf)
    m = re.match(r'\s*(\w+)\s*\(\s*[\w\.\->\[\]]*' + acc + r'\s*\)\s*,?\s*$', ln)   # init-list copy  dest(x.leaf),
    if m and not PRINT_RE.search(ln):
        return "copy", m.group(1)
    m = re.match(r'\s*([\w\.\->\[\]\*\(\)]*(?:\.|->))(\w+)\s*=(?!=)\s*[^;]*' + acc, ln)   # member copy x.dest = ..leaf..
    if m and not IF_RE.match(ln.strip()) and not PRINT_RE.search(ln):
        return "copy", m.group(2)
    # the field is an ARGUMENT of a real call inside a log line (ggml_backend_prefetch_init(params.n) ? "a" : "b"):
    # the call runs, so it is a use. Accessors that only format a value do not count.
    for cm in re.finditer(r'\b([A-Za-z_][\w:]*)\s*\([^()]*' + acc + r'[^()]*\)', ln):
        nm = cm.group(1)
        if not PRINT_RE.search(nm) and not re.search(r'(_name|name|to_string|c_str|str|string_format|static_cast|min|max)$', nm):
            return "use", None
    if PRINT_RE.search(ln):
        return "print", None
    j = i - 2
    while j >= 0 and i - 2 - j < 10:       # continuation line of a multi-line print/log call
        prev = idx.lines[f][j].rstrip()
        if prev.endswith(";") or prev.endswith("{") or prev.endswith("}"):
            break
        if PRINT_RE.search(prev):
            return "print", None
        j -= 1
    return "use", None


def liveness(idx, leaf, max_depth=4):
    seen, work, uses, copies = {leaf}, [(leaf, 0, [leaf])], [], []
    while work:
        cur, d, path = work.pop(0)
        for (f, i, txt) in idx.reads(cur):
            if CHECK_FILE_RE.search(f):
                continue
            c, dest = classify_line(idx, f, i, cur)
            if c == "use":
                uses.append((f.replace("#parser-outside-options", ""), i, txt, "->".join(path)))
            elif c == "copy" and dest not in seen and d < max_depth:
                seen.add(dest)
                work.append((dest, d + 1, path + [dest]))
                copies.append(dest)
    return dict(live=bool(uses), n_uses=len(uses), uses=uses[:5], copies=copies[:6])


# ---------------------------------------------------------------- baseline
def load_baseline(path):
    base, gate = {}, False
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        s = line.strip()
        if s == GATE_MARKER:
            gate = True
        if not s or s.startswith("#"):
            continue
        parts = s.split(None, 2)
        if len(parts) < 2 or parts[1] not in ("REGISTERED", "OPEN"):
            raise ValueError("%s:%d: expected '<flag> REGISTERED|OPEN [reason]', got %r" % (path, n, line.rstrip()))
        base[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return base, gate


# ---------------------------------------------------------------- analysis
def analyse(root, plant=None):
    opts, masks = parse_options(root, [plant] if plant else [])
    idx = Index(root, masks)
    flag2opt = {}
    for o in opts:
        for f in o["flags"]:
            flag2opt.setdefault(f, o)
    for o in opts:
        paths = fields_written(o["handler"])
        eff = handler_effect(o["handler"], paths)
        throws = bool(re.search(r'\bthrow\b|\barg_removed\b', o["handler"]))
        fl = [dict(path=p, leaf=p.split('.')[-1], live=liveness(idx, p.split('.')[-1])) for p in paths]
        if not o["handler"]:
            verdict = "UNPARSED"
        elif throws and not fl:      # a throw next to field writes is value VALIDATION, not a refusal
            verdict = "REFUSE"
        elif any(f["live"]["live"] for f in fl) or eff or (not fl and handler_body_acts(o["handler"])):
            verdict = "WIRED"
        elif handler_declares_inert(o["handler"], o["raw_handler"]):
            verdict = "DECLARED"
        elif fl:
            verdict = "NO-READER"
        else:
            verdict = "NO-EFFECT"
        o["verdict"], o["fields"], o["effect_call"] = verdict, fl, eff
    return opts, flag2opt, idx


def check(root, plant=None, table=None):
    bpath = os.path.join(root, BASELINE_NAME)
    if not os.path.exists(bpath):
        print("[cli-wired] NOT ADOPTED: no %s at %s (nothing checked; not a pass, not a fail)." % (BASELINE_NAME, root))
        return 0
    try:
        base, gate = load_baseline(bpath)
    except (OSError, ValueError) as e:
        print("[cli-wired] UNTRUSTWORTHY: baseline unreadable: %s" % e)
        return 2
    if not gate:
        print("[cli-wired] NOT ADOPTED in this tree: %s lacks the line '%s'. Nothing is checked here "
              "(not a pass, not a fail); add the line once every REGISTERED flag is wired or declared inert." % (BASELINE_NAME, GATE_MARKER))
        return 0
    if not os.path.exists(os.path.join(root, "common", "arg.cpp")):
        print("[cli-wired] UNTRUSTWORTHY: no common/arg.cpp under %s" % root)
        return 2
    opts, flag2opt, idx = analyse(root, plant)
    flags = {f for f, (st, _) in base.items() if st == "REGISTERED"}
    if plant:     # planted flags are checked as if REGISTERED
        for o in opts:
            if o["file"] == os.path.basename(plant):
                flags |= set(o["flags"])
    if idx.n_files < 50 or len(opts) < 5:
        print("[cli-wired] UNTRUSTWORTHY: indexed %d files, parsed %d options -- far below the expected ~700 / ~500" % (idx.n_files, len(opts)))
        return 2
    pc = flag2opt.get(POS_CONTROL)
    if pc is None or pc["verdict"] != "WIRED":
        print("[cli-wired] UNTRUSTWORTHY: positive control %s is %s, expected WIRED" % (POS_CONTROL, pc["verdict"] if pc else "absent"))
        return 2
    bad, counts = [], collections.Counter()
    for f in sorted(flags):
        o = flag2opt.get(f)
        if o is None:
            continue    # presence is check_fork_cli_flags_registered.py's job
        counts[o["verdict"]] += 1
        if o["verdict"] in ("NO-READER", "NO-EFFECT"):
            bad.append((f, o))
    print("tree        : %s" % root)
    print("coverage    : %d options parsed, %d source files indexed, %d REGISTERED flags checked (%s)"
          % (len(opts), idx.n_files, sum(counts.values()), ", ".join("%s=%d" % kv for kv in sorted(counts.items()))))
    shown = set()
    for f, o in bad:
        key = (o["file"], o["line"])
        if key in shown:
            continue
        shown.add(key)
        al = ",".join(o["flags"])
        if o["verdict"] == "NO-READER":
            fs = "; ".join("%s (copies->%s)" % (x["path"], ",".join(x["live"]["copies"]) or "none") for x in o["fields"])
            print("  NO-READER  %-34s %s:%d writes %s -- nothing reads it (only copies/prints) and the handler does not say 'no effect'"
                  % (al, o["file"], o["line"], fs))
        else:
            print("  NO-EFFECT  %-34s %s:%d -- the handler writes nothing, throws nothing, prints nothing" % (al, o["file"], o["line"]))
    if table:
        write_table(table, base, flag2opt)
    if bad:
        print("[cli-wired] FAILED: %d option group(s) accept a value and change nothing. Wire a reader, or make the handler say "
              "'no effect' at parse time (then it is DECLARED, visible to the user)." % len(shown))
        return 1
    print("[cli-wired] OK: every REGISTERED flag is WIRED, REFUSEs with a hint, or DECLAREs itself inert")
    return 0


def write_table(path, base, flag2opt):
    rows = ["| flag | group aliases | handler | fields written | live use (first) | verdict |", "|---|---|---|---|---|---|"]
    for f in sorted(base):
        o = flag2opt.get(f)
        if o is None:
            rows.append("| %s | | NOT FOUND | | | MISSING |" % f)
            continue
        fl = "; ".join(x["path"] for x in o["fields"]) or "-"
        use = "-"
        for x in o["fields"]:
            if x["live"]["uses"]:
                u = x["live"]["uses"][0]
                use = "%s:%d" % (u[0], u[1])
                break
        if use == "-" and o["effect_call"]:
            use = "handler call"
        rows.append("| %s | %s | %s:%d | %s | %s | %s |" % (f, ",".join(o["flags"]), o["file"], o["line"], fl, use, o["verdict"]))
    open(path, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print("table       : %s (%d rows)" % (path, len(rows) - 2))


# ---------------------------------------------------------------- selftest
_MINI_ARG = r'''
void init(common_params & params) {
    add_opt(common_arg(
        {"-c", "--ctx-size"}, "N", "ctx",
        [](common_params & params, int value) { params.n_ctx = value; }));
    add_opt(common_arg(
        {"--zz-dead"}, "N", "dead flag",
        [](common_params & params, int value) { params.zz_dead = value; }));
    add_opt(common_arg(
        {"--zz-printed"}, "N", "printed only",
        [](common_params & params, int value) { params.zz_printed = value; }));
    add_opt(common_arg(
        {"--zz-noop"}, "N", "declared inert",
        [](common_params & params, int value) { params.zz_noop = value; fprintf(stderr, "warning: --zz-noop has no effect\n"); }));
    add_opt(common_arg(
        {"--zz-refuse"}, "N", "removed",
        [](common_params &, int) { throw std::invalid_argument("was replaced"); }));
    add_opt(common_arg(
        {"--zz-empty"}, "N", "does nothing",
        [](common_params &, int) { }));
}
'''
_MINI_CONSUMER = '''
void use(common_params & params, llama_context_params & cp) {
    cp.n_ctx = params.n_ctx;                 // copy
    run(cp.n_ctx);                           // read of the copy
    cp.zz_printed = params.zz_printed;       // copy only ...
    LLAMA_LOG_INFO("%d", cp.zz_printed);     // ... and a print
}
'''


def selftest():
    fails = []

    def ok(name, cond):
        print("%-4s %s" % ("ok" if cond else "FAIL", name))
        if not cond:
            fails.append(name)

    t = tempfile.mkdtemp(prefix="cliwired_")
    os.makedirs(os.path.join(t, "common"))
    os.makedirs(os.path.join(t, "src"))
    for k in range(60):      # reach floor: the gate refuses to trust a near-empty tree
        open(os.path.join(t, "src", "f%d.cpp" % k), "w").write("int f%d() { return 0; }\n" % k)
    open(os.path.join(t, "common", "arg.cpp"), "w").write(_MINI_ARG)
    open(os.path.join(t, "src", "use.cpp"), "w").write(_MINI_CONSUMER)
    base = os.path.join(t, BASELINE_NAME)
    names = ["--ctx-size", "--zz-dead", "--zz-printed", "--zz-noop", "--zz-refuse", "--zz-empty"]
    open(base, "w").write("\n".join("%s REGISTERED" % n for n in names) + "\n")
    ok("tree without the WIRING-GATE marker is 'not adopted' (rc 0)", check(t) == 0)
    open(base, "a").write(GATE_MARKER + "\n")
    opts, f2o, idx = analyse(t)
    v = {n: f2o[n]["verdict"] for n in names}
    ok("positive control --ctx-size is WIRED (copy chain n_ctx -> cp.n_ctx read by run())", v["--ctx-size"] == "WIRED")
    ok("a field written and read by nothing -> NO-READER", v["--zz-dead"] == "NO-READER")
    ok("a field copied and only PRINTED -> NO-READER (a print is not a reader)", v["--zz-printed"] == "NO-READER")
    ok("a handler that says 'no effect' -> DECLARED", v["--zz-noop"] == "DECLARED")
    ok("a handler that throws -> REFUSE", v["--zz-refuse"] == "REFUSE")
    ok("a handler that does nothing and says nothing -> NO-EFFECT", v["--zz-empty"] == "NO-EFFECT")
    ok("gate fails (rc 1) on the dead / printed-only / empty flags", check(t) == 1)
    open(os.path.join(t, "src", "use.cpp"), "a").write("void more(common_params & params) { act(params.zz_dead); act(params.zz_printed); }\n")
    cur_base = open(base).read()      # read BEFORE truncating (open(..,'w') empties the file first)
    open(base, "w").write(cur_base.replace("--zz-empty REGISTERED\n", ""))
    ok("the marker survived the rewrite (control for the test itself)", GATE_MARKER in open(base).read())
    ok("after readers are added and the empty flag dropped the gate passes (rc 0)", check(t) == 0)
    open(base, "w").write("--ctx-size REGISTERED\n")
    ok("marker missing again -> not adopted rc 0", check(t) == 0)
    open(base, "w").write("--ctx-size REGISTERED\n" + GATE_MARKER + "\n")
    open(os.path.join(t, "src", "use.cpp"), "w").write("int unrelated() { return 1; }\n")
    ok("positive control not WIRED (its reader gone) -> rc 2 untrustworthy", check(t) == 2)
    print("\n%d failed" % len(fails))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=DEFAULT_FORK)
    ap.add_argument("--table", help="write the per-flag wiring table (markdown) to this path")
    ap.add_argument("--plant", help="an extra .inc file whose option groups are parsed and checked as REGISTERED (proves the gate on a real tree)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return check(os.path.abspath(a.root), plant=os.path.abspath(a.plant) if a.plant else None, table=a.table)


if __name__ == "__main__":
    sys.exit(main())
