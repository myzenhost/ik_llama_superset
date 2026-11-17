"""Fork MODULE / TYPE-ID wiring gate: built != reachable.

WHY. Several cloud-branch modules (KIVI cache, koboldcpp port, cram tier) and the MX type ids 70-73 were merged
built and unit-tested while NOTHING in the inference path called them (row S69-MXFP6-MXFP8-INTEGRATION-FTYPE-GGUFPY-SIMD-MSVC-BUILD-2026-10-08-01).

METHOD (static, textual)
  1. MODULES: for every .cpp under src/ and common/ collect its exported symbols (non-static function definitions at
     column 0, struct/class names, plus the same from its paired header <stem>.h). A module is WIRED when some other
     file (not the module's own .cpp/.h, not under tests/) references one of those symbols. Comments/strings are blanked.
  2. TYPE IDS: every `GGML_TYPE_<NAME>` enumerator in ggml/include/ggml.h (except *_COUNT) must be referenced in the
     application layer (src/ common/ tools/ examples/ include/), not only in ggml/ and tests/.
  3. scripts/modules-wired-allowlist.txt: lines `module:<path>|<task-row>|<reason>` or `type:<GGML_TYPE_X>|<task-row>|<reason>`.
     An entry that is no longer needed (the thing IS wired) is reported STALE and fails, so the list cannot rot.
LIMITS: matching is by NAME, not by type or overload; a symbol reached only through a macro or a registry reads as unwired
(allowlist it with a reason); a generic name used elsewhere makes a module look wired.
EXIT: 0 ok | 1 unwired / stale entry | 2 untrustworthy (a control failed)
CONTROLS: --selftest builds a scratch tree: a wired module passes (positive), a planted unwired module and type id fail (negative).
Usage: python scripts/check-fork-modules-wired.py [--root <fork>] [--selftest]
"""
import argparse, os, re, sys, tempfile

EXT = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cu", ".cuh", ".m", ".mm", ".inc")
APP_DIRS = ("src", "common", "tools", "examples", "include")
MOD_DIRS = ("src", "common")
ALLOW = os.path.join("scripts", "modules-wired-allowlist.txt")

_STRIP = re.compile(r'//[^\n]*|/\*.*?\*/|R"([^(\s]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)
_FUNC = re.compile(r'^(?!static\b|typedef\b|using\b|return\b|#|\s|\}|namespace\b)[A-Za-z_][\w:<>,\*&\s]*?[\s\*&]((?:\w+::)*[~A-Za-z_]\w*)\s*\([^;]*$', re.M)
_TYPE = re.compile(r'\b(?:struct|class)\s+([A-Za-z_]\w+)\s*(?:final\s*)?[:{]')
_CTOR = re.compile(r'^(\w+)::~?\1\s*\(', re.M)
_NS = re.compile(r'^namespace\s+([A-Za-z_]\w{3,})\s*\{', re.M)
_SKIPNAMES = {"main", "if", "for", "while", "switch", "sizeof", "operator"}


def read(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return _STRIP.sub(lambda m: " " * len(m.group(0)) if "\n" not in m.group(0) else "\n" * m.group(0).count("\n"), f.read())
    except OSError:
        return ""


def walk(root, dirs):
    for d in dirs:
        for dp, dn, fn in os.walk(os.path.join(root, d)):
            dn[:] = [x for x in dn if x not in ("tests", "build", ".git")]
            for f in fn:
                if f.endswith(EXT):
                    yield os.path.join(dp, f)


def symbols(text):
    out = set()
    for m in _FUNC.finditer(text):
        n = m.group(1).split("::")[-1].lstrip("~")
        if n not in _SKIPNAMES and len(n) >= 4:
            out.add(n)
    for m in _TYPE.finditer(text):
        out.add(m.group(1))
    for rx in (_CTOR, _NS):
        for m in rx.finditer(text):
            out.add(m.group(1))
    return out


def load_allow(root):
    mods, types = {}, {}
    p = os.path.join(root, ALLOW)
    if not os.path.exists(p):
        return mods, types, []
    bad = []
    for ln, line in enumerate(open(p, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) != 3 or not parts[1] or not parts[2] or ":" not in parts[0]:
            bad.append("%s:%d malformed (need kind:name|task-row|reason)" % (ALLOW, ln)); continue
        kind, name = parts[0].split(":", 1)
        (mods if kind == "module" else types)[name] = (parts[1], parts[2])
    return mods, types, bad


def run(root, verbose=True):
    files = {p: read(p) for p in walk(root, APP_DIRS)}
    ggml_dir = os.path.join(root, "ggml")
    # word-set per file, built once
    words = {p: set(re.findall(r"[A-Za-z_]\w*", t)) for p, t in files.items()}
    mods_allow, types_allow, bad = load_allow(root)
    problems = list(bad)
    used_allow = set()
    wired, unwired = [], []
    for p in sorted(files):
        # A file configured from <name>.in (common/build-info.cpp, git-ignored) is build output, not a module.
        if not p.endswith((".cpp", ".cc", ".cxx", ".c")) or os.path.relpath(p, root).split(os.sep)[0] not in MOD_DIRS or os.path.exists(p + ".in"):
            continue
        stem = os.path.splitext(p)[0]
        own = {p} | {stem + e for e in (".h", ".hpp")}
        syms = symbols(files[p])
        for h in own:
            if h in files:
                syms |= symbols(files[h])
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        if not syms:
            hit = None  # nothing extractable: cannot judge, treat as unwired so a human allowlists it with a reason
        else:
            hit = next((q for q in files if q not in own and words[q] & syms), None)
        if hit:
            wired.append(rel)
        elif rel in mods_allow:
            used_allow.add(("module", rel))
        else:
            unwired.append(rel)
    for rel in mods_allow:
        if rel in wired:
            problems.append("STALE allowlist entry module:%s (it is wired now; remove the entry)" % rel)
        elif ("module", rel) not in used_allow:
            problems.append("allowlist entry module:%s names no known module" % rel)
    for u in unwired:
        problems.append("UNWIRED module %s: no caller of its symbols outside tests/ and its own files" % u)
    # type ids
    ids = []
    gh = read(os.path.join(root, "ggml", "include", "ggml.h"))
    enum = re.search(r"enum\s+ggml_type\s*\{(.*?)\};", gh, re.S)
    if enum:
        ids = [n for n in re.findall(r"\b(GGML_TYPE_\w+)\b\s*=\s*\d+", enum.group(1)) if not n.endswith("_COUNT")]
    app_words = set().union(*words.values()) if words else set()
    tw = 0
    for t in ids:
        if t in app_words:
            if t in types_allow:
                problems.append("STALE allowlist entry type:%s (it is referenced now; remove the entry)" % t)
            tw += 1
        elif t in types_allow:
            used_allow.add(("type", t))
        else:
            problems.append("UNREFERENCED type id %s: no use in src/ common/ tools/ examples/ include/" % t)
    for t in types_allow:
        if t not in ids:
            problems.append("allowlist entry type:%s names no known GGML_TYPE" % t)
    if verbose:
        print("modules checked: %d (wired %d, allowlisted %d, unwired %d); type ids: %d (referenced %d); files indexed: %d"
              % (len(wired) + len(unwired) + len(used_allow & {m for m in used_allow if m[0] == 'module'}),
                 len(wired), sum(1 for m in used_allow if m[0] == "module"), len(unwired), len(ids), tw, len(files)))
        for pr in problems:
            print("FAIL", pr)
        if not problems:
            print("OK")
    return problems, wired, ids


def selftest():
    with tempfile.TemporaryDirectory() as d:
        def w(rel, txt):
            p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(txt)
        w("ggml/include/ggml.h", "enum ggml_type {\n GGML_TYPE_F32 = 0,\n GGML_TYPE_ORPHAN = 9,\n GGML_TYPE_COUNT = 10,\n};\n")
        w("src/good.h", "int good_entry_point(int x);\n")
        w("src/good.cpp", '#include "good.h"\nint good_entry_point(int x) {\n return x;\n}\n')
        w("src/main_use.cpp", '#include "good.h"\nint use_it() {\n return good_entry_point(1) + GGML_TYPE_F32;\n}\n')
        w("tools/main.cpp", "int use_it();\nint main() { return use_it(); }\n")
        w("src/orphan.cpp", "int orphan_unreachable_fn(int x) {\n return x;\n}\n")
        w("tests/t.cpp", "int z = orphan_unreachable_fn(1) + GGML_TYPE_ORPHAN;\n")  # tests/ must NOT count
        pr, wired, ids = run(d, verbose=False)
        ok = True
        if "src/good.cpp" not in wired:
            print("SELFTEST FAIL positive control: wired module not recognised"); ok = False
        if not any("UNWIRED module src/orphan.cpp" in x for x in pr):
            print("SELFTEST FAIL negative control: planted unwired module not caught"); ok = False
        if not any("UNREFERENCED type id GGML_TYPE_ORPHAN" in x for x in pr):
            print("SELFTEST FAIL negative control: planted unreferenced type id not caught"); ok = False
        if any("GGML_TYPE_F32" in x for x in pr):
            print("SELFTEST FAIL positive control: referenced type id flagged"); ok = False
        w("scripts/modules-wired-allowlist.txt", "module:src/orphan.cpp|ROW-1|planted\ntype:GGML_TYPE_ORPHAN|ROW-1|planted\n")
        pr, _, _ = run(d, verbose=False)
        if pr:
            print("SELFTEST FAIL allowlist did not silence planted entries:", pr); ok = False
        w("scripts/modules-wired-allowlist.txt", "module:src/good.cpp|ROW-1|stale\n")
        pr, _, _ = run(d, verbose=False)
        if not any("STALE" in x for x in pr):
            print("SELFTEST FAIL stale allowlist entry not caught"); ok = False
        print("SELFTEST", "OK" if ok else "FAILED")
        return 0 if ok else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    pr, _, _ = run(a.root)
    sys.exit(1 if pr else 0)
