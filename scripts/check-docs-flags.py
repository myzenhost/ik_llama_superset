"""docs <-> flags two-way gate (S69-ENGREV-FERROX-DOCS-FLAGS-TEST-2026-10-27-01).
CAPABILITY-GENUINELY-NA: a docs/flag lint script, not a model capability.

Method from ferrox crates/frink-cli/tests/documented_flags.rs (Apache-2.0; behaviour re-specified, no code copied).

DIRECTION 1  every long flag DECLARED in common/arg.cpp (first brace-list of each common_arg(...)) must appear,
             backticked, in some doc (docs/**/*.md, tools/*/README.md)  -- or be in scripts/docs_flags_known_undocumented.txt.
DIRECTION 2  every backticked --flag in docs/parameters.md must be ACCEPTED by some CLI source (a "--flag" string literal
             under common/ tools/ examples/ src/) -- or be in scripts/docs_flags_known_ghosts.txt (a flag the docs mention
             that no source accepts: either mentioned to say it is absent, or a doc defect awaiting triage).
BOUNDARY     a flag token is the whole run of [A-Za-z0-9_-]; `--draft-p-minSABOTAGE` documents NOTHING about `--draft-p-min`.
BASELINES    both known-* files are ratchets: a NEW undocumented/ghost flag fails, and so does a baseline entry that is no
             longer true (it must be deleted, so the lists only shrink).
VACUOUS      fewer than MIN_FLAGS declared flags parsed -> FAIL (a broken parser must not pass by finding nothing).
--selftest   runs the must-fail / must-pass controls (a doc flag dropped, a made-up doc flag, boundary sabotage,
             stale baseline, vacuous parse) on in-memory text first.
--write-baseline   regenerate the two known-* files from the current tree (a reviewed act, never part of the gate).
EXIT: 0 ok | 1 finding(s) | 2 selftest failed / unreadable
"""
import argparse, glob, os, re, sys

MIN_FLAGS = 300
TOKEN = r"(--[A-Za-z0-9][A-Za-z0-9_-]*)"
DECL = re.compile(r"common_arg\(\s*\{([^}]*)\}")
LIT = re.compile(r'"' + TOKEN + r'"')
CODE = re.compile(r"`[^`\n]*`")


def declared_flags(arg_cpp_text):
    out = set()
    for m in DECL.finditer(arg_cpp_text):
        out.update(LIT.findall(m.group(1)))
    return out


def literal_flags(texts):
    out = set()
    for t in texts:
        out.update(LIT.findall(t))
    return out


def doc_flags(md_text):
    """Flags inside backticks. TOKEN is greedy over [A-Za-z0-9_-], so `--x-SABOTAGE` is the flag `--x-SABOTAGE`, never `--x-`."""
    out = set()
    for m in CODE.finditer(md_text):
        for f in re.finditer(r"(?<![A-Za-z0-9_-])" + TOKEN, m.group(0)):
            out.add(f.group(1))
    return out


def evaluate(declared, accepted, docs_all, docs_params, known_undoc, known_ghost, min_flags=None):
    f = []
    need = MIN_FLAGS if min_flags is None else min_flags
    if len(declared) < need:
        f.append("VACUOUS: only %d declared flag(s) parsed, need >= %d" % (len(declared), need))
    for x in sorted(declared - docs_all - known_undoc):
        f.append("UNDOCUMENTED: %s is declared in common/arg.cpp but no doc names it" % x)
    for x in sorted(docs_params - accepted - known_ghost):
        f.append("GHOST: %s is in docs/parameters.md but no CLI source accepts it" % x)
    for x in sorted(known_undoc - (declared - docs_all)):
        f.append("STALE-BASELINE: %s no longer undocumented; delete it from docs_flags_known_undocumented.txt" % x)
    for x in sorted(known_ghost - (docs_params - accepted)):
        f.append("STALE-BASELINE: %s no longer a ghost; delete it from docs_flags_known_ghosts.txt" % x)
    return f


def _read(p):
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load(root):
    declared = declared_flags(_read(os.path.join(root, "common", "arg.cpp")))
    texts = []
    for pat in ("common/*.cpp", "common/*.h", "tools/*/*.cpp", "tools/*/*.h", "examples/*/*.cpp", "src/*.cpp"):
        texts += [_read(p) for p in glob.glob(os.path.join(root, pat))]
    accepted = literal_flags(texts)
    mds = [os.path.join(root, "docs", "parameters.md")] + glob.glob(os.path.join(root, "tools", "*", "README.md")) \
        + glob.glob(os.path.join(root, "docs", "**", "*.md"), recursive=True)
    docs_all = set()
    for p in mds:
        docs_all |= doc_flags(_read(p))
    docs_params = doc_flags(_read(os.path.join(root, "docs", "parameters.md")))
    return declared, accepted, docs_all, docs_params


def read_list(p):
    if not os.path.exists(p):
        return set()
    return {l.split("#")[0].strip() for l in _read(p).splitlines() if l.split("#")[0].strip()}


def selftest():
    arg = 'add_opt(common_arg(\n {"--alpha", "-a"}, "N", "x"));\nadd_opt(common_arg({"--spec-draft-p-min", "--draft-p-min"}, "P", "y"));\n'
    ok = True

    def case(name, findings, want):
        nonlocal ok
        hit = any(want in x for x in findings) if want else not findings
        ok &= hit
        print("selftest %-34s findings=%d expect %-26s -> %s" % (name, len(findings), want or "none", "ok" if hit else "FAIL"))

    dec = declared_flags(arg)
    if dec != {"--alpha", "--spec-draft-p-min", "--draft-p-min"}:
        print("selftest declared_flags parse FAIL: %r" % sorted(dec))
        return False
    docs = doc_flags("use `--alpha N` and `--spec-draft-p-min, --draft-p-min P`")
    ev = lambda d, a, da, dp, ku=frozenset(), kg=frozenset(), mn=1: evaluate(d, a, da, dp, set(ku), set(kg), mn)
    case("clean", ev(dec, dec, docs, docs), "")
    case("flag dropped from docs", ev(dec, dec, doc_flags("use `--alpha N`"), set()), "UNDOCUMENTED: --draft-p-min")
    case("made-up flag in docs", ev(dec, dec, docs | {"--made-up"}, docs | {"--made-up"}), "GHOST: --made-up")
    sab = doc_flags("`--alpha` and `--draft-p-minSABOTAGE` and `--spec-draft-p-min`")
    case("boundary: --draft-p-minSABOTAGE", ev(dec, dec, sab, set()), "UNDOCUMENTED: --draft-p-min")
    case("boundary: sabotage is a ghost", ev(dec, dec, sab, sab), "GHOST: --draft-p-minSABOTAGE")
    case("baseline hides known gap", ev(dec, dec, doc_flags("`--alpha`"), set(), ku={"--spec-draft-p-min", "--draft-p-min"}), "")
    case("stale baseline entry", ev(dec, dec, docs, docs, ku={"--alpha"}), "STALE-BASELINE: --alpha")
    case("vacuous parse", ev(dec, dec, docs, docs, mn=50), "VACUOUS")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    a = ap.parse_args()
    if a.selftest and not selftest():
        print("SELFTEST FAILED: the gate cannot be trusted")
        return 2
    try:
        declared, accepted, docs_all, docs_params = load(a.root)
    except OSError as e:
        print("cannot read sources: %s" % e)
        return 2
    ku = os.path.join(a.root, "scripts", "docs_flags_known_undocumented.txt")
    kg = os.path.join(a.root, "scripts", "docs_flags_known_ghosts.txt")
    if a.write_baseline:
        with open(ku, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# declared in common/arg.cpp, named in no doc yet. Regenerate only after review; the list may only shrink.\n")
            fh.writelines(x + "\n" for x in sorted(declared - docs_all))
        with open(kg, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# in docs/parameters.md, accepted by no CLI source (absent-by-design mentions or doc defects to triage). May only shrink.\n")
            fh.writelines(x + "\n" for x in sorted(docs_params - accepted))
        print("baselines written")
        return 0
    f = evaluate(declared, accepted, docs_all, docs_params, read_list(ku), read_list(kg))
    print("SCOPE: %d declared flag(s), %d accepted literal(s), %d doc flag(s), %d in parameters.md" %
          (len(declared), len(accepted), len(docs_all), len(docs_params)))
    for x in f:
        print("FINDING", x)
    print("%d finding(s)" % len(f))
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
