#!/usr/bin/env python3
"""Prevention scan: a LoRA fast-path guard must not test only the legacy adapter map.

Row S69-ENGREV-PEFT-LORA-FASTPATH-GUARD-LEGACY-MAP-ONLY-2026-10-27-01.

llama_context keeps adapters in THREE places: the legacy map `lora_adapters`, the context-wide `loras` (server / --lora) and the
per-sequence `seq_loras`. A fast path guarded by `lctx.lora_adapters.empty()` therefore stays ON while a server adapter is active, and the
adapter is silently skipped (output.weight LoRA while llama_set_output_vocab_subset is active; expert LoRA while the MoE expert cache is
active). llm_build_context::llm_lora_targets() checks all three; graph code must use it. `lora_adapters.empty()` may appear only in
src/llama.cpp (adapter teardown / the "no adapter at all" epoch shortcut).

Exit 0 = pass, 1 = fail. Usage: test-lora-fastpath-guard-scan.py [repo-root]
"""
import os
import re
import sys

ALLOWED = {os.path.join("src", "llama.cpp")}
# `X.lora_adapters.empty()`; group "obj" is X when the member is reached through a variable.
PAT = re.compile(r"(?:(?P<obj>\w+)\s*(?:\.|->)\s*)?(?<!\w)lora_adapters\s*\.\s*empty\s*\(\s*\)")
# `params.lora_adapters` / `params_base.lora_adapters` is the list of --lora ARGUMENTS (common_params), not the context's adapter map
ARG_NAMES = {"params", "params_base"}
FUNC_WINDOW = 400


def code_part(line):
    """The line without a trailing // comment (a comment may name the pattern it forbids)."""
    i = line.find("//")
    return line if i < 0 else line[:i]


def declared_common_params(lines, idx, obj):
    """True when `obj` is declared as a common_params (value, reference or pointer) in the same function.

    The window runs from the previous column-0 closing brace (the end of the preceding function) down to the hit line, so it
    holds the signature (parameter declarations) and the body above the hit.
    """
    start = idx
    while start > 0 and idx - start < FUNC_WINDOW and not lines[start - 1].startswith("}"):
        start -= 1
    decl = re.compile(r"\bcommon_params\b\s*(?:const\b\s*)?[&*]*\s*" + re.escape(obj) + r"\b")
    return any(decl.search(code_part(l)) for l in lines[start:idx + 1])


def scan_lines(lines):
    """Line numbers (1-based) and text of every legacy-map-only guard in `lines`."""
    bad = []
    for i, line in enumerate(lines):
        code = code_part(line)
        for m in PAT.finditer(code):
            obj = m.group("obj")
            if obj in ARG_NAMES:
                continue
            if obj and declared_common_params(lines, i, obj):
                continue
            bad.append((i + 1, line.strip()))
            break
    return bad


def selftest():
    """Synthetic cases: the --lora argument list reached through ANY common_params variable passes, the context map never does."""
    cases = [
        ("common_params copy named p", "static void f(const common_params & p) {\n    if (!p.lora_adapters.empty()) {}\n}\n", 0),
        ("common_params value named opts", "void g() {\n    common_params opts;\n    if (opts.lora_adapters.empty()) {}\n}\n", 0),
        ("common_params pointer", "void g(common_params * cp) {\n    if (cp->lora_adapters.empty()) {}\n}\n", 0),
        ("params. prefix", "void g() {\n    if (params.lora_adapters.empty()) {}\n}\n", 0),
        ("context map lctx", "void g(llama_context & lctx) {\n    if (lctx.lora_adapters.empty()) {}\n}\n", 1),
        ("bare member", "void g() {\n    if (lora_adapters.empty()) {}\n}\n", 1),
        ("same name declared in the PREVIOUS function only",
         "void a(const common_params & p) {\n    (void) p;\n}\nvoid b(llama_context & p) {\n    if (p.lora_adapters.empty()) {}\n}\n", 1),
        ("comment naming the pattern", "void g(llama_context & lctx) {\n    int x = 0; // lctx.lora_adapters.empty()\n}\n", 0),
    ]
    fails = 0
    for name, text, want in cases:
        got = len(scan_lines(text.splitlines()))
        ok = got == want
        fails += 0 if ok else 1
        print("selftest %-48s want %d got %d %s" % (name, want, got, "ok" if ok else "FAIL"))
    return fails


def scan(root):
    bad = []
    n_files = 0
    for base in ("src", "common", "tools", "examples"):
        top = os.path.join(root, base)
        for dirpath, _dirs, files in os.walk(top):
            for name in files:
                if not name.endswith((".cpp", ".h", ".cu", ".cuh", ".hpp")):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, root)
                n_files += 1
                if rel in ALLOWED:
                    continue
                with open(path, encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().splitlines()
                for n, text in scan_lines(lines):
                    bad.append("%s:%d: %s" % (rel, n, text))
    return bad, n_files


def main():
    if selftest():
        print("FAIL: scan selftest")
        return 1
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    root = os.path.normpath(root)
    # positive control: the allowed file really holds the pattern, so a zero elsewhere measured the tree, not the regex
    with open(os.path.join(root, "src", "llama.cpp"), encoding="utf-8", errors="replace") as fh:
        control = sum(1 for line in fh if PAT.search(code_part(line)))
    bad, n_files = scan(root)
    print("scanned %d source files; positive control (src/llama.cpp) = %d hit(s)" % (n_files, control))
    if control == 0:
        print("FAIL: positive control found no lora_adapters.empty() in src/llama.cpp; the scan cannot be trusted")
        return 1
    for b in bad:
        print("FAIL (legacy-map-only LoRA guard, use llm_lora_targets): " + b)
    if bad:
        return 1
    print("PASS: no graph-side lora_adapters.empty() guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
