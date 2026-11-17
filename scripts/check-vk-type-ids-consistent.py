#!/usr/bin/env python3
"""Check that ggml/src/ggml-vulkan/vulkan-shaders/ggml_type_ids.glsl carries ggml.h's enum ggml_type values.

The Vulkan shaders switch on the ggml_type the host passes as a specialization constant (MmTypeA, FaTypeK, FaTypeV), so every
`#define GGML_TYPE_<NAME> <N>u` in ggml_type_ids.glsl must equal `GGML_TYPE_<NAME> = <N>` in ggml/include/ggml.h. A wrong id compiles
and runs: the tensor just never reaches its decode branch. (This fork renumbered upstream's Q1_0 / Q2_0 to 400 / 401; upstream's table
says 41 / 42, which are this fork's Q1_0_G128 / Q2_0_G128.)

Exit 0 when every shader id matches; 1 on any mismatch or on a shader name that ggml.h does not define.
--selftest: a correct table must pass and a planted wrong id must fail, before the real check runs.
"""
import argparse
import os
import re
import sys

ENUM_RX = re.compile(r'^\s*(GGML_TYPE_[A-Z0-9_]+)\s*=\s*(\d+)\s*,', re.M)
GLSL_RX = re.compile(r'^\s*#define\s+(GGML_TYPE_[A-Z0-9_]+)\s+(\d+)u?\b', re.M)


def parse_enum(text):
    i = text.index('enum ggml_type {')
    j = text.index('};', i)
    return {k: int(v) for k, v in ENUM_RX.findall(text[i:j])}


def check(enum, glsl_text, where):
    errs = []
    ids = GLSL_RX.findall(glsl_text)
    if not ids:
        errs.append(f'{where}: no #define GGML_TYPE_* found (the parser did not reach the table)')
    for name, val in ids:
        if name not in enum:
            errs.append(f'{where}: {name} = {val} is not in ggml.h enum ggml_type')
        elif enum[name] != int(val):
            errs.append(f'{where}: {name} = {val}, ggml.h has {enum[name]}')
    return errs, len(ids)


def selftest():
    enum = {'GGML_TYPE_F32': 0, 'GGML_TYPE_Q1_0': 400}
    good = '#define GGML_TYPE_F32 0u\n#define GGML_TYPE_Q1_0 400u\n'
    bad = '#define GGML_TYPE_F32 0u\n#define GGML_TYPE_Q1_0 41u\n'
    e_good, n_good = check(enum, good, 'selftest-good')
    e_bad, _ = check(enum, bad, 'selftest-bad')
    e_unknown, _ = check(enum, '#define GGML_TYPE_NOPE 3u\n', 'selftest-unknown')
    ok = not e_good and n_good == 2 and len(e_bad) == 1 and len(e_unknown) == 1
    print('selftest:', 'PASS' if ok else 'FAIL', '(correct table passes, a wrong id fails, an unknown name fails)')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest and not selftest():
        return 1
    with open(os.path.join(a.root, 'ggml', 'include', 'ggml.h'), encoding='utf-8') as f:
        enum = parse_enum(f.read())
    p = os.path.join(a.root, 'ggml', 'src', 'ggml-vulkan', 'vulkan-shaders', 'ggml_type_ids.glsl')
    with open(p, encoding='utf-8') as f:
        errs, n = check(enum, f.read(), 'ggml_type_ids.glsl')
    for e in errs:
        print('MISMATCH', e)
    print(f'{n} shader type ids checked against {len(enum)} ggml.h enum values: {"OK" if not errs else "FAIL"}')
    return 0 if not errs else 1


if __name__ == '__main__':
    sys.exit(main())
