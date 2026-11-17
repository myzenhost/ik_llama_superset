#!/usr/bin/env python3
# qwen4exp load-time metadata validation (src/llama-hparams.cpp, row S69-QWEN4EXP-VALIDATION-PARITY-2026-10-28-01).
# Each malformed file must be refused IN hparams with its own message; the GOOD file must trip none of them (it
# fails later, on its missing tensors). With a second binary (the build before the change) the same messages must
# be ABSENT there -- the fail-before half.
#
#   python tests/test_qwen4exp_meta_validation.py <tree-root> <work-dir> <llama-cli> [<llama-cli before the change>]
import pathlib
import subprocess
import sys

root, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
after = sys.argv[3]
before = sys.argv[4] if len(sys.argv) > 4 else None
work.mkdir(parents=True, exist_ok=True)
CASES = {
    "good": None,
    "ssm_conv0": "ssm.conv_kernel must be greater than zero",
    "idx_top_k0": "indexer.top_k must be greater than zero",
    "hc1": "hyper_connection.count must be greater than one",
    "ple2": "only one PLE layer is supported",
    "ple_ngram1": "PLE n-gram size 1 is below 2",
}
NOWIN = 0x08000000 if sys.platform == "win32" else 0


def load(exe, f):
    # PROGRESS-OK: a metadata-only file fails at load in well under a second
    r = subprocess.run([exe, "-m", str(f), "-st", "-p", "hi", "-n", "1", "--no-warmup"], capture_output=True,
                       text=True, errors="replace", creationflags=NOWIN, timeout=120)
    return r.stdout + r.stderr


fails = 0
for case, want in CASES.items():
    f = work / ("q4meta_%s.gguf" % case)
    # PROGRESS-OK: writes one small metadata-only file
    subprocess.run([sys.executable, "-I", str(root / "tests/make_qwen4exp_meta_gguf.py"), str(root), str(f), case],
                   check=True, capture_output=True, creationflags=NOWIN, timeout=60)
    out_after = load(after, f)
    out_before = load(before, f) if before else ""
    if want is None:
        ok = all(m not in out_after for m in CASES.values() if m)
    else:
        ok = want in out_after and (before is None or want not in out_before)
    fails += not ok
    print("%-11s %s" % (case, "PASS" if ok else "FAIL"))
    if not ok:
        print("   tail:", out_after.strip().splitlines()[-3:])
print("RESULT:", "PASS" if fails == 0 else "FAIL (%d)" % fails)
sys.exit(1 if fails else 0)
