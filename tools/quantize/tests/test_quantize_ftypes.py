#!/usr/bin/env python3
"""End-to-end check that llama-quantize can WRITE the adopted ftypes, and survives a tensor its type cannot hold.

Two defects this guards, both measured 2026-09-27 on the wave3 CPU build:
  1. Q1_0 / Q2_0 / NVFP4 were declared in llama.h and quantized + validated by ggml, but src/llama-quantize.cpp had no
     ftype -> type case ("invalid output file type 40/44") and the CLI table had no NVFP4 row -- nothing could write them.
  2. A tensor whose row is not a whole number of blocks of a type OUTSIDE the K/IQ/BN lists (e.g. stories260K's
     172-wide ffn_down -> Q8_0) reached gguf.cpp's GGML_ASSERT and aborted the whole run instead of falling back.

Judged from the OUTPUT GGUF's own tensor-info table (read_gguf from test_quantize_flags.py), never from the log.

usage: test_quantize_ftypes.py --bin-dir <dir with llama-quantize[.exe]> --model <f16/f32 gguf with some tensors whose
       row length is a multiple of 128> --odd-model <f32 gguf with a 172-wide blk.0.ffn_down, e.g. stories260K-f32>
       [--work-dir DIR]
Exit 0 = all pass, 1 = a check failed, 2 = setup error.
"""
# DELETE-VERIFIED: the only files removed are the ones this test writes into its own work dir moments earlier
import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_quantize_flags import read_gguf  # noqa: E402

F16, Q8_0, NVFP4, Q1_0, Q2_0 = 1, 8, 40, 400, 401  # ggml/include/ggml.h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin-dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--odd-model', required=True)
    ap.add_argument('--work-dir')
    a = ap.parse_args()
    exe = os.path.join(a.bin_dir, 'llama-quantize' + ('.exe' if os.name == 'nt' else ''))
    for p in (exe, a.model, a.odd_model):
        if not os.path.isfile(p):
            print(f'setup: missing {p}')
            return 2
    work = a.work_dir or tempfile.mkdtemp(prefix='qftypes-')
    os.makedirs(work, exist_ok=True)
    failed = 0

    def judge(name, ok, detail):
        nonlocal failed
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: {detail}")
        failed += 0 if ok else 1

    def quant(src, ftype, tag):
        out = os.path.join(work, f'{tag}-{ftype}.gguf')
        r = subprocess.run([exe, src, out, ftype], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           errors='replace')
        return r.returncode, out

    # 1. each adopted ftype is writable, and at least one tensor really lands in that type
    for ftype, tid in (('NVFP4', NVFP4), ('Q1_0', Q1_0), ('Q2_0', Q2_0)):
        rc, out = quant(a.model, ftype, 'model')
        if rc != 0 or not os.path.isfile(out):
            judge(f'{ftype} writable', False, f'llama-quantize rc={rc}')
            continue
        _, types = read_gguf(out)
        n = sum(1 for t in types.values() if t == tid)
        judge(f'{ftype} writable', n > 0, f'{n} of {len(types)} tensors are {ftype}')
        os.remove(out)

    # 2. negative control: an ftype name that does not exist must still be REFUSED
    rc, out = quant(a.model, 'NOT_A_REAL_FTYPE', 'model')
    judge('invented ftype refused', rc != 0 and not os.path.isfile(out), f'rc={rc}')

    # 3. an odd-width tensor falls back instead of aborting; its even-width neighbour keeps the requested type
    rc, out = quant(a.odd_model, 'Q8_0', 'odd')
    if rc != 0 or not os.path.isfile(out):
        judge('Q8_0 odd-width fallback', False, f'llama-quantize rc={rc} (was rc 3: gguf.cpp GGML_ASSERT)')
    else:
        _, types = read_gguf(out)
        down, gate = types.get('blk.0.ffn_down.weight'), types.get('blk.0.ffn_gate.weight')
        judge('Q8_0 odd-width fallback', down == F16, f'blk.0.ffn_down (172 wide) type={down}, want F16={F16}')
        judge('Q8_0 control tensor kept', gate == Q8_0, f'blk.0.ffn_gate (64 wide) type={gate}, want Q8_0={Q8_0}')
        os.remove(out)

    print(f'{failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
