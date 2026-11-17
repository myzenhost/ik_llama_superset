#!/usr/bin/env python3
"""llama-quantize --slab-size N is accepted, reaches the slabbed path, and changes memory but not the output bytes.

usage: test-quantize-slab-size.py <llama-quantize> <F16 or BF16 GGUF> <work dir> [type=Q8_0] [threads=4]

ik_llama's --slab-size (MiB of f32 per slab, 0 = never) was merged into the unbuilt examples/quantize copy only (fork merge
a9fc5f64b). src/llama-quantize.cpp honours llama_model_quantize_params::slab_size (default 1 GiB) whatever the CLI says,
so before the port no user could set it, and with the 1 GiB default a small model never takes the slabbed path at all.

Checks:
  1. --slab-size 1 quantizes (exit 0, output written);
  2. --slab-size 0 quantizes (exit 0, output written);
  3. the two outputs are byte-identical (slabbing must not change the quantized data);
  4. --slab-size abc exits non-zero (the value parse is strict).
Peak working set of runs 1 and 2 is printed (informational, not a pass condition).
Exit code = number of failed checks. Each run writes a fresh, timestamped output name, so nothing is deleted.
"""
import hashlib
import os
import subprocess
import sys
import time

import psutil

qbin, model, work = sys.argv[1:4]
qtype = sys.argv[4] if len(sys.argv) > 4 else 'Q8_0'
threads = sys.argv[5] if len(sys.argv) > 5 else '4'
os.makedirs(work, exist_ok=True)
stamp = time.strftime('%Y%m%d_%H%M%S')
failed = 0
N_CHECKS = 4


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


def run(slab):
    out = os.path.join(work, f'slab{slab}_{stamp}.gguf')
    logp = os.path.join(work, f'quantize_slab{slab}_{stamp}.log')
    t0 = time.time()
    with open(logp, 'wb') as log:
        p = subprocess.Popen([qbin, '--slab-size', str(slab), model, out, qtype, threads], stdout=log,
                             stderr=subprocess.STDOUT, creationflags=0x08000000 if os.name == 'nt' else 0)
        peak = 0
        try:
            ps = psutil.Process(p.pid)
            while p.poll() is None:
                try:
                    mi = ps.memory_info()
                    peak = max(peak, getattr(mi, 'peak_wset', 0) or mi.rss)
                except psutil.Error:
                    break
                time.sleep(0.01)
        except psutil.Error:
            pass
        try:
            rc = p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -9
    print(f'slab {slab}: rc {rc}, {time.time() - t0:.1f} s, peak working set {peak / 2**20:.1f} MiB, log {logp}', flush=True)
    return rc, out, peak


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


rc1, out1, peak1 = run(1)
check('--slab-size 1 quantizes', rc1 == 0 and os.path.exists(out1), f'rc {rc1}')
rc0, out0, peak0 = run(0)
check('--slab-size 0 quantizes', rc0 == 0 and os.path.exists(out0), f'rc {rc0}')
both = os.path.exists(out1) and os.path.exists(out0)
h1, h0 = (sha(out1), sha(out0)) if both else ('', '')
check('outputs byte-identical', rc1 == 0 and rc0 == 0 and both and h1 == h0,
      f'{h1[:16]} vs {h0[:16]}' if both else 'missing output')
rcx, _, _ = run('abc')
check('--slab-size abc exits non-zero', rcx != 0, f'rc {rcx}')
print(f'peak working set: slab1 {peak1 / 2**20:.1f} MiB, slab0 {peak0 / 2**20:.1f} MiB', flush=True)
print(f'{N_CHECKS - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
