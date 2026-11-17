#!/usr/bin/env python3
"""Embedding-only contexts skip the construction-time compute-buffer reserve.

usage: test-embd-reserve-skip.py <llama-embedding> <bert-family embedding GGUF> [n_ctx]

A BERT-family encoder has no memory module, and examples/embedding sets n_ubatch = n_batch = n_ctx, so the
construction-time reserve sizes an n_ctx-token graph that a short input never uses (bge-small, -c 8192:
3504 MiB of compute buffer for a 10-token prompt). With the skip, the buffer is allocated at the first
decode for the real batch. LLAMA_EMBD_SKIP_RESERVE=0 restores the reserve; both runs must give the same
embedding.

Checks, on one binary:
  1. the default run logs that it skipped the reserve; the =0 run does not;
  2. the default run's peak commit is under half of the =0 run's (Windows: PeakPagefileUsage read from the
     exited process; elsewhere ru_maxrss of the child);
  3. the two embeddings are identical.
Exit code = number of failed checks.
"""
import json
import os
import subprocess
import sys

exe, model = sys.argv[1], sys.argv[2]
n_ctx = sys.argv[3] if len(sys.argv) > 3 else '8192'
PROMPT = 'The committee approved the revised budget.'


def peak_bytes(p):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                                       'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                                       'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        m = PMC()
        m.cb = ctypes.sizeof(PMC)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(p._handle)), ctypes.byref(m), m.cb)
        return m.PeakPagefileUsage if ok else None
    import resource
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024


def run(skip_env):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    env.pop('LLAMA_EMBD_SKIP_RESERVE', None)
    if skip_env is not None:
        env['LLAMA_EMBD_SKIP_RESERVE'] = skip_env
    args = [exe, '-m', model, '-c', n_ctx, '-ngl', '0', '-t', '2', '--embd-output-format', 'json', '-p', PROMPT]
    flags = 0x08000000 if os.name == 'nt' else 0
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, creationflags=flags)
    out, err = p.communicate(timeout=300)
    peak = peak_bytes(p)
    err = err.decode('utf-8', 'replace')
    try:
        emb = json.loads(out.decode('utf-8', 'replace'))['data'][0]['embedding']
    except (ValueError, KeyError, IndexError):
        emb = None
    return p.returncode, peak, 'skipping the construction-time reserve' in err, emb


failed = 0


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


rc_d, peak_d, skip_d, emb_d = run(None)
rc_r, peak_r, skip_r, emb_r = run('0')
check('both runs exit 0', rc_d == 0 and rc_r == 0, f'default rc {rc_d}, reserve rc {rc_r}')
check('default run skips the reserve', skip_d)
check('LLAMA_EMBD_SKIP_RESERVE=0 run reserves', not skip_r)
mib = lambda b: f'{b / 2**20:.1f} MiB' if b else 'n/a'  # noqa: E731
check('default peak commit < half of the reserving run', bool(peak_d and peak_r and peak_d * 2 < peak_r),
      f'default {mib(peak_d)}, reserve {mib(peak_r)}')
check('embeddings identical', emb_d is not None and emb_d == emb_r, f'dim {len(emb_d) if emb_d else 0}')
print(f'{5 - failed} passed, {failed} failed', flush=True)
sys.exit(failed)
