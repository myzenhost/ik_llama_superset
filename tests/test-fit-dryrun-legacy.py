#!/usr/bin/env python3
"""The --fit dry run sizes a legacy-graph model's compute buffer without allocating it.

usage: test-fit-dryrun-legacy.py <bin dir> <legacy-graph generative GGUF> [n_ctx] [n_ubatch]

common/fit.cpp builds the model and context with no_alloc = true and reads the per-device model / context /
compute sizes back. For an arch built by the legacy llm_build_context (uses_legacy_graph_builder() == true,
the default), llama_context::sched_reserve called ggml_backend_sched_reserve(), which ALLOCATES the compute
buffer, and then skipped the size read-back because no_alloc was set. So the dry run reported compute = 0 MiB
for every legacy model (the fit then under-counts the device by the whole compute buffer) while committing
that buffer in host memory anyway (row S69-ENGINE-REVIEW-1016-FIT-DRYRUN-LEGACY-RESERVE-ALLOCATES-01).

Checks:
  1. llama-fit-params --fit-print on and the real llama-completion run both exit 0;
  2. the dry run's Host compute column is nonzero;
  3. it equals the compute buffer the real context allocates (same -c / -b / -ub / -ngl 0), to the MiB;
  4. the dry run's peak commit is under half of the real run's (Windows: PeakPagefileUsage of the exited
     process; elsewhere ru_maxrss of the child), i.e. the dry run did not allocate the buffers it sized;
  5. the dry run's peak does not grow with the context it sizes: both runs repeated at -c 512, the Host
     context + compute columns grow by D between -c 512 and -c n_ctx, the real run's peak grows by at least
     D/2 (positive control: the instrument sees allocations of that size) and the dry run's by under D/4.
Check 4 alone measured the process floor, not the dry run: a 135M model's buffers are smaller than the
startup commit of a CUDA-linked exe, so `llama-fit-params --version` and the dry run peaked at the same
481.8 MiB (row S69-B1019F-STARTUP-COMMIT-TRANSIENT-AND-FIT-CHECK4-REACH-2026-10-26-01). Check 4 stays so
a startup-commit regression still shows; check 5 is the one that does not depend on the floor.
Exit code = number of failed checks.
"""
import os
import re
import subprocess
import sys

bindir, model = sys.argv[1], sys.argv[2]
n_ctx = sys.argv[3] if len(sys.argv) > 3 else '8192'
n_ub = sys.argv[4] if len(sys.argv) > 4 else '2048'
ext = '.exe' if os.name == 'nt' else ''
N_CTX_SMALL = '512'
common = lambda c: ['-m', model, '-c', c, '-b', n_ub, '-ub', n_ub, '-ngl', '0', '-t', '2']  # noqa: E731
COMMON = common(n_ctx)


def peak_bytes(p):
    if os.name != 'nt':
        return p.maxrss  # this child's own ru_maxrss from os.wait4 (RUSAGE_CHILDREN is the max over every child)
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


def run(args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    flags = 0x08000000 if os.name == 'nt' else 0
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, env=env,
                         creationflags=flags)
    if os.name == 'nt':
        out, err = p.communicate(timeout=600)
    else:
        import threading
        bufs = {}
        readers = [threading.Thread(target=lambda k, f: bufs.__setitem__(k, f.read()), args=(k, f))
                   for k, f in (('out', p.stdout), ('err', p.stderr))]
        for t in readers:
            t.start()
        _, status, ru = os.wait4(p.pid, 0)
        for t in readers:
            t.join()
        p.returncode = os.waitstatus_to_exitcode(status)
        p.maxrss = ru.ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        out, err = bufs['out'], bufs['err']
    return p.returncode, peak_bytes(p), out.decode('utf-8', 'replace'), err.decode('utf-8', 'replace')


failed = 0


def check(name, ok, detail=''):
    global failed
    print(('PASS ' if ok else 'FAIL ') + name + (f' -- {detail}' if detail else ''), flush=True)
    failed += 0 if ok else 1


rc_f, peak_f, out_f, err_f = run([os.path.join(bindir, 'llama-fit-params' + ext)] + COMMON + ['--fit-print', 'on'])
rc_r, peak_r, out_r, err_r = run([os.path.join(bindir, 'llama-completion' + ext)] + COMMON +
                                 ['-n', '1', '-p', 'The court granted the motion.', '-no-cnv', '--temp', '0'])
host = re.search(r'^Host (\d+) (\d+) (\d+)\s*$', out_f, re.M)
real = re.findall(r'CPU compute buffer size =\s*([0-9.]+) MiB', err_r)
fit_mib = int(host.group(3)) if host else None
real_mib = float(real[-1]) if real else None
mib = lambda b: f'{b / 2**20:.1f} MiB' if b else 'n/a'  # noqa: E731

check('both runs exit 0', rc_f == 0 and rc_r == 0 and host is not None,
      f'fit rc {rc_f}, real rc {rc_r}, Host line {"found" if host else "MISSING"}')
check('dry-run Host compute column is nonzero', bool(fit_mib), f'Host line {host.group(0).strip() if host else None!r}')
check('dry-run compute == real compute buffer (MiB)', fit_mib is not None and real_mib is not None and
      fit_mib == int(real_mib), f'dry run {fit_mib} MiB, real {real_mib} MiB')
check('dry-run peak commit < half of the real run', bool(peak_f and peak_r and peak_f * 2 < peak_r),
      f'dry run {mib(peak_f)}, real {mib(peak_r)}')

rc_fs, peak_fs, out_fs, _ = run([os.path.join(bindir, 'llama-fit-params' + ext)] + common(N_CTX_SMALL) + ['--fit-print', 'on'])
rc_rs, peak_rs, _, _ = run([os.path.join(bindir, 'llama-completion' + ext)] + common(N_CTX_SMALL) +
                           ['-n', '1', '-p', 'The court granted the motion.', '-no-cnv', '--temp', '0'])
host_s = re.search(r'^Host (\d+) (\d+) (\d+)\s*$', out_fs, re.M)
d_mib = (int(host.group(2)) + int(host.group(3)) - int(host_s.group(2)) - int(host_s.group(3))) if host and host_s else None
ok5 = (rc_fs == 0 and rc_rs == 0 and d_mib is not None and d_mib >= 32 and None not in (peak_f, peak_r, peak_fs, peak_rs))
grow_r = (peak_r - peak_rs) / 2**20 if ok5 else None
grow_f = (peak_f - peak_fs) / 2**20 if ok5 else None
check(f'real-run peak grows >= D/2 from -c {N_CTX_SMALL} to -c {n_ctx} (control)', ok5 and grow_r >= d_mib / 2,
      f'D {d_mib} MiB (Host ctx+compute), real grew {grow_r if grow_r is None else round(grow_r, 1)} MiB'
      f' (rc {rc_fs}/{rc_rs}; D must be >= 32 MiB for the check to mean anything)')
check(f'dry-run peak grows < D/4 from -c {N_CTX_SMALL} to -c {n_ctx}', ok5 and grow_f < d_mib / 4,
      f'D {d_mib} MiB, dry run grew {grow_f if grow_f is None else round(grow_f, 1)} MiB')
print(f'{6 - failed} passed, {failed} failed', flush=True)
if failed:
    sys.stderr.write('--- fit stdout ---\n' + out_f[-2000:] + '\n--- fit stderr tail ---\n' + err_f[-3000:] + '\n')
sys.exit(failed)
