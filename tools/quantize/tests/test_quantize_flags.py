#!/usr/bin/env python3
"""End-to-end check that llama-quantize's options actually reach the output file.

Every check is judged from the OUTPUT GGUF's own header (KV pairs and the tensor-info table), not from the tool's log:
a log-scraper reported four false failures while this was being written (records wrap across lines, and --repack
ends a record with ", type = q8_0_r8").

Why it exists: after the merge the built tool filled upstream's typed params (imatrix, kv_overrides, tt_overrides,
prune_layers) while src/llama-quantize.cpp read only the fork's void* forms, so --imatrix, --override-kv and
--tensor-type changed no byte of the output -- the tool even printed "loaded N importance matrix entries".

usage: test_quantize_flags.py --bin-dir <dir with llama-quantize[.exe] and llama-imatrix[.exe]> --model <q8_0 gguf>
       [--threads N] [--work-dir DIR] [--skip-slow] [--moe-model <gguf with ffn_gate_inp>] [--ple-model <gguf with per_layer_token_embd>]
The model must be a dense Q8_0 GGUF with blk.N.attn_q/attn_k/ffn_down/ffn_up tensors and a tied embedding
(no output.weight) -- e.g. a Qwen3-0.6B. Exit 0 = all pass, 1 = a check failed, 2 = setup error.
"""
# DELETE-VERIFIED: the only files removed are the ones this test writes into its own work dir moments earlier
import argparse
import hashlib
import os
import struct
import subprocess
import sys
import tempfile

SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
Q4_0, Q5_0, Q5_1, Q8_0 = 2, 6, 7, 8  # ggml/include/ggml.h


def read_gguf(path):
    """(kv, name -> ggml type id) from a GGUF v2/v3 header."""
    kv, tensors = {}, {}
    with open(path, 'rb') as f:
        def rd(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

        def rstr():
            return f.read(rd('Q')).decode('utf-8', 'replace')

        def val(t):
            if t in SCALAR:
                return rd(SCALAR[t])
            if t == 8:
                return rstr()
            if t == 9:
                it, n = rd('I'), rd('Q')
                return [val(it) for _ in range(n)]
            raise ValueError('unknown gguf value type %d' % t)
        if f.read(4) != b'GGUF':
            raise ValueError('%s is not a GGUF file' % path)
        rd('I')
        n_t, n_kv = rd('Q'), rd('Q')
        for _ in range(n_kv):
            k = rstr()
            kv[k] = val(rd('I'))
        for _ in range(n_t):
            name = rstr()
            for _ in range(rd('I')):
                rd('Q')
            tensors[name] = rd('I')
            rd('Q')
    return kv, tensors


class Runner:
    def __init__(self, args):
        ext = '.exe' if os.name == 'nt' else ''
        self.quantize = os.path.join(args.bin_dir, 'llama-quantize' + ext)
        self.imatrix = os.path.join(args.bin_dir, 'llama-imatrix' + ext)
        self.model, self.threads, self.work = args.model, str(args.threads), args.work_dir
        self.flags = 0x08000000 if os.name == 'nt' else 0  # CREATE_NO_WINDOW
        self.results = []

    def quant(self, tag, extra, ftype='Q4_0'):
        dst = os.path.join(self.work, 'tqf_%s.gguf' % tag)
        r = subprocess.run([self.quantize, '--allow-requantize'] + extra + [self.model, dst, ftype, self.threads],
                           capture_output=True, text=True, creationflags=self.flags)
        sha, kv, t = '-', {}, {}
        if r.returncode == 0 and os.path.exists(dst):
            with open(dst, 'rb') as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()[:16]
            kv, t = read_gguf(dst)
        if os.path.exists(dst):
            os.remove(dst)
        return r.returncode, sha, kv, t, r.stdout + r.stderr

    def judge(self, name, ok, detail):
        self.results.append((name, bool(ok)))
        print('%-26s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail), flush=True)


def ids(t, sub):
    return sorted({v for n, v in t.items() if sub in n})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin-dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--work-dir', default=None)
    ap.add_argument('--skip-slow', action='store_true', help='skip the --slow-iq2ks comparison')
    ap.add_argument('--moe-model', default=None, help='optional GGUF with ffn_gate_inp (e.g. stories15M_MOE-F16) for --ffn-gate-inp-type')
    ap.add_argument('--ple-model', default=None, help='optional GGUF with per_layer_token_embd (e.g. gemma-4-E2B) for --per-layer-token-embedding-type')
    args = ap.parse_args()
    args.work_dir = args.work_dir or tempfile.mkdtemp(prefix='tqf_')
    R = Runner(args)
    for p in (R.quantize, R.imatrix, R.model):
        if not os.path.exists(p):
            print('setup error: missing %s' % p)
            return 2

    src_kv, src = read_gguf(R.model)
    if src.get('blk.0.attn_q.weight') != Q8_0:
        print('setup error: the model must be Q8_0 (blk.0.attn_q.weight id %s)' % src.get('blk.0.attn_q.weight'))
        return 2

    rc0, h0, kv0, t0, _ = R.quant('none', [])
    R.judge('instrument control', rc0 == 0 and t0.get('blk.0.attn_q.weight') == Q4_0 and len(t0) == len(src),
            'Q4_0 run attn_q id=%s (q4_0=%d), tensors %d/%d' % (t0.get('blk.0.attn_q.weight'), Q4_0, len(t0), len(src)))

    # --- upstream-API options the library used to ignore ---
    txt, im = os.path.join(R.work, 'tqf_text.txt'), os.path.join(R.work, 'tqf_imatrix.gguf')
    with open(txt, 'w', encoding='utf-8') as fh:
        fh.write('The committee approved the revised budget. The treasurer filed a notice of change.' * 200)
    r = subprocess.run([R.imatrix, '-m', R.model, '-f', txt, '-o', im, '-c', '128', '--chunks', '2', '-t', R.threads],
                       capture_output=True, text=True, creationflags=R.flags)
    if r.returncode != 0 or not os.path.exists(im):
        print('setup error: llama-imatrix rc=%d\n%s' % (r.returncode, (r.stdout + r.stderr)[-1500:]))
        return 2
    rc, h, kv, t, _ = R.quant('imat', ['--imatrix', im])
    ik = {k: v for k, v in kv.items() if 'imatrix' in k}
    R.judge('--imatrix', rc == 0 and h != h0 and any(isinstance(v, int) and v > 0 for v in ik.values()),
            'changes output=%s kv=%s' % (h != h0, ik))
    rc, h, kv, t, _ = R.quant('hide', ['--imatrix', im, '--hide-imatrix'])
    hk = {k: v for k, v in kv.items() if 'imatrix' in k}
    fk = [k for k in hk if k.endswith('.file')]
    R.judge('--hide-imatrix', rc == 0 and fk and hk[fk[0]] == 'top_secret' and all(v == 0 for v in hk.values() if isinstance(v, int)),
            'kv=%s' % hk)
    rc, h, kv, t, _ = R.quant('okv', ['--override-kv', 'general.name=str:OVERRIDE_PROBE'])
    R.judge('--override-kv', rc == 0 and kv.get('general.name') == 'OVERRIDE_PROBE', 'general.name=%r' % kv.get('general.name'))
    rc, h, kv, t, _ = R.quant('tt', ['--tensor-type', 'attn_q=q8_0'])
    R.judge('--tensor-type', rc == 0 and ids(t, 'attn_q.weight') == [Q8_0] and ids(t, 'attn_k.weight') == ids(t0, 'attn_k.weight'),
            'attn_q=%s attn_k=%s' % (ids(t, 'attn_q.weight'), ids(t, 'attn_k.weight')))
    rc, h, kv, t, _ = R.quant('ttpure', ['--pure', '--tensor-type', 'attn_q=q8_0'])
    R.judge('--tensor-type under --pure', rc == 0 and ids(t, 'attn_q.weight') == [Q8_0] and ids(t, 'attn_k.weight') == [Q4_0],
            'attn_q=%s attn_k=%s' % (ids(t, 'attn_q.weight'), ids(t, 'attn_k.weight')))
    rc, h, kv, t, log = R.quant('prune', ['--prune-layers', '3'])
    R.judge('--prune-layers refuses', rc != 0 and 'refusing' in log, 'rc=%d' % rc)
    for p in (txt, im):
        os.remove(p)

    # --- fork flags restored into the built tool ---
    for flag, sub in [('--attn-q-type', 'attn_q.weight'), ('--attn-k-type', 'attn_k.weight'), ('--attn-v-type', 'attn_v.weight'),
                      ('--attn-output-type', 'attn_output.weight'), ('--ffn-gate-type', 'ffn_gate.weight'),
                      ('--ffn-down-type', 'ffn_down.weight'), ('--ffn-up-type', 'ffn_up.weight')]:
        rc, h, kv, t, _ = R.quant(flag.strip('-'), [flag, 'q5_0'])
        same = {n: v for n, v in t.items() if sub not in n} == {n: v for n, v in t0.items() if sub not in n}
        R.judge(flag, rc == 0 and ids(t, sub) == [Q5_0] and same, '%s=%s others-unchanged=%s' % (sub, ids(t, sub), same))
    rc, h, kv, t, _ = R.quant('customq', ['--custom-q', r'blk\.[0-9]\.attn_k=q5_1'])
    lo = sorted({v for n, v in t.items() if n.startswith('blk.') and len(n.split('.')[1]) == 1 and n.endswith('attn_k.weight')})
    hi = sorted({v for n, v in t.items() if n.startswith('blk.') and len(n.split('.')[1]) >= 2 and n.endswith('attn_k.weight')})
    R.judge('--custom-q', rc == 0 and lo == [Q5_1] and Q5_1 not in hi, 'blk.0-9 attn_k=%s blk.10+ attn_k=%s' % (lo, hi))
    rc, h, kv, t, _ = R.quant('customq_bad', ['--custom-q', 'attn_k=notatype'])
    R.judge('--custom-q bad type', rc != 0, 'rc=%d' % rc)
    rc, h, kv, t, _ = R.quant('sym', ['--symmetric-q40'])
    R.judge('--symmetric-q40', rc == 0 and h != h0 and t == t0, 'changes bytes=%s same types=%s' % (h != h0, t == t0))
    rc, h, kv, t, _ = R.quant('extra', ['--extra-output-tensor', 'q4_0'], ftype='Q8_0')
    added = sorted(set(t) - set(src))
    R.judge('--extra-output-tensor', rc == 0 and len(added) == 1, 'added=%s' % added)
    rc, h, kv, t, _ = R.quant('repack', ['--repack'], ftype='Q8_0')
    R.judge('--repack', rc == 0 and ids(t, 'ffn_down.weight') not in ([], [Q8_0]), 'ffn_down=%s' % ids(t, 'ffn_down.weight'))
    rc, h, kv, t, _ = R.quant('repackpat', ['--repack', '--repack-pattern', 'ffn_down'], ftype='Q8_0')
    fd, fu = ids(t, 'ffn_down.weight'), ids(t, 'ffn_up.weight')
    R.judge('--repack-pattern', rc == 0 and fd not in ([], [Q8_0]) and fu == [Q8_0], 'ffn_down=%s ffn_up=%s' % (fd, fu))
    # IQ2_KT is on the llama-level refusal list but NOT on ggml's hard-assert list (ggml_quantize_requires_imatrix);
    # IQ2_S cannot be used: --pure routes token_embd to iq2_xs, which ggml asserts on before the rule is reached
    rcn, _, _, _, logn = R.quant('noimat', ['--pure'], ftype='IQ2_KT')
    rcy, _, _, _, _ = R.quant('ignore', ['--pure', '--ignore-imatrix-rules'], ftype='IQ2_KT')
    R.judge('--ignore-imatrix-rules', rcn != 0 and 'Missing importance matrix' in logn and rcy == 0,
            'without rc=%d, with rc=%d' % (rcn, rcy))
    if not args.skip_slow:
        rcf, hf, _, _, _ = R.quant('iq2ks_fast', ['--pure'], ftype='IQ2_KS')
        rcs, hs, _, _, _ = R.quant('iq2ks_slow', ['--pure', '--slow-iq2ks'], ftype='IQ2_KS')
        R.judge('--slow-iq2ks', rcf == 0 and rcs == 0 and hf != hs, 'fast=%s slow=%s' % (hf, hs))

    for model, flag, sub, ftype, extra in ((args.moe_model, '--ffn-gate-inp-type', 'ffn_gate_inp', 'Q8_0', []),
                                           (args.ple_model, '--per-layer-token-embedding-type', 'per_layer_token_embd', 'Q4_0', ['--pure'])):
        if not model:
            continue
        R.model = model
        rcb, _, _, tb, _ = R.quant(sub + '_base', extra, ftype=ftype)
        rcf, _, _, tf, _ = R.quant(sub + '_flag', extra + [flag, 'q8_0'], ftype=ftype)
        same = {n: v for n, v in tf.items() if sub not in n} == {n: v for n, v in tb.items() if sub not in n}
        R.judge(flag, rcb == 0 and rcf == 0 and ids(tb, sub) not in ([], [Q8_0]) and ids(tf, sub) == [Q8_0] and same,
                'base %s=%s flag=%s others-unchanged=%s' % (sub, ids(tb, sub), ids(tf, sub), same))

    fails = [n for n, ok in R.results if not ok]
    print('\n%d checks, %d failed%s' % (len(R.results), len(fails), (': ' + ', '.join(fails)) if fails else ''))
    print('not exercised here: --partial-requant, --attn-qkv-type%s' % (
        '' if args.moe_model and args.ple_model else ', and without --moe-model/--ple-model: --ffn-gate-inp-type, --per-layer-token-embedding-type'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
