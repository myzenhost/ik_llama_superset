#!/usr/bin/env python3
"""ik 77eff8798 (#2596) port: quantize fallback cases + ftype names.

usage: test-quant-fallback-ftype-names.py <bin dir with llama-quantize(.exe), llama-perplexity(.exe)> <stories260K-f32.gguf> <work dir>

Checks (each FAILS on the parent commit and PASSES with the port):
  1. ftype names: a model quantised to PQ2_0_R8 / PTQ1_0_R8 / IQ4_KS_R16 reports its name at load. Before, the
     declared general.file_type had no name, the loader ignored it (ik #2550 rule) and printed a "(guessed)" type.
  2. interleave fallback: when a repacked type is refused (token_embd, or a row count that is not a multiple of the
     interleave), its base type is re-checked against the row length: IQ1_S/IQ1_M -> IQ1_KT (ik 77eff8798, #2596; was IQ3_KT), IQ4_KS -> IQ4_NL.
  3. --token-embedding-type iq4_ks_r16 on a 64-wide token_embd: IQ4_KS (block 256) does not fit, so -> IQ4_NL.
  4. per_layer_token_embd is never repacked: --per-layer-token-embedding-type q8_0_r8 is changed to q8_0.
Control: --pure Q8_0 of the same file: 64-wide tensors Q8_0, the 172-wide ffn_down F16, before and after.
  5. (fork fix found by check 1) --pure Q8_0 / Q4_0 / IQ4_NL / IQ4_KS_R16 models run llama-perplexity: the fused ffn
     up/gate dequant path asserted Nx % 8 on the 172-row ffn tensors before its row-count guard.
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_raw_util as gr  # noqa: E402

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
EXE = '.exe' if os.name == 'nt' else ''

bindir, model, work = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(work, exist_ok=True)
fails = []
passes = 0

# ggml type ids (ggml/include/ggml.h)
GGML = {'F16': 1, 'Q8_0': 8, 'IQ4_NL': 20, 'IQ4_KS': 144, 'IQ3_KT': 154, 'IQ1_KT': 158, 'IQ1_S_R4': 219, 'IQ1_M_R4': 229,
        'IQ4_KS_R16': 345}


def check(cond, msg):
    global passes
    if cond:
        passes += 1
        print('PASS', msg)
    else:
        fails.append(msg)
        print('FAIL', msg)


def run(argv):
    p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
    return p.returncode, p.stdout.decode('utf-8', 'replace')


def quantize(src, dst, qtype, extra=()):
    return run([os.path.join(bindir, 'llama-quantize' + EXE), '--pure', *extra, src, dst, qtype, '1'])


def tensor_types(path):
    g = gr.load(path)
    return {t['name']: t['type'] for t in g.tensors}


def ftype_line(path):
    # load through llama-perplexity: print_meta logs "model ftype = <name>"
    txt = os.path.join(work, 'tiny.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        f.write('Once upon a time there was a little girl named Lily. ' * 40)
    rc, out = run([os.path.join(bindir, 'llama-perplexity' + EXE), '-m', path, '-f', txt, '-c', '64', '--chunks', '1', '-t', '2'])
    m = re.search(r'model ftype\s*=\s*(.+)', out)
    return rc, (m.group(1).strip() if m else None), out


# 1. ftype names
for qt, want in (('PQ2_0_R8', 'PQ2_0_R8 - 2.13 bpw ternary (Prism)'),
                 ('PTQ1_0_R8', 'PTQ1_0_R8 - 1.75 bpw ternary (Prism)'),
                 ('IQ4_KS_R16', 'IQ4_KS_R16 - 4.25 bpw')):
    out_path = os.path.join(work, f'm-{qt}.gguf')
    rc, log = quantize(model, out_path, qt)
    check(rc == 0, f'quantize {qt} rc={rc}')
    if rc == 0:
        lrc, name, _ = ftype_line(out_path)
        check(name == want, f'{qt}: load reports model ftype {name!r}, want {want!r} (perplexity rc={lrc})')

# 2. interleave fallback re-runs change_type_if_necessary. IQ1_S_R4 / IQ1_M_R4 / IQ4_KS_R16 have 32-element blocks, so a
#    64-wide row passes the row-length check; token_embd (never repacked) and any tensor whose row COUNT is not a multiple
#    of the interleave then drop to the base type IQ1_S / IQ1_M / IQ4_KS, whose 256-element blocks do NOT fit 64 columns.
#    Before the port that base type was used as-is (IQ1_S: GGML_ASSERT(imatrix), IQ4_KS: row-size abort; rc 3).
#    stories260K: token_embd 64 x 512, blk.N.ffn_gate 64 x 172 (172 % 16 != 0, 172 % 4 == 0), attn_q 64 x 64.
for qt, extra, want_te, ffn_gate_want in (
        ('IQ1_S_R4',   ('--ignore-imatrix-rules',), 'IQ1_KT', 'IQ1_S_R4'),
        ('IQ1_M_R4',   ('--ignore-imatrix-rules',), 'IQ1_KT', 'IQ1_M_R4'),
        ('IQ4_KS_R16', (),                          'IQ4_NL', 'IQ4_NL')):
    out_path = os.path.join(work, f'fb-{qt}.gguf')
    rc, log = quantize(model, out_path, qt, extra)
    check(rc == 0, f'quantize --pure {qt} rc={rc}')
    if rc == 0:
        ty = tensor_types(out_path)
        got = ty.get('token_embd.weight')
        check(got == GGML[want_te], f'{qt}: token_embd (64 wide, interleave refused) -> {want_te} ({GGML[want_te]}), got {got}')
        got = ty.get('blk.0.ffn_gate.weight')
        check(got == GGML[ffn_gate_want], f'{qt}: blk.0.ffn_gate.weight (172 rows) -> {ffn_gate_want} ({GGML[ffn_gate_want]}), got {got}')
        got = ty.get('blk.0.attn_q.weight')
        check(got == GGML[qt], f'{qt}: blk.0.attn_q.weight (64 x 64) keeps {qt} ({GGML[qt]}), got {got}')

# 3. token-embd override to a repacked type whose base does not fit the row
out_path = os.path.join(work, 'te-iq4ksr16.gguf')
rc, log = quantize(model, out_path, 'Q8_0', ('--token-embedding-type', 'iq4_ks_r16'))
check(rc == 0, f'quantize Q8_0 --token-embedding-type iq4_ks_r16 rc={rc}')
if rc == 0:
    got = tensor_types(out_path).get('token_embd.weight')
    check(got == GGML['IQ4_NL'], f'token_embd -> IQ4_NL ({GGML["IQ4_NL"]}), got {got}')

# 4. per_layer_token_embd is never repacked (synthetic: stories260K + a per_layer_token_embd.weight copy of token_embd)
g = gr.load(model)
te = next(t for t in g.tensors if t['name'] == 'token_embd.weight')
g.tensors.append({'name': 'per_layer_token_embd.weight', 'dims': list(te['dims']), 'type': te['type'], 'data': te['data']})
syn = os.path.join(work, 'stories260K-ple.gguf')
gr.dump(g, syn)
out_path = os.path.join(work, 'ple-q8r8.gguf')
rc, log = quantize(syn, out_path, 'Q8_0', ('--per-layer-token-embedding-type', 'q8_0_r8'))
check(rc == 0, f'quantize synthetic per-layer model rc={rc}')
if rc == 0:
    got = tensor_types(out_path).get('per_layer_token_embd.weight')
    check(got == GGML['Q8_0'], f'per_layer_token_embd -> Q8_0 (8), not repacked Q8_0_R8 (208), got {got}')

# control: plain --pure Q8_0
out_path = os.path.join(work, 'ctl-q8_0.gguf')
rc, log = quantize(model, out_path, 'Q8_0')
check(rc == 0, f'control quantize Q8_0 rc={rc}')
if rc == 0:
    ty = tensor_types(out_path)
    check(ty.get('blk.0.attn_q.weight') == GGML['Q8_0'] and ty.get('blk.0.ffn_down.weight') == GGML['F16'],
          f'control Q8_0: attn_q {ty.get("blk.0.attn_q.weight")} ffn_down {ty.get("blk.0.ffn_down.weight")}')

# 5. the models above RUN: prompt processing goes through iqk_moe_fused_up_gate (ffn up/gate 64 x 172), whose dequant path
#    (Q8_0 / Q4_0 / IQ4_NL -> Q8_0_R8, 8-row interleave) asserted Nx % 8 on 172 rows (iqk_mul_mat.cpp, rc 3) before the guard.
for qt in ('Q8_0', 'Q4_0', 'IQ4_NL', 'IQ4_KS_R16'):
    out_path = os.path.join(work, f'run-{qt}.gguf')
    rc, log = quantize(model, out_path, qt)
    check(rc == 0, f'quantize --pure {qt} for the run check rc={rc}')
    if rc == 0:
        lrc, name, plog = ftype_line(out_path)
        m = re.search(r'Final estimate: PPL = ([0-9.]+)', plog)
        check(lrc == 0 and m is not None, f'{qt}: perplexity runs (rc={lrc}, {m.group(0) if m else "no PPL line"})')

print(f'{passes} passed, {len(fails)} failed')
sys.exit(1 if fails else 0)
