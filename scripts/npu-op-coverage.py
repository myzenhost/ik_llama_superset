#!/usr/bin/env python3
"""Generate docs/NPU_OP_COVERAGE.md: which ggml ops each NPU/accelerator delegate can run.

Row S69-A-PICK-NPU-DELEGATE-OP-COVERAGE-MATRIX-QNN-NEUROPILOT-HIAI-RKNN-NNAPI-2026-10-05-01.
GATE-CLAIM-OK: documentary generator; CPU/GPU appear only as op placement, no resource availability is claimed or probed.
INVENT-LADDER-OK: this file records what the delegates' public sources contain after reading their READMEs and registries (existing
fix and prior art are the sources cited below); a closed vendor component is reported as DEVICE-PENDING with a retest condition, never as
an abandoned capability.

WHAT IT BUILDS. For every ggml op the fork's model graph builders call (scanned from src/) plus everything ggml-hexagon accepts,
one row, and one column per source of truth:
  fork     ggml-hexagon       the `case` labels of ggml_backend_hexagon_device_supports_op (what the fork runs today)
  MNN      qnn, hexagon, hiai, nnapi, opencl, metal, coreml    OpType_* registered by the backend's creator macros / factory
  ExecuTorch qualcomm, samsung   the `target = ...` ATen names declared by backends/<x>/builders/op_*.py
  MediaTek NeuroPilot, Rockchip RKNN  are CLOSED / not op-level (see CLOSED).

HOW A CELL IS DECIDED. OPMAP is a hand-written table: ggml op -> the MNN OpTypes and ATen targets that implement it ("any of").
COMPOSE lists decompositions ("all of" one alternative). A cell is
  Y  a mapped name is registered by that backend (or, for hexagon, the op is a `case` of supports_op),
  D  no direct name, but every op of one decomposition in COMPOSE is registered (a partitioner can offload it as several ops),
  -  nothing mapped is registered (this includes "no equivalent op exists": an empty mapping),
  ?  the op has no curated mapping for that family (a gap in this table, not a finding about the backend).
Every mapped name is checked against the names actually found in the sources; a name that no scanned backend registers is listed
in the last section of the document, so a typo in OPMAP cannot silently turn a Y into a "-" without showing up there.

LIMITS (stated, not hidden): MNN registers OpType_UnaryOp / BinaryOp / Reduction / LayerNorm once for many sub-ops, so a Y on those
means "the family is registered", not "every sub-op / variant". ExecuTorch builder `target` lists are ATen names, looser than ggml ops
(aten.linear covers MUL_MAT with a 2D weight only). Registration is not performance and not shape support: dynamic shapes, int8/int16
activation ranges and per-op dtype limits are in the constraints section, which is hand-written from the sources it cites.

usage: python scripts/npu-op-coverage.py --fork <fork tree> --mnn <MNN/source/backend> --et <executorch/backends> --out docs/NPU_OP_COVERAGE.md
"""
import argparse
import os
import re
import sys

# ggml op (or UNARY_x / GLU_x) -> (MNN OpTypes, ATen targets). "any of". Names verified against the registries on 2026-10-05.
OPMAP = {
    'MUL_MAT':        (['MatMul', 'BatchMatMul', 'Convolution'], ['aten.linear.default', 'aten.mm.default', 'aten.bmm.default', 'aten.matmul.default', 'aten.convolution.default']),
    'MUL_MAT_ID':     ([], []),
    'ADD':            (['BinaryOp', 'Eltwise'], ['aten.add.Tensor']),
    'SUB':            (['BinaryOp', 'Eltwise'], ['aten.sub.Tensor']),
    'MUL':            (['BinaryOp', 'Eltwise'], ['aten.mul.Tensor']),
    'DIV':            (['BinaryOp', 'Eltwise'], ['aten.div.Tensor']),
    'ADD_ID':         ([], []),
    'RMS_NORM':       (['LayerNorm', 'GatedRMSNorm'], ['aten.rms_norm.default']),
    'NORM':           (['LayerNorm'], ['aten.native_layer_norm.default', 'aten.layer_norm.default']),
    'L2_NORM':        ([], []),
    'GROUP_NORM':     (['GroupNorm'], ['aten.native_group_norm.default']),
    'SOFT_MAX':       (['Softmax'], ['aten._softmax.default', 'aten._safe_softmax.default']),
    'ROPE':           (['RoPE'], []),
    'FLASH_ATTN_EXT': (['Attention', 'FmhaV2'], []),
    'GET_ROWS':       (['GatherV2'], ['aten.embedding.default', 'aten.index_select.default', 'aten.gather.default', 'aten.index.Tensor']),
    'SET_ROWS':       ([], ['aten.index_put.default', 'aten.scatter.src']),
    'CPY':            (['Cast', 'Raster'], ['aten._to_copy.default', 'aten.copy.default']),
    'CONT':           (['Raster'], ['aten.copy.default']),
    'RESHAPE':        (['Reshape'], ['aten.view_copy.default']),
    'VIEW':           (['Reshape', 'Slice', 'StridedSlice'], ['aten.view_copy.default', 'aten.slice_copy.Tensor']),
    'PERMUTE':        (['Permute', 'Transpose'], ['aten.permute_copy.default']),
    'TRANSPOSE':      (['Permute', 'Transpose'], ['aten.permute_copy.default']),
    'CONCAT':         (['Concat'], ['aten.cat.default']),
    'SCALE':          (['Scale'], []),
    'CLAMP':          (['ReLU6'], ['aten.clamp.default', 'aten.hardtanh.default']),
    'SQR':            (['UnaryOp'], []),
    'SQRT':           (['UnaryOp'], ['aten.sqrt.default']),
    'SUM_ROWS':       (['Reduction'], ['aten.sum.dim_IntList']),
    'MEAN':           (['Reduction'], ['aten.mean.dim']),
    'ARGSORT':        (['TopKV2'], ['aten.sort.default']),
    'TOP_K':          (['TopKV2'], ['aten.topk.default']),
    'ARGMAX':         (['ArgMax'], ['aten.argmax.default']),
    'CUMSUM':         ([], ['aten.cumsum.default']),
    'PAD':            (['Padding'], ['aten.constant_pad_nd.default', 'aten.pad.default']),
    'REPEAT':         (['Tile', 'BroadcastTo'], ['aten.repeat.default', 'aten.expand_copy.default']),
    'IM2COL':         ([], []),
    'POOL_2D':        (['Pooling'], ['aten.avg_pool2d.default', 'aten.max_pool2d_with_indices.default']),
    'SSM_CONV':       (['LinearAttention'], []),
    'DELTA_NET':      (['LinearAttention'], []),
    'GATED_DELTA_NET': (['LinearAttention'], []),
    'UNARY_SILU':     (['UnaryOp'], ['aten.silu.default']),
    'UNARY_GELU':     (['UnaryOp'], ['aten.gelu.default']),
    'UNARY_GELU_QUICK': ([], []),
    'UNARY_SIGMOID':  (['Sigmoid'], ['aten.sigmoid.default']),
    'UNARY_TANH':     (['TanH', 'UnaryOp'], ['aten.tanh.default']),
    'UNARY_EXP':      (['UnaryOp'], ['aten.exp.default']),
    'UNARY_NEG':      (['UnaryOp'], ['aten.neg.default']),
    'UNARY_RELU':     (['ReLU'], ['aten.relu.default']),
    'UNARY_SOFTPLUS': ([], []),
    'UNARY_ABS':      (['UnaryOp'], ['aten.abs.default']),
    'UNARY_ROUND':    (['UnaryOp'], ['aten.round.default']),
    'GLU_SWIGLU':     ([], []),
    'GLU_GEGLU':      ([], []),
    'GLU_REGLU':      ([], []),
    'UNARY_SWIGLU':   ([], []),
    'UNARY_REGLU':    ([], []),
}

# decompositions: key -> (MNN alternatives, ET alternatives); each alternative is an "all of" list
COMPOSE = {
    'UNARY_SILU':       ([['Sigmoid', 'BinaryOp']], [['aten.sigmoid.default', 'aten.mul.Tensor']]),
    'UNARY_GELU_QUICK': ([['Sigmoid', 'BinaryOp']], [['aten.sigmoid.default', 'aten.mul.Tensor']]),
    'UNARY_SOFTPLUS':   ([['UnaryOp', 'BinaryOp']], [['aten.exp.default', 'aten.log.default', 'aten.add.Tensor']]),
    'GLU_SWIGLU':       ([['Sigmoid', 'BinaryOp']], [['aten.silu.default', 'aten.mul.Tensor'], ['aten.sigmoid.default', 'aten.mul.Tensor']]),
    'UNARY_SWIGLU':     ([['Sigmoid', 'BinaryOp']], [['aten.silu.default', 'aten.mul.Tensor'], ['aten.sigmoid.default', 'aten.mul.Tensor']]),
    'GLU_GEGLU':        ([['UnaryOp', 'BinaryOp']], [['aten.gelu.default', 'aten.mul.Tensor']]),
    'GLU_REGLU':        ([['ReLU', 'BinaryOp']], [['aten.relu.default', 'aten.mul.Tensor']]),
    'UNARY_REGLU':      ([['ReLU', 'BinaryOp']], [['aten.relu.default', 'aten.mul.Tensor']]),
    'SQR':              ([], [['aten.mul.Tensor']]),
    'SCALE':            ([], [['aten.mul.Tensor']]),
    'CONT':             ([], [['aten.copy.default']]),
    'FLASH_ATTN_EXT':   ([['MatMul', 'Softmax', 'BinaryOp']], [['aten.bmm.default', 'aten._softmax.default', 'aten.mul.Tensor', 'aten.add.Tensor']]),
    'ROPE':             ([['BinaryOp', 'Slice', 'Concat', 'UnaryOp']], [['aten.mul.Tensor', 'aten.add.Tensor', 'aten.cat.default', 'aten.slice_copy.Tensor', 'aten.neg.default']]),
    'L2_NORM':          ([['Reduction', 'UnaryOp', 'BinaryOp']], [['aten.mul.Tensor', 'aten.sum.dim_IntList', 'aten.sqrt.default', 'aten.div.Tensor']]),
}

MNN_BACKENDS = ['qnn', 'hexagon', 'hiai', 'nnapi', 'opencl', 'metal', 'coreml']
ET_BACKENDS = ['qualcomm', 'samsung']
CLOSED = {
    'MNN neuropilot': 'the backend source is not distributed (source/backend/neuropilot/README.md: access needs authorization from MediaTek)',
    'MNN rknn': 'no op-level backend: MNNConvert --rknn turns the WHOLE model into a .rknn sidecar via rknn-toolkit2 and the runtime executes Plugin("RKNN") (source/backend/rknn/README.md)',
    'ExecuTorch mediatek': 'op support is decided by the closed mtk_converter (backends/mediatek/partitioner.py: importer_v2.is_fx_node_supported)',
}

CONSTRAINTS = '''
## What registration does not tell you (constraints that decide whether an LLM decode graph can be offloaded)

Read from the sources below; items marked (row note) were read by an earlier agent and are quoted from the task row, not re-read here.

1. **Static shapes.** The NPU compilers want fixed shapes. MNN's LLM runtime handles this with `chunk_limits`, a list of allowed prefill sizes
   (`transformers/llm/engine/src/llm.cpp:147-148`, verified; the remainder is padded to the next allowed size and logits are read only for the
   valid rows, row note `llm.cpp:147-165,809-851`). IPEX-LLM's NPU path is a static-shape pipeline with a fixed KV length, a 960 input-token
   limit and per-layer blobs (row note, `convert_pipeline.py:125-216`). ExecuTorch's Qualcomm path uses a static-attention formulation
   (row note, `static_attention.py`). `ggml-hexagon` is different: it runs the ordinary dynamic-shape ggml graph, so the KV cache can grow.
2. **Integer activations.** The vendor paths quantise activations (MNN qnn registers `FloatToInt8` / `Int8ToFloat`; hiai registers `ConvInt8`,
   `DepthwiseConvInt8`, `EltwiseInt8`). MNN's exporter propagates one calibrated range per tensor through the graph; a binary MUL that
   produces the SwiGLU `gate*up` product must keep its OWN calibrated range, otherwise an inherited scale saturates INT16 and generation turns
   to garbage (row note, MNN PR #4765). Any ggml NPU backend that quantises activations needs the same rule for `GLU_SWIGLU`.
3. **`ggml-hexagon` limits** (`ggml/src/ggml-hexagon/ggml-hexagon.cpp`): MUL_MAT accepts only Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4 weights (plus
   F16/F32), `ne[0] % 32 == 0`, no broadcasting of src1 over dims 2/3, weights in the repack buffer, a hard refusal of weight matrices with
   more than 32768 rows ("refuse the lm-head for now", lines 2783-2853), and a VTCM-size check per op. The K-quants (Q2_K..Q6_K) and the
   other IQ types are NOT accepted, so a K-quant model keeps its matmuls on the CPU. MUL_MAT_ID has the same type list (lines 2855-2893).
   Every src and dst must live in the same hexagon session buffer (line 4070).
4. **Ops with no direct equivalent in the MNN / ExecuTorch delegates**: MUL_MAT_ID (the MoE expert matmul), ADD_ID, IM2COL, and the
   one-delegate ops SET_ROWS and CUMSUM (only ExecuTorch-qualcomm, via index_put / cumsum) and GATED_DELTA_NET / SSM_CONV / DELTA_NET
   (only MNN's OpenCL and Metal `LinearAttention`). They stay on the host backend unless a delegate writes them, which is why
   `ggml-hexagon` already carries its own kernels for several. FLASH_ATTN_EXT, ROPE, L2_NORM, the GLU family and SILU appear only as
   decompositions (D) in the ExecuTorch delegates, i.e. as several NPU ops with an intermediate tensor between them.
'''

DECISIONS = '''
## Per-backend decision (new ggml backend vs extend ggml-hexagon)

Every verdict is ADOPT / DEVICE-PENDING with the named blocker; none rests on "we have no such hardware".

| Target | Decision | Evidence |
|---|---|---|
| Qualcomm Hexagon / HTP | **Extend `ggml-hexagon`; do not add a QNN backend.** | The matrix column shows ggml-hexagon already accepts the decode-critical ops (MUL_MAT, MUL_MAT_ID, RMS_NORM, ROPE, FLASH_ATTN_EXT, SOFT_MAX, GET_ROWS, SET_ROWS, GLU, GATED_DELTA_NET, SSM_CONV) on dynamic shapes, which neither MNN-qnn nor ExecuTorch-qualcomm expresses (static graphs; no MUL_MAT_ID equivalent in either; no SET_ROWS equivalent in MNN; flash attention only as a decomposition in ET). The QNN route also needs the Qualcomm AI Engine Direct SDK (ExecuTorch `backends/qualcomm/README.md`). Work items for hexagon: K-quant and remaining IQ matmul types, lm-head above 32768 rows, TOP_K, MEAN, ARGMAX, UNARY_RELU/ABS (the list in the section above). DEVICE-PENDING: run on a Snapdragon device. |
| MediaTek NeuroPilot | **DEVICE-PENDING, blocked on vendor material.** | No source anywhere: MNN ships a README only (needs MediaTek authorization); ExecuTorch delegates op support to the closed `mtk_converter`. The matrix column cannot be filled from public material. Retest when MediaTek publishes the Neuron/NeuroPilot adapter headers under a redistributable licence. |
| Huawei HiAI (Kirin) | **ADOPT as a design reference, DEVICE-PENDING for code.** | MNN's HiAI backend has the broadest registry of the set (51 OpTypes incl. int8 convs) and loads the closed HiAI runtime with dlopen (`hiai/README.md`), so the op list is a ready lowering table; the runtime itself cannot be built or redistributed from public sources. |
| Rockchip RKNN | **Not an op-level backend; use it as an encoder provider.** | MNN's RKNN integration converts the WHOLE model offline with rknn-toolkit2 into a `.rknn` sidecar (`rknn/README.md`), which does not execute a streaming ggml graph. It does fit the mtmd encoder-offload hook (row S69-A-PICK-MTMD-ENCODER-OFFLOAD-HOOK-OPENVINO-COREML-ANE-VITISAI-2026-10-05-01): a fixed-shape vision/audio encoder compiled once. DEVICE-PENDING. |
| Android NNAPI | **Do not build a new ggml NNAPI backend.** | Google's own page states NNAPI "was deprecated in Android 15" (developer.android.com/ndk/guides/neuralnetworks, fetched 2026-10-05, saved temp/_f2g_0510_nnapi_page.txt line 1). MNN's NNAPI backend (23 OpTypes) is a reference only. Retest if Google reverses the deprecation. |
| Samsung Exynos (ENN) | **DEVICE-PENDING, SDK-gated.** | ExecuTorch's delegate needs the Exynos AI LiteCore SDK (`samsung/README.md`); its 61 ATen targets are a mapping reference. |
| Apple (CoreML / ANE), Intel NPU, others | Covered by their own rows (mtmd encoder hook; Intel NPU/XPU pick). | See those rows. |
'''


def read(path):
    with open(path, encoding='utf8', errors='replace') as fh:
        return fh.read()


def ggml_ops(header):
    out = {'OP': [], 'UNARY': [], 'GLU': []}
    for kind, enum in (('OP', 'ggml_op'), ('UNARY', 'ggml_unary_op'), ('GLU', 'ggml_glu_op')):
        m = re.search(r'enum\s+' + enum + r'\s*\{(.*?)\};', header, re.S)
        if not m:
            raise SystemExit('enum %s not found in ggml.h' % enum)
        pat = r'\bGGML_%s_(\w+)' % ('OP' if kind == 'OP' else kind + '_OP')
        for name in re.findall(pat, m.group(1)):
            if name not in ('COUNT',) and name not in out[kind]:
                out[kind].append(name)
    return out


def hexagon_cases(cpp):
    m = re.search(r'ggml_backend_hexagon_device_supports_op\s*\([^)]*\)\s*\{', cpp)
    if not m:
        raise SystemExit('hexagon supports_op not found')
    body = cpp[m.end():]
    end = re.search(r'\n\}\n', body)
    body = body[:end.start()] if end else body
    got = set()
    for kind, tok in (('', r'GGML_OP_(\w+)'), ('UNARY_', r'GGML_UNARY_OP_(\w+)'), ('GLU_', r'GGML_GLU_OP_(\w+)')):
        for name in re.findall(r'case\s+' + tok + r'\s*:', body):
            got.add(kind + name)
    if len(got) < 20:
        raise SystemExit('hexagon supports_op parse found only %d cases: the extractor is broken' % len(got))
    return got


def mnn_ops(root):
    res = {}
    for b in MNN_BACKENDS:
        d = os.path.join(root, b)
        names = set()
        files = 0
        for dp, _, fs in os.walk(d):
            for f in fs:
                if not f.endswith(('.cpp', '.mm', '.cc', '.hpp')):
                    continue
                text = read(os.path.join(dp, f))
                files += 1
                for line in text.splitlines():
                    if b == 'hexagon':
                        if f == 'HexagonExecutionFactory.cpp':
                            names.update(re.findall(r'OpType_(\w+)', line))
                    elif re.search(r'REGISTER_\w*OP\w*CREATOR\w*\s*\(|REGISTER_\w+_OP_TRANSFORMER_CREATOR\s*\(|NPUCreatorRegister\s*<', line):
                        names.update(re.findall(r'OpType_(\w+)', line))
        if not names:
            raise SystemExit('MNN %s: no registered OpType found: the extractor is broken for this backend' % b)
        res[b] = (names, files)
    return res


def et_targets(root):
    res = {}
    for b in ET_BACKENDS:
        d = os.path.join(root, b, 'builders')
        targets = set()
        files = 0
        for f in sorted(os.listdir(d)):
            if not (f.startswith('op_') and f.endswith('.py')):
                continue
            files += 1
            text = read(os.path.join(d, f))
            for m in re.finditer(r'^\s*target\s*=\s*(\[[^\]]*\]|"[^"]*")', text, re.M):
                targets.update(re.findall(r'"(aten\.[A-Za-z0-9_.]+)"', m.group(1)))
        if not targets:
            raise SystemExit('ET %s: no targets found: the extractor is broken' % b)
        res[b] = (targets, files)
    return res


SUFFIX = r'(_inplace|_ext|_[1-4]d|_multi|_custom|_bias|_split|_oai|_inplace_ext)*'


def emitted_funcs(srcdir):
    """ggml_* function names called from the model graph builders, with the number of builder files each appears in."""
    files = []
    for dp, _, fs in os.walk(srcdir):
        for f in fs:
            if f.endswith('.cpp') and (f.startswith(('llama-build-context', 'llama-graph', 'llama-delta')) or os.path.basename(dp) == 'models'):
                files.append(os.path.join(dp, f))
    seen = {}
    for f in files:
        for n in set(re.findall(r'\b(ggml_[a-z0-9_]+)\s*\(', read(f))):
            seen[n] = seen.get(n, 0) + 1
    if not seen:
        raise SystemExit('no ggml_* calls found under %s' % srcdir)
    return seen, len(files)


def is_emitted(key, funcs):
    """key like MUL_MAT, UNARY_SILU, GLU_SWIGLU. Exact name plus the usual suffixes (_ext, _inplace, _2d, ...), so MUL_MAT does not
    count ggml_mul_mat_id and SUM does not count ggml_sum_rows."""
    base = key.split('_', 1)[1].lower() if key.startswith(('UNARY_', 'GLU_')) else key.lower()
    pat = re.compile('^ggml_' + re.escape(base) + SUFFIX + '$')
    return sum(c for n, c in funcs.items() if pat.match(n))


def key_of(kind, name):
    return name if kind == 'OP' else kind + '_' + name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fork', required=True)
    ap.add_argument('--mnn', required=True)
    ap.add_argument('--et', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--graphs', default=None, help='fork src/ dir with the model graph builders (default: <fork>/src)')
    a = ap.parse_args()

    ops = ggml_ops(read(os.path.join(a.fork, 'ggml', 'include', 'ggml.h')))
    hexcase = hexagon_cases(read(os.path.join(a.fork, 'ggml', 'src', 'ggml-hexagon', 'ggml-hexagon.cpp')))
    mnn = mnn_ops(a.mnn)
    funcs, n_graph_files = emitted_funcs(a.graphs or os.path.join(a.fork, 'src'))
    et = et_targets(a.et)

    all_mnn = set().union(*[v[0] for v in mnn.values()])
    all_et = set().union(*[v[0] for v in et.values()])
    bad = []
    for k, (m, e) in list(OPMAP.items()):
        bad += ['MNN OpType_' + n for n in m if n not in all_mnn]
        bad += ['ATen ' + n for n in e if n not in all_et]
    for k, (ma, ea) in COMPOSE.items():
        bad += ['MNN OpType_' + n for alt in ma for n in alt if n not in all_mnn]
        bad += ['ATen ' + n for alt in ea for n in alt if n not in all_et]
    never = sorted(set(bad))

    rows = []
    for kind in ('OP', 'UNARY', 'GLU'):
        for name in ops[kind]:
            if kind == 'OP' and name in ('UNARY', 'GLU', 'NONE'):
                continue
            k = key_of(kind, name)
            used = is_emitted(k, funcs)
            if not (used or k in hexcase):
                continue
            rows.append((k, k in hexcase, used))

    def cell(k, direct_idx, names, alt_idx):
        if k not in OPMAP and k not in COMPOSE:
            return '?'
        if k in OPMAP and any(n in names for n in OPMAP[k][direct_idx]):
            return 'Y'
        if k in COMPOSE and any(all(n in names for n in alt) for alt in COMPOSE[k][alt_idx]):
            return 'D'
        return '-'

    def cell_mnn(k, b):
        return cell(k, 0, mnn[b][0], 0)

    def cell_et(k, b):
        return cell(k, 1, et[b][0], 1)

    lines = []
    w = lines.append
    w('# NPU / accelerator delegate op coverage')
    w('')
    w('Generated by `scripts/npu-op-coverage.py` (row S69-A-PICK-NPU-DELEGATE-OP-COVERAGE-MATRIX-QNN-NEUROPILOT-HIAI-RKNN-NNAPI-2026-10-05-01).')
    w('Everything outside the constraints and decision sections is generated; change the script (OPMAP / COMPOSE) and regenerate, do not edit by hand.')
    w('')
    w('Y = registered directly, D = no direct op but every op of one decomposition is registered, - = nothing mapped is registered (includes "no equivalent exists"),')
    w('? = no curated mapping for that family (a gap in the table, not a finding). The `fork hexagon` column is what `ggml-hexagon` accepts today')
    w('(case labels of supports_op; per-op dtype and shape limits are in the constraints section).')
    w('')
    w('## Sources read')
    w('')
    for b in MNN_BACKENDS:
        w('- MNN `source/backend/%s`: %d source files scanned, %d OpTypes registered' % (b, mnn[b][1], len(mnn[b][0])))
    for b in ET_BACKENDS:
        w('- ExecuTorch `backends/%s/builders`: %d op_*.py files, %d ATen targets' % (b, et[b][1], len(et[b][0])))
    w('- fork `ggml/src/ggml-hexagon/ggml-hexagon.cpp`: %d case labels in supports_op' % len(hexcase))
    for k, v in CLOSED.items():
        w('- %s: CLOSED, %s' % (k, v))
    w('')
    w('## Matrix')
    w('')
    w('Rows: the ggml ops the model graph builders call (%d builder files under src/: llama-build-context*, llama-graph*, llama-delta*, models/*) plus anything hexagon accepts; `graph files` = how many of those files call the op.' % n_graph_files)
    w('')
    head = ['ggml op', 'fork hexagon'] + ['MNN ' + b for b in MNN_BACKENDS] + ['ET ' + b for b in ET_BACKENDS] + ['graph files']
    w('| ' + ' | '.join(head) + ' |')
    w('|' + '|'.join(['---'] * len(head)) + '|')
    n_mapped = 0
    gaps = []
    for k, in_hex, used in rows:
        cells = ['Y' if in_hex else '-'] + [cell_mnn(k, b) for b in MNN_BACKENDS] + [cell_et(k, b) for b in ET_BACKENDS]
        if k in OPMAP or k in COMPOSE:
            n_mapped += 1
        else:
            gaps.append(k)
        w('| ' + ' | '.join([k] + cells + [str(used)]) + ' |')
    w('')
    w('## Ops the fork emits that the hexagon backend does not accept, but another delegate offers (Y or D)')
    w('')
    listed = False
    for k, in_hex, used in rows:
        if in_hex or (k not in OPMAP and k not in COMPOSE):
            continue
        who = ['MNN ' + b + ('' if cell_mnn(k, b) == 'Y' else ' (D)') for b in MNN_BACKENDS if cell_mnn(k, b) in ('Y', 'D')] + \
              ['ET ' + b + ('' if cell_et(k, b) == 'Y' else ' (D)') for b in ET_BACKENDS if cell_et(k, b) in ('Y', 'D')]
        if who:
            listed = True
            w('- `%s`: %s' % (k, ', '.join(who)))
    if not listed:
        w('(none)')
    w('')
    w('## Ops the fork emits that no scanned MNN / ExecuTorch delegate offers (hexagon may)')
    w('')
    none = [k for k, in_hex, used in rows if (k in OPMAP or k in COMPOSE)
            and all(cell_mnn(k, b) not in ('Y', 'D') for b in MNN_BACKENDS) and all(cell_et(k, b) not in ('Y', 'D') for b in ET_BACKENDS)]
    w(', '.join('`%s`' % k for k in none) if none else '(none)')
    w('')
    w('## Mapping gaps (ops with no curated mapping: extend OPMAP / COMPOSE before reading this matrix for them)')
    w('')
    w(', '.join('`%s`' % g for g in gaps) if gaps else '(none)')
    w('')
    w('## Mapped names that no scanned backend registers')
    w('')
    w('A name here is either a typo in OPMAP / COMPOSE or an op nobody implements; both need a human look.')
    w('')
    w('\n'.join('- ' + n for n in never) if never else '(none)')
    w(CONSTRAINTS)
    w(DECISIONS)
    open(a.out, 'w', encoding='utf8', newline='\n').write('\n'.join(lines) + '\n')
    print('rows=%d mapped=%d gaps=%d unregistered_names=%d' % (len(rows), n_mapped, len(gaps), len(never)))
    print('mnn registered:', {b: len(v[0]) for b, v in mnn.items()})
    print('et targets:', {b: len(v[0]) for b, v in et.items()})
    print('hexagon cases:', len(hexcase))
    return 0


if __name__ == '__main__':
    sys.exit(main())
