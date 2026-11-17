// XAttention block selection as a ggml custom op on the flash-attention mask (src/llama-kv-xattn-op.h, the op llm_build_kqv splices when
// --xattn-threshold is set). Row S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01 (step 2: wiring).   GATE-CLAIM-OK: documentation only.
//
// The op runs on tensors laid out exactly as the graph lays them out: K is a 3D view of a cache whose rows interleave the kv heads per cell
// ([head_dim, n_kv, n_head_kv] with nb1 = head_dim * n_head_kv), Q is the permuted view of q_cur ([head_dim, n_tok, n_head], not contiguous).
// Its output is compared with an expectation built HERE from the raw arrays: for every query block and every head, justician_kv::xattn_block_mask
// on contiguous copies, union over heads, deselected tiles set to -inf. Checks:
//   1. op output == expectation, bit for bit, and something IS dropped (a planted heavy key block makes the selection non-trivial);
//   2. first key block and diagonal block are never dropped; nothing visible right of the causal limit appears;
//   3. the same op on ggml_cont copies of K and Q gives the same mask (the strides are honoured, not assumed contiguous);
//   4. threshold 1.0 passes the mask through unchanged; a two-sequence mask (rows alternate sequences) passes through unchanged;
//   5. GQA mapping: query head h scores against kv head h / (n_head / n_head_kv). kv head 0 holds a heavy key block 2 along dim 0 and kv head 1 a
//      heavy block 3 along dim 1; query heads 0,1 point along dim 0 and heads 2,3 along dim 1. Under the grouped mapping every head concentrates
//      on its planted block, so the union drops blocks; under an interleaved mapping (h % n_head_kv) heads 1 and 2 meet unstructured keys and keep
//      everything. CONTROL: the interleaved expectation must differ from the op's output (else check 1 could not tell the mappings apart).
// FAIL-BEFORE: src/llama-kv-xattn-op.h did not exist (no consumer of the step-1 estimator in the tree).
#include "../src/llama-kv-xattn-op.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

struct setup {
    int64_t hd = 32, n_head = 4, n_hkv = 2, n_tok = 512, n_kv = 1024;   // past = 512
    int     block = 128, stride = 8;
};

// raw data in GRAPH layout: kc[(s * n_hkv + kh) * hd + d] (cache rows interleave heads), qc[(t * n_head + h) * hd + d] (q_cur)
static void make_data(const setup & S, std::vector<float> & kc, std::vector<float> & qc) {
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    kc.resize((size_t) (S.n_kv * S.n_hkv * S.hd));
    qc.resize((size_t) (S.n_tok * S.n_head * S.hd));
    for (auto & x : kc) x = nd(rng);
    for (auto & x : qc) x = nd(rng);
    // plant: kv head 0 = key block 2 along dim 0, kv head 1 = key block 3 along dim 1; query heads 0,1 along dim 0, heads 2,3 along dim 1
    for (int64_t s = 2 * S.block; s < 3 * S.block; ++s) kc[(size_t) ((s * S.n_hkv + 0) * S.hd + 0)] = 8.0f;
    for (int64_t s = 3 * S.block; s < 4 * S.block; ++s) kc[(size_t) ((s * S.n_hkv + 1) * S.hd + 1)] = 8.0f;
    for (int64_t t = 0; t < S.n_tok; ++t) for (int64_t h = 0; h < S.n_head; ++h) qc[(size_t) ((t * S.n_head + h) * S.hd + (h < 2 ? 0 : 1))] = 8.0f;
}

static std::vector<float> causal_mask(const setup & S, int64_t rows, bool two_seq) {
    const int64_t past = S.n_kv - S.n_tok;
    std::vector<float> m((size_t) (S.n_kv * rows), -INFINITY);
    for (int64_t t = 0; t < S.n_tok; ++t) for (int64_t s = 0; s < S.n_kv; ++s) {
        bool vis = s <= past + t;
        if (two_seq) vis = vis && ((s % 2) == (t % 2));
        if (vis) m[(size_t) (t * S.n_kv + s)] = 0.0f;
    }
    return m;
}

// run the op; cont = true feeds contiguous copies of K and Q instead of the graph views
static std::vector<float> run_op(const setup & S, const std::vector<float> & kc, const std::vector<float> & qc, const std::vector<float> & mask,
                                 int64_t rows, float threshold, bool cont) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 4);
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * kcache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S.hd * S.n_hkv, S.n_kv);
    ggml_tensor * qcur   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S.hd, S.n_head, S.n_tok);
    ggml_tensor * m      = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S.n_kv, rows);
    ggml_tensor * k = ggml_view_3d(ctx, kcache, S.hd, S.n_kv, S.n_hkv, ggml_row_size(GGML_TYPE_F16, S.hd) * S.n_hkv, ggml_row_size(GGML_TYPE_F16, S.hd), 0);
    ggml_tensor * q = ggml_permute(ctx, qcur, 0, 2, 1, 3);
    if (cont) { k = ggml_cont(ctx, k); q = ggml_cont(ctx, q); }
    justician_kv::xattn_op_cfg cfg;
    cfg.threshold = threshold; cfg.stride = S.stride; cfg.block = S.block;
    ggml_tensor * out = ggml_map_custom3(ctx, m, k, q, justician_kv::xattn_mask_op, GGML_N_TASKS_MAX, &cfg);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<ggml_fp16_t> kh(kc.size()), mh(mask.size());
    for (size_t i = 0; i < kc.size(); ++i) kh[i] = ggml_fp32_to_fp16(kc[i]);
    for (size_t i = 0; i < mask.size(); ++i) mh[i] = ggml_fp32_to_fp16(mask[i]);
    ggml_backend_tensor_set(kcache, kh.data(), 0, kh.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(qcur, qc.data(), 0, qc.size() * sizeof(float));
    ggml_backend_tensor_set(m, mh.data(), 0, mh.size() * sizeof(ggml_fp16_t));
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(gf, out);
    CHECK(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
    std::vector<ggml_fp16_t> oh((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(out, oh.data(), 0, oh.size() * sizeof(ggml_fp16_t));
    std::vector<float> o(oh.size());
    for (size_t i = 0; i < oh.size(); ++i) o[i] = ggml_fp16_to_fp32(oh[i]);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return o;
}

// expectation from the raw arrays (K and Q rounded through f16 for K, as the op reads them; Q is f32)
static std::vector<float> expect_mask(const setup & S, const std::vector<float> & kc, const std::vector<float> & qc, const std::vector<float> & mask,
                                      float threshold, int64_t * dropped_out, bool interleaved = false) {
    std::vector<float> e = mask;
    const int64_t past = S.n_kv - S.n_tok, B = S.block, gqa = S.n_head / S.n_hkv;
    justician_kv::xattn_cfg cfg; cfg.block = S.block; cfg.stride = S.stride; cfg.threshold = threshold;
    int64_t dropped = 0;
    for (int64_t qb = 0; qb < S.n_tok / B; ++qb) {
        const int64_t kb_n = past / B + qb + 1, nk = kb_n * B;
        std::vector<uint8_t> keep((size_t) kb_n, 0);
        for (int64_t h = 0; h < S.n_head; ++h) {
            const int64_t kh = interleaved ? h % S.n_hkv : h / gqa;
            std::vector<float> K((size_t) (nk * S.hd)), Q((size_t) (B * S.hd));
            for (int64_t s = 0; s < nk; ++s) for (int64_t d = 0; d < S.hd; ++d)
                K[(size_t) (s * S.hd + d)] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(kc[(size_t) ((s * S.n_hkv + kh) * S.hd + d)]));
            for (int64_t i = 0; i < B; ++i) for (int64_t d = 0; d < S.hd; ++d)
                Q[(size_t) (i * S.hd + d)] = qc[(size_t) (((qb * B + i) * S.n_head + h) * S.hd + d)];
            const auto m = justician_kv::xattn_block_mask(Q.data(), K.data(), 1, (int) B, (int) nk, (int) S.hd, cfg);
            CHECK(m.size() == (size_t) kb_n);
            for (int64_t kb = 0; kb < kb_n && kb < (int64_t) m.size(); ++kb) keep[(size_t) kb] |= m[(size_t) kb];
        }
        for (int64_t kb = 0; kb < kb_n; ++kb) {
            if (keep[(size_t) kb]) continue;
            ++dropped;
            for (int64_t t = qb * B; t < (qb + 1) * B; ++t) for (int64_t s = kb * B; s < (kb + 1) * B; ++s) e[(size_t) (t * S.n_kv + s)] = -INFINITY;
        }
    }
    if (dropped_out) *dropped_out = dropped;
    return e;
}

static int64_t n_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return -1;
    int64_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const bool ai = std::isinf(a[i]) && a[i] < 0, bi = std::isinf(b[i]) && b[i] < 0;
        if (ai != bi || (!ai && a[i] != b[i])) ++n;
    }
    return n;
}

int main() {
    setup S;
    const int64_t rows = GGML_PAD(S.n_tok, GGML_KQ_MASK_PAD);
    std::vector<float> kc, qc;
    make_data(S, kc, qc);
    const std::vector<float> mask = causal_mask(S, rows, false);

    // 1 + 2 + 5
    const float thr = 0.9f;
    int64_t dropped = 0;
    const std::vector<float> want = expect_mask(S, kc, qc, mask, thr, &dropped);
    const std::vector<float> got  = run_op(S, kc, qc, mask, rows, thr, false);
    const int64_t nd = n_diff(got, want);
    const int64_t total_blocks = (S.n_tok / S.block) * ((S.n_kv - S.n_tok) / S.block) + (S.n_tok / S.block) * (S.n_tok / S.block + 1) / 2;
    printf("threshold %.2f: op vs expectation differing elements %lld; dropped %lld of %lld causal blocks\n", thr, (long long) nd, (long long) dropped, (long long) total_blocks);
    CHECK(nd == 0);
    CHECK(dropped > 0);
    {
        const int64_t past = S.n_kv - S.n_tok;
        int64_t bad = 0;
        for (int64_t t = 0; t < S.n_tok; ++t) {
            const int64_t diag = (past + t) / S.block * S.block;
            for (int64_t s = 0; s < S.n_kv; ++s) {
                const bool vis = !(std::isinf(got[(size_t) (t * S.n_kv + s)]));
                if (s > past + t && vis) ++bad;                                 // never right of the causal limit
                if ((s < S.block || (s >= diag && s <= past + t)) && !vis) ++bad; // first block and diagonal block always visible
            }
            if (std::isinf(got[(size_t) (t * S.n_kv + 2 * S.block)]) || std::isinf(got[(size_t) (t * S.n_kv + 3 * S.block)])) ++bad;   // planted blocks selected
        }
        printf("structural violations (first/diagonal/causal/planted): %lld\n", (long long) bad);
        CHECK(bad == 0);
    }

    // 3. contiguous inputs give the same mask
    const std::vector<float> got_c = run_op(S, kc, qc, mask, rows, thr, true);
    printf("contiguous-input run vs view run: differing elements %lld\n", (long long) n_diff(got_c, got));
    CHECK(n_diff(got_c, got) == 0);

    // 5 (GQA control): the expectation with the WRONG head mapping (interleaved h % n_hkv) differs from the op's output
    {
        int64_t dropped_il = 0;
        const std::vector<float> want_il = expect_mask(S, kc, qc, mask, thr, &dropped_il, true);
        printf("GQA control: interleaved-mapping expectation drops %lld blocks and differs from the op in %lld elements (want > 0)\n",
               (long long) dropped_il, (long long) n_diff(want_il, got));
        CHECK(n_diff(want_il, got) > 0);
    }

    // 4. pass-through cases
    const std::vector<float> got_1 = run_op(S, kc, qc, mask, rows, 1.0f, false);
    printf("threshold 1.00: differing from input %lld (want 0)\n", (long long) n_diff(got_1, mask));
    CHECK(n_diff(got_1, mask) == 0);
    const std::vector<float> mask2 = causal_mask(S, rows, true);
    const std::vector<float> got_2 = run_op(S, kc, qc, mask2, rows, thr, false);
    printf("two-sequence mask: differing from input %lld (want 0)\n", (long long) n_diff(got_2, mask2));
    CHECK(n_diff(got_2, mask2) == 0);

    const auto & st = justician_kv::xattn_stats();
    printf("stats: calls=%llu selected=%llu passthrough=%llu causal_blocks=%llu dropped=%llu\n",
           (unsigned long long) st.calls.load(), (unsigned long long) st.units_masked.load(), (unsigned long long) st.units_pass.load(),
           (unsigned long long) st.blocks_causal.load(), (unsigned long long) st.blocks_dropped.load());
    if (n_fail) { fprintf(stderr, "%d check(s) failed\n", n_fail); return 1; }
    printf("test-kv-xattn-op: all checks passed\n");
    return 0;
}
