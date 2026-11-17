// CPU flash attention SKIPS interior all -inf mask tiles (iqk_fa_tile_all_masked in ggml/src/iqk/fa/iqk_fa_templates.h), the kernel half of the
// XAttention prefill block selection. Row S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01.   GATE-CLAIM-OK: documentation only.
//
// The mask is what src/llama-kv-xattn-op.h writes: a causal prefill mask in which whole (128-row query block, 128-key block) tiles in the
// INTERIOR are -inf (first key block and the diagonal block always visible). The trailing clamp (mask_effective_nk1) cannot reach them.
//
// FAIL-BEFORE / PASS-AFTER, deterministic (no timing): V holds NaN in every cell of a deselected block. Such a cell has softmax weight exactly 0, and
// ggml's generic CPU kernel never reads V for a -inf score, so the right output is finite. The iqk kernel without the skip processes the tile and
// adds 0 * NaN = NaN to the row; with the skip the tile is never read. CONTROL (must stay NaN in both builds): the same poison in a VISIBLE block
// reaches the output, so the test does see V. Every live output element is also compared with a double-precision reference that excludes the
// masked cells. With --time the dense-causal and the block-sparse mask are timed (same shapes, finite V): the sparse/dense ratio is reported, not
// asserted, because the host is shared; IQK_FA_TILE_SKIP=0 turns the skip off in the same binary for an A/B.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static const int64_t BLK = 128;

// block (qb, kb) of the sparse mask: kept = first column, diagonal, and every third block in between (a fixed pattern so the test is reproducible)
static bool block_kept(int64_t qb_abs, int64_t kb) { return kb == 0 || kb == qb_abs || kb % 3 == 1; }

enum class poison { none, dropped_block, kept_block };

struct result { double max_err; int64_t n_nan; int64_t n_cmp; double ms; };

static result run(ggml_backend_t backend, int64_t d, int64_t nq, int64_t nkv, int64_t hq, int64_t hkv, bool sparse, poison pz, int time_reps) {
    const int64_t past   = nkv - nq;
    const int64_t nq_pad = GGML_PAD(nq, GGML_KQ_MASK_PAD);
    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, hq, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nkv, hkv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, nkv, hkv, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, nkv, nq_pad, 1, 1);
    const float scale = 1.0f / sqrtf((float) d);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(1234u + (unsigned) (nkv + 7 * nq + 131 * hq));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> qd(ggml_nelements(q)), kf(ggml_nelements(k)), vf(ggml_nelements(v));
    for (auto & x : qd) x = nd(rng);
    for (auto & x : kf) x = nd(rng);
    for (auto & x : vf) x = nd(rng);

    auto visible = [&](int64_t iq, int64_t ik) -> bool {
        if (iq >= nq || ik > iq + past) return false;
        if (!sparse) return true;
        return block_kept((iq + past) / BLK, ik / BLK);
    };
    // poison one key block for every kv head: a deselected interior block (dropped) or a block every row sees (kept = block 0)
    int64_t poison_kb = -1;
    if (pz == poison::dropped_block) {
        for (int64_t kb = 1; kb < (past / BLK); ++kb) if (!block_kept(past / BLK, kb)) { poison_kb = kb; break; }   // dropped for the FIRST query block
    } else if (pz == poison::kept_block) {
        poison_kb = 0;
    }
    if (poison_kb >= 0) {
        for (int64_t h = 0; h < hkv; ++h) for (int64_t ik = poison_kb * BLK; ik < (poison_kb + 1) * BLK; ++ik)
            for (int64_t dd = 0; dd < d; ++dd) vf[(size_t) ((h * nkv + ik) * d + dd)] = NAN;
    }

    std::vector<ggml_fp16_t> kh(kf.size()), vh(vf.size()), mh((size_t) (nkv * nq_pad));
    ggml_fp32_to_fp16_row(kf.data(), kh.data(), kh.size());
    ggml_fp32_to_fp16_row(vf.data(), vh.data(), vh.size());
    for (int64_t iq = 0; iq < nq_pad; ++iq)
        for (int64_t ik = 0; ik < nkv; ++ik) mh[(size_t) (iq * nkv + ik)] = ggml_fp32_to_fp16(visible(iq, ik) ? 0.0f : -INFINITY);
    ggml_fp16_to_fp32_row(kh.data(), kf.data(), kf.size());
    ggml_fp16_to_fp32_row(vh.data(), vf.data(), vf.size());
    ggml_backend_tensor_set(q, qd.data(), 0, qd.size()*sizeof(float));
    ggml_backend_tensor_set(k, kh.data(), 0, kh.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, vh.data(), 0, vh.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m, mh.data(), 0, mh.size()*sizeof(ggml_fp16_t));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, out);
    result r = { 0.0, 0, 0, 0.0 };
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL graph compute\n");
        n_fail++;
        r.max_err = INFINITY;
    }
    if (time_reps > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < time_reps; ++i) ggml_backend_graph_compute(backend, gf);
        r.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / time_reps;
    }
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));

    // out [dv, hq, nq] ; reference over visible cells only (a masked cell has weight exactly 0 and its V is never read)
    const int64_t rep = hq / hkv;
    std::vector<double> sc((size_t) nkv);
    for (int64_t iq = 0; iq < nq; ++iq) {
        for (int64_t ih = 0; ih < hq; ++ih) {
            const int64_t ikv = ih / rep;
            double mx = -INFINITY;
            for (int64_t ik = 0; ik < nkv; ++ik) {
                if (!visible(iq, ik)) { sc[(size_t) ik] = -INFINITY; continue; }
                double s = 0.0;
                for (int64_t dd = 0; dd < d; ++dd) s += (double) qd[(size_t) ((ih * nq + iq) * d + dd)] * (double) kf[(size_t) ((ikv * nkv + ik) * d + dd)];
                sc[(size_t) ik] = s * scale;
                if (sc[(size_t) ik] > mx) mx = sc[(size_t) ik];
            }
            double den = 0.0;
            for (int64_t ik = 0; ik < nkv; ++ik) { sc[(size_t) ik] = std::isfinite(sc[(size_t) ik]) ? std::exp(sc[(size_t) ik] - mx) : 0.0; den += sc[(size_t) ik]; }
            for (int64_t dd = 0; dd < d; ++dd) {
                double o = 0.0;
                for (int64_t ik = 0; ik < nkv; ++ik) if (sc[(size_t) ik] != 0.0) o += sc[(size_t) ik] * (double) vf[(size_t) ((ikv * nkv + ik) * d + dd)];
                o /= den;
                const float got = res[(size_t) ((iq * hq + ih) * d + dd)];
                if (std::isnan(got)) { r.n_nan++; continue; }
                if (std::isnan(o)) continue;   // the reference itself reads a poisoned VISIBLE cell (control case)
                const double err = std::fabs((double) got - o);
                if (err > r.max_err) r.max_err = err;
                r.n_cmp++;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    bool time_it = false;
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "--time")) time_it = true;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "no CPU backend\n"); return 2; }
    ggml_backend_cpu_set_n_threads(backend, 4);
    const char * skip_env = getenv("IQK_FA_TILE_SKIP");
    printf("IQK_FA_TILE_SKIP=%s\n", skip_env ? skip_env : "(unset)");

    struct shape { int64_t d, nq, nkv, hq, hkv; };
    // hq=8/nth=4: whole heads per thread; hq=9/nth=4: the kernel splits the 512 rows into 4 groups of 128 (one query block each)
    const shape shapes[] = { { 64, 512, 2048, 8, 2 }, { 128, 512, 2048, 8, 2 }, { 64, 512, 2048, 9, 3 } };
    for (const shape & s : shapes) {
        // 1. sparse mask, finite V: matches the reference
        result a = run(backend, s.d, s.nq, s.nkv, s.hq, s.hkv, true, poison::none, 0);
        printf("  d=%-3lld hq=%lld sparse, finite V              max|err| %.2e over %lld, NaN %lld\n", (long long) s.d, (long long) s.hq, a.max_err, (long long) a.n_cmp, (long long) a.n_nan);
        CHECK(a.n_nan == 0 && a.n_cmp > 0 && a.max_err <= 2.0e-3);
        // 2. NaN in a DESELECTED block: the skip never reads it (fail-before: 0 * NaN = NaN)
        result b = run(backend, s.d, s.nq, s.nkv, s.hq, s.hkv, true, poison::dropped_block, 0);
        printf("  d=%-3lld hq=%lld sparse, NaN in a dropped block  max|err| %.2e over %lld, NaN %lld  (want 0 NaN)\n", (long long) s.d, (long long) s.hq, b.max_err, (long long) b.n_cmp, (long long) b.n_nan);
        CHECK(b.n_nan == 0 && b.n_cmp > 0 && b.max_err <= 2.0e-3);
        // 3. CONTROL: NaN in a block every row sees must reach the output
        result c = run(backend, s.d, s.nq, s.nkv, s.hq, s.hkv, true, poison::kept_block, 0);
        printf("  d=%-3lld hq=%lld sparse, NaN in a KEPT block     NaN %lld of %lld  (control: want all NaN)\n", (long long) s.d, (long long) s.hq, (long long) c.n_nan, (long long) (s.nq * s.hq * s.d));
        CHECK(c.n_nan == s.nq * s.hq * s.d);
        // 4. dense causal, finite V: unchanged behaviour (nothing interior to skip)
        result e = run(backend, s.d, s.nq, s.nkv, s.hq, s.hkv, false, poison::none, 0);
        printf("  d=%-3lld hq=%lld dense causal, finite V          max|err| %.2e over %lld, NaN %lld\n", (long long) s.d, (long long) s.hq, e.max_err, (long long) e.n_cmp, (long long) e.n_nan);
        CHECK(e.n_nan == 0 && e.max_err <= 2.0e-3);
    }
    if (time_it) {
        for (int trial = 0; trial < 3; ++trial) {
            result dn = run(backend, 128, 512, 4096, 8, 2, false, poison::none, 10);
            result sp = run(backend, 128, 512, 4096, 8, 2, true,  poison::none, 10);
            printf("  TIME d=128 nq=512 nkv=4096: dense %.3f ms  sparse %.3f ms  ratio %.3f\n", dn.ms, sp.ms, sp.ms / dn.ms);
        }
    }
    ggml_backend_free(backend);
    if (n_fail) { fprintf(stderr, "%d check(s) failed\n", n_fail); return 1; }
    printf("test-fa-xattn-skip: all cases passed\n");
    return 0;
}
