// CPU flash attention with an EXACT trailing-mask KV clamp (mask_effective_nk1 in ggml/src/iqk/fa/iqk_fa_templates.h).
// Row S69-EXECUTORCH-PICK-CPU-FA-EXACT-CAUSAL-KV-CLAMP-2026-10-05-01.   GATE-CLAIM-OK: documentation only.
//
// The skip was switched off upstream (ik_llama a38d29232, #1935 / #1910) because it sampled mask[ik] once per k_step and so missed the
// few visible entries of a sliding-window row ("S > 0" assert). The reproducer shape is what this test builds: rows whose ONLY visible
// entries are 1-3 isolated positions (not on a k_step grid), a causal prefill mask, a mask that is -inf only in the tail, a mask with
// FINITE very negative tail values (which are visible and must NOT be skipped), and rows that are masked out entirely.
// Every output element is compared with a double-precision softmax(QK^T*scale + mask)V computed here.  A fully masked row is only
// required not to abort.  With --time the causal prefill case is timed so a build before/after can be compared.
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
#include <functional>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

// visible(iq, ik) -> mask value as float (use -INFINITY for masked); NAN-free
using mask_fn = std::function<float(int64_t, int64_t)>;

static bool run_case(ggml_backend_t backend, const char * name, int64_t d, int64_t nq, int64_t nkv, int64_t hq, int64_t hkv,
        const mask_fn & mfn, bool time_it, double * ms_out) {
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
    if (!ggml_backend_supports_op(backend, out)) {
        printf("  %-34s SKIP (op unsupported for this shape)\n", name);
        ggml_free(ctx);
        return true;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::mt19937 rng(7u + (unsigned) (nkv * 31 + nq));
    std::normal_distribution<float> nd(0.0f, 0.5f);
    std::vector<float> qd(ggml_nelements(q)), kf(ggml_nelements(k)), vf(ggml_nelements(v));
    for (auto & x : qd) x = nd(rng);
    for (auto & x : kf) x = nd(rng);
    for (auto & x : vf) x = nd(rng);
    std::vector<ggml_fp16_t> kh(kf.size()), vh(vf.size()), mh((size_t) (nkv * nq_pad));
    ggml_fp32_to_fp16_row(kf.data(), kh.data(), kh.size());
    ggml_fp32_to_fp16_row(vf.data(), vh.data(), vh.size());
    std::vector<float> mf((size_t) (nkv * nq_pad));
    for (int64_t iq = 0; iq < nq_pad; ++iq) {
        for (int64_t ik = 0; ik < nkv; ++ik) {
            const float val = iq < nq ? mfn(iq, ik) : -INFINITY;
            mh[iq*nkv + ik] = ggml_fp32_to_fp16(val);
            mf[iq*nkv + ik] = ggml_fp16_to_fp32(mh[iq*nkv + ik]);
        }
    }
    // the K/V the kernel sees are the f16-rounded ones
    ggml_fp16_to_fp32_row(kh.data(), kf.data(), kf.size());
    ggml_fp16_to_fp32_row(vh.data(), vf.data(), vf.size());
    ggml_backend_tensor_set(q, qd.data(), 0, qd.size()*sizeof(float));
    ggml_backend_tensor_set(k, kh.data(), 0, kh.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(v, vh.data(), 0, vh.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(m, mh.data(), 0, mh.size()*sizeof(ggml_fp16_t));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, out);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL %s: graph compute failed\n", name);
        n_fail++;
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }
    if (time_it && ms_out) {
        const int reps = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(backend, gf);
        const auto t1 = std::chrono::steady_clock::now();
        *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    }
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));

    // out layout: [dv, nh, nq, 1]  -> res[(iq*hq + ih)*d + dd]
    double max_err = 0.0;
    int64_t n_cmp = 0, n_masked_rows = 0;
    const int64_t rep = hq / hkv;
    std::vector<double> sc((size_t) nkv);
    for (int64_t iq = 0; iq < nq; ++iq) {
        bool any = false;
        for (int64_t ik = 0; ik < nkv; ++ik) any |= std::isfinite(mf[iq*nkv + ik]);
        if (!any) { ++n_masked_rows; continue; }
        for (int64_t ih = 0; ih < hq; ++ih) {
            const int64_t ikv = ih / rep;
            double mx = -INFINITY;
            for (int64_t ik = 0; ik < nkv; ++ik) {
                double s = 0.0;
                for (int64_t dd = 0; dd < d; ++dd) s += (double) qd[((ih*nq + iq)*d) + dd] * (double) kf[((ikv*nkv + ik)*d) + dd];
                s = s * scale + (double) mf[iq*nkv + ik];
                sc[ik] = s;
                if (s > mx) mx = s;
            }
            double den = 0.0;
            for (int64_t ik = 0; ik < nkv; ++ik) { sc[ik] = std::isfinite(sc[ik]) ? std::exp(sc[ik] - mx) : 0.0; den += sc[ik]; }
            for (int64_t dd = 0; dd < d; ++dd) {
                double o = 0.0;
                for (int64_t ik = 0; ik < nkv; ++ik) if (sc[ik] != 0.0) o += sc[ik] * (double) vf[((ikv*nkv + ik)*d) + dd];
                o /= den;
                const double got = res[(iq*hq + ih)*d + dd];
                const double err = std::fabs(got - o);
                if (!(err <= max_err)) max_err = std::isnan(err) ? INFINITY : err;
                ++n_cmp;
            }
        }
    }
    const bool ok = max_err <= 2.0e-3;
    printf("  %-34s d=%-3lld nq=%-3lld nkv=%-5lld max|err| %.2e over %lld values (%lld masked rows) %s", name, (long long) d, (long long) nq,
           (long long) nkv, max_err, (long long) n_cmp, (long long) n_masked_rows, ok ? "ok" : "FAIL");
    if (time_it && ms_out) printf("   %.3f ms/graph", *ms_out);
    printf("\n");
    if (!ok) n_fail++;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    bool time_it = false, time_all = false;   // --time: the d=128 causal cases; --time-all: every case (A/B under IQK_FA_KSTEP_CAP=32|64|128)
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--time")) time_it = true;
        if (!strcmp(argv[i], "--time-all")) time_all = time_it = true;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "no CPU backend\n"); return 2; }
    ggml_backend_cpu_set_n_threads(backend, 4);

    for (int64_t d : { 64, 128, 256 }) {
        for (int64_t nkv : { 1024, 4096 }) {
            const int64_t nq = 64;
            const int64_t past = nkv - nq;
            char nm[64];

            snprintf(nm, sizeof nm, "causal prefill");
            double ms = 0.0;
            run_case(backend, nm, d, nq, nkv, 8, 2, [&](int64_t iq, int64_t ik) { return ik <= iq + past ? 0.0f : -INFINITY; }, (time_it && d == 128) || time_all, &ms);

            snprintf(nm, sizeof nm, "causal, n_past=0 (ubatch only)");
            run_case(backend, nm, d, nq, nkv, 8, 2, [&](int64_t iq, int64_t ik) { return ik <= iq ? 0.0f : -INFINITY; }, (time_it && d == 128) || time_all, &ms);

            // #1910 reproducer: sliding window of 1-3 entries, and positions that fall BETWEEN the k_step grid points
            snprintf(nm, sizeof nm, "SWA window 1..3 (reproducer)");
            run_case(backend, nm, d, nq, nkv, 8, 2, [&](int64_t iq, int64_t ik) {
                const int64_t pos = iq + past, w = 1 + iq % 3;
                return (ik <= pos && ik > pos - w) ? 0.0f : -INFINITY;
            }, false, nullptr);

            snprintf(nm, sizeof nm, "isolated visible entries");
            run_case(backend, nm, d, nq, nkv, 8, 2, [&](int64_t iq, int64_t ik) {
                const int64_t pos = iq + past;
                return (ik == pos - 37 || ik == 5 || ik == (pos * 7 + 3) % nkv || ik == pos - 61 * (iq % 4)) ? 0.0f : -INFINITY;
            }, false, nullptr);

            // finite very negative values are VISIBLE: they must never be treated as the -inf tail
            snprintf(nm, sizeof nm, "finite -30000 tail (not -inf)");
            run_case(backend, nm, d, nq, nkv, 8, 2, [&](int64_t iq, int64_t ik) { return ik <= iq + past ? 0.0f : -30000.0f; }, false, nullptr);

            // rows masked out entirely mixed with live rows: must not abort
            snprintf(nm, sizeof nm, "fully masked rows mixed in");
            run_case(backend, nm, d, nq, nkv, 8, 2, [&](int64_t iq, int64_t ik) {
                if (iq % 5 == 2) return -INFINITY;
                return ik <= iq + past ? 0.0f : -INFINITY;
            }, false, nullptr);

            // multi-slot style: neighbouring rows have different lengths (union of masks must be used)
            snprintf(nm, sizeof nm, "per-row different lengths");
            run_case(backend, nm, d, nq, nkv, 8, 2, [&](int64_t iq, int64_t ik) {
                const int64_t len = 8 + ((iq * 977) % (nkv - 8));
                return ik < len ? 0.0f : -INFINITY;
            }, false, nullptr);

            // single-token decode over a fully visible cache (the flash-decoding path): timed under --time-all
            snprintf(nm, sizeof nm, "decode nq=1, all visible");
            run_case(backend, nm, d, 1, nkv, 8, 2, [&](int64_t, int64_t) { return 0.0f; }, time_all, &ms);
        }
    }
    ggml_backend_free(backend);
    if (n_fail) { fprintf(stderr, "%d check(s) failed\n", n_fail); return 1; }
    printf("test-fa-causal-clamp: all cases passed\n");
    return 0;
}
