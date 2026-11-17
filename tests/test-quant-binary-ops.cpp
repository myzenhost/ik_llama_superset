// CPU ADD / ADD1 / OUT_PROD with a QUANTISED src0, checked numerically per type.
//
// WHY. The per-backend merge moved CPU compute to upstream's dispatchers in ggml-cpu/ops.cpp. For add,
// add1 and out_prod the fork's own (now dead) ggml.c routed its extra weight types -- the IQ*_K family,
// Q6_0, the BitNet types -- to add_q_f32 / add1_q_f32 / out_prod_q_f32, handlers that still exist in
// ops.cpp; the live switches list only upstream's types, so every fork type falls to
// `default: GGML_ABORT`. Measured 2026-09-26 by comparing the two files' case labels (acc, set and clamp
// ALSO lost labels, but ggml.c aborted on those types too, so they are not regressions).
//
// WHAT IS CHECKED, per type, against an f32 reference built from the SAME dequantised weights:
//   add      dst = src0 + b         (dst has src0's type: dequantise, add, requantise by the CPU
//                                    from_float -- so the expected result is DETERMINED, and the test
//                                    demands it exactly: deq(dst) == deq(from_float(deq(src0) + b)))
//   add1     dst = src0 + scalar    (same determinism)
//   out_prod dst = src0 x b^T       (f32 result; compared to a double-precision product of deq(src0))
// SIGNAL CHECK: the handler memcpy's raw floats into a quantised row when the CPU from_float is NULL
// (ops.cpp add_q_f32), which is silent garbage. So every type's traits are printed, a type without a
// base to_float or a CPU from_float is reported as NOT ROUTABLE rather than run, and a result must carry
// signal (rel error well below 1) as well as match.
// POSITIVE CONTROL: q4_0, which the live switches already route, must pass all three. If it does not, the
// test is wrong and no verdict below means anything.
//
// Usage: test-quant-binary-ops [type_name]   (one type only, e.g. to show the abort on an unrouted type)

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

static double rel_to_max(const float * a, const float * b, size_t n) {
    double worst = 0.0, scale = 1e-12;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        worst = std::max(worst, std::fabs((double) a[i] - (double) b[i]));
        scale = std::max(scale, std::fabs((double) b[i]));
    }
    return worst / scale;
}

// Relative RMS error: ~1.0 for an all-zero or uncorrelated result, well below 1 for a real one. This is the
// SIGNAL measure. A max-based measure is not: the coarse trellis types (iq1/2_kt at 1.75-2.1 bpw) requantise
// uniform data with a MAX error of 0.67-0.80 of the largest value while matching the defined result exactly
// (measured 2026-09-26) -- that is coarse, not broken, and a max-based bar would call it broken.
static double rel_rms(const float * a, const float * b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

struct ctx_t {
    ggml_context * ctx;
    ctx_t() { ggml_init_params p = { (size_t) 64 * 1024 * 1024, nullptr, false }; ctx = ggml_init(p); }
    ~ctx_t() { ggml_free(ctx); }
    void run(ggml_tensor * t) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
    }
};

// dequantise a row-independent tensor row by row
static std::vector<float> deq(const ggml_tensor * t) {
    const int64_t K = t->ne[0], M = t->ne[1];
    std::vector<float> out((size_t) (K * M));
    const auto * tr = ggml_get_type_traits(t->type);
    for (int64_t r = 0; r < M; r++) {
        tr->to_float((const char *) t->data + r * t->nb[1], out.data() + r * K, K);
    }
    return out;
}

// what the handler must produce: from_float of the f32 row, then dequantised
static std::vector<float> requant(ggml_type type, const std::vector<float> & x, int64_t K, int64_t M) {
    const auto * cpu = ggml_get_type_traits_cpu(type);
    const auto * tr = ggml_get_type_traits(type);
    const size_t rs = ggml_row_size(type, K);
    std::vector<char> q(rs * M);
    std::vector<float> out((size_t) (K * M));
    for (int64_t r = 0; r < M; r++) {
        cpu->from_float(x.data() + r * K, q.data() + r * rs, K);
        tr->to_float(q.data() + r * rs, out.data() + r * K, K);
    }
    return out;
}

// match  : max-relative difference from the DEFINED result (must be ~0 -- the handler is deterministic)
// signal : relative RMS error vs the exact f32 value (must be < 0.9: zeros or garbage sit at ~1.0)
static bool check(const char * what, ggml_type type, double match, double signal, double match_tol) {
    const bool ok = std::isfinite(match) && match <= match_tol && std::isfinite(signal) && signal < 0.9;
    printf("  %-9s %-9s match %.3e (tol %.0e)  rel-RMS vs exact %.3e  %s\n", ggml_type_name(type), what, match,
           match_tol, signal, ok ? "ok" : "FAIL");
    if (!ok) g_fail++;
    return ok;
}

static void test_type(ggml_type type, std::mt19937 & rng) {
    const auto * tr = ggml_get_type_traits(type);
    const auto * cpu = ggml_get_type_traits_cpu(type);
    const char * name = ggml_type_name(type);
    if (!tr || !name) {
        printf("  type#%d has no traits row -- NOT ROUTABLE\n", (int) type);
        return;
    }
    if (!tr->to_float || !cpu || !cpu->from_float) {
        printf("  %-9s NOT ROUTABLE: to_float %s, cpu from_float %s -- add_q_f32 would write raw floats\n", name,
               tr->to_float ? "yes" : "NULL", (cpu && cpu->from_float) ? "yes" : "NULL");
        return;
    }
    const int64_t K = 512, M = 64, P = 8;
    std::uniform_real_distribution<float> uw(-0.1f, 0.1f), ub(-0.05f, 0.05f);
    std::vector<float> Wf((size_t) (K * M)), bf((size_t) (K * M)), cf((size_t) (K * P));
    for (auto & v : Wf) v = uw(rng);
    for (auto & v : bf) v = ub(rng);
    for (auto & v : cf) v = ub(rng);
    std::vector<float> imat((size_t) K, 1.0f);

    // ---- add ----
    {
        ctx_t c;
        ggml_tensor * W = ggml_new_tensor_2d(c.ctx, type, K, M);
        ggml_quantize_init(type);
        ggml_quantize_chunk(type, Wf.data(), W->data, 0, M, K, imat.data(), nullptr);
        ggml_tensor * b = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, K, M);
        memcpy(b->data, bf.data(), bf.size() * sizeof(float));
        ggml_tensor * y = ggml_add(c.ctx, W, b);
        c.run(y);
        std::vector<float> w = deq(W), exact(w.size());
        for (size_t i = 0; i < w.size(); i++) exact[i] = w[i] + bf[i];
        const std::vector<float> got = deq(y), want = requant(type, exact, K, M);
        check("add", type, rel_to_max(got.data(), want.data(), got.size()), rel_rms(got.data(), exact.data(), got.size()), 1e-6);
    }
    // ---- add1 ----
    {
        ctx_t c;
        ggml_tensor * W = ggml_new_tensor_2d(c.ctx, type, K, M);
        ggml_quantize_chunk(type, Wf.data(), W->data, 0, M, K, imat.data(), nullptr);
        ggml_tensor * s = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 1);
        ((float *) s->data)[0] = 0.03125f;
        ggml_tensor * y = ggml_add1(c.ctx, W, s);
        c.run(y);
        std::vector<float> w = deq(W), exact(w.size());
        for (size_t i = 0; i < w.size(); i++) exact[i] = w[i] + 0.03125f;
        const std::vector<float> got = deq(y), want = requant(type, exact, K, M);
        check("add1", type, rel_to_max(got.data(), want.data(), got.size()), rel_rms(got.data(), exact.data(), got.size()), 1e-6);
    }
    // ---- out_prod: a = src0 [K x M] quantised, b = [P x M] f32 -> [K x P] ----
    {
        ctx_t c;
        ggml_tensor * A = ggml_new_tensor_2d(c.ctx, type, K, M);
        ggml_quantize_chunk(type, Wf.data(), A->data, 0, M, K, imat.data(), nullptr);
        std::vector<float> bm((size_t) (P * M));
        for (size_t i = 0; i < bm.size(); i++) bm[i] = cf[i % cf.size()];
        ggml_tensor * B = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, P, M);
        memcpy(B->data, bm.data(), bm.size() * sizeof(float));
        ggml_tensor * y = ggml_out_prod(c.ctx, A, B);
        c.run(y);
        const std::vector<float> a = deq(A);
        std::vector<float> want((size_t) (K * P));
        for (int64_t p = 0; p < P; p++)
            for (int64_t k = 0; k < K; k++) {
                double s = 0.0;
                for (int64_t m = 0; m < M; m++) s += (double) a[(size_t) (m * K + k)] * bm[(size_t) (m * P + p)];
                want[(size_t) (p * K + k)] = (float) s;
            }
        const double d = rel_to_max((const float *) y->data, want.data(), want.size());
        // f32 result vs a double product of the SAME dequantised weights: only summation order differs
        check("out_prod", type, d, rel_rms((const float *) y->data, want.data(), want.size()), 1e-4);
    }
    // ---- SIGNAL-CHECK CONTROL (once, on the positive-control type): an all-zero result MUST fail. ----
    if (type == GGML_TYPE_Q4_0) {
        std::vector<float> zeros((size_t) (K * M), 0.0f), exact((size_t) (K * M));
        for (size_t i = 0; i < exact.size(); i++) exact[i] = Wf[i] + bf[i];
        const double z = rel_rms(zeros.data(), exact.data(), exact.size());
        const bool caught = !(z < 0.9);
        printf("  %-4s SIGNAL CONTROL: an all-zero result scores rel-RMS %.3f (must be >= 0.9)\n", caught ? "ok" : "FAIL", z);
        if (!caught) g_fail++;
    }
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    ggml_cpu_init();
    std::mt19937 rng(20260926);

    // positive control first, then the fork's row-independent weight types the live switches do not route
    const ggml_type types[] = {
        GGML_TYPE_Q4_0,
        GGML_TYPE_IQ2_K, GGML_TYPE_IQ2_KS, GGML_TYPE_IQ2_KL, GGML_TYPE_IQ3_K, GGML_TYPE_IQ3_KS,
        GGML_TYPE_IQ4_K, GGML_TYPE_IQ4_KS, GGML_TYPE_IQ4_KSS, GGML_TYPE_IQ5_K, GGML_TYPE_IQ5_KS, GGML_TYPE_IQ6_K,
        GGML_TYPE_IQ1_KT, GGML_TYPE_IQ2_KT, GGML_TYPE_IQ3_KT, GGML_TYPE_IQ4_KT,
        GGML_TYPE_Q6_0, GGML_TYPE_IQ1_BN, GGML_TYPE_IQ2_BN,
    };
    int ran = 0;
    for (ggml_type t : types) {
        const char * n = ggml_type_name(t);
        if (argc > 1 && (!n || std::string(argv[1]) != n)) continue;
        const int before = g_fail;
        test_type(t, rng);
        ran++;
        if (t == GGML_TYPE_Q4_0 && g_fail != before) {
            printf("POSITIVE CONTROL q4_0 FAILED -- the test itself is wrong; stopping\n");
            return 1;
        }
    }
    if (ran == 0) {
        printf("no type matched -- nothing was tested\n");
        return 1;
    }
    printf("%s -- %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
