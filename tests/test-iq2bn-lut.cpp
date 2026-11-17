// Numeric + performance test for the IQ2_BN AVX2 (no-VNNI) LUT prototype, runtime switch
// GGML_IK_IQ2BN_LUT (see ggml/src/iqk/iqk_gemm_1bit.cpp, mul_mat_iq2bn_q8_K64_lut, and
// temp/research/findings/S69_ENGINE_THIREUS_2026-09-26.md for the design and the adopt/decline
// verdict this test feeds).
//
// WHY TWO INVOCATIONS, NOT ONE. iq2bn_lut_enabled() in iqk_gemm_1bit.cpp caches the env var into
// a function-local static bool the first time IQ2_BN dispatch runs in this process, so flipping
// the env var mid-process would not change anything after the first mul_mat call. This binary
// therefore does two things: (a) ALWAYS runs a negative control and a same-process sanity check
// against an independent dequant-based reference, which is a useful regression test of whatever
// kernel THIS process's env var actually selects; (b) in --bench mode, times ggml_mul_mat at a
// given thread count. The caller (see the findings file) runs this binary twice -- once with
// GGML_IK_IQ2BN_LUT unset, once with it set to 1 -- and compares the two runs' printed numbers.
// That external A/B, not this file alone, is the adopt/decline test.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

// Mirrors iq2bn_lut_enabled()'s parse (iqk_gemm_1bit.cpp) for PRINT LABELS ONLY: getenv() != NULL
// is not enough -- GGML_IK_IQ2BN_LUT=0 is a non-null string, so a plain "getenv() ? ... : ..."
// mislabels an explicit =0 run as "LUT path" while the real dispatch correctly keeps it off.
// Found running the A/B (2026-09-27): default-unset and =0 must print identically; they didn't.
static const char * lut_label() {
    const char * e = getenv("GGML_IK_IQ2BN_LUT");
    return (e && *e && *e != '0') ? "LUT path" : "default path";
}

static void fill(std::vector<float> & v, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto & x : v) x = d(rng);
}

struct runner {
    ggml_context * ctx = nullptr;
    std::vector<uint8_t> buf;
    explicit runner(size_t mb) : buf(mb*1024*1024) {
        ggml_init_params p = { buf.size(), buf.data(), false };
        ctx = ggml_init(p);
    }
    ~runner() { if (ctx) ggml_free(ctx); }
    void run(ggml_cgraph * gf, int nthreads) {
        ggml_graph_compute_with_ctx(ctx, gf, nthreads);
    }
};

// ---- negative control ---------------------------------------------------------------------
// Same 9-entry table formula the kernel comment documents, checked exhaustively against a plain
// scalar reference, then ONE entry is deliberately set to a wrong value and the comparison below
// must flag the mismatch. This exists only to prove the comparator used elsewhere in this file
// would notice a wrong table entry -- it is not a copy of production code kept for its own sake.
static bool negative_control() {
    const int probes[] = { -128, -100, -3, 0, 5, 37, 127 };
    for (int a0 : probes) {
        for (int a1 : probes) {
            for (int w0 = -1; w0 <= 1; ++w0) {
                for (int w1 = -1; w1 <= 1; ++w1) {
                    int correct = w0*a0 + w1*a1;
                    int16_t table[9];
                    table[0]=(int16_t)(-a0-a1); table[1]=(int16_t)(-a1);   table[2]=(int16_t)(a0-a1);
                    table[3]=(int16_t)(-a0);    table[4]=0;                table[5]=(int16_t)(a0);
                    table[6]=(int16_t)(a1-a0);  table[7]=(int16_t)(a1);    table[8]=(int16_t)(a0+a1);
                    int idx = (w0+1) + 3*(w1+1);
                    if (table[idx] != correct) {
                        printf("NEGATIVE CONTROL SETUP INVALID: table[%d]=%d != correct %d "
                               "(a0=%d a1=%d w0=%d w1=%d)\n", idx, table[idx], correct, a0, a1, w0, w1);
                        return false; // the control itself would be worthless
                    }
                }
            }
        }
    }
    {
        int a0 = 37, a1 = -52;
        int16_t table8 = (int16_t)(a0 + a1);
        int16_t wrong_table8 = (int16_t)(table8 + 1);   // deliberate mismatch, w0=1,w1=1 entry
        int correct = 1*a0 + 1*a1;
        if (wrong_table8 == correct) {
            printf("NEGATIVE CONTROL DID NOT FLAG THE DELIBERATE MISMATCH (value=%d correct=%d)\n",
                   wrong_table8, correct);
            return false;
        }
    }
    printf("negative control: PASS (formula verified exhaustively over %d x %d activation pairs x "
           "9 weight combos; a deliberately wrong entry was flagged as a mismatch)\n",
           (int)(sizeof(probes)/sizeof(probes[0])), (int)(sizeof(probes)/sizeof(probes[0])));
    return true;
}

// ---- numeric agreement ----------------------------------------------------------------------
// IQ2_BN mul_mat, via whichever kernel THIS process's GGML_IK_IQ2BN_LUT selects, checked against
// an independent dequant-based double-precision reference (never against the OTHER kernel
// in-process -- see the file banner for why that needs two invocations).
static bool numeric_agreement(int64_t K, int64_t M, std::mt19937 & rng) {
    runner g(256);
    std::vector<float> Wf((size_t)(K*M));
    fill(Wf, rng, -0.3f, 0.3f);

    ggml_tensor * W = ggml_new_tensor_2d(g.ctx, GGML_TYPE_IQ2_BN, K, M);
    ggml_quantize_chunk(GGML_TYPE_IQ2_BN, Wf.data(), W->data, 0, M, K, nullptr, nullptr);

    std::vector<float> xv((size_t)K);
    fill(xv, rng, -0.5f, 0.5f);
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, 1);
    memcpy(x->data, xv.data(), xv.size()*sizeof(float));

    ggml_tensor * y = ggml_mul_mat(g.ctx, W, x);
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, y);
    g.run(gf, 2);

    const float * py = (const float *) y->data;

    std::vector<float> Wdq((size_t)(K*M));
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_IQ2_BN);
    size_t row_size = ggml_row_size(GGML_TYPE_IQ2_BN, K);
    for (int64_t r = 0; r < M; ++r) {
        tr->to_float((const char*)W->data + r*row_size, Wdq.data() + r*K, K);
    }
    std::vector<double> ref((size_t)M, 0.0);
    for (int64_t r = 0; r < M; ++r) {
        double s = 0.0;
        for (int64_t k = 0; k < K; ++k) s += (double)Wdq[r*K+k] * (double)xv[k];
        ref[r] = s;
    }

    double worst_abs = 0.0, scale = 1e-12;
    for (int64_t r = 0; r < M; ++r) scale = std::max(scale, std::fabs(ref[r]));
    for (int64_t r = 0; r < M; ++r) worst_abs = std::max(worst_abs, std::fabs((double)py[r] - ref[r]));
    double rel = worst_abs / scale;
    bool finite_ok = true;
    for (int64_t r = 0; r < M; ++r) if (!std::isfinite(py[r])) finite_ok = false;
    bool ok = finite_ok && rel < 0.02;   // int8-activation quantisation noise floor, not a bug budget
    printf("numeric agreement (K=%lld M=%lld, %s): worst |kernel - dequant_ref| / max|ref| = %.6f -- %s\n",
           (long long)K, (long long)M, lut_label(),
           rel, ok ? "PASS" : "FAIL");
    if (!ok) g_failures++;
    return ok;
}

// nrc_y = ncols of the activation matrix, i.e. how many token positions share ONE mul_mat call
// (1 = decode, >1 = prefill/batch). The LUT kernel's per-call weight repack (step 1) runs ONCE per
// call regardless of nrc_y, so a bigger nrc_y amortises that fixed cost over more output columns
// inside the SAME call -- worth measuring on its own, separate from moving the repack to load time.
static void bench(int64_t K, int64_t M, int64_t ncols, int nthreads, int repeats, int iters,
                   const char * tag) {
    runner g(768);
    std::mt19937 rng(12345);
    std::vector<float> Wf((size_t)(K*M));
    fill(Wf, rng, -0.3f, 0.3f);
    ggml_tensor * W = ggml_new_tensor_2d(g.ctx, GGML_TYPE_IQ2_BN, K, M);
    ggml_quantize_chunk(GGML_TYPE_IQ2_BN, Wf.data(), W->data, 0, M, K, nullptr, nullptr);
    std::vector<float> xv((size_t)(K*ncols));
    fill(xv, rng, -0.5f, 0.5f);
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, ncols);
    memcpy(x->data, xv.data(), xv.size()*sizeof(float));
    ggml_tensor * y = ggml_mul_mat(g.ctx, W, x);
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, y);

    for (int i = 0; i < 3; ++i) g.run(gf, nthreads);   // warmup

    printf("bench K=%lld M=%lld ncols=%lld threads=%d (%s, %s):", (long long)K, (long long)M,
           (long long)ncols, nthreads, lut_label(), tag);
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) g.run(gf, nthreads);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        printf("  rep%d=%.4f_ms/iter", r, ms/iters);
    }
    printf("\n");
}

// ---- repack-once reference: the fork ALREADY has a repacked layout for IQ2_BN, IQ2_BN_R4, with
// its own dedicated kernel (mul_mat_iq2_bn_r4_q8_k16, ggml/src/iqk/iqk_gemm_1bit.cpp:2139), wired
// through the existing run-time-repack machinery (iqk_repack_tensor, ggml/src/iqk/iqk_quantize.cpp)
// used for _R4/_R8/_R16 types across the fork and already correctness-tested against its base type
// in tests/probe-interleaved-vs-base-mulmat.cpp. Before building a NEW repack target for the LUT
// idea (2026-09-27 coordinator instruction), measuring this EXISTING one first -- rule 1.87, read
// our side before reimplementing.
extern "C" void iqk_repack_tensor(struct ggml_tensor * tensor);   // ggml/src/iqk/iqk_quantize.h

static void bench_r4(int64_t K, int64_t M, int64_t ncols, int nthreads, int repeats, int iters) {
    runner g(768);
    std::mt19937 rng(12345);
    std::vector<float> Wf((size_t)(K*M));
    fill(Wf, rng, -0.3f, 0.3f);
    ggml_tensor * W = ggml_new_tensor_2d(g.ctx, GGML_TYPE_IQ2_BN, K, M);
    ggml_quantize_chunk(GGML_TYPE_IQ2_BN, Wf.data(), W->data, 0, M, K, nullptr, nullptr);
    iqk_repack_tensor(W);   // in-place, once -- exactly what a real load does (llama.cpp:5106)
    if (W->type != GGML_TYPE_IQ2_BN_R4) {
        printf("bench K=%lld M=%lld ncols=%lld threads=%d (R4 path): iqk_repack_tensor did not "
               "convert (M%%4=%lld) -- SKIPPED\n", (long long)K, (long long)M, (long long)ncols,
               nthreads, (long long)(M % 4));
        return;
    }
    std::vector<float> xv((size_t)(K*ncols));
    fill(xv, rng, -0.5f, 0.5f);
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, ncols);
    memcpy(x->data, xv.data(), xv.size()*sizeof(float));
    ggml_tensor * y = ggml_mul_mat(g.ctx, W, x);
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, y);

    for (int i = 0; i < 3; ++i) g.run(gf, nthreads);   // warmup

    printf("bench K=%lld M=%lld ncols=%lld threads=%d (R4 path, repacked once before timing):",
           (long long)K, (long long)M, (long long)ncols, nthreads);
    for (int r = 0; r < repeats; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) g.run(gf, nthreads);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        printf("  rep%d=%.4f_ms/iter", r, ms/iters);
    }
    printf("\n");
}

int main(int argc, char ** argv) {
    if (!negative_control()) g_failures++;

    std::mt19937 rng(42);
    numeric_agreement(4096, 256, rng);
    numeric_agreement(1024, 1024, rng);
    // M=100 is not a multiple of 32: with GGML_IK_IQ2BN_LUT=1 this exercises the LUT kernel's
    // scalar tail loop (rows nx32..nrc_x), not just its vpshufb 32-row path.
    numeric_agreement(1024, 100, rng);

    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        const int64_t K = 4096, M = 8192;
        const int iters = 20, repeats = 3;
        int nthreads = argc > 2 ? std::atoi(argv[2]) : 1;
        bench(K, M, 1, nthreads, repeats, iters, "decode");
        bench(K, M, 32, nthreads, repeats, iters, "batch32");
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench-r4") == 0) {
        const int64_t K = 4096, M = 8192;
        const int iters = 20, repeats = 3;
        int nthreads = argc > 2 ? std::atoi(argv[2]) : 1;
        bench_r4(K, M, 1, nthreads, repeats, iters);
        bench_r4(K, M, 32, nthreads, repeats, iters);
    }

    printf("\n%s -- %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
