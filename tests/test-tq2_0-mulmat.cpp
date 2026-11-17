// TQ2_0 MUL_MAT / MUL_MAT_ID on the CPU backend: does the iqk GEMM agree with the vec_dot it replaced?
//
// WHY. Row S69-TQ2-0-CPU-PATH-5X-SLOWER-THAN-IQ2-BN-2026-10-16-01: the same BitNet-2B-4T ternary weights ran
// 121.7 s/pass as TQ2_0 against 24.0 s/pass as IQ2_BN. TQ2_0 had no iqk GEMM (every iqk entry declined it), so
// each output was one ggml_vec_dot_tq2_0_q8_K call -- and that vec_dot linked to the ggml-base copy, compiled
// without -mavx2, i.e. the scalar loop. The fix adds mul_mat_tq2_0_q8_K (iqk_gemm_1bit.cpp) and an AVX2
// vec_dot in the CPU backend (arch/x86/quants.c). This test pins both.
//
// THE ORACLES.
//   A. the type's own CPU vec_dot (ggml_get_type_traits_cpu(TQ2_0)->vec_dot) on the activation quantised by the
//      vec_dot_type's from_float -- i.e. exactly what the generic path computes. The GEMM reads the same Q8_K
//      bytes and does the same integer sums, so it must agree to float rounding (and is designed to be bitwise
//      equal; the count of bit-equal outputs is printed).
//   B. dequantise(W) . dequantise(q8_K(x)) in double precision -- independent of every kernel, so a defect shared
//      by the GEMM and the vec_dot cannot pass. Only float rounding of the kernel's accumulation is left, bounded
//      per output by B_TOL * sum_i |w_i x_i|. NEGATIVE control: the same measure against a reference built from
//      the activation shifted by one element must exceed B_CTL_MIN (rms), otherwise B could not have failed.
//   S. the ggml-base SCALAR loop, ggml_vec_dot_tq2_0_q8_K -- what every output was before the fix. The GEMM
//      and the CPU vec_dot must both equal it BIT FOR BIT on every output, so a model's perplexity cannot move.
//      (MEASURED: a first version with mainline's per-lane float accumulation passed A and B and still moved
//      BitNet-2B TQ2_0 wikitext-2 PPL, --chunks 10, from 20.1866 to 20.2233.)
//   C. (x86 + AVX2 + IQK static builds) the dispatch itself: iqk_gemm_type_b(TQ2_0) must name Q8_K. Before the
//      fix it returned -1, so this is the check that FAILS on the old tree; A and B passed there too, slowly.
//
// Shapes: K in {256, 2560} (1 and 10 blocks), rows M in {1, 7, 64} (odd counts reach the tail), columns N across
// 1..33 (the GEMM takes up to 8 columns per call, so 9, 16 and 33 cross that), 4 threads. MUL_MAT_ID: 4 experts,
// 2 used per token, 9 tokens, which goes through iqk_mul_mat_moe's row_mapping.
//
// Weights are exact ternary values times a per-256 scale that varies block to block (so the per-block d is
// exercised), one all-zero block (d = 0) and one Gaussian row (quantised to the nearest ternary point).

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifdef TEST_HAVE_IQK
extern "C" int iqk_gemm_type_b(int typeA, long ne00, int preferred);
#endif
// the ggml-base scalar loop (ggml-quants.c): what every TQ2_0 output was before the fix, so the
// GEMM and the CPU vec_dot must reproduce it BIT FOR BIT (a model's perplexity then cannot move)
extern "C" void ggml_vec_dot_tq2_0_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static const double A_TOL = 1e-5;   // relative to the largest |output| of the case
// B compares with dequant(W) . dequant(q8_K(x)) in double, so the only difference left is float rounding of the
// kernel's per-block accumulation. It is measured per output against sum_i |w_i x_i| (the rounding scale), not
// against |output|: a single output that cancels to near zero made a plain relative error meaningless (the first
// version of this test compared with the float x, and 3 of 49 one-to-three-output cases failed at rel_rms
// 0.03-0.14 while every output was bit-equal to the vec_dot -- a defect of the oracle, not of the kernel).
static const double B_TOL = 1e-5;
static const double B_CTL_MIN = 1e-3; // the shifted-x control must be at least this far off

// block_q8_K as ggml-common.h lays it out: float d, float sum, int8 qs[256], int16 bsums[16].
struct test_block_q8_K { float d; float sum; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(test_block_q8_K) == 296, "block_q8_K layout");

// the activation the kernels actually see: q8_K(x) through the vec_dot_type's from_float, dequantised in double
static void q8k_dequant(const float * x, int64_t K, std::vector<double> & out) {
    const auto * tr  = ggml_get_type_traits_cpu(GGML_TYPE_TQ2_0);
    const auto * tq8 = ggml_get_type_traits_cpu(tr->vec_dot_type);
    std::vector<test_block_q8_K> q(K/256);
    if ((size_t) ggml_row_size(tr->vec_dot_type, K) != q.size()*sizeof(test_block_q8_K)) { printf("q8_K row size mismatch\n"); exit(2); }
    tq8->from_float(x, q.data(), K);
    out.resize(K);
    for (int64_t b = 0; b < K/256; ++b) for (int j = 0; j < 256; ++j) out[b*256 + j] = (double) q[b].d * q[b].qs[j];
}

// max over outputs of |a - b| / scale
static double max_scaled(const std::vector<float> & a, const std::vector<double> & b, const std::vector<double> & scale) {
    double r = 0;
    for (size_t i = 0; i < a.size(); ++i) r = std::max(r, scale[i] > 0 ? std::fabs(a[i] - b[i])/scale[i] : std::fabs(a[i] - b[i]));
    return r;
}
// rms over outputs of |a - b| / scale (for the control: one shifted output can land near the true one by chance,
// so a min over 2000 outputs would be noise; the rms is not)
static double rms_scaled(const std::vector<float> & a, const std::vector<double> & b, const std::vector<double> & scale) {
    double r = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double e = scale[i] > 0 ? (a[i] - b[i])/scale[i] : a[i] - b[i]; r += e*e; }
    return a.empty() ? 0 : std::sqrt(r/a.size());
}

static void make_weights(std::mt19937 & rng, int64_t K, int64_t M, std::vector<float> & w) {
    std::uniform_int_distribution<int> tern(0, 9);
    std::uniform_real_distribution<float> sc(0.01f, 0.05f);
    std::normal_distribution<float> nd(0.0f, 0.03f);
    w.assign(K*M, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t b = 0; b < K/256; ++b) {
            const float s = sc(rng);
            const bool zero_block = (m == 0 && b == K/256 - 1 && K > 256);
            for (int64_t j = 0; j < 256; ++j) {
                float v;
                if (zero_block) v = 0.0f;
                else if (m == M - 1 && M > 1) v = nd(rng);                    // one Gaussian row
                else { const int t = tern(rng); v = t < 4 ? 0.0f : t < 7 ? s : -s; }
                w[m*K + b*256 + j] = v;
            }
        }
    }
}

// oracle A for one weight row block: out[n*M + m] = vec_dot(W row m, q8(x col n))
static void oracle_vecdot(const void * W, size_t row_w, int64_t K, int64_t M, const float * x, int64_t N, std::vector<float> & out, bool scalar = false) {
    const auto * tr  = ggml_get_type_traits_cpu(GGML_TYPE_TQ2_0);
    const auto * tq8 = ggml_get_type_traits_cpu(tr->vec_dot_type);
    std::vector<uint8_t> q(ggml_row_size(tr->vec_dot_type, K));
    out.assign(M*N, 0.0f);
    for (int64_t n = 0; n < N; ++n) {
        tq8->from_float(x + n*K, q.data(), K);
        for (int64_t m = 0; m < M; ++m) {
            float s = 0;
            (scalar ? ggml_vec_dot_tq2_0_q8_K : tr->vec_dot)((int) K, &s, 0, (const char *) W + m*row_w, 0, q.data(), 0, 1);
            out[n*M + m] = s;
        }
    }
}

struct cmp_result { double a_rel; int n_bitwise; double b_rel; double b_ctl; bool finite; int n_bit_s; int n_bit_as; };

static cmp_result compare(const std::vector<float> & y, const std::vector<float> & ya,
                          const std::vector<double> & yb, const std::vector<double> & ybad, const std::vector<double> & scale,
                          const std::vector<float> & ys) {
    cmp_result r{0, 0, 0, 0, true, 0, 0};
    double maxabs = 0, maxdiff = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!std::isfinite(y[i])) r.finite = false;
        maxabs  = std::max(maxabs, (double) std::fabs(ya[i]));
        maxdiff = std::max(maxdiff, (double) std::fabs(y[i] - ya[i]));
        if (memcmp(&y[i], &ya[i], sizeof(float)) == 0) ++r.n_bitwise;
        if (memcmp(&y[i], &ys[i], sizeof(float)) == 0) ++r.n_bit_s;
        if (memcmp(&ya[i], &ys[i], sizeof(float)) == 0) ++r.n_bit_as;
    }
    r.a_rel = maxabs > 0 ? maxdiff/maxabs : maxdiff;
    r.b_rel = max_scaled(y, yb, scale);
    r.b_ctl = rms_scaled(y, ybad, scale);
    return r;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_cpu_init();
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 4);
    std::mt19937 rng(20261016);
    std::normal_distribution<float> nx(0.0f, 1.0f);
    int failed = 0, tested = 0;

    const auto * tr = ggml_get_type_traits(GGML_TYPE_TQ2_0);

    // ---- C: dispatch
#ifdef TEST_HAVE_IQK
#if defined(__x86_64__) || defined(_M_X64)
    if (ggml_cpu_has_avx2()) {
        const int t = iqk_gemm_type_b(GGML_TYPE_TQ2_0, 2560, GGML_TYPE_Q8_K);
        const bool ok = t == GGML_TYPE_Q8_K;
        printf("dispatch  iqk_gemm_type_b(tq2_0, 2560, q8_K) = %d (%s)  %s\n", t, t >= 0 ? ggml_type_name((ggml_type) t) : "declined",
               ok ? "ok" : "FAIL: no iqk GEMM for TQ2_0, every output is one vec_dot");
        ++tested; if (!ok) ++failed;
    }
#endif
#else
    printf("dispatch  not checked (not an x86 static IQK build)\n");
#endif
    if (getenv("GGML_NO_IQK_GEMM")) printf("note      GGML_NO_IQK_GEMM is set: the graph runs the vec_dot, A compares it with itself\n");

    // ---- MUL_MAT
    const int64_t Ks[] = { 256, 2560 };
    const int64_t Ms[] = { 1, 7, 64 };
    const int64_t Ns[] = { 1, 2, 3, 5, 8, 9, 16, 33 };
    for (int64_t K : Ks) for (int64_t M : Ms) {
        std::vector<float> wf;
        make_weights(rng, K, M, wf);
        ggml_init_params ip = { (size_t) 256*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ2_0, K, M);
        const size_t row_w = ggml_row_size(GGML_TYPE_TQ2_0, K);
        ggml_quantize_chunk(GGML_TYPE_TQ2_0, wf.data(), W->data, 0, M, K, nullptr, nullptr);
        std::vector<float> wd(K*M);
        for (int64_t m = 0; m < M; ++m) tr->to_float((const char *) W->data + m*row_w, wd.data() + m*K, K);   // per row
        for (int64_t N : Ns) {
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            std::vector<float> xf(K*N);
            for (auto & v : xf) v = nx(rng);
            memcpy(X->data, xf.data(), xf.size()*sizeof(float));
            ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("MUL_MAT K=%lld M=%lld N=%lld compute FAILED\n", (long long) K, (long long) M, (long long) N); ++failed; continue; }
            std::vector<float> y((float *) Y->data, (float *) Y->data + M*N), ya;
            std::vector<float> ys;
            oracle_vecdot(W->data, row_w, K, M, xf.data(), N, ya);
            oracle_vecdot(W->data, row_w, K, M, xf.data(), N, ys, true);
            std::vector<double> yb(M*N), ybad(M*N), sc(M*N), xq;
            for (int64_t n = 0; n < N; ++n) {
                q8k_dequant(xf.data() + n*K, K, xq);
                for (int64_t m = 0; m < M; ++m) {
                    double s = 0, s2 = 0, a = 0;
                    for (int64_t i = 0; i < K; ++i) { s += wd[m*K + i]*xq[i]; s2 += wd[m*K + i]*xq[(i + 1) % K]; a += std::fabs(wd[m*K + i]*xq[i]); }
                    yb[n*M + m] = s; ybad[n*M + m] = s2; sc[n*M + m] = a;
                }
            }
            const cmp_result r = compare(y, ya, yb, ybad, sc, ys);
            const bool ok = r.finite && r.a_rel <= A_TOL && r.b_rel < B_TOL && r.b_ctl > B_CTL_MIN
                         && r.n_bit_s == (int) (M*N) && r.n_bit_as == (int) (M*N);
            printf("MUL_MAT    K=%5lld M=%3lld N=%3lld  vs vec_dot max_rel=%.2e bitwise=%d/%lld  vs scalar bitwise gemm=%d vec_dot=%d  vs double max_err/sum|wx|=%.2e negctl_rms=%.2e  %s\n",
                   (long long) K, (long long) M, (long long) N, r.a_rel, r.n_bitwise, (long long) (M*N), r.n_bit_s, r.n_bit_as, r.b_rel, r.b_ctl, ok ? "ok" : "FAIL");
            ++tested; if (!ok) ++failed;
        }
        ggml_free(ctx);
    }

    // ---- MUL_MAT_ID: 4 experts, 2 used per token, 9 tokens
    {
        const int64_t K = 512, M = 64, n_as = 4, n_used = 2, T = 9;
        ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * AS = ggml_new_tensor_3d(ctx, GGML_TYPE_TQ2_0, K, M, n_as);
        const size_t row_w = ggml_row_size(GGML_TYPE_TQ2_0, K);
        std::vector<std::vector<float>> wde(n_as);
        for (int64_t e = 0; e < n_as; ++e) {
            std::vector<float> wf;
            make_weights(rng, K, M, wf);
            ggml_quantize_chunk(GGML_TYPE_TQ2_0, wf.data(), (char *) AS->data + e*M*row_w, 0, M, K, nullptr, nullptr);
            wde[e].resize(K*M);
            for (int64_t m = 0; m < M; ++m) tr->to_float((const char *) AS->data + (e*M + m)*row_w, wde[e].data() + m*K, K);
        }
        ggml_tensor * B   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, T);
        ggml_tensor * IDS = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
        std::vector<float> bf(K*n_used*T);
        for (auto & v : bf) v = nx(rng);
        memcpy(B->data, bf.data(), bf.size()*sizeof(float));
        std::uniform_int_distribution<int> pick(0, (int) n_as - 1);
        for (int64_t t = 0; t < T; ++t) {
            const int e0 = pick(rng); int e1 = pick(rng); if (e1 == e0) e1 = (e0 + 1) % (int) n_as;
            ((int32_t *) IDS->data)[t*n_used + 0] = e0;
            ((int32_t *) IDS->data)[t*n_used + 1] = e1;
        }
        ggml_tensor * Y = ggml_mul_mat_id(ctx, AS, B, IDS);   // [M, n_used, T]
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, Y);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("MUL_MAT_ID compute FAILED\n"); ++failed; }
        else {
            std::vector<float> y((float *) Y->data, (float *) Y->data + M*n_used*T), ya(M*n_used*T);
            std::vector<double> yb(M*n_used*T), ybad(M*n_used*T), sc(M*n_used*T), xq;
            std::vector<float> ys(M*n_used*T), cols;
            for (int64_t t = 0; t < T; ++t) for (int64_t j = 0; j < n_used; ++j) {
                const int e = ((int32_t *) IDS->data)[t*n_used + j];
                const float * x = bf.data() + (t*n_used + j)*K;
                std::vector<float> col;
                oracle_vecdot((const char *) AS->data + e*M*row_w, row_w, K, M, x, 1, col);
                oracle_vecdot((const char *) AS->data + e*M*row_w, row_w, K, M, x, 1, cols, true);
                q8k_dequant(x, K, xq);
                for (int64_t m = 0; m < M; ++m) {
                    const int64_t o = (t*n_used + j)*M + m;
                    ya[o] = col[m];
                    ys[o] = cols[m];
                    double s = 0, s2 = 0, a = 0;
                    for (int64_t i = 0; i < K; ++i) { s += wde[e][m*K + i]*xq[i]; s2 += wde[e][m*K + i]*xq[(i + 1) % K]; a += std::fabs(wde[e][m*K + i]*xq[i]); }
                    yb[o] = s; ybad[o] = s2; sc[o] = a;
                }
            }
            const cmp_result r = compare(y, ya, yb, ybad, sc, ys);
            const bool ok = r.finite && r.a_rel <= A_TOL && r.b_rel < B_TOL && r.b_ctl > B_CTL_MIN
                         && r.n_bit_s == (int) (M*n_used*T) && r.n_bit_as == (int) (M*n_used*T);
            printf("MUL_MAT_ID K=%5lld M=%3lld experts=%lld used=%lld tokens=%lld  vs vec_dot max_rel=%.2e bitwise=%d/%lld  vs scalar bitwise gemm=%d vec_dot=%d  vs double max_err/sum|wx|=%.2e negctl_rms=%.2e  %s\n",
                   (long long) K, (long long) M, (long long) n_as, (long long) n_used, (long long) T, r.a_rel, r.n_bitwise,
                   (long long) (M*n_used*T), r.n_bit_s, r.n_bit_as, r.b_rel, r.b_ctl, ok ? "ok" : "FAIL");
            ++tested; if (!ok) ++failed;
        }
        ggml_free(ctx);
    }

    ggml_backend_free(be);
    printf("%d check(s), %d failure(s)\n", tested, failed);
    return failed ? 1 : 0;
}
