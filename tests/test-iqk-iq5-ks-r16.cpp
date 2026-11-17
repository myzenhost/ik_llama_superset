// IQ5_KS_R16 (ik 1b18524bd #2703, 44db69c5d #2731): 16-row interleave of IQ5_KS.
//
// test-quantize-fns SKIPS every row-interleaved type ("needs a multi-row harness"), so this is that harness:
//   1. REPACK: ggml_quantize_chunk(IQ5_KS_R16) of 16*R rows dequantizes BIT-EXACTLY to ggml_quantize_chunk(IQ5_KS) of the
//      same rows (both take quantize_iq5_ks_bulk's row search; R16 only re-lays the bytes) -- for n_per_row a multiple of
//      256 and, through the padded path, for n_per_row = 32 mod 256.
//   2. FROM_FLOAT_REF: the reference entry takes 16 rows of k/16 (fork fix: ik passes 4 rows and aborts on its own assert).
//   3. MATMUL: ggml_mul_mat(W[IQ5_KS_R16], X) on the CPU backend (iqk AVX2 mul_mat_iq5_ks_r16_q8_2, the Q8_K_R16 repack at
//      >= 32 columns on AVX512, or NEON) against a double reference built from 1's dequantized weights.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::vector<float> dequant(ggml_type t, const std::vector<uint8_t> & q, int64_t nrows, int64_t n) {
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    std::vector<float> y((size_t) (nrows * n));
    if (t == GGML_TYPE_IQ5_KS_R16) {
        // to_float of an R16 type decodes a whole 16-row group: k = 16 * n_per_row
        const size_t group = 16 * ggml_row_size(t, n);
        for (int64_t g = 0; g < nrows/16; ++g) tr->to_float(q.data() + g*group, y.data() + g*16*n, 16*n);
    } else {
        const size_t rs = ggml_row_size(t, n);
        for (int64_t r = 0; r < nrows; ++r) tr->to_float(q.data() + r*rs, y.data() + r*n, n);
    }
    return y;
}

static void test_repack(int64_t n, int64_t nrows, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) (n * nrows));
    for (auto & v : x) v = nd(rng);
    std::vector<uint8_t> q16(ggml_row_size(GGML_TYPE_IQ5_KS_R16, n) * nrows);
    const size_t w16 = ggml_quantize_chunk(GGML_TYPE_IQ5_KS_R16, x.data(), q16.data(), 0, nrows, n, nullptr);
    CHECK(w16 == q16.size(), "n=%lld: R16 quantize wrote %zu, want %zu", (long long) n, w16, q16.size());
    std::vector<float> y16 = dequant(GGML_TYPE_IQ5_KS_R16, q16, nrows, n);
    double err2 = 0, sig2 = 0;
    for (size_t i = 0; i < x.size(); ++i) { err2 += (x[i] - y16[i]) * (x[i] - y16[i]); sig2 += x[i] * x[i]; }
    const double rmse = std::sqrt(err2 / sig2);
    CHECK(rmse < 0.06, "n=%lld: R16 relative rmse %.4f (IQ5_KS is ~0.03)", (long long) n, rmse);
    if (n % 256 == 0) {
        std::vector<uint8_t> q1(ggml_row_size(GGML_TYPE_IQ5_KS, n) * nrows);
        ggml_quantize_chunk(GGML_TYPE_IQ5_KS, x.data(), q1.data(), 0, nrows, n, nullptr);
        std::vector<float> y1 = dequant(GGML_TYPE_IQ5_KS, q1, nrows, n);
        int64_t bad = 0;
        for (size_t i = 0; i < y1.size(); ++i) if (std::memcmp(&y1[i], &y16[i], 4) != 0) ++bad;
        CHECK(bad == 0, "n=%lld: %lld of %zu R16 weights differ from IQ5_KS of the same rows", (long long) n, (long long) bad, y1.size());
        printf("  repack n=%-5lld rows=%lld: rmse %.4f, %lld of %zu weights differ from IQ5_KS\n", (long long) n, (long long) nrows, rmse, (long long) bad, y1.size());
    } else {
        printf("  padded n=%-5lld rows=%lld: rmse %.4f\n", (long long) n, (long long) nrows, rmse);
    }
}

static void test_ref(std::mt19937 & rng) {
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_IQ5_KS_R16);
    CHECK(tr && tr->from_float_ref, "IQ5_KS_R16 has no from_float_ref");
    if (!tr || !tr->from_float_ref) return;
    const int64_t k = 16*512;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(k); for (auto & v : x) v = nd(rng);
    std::vector<uint8_t> q(16*ggml_row_size(GGML_TYPE_IQ5_KS_R16, k/16) + 64, 0xAB);
    tr->from_float_ref(x.data(), q.data(), k);   // aborted before the fork fix (4 rows -> GGML_ASSERT(nrows%16 == 0))
    std::vector<float> y(k);
    tr->to_float(q.data(), y.data(), k);
    double err2 = 0, sig2 = 0;
    for (int64_t i = 0; i < k; ++i) { err2 += (x[i] - y[i]) * (x[i] - y[i]); sig2 += x[i] * x[i]; }
    CHECK(std::sqrt(err2/sig2) < 0.06, "from_float_ref round trip rmse %.4f", std::sqrt(err2/sig2));
    printf("  from_float_ref 16 x %lld: rmse %.4f\n", (long long) (k/16), std::sqrt(err2/sig2));
}

static void test_matmul(std::mt19937 & rng) {
    const int64_t shapes[][3] = { {256, 16, 1}, {512, 32, 3}, {1024, 16, 8}, {256, 48, 16}, {512, 32, 33}, {2048, 16, 4} };  // K, N rows, M cols
    double worst = 0;
    for (auto & sh : shapes) {
        for (int nth : { 1, 4 }) {
            const int64_t K = sh[0], N = sh[1], M = sh[2];
            std::vector<uint8_t> buf(64u * 1024 * 1024);
            ggml_init_params ip = { buf.size(), buf.data(), false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_IQ5_KS_R16, K, N);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> wf((size_t) (K*N)); for (auto & v : wf) v = nd(rng);
            ggml_quantize_chunk(GGML_TYPE_IQ5_KS_R16, wf.data(), W->data, 0, N, K, nullptr);
            for (int64_t i = 0; i < K*M; ++i) ((float *) X->data)[i] = nd(rng);
            ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            std::vector<uint8_t> q((uint8_t *) W->data, (uint8_t *) W->data + ggml_nbytes(W));
            std::vector<float> Wd = dequant(GGML_TYPE_IQ5_KS_R16, q, N, K);
            double mx = 0, err = 0;
            for (int64_t c = 0; c < M; ++c)
                for (int64_t r = 0; r < N; ++r) {
                    double s = 0;
                    for (int64_t e = 0; e < K; ++e) s += (double) Wd[r*K + e] * ((float *) X->data)[c*K + e];
                    mx = std::max(mx, std::fabs(s));
                    err = std::max(err, std::fabs(s - ((float *) Y->data)[c*N + r]));
                }
            const double e = err / mx;
            CHECK(e <= 2e-2, "mul_mat K=%lld N=%lld M=%lld nth=%d rel err %.3e", (long long) K, (long long) N, (long long) M, nth, e);
            worst = std::max(worst, e);
            ggml_free(ctx);
        }
    }
    printf("  mul_mat: %zu shapes x 2 thread counts, worst relative-to-max error %.3e\n", sizeof(shapes)/sizeof(shapes[0]), worst);
}

int main() {
    ggml_cpu_init();
    std::mt19937 rng(20261020);
    printf("IQ5_KS_R16: repack vs IQ5_KS, from_float_ref, CPU mul_mat\n");
    CHECK(ggml_type_name(GGML_TYPE_IQ5_KS_R16) && std::strcmp(ggml_type_name(GGML_TYPE_IQ5_KS_R16), "iq5_ks_r16") == 0, "type name");
    test_repack(256, 16, rng);
    test_repack(1024, 32, rng);
    test_repack(4096, 16, rng);
    test_repack(288, 16, rng);    // 256 + 32: the padded path of quantize_iq5_ks_r16_impl
    test_ref(rng);
    test_matmul(rng);
    printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
