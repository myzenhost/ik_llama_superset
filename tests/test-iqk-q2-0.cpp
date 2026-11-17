// Does the iqk x86 GEMM compute mainline Q2_0 (GGML_TYPE_Q2_0 = 401: 64-element blocks, fp16 d, codes {0..3} -> {-1,0,1,2}*d)
// x Q8_2_X4 correctly? Port of ik_llama.cpp 422c259a2 (#2659) mul_mat_q2_0_q8_2_x4 (ik merge-owed item 5).
//
// GATE-CLAIM-OK: "CPU" names the backend source directory this test links; it computes on the host and touches no
// accelerator. No device is claimed unavailable.
//
// WHAT IT CHECKS.
//   1. iqk_gemm_type_b(Q2_0, ne00, vec_dot_type) names Q8_2_X4. Before the port iqk had no Q2_0 kernel and returned -1:
//      THIS LINE FAILS BEFORE AND PASSES AFTER.
//   2. iqk_mul_mat() accepts the pairing and matches a double-precision reference built from the type's own to_float
//      (dequantize_row_q2_0) and an exact decode of the Q8_2_X4 bytes. RANDOM weight bytes, so all four codes appear at
//      every position. Row lengths include an ODD number of 64-blocks >= 3 (K = 192, 320): ik's tail indexed
//      q2[2*ib] / yb[2*ib], which is only right for K = 64; the port reads q2[ib] / yb[0..1].
//   3. Graph level: ggml_mul_mat on a Q2_0 weight through ggml_graph_compute (the dense iqk_ok site in ggml-cpu.c),
//      1 and 4 threads, against the f32 activations.
// CONTROLS: a reference without the "-1" must disagree; the comparator must see a perturbation and a NaN.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);
int iqk_gemm_type_b(int typeA, long ne00, int preferred);
}

static int g_fail = 0;
static double g_min_signal = 1e30;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double rel_to_max(const std::vector<double> & ref, const std::vector<float> & got) {
    double worst = 0.0, scale = 1e-12;
    for (size_t i = 0; i < ref.size(); i++) {
        if (!std::isfinite((double) got[i])) return INFINITY;
        worst = std::max(worst, std::fabs((double) got[i] - ref[i]));
        scale = std::max(scale, std::fabs(ref[i]));
    }
    return worst / scale;
}

static ggml_fp16_t to_f16(float v) {
    ggml_fp16_t h;
    ggml_fp32_to_fp16_row(&v, &h, 1);
    return h;
}

static float bf16_bits_to_f32(uint16_t bits) {
    const uint32_t u = (uint32_t) bits << 16;
    float d; memcpy(&d, &u, 4);
    return d;
}

static void fill_q2_0_row(uint8_t * row, int64_t K, std::mt19937 & rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.005f, 0.08f);
    for (int64_t b = 0; b < K / 64; b++) {
        uint8_t * blk = row + (size_t) b * 18;
        const ggml_fp16_t d = to_f16(sc(rng));
        memcpy(blk, &d, 2);
        for (int i = 0; i < 16; i++) blk[2 + i] = (uint8_t) byte(rng);
    }
}

// Q8_2_X4 row decode, from the layout quantize_row_q8_1_x4_T writes (iqk_quantize.cpp quantize_row_q8_2_x4):
// K/128 groups of { uint16 d[8] (bf16 d[0..3], int16 sums[4..7]); int8 qs[128] }, then plain block_q8_2
// { uint16 d (bf16); int16 s; int8 qs[32] } for the remaining 32-blocks.
static void decode_q8_2_x4(const uint8_t * row, int64_t K, float * out) {
    const int64_t n4 = K / 128;
    for (int64_t g = 0; g < n4; g++) {
        const uint8_t * blk = row + (size_t) g * 144;
        for (int j = 0; j < 4; j++) {
            uint16_t bits; memcpy(&bits, blk + 2 * j, 2);
            const float d = bf16_bits_to_f32(bits);
            for (int i = 0; i < 32; i++) out[g * 128 + 32 * j + i] = d * (float)(int8_t) blk[16 + 32 * j + i];
        }
    }
    const uint8_t * tail = row + (size_t) n4 * 144;
    for (int64_t b = 0; b < (K - n4 * 128) / 32; b++) {
        const uint8_t * blk = tail + (size_t) b * 36;
        uint16_t bits; memcpy(&bits, blk, 2);
        const float d = bf16_bits_to_f32(bits);
        for (int i = 0; i < 32; i++) out[n4 * 128 + b * 32 + i] = d * (float)(int8_t) blk[4 + i];
    }
}

static double run_case(int64_t K, int64_t Nx, int64_t Ny, std::mt19937 & rng, bool drop_minus_one, bool * declined) {
    const ggml_type tA = GGML_TYPE_Q2_0;
    const ggml_type tB = GGML_TYPE_Q8_2_X4;
    const size_t rowA = ggml_row_size(tA, K);
    const size_t rowB = ggml_row_size(tB, K);

    std::vector<uint8_t> A((size_t) Nx * rowA);
    for (int64_t r = 0; r < Nx; r++) fill_q2_0_row(A.data() + (size_t) r * rowA, K, rng);
    std::vector<float> Wdeq((size_t) Nx * K);
    const ggml_type_traits * trA = ggml_get_type_traits(tA);
    for (int64_t r = 0; r < Nx; r++) trA->to_float(A.data() + (size_t) r * rowA, Wdeq.data() + (size_t) r * K, K);
    if (drop_minus_one) {
        for (int64_t r = 0; r < Nx; r++) {
            for (int64_t e = 0; e < K; e++) {
                const uint8_t * blk = A.data() + (size_t) r * rowA + (size_t) (e / 64) * 18;
                ggml_fp16_t dh; memcpy(&dh, blk, 2);
                float d; ggml_fp16_to_fp32_row(&dh, &d, 1);
                const int j = (int) (e % 64);
                Wdeq[(size_t) r * K + e] = d * (float) ((blk[2 + j / 4] >> (2 * (j % 4))) & 3);
            }
        }
    }

    std::vector<float> X((size_t) Ny * K);
    std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
    for (auto & v : X) v = ad(rng);
    const ggml_type_traits_cpu * cpuB = ggml_get_type_traits_cpu(tB);
    std::vector<uint8_t> B((size_t) Ny * rowB);
    for (int64_t c = 0; c < Ny; c++) cpuB->from_float(X.data() + (size_t) c * K, B.data() + (size_t) c * rowB, K);
    std::vector<float> Xref((size_t) Ny * K);
    for (int64_t c = 0; c < Ny; c++) decode_q8_2_x4(B.data() + (size_t) c * rowB, K, Xref.data() + (size_t) c * K);

    std::vector<double> ref((size_t) Ny * Nx);
    for (int64_t c = 0; c < Ny; c++) {
        for (int64_t r = 0; r < Nx; r++) {
            double s = 0.0;
            for (int64_t e = 0; e < K; e++) s += (double) Wdeq[(size_t) r * K + e] * (double) Xref[(size_t) c * K + e];
            ref[(size_t) c * Nx + r] = s;
        }
    }

    std::vector<float> C((size_t) Ny * Nx, -12345.0f);
    const bool ok = iqk_mul_mat(Nx, Ny, K, (int) tA, A.data(), (long) rowA, (int) tB, B.data(), (long) rowB,
                                C.data(), (long) Nx, 0, 1);
    if (!ok) { *declined = true; return -1.0; }
    *declined = false;
    const double err = rel_to_max(ref, C);
    double mr = 0.0, mg = 0.0;
    for (size_t i = 0; i < ref.size(); i++) { mr = std::max(mr, std::fabs(ref[i])); mg = std::max(mg, (double) std::fabs(C[i])); }
    if (!drop_minus_one) g_min_signal = std::min(g_min_signal, std::min(mr, mg));
    if (getenv("IQK_Q2_VERBOSE")) {
        printf("  K=%lld Nx=%lld Ny=%lld max|ref|=%.4g max|got|=%.4g err=%.3e\n", (long long) K, (long long) Nx, (long long) Ny, mr, mg, err);
    }
    if (!drop_minus_one) {
        CHECK(std::isfinite(err), "non-finite output K=%lld Nx=%lld Ny=%lld", (long long) K, (long long) Nx, (long long) Ny);
        CHECK(err <= 1e-4, "K=%lld Nx=%lld Ny=%lld rel err %.3e > 1e-4", (long long) K, (long long) Nx, (long long) Ny, err);
    }
    return err;
}

int main() {
    ggml_cpu_init();   // fp16 table + CPU traits
    std::mt19937 rng(20261017);

    const ggml_type_traits_cpu * cpuA = ggml_get_type_traits_cpu(GGML_TYPE_Q2_0);
    const int pref = cpuA ? (int) cpuA->vec_dot_type : -1;
    const int tb = iqk_gemm_type_b((int) GGML_TYPE_Q2_0, 256, pref);
    printf("iqk_gemm_type_b(Q2_0=%d, 256, vec_dot_type=%d) = %d (Q8_2_X4 = %d)\n", (int) GGML_TYPE_Q2_0, pref, tb, (int) GGML_TYPE_Q8_2_X4);
    CHECK(tb == (int) GGML_TYPE_Q8_2_X4, "iqk has no Q2_0 kernel (activation type %d, want %d)", tb, (int) GGML_TYPE_Q8_2_X4);
    if (tb != (int) GGML_TYPE_Q8_2_X4) {
        printf("FAILED: %d check(s) -- kernel path not reachable, numeric cases not run\n", g_fail);
        return 1;
    }
    // Q2_0_G128 (id 42) must keep its own pairing: the new case must not have changed what the G128 selector answers
    const int tb128 = iqk_gemm_type_b((int) GGML_TYPE_Q2_0_G128, 256, -1);
    CHECK(tb128 == (int) GGML_TYPE_Q8_2_X4, "Q2_0_G128 selector changed: %d", tb128);

    const int64_t Ks[]  = { 64, 128, 192, 256, 320, 640 };
    const int64_t Nxs[] = { 8, 5, 33 };
    const int64_t Nys[] = { 1, 2, 3, 4, 5, 7, 8, 9, 16, 17 };
    int n_cases = 0;
    double worst = 0.0;
    for (int64_t K : Ks) {
        for (int64_t Nx : Nxs) {
            for (int64_t Ny : Nys) {
                bool declined = false;
                const double e = run_case(K, Nx, Ny, rng, false, &declined);
                n_cases++;
                if (declined) { CHECK(false, "iqk declined K=%lld Nx=%lld Ny=%lld", (long long) K, (long long) Nx, (long long) Ny); continue; }
                worst = std::max(worst, e);
            }
        }
    }
    printf("%d cases, worst relative error %.3e, smallest signal magnitude %.3e\n", n_cases, worst, g_min_signal);
    CHECK(g_min_signal > 1e-2, "degenerate: an (almost) all-zero reference or result (min signal %.3e)", g_min_signal);

    // graph level through ggml-cpu.c
    {
        const int64_t shapes[][3] = { {256, 16, 1}, {192, 16, 7}, {320, 33, 5}, {128, 64, 17} };  // K, N, M
        const int nths[] = { 1, 4 };
        double gworst = 0.0;
        for (auto & sh : shapes) {
            for (int nth : nths) {
                const int64_t K = sh[0], N = sh[1], M = sh[2];
                std::vector<uint8_t> buf(64u * 1024 * 1024);
                ggml_init_params ip = { buf.size(), buf.data(), false };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0, K, N);
                ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
                const size_t rowA = ggml_row_size(GGML_TYPE_Q2_0, K);
                for (int64_t r = 0; r < N; r++) fill_q2_0_row((uint8_t *) W->data + (size_t) r * rowA, K, rng);
                std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
                for (int64_t i = 0; i < K * M; i++) ((float *) X->data)[i] = ad(rng);
                ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, Y);
                ggml_graph_compute_with_ctx(ctx, gf, nth);

                std::vector<float> Wd((size_t) N * K);
                const ggml_type_traits * trA = ggml_get_type_traits(GGML_TYPE_Q2_0);
                for (int64_t r = 0; r < N; r++) trA->to_float((uint8_t *) W->data + (size_t) r * rowA, Wd.data() + (size_t) r * K, K);
                std::vector<double> ref((size_t) M * N);
                for (int64_t c = 0; c < M; c++)
                    for (int64_t r = 0; r < N; r++) {
                        double s = 0.0;
                        for (int64_t e = 0; e < K; e++) s += (double) Wd[(size_t) r * K + e] * (double) ((float *) X->data)[(size_t) c * K + e];
                        ref[(size_t) c * N + r] = s;
                    }
                std::vector<float> got((float *) Y->data, (float *) Y->data + (size_t) M * N);
                const double e = rel_to_max(ref, got);
                CHECK(e <= 3e-2, "graph K=%lld N=%lld M=%lld nth=%d rel err %.3e > 3e-2", (long long) K, (long long) N, (long long) M, nth, e);
                gworst = std::max(gworst, e);
                ggml_free(ctx);
            }
        }
        printf("graph-level: worst relative error %.3e\n", gworst);
    }

    // controls
    {
        bool declined = false;
        const double e_wrong = run_case(256, 8, 4, rng, true, &declined);
        printf("control: reference without the -1 differs by %.3e (must be > 0.05)\n", e_wrong);
        CHECK(!declined && e_wrong > 0.05, "the check is NOT sensitive to the -1 correction (err %.3e)", e_wrong);
        const double e_wrong_tail = run_case(192, 8, 4, rng, true, &declined);
        printf("control: same on the odd-block tail (K=192) differs by %.3e (must be > 0.05)\n", e_wrong_tail);
        CHECK(!declined && e_wrong_tail > 0.05, "the tail check is NOT sensitive to the -1 correction (err %.3e)", e_wrong_tail);

        std::vector<double> r(64, 1.0);
        std::vector<float>  g(64, 1.0f);
        const double clean = rel_to_max(r, g);
        g[17] = 1.5f;
        const double dirty = rel_to_max(r, g);
        CHECK(clean == 0.0 && dirty > 0.4, "comparator cannot see a perturbation");
        g[17] = NAN;
        CHECK(!std::isfinite(rel_to_max(r, g)), "comparator passes a NaN");
    }

    if (g_fail) {
        printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    printf("OK: Q2_0 x Q8_2_X4 iqk kernel matches the reference (%d cases)\n", n_cases);
    return 0;
}
