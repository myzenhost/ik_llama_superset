// Does the iqk x86 GEMM compute Q2_0_G128 (Bonsai 2-bit ternary) x Q8_2_X4 correctly?
//
// GATE-CLAIM-OK: "CPU" names the backend source directory this test links; it computes on the host
// and touches no accelerator. No device is claimed unavailable.
//
// WHY THIS EXISTS. Q2_0_G128 had no iqk kernel in this fork, so its dense MUL_MAT was excluded from
// iqk at ggml-cpu.c and ran the scalar vec_dot_q2_0_g128_q8_0 (~1.8 t/s in the authoring tree against
// ~6 t/s for the 1-bit sibling's AVX2 kernel). iqk_gemm_1bit.cpp now carries a kernel for the type;
// this test is what lets that kernel exist without trusting it.
//
// WHAT IT CHECKS.
//   1. iqk_gemm_type_b(Q2_0_G128, ne00, -1) names Q8_2_X4. Without the kernel it returns -1, so THIS
//      LINE FAILS BEFORE THE PORT and passes after; it is the discriminating assertion.
//   2. iqk_mul_mat() ACCEPTS the pairing (returns true) and its answer matches an independent
//      reference: the weights decoded by the type's own to_float (dequantize_row_q2_0_g128) and the
//      activations decoded by Q8_2_X4's to_float, multiplied out in double precision. The weights are
//      RANDOM BYTES, not quantiser output, so every one of the four 2-bit codes (including the
//      unused code 3 = +2) lands at every byte position and every shift -- an unpack that is only
//      right for ternary data would not survive that.
//   3. Across Ny = 1..17 columns (the template tiers), several row counts (including a count that is
//      not a multiple of the kernel tile) and several row lengths.
//
// CONTROLS (a verifier that cannot fail is not an oracle):
//   * NEGATIVE, comparator: a deliberately perturbed result must be reported.
//   * NEGATIVE, reference: a reference that omits the "-1" (decodes the raw code) must DISAGREE with
//     the kernel by far more than the tolerance -- proving the check is sensitive to the one
//     arithmetic detail (the sy correction) that a copy of the 1-bit kernel would get wrong.
//   * non-finite outputs fail the case on their own.

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

// The two iqk entry points this test drives. Declared here (not via iqk_mul_mat.h, which lives under
// ggml/src and is not on the test include path); both are extern "C" in iqk_mul_mat.cpp.
extern "C" {
bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);
int iqk_gemm_type_b(int typeA, long ne00, int preferred);
}

static int g_fail = 0;
static double g_min_signal = 1e30;   // smallest max|ref| / max|got| over all cases (must be well above 0)
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

// max |a-b| over max |ref|, the right normalisation for whole vectors (see test-mulmat-gemv.cpp).
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

// One case: Nx weight rows of length K, Ny activation columns. Returns the worst relative error, or
// -1 if iqk declined the pairing.
static double run_case(int64_t K, int64_t Nx, int64_t Ny, std::mt19937 & rng, bool drop_minus_one, bool * declined) {
    const ggml_type tA = GGML_TYPE_Q2_0_G128;
    const ggml_type tB = GGML_TYPE_Q8_2_X4;
    const size_t rowA = ggml_row_size(tA, K);
    const size_t rowB = ggml_row_size(tB, K);

    // ---- weights: RANDOM BYTES with a sane fp16 scale per 128-group (34 bytes: d, qs[32]) ----------
    std::vector<uint8_t> A((size_t) Nx * rowA);
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.005f, 0.08f);
    for (int64_t r = 0; r < Nx; r++) {
        for (int64_t g = 0; g < K / 128; g++) {
            uint8_t * blk = A.data() + (size_t) r * rowA + (size_t) g * 34;
            const ggml_fp16_t d = to_f16(sc(rng));
            memcpy(blk, &d, 2);
            for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t) byte(rng);
        }
    }
    std::vector<float> Wdeq((size_t) Nx * K);
    const ggml_type_traits * trA = ggml_get_type_traits(tA);
    for (int64_t r = 0; r < Nx; r++) {
        trA->to_float(A.data() + (size_t) r * rowA, Wdeq.data() + (size_t) r * K, K);
    }
    if (drop_minus_one) {
        // the reference variant for the sensitivity control: raw code instead of code-1, i.e. the
        // decoded value plus d, per element. Reconstruct from the bytes rather than from Wdeq.
        for (int64_t r = 0; r < Nx; r++) {
            for (int64_t e = 0; e < K; e++) {
                const uint8_t * blk = A.data() + (size_t) r * rowA + (size_t) (e / 128) * 34;
                ggml_fp16_t dh; memcpy(&dh, blk, 2);
                float d; ggml_fp16_to_fp32_row(&dh, &d, 1);
                const int j = (int) (e % 128);
                const int code = (blk[2 + j / 4] >> (2 * (j % 4))) & 3;
                Wdeq[(size_t) r * K + e] = d * (float) code;
            }
        }
    }

    // ---- activations: f32 -> Q8_2_X4 through the CPU traits, the same call ggml-cpu.c makes --------
    std::vector<float> X((size_t) Ny * K);
    std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
    for (auto & v : X) v = ad(rng);
    const ggml_type_traits_cpu * cpuB = ggml_get_type_traits_cpu(tB);
    const ggml_type_traits    * trB  = ggml_get_type_traits(tB);
    std::vector<uint8_t> B((size_t) Ny * rowB);
    for (int64_t c = 0; c < Ny; c++) {
        cpuB->from_float(X.data() + (size_t) c * K, B.data() + (size_t) c * rowB, K);
    }
    // Reference activations: Q8_2_X4's own decoder when it has one (tight comparison, only float
    // accumulation order differs); otherwise the f32 originals (q8 rounding error, looser tolerance).
    std::vector<float> Xref((size_t) Ny * K);
    const bool have_b_dec = trB && trB->to_float != nullptr;
    if (have_b_dec) {
        for (int64_t c = 0; c < Ny; c++) {
            trB->to_float(B.data() + (size_t) c * rowB, Xref.data() + (size_t) c * K, K);
        }
    } else {
        // Q8_2_X4 has no to_float. Decode it here from the layout quantize_row_q8_1_x4_T writes
        // (iqk_quantize.cpp:855-915): per 128 elements a block_q8_2_x4 = { uint16 d[8]; int8 qs[128] };
        // d[0..3] are the bf16 scales (already rounded before quantising), qs[32*j+i] sequential.
        // Exact for K % 128 == 0, the only K this test uses.
        for (int64_t c = 0; c < Ny; c++) {
            const uint8_t * row = B.data() + (size_t) c * rowB;
            for (int64_t b = 0; b < K / 128; b++) {
                const uint8_t * blk = row + (size_t) b * 144;
                for (int j = 0; j < 4; j++) {
                    uint16_t bits; memcpy(&bits, blk + 2 * j, 2);
                    const uint32_t u = (uint32_t) bits << 16;
                    float d; memcpy(&d, &u, 4);
                    for (int i = 0; i < 32; i++) {
                        Xref[(size_t) c * K + b * 128 + 32 * j + i] = d * (float)(int8_t) blk[16 + 32 * j + i];
                    }
                }
            }
        }
    }

    if (getenv("IQK_Q2_VERBOSE")) {
        double mw = 0.0, mx = 0.0; int nb = 0;
        for (float v : Wdeq) mw = std::max(mw, (double) std::fabs(v));
        for (float v : Xref) mx = std::max(mx, (double) std::fabs(v));
        for (uint8_t v : B) nb += v != 0;
        printf("  dbg: max|Wdeq|=%.4g max|Xref|=%.4g nonzero B bytes=%d/%zu have_b_dec=%d\n", mw, mx, nb, B.size(), (int) have_b_dec);
    }
    // ---- reference in double ---------------------------------------------------------------------
    std::vector<double> ref((size_t) Ny * Nx);
    for (int64_t c = 0; c < Ny; c++) {
        for (int64_t r = 0; r < Nx; r++) {
            double s = 0.0;
            for (int64_t e = 0; e < K; e++) s += (double) Wdeq[(size_t) r * K + e] * (double) Xref[(size_t) c * K + e];
            ref[(size_t) c * Nx + r] = s;
        }
    }

    // ---- kernel ------------------------------------------------------------------------------------
    std::vector<float> C((size_t) Ny * Nx, -12345.0f);   // sentinel: an untouched cell stays visible
    const bool ok = iqk_mul_mat(Nx, Ny, K, (int) tA, A.data(), (long) rowA, (int) tB, B.data(), (long) rowB,
                                C.data(), (long) Nx, 0, 1);
    if (!ok) {
        *declined = true;
        return -1.0;
    }
    *declined = false;
    const double err = rel_to_max(ref, C);
    {   // guard against a degenerate pass: an all-zero reference matched by an all-zero result
        double mr = 0.0, mg = 0.0;
        for (size_t i = 0; i < ref.size(); i++) { mr = std::max(mr, std::fabs(ref[i])); mg = std::max(mg, (double) std::fabs(C[i])); }
        g_min_signal = std::min(g_min_signal, std::min(mr, mg));
        if (getenv("IQK_Q2_VERBOSE")) {
            printf("  K=%lld Nx=%lld Ny=%lld max|ref|=%.4g max|got|=%.4g err=%.3e\n", (long long) K, (long long) Nx, (long long) Ny, mr, mg, err);
        }
    }
    // tolerance: tight when the reference decodes the same Q8_2_X4 bytes, looser when it cannot.
    // both branches now compare against the exact decode of the same Q8_2_X4 bytes the kernel reads,
    // so only float accumulation order differs.
    const double tol = 1e-4;
    if (!drop_minus_one) {
        CHECK(std::isfinite(err), "non-finite output K=%lld Nx=%lld Ny=%lld", (long long) K, (long long) Nx, (long long) Ny);
        CHECK(err <= tol, "K=%lld Nx=%lld Ny=%lld rel err %.3e > tol %.1e", (long long) K, (long long) Nx, (long long) Ny, err, tol);
    }
    return err;
}

int main() {
    // GGML_FP16_TO_FP32 is a table lookup on this build; the table is filled by ggml_cpu_init() (via
    // ggml_init in a real run). Without it every fp16 scale decodes to 0 and the whole comparison
    // degenerates to 0 == 0 -- the g_min_signal guard below exists because that happened here once.
    ggml_cpu_init();
    std::mt19937 rng(20261016);

    // ---- 1. the discriminating assertion ------------------------------------------------------------
    const int tb = iqk_gemm_type_b((int) GGML_TYPE_Q2_0_G128, 256, -1);
    printf("iqk_gemm_type_b(Q2_0_G128, 256, -1) = %d (Q8_2_X4 = %d)\n", tb, (int) GGML_TYPE_Q8_2_X4);
    CHECK(tb == (int) GGML_TYPE_Q8_2_X4, "iqk has no Q2_0_G128 kernel (activation type %d, want %d)", tb, (int) GGML_TYPE_Q8_2_X4);
    const ggml_type_traits_cpu * cpuB = ggml_get_type_traits_cpu(GGML_TYPE_Q8_2_X4);
    CHECK(cpuB && cpuB->from_float, "no CPU from_float for Q8_2_X4: the activation cannot be produced");
    if (!(cpuB && cpuB->from_float) || tb != (int) GGML_TYPE_Q8_2_X4) {
        printf("FAILED: %d check(s) -- kernel path not reachable, numeric cases not run\n", g_fail);
        return 1;
    }
    // a row length that is not a whole number of 128-groups must be refused, not read past the end
    CHECK(iqk_gemm_type_b((int) GGML_TYPE_Q2_0_G128, 96, -1) == -1, "ne00=96 (not a multiple of 128) was accepted");

    // ---- 2. numeric cases -----------------------------------------------------------------------------
    const int64_t Ks[]  = { 128, 256, 640 };
    const int64_t Nxs[] = { 8, 5, 33, 64 };
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
    CHECK(g_min_signal > 1e-2, "degenerate: a case had an (almost) all-zero reference or result (min signal %.3e)", g_min_signal);

    // ---- 2b. graph level: ggml_mul_mat on a Q2_0_G128 weight through ggml_graph_compute --------------------
    // This is the dispatch in ggml-cpu.c (the dense iqk_ok site), not iqk_mul_mat called directly. It
    // does not discriminate before/after by itself (the scalar vec_dot path answers correctly too);
    // it pins that lifting the exclusion produced no garbage, single- and multi-threaded, for a
    // matvec (M=1) and batched shapes. Tolerance: q8 rounding of the activations vs the f32 originals.
    {
        const int64_t shapes[][3] = { {256, 16, 1}, {256, 16, 7}, {640, 33, 5}, {128, 64, 17} };  // K, N, M
        const int nths[] = { 1, 4 };
        double gworst = 0.0;
        for (auto & sh : shapes) {
            for (int nth : nths) {
                const int64_t K = sh[0], N = sh[1], M = sh[2];
                std::vector<uint8_t> buf(64u * 1024 * 1024);
                ggml_init_params ip = { buf.size(), buf.data(), false };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0_G128, K, N);
                ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
                const size_t rowA = ggml_row_size(GGML_TYPE_Q2_0_G128, K);
                std::uniform_int_distribution<int> byte(0, 255);
                std::uniform_real_distribution<float> sc(0.005f, 0.08f), ad(-0.5f, 0.5f);
                for (int64_t r = 0; r < N; r++) {
                    for (int64_t g = 0; g < K / 128; g++) {
                        uint8_t * blk = (uint8_t *) W->data + (size_t) r * rowA + (size_t) g * 34;
                        const ggml_fp16_t d = to_f16(sc(rng));
                        memcpy(blk, &d, 2);
                        for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t) byte(rng);
                    }
                }
                for (int64_t i = 0; i < K * M; i++) ((float *) X->data)[i] = ad(rng);
                ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, Y);
                ggml_graph_compute_with_ctx(ctx, gf, nth);

                std::vector<float> Wd((size_t) N * K);
                const ggml_type_traits * trA = ggml_get_type_traits(GGML_TYPE_Q2_0_G128);
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
                double mr = 0.0;
                for (double v : ref) mr = std::max(mr, std::fabs(v));
                CHECK(mr > 1e-2, "graph case K=%lld N=%lld M=%lld has an (almost) all-zero reference", (long long) K, (long long) N, (long long) M);
                CHECK(e <= 3e-2, "graph K=%lld N=%lld M=%lld nth=%d rel err %.3e > 3e-2", (long long) K, (long long) N, (long long) M, nth, e);
                gworst = std::max(gworst, e);
                ggml_free(ctx);
            }
        }
        printf("graph-level: %zu shapes x %zu thread counts, worst relative error %.3e\n", sizeof(shapes) / sizeof(shapes[0]), sizeof(nths) / sizeof(nths[0]), gworst);
    }

    // ---- 3. controls ----------------------------------------------------------------------------------
    {
        // reference variant without the "-1": must disagree with the kernel by far more than tol
        bool declined = false;
        const double e_wrong = run_case(256, 8, 4, rng, true, &declined);
        printf("control: reference without the -1 differs by %.3e (must be > 0.05)\n", e_wrong);
        CHECK(!declined && e_wrong > 0.05, "the check is NOT sensitive to the -1 correction (err %.3e)", e_wrong);

        // comparator: a perturbed result must be reported
        std::vector<double> r(64, 1.0);
        std::vector<float>  g(64, 1.0f);
        const double clean = rel_to_max(r, g);
        g[17] = 1.5f;
        const double dirty = rel_to_max(r, g);
        printf("control: comparator clean %.3e, perturbed %.3e\n", clean, dirty);
        CHECK(clean == 0.0 && dirty > 0.4, "comparator cannot see a perturbation");
        g[17] = NAN;
        CHECK(!std::isfinite(rel_to_max(r, g)), "comparator passes a NaN");
    }

    if (g_fail) {
        printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    printf("OK: Q2_0_G128 x Q8_2_X4 iqk kernel matches the reference (%d cases)\n", n_cases);
    return 0;
}
