// Do the SCALAR vec_dots of the Bonsai types (Q1_0_G128, Q2_0_G128) agree with the activation layout
// the CPU backend feeds them?  (dedup check for orphan commits d933557be / 942d9c965 / d3eea50ef)
//
// GATE-CLAIM-OK: "CPU" names the backend source directory this test links; it computes on the host and
// touches no accelerator. No device is claimed unavailable.
//
// WHY THIS EXISTS. The orphan tree declared vec_dot_type Q8_2_X4 for these types while the scalar
// vec_dot cast its activation pointer to block_q8_0_x4 (an 8-byte-smaller header), so a run forced onto
// the scalar path read qs short and misread every scale (GGML_ASSERT(S > 0) in the FA kernels). The
// orphan fixed it by changing the CAST to block_q8_2_x4 and the declaration to follow.
// This tree is different: the CPU traits row (ggml-cpu.c type_traits_cpu) is what the generic mul_mat
// reads, and it declares Q8_0_X4 for both types, matching the cast; iqk reads its own activation type
// through ggml_cpu_iqk_act_type (Q8_2_X4), independent of vec_dot_type. So the defect should not exist
// here, and the orphan's cast change should BREAK this tree. This test measures both halves.
//
// WHAT IT CHECKS (per type)
//   1. the CPU traits vec_dot_type of the type, printed; the scalar vec_dot, fed activations quantized
//      to THAT type through the CPU from_float, matches a double-precision reference built from the
//      weights' own dequantiser and the decoded activation bytes (Q8_0_X4 layout read here:
//      { ggml_half d[4]; int8 qs[128] }, scale k covers qs[32k .. 32k+31]).
//   2. CONTROL: the same scalar vec_dot fed the OTHER grouped layout (Q8_2_X4, what the orphan's
//      declaration would hand it) must DISAGREE with the reference by far more than the tolerance.
//      The cast and the declaration have to move together: changing only the cast (the orphan fix) is a
//      regression in a tree whose CPU traits row says Q8_0_X4.
//   3. graph level: ggml_mul_mat(W, X) through ggml_graph_compute (1 and 4 threads), in the iqk mode of
//      this process. Run twice by ctest: default, and GGML_NO_IQK_GEMM=1 (forces the scalar path).
//   4. comparator controls: a perturbed value and a NaN must be reported; an all-zero signal fails.

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

// Random weight bytes with a sane fp16 scale in the first two bytes of each 128-group (both Bonsai
// blocks are { ggml_half d; uint8 qs[] }), so every code at every position is exercised.
static void fill_weights(std::vector<uint8_t> & A, ggml_type t, int64_t Nx, int64_t K, std::mt19937 & rng) {
    const size_t rowA = ggml_row_size(t, K);
    const size_t grp  = ggml_row_size(t, 128);
    A.assign((size_t) Nx * rowA, 0);
    if (t != GGML_TYPE_Q1_0_G128 && t != GGML_TYPE_Q2_0_G128) {   // harness control types: quantize real floats
        std::uniform_real_distribution<float> wv(-1.0f, 1.0f);
        std::vector<float> f((size_t) K);
        for (int64_t r = 0; r < Nx; r++) {
            for (auto & v : f) v = wv(rng);
            ggml_get_type_traits(t)->from_float_ref(f.data(), A.data() + (size_t) r * rowA, K);
        }
        return;
    }
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.005f, 0.08f);
    for (int64_t r = 0; r < Nx; r++) {
        for (int64_t g = 0; g < K / 128; g++) {
            uint8_t * blk = A.data() + (size_t) r * rowA + (size_t) g * grp;
            const ggml_fp16_t d = to_f16(sc(rng));
            memcpy(blk, &d, 2);
            for (size_t i = 2; i < grp; i++) blk[i] = (uint8_t) byte(rng);
        }
    }
}

// Decode the activation bytes the scalar function will read, for the two grouped layouts used here.
//   Q8_0_X4:  { ggml_half d[4]; int8 qs[128] }   per 128 elements, d[k] scales qs[32k..32k+31]
//   Q8_2_X4:  { uint16 d[8]; int8 qs[128] }      d[0..3] bf16 scales (d[4..7] are int16 sub-block sums)
static void decode_act(ggml_type tB, const uint8_t * row, int64_t K, float * out) {
    for (int64_t b = 0; b < K / 128; b++) {
        if (tB == GGML_TYPE_Q8_0_X4) {
            const uint8_t * blk = row + (size_t) b * (8 + 128);
            for (int j = 0; j < 4; j++) {
                ggml_fp16_t h; memcpy(&h, blk + 2 * j, 2);
                float d; ggml_fp16_to_fp32_row(&h, &d, 1);
                for (int i = 0; i < 32; i++) out[b * 128 + 32 * j + i] = d * (float)(int8_t) blk[8 + 32 * j + i];
            }
        } else {   // Q8_2_X4
            const uint8_t * blk = row + (size_t) b * (16 + 128);
            for (int j = 0; j < 4; j++) {
                uint16_t bits; memcpy(&bits, blk + 2 * j, 2);
                const uint32_t u = (uint32_t) bits << 16;
                float d; memcpy(&d, &u, 4);
                for (int i = 0; i < 32; i++) out[b * 128 + 32 * j + i] = d * (float)(int8_t) blk[16 + 32 * j + i];
            }
        }
    }
}

// Scalar vec_dot of the type, activations quantized to tB (through the CPU from_float), against the
// reference that decodes the same activation bytes. Returns the worst relative error.
static double scalar_case(ggml_type tA, ggml_type tB, int64_t K, int64_t Nx, int64_t Ny, std::mt19937 & rng) {
    const ggml_type_traits_cpu * cpuA = ggml_get_type_traits_cpu(tA);
    const ggml_type_traits_cpu * cpuB = ggml_get_type_traits_cpu(tB);
    const ggml_type_traits     * trA  = ggml_get_type_traits(tA);
    const size_t rowA = ggml_row_size(tA, K), rowB = ggml_row_size(tB, K);

    std::vector<uint8_t> A;
    fill_weights(A, tA, Nx, K, rng);
    std::vector<float> Wd((size_t) Nx * K);
    for (int64_t r = 0; r < Nx; r++) trA->to_float(A.data() + (size_t) r * rowA, Wd.data() + (size_t) r * K, K);

    std::vector<float> X((size_t) Ny * K);
    std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
    for (auto & v : X) v = ad(rng);
    std::vector<uint8_t> B((size_t) Ny * rowB);
    std::vector<float> Xd((size_t) Ny * K);
    for (int64_t c = 0; c < Ny; c++) {
        cpuB->from_float(X.data() + (size_t) c * K, B.data() + (size_t) c * rowB, K);
        decode_act(tB, B.data() + (size_t) c * rowB, K, Xd.data() + (size_t) c * K);
    }

    std::vector<double> ref((size_t) Ny * Nx);
    std::vector<float>  got((size_t) Ny * Nx);
    for (int64_t c = 0; c < Ny; c++) {
        for (int64_t r = 0; r < Nx; r++) {
            double s = 0.0;
            for (int64_t e = 0; e < K; e++) s += (double) Wd[(size_t) r * K + e] * (double) Xd[(size_t) c * K + e];
            ref[(size_t) c * Nx + r] = s;
            float o = -12345.0f;
            cpuA->vec_dot((int) K, &o, 0, A.data() + (size_t) r * rowA, 0, B.data() + (size_t) c * rowB, 0, 1);
            got[(size_t) c * Nx + r] = o;
        }
    }
    double mr = 0.0;
    for (double v : ref) mr = std::max(mr, std::fabs(v));
    g_min_signal = std::min(g_min_signal, mr);
    return rel_to_max(ref, got);
}

static double graph_case(ggml_type tA, int64_t K, int64_t N, int64_t M, int nth, std::mt19937 & rng) {
    std::vector<uint8_t> buf(64u * 1024 * 1024);
    ggml_init_params ip = { buf.size(), buf.data(), false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, tA, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::vector<uint8_t> A;
    fill_weights(A, tA, N, K, rng);
    memcpy(W->data, A.data(), A.size());
    std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
    for (int64_t i = 0; i < K * M; i++) ((float *) X->data)[i] = ad(rng);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    const size_t rowA = ggml_row_size(tA, K);
    std::vector<float> Wd((size_t) N * K);
    for (int64_t r = 0; r < N; r++) ggml_get_type_traits(tA)->to_float(A.data() + (size_t) r * rowA, Wd.data() + (size_t) r * K, K);
    std::vector<double> ref((size_t) M * N);
    for (int64_t c = 0; c < M; c++)
        for (int64_t r = 0; r < N; r++) {
            double s = 0.0;
            for (int64_t e = 0; e < K; e++) s += (double) Wd[(size_t) r * K + e] * (double) ((float *) X->data)[(size_t) c * K + e];
            ref[(size_t) c * N + r] = s;
        }
    std::vector<float> got((float *) Y->data, (float *) Y->data + (size_t) M * N);
    double mr = 0.0;
    for (double v : ref) mr = std::max(mr, std::fabs(v));
    g_min_signal = std::min(g_min_signal, mr);
    {   // where do the bad values sit? (diagnostic, printed only when something is not finite)
        size_t nbad = 0, first = 0;
        for (size_t i = 0; i < got.size(); i++) if (!std::isfinite((double) got[i])) { if (!nbad) first = i; nbad++; }
        if (nbad) printf("    nonfinite outputs: %zu of %zu, first at column %zu row %zu (N=%d)\n", nbad, got.size(), first / (size_t) N, first % (size_t) N, (int) N);
    }
    const double e = rel_to_max(ref, got);
    ggml_free(ctx);
    return e;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    ggml_cpu_init();   // fills the fp16 table; without it every scale decodes to 0 and 0 == 0 "passes"
    std::mt19937 rng(20261016);
    const char * ni = getenv("GGML_NO_IQK_GEMM");
    const bool no_iqk = ni && ni[0] && ni[0] != '0';
    printf("mode: GGML_NO_IQK_GEMM=%s\n", no_iqk ? "1 (graph level forced onto the scalar path)" : "unset (iqk may take the type)");

    {   // harness control: a long-established type through the same graph path at 1 and 4 threads
        double hc = 0.0;
        for (int nth : { 1, 2, 3, 4 }) {
            const double e = graph_case(GGML_TYPE_Q4_0, 256, 16, 7, nth, rng);
            printf("  harness control q4_0, %d thread(s): rel err %.3e\n", nth, e);
            hc = std::max(hc, e);
        }
        printf("harness control q4_0 graph-level (1 and 4 threads): worst relative error %.3e (tol 3e-2)\n", hc);
        CHECK(hc <= 3e-2, "harness control q4_0 failed: %.3e (the graph helper itself is wrong)", hc);
    }
    const ggml_type types[] = { GGML_TYPE_Q1_0_G128, GGML_TYPE_Q2_0_G128 };
    for (ggml_type tA : types) {
        const ggml_type_traits_cpu * cpuA = ggml_get_type_traits_cpu(tA);
        const ggml_type vt = cpuA->vec_dot_type;
        printf("%s: CPU traits vec_dot_type = %s (%d); ggml.c traits vec_dot_type = %s\n", ggml_type_name(tA),
               ggml_type_name(vt), (int) vt, ggml_type_name(ggml_get_type_traits(tA)->vec_dot_type));
        CHECK(vt == GGML_TYPE_Q8_0_X4 || vt == GGML_TYPE_Q8_2_X4, "vec_dot_type %d is not one of the two grouped layouts this test decodes", (int) vt);
        if (g_fail) return 1;

        // 1. scalar vec_dot with the activation type the CPU backend would hand it
        double worst = 0.0;
        for (int64_t K : { 128, 256, 640 })
            for (int64_t Nx : { 8, 5 })
                for (int64_t Ny : { 1, 3 })
                    worst = std::max(worst, scalar_case(tA, vt, K, Nx, Ny, rng));
        printf("  scalar vec_dot with %s activations: worst relative error %.3e (tol 1e-4)\n", ggml_type_name(vt), worst);
        CHECK(worst <= 1e-4, "%s scalar vec_dot disagrees with its declared activation layout (%s): %.3e", ggml_type_name(tA), ggml_type_name(vt), worst);

        // 2. control: the OTHER grouped layout must be wrong for this scalar function
        const ggml_type other = (vt == GGML_TYPE_Q8_0_X4) ? GGML_TYPE_Q8_2_X4 : GGML_TYPE_Q8_0_X4;
        const double e_other = scalar_case(tA, other, 256, 8, 3, rng);
        printf("  control: same scalar vec_dot fed %s activations: relative error %.3e (must be > 0.05)\n", ggml_type_name(other), e_other);
        CHECK(e_other > 0.05, "control is not sensitive: the %s scalar vec_dot also accepts %s (err %.3e)", ggml_type_name(tA), ggml_type_name(other), e_other);

        // 3. graph level
        double gw = 0.0;
        const int64_t shapes[][3] = { {256, 16, 1}, {256, 16, 7}, {640, 33, 5}, {128, 64, 17} };
        for (auto & sh : shapes)
            for (int nth : { 1, 4 }) {
                printf("    graph K=%d N=%d M=%d threads=%d: running\n", (int) sh[0], (int) sh[1], (int) sh[2], nth);
                const double ge = graph_case(tA, sh[0], sh[1], sh[2], nth, rng);
                printf("    graph K=%d N=%d M=%d threads=%d: rel err %.3e\n", (int) sh[0], (int) sh[1], (int) sh[2], nth, ge);
                gw = std::max(gw, ge);
            }
        printf("  graph-level ggml_mul_mat: worst relative error %.3e (tol 3e-2: q8 rounding of the f32 activations)\n", gw);
        CHECK(gw <= 3e-2, "%s graph-level mul_mat error %.3e", ggml_type_name(tA), gw);
    }
    CHECK(g_min_signal > 1e-2, "degenerate: a case had an (almost) all-zero reference (min signal %.3e)", g_min_signal);

    // 4. comparator controls
    {
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
    printf("OK: Bonsai scalar vec_dots agree with the CPU traits activation layout\n");
    return 0;
}
