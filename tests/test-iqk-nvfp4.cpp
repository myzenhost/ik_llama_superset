// Does the iqk x86 GEMM take NVFP4 weights x plain Q8_0 activations, and does it agree with the type's own scalar
// vec_dot?  (S69-FASTLLM-PICK-NVFP4-CPU-IQK-2026-10-27-01; review of ztxz16/fastllm, Apache-2.0, no code copied)
//
// GATE-CLAIM-OK: "CPU" names the backend source directory this test links; it computes on the host and touches no
// accelerator. No device is claimed unavailable.
//
// WHY. Before the kernel NVFP4 had no iqk entry: every CPU matmul fell to the per-row generic
// ggml_vec_dot_nvfp4_q8_0 (llama-bench Qwen3-0.6B, same source, p512 / tg32, MEASURED 2026-10-27).
//
// WHAT IT CHECKS.
//   1. REACH: iqk_gemm_type_b(NVFP4, ne00, vec_dot_type) names Q8_0.  THIS LINE FAILS BEFORE AND PASSES AFTER
//      (before: iqk has no NVFP4 case and returns -1).
//   2. iqk_mul_mat() accepts the pairing for Ny = 1..9, 16, 17, 33 and matches (a) the type's CPU vec_dot fed the
//      same Q8_0 bytes -- the arithmetic order is the reference's, so the match is expected to be exact; the test
//      requires <= 1e-5 of the largest value and prints how many outputs are bit-identical -- and (b) a double
//      precision reference built from the type's to_float and the decoded Q8_0 activations (tol 1e-4).
//      RANDOM weight bytes: every code and every scale byte class (zero, subnormal, 0x7F) appears.
//      Row lengths include 1, 3, 5, 10 and 16 blocks of 64, so the per-block loop has no multiple-of-four assumption.
//   3. Graph level through ggml-cpu.c: ggml_mul_mat (1 and 4 threads), ggml_mul_mat_id (broadcast and per-slot
//      activations) and the fused ggml_moe_up_gate, against the same vec_dot reference.
// CONTROLS: a reference with two sub-block scales exchanged must disagree; the comparator sees a perturbation and a NaN.

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

static const ggml_type tA = GGML_TYPE_NVFP4;
static const ggml_type tB = GGML_TYPE_Q8_0;

// Random block bytes. Scale bytes: mostly sane UE4M3 (exponent 4..10), with the special classes
// 0x00, a subnormal (0x01..0x07) and 0x7F (decodes to 0) mixed in so the scale table is exercised.
static void fill_nvfp4_row(uint8_t * row, int64_t K, std::mt19937 & rng) {
    std::uniform_int_distribution<int> byte(0, 255), cls(0, 15), sane(0x20, 0x57);
    for (int64_t b = 0; b < K / 64; b++) {
        uint8_t * blk = row + (size_t) b * ggml_type_size(tA);
        for (int s = 0; s < 4; s++) {
            const int c = cls(rng);
            blk[s] = (uint8_t) (c == 0 ? 0x00 : c == 1 ? 0x01 + (byte(rng) & 7) : c == 2 ? 0x7F : sane(rng));
        }
        for (int i = 0; i < 32; i++) blk[4 + i] = (uint8_t) byte(rng);
    }
}

static void fill_weights(std::vector<uint8_t> & A, int64_t Nx, int64_t K, std::mt19937 & rng) {
    const size_t rowA = ggml_row_size(tA, K);
    A.assign((size_t) Nx * rowA, 0);
    for (int64_t r = 0; r < Nx; r++) fill_nvfp4_row(A.data() + (size_t) r * rowA, K, rng);
}

static void decode_q8_0(const uint8_t * row, int64_t K, float * out) {
    for (int64_t b = 0; b < K / 32; b++) {
        const uint8_t * blk = row + (size_t) b * 34;
        ggml_fp16_t h; memcpy(&h, blk, 2);
        float d; ggml_fp16_to_fp32_row(&h, &d, 1);
        for (int i = 0; i < 32; i++) out[b * 32 + i] = d * (float)(int8_t) blk[2 + i];
    }
}

// the type's own scalar vec_dot, activations quantised to Q8_0 by the CPU from_float
static float vec_dot_ref(const uint8_t * a_row, const uint8_t * b_row, int64_t K) {
    const ggml_type_traits_cpu * cpuA = ggml_get_type_traits_cpu(tA);
    float o = -12345.0f;
    cpuA->vec_dot((int) K, &o, 0, a_row, 0, b_row, 0, 1);
    return o;
}

struct case_result { double err_vd, err_dbl; size_t exact, total; bool declined; };

static case_result run_case(int64_t K, int64_t Nx, int64_t Ny, std::mt19937 & rng, bool swap_scale = false) {
    const size_t rowA = ggml_row_size(tA, K), rowB = ggml_row_size(tB, K);
    std::vector<uint8_t> A;
    fill_weights(A, Nx, K, rng);
    std::vector<float> Wd((size_t) Nx * K);
    const ggml_type_traits * trA = ggml_get_type_traits(tA);
    for (int64_t r = 0; r < Nx; r++) trA->to_float(A.data() + (size_t) r * rowA, Wd.data() + (size_t) r * K, K);

    std::vector<float> X((size_t) Ny * K);
    std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
    for (auto & v : X) v = ad(rng);
    const ggml_type_traits_cpu * cpuB = ggml_get_type_traits_cpu(tB);
    std::vector<uint8_t> B((size_t) Ny * rowB);
    std::vector<float> Xd((size_t) Ny * K);
    for (int64_t c = 0; c < Ny; c++) {
        cpuB->from_float(X.data() + (size_t) c * K, B.data() + (size_t) c * rowB, K);
        decode_q8_0(B.data() + (size_t) c * rowB, K, Xd.data() + (size_t) c * K);
    }

    std::vector<double> ref_dbl((size_t) Ny * Nx);
    std::vector<double> ref_vd((size_t) Ny * Nx);
    for (int64_t c = 0; c < Ny; c++) {
        for (int64_t r = 0; r < Nx; r++) {
            double s = 0.0;
            for (int64_t e = 0; e < K; e++) {
                int64_t we = e;
                if (swap_scale) {   // control: weights of sub-blocks 0 and 1 of every block exchanged
                    const int64_t within = e % 64;
                    if (within < 16) we = e + 16; else if (within < 32) we = e - 16;
                }
                s += (double) Wd[(size_t) r * K + we] * (double) Xd[(size_t) c * K + e];
            }
            ref_dbl[(size_t) c * Nx + r] = s;
            ref_vd[(size_t) c * Nx + r] = vec_dot_ref(A.data() + (size_t) r * rowA, B.data() + (size_t) c * rowB, K);
        }
    }

    std::vector<float> C((size_t) Ny * Nx, -12345.0f);
    case_result res{0, 0, 0, (size_t) Ny * Nx, false};
    if (!iqk_mul_mat(Nx, Ny, K, (int) tA, A.data(), (long) rowA, (int) tB, B.data(), (long) rowB, C.data(), (long) Nx, 0, 1)) {
        res.declined = true;
        return res;
    }
    res.err_vd  = rel_to_max(ref_vd, C);
    res.err_dbl = rel_to_max(ref_dbl, C);
    for (size_t i = 0; i < C.size(); i++) if (C[i] == (float) ref_vd[i]) res.exact++;
    double mr = 0.0, mg = 0.0;
    for (size_t i = 0; i < ref_dbl.size(); i++) { mr = std::max(mr, std::fabs(ref_dbl[i])); mg = std::max(mg, (double) std::fabs(C[i])); }
    if (!swap_scale) g_min_signal = std::min(g_min_signal, std::min(mr, mg));
    return res;
}

static ggml_tensor * new_weight(ggml_context * ctx, int64_t K, int64_t N, int64_t E, std::mt19937 & rng, std::vector<uint8_t> * keep) {
    ggml_tensor * W = E > 0 ? ggml_new_tensor_3d(ctx, tA, K, N, E) : ggml_new_tensor_2d(ctx, tA, K, N);
    const size_t rowA = ggml_row_size(tA, K);
    const int64_t rows = N * (E > 0 ? E : 1);
    for (int64_t r = 0; r < rows; r++) fill_nvfp4_row((uint8_t *) W->data + (size_t) r * rowA, K, rng);
    if (keep) keep->assign((uint8_t *) W->data, (uint8_t *) W->data + (size_t) rows * rowA);
    return W;
}

// activations -> Q8_0 rows as the CPU backend would, so the reference sees the same bytes
static void quantize_acts(const float * x, int64_t K, int64_t n, std::vector<uint8_t> & out) {
    const size_t rowB = ggml_row_size(tB, K);
    out.assign((size_t) n * rowB, 0);
    for (int64_t i = 0; i < n; i++) ggml_get_type_traits_cpu(tB)->from_float(x + (size_t) i * K, out.data() + (size_t) i * rowB, K);
}

static void pick_experts(ggml_tensor * ids, int64_t E, int64_t n_used, int64_t T, std::mt19937 & rng) {
    for (int64_t t = 0; t < T; t++) {   // distinct experts per token
        std::vector<int> pool(E);
        for (int i = 0; i < (int) E; i++) pool[i] = i;
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int64_t s = 0; s < n_used; s++) ((int32_t *) ids->data)[t * n_used + s] = pool[s];
    }
}

static double graph_mul_mat(int64_t K, int64_t N, int64_t M, int nth, std::mt19937 & rng) {
    std::vector<uint8_t> buf(64u * 1024 * 1024);
    ggml_init_params ip = { buf.size(), buf.data(), false };
    ggml_context * ctx = ggml_init(ip);
    std::vector<uint8_t> A;
    ggml_tensor * W = new_weight(ctx, K, N, 0, rng, &A);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
    for (int64_t i = 0; i < K * M; i++) ((float *) X->data)[i] = ad(rng);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    std::vector<uint8_t> B;
    quantize_acts((float *) X->data, K, M, B);
    const size_t rowA = ggml_row_size(tA, K), rowB = ggml_row_size(tB, K);
    std::vector<double> ref((size_t) M * N);
    for (int64_t c = 0; c < M; c++) for (int64_t r = 0; r < N; r++)
        ref[(size_t) c * N + r] = vec_dot_ref(A.data() + (size_t) r * rowA, B.data() + (size_t) c * rowB, K);
    std::vector<float> got((float *) Y->data, (float *) Y->data + (size_t) M * N);
    double mr = 0.0;
    for (double v : ref) mr = std::max(mr, std::fabs(v));
    g_min_signal = std::min(g_min_signal, mr);
    const double e = rel_to_max(ref, got);
    ggml_free(ctx);
    return e;
}

static double graph_mul_mat_id(int64_t K, int64_t N, int64_t E, int64_t n_used, int64_t T, int64_t nb1, int nth, std::mt19937 & rng) {
    std::vector<uint8_t> buf(96u * 1024 * 1024);
    ggml_init_params ip = { buf.size(), buf.data(), false };
    ggml_context * ctx = ggml_init(ip);
    std::vector<uint8_t> A;
    ggml_tensor * W = new_weight(ctx, K, N, E, rng, &A);
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, nb1, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
    for (int64_t i = 0; i < K * nb1 * T; i++) ((float *) X->data)[i] = ad(rng);
    pick_experts(ids, E, n_used, T, rng);
    ggml_tensor * Y = ggml_mul_mat_id(ctx, W, X, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    std::vector<uint8_t> B;
    quantize_acts((float *) X->data, K, nb1 * T, B);
    const size_t rowA = ggml_row_size(tA, K), rowB = ggml_row_size(tB, K);
    std::vector<double> ref((size_t) T * n_used * N);
    for (int64_t t = 0; t < T; t++) for (int64_t s = 0; s < n_used; s++) {
        const int e = ((int32_t *) ids->data)[t * n_used + s];
        const int64_t col = nb1 == 1 ? 0 : s;
        for (int64_t r = 0; r < N; r++)
            ref[(size_t) (t * n_used + s) * N + r] = vec_dot_ref(A.data() + ((size_t) e * N + r) * rowA, B.data() + (size_t) (t * nb1 + col) * rowB, K);
    }
    std::vector<float> got((float *) Y->data, (float *) Y->data + ref.size());
    double mr = 0.0;
    for (double v : ref) mr = std::max(mr, std::fabs(v));
    g_min_signal = std::min(g_min_signal, mr);
    const double err = rel_to_max(ref, got);
    ggml_free(ctx);
    return err;
}

static double graph_moe_up_gate(int64_t K, int64_t N, int64_t E, int64_t n_used, int64_t T, int nth, std::mt19937 & rng) {
    std::vector<uint8_t> buf(128u * 1024 * 1024);
    ggml_init_params ip = { buf.size(), buf.data(), false };
    ggml_context * ctx = ggml_init(ip);
    std::vector<uint8_t> Aup, Agate;
    ggml_tensor * up   = new_weight(ctx, K, N, E, rng, &Aup);
    ggml_tensor * gate = new_weight(ctx, K, N, E, rng, &Agate);
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    std::uniform_real_distribution<float> ad(-0.5f, 0.5f);
    for (int64_t i = 0; i < K * T; i++) ((float *) X->data)[i] = ad(rng);
    pick_experts(ids, E, n_used, T, rng);
    ggml_tensor * Y = ggml_moe_up_gate(ctx, up, gate, X, ids, GGML_UNARY_OP_SILU);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_graph_compute_with_ctx(ctx, gf, nth);

    std::vector<uint8_t> B;
    quantize_acts((float *) X->data, K, T, B);
    const size_t rowA = ggml_row_size(tA, K), rowB = ggml_row_size(tB, K);
    std::vector<double> ref((size_t) T * n_used * N);
    for (int64_t t = 0; t < T; t++) for (int64_t s = 0; s < n_used; s++) {
        const int e = ((int32_t *) ids->data)[t * n_used + s];
        for (int64_t r = 0; r < N; r++) {
            const size_t ar = ((size_t) e * N + r) * rowA;
            const double u = vec_dot_ref(Aup.data() + ar,   B.data() + (size_t) t * rowB, K);
            const double g = vec_dot_ref(Agate.data() + ar, B.data() + (size_t) t * rowB, K);
            ref[(size_t) (t * n_used + s) * N + r] = g / (1.0 + std::exp(-g)) * u;
        }
    }
    std::vector<float> got((float *) Y->data, (float *) Y->data + ref.size());
    double mr = 0.0;
    for (double v : ref) mr = std::max(mr, std::fabs(v));
    g_min_signal = std::min(g_min_signal, mr);
    const double err = rel_to_max(ref, got);
    ggml_free(ctx);
    return err;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
#if !defined(__x86_64__) && !defined(_M_X64)
    printf("SKIP: the NVFP4 iqk kernel is x86 only\n");
    return 77;
#endif
    ggml_cpu_init();   // fp16 table + CPU traits; without it every scale decodes to 0 and 0 == 0 "passes"
    std::mt19937 rng(20261027);
    const char * ni = getenv("GGML_NO_IQK_GEMM");
    const bool no_iqk = ni && ni[0] && ni[0] != '0';
    printf("mode: GGML_NO_IQK_GEMM=%s\n", no_iqk ? "1 (graph level forced onto the scalar path)" : "unset (iqk may take the type)");

    const ggml_type_traits_cpu * cpuA = ggml_get_type_traits_cpu(tA);
    const int pref = cpuA ? (int) cpuA->vec_dot_type : -1;
    CHECK(pref == (int) GGML_TYPE_Q8_0, "NVFP4 vec_dot_type is %d, this test feeds Q8_0 (%d)", pref, (int) GGML_TYPE_Q8_0);
    const int tb = iqk_gemm_type_b((int) tA, 256, pref);
    printf("iqk_gemm_type_b(NVFP4=%d, 256, vec_dot_type=%d) = %d (Q8_0 = %d)\n", (int) tA, pref, tb, (int) GGML_TYPE_Q8_0);
    const bool reach = tb == (int) GGML_TYPE_Q8_0;
    CHECK(reach, "iqk has no NVFP4 kernel (activation type %d, want %d)", tb, (int) GGML_TYPE_Q8_0);
    if (reach) {
        const int tb96 = iqk_gemm_type_b((int) tA, 96, pref);
        CHECK(tb96 == -1, "iqk accepts ne00 = 96, which is not a multiple of the 64-wide block (answer %d)", tb96);
    }

    // 2. direct iqk_mul_mat
    const int64_t Ks[]  = { 64, 192, 320, 640, 1024 };
    const int64_t Nxs[] = { 8, 5, 33 };
    const int64_t Nys[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 33 };
    int n_cases = 0, n_declined = 0;
    double worst_vd = 0.0, worst_dbl = 0.0;
    size_t exact = 0, total = 0;
    for (int64_t K : Ks) for (int64_t Nx : Nxs) for (int64_t Ny : Nys) {
        const case_result r = run_case(K, Nx, Ny, rng);
        n_cases++;
        if (r.declined) { n_declined++; continue; }
        worst_vd = std::max(worst_vd, r.err_vd);
        worst_dbl = std::max(worst_dbl, r.err_dbl);
        exact += r.exact; total += r.total;
        CHECK(std::isfinite(r.err_vd) && r.err_vd <= 1e-5, "K=%lld Nx=%lld Ny=%lld differs from the scalar vec_dot by %.3e", (long long) K, (long long) Nx, (long long) Ny, r.err_vd);
        CHECK(r.err_dbl <= 1e-4, "K=%lld Nx=%lld Ny=%lld differs from the double reference by %.3e", (long long) K, (long long) Nx, (long long) Ny, r.err_dbl);
    }
    CHECK(n_declined == 0, "iqk_mul_mat declined %d of %d cases", n_declined, n_cases);
    printf("direct iqk_mul_mat: %d cases (%d declined), worst rel err vs vec_dot %.3e, vs double %.3e, bit-identical outputs %zu of %zu\n",
           n_cases, n_declined, worst_vd, worst_dbl, exact, total);

    // 3. graph level
    {
        double gw = 0.0;
        const int64_t shapes[][3] = { {256, 16, 1}, {192, 16, 7}, {320, 33, 5}, {128, 64, 17}, {640, 40, 64} };  // K, N, M
        for (auto & sh : shapes) for (int nth : { 1, 4 }) {
            const double e = graph_mul_mat(sh[0], sh[1], sh[2], nth, rng);
            CHECK(std::isfinite(e) && e <= 1e-5, "mul_mat K=%lld N=%lld M=%lld nth=%d rel err %.3e", (long long) sh[0], (long long) sh[1], (long long) sh[2], nth, e);
            gw = std::max(gw, e);
        }
        printf("graph ggml_mul_mat: worst rel err vs vec_dot %.3e\n", gw);

        double iw = 0.0;
        const int64_t id_shapes[][6] = { {256, 32, 4, 2, 5, 1}, {256, 32, 4, 2, 5, 2}, {192, 24, 8, 3, 40, 1}, {128, 16, 6, 2, 9, 2} }; // K,N,E,n_used,T,nb1
        for (auto & sh : id_shapes) for (int nth : { 1, 4 }) {
            const double e = graph_mul_mat_id(sh[0], sh[1], sh[2], sh[3], sh[4], sh[5], nth, rng);
            CHECK(std::isfinite(e) && e <= 1e-5, "mul_mat_id K=%lld N=%lld E=%lld used=%lld T=%lld nb1=%lld nth=%d rel err %.3e",
                  (long long) sh[0], (long long) sh[1], (long long) sh[2], (long long) sh[3], (long long) sh[4], (long long) sh[5], nth, e);
            iw = std::max(iw, e);
        }
        printf("graph ggml_mul_mat_id: worst rel err vs vec_dot %.3e\n", iw);

        double uw = 0.0;
        const int64_t ug_shapes[][5] = { {256, 32, 4, 2, 5}, {192, 24, 8, 3, 40} };   // K,N,E,n_used,T
        for (auto & sh : ug_shapes) for (int nth : { 1, 4 }) {
            const double e = graph_moe_up_gate(sh[0], sh[1], sh[2], sh[3], sh[4], nth, rng);
            CHECK(std::isfinite(e) && e <= 2e-3, "moe_up_gate K=%lld N=%lld E=%lld used=%lld T=%lld nth=%d rel err %.3e",
                  (long long) sh[0], (long long) sh[1], (long long) sh[2], (long long) sh[3], (long long) sh[4], nth, e);
            uw = std::max(uw, e);
        }
        printf("graph ggml_moe_up_gate: worst rel err vs vec_dot-derived reference %.3e (tol 2e-3: silu in float)\n", uw);
    }
    CHECK(g_min_signal > 1e-2, "degenerate: a case had an (almost) all-zero reference or result (min signal %.3e)", g_min_signal);

    // controls
    {
        const case_result r = run_case(256, 8, 4, rng, true);
        printf("control: double reference with sub-block weights 0/1 exchanged differs by %.3e (must be > 0.05)\n", r.err_dbl);
        CHECK(r.declined || r.err_dbl > 0.05, "the double reference is NOT sensitive to a sub-block exchange (err %.3e)", r.err_dbl);
        std::vector<double> rr(64, 1.0);
        std::vector<float>  g(64, 1.0f);
        const double clean = rel_to_max(rr, g);
        g[17] = 1.5f;
        const double dirty = rel_to_max(rr, g);
        printf("control: comparator clean %.3e, perturbed %.3e\n", clean, dirty);
        CHECK(clean == 0.0 && dirty > 0.4, "comparator cannot see a perturbation");
        g[17] = NAN;
        CHECK(!std::isfinite(rel_to_max(rr, g)), "comparator passes a NaN");
    }

    if (g_fail) {
        printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    printf("OK: NVFP4 iqk kernel reachable and agrees with the scalar vec_dot\n");
    return 0;
}
