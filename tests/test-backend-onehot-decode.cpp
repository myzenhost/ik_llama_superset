// Which weights does a BACKEND's matmul decode wrongly, and how?
//
// test-backend-ops reports one NMSE per MUL_MAT case. When a type fails with NMSE ~1 (output
// uncorrelated with the reference) that number cannot say whether the scale, the grid lookup,
// the signs or the block addressing is wrong. This test recovers, exactly, the weight value the
// backend's matmul believes is stored at each position, and prints it next to the reference
// decoder (the type's to_float) so the pattern of disagreement is visible.
//
// THE EXTRACTION IS EXACT. src1 is a set of one-hot columns e_i. Whatever activation format the
// backend quantises src1 into (q8_1 on CUDA), a one-hot column quantises losslessly: d = 1/127,
// q[i] = 127, every other q = 0. So column i of W^T * E is the backend's own decoded weight at i,
// with no quantisation error to explain away.
//
// TWO PATHS PER TYPE, because they share no kernel code on CUDA: N = 1 column at a time goes to
// the matrix-vector kernel (MMVQ), N = K columns at once goes to MMQ or dequantise + GEMM. A fault
// in both paths points at something they share (data upload, tables, the block layout); a fault
// in one names that path.
//
// SCOPE: M = 2 rows, K = 512 (two QK_K blocks, so block addressing is exercised), uniform-random
// weights, one backend given by argv[1] (default CUDA0), compared against to_float on the SAME
// bytes. Controls: q4_K and iq2_xs are included and are expected to agree; if they do not, the
// harness is what is wrong and no verdict below is meaningful.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr int64_t K = 512;
static constexpr int64_t M = 2;

// Run W (M x K, quantised) against ncols one-hot columns starting at position col0.
// Returns out[c*M + r] = backend's decoded weight W[r][col0 + c].
static bool run_backend(ggml_backend_t be, ggml_type t, const std::vector<uint8_t> & wbytes,
                        int64_t col0, int64_t ncols, std::vector<float> & out) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, t, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, ncols);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(w, wbytes.data(), 0, wbytes.size());
    std::vector<float> xs(K*ncols, 0.0f);
    for (int64_t c = 0; c < ncols; ++c) xs[c*K + col0 + c] = 1.0f;
    ggml_backend_tensor_set(x, xs.data(), 0, xs.size()*sizeof(float));

    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    out.assign(M*ncols, 0.0f);
    if (ok) ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));

    // The weight bytes the backend holds AFTER compute must still be the bytes uploaded. A
    // backend that transforms weights on upload or in place would otherwise be indistinguishable
    // from a wrong kernel.
    std::vector<uint8_t> back(wbytes.size());
    ggml_backend_tensor_get(w, back.data(), 0, back.size());
    if (back != wbytes) {
        size_t first = 0, ndiff = 0;
        for (size_t k = 0; k < back.size(); ++k) if (back[k] != wbytes[k]) { if (!ndiff) first = k; ++ndiff; }
        printf("  %-8s WEIGHT BYTES CHANGED on the backend: %zu of %zu differ, first at byte %zu\n",
               ggml_type_name(t), ndiff, back.size(), first);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static int g_fail = 0;

static void report(const char * tn, const char * path, const std::vector<float> & ref,
                   const std::vector<float> & got /* [pos*M + r] */) {
    double maxref = 0;
    for (float v : ref) maxref = std::max(maxref, (double) std::fabs(v));
    // The wide path on CUDA is MMQ, which holds per-block weight scales (and k-quant mins) as
    // half2 in shared memory: ~2^-11 relative each, and d*sc*q - dmin*m can compound them.
    // MEASURED on q4_K: 7 of 1024 weights at 0.11-0.14% relative, no zeros, no sign flips.
    // A decode fault is orders of magnitude larger (the IQ*_S miscompile gave exact zeros), so
    // the looser bound on that path costs no sensitivity to the faults this test exists for.
    const double rel = strcmp(path, "N=1") == 0 ? 1e-3 : 2e-3;
    const double tol = rel * (maxref > 0 ? maxref : 1.0);
    int bad = 0, sign_flip = 0, zero_got = 0;
    int bad_per32[K/32] = {0};
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t i = 0; i < K; ++i) {
            const float a = ref[r*K + i], b = got[i*M + r];
            if (!(std::fabs(a - b) <= tol)) {
                ++bad; ++bad_per32[i/32];
                if (std::fabs(a + b) <= tol && std::fabs(a) > tol) ++sign_flip;
                if (b == 0.0f) ++zero_got;
            }
        }
    }
    printf("  %-8s %-5s bad %4d / %lld   (sign-flipped %d, backend-zero %d)   tol %.3g\n",
           tn, path, bad, (long long)(M*K), sign_flip, zero_got, tol);
    if (bad) {
        ++g_fail;
        int shown = 0;
        for (int64_t r = 0; r < M && shown < 8; ++r) {
            for (int64_t i = 0; i < K && shown < 8; ++i) {
                const float a = ref[r*K + i], b = got[i*M + r];
                if (!(std::fabs(a - b) <= tol)) {
                    printf("           bad r=%lld i=%3lld  ref %10.6f  backend %10.6f  |diff| %.3g\n",
                           (long long) r, (long long) i, a, b, std::fabs(a - b));
                    ++shown;
                }
            }
        }
        printf("           bad per 32-weight block (row-major over both rows):");
        for (int b = 0; b < K/32; ++b) printf(" %d", bad_per32[b]);
        printf("\n           row 0, first 32 weights  ref | backend | backend/ref\n");
        for (int i = 0; i < 32; ++i) {
            const float a = ref[i], b = got[i*M + 0];
            printf("           [%2d] %10.5f | %10.5f | %8.4f\n", i, a, b, a != 0 ? b/a : 0.0f);
        }
    }
}

int main(int argc, char ** argv) {
    const char * be_name = argc > 1 ? argv[1] : "CUDA0";
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name(be_name, nullptr);
    if (!be) {
        printf("SKIPPED: backend %s could not be initialised in this build. Not a verdict.\n", be_name);
        return 0;
    }
    printf("backend %s, M=%lld K=%lld\n", be_name, (long long) M, (long long) K);

    // Controls first (q4_K, iq2_xs), then the three types nvcc 13.2 miscompiled (#28581), then every
    // other type whose CUDA kernels read bytes through a uint8_t* view of a packed int -- the idiom
    // that was miscompiled -- including the fork's own IQ*_K family, which test-backend-ops'
    // MUL_MAT sweep does not exercise on CUDA.
    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XS,
                                GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_S,
                                GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_M,
                                GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                                GGML_TYPE_IQ2_K, GGML_TYPE_IQ2_KS, GGML_TYPE_IQ2_KL, GGML_TYPE_IQ3_K,
                                GGML_TYPE_IQ3_KS, GGML_TYPE_IQ4_K, GGML_TYPE_IQ4_KS, GGML_TYPE_IQ4_KSS,
                                GGML_TYPE_IQ5_K, GGML_TYPE_IQ5_KS, GGML_TYPE_IQ6_K };
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);

    for (ggml_type t : types) {
        const char * tn = ggml_type_name(t);
        std::vector<float> src(M*K);
        for (float & v : src) v = U(rng);
        std::vector<float> imat(K, 1.0f);
        const size_t rs = ggml_row_size(t, K);
        std::vector<uint8_t> wb(rs*M);
        ggml_quantize_chunk(t, src.data(), wb.data(), 0, M, K, imat.data(), nullptr);

        std::vector<float> ref(M*K);
        const ggml_type_traits * tr = ggml_get_type_traits(t);
        for (int64_t r = 0; r < M; ++r) tr->to_float(wb.data() + r*rs, ref.data() + r*K, K);

        // MMVQ path: one column at a time.
        std::vector<float> got1(K*M), tmp;
        bool ok = true;
        for (int64_t i = 0; i < K && ok; ++i) {
            ok = run_backend(be, t, wb, i, 1, tmp);
            for (int64_t r = 0; r < M; ++r) got1[i*M + r] = tmp[r];
        }
        if (!ok) { printf("  %-8s N=1   compute FAILED\n", tn); ++g_fail; }
        else report(tn, "N=1", ref, got1);

        // Wide path: all K columns at once.
        std::vector<float> gotK;
        if (!run_backend(be, t, wb, 0, K, gotK)) { printf("  %-8s N=%lld compute FAILED\n", tn, (long long) K); ++g_fail; }
        else report(tn, "N=K", ref, gotK);
    }

    ggml_backend_free(be);
    printf("%s: %d path(s) disagree with to_float\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
