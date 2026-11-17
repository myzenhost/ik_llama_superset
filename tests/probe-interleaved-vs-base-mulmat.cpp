// Standalone probe (links the static ggml archives; build line as tests/probe-cpu-op-reference.cpp).
// Measured 2026-09-26 (Zen 2, AVX2), after fork a35a8d38e: 53 checks (28 types + 25 kernel-only repack checks), 0 FAIL.
//   q4/q5/q6_k_r4 and iq2_bn_r4 differ from base by ~7.8e-3 even on identical bytes but are as accurate vs f32 (activation
//   rounding; q6_k_r4 ratio 1.049 is the closest to the 1.05 slack). bf16_r16 WAS 890x worse before the AVX2 kernel +
//   RNE repack. Not covered: q8_k_r8 / q8_k_r16 (activation types).
// Multi-row harness for row-interleaved repacks (_r4/_r8/_r16): each is checked against its BASE type
// through a real CPU ggml_mul_mat. Same weights quantised both ways, same activations; a lossless
// repack must give (near-)identical outputs. Oracle = the base type, which test-quantize-fns covers.
// NEGATIVE CONTROL: base type vs a DIFFERENT base type (q4_0 vs q5_0) must disagree noticeably.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_type type_by_name(const std::string & nm) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * tn = ggml_type_name((ggml_type)t);
        if (tn && nm == tn) return (ggml_type)t;
    }
    return GGML_TYPE_COUNT;
}

// y = W x for W [nrows x n] quantised as `wt`, x [n x ncols] f32. Returns y (nrows*ncols) or empty on failure.
extern "C" void iqk_repack_tensor(struct ggml_tensor * tensor);   // ggml/src/iqk/iqk_quantize.h

// KERNEL-ONLY path: quantise as the BASE type, then iqk_repack_tensor() the same bytes in place (what run-time
// repack does at load). The _rN matmul must then reproduce the base result to rounding -- this isolates the
// kernel + repacker from the _rN type's own quantiser. Returns empty if the repacker did not convert to `want`.
static std::vector<float> mm_repacked(ggml_type base, ggml_type want, const std::vector<float> & W, const std::vector<float> & X, int n, int nrows, int ncols) {
    ggml_init_params ip = { (size_t)256 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, base, n, nrows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, ncols);
    ggml_quantize_init(base);
    std::vector<float> imat(n, 1.0f);   // same flat imatrix as mm(), always
    ggml_quantize_chunk(base, W.data(), w->data, 0, nrows, n, imat.data(), nullptr);
    iqk_repack_tensor(w);
    std::vector<float> out;
    if (w->type == want) {
        memcpy(x->data, X.data(), X.size() * sizeof(float));
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 4);
        out.resize((size_t)nrows * ncols);
        memcpy(out.data(), y->data, out.size() * sizeof(float));
    }
    ggml_free(ctx);
    return out;
}

static std::vector<float> mm(ggml_type wt, const std::vector<float> & W, const std::vector<float> & X, int n, int nrows, int ncols) {
    ggml_init_params ip = { (size_t)256 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, wt, n, nrows);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, ncols);
    ggml_quantize_init(wt);
    // Types that cannot quantise without an importance matrix (iq2_xxs, iq1_s, ...) get a flat one: every
    // column weighted equally. The SAME imatrix goes to the base and the repack, so the comparison is fair.
    // Always pass it: ggml_quantize_requires_imatrix() lists IQ2_XXS / IQ2_XS / IQ1_S but NOT their _r4 twins
    // (identical upstream; llama-quantize enforces the _r4 ones itself, llama-quantize.cpp:1788). Keying on it
    // gave the base an imatrix and the _r4 none, and produced a FALSE "iq2_xxs_r4 16.6% worse" on 2026-09-26 --
    // the two paths are byte-identical when both get the same imatrix (probe: 0 of 8448 bytes differ).
    std::vector<float> imat(n, 1.0f);
    ggml_quantize_chunk(wt, W.data(), w->data, 0, nrows, n, imat.data(), nullptr);
    memcpy(x->data, X.data(), X.size() * sizeof(float));
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    std::vector<float> out(nrows * ncols);
    memcpy(out.data(), y->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

static double rel(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return std::sqrt(num / (den + 1e-30));
}

int main(int argc, char ** argv) {
    ggml_cpu_init();
    const int n = 1024, nrows = 32, ncols = 8;
    std::vector<float> W((size_t)n * nrows), X((size_t)n * ncols);
    std::mt19937 g(11); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : W) v = nd(g);
    for (auto & v : X) v = nd(g);

    const double TOL = 2e-3;   // repack of the SAME quants: only activation-path rounding may differ
    // PASS RULE: a repack may quantise DIFFERENTLY from its base (q4/q5/q6_k_r4, iq2_bn_r4 do: ~0.8% apart),
    // so equality with the base is reported, not required. What is required: its error against the EXACT
    // f32 product is no more than 5% worse than the base type's. That catches a faulty kernel or repack
    // (bf16_r16 measured 890x worse before the AVX2 kernel + round-to-nearest-even repack of 2026-09-26).
    const double ACC_SLACK = 1.05;
    std::vector<float> ref((size_t)nrows * ncols);
    for (int c = 0; c < ncols; c++) for (int r = 0; r < nrows; r++) {
        double s = 0; for (int i = 0; i < n; i++) s += (double)W[(size_t)r*n+i] * X[(size_t)c*n+i];
        ref[(size_t)c*nrows+r] = (float)s;
    }
    int fails = 0, checked = 0;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * tn = ggml_type_name((ggml_type)t);
        if (!tn) continue;
        std::string nm = tn;
        size_t us = nm.rfind('_');
        if (us == std::string::npos || us + 2 > nm.size() || nm[us + 1] != 'r' || !isdigit((unsigned char)nm[us + 2])) continue;
        if (argc > 1 && nm != argv[1]) continue;
        std::string base = nm.substr(0, us);
        ggml_type bt = type_by_name(base);
        if (bt == GGML_TYPE_COUNT) { // q2_k_r4 -> q2_K, q8_k_r8 -> q8_K ...
            std::string b2 = base; for (auto & c : b2) if (c == 'k' && &c == &b2.back()) c = 'K';
            bt = type_by_name(b2);
            if (bt == GGML_TYPE_COUNT) { printf("%-12s base '%s' not found -- SKIPPED\n", nm.c_str(), base.c_str()); continue; }
            base = b2;
        }
        // Only activation types are out of scope (q8_K has no weight quantiser for ggml_quantize_chunk). The
        // old test here also required the CPU from_float trait, which ggml_quantize_chunk does not use -- that
        // excluded 7 imatrix-only types that are perfectly checkable.
        if (!ggml_get_type_traits((ggml_type)bt)->to_float || bt == GGML_TYPE_Q8_K) {
            printf("%-12s base %s is an activation type -- SKIPPED\n", nm.c_str(), base.c_str()); continue;
        }
        fflush(stdout);
        auto yb = mm(bt, W, X, n, nrows, ncols);
        auto yr = mm((ggml_type)t, W, X, n, nrows, ncols);
        double r = rel(yr, yb);
        double eb = rel(yb, ref), er = rel(yr, ref);
        bool ok = std::isfinite(er) && er <= ACC_SLACK * eb;
        checked++; fails += !ok;
        printf("%-12s vs %-8s rel diff %.3e (%s) | err vs f32 %.3e base %.3e ratio %.3f  %s\n", nm.c_str(), base.c_str(), r,
               r < TOL ? "same quants" : "different quants", er, eb, er / eb, ok ? "ok" : "FAIL");
        // kernel-only check: base bytes repacked in place must reproduce the base output
        auto yk = mm_repacked(bt, (ggml_type)t, W, X, n, nrows, ncols);
        if (yk.empty()) {
            printf("%-12s   kernel check: iqk_repack_tensor has no %s -> %s entry on this build -- not run\n", "", base.c_str(), nm.c_str());
        } else {
            // Identical weight bytes should reproduce the base output. q4/q5/q6_k_r4 and iq2_bn_r4 do NOT (~7.8e-3,
            // the size of an int8 activation-rounding difference) yet are exactly as accurate vs f32 (ratio 0.995-1.009,
            // measured 2026-09-26): their kernels round activations differently. So: identical -> ok; otherwise the
            // repacked kernel must still be no more than ACC_SLACK worse than base vs the exact f32 product.
            double rk = rel(yk, yb);
            double ek = rel(yk, ref);
            bool same = std::isfinite(rk) && rk < TOL;
            bool kok  = same || (std::isfinite(ek) && ek <= ACC_SLACK * eb);
            checked++; fails += !kok;
            printf("%-12s   kernel check (repacked base bytes) vs base rel %.3e (%s) err vs f32 %.3e ratio %.3f  %s\n", "", rk,
                   same ? "identical" : "activation path differs", ek, ek / eb, kok ? "ok" : "FAIL");
        }
        fflush(stdout);
    }
    // negative control: two different base types must NOT agree to TOL
    double ctl = rel(mm(type_by_name("q4_0"), W, X, n, nrows, ncols), mm(type_by_name("q5_0"), W, X, n, nrows, ncols));
    printf("CONTROL q4_0 vs q5_0 rel diff %.3e  (must exceed %.0e: %s)\n", ctl, TOL, ctl > TOL ? "ok" : "CONTROL FAILED");
    // A failed control means "same quants" cannot be told from "different quants", so no verdict above stands.
    // It used to be printed and then ignored by the exit code. The same method now runs inside ctest as
    // test-mulmat-gemv's test_interleaved(), with its own control that also counts.
    if (!(ctl > TOL)) fails++;
    printf("%d checked, %d failed\n", checked, fails);
    return fails ? 1 : 0;
}
