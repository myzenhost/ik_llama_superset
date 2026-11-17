// Model-free tests for TEAL training-free activation sparsity (S69-TEAL-STAGE1, arXiv:2408.14690
// Sec 4.2 Definition 1): stp(x)_i = 0 if |x_i| <= tp else x_i, applied to the INPUT of a linear
// projection, tp = the s-quantile of |x|.
//
// WHAT THIS COVERS (the end-to-end model gates live in tests/test-teal-sparsity-e2e.py):
//   1. the mask op itself (teal_apply_mask, src/llama-build-context.cpp) against a plain scalar
//      reference, INCLUDING the boundary |x| == tp (Definition 1 says "<=" -> zeroed; ggml_step is
//      strict ">" on |x|-tp, so the two must agree at the boundary or the helper has the wrong
//      sign convention);
//   2. the calibration contract: the threshold tools/cats-calibrate writes is v[k] with
//      k = floor(s*n) of the sorted |x| (cats-calibrate.cpp, nth_element), so masking with it must
//      zero EXACTLY k+1 elements of a tie-free sample;
//   3. the command-line flag pair --teal-sparsity / --teal-thresholds: default off, parses, rejects
//      s outside [0,1), and reaches llama_context_params through common_context_params_to_llama;
//   4. the library default of the three C-API fields is OFF (the bit-identity contract: a caller
//      that never touches them must get the pre-TEAL graph).
//
// CONTROLS (a comparison that cannot fail proves nothing):
//   * a NEGATIVE control perturbs the threshold by 10% and asserts the exact-count check then FAILS;
//   * inputs are non-trivial (mixed signs, a wide range, exact boundary values), so a helper that
//     returns its input, zero, or masks on x instead of |x| is caught.
//
// FAIL-BEFORE: this file cannot be built against the pre-TEAL tree at all (no teal_apply_mask
// symbol, no teal_* members in common_params / llama_context_params) -- that compile error is the
// "before" run; the run recorded in the commit message is the "after".

#include "ggml.h"
#include "ggml-cpu.h"
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Defined in src/llama-build-context.cpp, declared in src/llama-build-context.h (non-static since
// 2026-10-01 so per-arch graph files can call it). Declared here instead of including that header
// to avoid dragging the whole graph-builder include set into a model-free test.
struct ggml_tensor * teal_apply_mask(struct ggml_context * ctx, struct ggml_tensor * a, float threshold);

static int g_failures = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            printf("  FAIL %s:%d  %s  --  ", __FILE__, __LINE__, #cond);      \
            printf(__VA_ARGS__);                                              \
            printf("\n");                                                     \
            g_failures++;                                                     \
        }                                                                     \
    } while (0)

static std::vector<float> run_mask(const std::vector<float> & x, float t) {
    std::vector<uint8_t> buf(64u * 1024u * 1024u);
    ggml_init_params p = { buf.size(), buf.data(), false };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) x.size());
    std::memcpy(a->data, x.data(), x.size() * sizeof(float));
    ggml_tensor * m = teal_apply_mask(ctx, a, t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, m);
    ggml_graph_compute_with_ctx(ctx, gf, 2);
    std::vector<float> out((const float *) m->data, (const float *) m->data + x.size());
    ggml_free(ctx);
    return out;
}

static size_t count_zeros(const std::vector<float> & v) {
    size_t n = 0;
    for (float f : v) {
        if (f == 0.0f) ++n;
    }
    return n;
}

static void test_mask_vs_scalar_reference() {
    printf("test_mask_vs_scalar_reference\n");
    std::mt19937 rng(1234);
    std::normal_distribution<float> d(0.0f, 1.5f);
    std::vector<float> x(4096);
    for (auto & v : x) v = d(rng);

    const float t = 0.7f;
    // exact boundary values: Definition 1 zeroes |x| <= tp, keeps |x| > tp
    x[0] = t;                                       // == t   -> must be ZEROED
    x[1] = -t;                                      // == -t  -> must be ZEROED
    x[2] = std::nextafter(t, 10.0f);                // just above -> KEPT
    x[3] = -std::nextafter(t, 10.0f);               // just above (neg) -> KEPT
    x[4] = 0.0f;                                    // zero stays zero
    x[5] = std::nextafter(t, 0.0f);                 // just below -> ZEROED
    x[6] = -9.5f;                                   // large negative -> KEPT (catches masking on x not |x|)

    const std::vector<float> got = run_mask(x, t);
    size_t mism = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const float ref = std::fabs(x[i]) > t ? x[i] : 0.0f;
        if (got[i] != ref) {
            if (mism < 5) printf("    mismatch i=%zu x=%.9g got=%.9g ref=%.9g\n", i, x[i], got[i], ref);
            ++mism;
        }
    }
    CHECK(mism == 0, "%zu elements differ from the scalar reference", mism);
    CHECK(got[0] == 0.0f && got[1] == 0.0f, "boundary |x| == tp must be zeroed (Definition 1 '<=')");
    CHECK(got[2] == x[2] && got[3] == x[3], "just-above-boundary values must be kept bit-exact");
    CHECK(got[6] == -9.5f, "large negative value must be kept (mask must use |x|)");
    printf("  zeros=%zu / %zu\n", count_zeros(got), got.size());
}

static void test_calibration_contract_and_negative_control() {
    printf("test_calibration_contract_and_negative_control\n");
    std::mt19937 rng(99);
    std::normal_distribution<float> d(0.0f, 2.0f);
    std::vector<float> x(10000);
    for (auto & v : x) v = d(rng);
    std::vector<float> ax(x.size());
    for (size_t i = 0; i < x.size(); ++i) ax[i] = std::fabs(x[i]);

    for (double s : {0.25, 0.40, 0.50}) {
        // exactly what tools/cats-calibrate does: k = floor(s*n); threshold = k-th order statistic
        std::vector<float> v = ax;
        size_t k = (size_t) (s * (double) v.size());
        std::nth_element(v.begin(), v.begin() + k, v.end());
        const float thr = v[k];

        const std::vector<float> got = run_mask(x, thr);
        const size_t zeros = count_zeros(got);
        printf("  s=%.2f k=%zu thr=%.6f zeros=%zu (expected k+1 = %zu)\n", s, k, thr, zeros, k + 1);
        CHECK(zeros == k + 1, "s=%.2f: masking with the calibrated threshold must zero exactly k+1 elements", s);

        // NEGATIVE CONTROL: a 10%% perturbed threshold must NOT satisfy the same exact-count check.
        const std::vector<float> bad = run_mask(x, thr * 1.10f);
        CHECK(count_zeros(bad) != k + 1, "s=%.2f: negative control did not change the zero count -- the check cannot fail", s);
    }
}

static void test_flags_and_defaults() {
    printf("test_flags_and_defaults\n");
    {
        common_params dflt;
        CHECK(dflt.teal_sparsity == 0.0f, "common_params default teal_sparsity must be 0 (off)");
        CHECK(dflt.teal_thresholds_file.empty(), "common_params default teal_thresholds_file must be empty");
        CHECK(!dflt.teal_calibrate, "common_params default teal_calibrate must be false");
    }
    {
        llama_context_params cp = llama_context_default_params();
        CHECK(cp.teal_sparsity == 0.0f, "llama_context_default_params teal_sparsity must be 0 (bit-identity contract)");
        CHECK(cp.teal_thresholds_file == nullptr, "llama_context_default_params teal_thresholds_file must be NULL");
        CHECK(!cp.teal_calibrate, "llama_context_default_params teal_calibrate must be false");
    }

    auto parse = [](std::vector<std::string> args, common_params & params) {
        std::vector<char *> argv;
        for (auto & a : args) argv.push_back(const_cast<char *>(a.c_str()));
        return common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON);
    };

    {
        common_params p;
        const bool ok = parse({"prog", "-m", "dummy.gguf", "--teal-sparsity", "0.4", "--teal-thresholds", "some_thresholds.json"}, p);
        CHECK(ok, "valid --teal-sparsity 0.4 --teal-thresholds must parse");
        CHECK(std::fabs(p.teal_sparsity - 0.4f) < 1e-7f, "teal_sparsity not set (got %g)", p.teal_sparsity);
        CHECK(p.teal_thresholds_file == "some_thresholds.json", "teal_thresholds_file not set (got '%s')", p.teal_thresholds_file.c_str());

        llama_context_params cp = common_context_params_to_llama(p);
        CHECK(std::fabs(cp.teal_sparsity - 0.4f) < 1e-7f, "teal_sparsity did not reach llama_context_params (got %g)", cp.teal_sparsity);
        CHECK(cp.teal_thresholds_file != nullptr && std::string(cp.teal_thresholds_file) == "some_thresholds.json",
              "teal_thresholds_file did not reach llama_context_params");
    }
    {
        // CONTROL for the two rejections below: the same command line WITHOUT the bad value parses, so a
        // rejection can only come from the --teal-sparsity range check, not from a missing -m.
        common_params p;
        CHECK(parse({"prog", "-m", "dummy.gguf"}, p), "control: plain '-m dummy.gguf' must parse");
        CHECK(p.teal_sparsity == 0.0f && p.teal_thresholds_file.empty(), "control: no TEAL flags => off");
    }
    {
        common_params p;
        CHECK(!parse({"prog", "-m", "dummy.gguf", "--teal-sparsity", "1.0"}, p), "--teal-sparsity 1.0 must be rejected (range is [0,1))");
    }
    {
        common_params p;
        CHECK(!parse({"prog", "-m", "dummy.gguf", "--teal-sparsity", "-0.1"}, p), "--teal-sparsity -0.1 must be rejected");
    }
    {
        common_params p;
        const bool ok = parse({"prog", "-m", "dummy.gguf", "--teal-sparsity", "0"}, p);
        CHECK(ok && p.teal_sparsity == 0.0f, "--teal-sparsity 0 is the explicit off state and must parse");
    }
}

int main() {
    test_mask_vs_scalar_reference();
    test_calibration_contract_and_negative_control();
    test_flags_and_defaults();
    if (g_failures) {
        printf("test-teal-sparsity: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    printf("test-teal-sparsity: all passed\n");
    return 0;
}
