// Merge-owed item 9 of S69-ENGINE-MERGE-IK-LLAMA-MAIN-TRIAL-2026-09-29-01: ik's AVX2 scale search for the
// IQ3_K / IQ3_KS / IQ5_K / IQ5_KS / IQ6_K quantizers sat behind `#ifdef __AVX2__` in iqk_quantize_base.cpp, a
// ggml-base TU built with no arch flags, so it was preprocessed out and never ran. ggml-cpu now registers its
// find_best_scale_avx2 instantiations through iqk_scale_search.h from ggml_cpu_init() when
// GGML_IQK_AVX2_SCALE_SEARCH=1 (opt-in: see the measured PPL cost at the registration site in ggml-cpu.c).
//
// Two ctest entries run this binary:
//   default (env unset)         -> after ggml_cpu_init NOTHING is registered: the scalar search runs and
//                                  from_float_ref stays host-independent (test-iqk-golden depends on it);
//   --opt-in (env set to 1)     -> checks 1-4 below.
// Checks:
//   1. before ggml_cpu_init no search is registered (the scalar path is what ggml-base alone runs);
//   2. after ggml_cpu_init every kind is registered when the CPU backend has AVX2 (fail-before: all NULL);
//   3. per type, the registered search and the scalar one (hook cleared) both quantize the same data; the AVX2
//      result must be deterministic and its weighted round-trip error within 10% of the scalar one. 10%, not
//      less: ik's AVX2 IQ5_K / IQ6_K weigh by x^2 where the scalar search adds sigma2 (ik's own note: the
//      scalar-matching flags are "not really better"), measured worst +5.6% (iq6_k, no imatrix);
//   4. control: a deliberately wrong search (scale 0 everywhere) MUST be caught by the error comparison.
// A ternary-weight data set (TriLM-like, where the model-level PPL moved) is quantized too and its errors are
// printed, not asserted. Times of both paths are printed (min of 3), not asserted.
//
// IQ5_K / IQ5_KS and the bulk fast path (merge of this branch with batch-1017f, which carries a second port of
// the same ik #2608 search): ggml_quantize_chunk sends IQ5_K / IQ5_KS to quantize_iq5_k(s)_bulk, which runs its
// own AVX2 search by default and calls the reference quantizer -- the one this hook plugs into -- only when
// GGML_IQK_QUANT_REFERENCE=1 (or the build has no AVX2). from_float_ref always takes the reference. So for those
// two types the hook is reached on the reference path, and the checks above are only meaningful there: measured
// on the merge without this, both IQ5 rows printed "bytes IDENTICAL" and the zero-scale control was NOT caught,
// because neither run touched the hook. The opt-in run therefore forces the reference path for every case and
//   5. asserts the bulk fast path did not run (iqk_quantize_fast_path_rows unchanged), and, as the counter's own
//      control, that without GGML_IQK_QUANT_REFERENCE one IQ5_K chunk DOES advance it by its row count on AVX2.
// GATE-CLAIM-OK: unit test; timings printed, not claimed.
#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/iqk/iqk_scale_search.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int64_t iqk_quantize_fast_path_rows(void);

static void set_reference(bool on) {
#ifdef _WIN32
    _putenv_s("GGML_IQK_QUANT_REFERENCE", on ? "1" : "0");
#else
    if (on) setenv("GGML_IQK_QUANT_REFERENCE", "1", 1); else unsetenv("GGML_IQK_QUANT_REFERENCE");
#endif
}

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

struct tcase { ggml_type type; iqk_scale_search_kind kind; const char * name; };

static const tcase k_cases[] = {
    { GGML_TYPE_IQ3_K,  IQK_SCALE_SEARCH_16_127, "iq3_k"  },
    { GGML_TYPE_IQ3_KS, IQK_SCALE_SEARCH_32_127, "iq3_ks" },
    { GGML_TYPE_IQ5_K,  IQK_SCALE_SEARCH_16_240, "iq5_k"  },
    { GGML_TYPE_IQ5_KS, IQK_SCALE_SEARCH_32_240, "iq5_ks" },
    { GGML_TYPE_IQ6_K,  IQK_SCALE_SEARCH_16_248, "iq6_k"  },
};

static iqk_scale_search_result zero_search(int, const int8_t *, const float *, const int8_t *, const float *, const float *,
        float, int, bool, float, bool) {
    return { 0.0f, 0.0f, false };
}

struct qrun { std::vector<uint8_t> q; double err; double us; };

// quantize nrows x n, dequantize, return the (imatrix-)weighted relative squared error and min time of `reps`
static qrun run(ggml_type type, const std::vector<float> & x, const std::vector<float> * imat, int64_t nrows, int64_t n, int reps) {
    qrun r;
    const size_t rs = ggml_row_size(type, n);
    r.q.assign(rs*nrows, 0);
    r.us = 1e30;
    for (int k = 0; k < reps; k++) {
        std::fill(r.q.begin(), r.q.end(), 0);
        const int64_t t0 = ggml_time_us();
        ggml_quantize_chunk(type, x.data(), r.q.data(), 0, nrows, n, imat ? imat->data() : nullptr, nullptr);
        r.us = std::min(r.us, (double) (ggml_time_us() - t0));
    }
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    std::vector<float> y(n);
    double num = 0, den = 0;
    for (int64_t row = 0; row < nrows; row++) {
        tr->to_float(r.q.data() + row*rs, y.data(), n);
        for (int64_t j = 0; j < n; j++) {
            const double w = imat ? (*imat)[j] : 1.0;
            const double d = y[j] - x[row*n + j];
            num += w*d*d;
            den += w*(double) x[row*n + j]*x[row*n + j];
        }
    }
    r.err = den > 0 ? num/den : 0;
    return r;
}

int main(int argc, char ** argv) {
    const bool opt_in = argc > 1 && strcmp(argv[1], "--opt-in") == 0;
    const char * env = getenv("GGML_IQK_AVX2_SCALE_SEARCH");
    printf("mode %s, GGML_IQK_AVX2_SCALE_SEARCH=%s\n", opt_in ? "--opt-in" : "default", env ? env : "(unset)");

    ggml_free(ggml_init({ /*.mem_size =*/ 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true }));

    // 1. nothing registered before the CPU backend initialises
    for (const tcase & c : k_cases) {
        CHECK(iqk_get_scale_search(c.kind) == nullptr, "%s: search registered before ggml_cpu_init", c.name);
    }

    ggml_cpu_init();
    const bool avx2 = ggml_cpu_has_avx2() != 0;
    printf("ggml_cpu_has_avx2 = %d\n", (int) avx2);

    if (!opt_in) {
        // default: the scalar search stays in force
        CHECK(!(env && atoi(env) == 1), "default mode run with GGML_IQK_AVX2_SCALE_SEARCH=1 set");
        for (const tcase & c : k_cases) {
            CHECK(iqk_get_scale_search(c.kind) == nullptr, "%s: AVX2 search registered without the opt-in", c.name);
        }
        if (fails) {
            fprintf(stderr, "%d check(s) FAILED\n", fails);
            return 1;
        }
        printf("default: no scale search registered, scalar quantizers in force\n");
        return 0;
    }
    CHECK(env && atoi(env) == 1, "--opt-in needs GGML_IQK_AVX2_SCALE_SEARCH=1 in the environment");

    // 2. registered after init on an AVX2 CPU backend (before this change: never, every kind NULL)
    for (const tcase & c : k_cases) {
        if (avx2) {
            CHECK(iqk_get_scale_search(c.kind) != nullptr, "%s: AVX2 search not registered by ggml_cpu_init", c.name);
        } else {
            printf("%s: no AVX2 in this CPU backend, search = %p (scalar expected)\n", c.name, (void *) iqk_get_scale_search(c.kind));
        }
    }

    // data: gaussian rows with a few outliers, plus a positive imatrix
    const int64_t nrows = 64, n = 4096;
    std::mt19937 rng(1017);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.1f, 2.0f);
    std::vector<float> x(nrows*n), imat(n);
    for (auto & v : x) v = nd(rng);
    for (int64_t i = 0; i < nrows*n; i += 97) x[i] *= 6.0f;
    for (auto & v : imat) v = ud(rng);

    // ternary rows: {-a, 0, +a} with a per-row a, about 1/3 zeros (TriLM-like). Printed only.
    std::vector<float> xt(nrows*n);
    {
        std::uniform_int_distribution<int> td(-1, 1);
        for (int64_t r = 0; r < nrows; r++) {
            const float a = 0.02f + 0.001f*(float) r;
            for (int64_t j = 0; j < n; j++) xt[r*n + j] = a*(float) td(rng);
        }
    }

    // 5. the hook lives in the reference quantizers; keep ggml_quantize_chunk on them (see the header note)
    set_reference(true);
    const int64_t fast_rows_before = iqk_quantize_fast_path_rows();

    const double tol = 1.10;
    for (const tcase & c : k_cases) {
        const iqk_scale_search_fn fn = iqk_get_scale_search(c.kind);
        if (!fn) { printf("%s: skipped (no search registered)\n", c.name); continue; }
        for (int use_imat = 0; use_imat < 2; use_imat++) {
            const std::vector<float> * im = use_imat ? &imat : nullptr;

            iqk_set_scale_search(c.kind, nullptr);
            const qrun sc = run(c.type, x, im, nrows, n, 3);
            iqk_set_scale_search(c.kind, fn);
            const qrun av = run(c.type, x, im, nrows, n, 3);
            const qrun av2 = run(c.type, x, im, nrows, n, 1);

            const bool same = sc.q == av.q;
            printf("%-6s imatrix=%d  scalar err %.6e %8.0f us | avx2 err %.6e %8.0f us | time ratio scalar/avx2 %.2f | bytes %s\n",
                    c.name, use_imat, sc.err, sc.us, av.err, av.us, sc.us/av.us, same ? "IDENTICAL" : "differ");
            CHECK(av.q == av2.q, "%s: AVX2 search is not deterministic", c.name);
            CHECK(std::isfinite(av.err) && av.err > 0, "%s: AVX2 error %g", c.name, av.err);
            CHECK(av.err <= sc.err*tol, "%s imatrix=%d: AVX2 error %.6e worse than scalar %.6e by more than 10%%",
                    c.name, use_imat, av.err, sc.err);

            // 4. control: a broken search must be caught by the same comparison
            if (use_imat == 0) {
                iqk_set_scale_search(c.kind, zero_search);
                const qrun bad = run(c.type, x, im, nrows, n, 1);
                iqk_set_scale_search(c.kind, fn);
                const bool caught = !(bad.err <= sc.err*tol);
                printf("%-6s control: zero-scale search err %.6e -> %s\n", c.name, bad.err, caught ? "caught" : "NOT caught");
                CHECK(caught, "%s: the error comparison does not catch a broken search", c.name);
            }
        }
        {
            iqk_set_scale_search(c.kind, nullptr);
            const qrun sc = run(c.type, xt, nullptr, nrows, n, 1);
            iqk_set_scale_search(c.kind, fn);
            const qrun av = run(c.type, xt, nullptr, nrows, n, 1);
            printf("%-6s ternary    scalar err %.6e | avx2 err %.6e | avx2/scalar %.3f (printed, not asserted)\n",
                    c.name, sc.err, av.err, sc.err > 0 ? av.err/sc.err : 0.0);
        }
    }

    {
        const int64_t fast_rows_after = iqk_quantize_fast_path_rows();
        printf("bulk fast path rows during the reference-path checks: %lld (must be 0)\n",
                (long long) (fast_rows_after - fast_rows_before));
        CHECK(fast_rows_after == fast_rows_before, "the bulk fast path ran %lld rows with GGML_IQK_QUANT_REFERENCE=1",
                (long long) (fast_rows_after - fast_rows_before));
        // control for the counter itself: the default path must move it on an AVX2 build
        set_reference(false);
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_IQ5_K, n)*nrows);
        const int64_t c0 = iqk_quantize_fast_path_rows();
        ggml_quantize_chunk(GGML_TYPE_IQ5_K, x.data(), q.data(), 0, nrows, n, nullptr, nullptr);
        const int64_t moved = iqk_quantize_fast_path_rows() - c0;
        printf("control: default-path IQ5_K chunk moved the fast-path counter by %lld rows (AVX2 expects %lld)\n",
                (long long) moved, (long long) (avx2 ? nrows : 0));
        if (avx2) {
            CHECK(moved == nrows, "control: the fast-path counter did not see a default IQ5_K chunk (%lld rows)", (long long) moved);
        }
    }

    if (fails) {
        fprintf(stderr, "%d check(s) FAILED\n", fails);
        return 1;
    }
    printf("all IQK scale-search checks passed\n");
    return 0;
}
