// IQ5_K / IQ5_KS bulk quantizer: AVX2 fast path (ik_llama #2608) vs the scalar reference.
//
// Contracts checked (row S69-DECL-OVERTURN-IQ5K-AVX2-QUANT-2026-10-17-01):
//   DISPATCH  ggml_quantize_chunk(IQ5_K / IQ5_KS) runs the fast path when the CPU backend was built with AVX2: the
//             iqk_quantize_fast_path_rows() counter advances by exactly nrows. With GGML_IQK_QUANT_REFERENCE=1 it must not
//             advance, and the bytes must equal type_traits from_float_ref (the scalar reference) row for row.
//             Without AVX2 the counter never advances and both runs are the reference.
//             IQ5_K_R4 / IQ5_KS_R4 are held to the same three contracts (their rows are quantized by the same row search,
//             then repacked); they run on the largest multiple of 4 rows, and from_float_ref / to_float take 4 rows a call.
//   QUALITY   the fast search is a different search (by design), so bytes differ; its weighted RMSE after a to_float
//             round trip must stay within 1% of the reference's: RMSE(fast) <= 1.01 * RMSE(reference), per type, with and
//             without an importance matrix. Inputs: the test-iqk-golden row (cos/sin, 4096) + 64 pseudo-random rows of
//             4096 (Gaussian with a few 8-sigma outliers), deterministic, no <random> distributions.
//   TIMING    printed (t_ref / t_fast), never asserted: the host is shared and thermally limited.
//             BENCH-PENDING: the measured ratio is recorded in the row, not here. --bench <rows> <cols> <reps> times larger inputs.
//
// Exit code: 0 all pass, 1 a finding.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" int64_t iqk_quantize_fast_path_rows(void);

static void set_reference(bool on) {
#ifdef _WIN32
    _putenv_s("GGML_IQK_QUANT_REFERENCE", on ? "1" : "0");
#else
    if (on) setenv("GGML_IQK_QUANT_REFERENCE", "1", 1); else unsetenv("GGML_IQK_QUANT_REFERENCE");
#endif
}

struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return (uint32_t)(s >> 33); }
    float uniform() { return (next() + 0.5f) / 2147483648.0f; } // (0,1)
    float gauss() {
        float u1 = uniform(), u2 = uniform();
        return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    }
};

static void make_data(int64_t nrows, int64_t ncols, std::vector<float> & x, uint64_t seed) {
    x.resize(nrows * ncols);
    // row 0: the test-iqk-golden input pattern
    for (int64_t i = 0; i < ncols; ++i) x[i] = 0.1f + 2.0f * cosf((float) i) + 0.37f * sinf(0.5f * (float) i);
    Lcg rng(seed);
    for (int64_t r = 1; r < nrows; ++r) {
        float scale = 0.01f + 0.05f * rng.uniform();
        for (int64_t i = 0; i < ncols; ++i) {
            float v = rng.gauss() * scale;
            if ((rng.next() & 1023) == 0) v *= 8.0f; // occasional outlier
            x[r * ncols + i] = v;
        }
    }
}

static void make_imatrix(int64_t ncols, std::vector<float> & w, uint64_t seed) {
    w.resize(ncols);
    Lcg rng(seed);
    for (int64_t i = 0; i < ncols; ++i) {
        float u = rng.uniform();
        w[i] = (rng.next() % 10 == 0) ? 0.0f : u * u * 4.0f;
    }
}

// Rows per interleaved group: the _R4 types store 4 rows together, and their to_float / from_float_ref take k = 4*ncols
// (4 rows of ncols) at a time. Every other type here is 1.
static int64_t rows_per_group(ggml_type type) {
    return (type == GGML_TYPE_IQ5_K_R4 || type == GGML_TYPE_IQ5_KS_R4) ? 4 : 1;
}

static double weighted_rmse(ggml_type type, const std::vector<float> & x, const std::vector<uint8_t> & q,
                            int64_t nrows, int64_t ncols, const float * imat) {
    const auto * tr = ggml_get_type_traits(type);
    const size_t row_size = ggml_row_size(type, ncols);
    const int64_t g = rows_per_group(type);
    std::vector<float> y(g * ncols);
    double num = 0, den = 0;
    for (int64_t r = 0; r < nrows; r += g) {
        tr->to_float(q.data() + r * row_size, y.data(), g * ncols);
        for (int64_t i = 0; i < g * ncols; ++i) {
            double w = imat ? imat[i % ncols] : 1.0;
            double d = (double) x[r * ncols + i] - (double) y[i];
            num += w * d * d;
            den += w;
        }
    }
    return den > 0 ? sqrt(num / den) : 0.0;
}

static double quantize_timed(ggml_type type, const std::vector<float> & x, std::vector<uint8_t> & q, int64_t nrows,
                             int64_t ncols, const float * imat, int reps) {
    q.assign(nrows * ggml_row_size(type, ncols), 0);
    double best = 1e30;
    for (int rep = 0; rep < reps; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        size_t n = ggml_quantize_chunk(type, x.data(), q.data(), 0, nrows, ncols, imat, nullptr);
        auto t1 = std::chrono::steady_clock::now();
        if (n != q.size()) { printf("  size mismatch %zu vs %zu\n", n, q.size()); return -1; }
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

int main(int argc, char ** argv) {
    printf("EXIT-CODE-SEMANTICS: findings=1\n");
    int64_t nrows = 65, ncols = 4096;
    int reps = 1;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0 && i + 3 < argc) {
            nrows = atoll(argv[i + 1]); ncols = atoll(argv[i + 2]); reps = atoi(argv[i + 3]); bench = true; i += 3;
        }
    }
    ggml_cpu_init();
    const bool has_avx2 = ggml_cpu_has_avx2() != 0;
    printf("avx2=%d rows=%lld cols=%lld reps=%d\n", (int) has_avx2, (long long) nrows, (long long) ncols, reps);

    std::vector<float> x, imat;
    make_data(nrows, ncols, x, 0x5eed1017ULL);
    make_imatrix(ncols, imat, 0x1a7e1017ULL);

    int findings = 0;
    // _R4: ggml_quantize_chunk quantizes each group of 4 rows with the IQ5_K / IQ5_KS row quantizer and then repacks, so
    // the same three contracts apply (row S69-IQ5K-R4-AVX2-QUANT-FASTPATH-2026-10-17-01). They need nrows % 4 == 0.
    const ggml_type types[] = { GGML_TYPE_IQ5_K, GGML_TYPE_IQ5_KS, GGML_TYPE_IQ5_K_R4, GGML_TYPE_IQ5_KS_R4 };
    for (ggml_type type : types) {
        const int64_t g = rows_per_group(type);
        const int64_t nrows_t = nrows - nrows % g;
        if (nrows_t <= 0) { printf("SKIPPED %s: needs at least %lld rows\n", ggml_type_name(type), (long long) g); continue; }
        for (int use_imat = 0; use_imat < 2; ++use_imat) {
            const float * im = use_imat ? imat.data() : nullptr;
            const char * tn = ggml_type_name(type);
            std::vector<uint8_t> q_ref, q_fast;

            set_reference(true);
            int64_t c0 = iqk_quantize_fast_path_rows();
            double t_ref = quantize_timed(type, x, q_ref, nrows_t, ncols, im, reps);
            int64_t c1 = iqk_quantize_fast_path_rows();

            set_reference(false);
            double t_fast = quantize_timed(type, x, q_fast, nrows_t, ncols, im, reps);
            int64_t c2 = iqk_quantize_fast_path_rows();

            if (t_ref < 0 || t_fast < 0) { ++findings; continue; }

            // dispatch
            if (c1 != c0) {
                printf("FINDING %s imat=%d: GGML_IQK_QUANT_REFERENCE=1 still ran the fast path (%lld rows)\n", tn, use_imat, (long long)(c1 - c0));
                ++findings;
            }
            const int64_t want = has_avx2 ? nrows_t * reps : 0;
            if (c2 - c1 != want) {
                printf("FINDING %s imat=%d: fast-path counter advanced %lld, expected %lld\n", tn, use_imat, (long long)(c2 - c1), (long long) want);
                ++findings;
            }
            // the reference run must be the from_float_ref bytes (no imatrix only: from_float_ref takes none)
            if (!use_imat) {
                const auto * tr = ggml_get_type_traits(type);
                const size_t row_size = ggml_row_size(type, ncols);
                std::vector<uint8_t> qr(g * row_size);
                int64_t bad = 0;
                for (int64_t r = 0; r < nrows_t; r += g) {
                    tr->from_float_ref(x.data() + r * ncols, qr.data(), g * ncols);
                    if (memcmp(qr.data(), q_ref.data() + r * row_size, g * row_size) != 0) bad += g;
                }
                if (bad) {
                    printf("FINDING %s: reference bulk run differs from from_float_ref in %lld of %lld rows\n", tn, (long long) bad, (long long) nrows_t);
                    ++findings;
                }
            }
            // quality
            double e_ref  = weighted_rmse(type, x, q_ref,  nrows_t, ncols, im);
            double e_fast = weighted_rmse(type, x, q_fast, nrows_t, ncols, im);
            size_t ndiff = 0;
            for (size_t i = 0; i < q_ref.size(); ++i) ndiff += q_ref[i] != q_fast[i];
            const bool ok = e_fast <= 1.01 * e_ref;
            printf("%-7s imat=%d  rmse_ref=%.6e rmse_fast=%.6e ratio=%.5f %s  bytes_differ=%zu/%zu  t_ref=%.4fs t_fast=%.4fs t_ref/t_fast=%.2f\n",
                   tn, use_imat, e_ref, e_fast, e_ref > 0 ? e_fast / e_ref : 0.0, ok ? "OK" : "FAIL",
                   ndiff, q_ref.size(), t_ref, t_fast, t_fast > 0 ? t_ref / t_fast : 0.0);
            if (!ok) ++findings;
        }
    }
    set_reference(false);
    printf("%s findings=%d%s\n", findings ? "FAIL" : "PASS", findings, bench ? " (bench)" : "");
    return findings ? 1 : 0;
}
