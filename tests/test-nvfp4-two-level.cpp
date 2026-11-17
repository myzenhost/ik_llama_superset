// NVFP4 encoder: block-scale code search + per-tensor (two-level) scale.   CPU only.
// COMPATIBILITY-INVESTIGATED: pure numerics on the ggml type traits; no device, no model file.
// GATE-CLAIM-OK: documentation only.
//
// Rows: S69-A-PICK-NVFP4-TWO-LEVEL-SCALE-IN-LLAMA-QUANTIZE-EMIT-TENSOR-SCALE-2026-10-05-01 and
//       S69-BURN-PICK-FP4-BLOCK-SCALE-CODE-SEARCH-2026-09-29-01.
//
// What it proves (each CHECK has a fail-before partner in the same run):
//  A. the code search beats the reference encoder (quantize_row_nvfp4_ref, which is UNCHANGED and reached through
//     type_traits.from_float_ref): reference error > 0.0100 on N(0, 0.02^2), searched error <= 0.0090;
//  B. the search is never worse than the reference on any single 64-block, unweighted;
//  C. with an importance weight the weighted error is never above the unweighted-search weighted error;
//  D. two-level (pts = amax/(6*448), quantise x/pts, dequantise * pts) <= 0.0090 AND is unchanged when the tensor is
//     multiplied by 2^-6, whereas the one-level encoder flushes that tensor (error >= 0.5): the must-fail partner.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static double rel_sq_err(const std::vector<float> & x, const std::vector<float> & y, const float * w = nullptr, int64_t n_per_row = 0) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double ww = w ? (double) w[i % (size_t) n_per_row] : 1.0;
        const double d = (double) x[i] - (double) y[i];
        num += ww * d * d;
        den += ww * (double) x[i] * (double) x[i];
    }
    return num / den;
}

int main() {
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    if (!tr || !tr->to_float || !tr->from_float_ref) {
        fprintf(stderr, "NVFP4 traits missing\n");
        return 1;
    }
    const int64_t n_per_row = 4096;
    const int64_t nrows     = 256;
    const int64_t n         = n_per_row * nrows;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> x(n);
    for (auto & v : x) v = nd(rng);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_NVFP4, n_per_row);
    std::vector<uint8_t> q_ref(row_bytes * nrows), q_srch(row_bytes * nrows);
    std::vector<float>   y_ref(n), y_srch(n);

    // reference encoder, row by row (the unchanged nominal-code rule)
    for (int64_t r = 0; r < nrows; ++r) {
        tr->from_float_ref(x.data() + r*n_per_row, q_ref.data() + r*row_bytes, n_per_row);
        tr->to_float(q_ref.data() + r*row_bytes, y_ref.data() + r*n_per_row, n_per_row);
    }
    // searched encoder (what llama-quantize runs)
    ggml_quantize_chunk(GGML_TYPE_NVFP4, x.data(), q_srch.data(), 0, nrows, n_per_row, nullptr, nullptr);
    for (int64_t r = 0; r < nrows; ++r) {
        tr->to_float(q_srch.data() + r*row_bytes, y_srch.data() + r*n_per_row, n_per_row);
    }

    const double e_ref  = rel_sq_err(x, y_ref);
    const double e_srch = rel_sq_err(x, y_srch);
    printf("A  N(0,0.02)  reference %.5f   searched %.5f   (%.1f%% lower)\n", e_ref, e_srch, 100.0 * (1.0 - e_srch / e_ref));
    CHECK(e_ref  > 0.0100);          // fail-before partner: this IS the old behaviour
    CHECK(e_srch <= 0.0090);         // pass-after
    CHECK(e_srch < 0.90 * e_ref);

    // B. never worse than the reference on any 64-block (unweighted)
    int worse = 0;
    for (int64_t b = 0; b < n / 64; ++b) {
        double er = 0.0, es = 0.0;
        for (int j = 0; j < 64; ++j) {
            const double dr = (double) x[b*64 + j] - y_ref[b*64 + j];
            const double ds = (double) x[b*64 + j] - y_srch[b*64 + j];
            er += dr * dr;
            es += ds * ds;
        }
        // a 64-block holds four independent 16-sub-blocks, each minimised on its own, so the sum can not exceed the reference
        if (es > er * (1.0 + 1e-6)) ++worse;
    }
    printf("B  blocks where search > reference: %d of %lld\n", worse, (long long) (n / 64));
    CHECK(worse == 0);

    // C. importance weights that VARY INSIDE every 16- and 32-block (a weight that is constant over a block rescales its error and
    // cannot change which scale code wins, so such a vector would not test the weighting at all)
    std::vector<float> w(n_per_row);
    for (int64_t j = 0; j < n_per_row; ++j) w[j] = (j % 16) < 3 ? 200.0f : 1.0f;
    std::vector<uint8_t> q_w(row_bytes * nrows);
    std::vector<float>   y_w(n);
    ggml_quantize_chunk(GGML_TYPE_NVFP4, x.data(), q_w.data(), 0, nrows, n_per_row, w.data(), nullptr);
    for (int64_t r = 0; r < nrows; ++r) {
        tr->to_float(q_w.data() + r*row_bytes, y_w.data() + r*n_per_row, n_per_row);
    }
    const double ew_w  = rel_sq_err(x, y_w,    w.data(), n_per_row);
    const double ew_u  = rel_sq_err(x, y_srch, w.data(), n_per_row);
    printf("C  weighted error: with imatrix %.5f   without %.5f\n", ew_w, ew_u);
    CHECK(ew_w <= ew_u * (1.0 + 1e-9));
    CHECK(ew_w <  ew_u);                                // the weights must actually move the choice

    // D. two-level scale
    auto two_level = [&](const std::vector<float> & src, float & pts_out) {
        float amax = 0.0f;
        for (float v : src) amax = std::fmax(amax, std::fabs(v));
        const float pts = amax / (6.0f * 448.0f);
        pts_out = pts;
        std::vector<float> scaled(src.size());
        for (size_t i = 0; i < src.size(); ++i) scaled[i] = src[i] / pts;
        std::vector<uint8_t> q(row_bytes * nrows);
        ggml_quantize_chunk(GGML_TYPE_NVFP4, scaled.data(), q.data(), 0, nrows, n_per_row, nullptr, nullptr);
        std::vector<float> y(src.size());
        for (int64_t r = 0; r < nrows; ++r) {
            tr->to_float(q.data() + r*row_bytes, y.data() + r*n_per_row, n_per_row);
        }
        for (auto & v : y) v *= pts;
        return y;
    };
    auto one_level = [&](const std::vector<float> & src) {
        std::vector<uint8_t> q(row_bytes * nrows);
        ggml_quantize_chunk(GGML_TYPE_NVFP4, src.data(), q.data(), 0, nrows, n_per_row, nullptr, nullptr);
        std::vector<float> y(src.size());
        for (int64_t r = 0; r < nrows; ++r) {
            tr->to_float(q.data() + r*row_bytes, y.data() + r*n_per_row, n_per_row);
        }
        return y;
    };

    float pts1 = 0.0f, pts2 = 0.0f;
    const auto y2 = two_level(x, pts1);
    const double e_two = rel_sq_err(x, y2);

    std::vector<float> xs(n);
    for (int64_t i = 0; i < n; ++i) xs[i] = x[i] * (1.0f / 64.0f);       // a model with 64x smaller weights
    const auto y2s = two_level(xs, pts2);
    const double e_two_s = rel_sq_err(xs, y2s);
    const auto y1s = one_level(xs);
    const double e_one_s = rel_sq_err(xs, y1s);
    printf("D  two-level %.5f   two-level at 2^-6 %.5f (pts %.3g -> %.3g)   one-level at 2^-6 %.5f\n", e_two, e_two_s, pts1, pts2, e_one_s);
    CHECK(e_two   <= 0.0090);
    CHECK(std::fabs(e_two_s - e_two) <= 1e-4);       // scale-equivariant
    CHECK(e_one_s >= 0.5);                            // the must-fail partner: one-level flushes it
    CHECK(pts2 > 0.0f && std::fabs(pts2 * 64.0f / pts1 - 1.0f) < 1e-5f);

    // E. MXFP4 exponent search (S69-BURN-PICK-FP4-BLOCK-SCALE-CODE-SEARCH, MXFP4 half). The from_float_ref entry keeps the OCP floor
    // rule (the old behaviour, the fail-before side); ggml_quantize_chunk is the offline quantiser and searches e-1, e, e+1.
    {
        const ggml_type_traits * tm = ggml_get_type_traits(GGML_TYPE_MXFP4);
        CHECK(tm && tm->to_float && tm->from_float_ref);
        if (tm && tm->to_float && tm->from_float_ref) {
            const size_t mrow = ggml_row_size(GGML_TYPE_MXFP4, n_per_row);
            std::vector<uint8_t> qa(mrow * nrows), qb(mrow * nrows), qc(mrow * nrows);
            std::vector<float>   ya(n), yb(n), yc(n);
            for (int64_t r = 0; r < nrows; ++r) {
                tm->from_float_ref(x.data() + r*n_per_row, qa.data() + r*mrow, n_per_row);
                tm->to_float(qa.data() + r*mrow, ya.data() + r*n_per_row, n_per_row);
            }
            ggml_quantize_chunk(GGML_TYPE_MXFP4, x.data(), qb.data(), 0, nrows, n_per_row, nullptr, nullptr);
            for (int64_t r = 0; r < nrows; ++r) {
                tm->to_float(qb.data() + r*mrow, yb.data() + r*n_per_row, n_per_row);
            }
            const double m_ref = rel_sq_err(x, ya);
            const double m_srch = rel_sq_err(x, yb);
            printf("E  MXFP4 N(0,0.02) reference %.5f   searched %.5f   (%.1f%% lower)\n", m_ref, m_srch, 100.0 * (1.0 - m_srch / m_ref));
            CHECK(m_srch < m_ref);                         // fail-before: the reference rule is strictly worse on the same data
            int mworse = 0;
            for (int64_t b = 0; b < n / 32; ++b) {
                double er = 0.0, es = 0.0;
                for (int j = 0; j < 32; ++j) {
                    const double dr = (double) x[b*32 + j] - ya[b*32 + j];
                    const double ds = (double) x[b*32 + j] - yb[b*32 + j];
                    er += dr * dr;
                    es += ds * ds;
                }
                if (es > er * (1.0 + 1e-6)) ++mworse;
            }
            printf("E  MXFP4 blocks where search > reference: %d of %lld\n", mworse, (long long) (n / 32));
            CHECK(mworse == 0);
            ggml_quantize_chunk(GGML_TYPE_MXFP4, x.data(), qc.data(), 0, nrows, n_per_row, w.data(), nullptr);
            for (int64_t r = 0; r < nrows; ++r) {
                tm->to_float(qc.data() + r*mrow, yc.data() + r*n_per_row, n_per_row);
            }
            const double mw_w = rel_sq_err(x, yc, w.data(), n_per_row);
            const double mw_u = rel_sq_err(x, yb, w.data(), n_per_row);
            printf("E  MXFP4 weighted error: with imatrix %.5f   without %.5f\n", mw_w, mw_u);
            CHECK(mw_w <= mw_u * (1.0 + 1e-9));
            CHECK(mw_w <  mw_u);
        }
    }

    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-nvfp4-two-level: all checks passed\n");
    return 0;
}
