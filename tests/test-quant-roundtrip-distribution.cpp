// Does the KT family's round-trip deficit come from the QUANTISER, or from the harness feeding it
// a distribution its codebook was never fitted to?
//
// INVENT-LADDER-OK: nothing is declared impossible or blocked here. No existing test in this tree
// answers this question -- test-mulmat-gemv measures one distribution and test-iqk-golden measures
// storage -- so the missing instrument is BUILT here rather than named as a limitation. Every
// outcome below leads to work, including the outcome that says the threshold must not move.
//
// THE QUESTION THIS FILE EXISTS TO SETTLE. test-mulmat-gemv rejects iq1_kt, iq2_kt and iq3_kt on a
// dequantisation round-trip control, so their CPU kernels get no verdict at all -- not a pass, not
// a fail. Before that threshold is touched in either direction, one alternative has to be excluded:
//
//   * the KT types are TRELLIS-quantised. A trellis codebook is fitted to an assumed weight
//     distribution, and real model weights are roughly Gaussian.
//   * test-mulmat-gemv generates weights with std::uniform_real_distribution (line 156).
//   * a Gaussian-fitted trellis on UNIFORM data is that codebook's worst case, while a round
//     quantiser (q*_K, iq*_K, iq*_KS) is distribution-agnostic by construction -- it scales to the
//     block's own range and rounds, so it does not care what shape the data has.
//
// If that is the whole story, the KT deficit must SHRINK or INVERT under a Gaussian, and the
// threshold question dissolves: the reference would be fine and the harness input would be the
// defect. If the deficit SURVIVES a Gaussian, the distribution explanation is refuted and the
// deficit is a property of the quantiser at every distribution tried.
//
// Either way the answer is measured here rather than argued, and the threshold is not touched
// until this file has spoken. Widening a bound to make a red line go green is how a real defect
// becomes permanent, which is why the row refuses that fix by name.
//
// WHAT WOULD MAKE THIS FILE LIE, and what is done about each:
//   * a type that dequantises to all zeros scores a relative error of exactly 1.0 and would read
//     as "merely coarse" rather than "produced nothing". Guarded: a NO SIGNAL verdict is printed
//     and counted as a failure, never as a datum.
//   * a metric that cannot see disagreement would report every type as fine. Guarded by a
//     NEGATIVE control: the same weights compared against an independent draw must score near
//     sqrt(2), and if it does not, this file's numbers mean nothing.
//   * a type whose row carries metadata must be dequantised ROW-WISE. test-mulmat-gemv got this
//     wrong once and reported all 10 such types as broken references; the same row-wise loop is
//     used here for the same reason.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_failures = 0;

enum dist_kind { DIST_UNIFORM = 0, DIST_GAUSSIAN = 1, DIST_LAPLACE = 2, DIST_COUNT = 3 };

static const char * dist_name(int d) {
    switch (d) {
        case DIST_UNIFORM:  return "uniform";
        case DIST_GAUSSIAN: return "gaussian";
        case DIST_LAPLACE:  return "laplace";
    }
    return "?";
}

// Three distributions, not two, so the result reads as a TREND rather than a coincidence of one
// pair. Laplace is included because real transformer weights have tails heavier than a Gaussian,
// so it brackets the truth from the other side: uniform has no tails at all, Laplace has more than
// a Gaussian, and if the KT deficit is distribution-driven it must move monotonically across them.
//
// All three are normalised to the SAME RMS afterwards. The metric is relative so this does not
// change any number, but it removes the one objection that could be raised without measuring --
// that the distributions were compared at different scales.
static double g_rms = 0.5;   // set by the magnitude experiment below

static void fill_dist(std::vector<float> & v, std::mt19937 & rng, int kind) {
    if (kind == DIST_UNIFORM) {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto & x : v) x = d(rng);
    } else if (kind == DIST_GAUSSIAN) {
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (auto & x : v) x = d(rng);
    } else {
        // Laplace by inverse transform, which needs no library support.
        std::uniform_real_distribution<float> u(-0.5f, 0.5f);
        for (auto & x : v) {
            const float t = u(rng);
            x = (t < 0.0f ? 1.0f : -1.0f) * std::log(1.0f - 2.0f * std::fabs(t) + 1e-12f);
        }
    }
    double s = 0.0;
    for (auto x : v) s += (double) x * (double) x;
    const double rms = std::sqrt(s / (double) std::max<size_t>(1, v.size()));
    const double k = (rms > 0.0) ? (g_rms / rms) : 1.0;
    for (auto & x : v) x = (float) ((double) x * k);
}

static double rel_rms(const std::vector<float> & a, const std::vector<float> & b) {
    double se = 0.0, sw = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i])) return INFINITY;
        const double d = (double) a[i] - (double) b[i];
        se += d * d;
        sw += (double) b[i] * (double) b[i];
    }
    return std::sqrt(se / std::max(sw, 1e-30));
}

struct result { double err[DIST_COUNT]; bool ok; };

static result measure(ggml_type wtype, int64_t K, int64_t M) {
    result r;
    r.ok = true;
    for (int d = 0; d < DIST_COUNT; d++) r.err[d] = INFINITY;

    const auto * tr = ggml_get_type_traits(wtype);
    if (!tr || tr->to_float == NULL) {
        r.ok = false;
        return r;
    }

    for (int d = 0; d < DIST_COUNT; d++) {
        // The SAME seed for every distribution and every type, so a difference between two cells
        // of the table is a difference in the quantiser or the distribution and never in the draw.
        std::mt19937 rng(20260917);
        std::vector<float> Wf((size_t) K * M);
        fill_dist(Wf, rng, d);

        std::vector<uint8_t> q((size_t) ggml_row_size(wtype, K) * M + 1024);
        if (wtype == GGML_TYPE_F16) {
            // f16 is NOT a block-quantised format and ggml_quantize_chunk is not its entry point;
            // it has a from_float in the traits and nothing else. Routing it through the chunk API
            // produced an all-zero buffer, which the NO SIGNAL guard caught on the first run --
            // the guard's first catch was this file's OWN bug, which is the right order for a
            // control to fail in.
            ggml_get_type_traits(wtype)->from_float(Wf.data(), q.data(), K * M);
        } else {
            ggml_quantize_chunk(wtype, Wf.data(), q.data(), 0, M, K, nullptr, nullptr);
        }

        // ROW-WISE, because ggml_row_size includes row_meta_size and 10 types in this tree carry
        // 2 or 4 bytes of it. A single to_float over K*M applies row 0's metadata to every row.
        std::vector<float> Wdq((size_t) K * M, 0.0f);
        const size_t row_bytes = ggml_row_size(wtype, K);
        for (int64_t i = 0; i < M; i++) {
            tr->to_float((const char *) q.data() + (size_t) i * row_bytes,
                         Wdq.data() + (size_t) i * K, K);
        }

        // NO SIGNAL guard. An all-zero dequantisation scores exactly 1.0 by the relative metric,
        // which is inside the range a genuinely coarse 1-bit format could occupy. Without this it
        // would be tabulated as a number rather than as the absence of one.
        double amax = 0.0;
        for (size_t i = 0; i < Wdq.size(); i++) amax = std::max(amax, (double) std::fabs(Wdq[i]));
        if (!(amax > 0.0)) {
            printf("  ---- %s / %s: NO SIGNAL -- every dequantised value is zero. This is not a\n"
                   "       round-trip error, it is the absence of a round-trip, and it is counted\n"
                   "       as a failure rather than tabulated as 1.0.\n",
                   ggml_type_name(wtype), dist_name(d));
            g_failures++;
            r.ok = false;
            continue;
        }
        r.err[d] = rel_rms(Wdq, Wf);
    }
    return r;
}

struct pair_row { ggml_type kt; ggml_type sib; const char * why; };

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);   // a crash must not hide behind a block-buffered tail

    const int64_t K = 512;
    const int64_t M = 64;

    printf("SCOPE: relative-RMS quantise/dequantise round-trip, K=%lld M=%lld, host CPU, one seed\n"
           "(20260917) shared by every cell so no difference here is a difference in the draw.\n"
           "Three weight distributions, all normalised to the same RMS. This file measures the\n"
           "QUANTISER and its dequantiser ONLY -- no kernel, no GEMM, no vec_dot is involved.\n\n",
           (long long) K, (long long) M);

    // ============================================================ EXPERIMENT 0, and it runs FIRST
    // This has to be the first thing in main(), because its subject is the state of the process
    // BEFORE ggml_init has ever been called. Anything else printed above it would destroy it.
    //
    // HOW THIS QUESTION AROSE. An earlier run of this file reported iq2_ks and iq3_ks dequantising
    // to ALL ZEROS at every M from 64 to 512, while test-mulmat-gemv reports those same types at
    // the same K with healthy round-trips (0.2537 and 0.1281). Two call sites, one type, opposite
    // answers -- so one of them was lying and it was worth finding out which.
    //
    // The two obvious differences were tested first and BOTH were refuted: weight magnitude
    // (-0.1..0.1 there, RMS 0.5 here) and destination buffer (a ggml tensor there, a std::vector
    // here). A 2x2 of those showed signal in all four cells, so neither was the cause.
    //
    // What was left is ORDER. The grid-based IQ quantisers read a lookup table that ggml builds
    // lazily; the trellis KT quantisers compute their values instead and read no such table. If
    // that is the mechanism, then before any ggml_init:
    //     PREDICTION  iq2_ks (grid)    -> ZERO
    //     PREDICTION  iq2_kt (trellis) -> signal          <- the control that makes it a test
    // and after a ggml_init both must produce signal. A result where BOTH are zero, or both are
    // fine, refutes it. The prediction is written here before the run so it cannot be fitted after.
    printf("EXPERIMENT 0 -- does a grid-based quantiser silently emit ZEROS before ggml_init?\n");
    {
        auto probe_one = [&](ggml_type t) -> bool {
            const auto * tr = ggml_get_type_traits(t);
            if (!tr || !tr->to_float) return false;
            std::mt19937 rng(20260917);
            std::vector<float> Wf((size_t) K * 192);
            fill_dist(Wf, rng, DIST_GAUSSIAN);
            const size_t rb = ggml_row_size(t, K);
            std::vector<uint8_t> q(rb * 192 + 1024);
            ggml_quantize_chunk(t, Wf.data(), q.data(), 0, 192, K, nullptr, nullptr);
            std::vector<float> Wdq((size_t) K * 192, 0.0f);
            for (int64_t i = 0; i < 192; i++) {
                tr->to_float((const char *) q.data() + (size_t) i * rb,
                             Wdq.data() + (size_t) i * K, K);
            }
            double amax = 0.0;
            for (float v : Wdq) amax = std::max(amax, (double) std::fabs(v));
            return amax > 0.0;
        };

        const bool ks_before = probe_one(GGML_TYPE_IQ2_KS);
        const bool kt_before = probe_one(GGML_TYPE_IQ2_KT);

        ggml_init_params ip = { 16u * 1024u * 1024u, NULL, false };
        ggml_context * c = ggml_init(ip);

        const bool ks_after = probe_one(GGML_TYPE_IQ2_KS);
        const bool kt_after = probe_one(GGML_TYPE_IQ2_KT);
        ggml_free(c);

        printf("   iq2_ks  (grid-based)  before ggml_init: %-7s   after: %-7s\n",
               ks_before ? "signal" : "ZERO", ks_after ? "signal" : "ZERO");
        printf("   iq2_kt  (trellis, CONTROL) before      : %-7s   after: %-7s\n",
               kt_before ? "signal" : "ZERO", kt_after ? "signal" : "ZERO");

        // NOW A REGRESSION TEST (2026-09-26). The prediction was CONFIRMED on 2026-09-17 (iq2_ks ZERO before
        // ggml_init, signal after; the trellis control signal both times). The mechanism turned out to be
        // the fp16 -> fp32 LOOKUP TABLE, not a grid table: on x86 GGML_FP16_TO_FP32 reads
        // ggml_base_table_f32_f16, which only ggml_init filled, so every fp16 block scale read back as 0.
        // Fixed in ggml.c: ggml_quantize_init (called by ggml_quantize_chunk) now fills that table itself.
        // So the verdicts are inverted from the experiment's: zeros before init is the FAILURE now.
        if (ks_before && ks_after && kt_before && kt_after) {
            printf("   => FIXED (regression check): the grid-based type produces signal with NO ggml_init,\n"
                   "      and the trellis control is unchanged. Before the 2026-09-26 fix iq2_ks wrote\n"
                   "      ZEROS here, silently -- ggml_quantize_chunk returned a byte count and no error.\n");
        } else if (!ks_before && ks_after && kt_before && kt_after) {
            printf("   => REGRESSION. ggml_quantize_chunk on a grid-based IQ type SILENTLY WRITES ZEROS\n"
                   "      again when no ggml_init has run: the fp16 table is not being filled by\n"
                   "      ggml_quantize_init. The trellis control is unaffected, as in the original finding.\n");
            g_failures++;
        } else {
            printf("   => NEITHER outcome. The prediction does not describe what happened, so it is\n"
                   "      reported as an open question rather than forced into a conclusion.\n");
            g_failures++;
        }
    }
    printf("\n");

    // ---------------------------------------------------------------- controls, before any datum
    printf("CONTROLS -- if either of these is wrong, every number below is meaningless.\n");
    {
        std::mt19937 rng(20260917);
        std::vector<float> a((size_t) K * M);
        fill_dist(a, rng, DIST_GAUSSIAN);

        // POSITIVE: f16 must round-trip to almost nothing. Proves the pipeline -- quantise,
        // dequantise, compare -- actually reaches the data and that the metric reads ~0 when the
        // two sides agree.
        const result f16 = measure(GGML_TYPE_F16, K, M);
        const bool pos_ok = f16.ok && f16.err[DIST_GAUSSIAN] < 1e-3;
        printf("   %-4s POSITIVE  f16 round-trips to %.3e (must be < 1e-3)\n",
               pos_ok ? "ok" : "FAIL", f16.err[DIST_GAUSSIAN]);
        if (!pos_ok) g_failures++;

        // NEGATIVE: the metric must be able to SEE disagreement. Two independent draws of the same
        // distribution are uncorrelated, so the relative RMS between them is about sqrt(2). A
        // metric that returned a small number here would call every quantiser perfect.
        std::vector<float> b((size_t) K * M);
        std::mt19937 rng2(777);
        fill_dist(b, rng2, DIST_GAUSSIAN);
        const double neg = rel_rms(b, a);
        const bool neg_ok = neg > 1.0;
        printf("   %-4s NEGATIVE  two independent draws differ by %.4f (must be > 1.0, ~sqrt(2))\n",
               neg_ok ? "ok" : "FAIL", neg);
        if (!neg_ok) g_failures++;
    }
    printf("\n");

    // ------------------------------------------------------ how many rows does a quantiser need?
    // MEASURED ON THE FIRST RUN OF THIS FILE: iq2_ks and iq3_ks dequantised to ALL ZEROS at M=64
    // while the same types, at the same K, report healthy round-trips in test-mulmat-gemv, whose
    // smallest M is 192. Some quantisers in this family fit something over the whole chunk -- the
    // iq4_kt run prints "cluster_points: Oops. Cluster 4 has no points" unprompted, which is a
    // k-means complaining about sparse input -- so the row count is not a free parameter.
    //
    // This sweep exists so that this file's own table can never be read at an M where its
    // quantiser silently produced nothing. It is a diagnosis attached to the guard rather than a
    // guard that only says no.
    printf("WHY DO iq2_ks / iq3_ks DEQUANTISE TO ZERO HERE BUT NOT IN test-mulmat-gemv?\n"
           "Two things differ between the two call sites, so both are varied together rather than\n"
           "one being assumed: the WEIGHT MAGNITUDE (that file fills -0.1..0.1; this one normalised\n"
           "to RMS 0.5) and the DESTINATION BUFFER (a ggml tensor there, a std::vector here, which\n"
           "differ in alignment). A 2x2 settles which, and a type that works in every cell is the\n"
           "positive control that says the grid itself reaches the quantiser.\n");
    {
        const ggml_type probe[] = {
            GGML_TYPE_IQ2_KS, GGML_TYPE_IQ3_KS, GGML_TYPE_IQ4_KSS,
            GGML_TYPE_IQ2_KT, GGML_TYPE_IQ3_KT, GGML_TYPE_IQ4_KT,
        };
        const double rmss[] = { 0.0577, 0.5 };   // 0.0577 is the RMS of uniform(-0.1, 0.1)
        const int64_t m = 192;                   // the smallest M test-mulmat-gemv uses
        printf("   %-9s %-26s %-26s\n", "type", "RMS 0.0577 (as mulmat)", "RMS 0.5 (as this file)");
        for (ggml_type t : probe) {
            printf("   %-9s", ggml_type_name(t));
            for (double rms : rmss) {
                g_rms = rms;
                const auto * tr = ggml_get_type_traits(t);
                std::mt19937 rng(20260917);
                std::vector<float> Wf((size_t) K * m);
                fill_dist(Wf, rng, DIST_GAUSSIAN);
                const size_t rb = ggml_row_size(t, K);

                // Cell A: plain std::vector destination.
                std::vector<uint8_t> qv(rb * m + 1024);
                ggml_quantize_chunk(t, Wf.data(), qv.data(), 0, m, K, nullptr, nullptr);
                // Cell B: a ggml tensor destination, exactly as test-mulmat-gemv does it.
                ggml_init_params ip = { rb * m + 64u * 1024u, NULL, false };
                ggml_context * c = ggml_init(ip);
                ggml_tensor * W = ggml_new_tensor_2d(c, t, K, m);
                ggml_quantize_chunk(t, Wf.data(), W->data, 0, m, K, nullptr, nullptr);

                double amax_v = 0.0, amax_t = 0.0;
                std::vector<float> Wdq((size_t) K * m, 0.0f);
                for (int64_t i = 0; i < m; i++) {
                    tr->to_float((const char *) qv.data() + (size_t) i * rb,
                                 Wdq.data() + (size_t) i * K, K);
                }
                for (float v : Wdq) amax_v = std::max(amax_v, (double) std::fabs(v));
                std::fill(Wdq.begin(), Wdq.end(), 0.0f);
                for (int64_t i = 0; i < m; i++) {
                    tr->to_float((const char *) W->data + (size_t) i * rb,
                                 Wdq.data() + (size_t) i * K, K);
                }
                for (float v : Wdq) amax_t = std::max(amax_t, (double) std::fabs(v));
                ggml_free(c);

                printf("  vec:%-7s tensor:%-7s",
                       amax_v > 0.0 ? "signal" : "ZERO", amax_t > 0.0 ? "signal" : "ZERO");
            }
            printf("\n");
        }
        g_rms = 0.5;
    }
    printf("\n");

    // ------------------------------------------------------------------------------- the table
    // Each KT type is paired with the NON-KT type closest to it in bits per weight. The pairing is
    // the whole experiment: an absolute round-trip number means little, but two formats at the
    // same bpw measured on the same bytes of the same draw is a controlled comparison.
    const pair_row pairs[] = {
        { GGML_TYPE_IQ2_KT, GGML_TYPE_IQ2_KS,  "2.125 vs 2.188 bpw" },
        { GGML_TYPE_IQ3_KT, GGML_TYPE_IQ3_KS,  "3.125 vs 3.188 bpw" },
        { GGML_TYPE_IQ4_KT, GGML_TYPE_IQ4_KSS, "4.000 vs 4.000 bpw -- IDENTICAL" },
    };

    printf("%-9s %-10s", "type", "family");
    for (int d = 0; d < DIST_COUNT; d++) printf(" %12s", dist_name(d));
    printf("\n");
    printf("--------------------------------------------------------------------------\n");

    double ratio[DIST_COUNT][3];
    int nrow = 0;
    for (const auto & p : pairs) {
        const result a = measure(p.kt,  K, M);
        const result b = measure(p.sib, K, M);
        for (const auto * e : { &a, &b }) {
            const ggml_type t = (e == &a) ? p.kt : p.sib;
            printf("%-9s %-10s", ggml_type_name(t), (e == &a) ? "TRELLIS" : "round");
            for (int d = 0; d < DIST_COUNT; d++) printf(" %12.4f", e->err[d]);
            printf("\n");
        }
        for (int d = 0; d < DIST_COUNT; d++) {
            ratio[d][nrow] = (b.err[d] > 0.0) ? a.err[d] / b.err[d] : INFINITY;
        }
        printf("          %-10s", "ratio");
        for (int d = 0; d < DIST_COUNT; d++) printf(" %11.2fx", ratio[d][nrow]);
        printf("   <- %s\n\n", p.why);
        nrow++;
    }

    // ------------------------------------------------------------------------------ the verdict
    // Stated as a rule BEFORE the numbers are read, so the conclusion cannot be fitted to whatever
    // came out. This is the same discipline the shape-independence probe used to decide that
    // iq4_kss was a value fault: assign the meaning first, then look.
    printf("HOW TO READ THIS, decided before the run:\n"
           "   * if the TRELLIS/round ratio falls markedly from uniform -> gaussian -> laplace,\n"
           "     the deficit is the harness's input distribution and the KT quantisers are fine.\n"
           "     The round-trip control would then be testing the harness, not the type.\n"
           "   * if the ratio stays flat and above 1, the distribution explanation is refuted and\n"
           "     the deficit belongs to the quantiser at every distribution tried.\n\n");

    int moved = 0;
    for (int r = 0; r < nrow; r++) {
        if (std::isfinite(ratio[DIST_UNIFORM][r]) && std::isfinite(ratio[DIST_LAPLACE][r]) &&
            ratio[DIST_LAPLACE][r] < 0.85 * ratio[DIST_UNIFORM][r]) {
            moved++;
        }
    }
    printf("VERDICT: %d of %d pairs improved by more than 15%% from uniform to laplace.\n",
           moved, nrow);
    if (moved == nrow) {
        printf("   => DISTRIBUTION-DRIVEN. The KT codebooks are fitted to a shape this harness\n"
               "      does not generate. Fix the harness's weights, not the threshold.\n");
    } else if (moved == 0) {
        printf("   => NOT distribution-driven. The deficit survives every distribution tried, so\n"
               "      the KT quantiser is worse than its bpw-matched sibling on this host, and\n"
               "      widening the threshold would certify that difference as acceptable without\n"
               "      anyone having decided that it is.\n");
    } else {
        printf("   => MIXED, and a mixed result is not a conclusion. Report it as open.\n");
    }

    printf("\nNOT COVERED: real model weights (all three distributions here are synthetic), any\n"
           "imatrix-guided quantisation (nullptr is passed, so this is the imatrix-free path),\n"
           "K other than %lld, and every non-host backend.\n", (long long) K);

    printf("\n%s -- %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
    return g_failures ? 1 : 0;
}
