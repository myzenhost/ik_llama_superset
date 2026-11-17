// Do a quant type's TWO decoders agree about the same bytes?
//
// Every quantised type in this tree is decoded in two independent places: dequantize_row_<t>(),
// which produces floats, and the GEMM kernel behind vec_dot, which consumes the packed bytes
// directly. Nothing checked that those two agree. test-mulmat-gemv compares a MATMUL against a
// reference built from the first decoder, so when they disagree it reports a number -- it cannot
// say which decoder is wrong, or which weights are affected.
//
// This test asks the narrower question and answers it exactly:
//
//     for weight i, what value does the KERNEL think is stored there?
//
// THE EXTRACTION IS EXACT, not approximate, and that is what makes this worth writing. The
// activation side is quantised to the weight type's .vec_dot_type before vec_dot runs. For a
// ONE-HOT activation e_i, q8_K quantisation is lossless: quantize_row_q8_K sets d = max|x|/127 =
// 1/127 and q[i] = round(x[i]/d) = 127, every other q = 0, so dequantising gives exactly 1.0 at i
// and exactly 0 elsewhere. A dot product against e_i therefore returns the kernel's own decoded
// weight at i, with no quantisation error to explain away. (A type whose vec_dot_type is not a
// q8_K variant is SKIPPED by name rather than silently mis-measured -- see the check below.)
//
// WHY IT EXISTS. iq4_kss and iq5_ks disagree with test-mulmat-gemv's reference by 0.26 and 0.21
// while agreeing with themselves at every shape and batch width. That was narrowed by elimination
// to the GEMM decode path, but "which weights, and how" was still unanswered, and a fix aimed at a
// mechanism nobody has seen is a guess. This prints the disagreement per weight and per 32-weight
// block, so the pattern is visible rather than inferred.
//
// SCOPE: one row, K = 256 (a single QK_K block group), host backend, nrc == 1, uniform-random
// weights, and only types whose vec_dot_type is a q8_K variant. It says nothing about batching,
// other shapes, other backends, or any type it SKIPs -- each of which prints its own reason.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_failures = 0;

static const char * tname(ggml_type t) {
    const char * n = ggml_type_name(t);
    return n ? n : "<no traits>";
}

// One type: quantise a row, decode it both ways, and report where they differ.
static void compare_decoders(ggml_type wtype, int64_t K, std::mt19937 & rng) {
    const ggml_type_traits     * base = ggml_get_type_traits(wtype);
    const ggml_type_traits_cpu * cpu  = ggml_get_type_traits_cpu(wtype);

    if (!base || !base->type_name || !base->to_float || !base->from_float) {
        printf("  ---- %-8s SKIPPED: this build has no to_float/from_float for it, so there is no\n"
               "       first decoder to compare against. Not a verdict about the kernel.\n",
               tname(wtype));
        return;
    }
    if (!cpu || !cpu->vec_dot) {
        printf("  ---- %-8s SKIPPED: no CPU vec_dot, so there is no second decoder.\n",
               tname(wtype));
        return;
    }
    // The one-hot trick is only exact for a q8_K-family activation. Say so by name rather than
    // quietly producing numbers that carry an unstated quantisation error.
    const ggml_type vdt = cpu->vec_dot_type;
    if (vdt != GGML_TYPE_Q8_K) {
        printf("  ---- %-8s SKIPPED: vec_dot_type is %s, not q8_K. A one-hot activation is only\n"
               "       exactly representable in q8_K here, and an inexact probe would report a\n"
               "       disagreement that is its own rounding. NOT a verdict about this type.\n",
               tname(wtype), tname(vdt));
        return;
    }
    const ggml_type_traits_cpu * yt = ggml_get_type_traits_cpu(vdt);
    if (!yt || !yt->from_float) {
        printf("  ---- %-8s SKIPPED: no from_float for the activation type %s.\n",
               tname(wtype), tname(vdt));
        return;
    }

    // ---- the weights, quantised once -------------------------------------------------------
    std::vector<float> wf((size_t) K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < K; i++) wf[i] = dist(rng);

    std::vector<char> wq(ggml_row_size(wtype, K) + 64, 0);
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, 1, K, nullptr, nullptr);

    // ---- decoder 1: dequantize_row_<t> -------------------------------------------------------
    std::vector<float> w_deq((size_t) K, 0.0f);
    base->to_float(wq.data(), w_deq.data(), K);

    // ---- decoder 2: the GEMM, probed one weight at a time ------------------------------------
    std::vector<float> w_ker((size_t) K, 0.0f);
    std::vector<float> onehot((size_t) K, 0.0f);
    std::vector<char>  yq(ggml_row_size(vdt, K) + 64, 0);
    for (int64_t i = 0; i < K; i++) {
        std::memset(onehot.data(), 0, (size_t) K * sizeof(float));
        onehot[i] = 1.0f;
        std::memset(yq.data(), 0, yq.size());
        yt->from_float(onehot.data(), yq.data(), K);
        float s = 0.0f;
        cpu->vec_dot((int) K, &s, 0, wq.data(), 0, yq.data(), 0, 1);
        w_ker[i] = s;
    }

    // ---- report ------------------------------------------------------------------------------
    double se = 0.0, sw = 0.0, worst = 0.0;
    int64_t worst_i = -1, n_bad = 0;
    bool finite = true;
    for (int64_t i = 0; i < K; i++) {
        if (!std::isfinite(w_ker[i]) || !std::isfinite(w_deq[i])) { finite = false; continue; }
        const double d = (double) w_ker[i] - (double) w_deq[i];
        se += d*d;
        sw += (double) w_deq[i] * (double) w_deq[i];
        if (std::fabs(d) > worst) { worst = std::fabs(d); worst_i = i; }
        if (std::fabs(d) > 1e-4) n_bad++;
    }
    // Same guard as the constant-activation probe, for the same measured reason: an all-zero
    // dequantised row makes both decoders agree about nothing and would print a clean "ok".
    if (!(sw > 1e-12)) {
        printf("  ---- %-8s NO SIGNAL: the dequantised row is all (or almost all) zero, so the two\n"
               "       decoders would agree about nothing. NOT a pass and NOT a verdict.\n",
               tname(wtype));
        return;
    }
    const double rel = finite ? std::sqrt(se / sw) : INFINITY;
    const bool ok = finite && rel <= 1e-3;

    printf("  %-4s %-8s K=%-5lld | disagreeing weights: %-4lld of %-4lld | relative RMS %-10.4g "
           "| worst %.4g at i=%lld\n",
           ok ? "ok" : "FAIL", tname(wtype), (long long) K,
           (long long) n_bad, (long long) K, rel, worst, (long long) worst_i);
    if (!finite) {
        printf("       non-finite values present in one of the two decodings.\n");
    }
    if (!ok) {
        g_failures++;
        // Per 32-weight block: a fault tied to the per-block scale or grid-shift bit shows up as
        // whole blocks being wrong while their neighbours are clean. A fault in the quant nibbles
        // scatters instead. The shape of this list is the diagnosis.
        printf("       per 32-weight block, relative RMS of (kernel - dequantiser):\n       ");
        for (int64_t b = 0; b*32 < K; b++) {
            double bse = 0.0, bsw = 0.0;
            for (int64_t j = b*32; j < (b+1)*32 && j < K; j++) {
                const double d = (double) w_ker[j] - (double) w_deq[j];
                bse += d*d;
                bsw += (double) w_deq[j] * (double) w_deq[j];
            }
            printf("%7.4f", std::sqrt(bse / (bsw > 0 ? bsw : 1e-30)));
            if ((b+1) % 8 == 0) printf("\n       ");
        }
        printf("\n");
        printf("       first 6 disagreements   i:  dequantiser  ->  kernel\n");
        int shown = 0;
        for (int64_t i = 0; i < K && shown < 6; i++) {
            const double d = (double) w_ker[i] - (double) w_deq[i];
            if (std::fabs(d) > 1e-4) {
                printf("           %4lld:  %12.6f  ->  %12.6f   (ratio %.4f)\n",
                       (long long) i, w_deq[i], w_ker[i],
                       w_deq[i] != 0.0f ? w_ker[i]/w_deq[i] : 0.0);
                shown++;
            }
        }
    }
}

// THE ONE-HOT PROBE HAS A BLIND SPOT, AND IT IS EXACTLY THE MECHANISM UNDER SUSPICION.
//
// These kernels do not compute a plain sum of w[i]*y[i]. They decode the quants against a grid
// whose entries are stored with an offset, and correct for that offset in a separate term built
// from the ACTIVATION'S BLOCK SUMS -- q8_K's bsums, consumed by accum_mins(). For iq4_kss the
// per-block grid-shift bit enters only there: new_block computes
//     shifts   = (scales & 1) ? 4 : 0
//     scales_s = scales * (-128 + shifts)
// and hands scales_s to accum_mins. That term is proportional to sum(y) over the block.
//
// A one-hot activation has a block sum of 127 in ONE block and ZERO in every other, so it
// suppresses the very term the grid-shift lives in. The one-hot probe agreeing therefore does NOT
// clear the kernel -- it establishes that the QUANT NIBBLES and the per-block SCALE decode
// identically, and says nothing about the offset correction.
//
// A CONSTANT activation is the complement: every block sum is at its maximum, so the mins term is
// as large as it can be relative to everything else. Running both and comparing is what separates
// "the nibbles are decoded wrong" from "the offset correction is wrong", and neither probe alone
// can do it.
//
// dot(W, 1) is just the sum of the decoded weights, so the dequantiser's answer is computed
// directly in double and needs no second implementation to be trusted.
static void compare_constant_activation(ggml_type wtype, int64_t K, std::mt19937 & rng) {
    const ggml_type_traits     * base = ggml_get_type_traits(wtype);
    const ggml_type_traits_cpu * cpu  = ggml_get_type_traits_cpu(wtype);
    if (!base || !base->type_name || !base->to_float || !base->from_float || !cpu || !cpu->vec_dot) {
        printf("  ---- %-8s SKIPPED: missing a decoder (see the one-hot section above).\n",
               tname(wtype));
        return;
    }
    const ggml_type vdt = cpu->vec_dot_type;
    const ggml_type_traits_cpu * yt = ggml_get_type_traits_cpu(vdt);
    if (vdt != GGML_TYPE_Q8_K || !yt || !yt->from_float) {
        printf("  ---- %-8s SKIPPED: vec_dot_type is %s, not q8_K.\n", tname(wtype), tname(vdt));
        return;
    }

    std::vector<float> wf((size_t) K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < K; i++) wf[i] = dist(rng);

    std::vector<char> wq(ggml_row_size(wtype, K) + 64, 0);
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, 1, K, nullptr, nullptr);

    std::vector<float> w_deq((size_t) K, 0.0f);
    base->to_float(wq.data(), w_deq.data(), K);

    // y = 1 everywhere. q8_K stores it exactly: d = 1/127, every q = 127.
    std::vector<float> ones((size_t) K, 1.0f);
    std::vector<char>  yq(ggml_row_size(vdt, K) + 64, 0);
    yt->from_float(ones.data(), yq.data(), K);

    float s_ker = 0.0f;
    cpu->vec_dot((int) K, &s_ker, 0, wq.data(), 0, yq.data(), 0, 1);

    double s_deq = 0.0, mag = 0.0;
    for (int64_t i = 0; i < K; i++) {
        s_deq += (double) w_deq[i];
        mag   += std::fabs((double) w_deq[i]);
    }
    // A DEGENERATE ROW MUST NOT READ AS A PASS. The first version of this check divided by
    // max(mag, 1e-30) and called the result ok -- and three control types came back
    // "kernel 0, dequantiser 0, gap 0, ok" because their dequantised weights were ALL ZERO, so
    // both decoders trivially agreed about nothing. That is the failure shape this whole test
    // exists to catch, reproduced inside the test itself: an instrument reporting healthy while
    // measuring nothing. If there is no signal, say so by name instead of passing.
    if (!(mag > 1e-6)) {
        printf("  ---- %-8s NO SIGNAL: the dequantised row is all (or almost all) zero, sum|w| = "
               "%.3g,\n"
               "       so the two decoders would agree about nothing. This build did not produce "
               "usable\n"
               "       weights for this type at K=%lld. NOT a pass and NOT a verdict about the "
               "kernel.\n",
               tname(wtype), mag, (long long) K);
        return;
    }
    // Relative to the sum of |weights|, not to the signed sum: the signed sum of random weights is
    // near zero, and dividing by it would manufacture a huge relative error from nothing.
    const double rel = std::fabs((double) s_ker - s_deq) / mag;
    const bool ok = std::isfinite(s_ker) && rel <= 1e-3;
    if (!ok) g_failures++;
    printf("  %-4s %-8s K=%-5lld | kernel %-13.6g dequantiser %-13.6g | absolute gap %-11.4g "
           "| relative to sum|w| %.4g\n",
           ok ? "ok" : "FAIL", tname(wtype), (long long) K,
           (double) s_ker, s_deq, std::fabs((double) s_ker - s_deq), rel);
}

// LINEARITY, AND IT NEEDS NO REFERENCE AT ALL.
//
// A dot product is linear in its second argument: dot(W, sum_i e_i) must equal sum_i dot(W, e_i).
// Both sides here are produced by THE SAME KERNEL, so this compares the kernel against itself and
// cannot be blamed on the dequantiser, the quantiser, the harness's reference, or the grid tables.
// Nothing outside the kernel is involved.
//
// It is worth its own probe because the first two probes produced a result that cannot both be
// true of a linear operator: one-hot said the kernel's decoded weight at every i equals the
// dequantiser's, so the sum of its own per-weight answers is 18.2101 -- while its answer for a
// constant activation was -1.39986. If that holds, the fault is not "the kernel decodes weights
// wrongly" at all; it is that the kernel's output depends on the activation in a way no dot
// product can.
//
// Both activations are represented EXACTLY in q8_K (all-ones: d = 1/127, every q = 127; one-hot:
// d = 1/127, one q = 127), so neither side carries a quantisation error to explain a gap.
static void check_linearity(ggml_type wtype, int64_t K, std::mt19937 & rng) {
    const ggml_type_traits     * base = ggml_get_type_traits(wtype);
    const ggml_type_traits_cpu * cpu  = ggml_get_type_traits_cpu(wtype);
    if (!base || !base->type_name || !base->from_float || !cpu || !cpu->vec_dot) {
        printf("  ---- %-8s SKIPPED: missing a decoder.\n", tname(wtype));
        return;
    }
    const ggml_type vdt = cpu->vec_dot_type;
    const ggml_type_traits_cpu * yt = ggml_get_type_traits_cpu(vdt);
    if (vdt != GGML_TYPE_Q8_K || !yt || !yt->from_float) {
        printf("  ---- %-8s SKIPPED: vec_dot_type is %s, not q8_K.\n", tname(wtype), tname(vdt));
        return;
    }

    std::vector<float> wf((size_t) K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < K; i++) wf[i] = dist(rng);
    std::vector<char> wq(ggml_row_size(wtype, K) + 64, 0);
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, 1, K, nullptr, nullptr);

    std::vector<float> y((size_t) K, 0.0f);
    std::vector<char>  yq(ggml_row_size(vdt, K) + 64, 0);

    // sum of the kernel's own per-weight answers
    double sum_parts = 0.0;
    for (int64_t i = 0; i < K; i++) {
        std::memset(y.data(), 0, (size_t) K * sizeof(float));
        y[i] = 1.0f;
        std::memset(yq.data(), 0, yq.size());
        yt->from_float(y.data(), yq.data(), K);
        float s = 0.0f;
        cpu->vec_dot((int) K, &s, 0, wq.data(), 0, yq.data(), 0, 1);
        sum_parts += (double) s;
    }
    // the kernel's answer for the sum of those same basis vectors
    for (int64_t i = 0; i < K; i++) y[i] = 1.0f;
    std::memset(yq.data(), 0, yq.size());
    yt->from_float(y.data(), yq.data(), K);
    float s_whole = 0.0f;
    cpu->vec_dot((int) K, &s_whole, 0, wq.data(), 0, yq.data(), 0, 1);

    const double scale = std::max(std::fabs(sum_parts), std::fabs((double) s_whole));
    if (!(scale > 1e-6)) {
        printf("  ---- %-8s NO SIGNAL: both sides are ~0, so linearity is untested here.\n",
               tname(wtype));
        return;
    }
    const double rel = std::fabs(sum_parts - (double) s_whole) / scale;
    const bool ok = std::isfinite(s_whole) && rel <= 1e-3;
    if (!ok) g_failures++;
    printf("  %-4s %-8s K=%-5lld | sum of dot(W,e_i) = %-13.6g | dot(W,1) = %-13.6g | "
           "relative gap %.4g\n",
           ok ? "ok" : "FAIL", tname(wtype), (long long) K, sum_parts, (double) s_whole, rel);
}

// LINEARITY AT REALISTIC MAGNITUDES, and this exists to stop the previous probe being over-read.
//
// The all-ones activation puts EVERY q8 value at its maximum of 127. If a kernel accumulates
// products in int16 through _mm256_maddubs_epi16 -- which saturates -- that input is the worst
// case in the whole domain, and a failure there would be a real hazard at an extreme rather than
// an explanation of the 0.21-0.26 disagreement test-mulmat-gemv sees with RANDOM activations.
// Reporting "the kernel is non-linear" from a maximal input alone would be a setup artifact
// dressed as a finding.
//
// So: split a RANDOM activation into two random halves, y = y1 + y2, all of moderate magnitude,
// and check dot(W, y1+y2) == dot(W, y1) + dot(W, y2). The identity is the same; the inputs are
// ordinary. If linearity fails here too, saturation at the extreme is not the story and the fault
// is present at magnitudes a real model would produce.
//
// NOTE the one inexactness, stated rather than hidden: y1, y2 and y1+y2 are quantised to q8_K
// SEPARATELY, so each carries its own rounding. That puts a floor under the achievable agreement
// -- which is why the control matters more here than anywhere else in this file. Whatever floor
// q8_K rounding imposes, it imposes on iq4_ks too, and iq4_ks's number is what "as good as this
// probe can get" looks like.
static void check_linearity_random(ggml_type wtype, int64_t K, std::mt19937 & rng) {
    const ggml_type_traits     * base = ggml_get_type_traits(wtype);
    const ggml_type_traits_cpu * cpu  = ggml_get_type_traits_cpu(wtype);
    if (!base || !base->type_name || !base->from_float || !cpu || !cpu->vec_dot) {
        printf("  ---- %-8s SKIPPED: missing a decoder.\n", tname(wtype));
        return;
    }
    const ggml_type vdt = cpu->vec_dot_type;
    const ggml_type_traits_cpu * yt = ggml_get_type_traits_cpu(vdt);
    if (vdt != GGML_TYPE_Q8_K || !yt || !yt->from_float) {
        printf("  ---- %-8s SKIPPED: vec_dot_type is %s, not q8_K.\n", tname(wtype), tname(vdt));
        return;
    }

    std::vector<float> wf((size_t) K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < K; i++) wf[i] = dist(rng);
    std::vector<char> wq(ggml_row_size(wtype, K) + 64, 0);
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, 1, K, nullptr, nullptr);

    std::vector<float> y1((size_t) K), y2((size_t) K), ys((size_t) K);
    for (int64_t i = 0; i < K; i++) {
        y1[i] = dist(rng);
        y2[i] = dist(rng);
        ys[i] = y1[i] + y2[i];
    }

    std::vector<char> yq(ggml_row_size(vdt, K) + 64, 0);
    auto dot = [&](const std::vector<float> & y) {
        std::memset(yq.data(), 0, yq.size());
        yt->from_float(y.data(), yq.data(), K);
        float s = 0.0f;
        cpu->vec_dot((int) K, &s, 0, wq.data(), 0, yq.data(), 0, 1);
        return (double) s;
    };
    const double d1 = dot(y1), d2 = dot(y2), ds = dot(ys);

    const double scale = std::max(std::fabs(d1) + std::fabs(d2), std::fabs(ds));
    if (!(scale > 1e-6)) {
        printf("  ---- %-8s NO SIGNAL: all three dot products are ~0.\n", tname(wtype));
        return;
    }
    const double rel = std::fabs((d1 + d2) - ds) / scale;
    // Looser than the exact probes: three independent q8_K roundings put a floor here. The control
    // shows where that floor actually is rather than this constant being asserted.
    const bool ok = std::isfinite(ds) && rel <= 2e-2;
    if (!ok) g_failures++;
    printf("  %-4s %-8s K=%-5lld | dot(y1)+dot(y2) = %-13.6g | dot(y1+y2) = %-13.6g | "
           "relative gap %.4g\n",
           ok ? "ok" : "FAIL", tname(wtype), (long long) K, d1 + d2, ds, rel);
}

// WHERE, INSIDE THE ROW, DOES THE NON-LINEARITY LIVE?
//
// The offset-correction constant is NOT the bug -- that was checked arithmetically and it holds:
//   iq4k_values[0:16]   == kvalues_iq4nl - 128   exactly (the grid is stored unsigned, +128)
//   iq4k_values[16:32] - iq4k_values[0:16] == 4, for every one of the 16 entries
// so value = (iq4nl[q] - 128) + 4*shift, and the kernel's
//   scales_s = scales * (-128 + shifts),  shifts in {0, 4}
// is exactly that correction. The constant is right and the single-grid trick is sound.
//
// What remains is how the correction is DISTRIBUTED across the row. q8_K carries one bsum per 16
// weights; the scales are per 32. A correct kernel pairs each scale with the two bsums under it.
// A mis-pairing, a wrong stride, or a term applied to the wrong sub-block is invisible to a
// one-hot activation -- which activates one bsum slot and can only ever check that slot against
// itself -- and shows up the moment more than one is live.
//
// So: activate y = 1 on exactly ONE 32-weight block at a time and compare the kernel against the
// sum of the dequantised weights over that block. Per-block agreement with whole-row disagreement
// means the fault is in the ACCUMULATION ACROSS blocks; per-block disagreement localises which
// blocks, and whether it tracks the shift bit.
static void check_per_block(ggml_type wtype, int64_t K, std::mt19937 & rng) {
    const ggml_type_traits     * base = ggml_get_type_traits(wtype);
    const ggml_type_traits_cpu * cpu  = ggml_get_type_traits_cpu(wtype);
    if (!base || !base->type_name || !base->to_float || !base->from_float || !cpu || !cpu->vec_dot) {
        printf("  ---- %-8s SKIPPED: missing a decoder.\n", tname(wtype));
        return;
    }
    const ggml_type vdt = cpu->vec_dot_type;
    const ggml_type_traits_cpu * yt = ggml_get_type_traits_cpu(vdt);
    if (vdt != GGML_TYPE_Q8_K || !yt || !yt->from_float) {
        printf("  ---- %-8s SKIPPED: vec_dot_type is %s, not q8_K.\n", tname(wtype), tname(vdt));
        return;
    }

    std::vector<float> wf((size_t) K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < K; i++) wf[i] = dist(rng);
    std::vector<char> wq(ggml_row_size(wtype, K) + 64, 0);
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, 1, K, nullptr, nullptr);
    std::vector<float> w_deq((size_t) K, 0.0f);
    base->to_float(wq.data(), w_deq.data(), K);

    double mag = 0.0;
    for (int64_t i = 0; i < K; i++) mag += std::fabs((double) w_deq[i]);
    if (!(mag > 1e-6)) {
        printf("  ---- %-8s NO SIGNAL: dequantised row is ~zero.\n", tname(wtype));
        return;
    }

    std::vector<float> y((size_t) K, 0.0f);
    std::vector<char>  yq(ggml_row_size(vdt, K) + 64, 0);
    printf("  %-8s per 32-weight block, kernel minus (sum of dequantised weights):\n", tname(wtype));
    int n_bad = 0;
    for (int64_t b = 0; b*32 < K; b++) {
        std::memset(y.data(), 0, (size_t) K * sizeof(float));
        double want = 0.0;
        for (int64_t j = b*32; j < (b+1)*32 && j < K; j++) {
            y[j]  = 1.0f;
            want += (double) w_deq[j];
        }
        std::memset(yq.data(), 0, yq.size());
        yt->from_float(y.data(), yq.data(), K);
        float s = 0.0f;
        cpu->vec_dot((int) K, &s, 0, wq.data(), 0, yq.data(), 0, 1);
        const double gap = (double) s - want;
        const bool bad = std::fabs(gap) > 1e-3 * std::max(1.0, std::fabs(want));
        if (bad) n_bad++;
        printf("       block %2lld: kernel %10.5f  dequantiser %10.5f  gap %11.5f  %s\n",
               (long long) b, (double) s, want, gap, bad ? "<-- WRONG" : "ok");
    }
    printf("       %d of %lld blocks wrong when activated ALONE.\n",
           n_bad, (long long) ((K + 31) / 32));
    if (n_bad == 0) {
        printf("       Every block is correct in isolation, so the fault is in the ACCUMULATION\n"
               "       ACROSS blocks, not in any single block's decode or correction.\n");
    }
}

// HOW DOES THE ERROR SCALE WITH THE NUMBER OF LIVE WEIGHTS IN ONE BLOCK?
//
// Two measurements that must be reconciled: ONE live weight is exact to float epsilon, and
// THIRTY-TWO live weights in the SAME block are wrong by several units. Every block fails alone,
// so it is not accumulation across blocks.
//
// The block's offset correction is proportional to that block's q8_K bsum, i.e. to the number of
// live weights. If the correction were simply mis-scaled by a factor, the error would be strictly
// proportional to the count -- and a 32-weight error of ~5 would imply a 1-weight error of ~0.15,
// which is four orders of magnitude above what the one-hot probe measured. So a plain scale factor
// is already excluded and the dependence must be something else.
//
// This sweeps the count 1, 2, 4, 8, 16, 32 inside a single block and prints the error against it.
// Strict proportionality, a step at some threshold, or a jump when the second 16-weight bsum group
// becomes live are all different faults and look different here.
static void check_count_scaling(ggml_type wtype, int64_t K, int64_t block, std::mt19937 & rng) {
    const ggml_type_traits     * base = ggml_get_type_traits(wtype);
    const ggml_type_traits_cpu * cpu  = ggml_get_type_traits_cpu(wtype);
    if (!base || !base->type_name || !base->to_float || !base->from_float || !cpu || !cpu->vec_dot) {
        printf("  ---- %-8s SKIPPED: missing a decoder.\n", tname(wtype));
        return;
    }
    const ggml_type vdt = cpu->vec_dot_type;
    const ggml_type_traits_cpu * yt = ggml_get_type_traits_cpu(vdt);
    if (vdt != GGML_TYPE_Q8_K || !yt || !yt->from_float) {
        printf("  ---- %-8s SKIPPED: vec_dot_type is %s, not q8_K.\n", tname(wtype), tname(vdt));
        return;
    }

    std::vector<float> wf((size_t) K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < K; i++) wf[i] = dist(rng);
    std::vector<char> wq(ggml_row_size(wtype, K) + 64, 0);
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, 1, K, nullptr, nullptr);
    std::vector<float> w_deq((size_t) K, 0.0f);
    base->to_float(wq.data(), w_deq.data(), K);

    double mag = 0.0;
    for (int64_t i = 0; i < K; i++) mag += std::fabs((double) w_deq[i]);
    if (!(mag > 1e-6)) {
        printf("  ---- %-8s NO SIGNAL: dequantised row is ~zero.\n", tname(wtype));
        return;
    }

    std::vector<float> y((size_t) K, 0.0f);
    std::vector<char>  yq(ggml_row_size(vdt, K) + 64, 0);
    printf("  %-8s block %lld, sweeping how many of its 32 weights are live:\n",
           tname(wtype), (long long) block);
    printf("       live   kernel      dequantiser   gap          gap/live\n");
    const int counts[] = { 1, 2, 4, 8, 16, 17, 32 };
    for (size_t c = 0; c < sizeof(counts)/sizeof(counts[0]); c++) {
        const int n = counts[c];
        std::memset(y.data(), 0, (size_t) K * sizeof(float));
        double want = 0.0;
        for (int j = 0; j < n; j++) {
            const int64_t idx = block*32 + j;
            if (idx >= K) break;
            y[idx] = 1.0f;
            want  += (double) w_deq[idx];
        }
        std::memset(yq.data(), 0, yq.size());
        yt->from_float(y.data(), yq.data(), K);
        float s = 0.0f;
        cpu->vec_dot((int) K, &s, 0, wq.data(), 0, yq.data(), 0, 1);
        const double gap = (double) s - want;
        printf("       %4d   %10.5f  %10.5f    %11.5f  %11.5f\n",
               n, (double) s, want, gap, gap / n);
    }
    // 17 is in the list on purpose: q8_K carries one bsum per 16 weights, so 16 -> 17 is where a
    // block's SECOND bsum group first becomes live. A fault in how the two groups under one scale
    // are paired shows up as a jump exactly there and nowhere else.
    printf("       (17 is included because q8_K has one bsum per 16 weights: 16 -> 17 is where\n"
           "        this block's SECOND bsum group first becomes live.)\n");
}

// ADJACENT vs SPACED: the same number of live weights, differing only in whether they PAIR UP.
//
// This is a hypothesis test with a sharp prediction, not another survey. _mm256_maddubs_epi16
// multiplies bytes and sums them in PAIRS into an int16, with saturation. iq4_kss shuffles the
// UNSIGNED grid kvalues_iq4nl (1 .. 241), so two adjacent products can reach 2 * 241 * 127 =
// 61214, well past int16's 32767. iq4_ks shuffles the SIGNED iq4k_values (|v| <= 127), where two
// products reach at most 2 * 127 * 127 = 32258 -- just inside.
//
// If that is the fault, then for iq4_kss:
//     live weights at EVEN indices only  -> each maddubs pair has ONE non-zero product -> EXACT
//     live weights CONSECUTIVE           -> pairs have TWO non-zero products -> WRONG
// with the SAME number of live weights in both cases. Nothing else distinguishes the two inputs,
// so a difference between them isolates the pairing and cannot be explained by count, magnitude,
// block, scale, grid or bsums.
//
// If both come back equally wrong, the pairwise-saturation hypothesis is dead and the step
// pattern in the count sweep needs another explanation.
static void check_adjacent_vs_spaced(ggml_type wtype, int64_t K, std::mt19937 & rng) {
    const ggml_type_traits     * base = ggml_get_type_traits(wtype);
    const ggml_type_traits_cpu * cpu  = ggml_get_type_traits_cpu(wtype);
    if (!base || !base->type_name || !base->to_float || !base->from_float || !cpu || !cpu->vec_dot) {
        printf("  ---- %-8s SKIPPED: missing a decoder.\n", tname(wtype));
        return;
    }
    const ggml_type vdt = cpu->vec_dot_type;
    const ggml_type_traits_cpu * yt = ggml_get_type_traits_cpu(vdt);
    if (vdt != GGML_TYPE_Q8_K || !yt || !yt->from_float) {
        printf("  ---- %-8s SKIPPED: vec_dot_type is %s, not q8_K.\n", tname(wtype), tname(vdt));
        return;
    }

    std::vector<float> wf((size_t) K);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < K; i++) wf[i] = dist(rng);
    std::vector<char> wq(ggml_row_size(wtype, K) + 64, 0);
    ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, 1, K, nullptr, nullptr);
    std::vector<float> w_deq((size_t) K, 0.0f);
    base->to_float(wq.data(), w_deq.data(), K);

    double mag = 0.0;
    for (int64_t i = 0; i < K; i++) mag += std::fabs((double) w_deq[i]);
    if (!(mag > 1e-6)) {
        printf("  ---- %-8s NO SIGNAL: dequantised row is ~zero.\n", tname(wtype));
        return;
    }

    std::vector<float> y((size_t) K, 0.0f);
    std::vector<char>  yq(ggml_row_size(vdt, K) + 64, 0);
    auto run = [&](bool spaced, int n_live) {
        std::memset(y.data(), 0, (size_t) K * sizeof(float));
        double want = 0.0;
        int placed = 0;
        for (int64_t i = 0; i < K && placed < n_live; i++) {
            if (spaced && (i % 2)) continue;
            y[i]  = 1.0f;
            want += (double) w_deq[i];
            placed++;
        }
        std::memset(yq.data(), 0, yq.size());
        yt->from_float(y.data(), yq.data(), K);
        float s = 0.0f;
        cpu->vec_dot((int) K, &s, 0, wq.data(), 0, yq.data(), 0, 1);
        return std::pair<double,double>((double) s, want);
    };

    const int n_live = 16;
    const auto adj = run(false, n_live);
    const auto spc = run(true,  n_live);
    const double gap_adj = adj.first - adj.second;
    const double gap_spc = spc.first - spc.second;
    printf("  %-8s %d live weights each, differing ONLY in whether they pair up:\n",
           tname(wtype), n_live);
    printf("       CONSECUTIVE indices: kernel %10.5f  dequantiser %10.5f  gap %11.5f\n",
           adj.first, adj.second, gap_adj);
    printf("       EVEN indices only  : kernel %10.5f  dequantiser %10.5f  gap %11.5f\n",
           spc.first, spc.second, gap_spc);
    if (std::fabs(gap_spc) < 1e-3 && std::fabs(gap_adj) > 1e-3) {
        printf("       SPACED IS EXACT AND ADJACENT IS NOT -> the fault is in the PAIRWISE step,\n"
               "       which is _mm256_maddubs_epi16 summing two byte products into a saturating\n"
               "       int16. The unsigned grid (1..241) makes two products overflow; the signed\n"
               "       grid iq4_ks uses (|v| <= 127) does not.\n");
    } else if (std::fabs(gap_spc) > 1e-3 && std::fabs(gap_adj) > 1e-3) {
        printf("       BOTH are wrong -> pairing is NOT the discriminator, and the pairwise-\n"
               "       saturation hypothesis is refuted. The step pattern needs another cause.\n");
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::mt19937 rng(20260915);

    printf("TWO DECODERS, ONE SET OF BYTES: does the GEMM agree with dequantize_row_<t>?\n");
    printf("The kernel's own decoded weight at i is recovered EXACTLY by a dot product against a\n");
    printf("one-hot activation, which q8_K represents without loss (d = 1/127, q[i] = 127).\n\n");

    // POSITIVE CONTROL FIRST, and the whole test is worthless without it. iq4_ks passes
    // test-mulmat-gemv at 0.003653 and shares the scale convention, vec_dot_type, nrows,
    // row_meta_size and grid family with the two suspects. If the one-hot extraction is sound it
    // must agree with the dequantiser for iq4_ks; if it does not, the PROBE is broken and nothing
    // below it means anything. q4_K and q6_K are the same argument from outside this family.
    printf("CONTROLS -- types that PASS test-mulmat-gemv. These must agree.\n");
    compare_decoders(GGML_TYPE_Q4_K,   256, rng);
    compare_decoders(GGML_TYPE_Q6_K,   256, rng);
    compare_decoders(GGML_TYPE_IQ4_K,  256, rng);
    compare_decoders(GGML_TYPE_IQ4_KS, 256, rng);

    printf("\nSUSPECTS -- disagree with test-mulmat-gemv's reference by 0.26 and 0.21.\n");
    compare_decoders(GGML_TYPE_IQ4_KSS, 256, rng);
    compare_decoders(GGML_TYPE_IQ5_KS,  256, rng);

    // SECOND PROBE, complementary to the first by construction. If the one-hot section passed and
    // this one fails, the quant nibbles and per-block scales decode correctly and the ACTIVATION-
    // SUM term -- where the grid-shift correction lives -- does not. If both pass, the kernel
    // decodes correctly at nrc == 1 for one row and the disagreement test-mulmat-gemv reports is
    // somewhere else entirely: batching, threading, or the mul_mat dispatch around vec_dot.
    printf("\nCONSTANT-ACTIVATION PROBE -- y = 1 everywhere, so every q8_K block sum is maximal\n");
    printf("and the offset-correction term (accum_mins) is at its most visible. The one-hot probe\n");
    printf("above nearly ZEROES that term, so it cannot see a fault in it.\n");
    compare_constant_activation(GGML_TYPE_Q4_K,    256, rng);
    compare_constant_activation(GGML_TYPE_Q6_K,    256, rng);
    compare_constant_activation(GGML_TYPE_IQ4_K,   256, rng);
    compare_constant_activation(GGML_TYPE_IQ4_KS,  256, rng);
    compare_constant_activation(GGML_TYPE_IQ4_KSS, 256, rng);
    compare_constant_activation(GGML_TYPE_IQ5_KS,  256, rng);

    // THIRD PROBE: the kernel against ITSELF. No reference, no dequantiser, no harness arithmetic
    // -- just whether the kernel is the linear operator a dot product has to be.
    printf("\nLINEARITY PROBE -- dot(W, sum e_i) must equal sum dot(W, e_i). Both sides come from\n");
    printf("the SAME kernel, so nothing outside it can be blamed for a gap.\n");
    check_linearity(GGML_TYPE_IQ4_KS,  256, rng);
    check_linearity(GGML_TYPE_IQ4_KSS, 256, rng);
    check_linearity(GGML_TYPE_IQ5_KS,  256, rng);

    // FOURTH PROBE, and it is the one that decides whether the third is a finding or an artifact.
    // Same identity, ORDINARY inputs: two random activations and their sum, none of them pinned at
    // q8's maximum. If this fails too, saturation at an extreme input is not the explanation.
    printf("\nLINEARITY AT REALISTIC MAGNITUDES -- dot(y1)+dot(y2) vs dot(y1+y2), y random.\n");
    printf("The all-ones probe above pins every activation at q8's maximum, which is the worst\n");
    printf("case for any int16 accumulation; this one does not, so it separates a real fault from\n");
    printf("a saturation artifact. iq4_ks shows the floor three q8_K roundings impose.\n");
    check_linearity_random(GGML_TYPE_IQ4_KS,  256, rng);
    check_linearity_random(GGML_TYPE_IQ4_KSS, 256, rng);
    check_linearity_random(GGML_TYPE_IQ5_KS,  256, rng);

    // FIFTH PROBE: localise the non-linearity inside the row. One 32-weight block live at a time.
    printf("\nPER-BLOCK PROBE -- one 32-weight block activated at a time. Diagnostic, not a gate:\n");
    printf("it reports where the fault is, and the pass/fail verdict stays with the probes above.\n");
    check_per_block(GGML_TYPE_IQ4_KS,  256, rng);
    check_per_block(GGML_TYPE_IQ4_KSS, 256, rng);

    printf("\nCOUNT-SCALING PROBE -- how the error grows with live weights inside ONE block.\n");
    check_count_scaling(GGML_TYPE_IQ4_KS,  256, 0, rng);
    check_count_scaling(GGML_TYPE_IQ4_KSS, 256, 0, rng);

    printf("\nADJACENT vs SPACED -- same live count, differing only in whether they pair up.\n");
    check_adjacent_vs_spaced(GGML_TYPE_IQ4_KS,  256, rng);
    check_adjacent_vs_spaced(GGML_TYPE_IQ4_KSS, 256, rng);
    check_adjacent_vs_spaced(GGML_TYPE_IQ5_KS,  256, rng);

    printf("\n%s -- %d failure(s)\n", g_failures ? "FAIL" : "PASS", g_failures);
    printf("SCOPE: one row, K = 256, host backend, nrc == 1, uniform-random weights, and only\n"
           "types whose vec_dot_type is q8_K (others print a SKIPPED line naming the reason).\n"
           "A PASS says the two decoders agree on these bytes at this shape. It does NOT certify\n"
           "either decoder against the format specification -- they could agree and both be wrong.\n"
           "NOT COVERED: batching, other K, other backends, and every SKIPPED type.\n");
    return g_failures ? 1 : 0;
}
