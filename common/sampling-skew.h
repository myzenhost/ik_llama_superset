#pragma once

// Skew sampling: reshape the final draw toward the head (skew < 0) or the tail (skew > 0) of the distribution.
//
// Row S69-APHRODITE-PICK-SKEW-SAMPLING-2026-10-18-01 (engine row S69-ENGINE-MERGE-APHRODITE-2026-09-27-01).
//
// SOURCE-READ (rule 1.87):
//   turboderp/exllamav2 exllamav2/exllamav2_ext/ext_sampling.cpp:239-242 and :270-281 (MIT): candidates are sorted by
//   probability, descending, whenever skew != 0; the uniform draw u is replaced by u^(exp(-skew)) and then walked
//   through the cumulative distribution (multinomial_cpu). For a sorted list with cumulative F_j that picks token j
//   with probability F_j^a - F_{j-1}^a, a = exp(skew): the same draw written as a distribution, which is what this
//   file computes, so the host sampler's own draw (and its seed handling) is unchanged.
//   aphrodite-engine aphrodite/v1/sample/sampler.py:463-467 and aphrodite/v1/sample/ops/skew.py:8-37 (repository
//   AGPL-3.0; skew.py carries an Apache-2.0 SPDX header): softmax, cumsum, pow(exp(skew)), diff, log, applied after
//   the ordered samplers (sampler.py:221-319: DRY, penalties, temperature, top-n-sigma, top-a, TFS, eta, epsilon,
//   typical, quadratic, XTC) and BEFORE top_k / top_p and the multinomial draw (sampler.py:469-475).
// SOURCE-NUANCE: Aphrodite's skew.py takes the cumulative sum in TOKEN-ID order (no sort before torch.cumsum), so its
//   result depends on how the vocabulary happens to be numbered; ExLlamaV2 sorts first. This port follows ExLlamaV2
//   (sorted by probability, ties by lower id), so skew means "toward the likely head / the unlikely tail" whatever
//   the vocabulary order. Placement also follows ExLlamaV2 (skew is part of the final multinomial, after top_k /
//   top_p): here it runs after the whole sampler queue, right before the draw, so top_k / top_p cut the model's
//   distribution and skew reshapes only what survives. Aphrodite cuts the skewed distribution instead
//   (SOURCE-READ: upstream/aphrodite-engine/aphrodite/v1/sample/sampler.py:463-475, no code copied).
//   Like both sources it acts only on the random draw: greedy decoding is unaffected.
//   skew.py also guesses whether its input is already a distribution (looks_like_probs); here the input is always
//   logits, so there is no guess.
// Fresh C++; no code copied from either source.

#include "llama.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

// Replaces the candidates' logits by log(F_j^a - F_{j-1}^a), a = exp(skew), over the candidates sorted by
// probability. skew == 0 (or fewer than 2 candidates) leaves the array untouched. Returns true when applied.
inline bool common_sampling_skew_apply(llama_token_data_array * cur_p, float skew) {
    if (cur_p == nullptr || cur_p->size < 2 || skew == 0.0f || !std::isfinite(skew)) {
        return false;
    }
    llama_token_data * d = cur_p->data;
    const size_t       n = cur_p->size;
    std::sort(d, d + n, [](const llama_token_data & x, const llama_token_data & y) {
        return x.logit > y.logit || (x.logit == y.logit && x.id < y.id);
    });
    const float mx = d[0].logit;
    if (!std::isfinite(mx)) {
        return false; // all -inf: nothing to reshape, the draw reports it
    }
    std::vector<double> p(n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        p[i] = std::exp((double) d[i].logit - (double) mx);
        sum += p[i];
    }
    const double a    = std::exp((double) skew);
    double       cum  = 0.0;
    double       prev = 0.0; // F_{j-1}^a
    for (size_t i = 0; i < n; ++i) {
        cum += p[i] / sum;
        const double fa = std::pow(std::min(cum, 1.0), a);
        const double q  = fa - prev;
        prev            = fa;
        d[i].logit      = q > 0.0 ? (float) std::log(q) : -INFINITY;
        d[i].p          = (float) std::max(q, 0.0);
    }
    cur_p->sorted = false; // the new weights need not be descending
    return true;
}
