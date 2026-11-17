#pragma once

// Speculative-decoding acceptance rules, as pure functions of the two distributions (no model, no context), so the
// statistics can be tested directly. Row S69-SPEC-DECODE-REJECTION-SAMPLING-ACCEPTANCE-PICK-2026-09-28-01.
//
// Two rules, both DISTRIBUTION-EXACT (the emitted token is distributed exactly as the target p):
//
//   sample-and-match (what common_sampler_sample_and_accept_n does today): draw y ~ p, accept the draft x iff y == x,
//       emit y.  P(accept) = sum_x p(x) q(x).
//
//   rejection (Leviathan et al. 2023, arXiv 2211.17192, Algorithm 1; Chen et al. 2023, arXiv 2302.01318): accept x
//       with probability min(1, p(x)/q(x)); on rejection emit a draw from norm(max(0, p - q)).
//       P(accept) = sum_x min(p(x), q(x)) >= sum_x p(x) q(x).
//
// The two coincide when q is one-hot (a GREEDY draft: min(1, p/1) = p(x) = sum p q), so the gain exists only for
// stochastically sampled drafts. MEASURED in common/speculative.cpp: both draft-model paths take the ARGMAX
// (:633, :1862 `cur_p->data[0].id`) unless --spec-draft-sample; EAGLE3 keeps its own sampler's draw (:806); the MTP
// path was not read for this. Design read from mistral.rs mistralrs-core/src/sampler.rs:1794-1851 (MIT) and MLC-LLM
// cpp/serve/sampler/gpu_sampler.cc:201-357 (Apache-2.0) -- reimplemented from the papers, no code copied.

#include "llama.h"

#include <random>
#include <vector>

// A (sub-)normalised distribution over token ids, in any order. Ids absent from `ids` have probability 0; the probs
// need not sum to exactly 1 (top-k / top-p supports are renormalised internally where it matters).
struct common_spec_dist {
    std::vector<llama_token> ids;
    std::vector<float>       probs;
};

struct common_spec_accept_result {
    llama_token token;     // the token to emit at this position
    bool        accepted;  // true iff token == the draft token and the draft was kept
    bool        relaxed = false; // accepted ONLY because the draft token lies inside the target's nucleus (common_spec_accept_relaxed)
};

// Probability of `id` under `d`, renormalised to d's own total mass (0 if absent or the total is not positive).
float common_spec_dist_prob(const common_spec_dist & d, llama_token id);

// Rejection rule for one draft position. x is the draft token (drawn from q by the drafter).
common_spec_accept_result common_spec_accept_rejection(const common_spec_dist & p, const common_spec_dist & q,
                                                       llama_token x, std::mt19937 & rng);

// RELAXED rule (opt-in, LOSSY): if the draft token x lies in the target's nucleus -- the smallest set of highest-probability tokens of p
// whose mass reaches top_p (ties by lower id) -- it is accepted whatever p(x)/q(x) is; otherwise the exact rejection rule above decides.
// Inferflow's "flexible accept schema" (report Sec. 8, Algorithm 1; MIT), written from that description. The emitted distribution is
// pulled toward the drafter's, so callers must report result.relaxed separately. top_p <= 0 or >= 1 means "exact rule only".
common_spec_accept_result common_spec_accept_relaxed(const common_spec_dist & p, const common_spec_dist & q,
                                                     llama_token x, std::mt19937 & rng, float top_p);

// Sample-and-match rule for one draft position (the current default), for comparison and tests.
common_spec_accept_result common_spec_accept_sample_match(const common_spec_dist & p, llama_token x, std::mt19937 & rng);

// Analytic acceptance probabilities, for tests and for logging expected vs observed acceptance.
double common_spec_accept_rate_rejection(const common_spec_dist & p, const common_spec_dist & q);   // sum min(p, q)
double common_spec_accept_rate_sample_match(const common_spec_dist & p, const common_spec_dist & q); // sum p * q
