#pragma once

// Gumbel-max text watermark with a keyed Philox PRF, plus its detector.
//
// Reimplemented from the behaviour of vLLM's watermarking package (Apache-2.0,
// https://github.com/vllm-project/vllm @ 924707f1). No code copied.
// SOURCE-READ: upstream/vllm/vllm/v1/watermarking/prfs/philox.py (Philox rounds :40-58,
// context absorption + candidate call :61-125), prfs/base.py:17-19 (uint32 -> uniform),
// gumbel.py:24-39 (Gamma survival) and :65-68 (clamp + noise), detector.py:52-77 (contexts + dedup),
// v1/worker/gpu/sample/watermark.py:105-135 (generation-side repeated-context rule).
// The PRF is bit-compatible with vLLM's CPU reference path, so a text generated here is detected by
// vLLM's GumbelWatermarkDetector with the same key, and vice versa.
//
// Scheme, per generated position:
//   context  = the last `width` GENERATED tokens; positions inside the prompt (or before the start)
//              are -1, so a detector that sees only the completion rebuilds the same contexts.
//   u(v)     = Philox4x32-10(key, context, v) mapped into (0,1).
//   token    = argmax_v( logit_v / T + g_v ),  g_v = -log(-log(u(v))).
//              Gumbel-max with KEYED noise: each token is still an exact sample from softmax(logit/T),
//              so the output distribution is unchanged per token.
//   dedup    = if the current context already occurred among this request's generated tokens, sample
//              that position with ordinary RNG instead. A repeated context reuses the same noise
//              vector, which (a) feeds repetition loops and (b) gives the detector no new evidence.
// Detection: skip repeated contexts, S = sum -log(1 - u(target)); under "not watermarked" each term is
// Exp(1), so S ~ Gamma(n, 1) and the p-value is its survival function at S.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace common_watermark {

constexpr int      DEFAULT_CONTEXT_WIDTH = 4;
constexpr int32_t  DEFAULT_MAX_HISTORY   = 8192;
constexpr double   DEFAULT_P_THRESHOLD   = 0.01;

// The context absorbed into a Philox state, reusable for every candidate token at one position.
struct context_state {
    uint32_t s[4];
    uint32_t key0;
    uint32_t key1;
};

// Absorb `width` context tokens (-1 = no token) under `key`.
context_state absorb(uint64_t key, const int32_t * context, int width);

// u in (0,1) for `token` at an absorbed context. Matches vLLM's float32 value exactly.
float uniform(const context_state & st, int32_t token);

// Convenience: absorb + uniform.
float uniform(uint64_t key, const int32_t * context, int width, int32_t token);

// Context for the NEXT position, given the tokens generated so far (prompt excluded). Writes `width`
// values, left-padded with -1.
void next_context(const std::vector<int32_t> & generated, int width, int32_t * out);

// True when the next position's context already occurred at an earlier generated position, searching
// the most recent `max_history` positions (<= 0 = all). Same rule as vLLM's 'single_turn' scope.
bool next_context_repeated(const std::vector<int32_t> & generated, int width, int32_t max_history);

// Keyed Gumbel-max over candidate (id, logit) pairs. `logits` are already temperature-scaled. Returns
// the index into the arrays, or -1 if n == 0.
int64_t gumbel_select(const context_state & st, const int32_t * ids, const float * logits, size_t n);

struct detection {
    double  score          = 0.0;
    double  p_value        = 1.0;
    int32_t n_scored       = 0;
    bool    is_watermarked = false;
};

// Survival function of Gamma(shape, 1) at x, integer shape (the detector's null distribution).
double gamma_survival_int(double x, int64_t shape);

// Detect over a token sequence that STARTS at the first generated token.
detection detect(uint64_t key, const std::vector<int32_t> & tokens, int width = DEFAULT_CONTEXT_WIDTH,
                 bool dedup = true, double p_threshold = DEFAULT_P_THRESHOLD);

// ---------------------------------------------------------------------------------------------------
// Two further schemes, selectable per server (--watermark-scheme) or per request ("watermark_scheme").
// All three share the server-side key and the generated-only context convention above, so a detector
// needs only the completion. Row S69-ADD-TGI-ALTERNATIVES-LLGUIDANCE-PAGEDKV-RADIX-SYNTHID-2026-09-27-01.
//
// KGW green list -- the scheme TGI ships (Kirchenbauer et al. 2023, arXiv:2301.10226).
//   SOURCE-READ: upstream/text-generation-inference @ clone of 2026-09-27 (Apache-2.0)
//   server/text_generation_server/utils/watermark.py:22-23 (gamma 0.5, delta 2.0), :53 (seed from the
//   previous token), :64-66 (green list = first gamma*|V| of a keyed permutation), :82 (+delta on green
//   logits). No code copied. Departures: membership is a keyed PRF test u(prev, v) < gamma instead of a
//   torch.randperm prefix (same expected green fraction, O(1) per token, no framework RNG to match); the
//   previous token is the previous GENERATED token (-1 for the first), not the prompt's last token.
//   The bias DISTORTS the distribution (that is the scheme); it is applied to raw logits before the
//   truncation samplers, where TGI applies it (before its warpers).
//   Detection: G green of T scored, one-sided binomial test against Bin(T, gamma); z reported too.
//
// SynthID-Text -- tournament sampling (Dathathri et al., Nature 2024, doi:10.1038/s41586-024-08025-4).
//   SOURCE-READ: transformers 5.9.0 (Apache-2.0) generation/logits_process.py:2681-2703 (per-layer
//   update p <- p * (1 + g - E_p[g])), :2768-2803 (LCG hash, multiplier 6364136223846793005, increment 1),
//   :2838-2876 (context hash, then token, then per-layer key), :2750-2765 (a repeated context is not
//   watermarked). No code copied. Departure: g = top bit of a 64-bit mix of the per-layer hash instead of
//   a torch.randint lookup table, so this is NOT bit-compatible with transformers' detectors; layer keys
//   are derived from the one server key. Applied to the candidates the truncation samplers left.
//   Detection: mean g over scored positions and layers; under "not watermarked" each g is a fair coin,
//   so the sum is Bin(n*depth, 0.5).
//   KNOWN WEAKNESS (the paper's own finding): low-entropy text -- short, factual, formulaic, e.g. legal
//   citations and boilerplate -- carries little signal, because a near-deterministic distribution leaves
//   the tournament nothing to choose between. Gumbel and KGW share it to a degree; check num_scored_tokens.

enum scheme : int32_t {
    SCHEME_GUMBEL  = 0,  // default; vLLM-compatible, distribution-preserving per token
    SCHEME_KGW     = 1,  // TGI's green list
    SCHEME_SYNTHID = 2,  // tournament
};

constexpr float   DEFAULT_KGW_GAMMA     = 0.5f;  // TGI watermark.py:22
constexpr float   DEFAULT_KGW_DELTA     = 2.0f;  // TGI watermark.py:23
constexpr int32_t DEFAULT_SYNTHID_DEPTH = 30;    // tournament layers (the paper's m = 30)

const char * scheme_name(int32_t s);
// -1 for an unknown name. Accepts "gumbel", "kgw" (alias "greenlist"), "synthid".
int32_t scheme_from_name(const char * name);

// 64-bit finaliser (splitmix64); every derived key and PRF bit below goes through it.
uint64_t mix64(uint64_t x);

// KGW: is `token` green after `prev_token` (-1 = no previous generated token)?
bool kgw_is_green(uint64_t key, int32_t prev_token, int32_t token, float gamma);
// The same test split in two, so a caller scanning a whole vocabulary absorbs `prev_token` once.
uint64_t kgw_prev_state(uint64_t key, int32_t prev_token);
bool     kgw_green(uint64_t prev_state, int32_t token, float gamma);
// dedup: score each (prev, token) pair once (Kirchenbauer's ignore-repeated-ngrams). score = green fraction.
detection detect_kgw(uint64_t key, const std::vector<int32_t> & tokens, float gamma = DEFAULT_KGW_GAMMA,
                     bool dedup = true, double p_threshold = DEFAULT_P_THRESHOLD);

// SynthID: LCG hash of a `width`-token context (-1 padded), then the g bit of `token` at `layer`.
uint64_t synthid_context_hash(const int32_t * context, int width);
int      synthid_g(uint64_t key, uint64_t ctx_hash, int32_t token, int layer);
// Tournament reweighting of normalised `probs` in place (sum stays 1).
void     synthid_reweight(uint64_t key, uint64_t ctx_hash, const int32_t * ids, float * probs, size_t n, int depth);
// dedup: skip positions whose context already occurred. score = mean g.
detection detect_synthid(uint64_t key, const std::vector<int32_t> & tokens, int width = DEFAULT_CONTEXT_WIDTH,
                         int depth = DEFAULT_SYNTHID_DEPTH, bool dedup = true,
                         double p_threshold = DEFAULT_P_THRESHOLD);

// P(X >= k) for X ~ Bin(n, p), summed in log space.
double binomial_survival(int64_t k, int64_t n, double p);

} // namespace common_watermark
