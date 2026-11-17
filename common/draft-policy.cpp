#include "draft-policy.h"

#include <algorithm>

namespace {

constexpr double kCostAlpha = 0.1;   // EMA weight of a new round time
constexpr double kTokAlpha  = 0.05;  // EMA weight of a new model-drafter window outcome
constexpr double kDecay     = 0.97;  // lookup counts: older windows fade
constexpr double kPriorN    = 4.0;   // the priors below are worth this many observations
constexpr int    kProbes    = 3;     // a window size is tried this often before its guessed cost can veto it

// Acceptance of a lookup draft before its bucket has data: the longer the match, the likelier the continuation.
constexpr double kPriorQ[common_draft_policy::kBuckets]      = {0.75, 0.88, 0.93, 0.96};
// A chained draft continues the model drafter's own drafts: weaker evidence than a plain match of that length.
constexpr double kPriorChainQ[common_draft_policy::kBuckets] = {0.45, 0.65, 0.80, 0.90};

}  // namespace

common_draft_policy::cost_shape common_draft_policy::default_shape() {
    return {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6};
}

common_draft_policy::common_draft_policy(int max_t, double margin, cost_shape shape)
    : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin), shape_(shape) {
    // a shape entry that is not positive would divide by zero when scaling: fall back to the default there
    const cost_shape def = default_shape();
    for (int t = 1; t <= kMaxT; ++t) {
        if (!(shape_[t] > 0.0)) {
            shape_[t] = def[t];
        }
    }
}

int common_draft_policy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double common_draft_policy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double common_draft_policy::chain_rate(int match) const {
    const int b = bucket(match);
    return (cok_[b] + kPriorN * kPriorChainQ[b]) / (cok_[b] + cbad_[b] + kPriorN);
}

int common_draft_policy::n_cost_samples(int t) const {
    return (int) cost_n_[std::clamp(t, 1, kMaxT)];
}

double common_draft_policy::cost_ms(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (cost_n_[t] > 0) {
        return cost_[t];
    }
    // scale from the measured sizes, each weighted by how often it was seen
    double num = 0.0, den = 0.0;
    for (int u = 1; u <= kMaxT; ++u) {
        if (cost_n_[u] > 0) {
            const double w = std::min(cost_n_[u], 20.0);
            num += w * cost_[u] * shape_[t] / shape_[u];
            den += w;
        }
    }
    return den > 0 ? num / den : shape_[t];
}

double common_draft_policy::mtp_tokens(int t) const {
    t = std::clamp(t, 1, kMaxT);
    if (mtp_n_[t] > 0) {
        return mtp_tok_[t];
    }
    return 1.0 + 0.7 * (t - 1); // before any window of this size: a typical acceptance
}

void common_draft_policy::add_cost(int t, double round_ms) {
    if (!(round_ms > 0)) {
        return;
    }
    cost_[t]    = cost_n_[t] > 0 ? (1.0 - kCostAlpha) * cost_[t] + kCostAlpha * round_ms : round_ms;
    cost_n_[t] += 1.0;
}

common_draft_policy::pick common_draft_policy::choose(int t_mtp, int lookup_k, int match) const {
    pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) {
        return p;
    }
    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q    = lookup_rate(match);
    double e = 1.0, qi = 1.0, best = 0.0;
    int best_t = 0;
    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e  += qi;
        const double r = e / cost_ms(k + 1);
        if (r > best) {
            best   = r;
            best_t = k + 1;
        }
    }
    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t      = best_t;
        return p;
    }
    // a guessed cost could keep a size from ever being measured: the first few times a confident lookup would need a size
    // not measured yet, it is tried (verification keeps the output; only that round's speed is at stake)
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (t_full > p.t && cost_n_[t_full] < kProbes && q >= 0.85) {
        p.lookup = true;
        p.t      = t_full;
    }
    return p;
}

void common_draft_policy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, kMaxT);
    add_cost(t, round_ms);
    if (lookup) {
        // per-draft outcomes: every accepted draft is one success; a window that stopped short evaluated exactly one more
        // draft, which was rejected (the ones after it were never judged), so q = ok / (ok + bad) is per evaluated draft
        const int b = bucket(match);
        ok_[b]  = kDecay * ok_[b]  + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t]  = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t]   += 1.0;
    }
}

int common_draft_policy::chain(int t_mtp, double p_mtp, int k_avail, int match) const {
    t_mtp = std::clamp(t_mtp, 1, max_t_);
    const int kmax = std::min(k_avail, max_t_ - t_mtp);
    if (kmax <= 0) {
        return 0;
    }
    const double e0 = mtp_tokens(t_mtp), base = e0 / cost_ms(t_mtp), c = chain_rate(match);
    p_mtp = std::clamp(p_mtp, 0.0, 1.0);
    double gain = 0.0, ci = 1.0, best = 0.0;
    int best_k = 0;
    for (int k = 1; k <= kmax; ++k) {
        ci   *= c;
        gain += ci;
        const double r = (e0 + p_mtp * gain) / cost_ms(t_mtp + k);
        if (r > best) {
            best   = r;
            best_k = k;
        }
    }
    if (best_k > 0 && best > base * (1.0 + margin_)) {
        return best_k;
    }
    // as choose(): a size whose cost is only guessed is tried a few times when the continuation looks likely
    const int t_full = t_mtp + kmax;
    if (cost_n_[t_full] < kProbes && p_mtp * c >= 0.6) {
        return kmax;
    }
    return 0;
}

void common_draft_policy::observe_chain(int t_mtp, int k, int accepted, int match, double round_ms) {
    t_mtp = std::clamp(t_mtp, 1, kMaxT);
    const int t = std::clamp(t_mtp + k, 1, kMaxT);
    add_cost(t, round_ms);
    // the model-drafter part is an ordinary window as far as its own drafts go
    const int    mtp_acc = std::min(accepted, t_mtp - 1);
    const double got     = mtp_acc + 1.0;
    mtp_tok_[t_mtp]  = mtp_n_[t_mtp] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t_mtp] + kTokAlpha * got : got;
    mtp_n_[t_mtp]   += 1.0;
    if (accepted < t_mtp - 1) {
        return; // the chain was never reached: it says nothing about the lookup
    }
    const int b = bucket(match), ok = accepted - (t_mtp - 1);
    cok_[b]  = kDecay * cok_[b]  + ok;
    cbad_[b] = kDecay * cbad_[b] + (ok < k ? 1.0 : 0.0);
}
