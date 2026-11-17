#pragma once

// Per verify round: take the model drafter's window (MTP), or a lookup (suffix / n-gram) window?
//
// A lookup draft taken whenever it proposes more than the model drafter loses on ordinary text: a long window costs more
// to verify than the usual 3-4 MTP tokens and pays only when enough of it is accepted. This policy learns both sides
// online and compares expected committed tokens per millisecond:
//
//   model drafter  E = mean tokens a window of that size has committed (EMA) / measured round cost of that size
//   lookup         E(k) = 1 + q + q^2 + .. + q^k for k <= the proposal, q = acceptance of lookup drafts whose match was about
//                  as long (4 buckets of match length, decayed counts), / measured round cost of k + 1
//
// and takes the lookup window only when its best E/cost beats the model drafter's by `margin`. Round costs are measured
// per window size (EMA); a size not seen yet is scaled from seen ones by a prior shape. It chooses which drafts are
// verified; verification still decides every committed token, so the output does not change.
//
// Spec: Strata serve draft policy (MIT, Niko1221/Strata @82f46a8c8, src/spec/draft_policy.{hpp,cpp}); written from that
// spec, no code copied. Row S69-STRATA-S2-DRAFT-POLICY-COST-MODEL-MTP-LOOKUP-CHAIN-2026-10-07-01.

#include <array>

class common_draft_policy {
public:
    static constexpr int kMaxT    = 8;
    static constexpr int kBuckets = 4;

    using cost_shape = std::array<double, kMaxT + 1>;

    // Relative round cost by window size, used only for sizes with no measurement yet. This default was fitted on a 12 GB
    // card with CPU-resident experts (steep); a host with the model resident should pass a flatter one.
    static cost_shape default_shape();

    explicit common_draft_policy(int max_t, double margin = 0.03, cost_shape shape = default_shape());

    struct pick {
        bool lookup = false;
        int  t      = 1; // window size (1 + drafts)
    };

    // t_mtp: the model drafter's window; lookup_k: the lookup proposal's length (0 = none); match: its match length.
    pick choose(int t_mtp, int lookup_k, int match) const;
    // After the round: the window it used, drafts accepted, and the round's wall time (verify + commit + draft).
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    // Chain: how many of k_avail lookup tokens to append after a model-drafter window of t_mtp whose drafts all hold with
    // probability p_mtp. Best k by expected tokens per ms, if it beats the window alone by `margin`. 0 = none.
    int  chain(int t_mtp, double p_mtp, int k_avail, int match) const;
    // After a chained round: the window's size, the chained tokens, the window's accepted drafts, its time.
    void observe_chain(int t_mtp, int k, int accepted, int match, double round_ms);

    double chain_rate(int match) const;
    double lookup_rate(int match) const;
    double cost_ms(int t) const;
    int    n_cost_samples(int t) const;

private:
    static int bucket(int match);
    double     mtp_tokens(int t) const;
    void       add_cost(int t, double round_ms);

    int        max_t_;
    double     margin_;
    cost_shape shape_;
    std::array<double, kMaxT + 1> cost_{}, cost_n_{};   // round ms by window size
    std::array<double, kMaxT + 1> mtp_tok_{}, mtp_n_{}; // tokens committed by model-drafter windows of that size
    std::array<double, kBuckets>  ok_{}, bad_{};        // lookup drafts accepted / not accepted (<= 1 per window), decayed
    std::array<double, kBuckets>  cok_{}, cbad_{};      // the same for drafts chained after a model-drafter window
};
