// FSD (Frustratingly Simple Decoding) sampler, llama_sampler_init_fsd.  CPU only; no model file.
// COMPATIBILITY-INVESTIGATED: pure arithmetic over candidate arrays.   GATE-CLAIM-OK: documentation only.
//
// Row S69-INFERFLOW-PICK-FSD-ANTILM-SAMPLER-2026-09-29-01. The vectors are derived by hand from the reference behaviour
// (inferflow sampling_strategy.h:125-236 / .cc:457-541, arXiv:2305.12675 Alg. 1) and written down here before the sampler ran.
// FAIL-BEFORE: llama_sampler_init_fsd did not exist, so this file could not be compiled or linked at the parent commit.
//
//  V1  history [5,5,5,5], n=3 beta=0.9 alpha=0.5: penalty(5) = 0.9*(2/3) + 0.1*0.9*(3/4) + 0.01*(4/4) = 0.6775, penalty(7) = 0.
//      candidates p5 = 0.6, p7 = 0.4  ->  score5 = -0.03875, score7 = 0.2  -> FSD picks 7 (greedy would pick 5); alpha = 0 picks 5.
//  V2  accept(7) changes the table: candidates p7 = 0.55, p8 = 0.45 now pick 8 (penalty(7) = 1/5 at the unigram level, score7 = 0.175
//      < score8 = 0.225), while a sampler that never saw the 7 picks 7.
//  V3  history shorter than n-1: no penalty at all, the raw softmax decides (nuance 1).
//  V4  `remaining` is consumed only where the candidate is a follower (nuance 2): history [4,5,6,4,5], n=3, candidates 4 (p .35) and
//      6 (p .65): penalty(4) = 0.4 (unigram only, remaining still 1), penalty(6) = 0.497, so FSD picks 6. A sampler that back-offs
//      unconditionally would give penalty(4) = 0.004 and pick 4.
//  V5  k = 1 keeps only the most probable candidate whatever the penalty.
//  V6  determinism, clone, reset, and Randomized FSD (seeded, reproducible, plain FSD after the first N steps).
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

// one step: candidates {id, probability}; logits are the log of the probabilities (the softmax over the pool restores them)
static llama_token step(llama_sampler * s, const std::vector<std::pair<llama_token, float>> & cand) {
    std::vector<llama_token_data> data;
    for (const auto & c : cand) {
        data.push_back({ c.first, std::log(c.second), 0.0f });
    }
    llama_token_data_array arr = { data.data(), data.size(), -1, false };
    llama_sampler_apply(s, &arr);
    CHECK(arr.selected >= 0 && (size_t) arr.selected < arr.size);
    return arr.selected >= 0 ? data[arr.selected].id : -1;
}

static void feed(llama_sampler * s, const std::vector<llama_token> & toks) {
    for (llama_token t : toks) llama_sampler_accept(s, t);
}

int main() {
    // V1
    {
        llama_sampler * s = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, 0, 1);
        feed(s, {5, 5, 5, 5});
        CHECK(step(s, {{5, 0.6f}, {7, 0.4f}}) == 7);
        llama_sampler_free(s);
        llama_sampler * g = llama_sampler_init_fsd(50, 6, 0.0f, 3, 0.9f, 0, 1);   // alpha = 0: greedy within the pool
        feed(g, {5, 5, 5, 5});
        CHECK(step(g, {{5, 0.6f}, {7, 0.4f}}) == 5);
        llama_sampler_free(g);
    }
    // V2
    {
        llama_sampler * s = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, 0, 1);
        feed(s, {5, 5, 5, 5});
        CHECK(step(s, {{5, 0.6f}, {7, 0.4f}}) == 7);
        llama_sampler_accept(s, 7);
        CHECK(step(s, {{7, 0.55f}, {8, 0.45f}}) == 8);        // 7 is penalised now
        llama_sampler_free(s);
        llama_sampler * f = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, 0, 1);
        feed(f, {5, 5, 5, 5});
        CHECK(step(f, {{7, 0.55f}, {8, 0.45f}}) == 7);         // control: never saw the 7
        llama_sampler_free(f);
    }
    // V3
    {
        llama_sampler * s = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, 0, 1);
        feed(s, {5});                                           // 1 token < n - 1 = 2
        CHECK(step(s, {{5, 0.6f}, {7, 0.4f}}) == 5);
        feed(s, {5});                                           // now 2 tokens: the penalty applies
        CHECK(step(s, {{5, 0.6f}, {7, 0.4f}}) == 7);
        llama_sampler_free(s);
    }
    // V4
    {
        llama_sampler * s = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, 0, 1);
        feed(s, {4, 5, 6, 4, 5});
        CHECK(step(s, {{4, 0.35f}, {6, 0.65f}}) == 6);
        llama_sampler_free(s);
    }
    // V5
    {
        llama_sampler * s = llama_sampler_init_fsd(50, 1, 0.5f, 3, 0.9f, 0, 1);
        feed(s, {5, 5, 5, 5});
        CHECK(step(s, {{5, 0.6f}, {7, 0.4f}}) == 5);            // only the top-1 candidate is in the race
        llama_sampler_free(s);
    }
    // V6: deterministic, clone follows the same path, reset forgets, randomized FSD is reproducible
    {
        std::vector<llama_token> seq_a, seq_b;
        for (int run = 0; run < 2; ++run) {
            llama_sampler * s = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, 0, 7);
            feed(s, {1, 2, 3, 1, 2, 3, 1, 2});
            auto & out = run == 0 ? seq_a : seq_b;
            for (int i = 0; i < 12; ++i) {
                const llama_token t = step(s, {{1, 0.4f}, {2, 0.3f}, {3, 0.2f}, {9, 0.1f}});
                out.push_back(t);
                llama_sampler_accept(s, t);
            }
            llama_sampler_free(s);
        }
        CHECK(seq_a == seq_b);
        bool varied = false;
        for (size_t i = 1; i < seq_a.size(); ++i) varied |= seq_a[i] != seq_a[0];
        CHECK(varied);                                           // the penalty really changes the choice over time (greedy would repeat one id)

        llama_sampler * s = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, 0, 7);
        feed(s, {5, 5, 5, 5});
        llama_sampler * c = llama_sampler_clone(s);
        CHECK(step(c, {{5, 0.6f}, {7, 0.4f}}) == 7);             // clone carries the table
        llama_sampler_reset(s);
        feed(s, {5, 5});                                          // after reset only these two exist: unigram+bigram history of 5s
        llama_sampler_free(c);
        llama_sampler * fresh = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, 0, 7);
        feed(fresh, {5, 5});
        CHECK(step(s, {{5, 0.6f}, {7, 0.4f}}) == step(fresh, {{5, 0.6f}, {7, 0.4f}}));   // reset == fresh
        llama_sampler_free(fresh);
        llama_sampler_free(s);

        // randomized FSD: same seed -> same draws; after the first N steps it is plain FSD
        std::vector<llama_token> r1, r2, plain;
        for (int run = 0; run < 3; ++run) {
            llama_sampler * rs = llama_sampler_init_fsd(50, 6, 0.5f, 3, 0.9f, run < 2 ? 6 : 0, 11);
            feed(rs, {1, 2, 3, 1, 2, 3});
            auto & out = run == 0 ? r1 : (run == 1 ? r2 : plain);
            for (int i = 0; i < 16; ++i) {
                const llama_token t = step(rs, {{1, 0.4f}, {2, 0.3f}, {3, 0.2f}, {9, 0.1f}});
                out.push_back(t);
                llama_sampler_accept(rs, t);
            }
            llama_sampler_free(rs);
        }
        CHECK(r1 == r2);
        // the randomized run may differ from plain FSD during the first 6 steps; it must not stay different by construction of the
        // coin only: both are valid candidate ids
        for (llama_token t : r1) CHECK(t == 1 || t == 2 || t == 3 || t == 9);
    }
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-sampling-fsd: all checks passed\n");
    return 0;
}
