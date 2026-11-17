// DRY sampler per-token host cost at 8k / 32k / 128k history (row S69-ENGREV-EXLLAMAV2-DRY-BREAKER-PRECOMPUTE-2026-10-27-01,
// combined scope: measure the per-token cost before deciding whether max_ngram / max_occurrences / early-exit caps are worth adding).
//
// CPU only, no model: uses llama_sampler_init_dry_impl, the production path common/sampling.cpp reaches
// (dry_penalty_last_n = -1 means the whole context). Prints microseconds per apply() for two histories:
//   random   : tokens drawn from a 4096 id range (few long repeats)
//   periodic : a 64-token pattern repeated (every suffix repeats everywhere)
// Timing never fails; the test fails only if the periodic history does not penalise the continuing token.
#include "../src/llama-sampling.h"
#include "../src/llama-vocab.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

static double time_apply(const std::vector<llama_token> & hist, int reps, float & out_logit_target, llama_token target) {
    llama_vocab dummy_vocab;
    llama_sampler_dry * smpl = llama_sampler_init_dry_impl(dummy_vocab, (int32_t) hist.size(), 0.8f, 1.75f, 2, -1, nullptr, 0);
    for (llama_token t : hist) {
        llama_sampler_dry_accept(smpl, t);
    }
    std::vector<llama_token_data> cand(16);
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        for (int i = 0; i < 16; ++i) {
            cand[i] = { (llama_token) (target + i), 0.0f, 0.0f };
        }
        llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
        auto t0 = std::chrono::steady_clock::now();
        llama_sampler_dry_apply(smpl, &arr);
        auto t1 = std::chrono::steady_clock::now();
        out_logit_target = cand[0].logit;
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (us < best) { best = us; }
    }
    llama_sampler_dry_free(smpl);
    return best;
}

int main() {
    const int sizes[] = { 8192, 32768, 131072 };
    int rc = 0;
    for (int n : sizes) {
        std::mt19937 rng(1234);
        std::vector<llama_token> rnd(n), per(n);
        for (int i = 0; i < n; ++i) {
            rnd[i] = 100 + (llama_token) (rng() % 4096);
            per[i] = 100 + (llama_token) (i % 64);
        }
        float lr = 0.0f, lp = 0.0f;
        // the pattern continues at index n % 64: that token is the penalised one
        const llama_token next_per = 100 + (llama_token) (n % 64);
        const double us_r = time_apply(rnd, 5, lr, rnd[0]);
        const double us_p = time_apply(per, 5, lp, next_per);
        std::printf("history %6d tokens: random %10.1f us/token   periodic %10.1f us/token (cand0 logit %.3f)\n", n, us_r, us_p, lp);
        if (!(lp < 0.0f)) {
            std::printf("FAIL: periodic history did not penalise the continuing token at n=%d\n", n);
            rc = 1;
        }
    }
    return rc;
}
