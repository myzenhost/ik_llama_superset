// adaptive-p must never draw a candidate that an earlier stage masked (grammar, logit_bias ban).
//
// llama_sample_adaptive_p_impl rewrites every candidate's logit from its distance to the target
// probability. A masked candidate (logit -inf, p = 0) is at distance `target` from it, which is a
// FINITE logit, so without the skip it becomes drawable. ikawrakow/ik_llama.cpp #2337 (0ed847d31)
// added `if (logit <= -999.f) continue;`; it arrived here with the ik main merge. Row
// S69-THIREUS-PICK-ADAPTIVE-P-MASKED-TOKEN-QC-2026-09-29-01.
//
// The test drives the same three calls common_sampler_sample makes, on a fresh candidate array per
// draw, and counts which ids come out. The controls matter as much as the assertion: an unmasked
// low-probability token must still be drawn (otherwise a sampler that never samples anything
// would pass), and a build without the skip must fail (checked by mutation, see the row).

#include "../src/llama-sampling.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail = 0;

static void expect(const char * what, bool ok, long got = 0) {
    printf("  %-4s %-58s (%ld)\n", ok ? "ok" : "FAIL", what, got);
    g_fail += !ok;
}

// One draw over a fresh copy of `logits`; returns the drawn id.
static llama_token draw(llama_sampling & smpl, llama_sampler_adaptive_p * ap, std::mt19937 & rng,
                        const std::vector<float> & logits) {
    std::vector<llama_token_data> data(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        data[i] = { (llama_token) i, logits[i], 0.0f };
    }
    llama_token_data_array cur = { data.data(), data.size(), /*selected=*/ -1, /*sorted=*/ false };
    llama_prep_adaptive_p_impl(&smpl, &cur, ap);
    llama_sample_adaptive_p_impl(&smpl, &cur, ap);
    return llama_sample_token_adaptive_p_impl(&smpl, &cur, ap, rng);
}

int main() {
    // ggml_time_us() divides by the timer frequency, which ggml_time_init() sets; without it the
    // samplers' timing calls fault with an integer divide by zero on Windows.
    ggml_time_init();

    const float ninf = -INFINITY;
    llama_sampling smpl(4);
    std::mt19937 rng(1234);

    {
        // three live tokens and one masked; target 0.1 sits where the masked one would be drawn most
        const std::vector<float> logits = { 2.0f, 1.0f, 0.0f, ninf };
        llama_sampler_adaptive_p * ap = llama_init_adaptive_p_impl(0.10f, 0.9f, /*updt_w_cur=*/false, 42);
        long n[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < 5000; ++i) {
            const llama_token id = draw(smpl, ap, rng, logits);
            if (id >= 0 && id < 4) { n[id]++; }
        }
        llama_free_adaptive_p(ap);
        expect("masked token is never drawn (5000 draws, target 0.10)", n[3] == 0, n[3]);
        expect("control: the lowest live token is still drawn", n[2] > 0, n[2]);
        expect("control: draws add up", n[0] + n[1] + n[2] + n[3] == 5000, n[0] + n[1] + n[2] + n[3]);
    }

    {
        // same with the current-probability update path
        const std::vector<float> logits = { 2.0f, 1.0f, 0.0f, ninf };
        llama_sampler_adaptive_p * ap = llama_init_adaptive_p_impl(0.10f, 0.9f, /*updt_w_cur=*/true, 43);
        long masked = 0, live = 0;
        for (int i = 0; i < 5000; ++i) {
            const llama_token id = draw(smpl, ap, rng, logits);
            (id == 3 ? masked : live)++;
        }
        llama_free_adaptive_p(ap);
        expect("masked token is never drawn, updt_w_cur path", masked == 0, masked);
        expect("control: live draws happen, updt_w_cur path", live == 5000, live);
    }

    {
        // a single survivor: always that token, no NaN, no abort
        const std::vector<float> logits = { 3.0f, ninf, ninf, ninf };
        llama_sampler_adaptive_p * ap = llama_init_adaptive_p_impl(0.30f, 0.9f, false, 44);
        long other = 0;
        for (int i = 0; i < 500; ++i) {
            if (draw(smpl, ap, rng, logits) != 0) { other++; }
        }
        llama_free_adaptive_p(ap);
        expect("single live token: always drawn", other == 0, other);
    }

    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
