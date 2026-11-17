// A candidate list in which every logit is -inf must come back as LLAMA_TOKEN_NULL, not abort the process.
//
// That list is what a grammar/choice the vocabulary cannot satisfy leaves behind (choice=["zzz"] on a vocab with no
// "z"). llama_sample_token_with_rng_impl took max = -inf, every expf(-inf - -inf) became NaN, upper_bound returned
// end(), and GGML_ABORT killed the whole llama-server -- every slot, every client -- for one bad request. Row
// S69-FORK-UNSATISFIABLE-GRAMMAR-ABORTS-SERVER-SAMPLING-839-2026-10-11-01.
//
// The controls keep the assertion honest: a single finite candidate among -inf ones must still be drawn every
// time (a sampler that always answered NULL would pass the first case), and an ordinary row must still return
// one of its own ids.

#include "../src/llama-sampling.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fail = 0;

static void expect(const char * what, bool ok, long got = 0) {
    printf("  %-4s %-62s (%ld)\n", ok ? "ok" : "FAIL", what, got);
    g_fail += !ok;
}

static llama_token draw(llama_sampling & smpl, std::mt19937 & rng, const std::vector<float> & logits) {
    std::vector<llama_token_data> d;
    for (size_t i = 0; i < logits.size(); ++i) {
        d.push_back({ (llama_token) i, logits[i], 0.0f });
    }
    llama_token_data_array arr = { d.data(), d.size(), -1, false };
    return llama_sample_token_with_rng_impl(&smpl, &arr, rng);
}

int main() {
    // ggml_time_us() needs ggml_time_init() on Windows (see test-engine-vllm-sampling.cpp).
    llama_backend_init();

    const float NINF = -INFINITY;
    llama_sampling smpl(8);
    std::mt19937 rng(1234);

    printf("all candidates -inf\n");
    {
        const llama_token id = draw(smpl, rng, { NINF, NINF, NINF, NINF });
        expect("returns LLAMA_TOKEN_NULL instead of aborting", id == LLAMA_TOKEN_NULL, id);
        const llama_token id8 = draw(smpl, rng, std::vector<float>(8, NINF));
        expect("same for a longer list", id8 == LLAMA_TOKEN_NULL, id8);
    }

    printf("control: one finite candidate among -inf\n");
    {
        int hits = 0;
        for (int i = 0; i < 200; ++i) {
            hits += draw(smpl, rng, { NINF, NINF, -3.0f, NINF }) == 2;
        }
        expect("the only allowed id is drawn every time", hits == 200, hits);
    }

    printf("control: ordinary logits\n");
    {
        bool all_valid = true;
        int  seen_low  = 0;
        for (int i = 0; i < 2000; ++i) {
            const llama_token id = draw(smpl, rng, { 1.0f, 0.5f, -0.5f, 0.0f });
            all_valid = all_valid && id >= 0 && id < 4;
            seen_low += id == 2;
        }
        expect("every draw is one of the row's ids", all_valid);
        expect("the lowest-probability id is still drawn sometimes", seen_low > 0, seen_low);
    }

    llama_backend_free();
    printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
