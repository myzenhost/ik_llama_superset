// Row S69-EXLLAMAV3-PICK-PENALTY-FADE-2026-10-18-01: penalty fade (ExLlamaV3 sustain + decay, MIT).
// Pins common/sampling-penalty-fade.h against values worked by hand from rep_pen.cu:47-84 and :163-199
// (factor 1 inside the sustain range, then 1 - (dist - sustain) / decay; repetition blends linearly
// between the raw and the fully penalised logit by the largest factor; frequency subtracts the sum of
// factors, presence the largest), and the window/accept contract in common_sampler_accept.

#include "sampling.h"
#include "sampling-penalty-fade.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static bool near(float a, float b) { return std::fabs(a - b) <= 1e-5f * (1.0f + std::fabs(b)); }

static void test_factor_shape() {
    // sustain 4, decay 4: dist 1..4 full, 5 -> 0.75, 6 -> 0.5, 7 -> 0.25, 8 -> 0
    const float want[] = { 1, 1, 1, 1, 0.75f, 0.5f, 0.25f, 0.0f, 0.0f };
    for (int d = 1; d <= 9; ++d) {
        assert(near(common_penalty_fade_factor(d, 4, 4), want[d - 1]));
    }
}

static std::vector<llama_token_data> make_cands(const std::vector<float> & logits) {
    std::vector<llama_token_data> c;
    for (size_t i = 0; i < logits.size(); ++i) c.push_back({ (llama_token) i, logits[i], 0.0f });
    return c;
}

static void test_repetition_blend() {
    // vocab 4. history oldest -> newest: 1 2 3 0 ; sustain 1, decay 2.
    // dist: token0 = 1 (f 1), token3 = 2 (f 0.5), token2 = 3 (f 0, outside), token1 = 4 (outside)
    std::vector<llama_token> hist = { 1, 2, 3, 0 };
    auto c = make_cands({ 2.0f, 2.0f, -2.0f, 2.0f });
    llama_token_data_array a = { c.data(), c.size(), -1, true };
    const size_t n = common_penalty_fade_apply(&a, hist.data(), hist.size(), 1, 2, 2.0f, 0.0f, 0.0f);
    assert(n == 2);
    assert(near(c[0].logit, 1.0f));                 // full: 2 / 2
    assert(near(c[3].logit, 2.0f * 0.5f + 1.0f * 0.5f)); // half way between 2 and 1 = 1.5
    assert(near(c[1].logit, 2.0f));                 // outside the window: untouched
    assert(near(c[2].logit, -2.0f));                // factor 0: untouched
    assert(!a.sorted);
}

static void test_negative_logit_and_max_factor() {
    // a token seen twice keeps its LARGEST factor for repetition (rep_pen.cu:64 shmemAtomicMaxF)
    // history: 5 0 0 9 ; sustain 0, decay 4 -> token0 at dist 2 (f 0.5) and 3 (f 0.25): max 0.5
    std::vector<llama_token> hist = { 5, 0, 0, 9 };
    auto c = make_cands({ -1.0f });
    llama_token_data_array a = { c.data(), c.size(), -1, false };
    common_penalty_fade_apply(&a, hist.data(), hist.size(), 0, 4, 3.0f, 0.0f, 0.0f);
    // full = -1 * 3 = -3 ; blend 0.5 -> -2
    assert(near(c[0].logit, -2.0f));
}

static void test_frequency_sums_presence_maxes() {
    // same history: token0 factors 0.5 + 0.25 -> f_sum 0.75, f_max 0.5
    std::vector<llama_token> hist = { 5, 0, 0, 9 };
    auto c = make_cands({ 1.0f });
    llama_token_data_array a = { c.data(), c.size(), -1, false };
    common_penalty_fade_apply(&a, hist.data(), hist.size(), 0, 4, 1.0f, 0.4f, 0.2f);
    assert(near(c[0].logit, 1.0f - 0.4f * 0.75f - 0.2f * 0.5f));
}

static void test_inside_sustain_matches_uniform_penalty() {
    // every occurrence inside the sustain range: identical to the classic count-based penalty
    std::vector<llama_token> hist = { 0, 1, 0 };
    auto c = make_cands({ 3.0f, -3.0f, 3.0f });
    llama_token_data_array a = { c.data(), c.size(), -1, false };
    common_penalty_fade_apply(&a, hist.data(), hist.size(), 8, 8, 1.5f, 0.1f, 0.3f);
    assert(near(c[0].logit, 3.0f / 1.5f - 0.1f * 2 - 0.3f));
    assert(near(c[1].logit, -3.0f * 1.5f - 0.1f * 1 - 0.3f));
    assert(near(c[2].logit, 3.0f));
}

static void test_empty_and_null() {
    auto c = make_cands({ 1.0f });
    llama_token_data_array a = { c.data(), c.size(), -1, true };
    assert(common_penalty_fade_apply(&a, nullptr, 0, 4, 4, 2.0f, 1.0f, 1.0f) == 0);
    assert(a.sorted);
    assert(common_penalty_fade_apply(nullptr, nullptr, 0, 4, 4, 2.0f, 1.0f, 1.0f) == 0);
}

static void test_history_covers_the_decay_range() {
    // common_sampler_accept must keep penalty_last_n + penalty_decay tokens, or the fade never sees its tail.
    common_sampler sampler{};
    sampler.params.n_prev = 2;
    sampler.params.penalty_last_n = 3;
    sampler.params.penalty_decay = 4;
    for (llama_token t = 0; t < 20; ++t) {
        common_sampler_accept(&sampler, nullptr, t, false);
    }
    assert(sampler.prev.size() == 7);
    assert(sampler.prev.front() == 13 && sampler.prev.back() == 19);
}

int main() {
    test_factor_shape();
    test_repetition_blend();
    test_negative_logit_and_max_factor();
    test_frequency_sums_presence_maxes();
    test_inside_sustain_matches_uniform_penalty();
    test_empty_and_null();
    test_history_covers_the_decay_range();
    std::printf("test-sampling-penalty-fade: all passed\n");
    return 0;
}
