#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <vector>

// CONTRACT (common/sampling.cpp, common_sampler_accept): the retained window is
//     max(params.n_prev, params.penalty_last_n, no-repeat-ngram window, 1)
// since 7b62d080f - the repeat penalty must see its whole penalty_last_n window even when n_prev is smaller, and a
// reset sampler must not collapse that window to one token. Upstream's #2410 test assumed the window is exactly
// n_prev; with this fork's default penalty_last_n (64) that is no longer true, so each case below pins ONE bound at a time.

static void test_history_grows_to_capacity() {
    common_sampler sampler{};
    sampler.params.n_prev = 3;
    sampler.params.penalty_last_n = 0;  // take the penalty window out so n_prev is the binding bound

    common_sampler_accept(&sampler, nullptr, 10, false);
    common_sampler_accept(&sampler, nullptr, 11, false);
    common_sampler_accept(&sampler, nullptr, 12, false);

    assert((sampler.prev == std::vector<llama_token>{10, 11, 12}));

    common_sampler_accept(&sampler, nullptr, 13, false);

    assert((sampler.prev == std::vector<llama_token>{11, 12, 13}));
}

static void test_penalty_window_widens_history() {
    common_sampler sampler{};
    sampler.params.n_prev = 3;
    sampler.params.penalty_last_n = 5;  // larger than n_prev: the penalty window wins

    for (llama_token t = 10; t < 16; ++t) {
        common_sampler_accept(&sampler, nullptr, t, false);
    }

    assert((sampler.prev == std::vector<llama_token>{11, 12, 13, 14, 15}));
}

static void test_zero_capacity_keeps_only_the_last_token() {
    common_sampler sampler{};
    sampler.params.n_prev = 0;
    sampler.params.penalty_last_n = 0;

    common_sampler_accept(&sampler, nullptr, 10, false);
    common_sampler_accept(&sampler, nullptr, 11, false);

    // The floor of 1 is deliberate: common_sampler_last() reads prev.back() unguarded, so the window never goes empty.
    assert((sampler.prev == std::vector<llama_token>{11}));
}

int main() {
    test_history_grows_to_capacity();
    test_penalty_window_widens_history();
    test_zero_capacity_keeps_only_the_last_token();

    return 0;
}
