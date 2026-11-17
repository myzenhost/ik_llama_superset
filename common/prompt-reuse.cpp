#include "prompt-reuse.h"

#include <algorithm>

size_t prompt_reuse_plan(const std::vector<int> & cached, const std::vector<int> & prompt, const prompt_reuse_opts & opts) {
    if (cached.empty() || prompt.empty()) {
        return 0;
    }

    size_t n = 0;
    if (opts.require_full_subset) {
        if (cached.size() > prompt.size()) {
            return 0;
        }
        if (!std::equal(cached.begin(), cached.end(), prompt.begin())) {
            return 0;
        }
        n = cached.size();
    } else {
        const size_t lim = std::min(cached.size(), prompt.size());
        while (n < lim && cached[n] == prompt[n]) {
            ++n;
        }
        // always leave one cached and one prompt token to evaluate
        n = std::min(n, lim - 1);
    }

    if (n < opts.min_to_proceed) {
        return 0;
    }
    if (opts.min_input_to_keep > 0 && n + opts.min_input_to_keep > prompt.size()) {
        n = prompt.size() > opts.min_input_to_keep ? prompt.size() - opts.min_input_to_keep : 0;
    }
    return n;
}
