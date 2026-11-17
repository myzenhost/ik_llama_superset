#pragma once
// Prompt-prefix reuse planner. Spec: docs/agent-reports/koboldcpp-spec.md, feature B. Standard library only.

#include <cstddef>
#include <vector>

struct prompt_reuse_opts {
    bool   require_full_subset = false; // cache must be a complete prefix of the prompt (recurrent models)
    size_t min_to_proceed      = 0;     // reuse below this is discarded
    size_t min_input_to_keep   = 0;     // leave at least this many prompt tokens unreused (0 = no constraint)
};

// Number of leading prompt tokens that need not be recomputed; 0 = start fresh.
size_t prompt_reuse_plan(const std::vector<int> & cached, const std::vector<int> & prompt, const prompt_reuse_opts & opts);
