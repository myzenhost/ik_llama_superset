#pragma once

// Boundary V: with a turbo2 V cache, keep the V of the first and last two layers at q8_0 unless the caller already
// chose per-layer V types. Idea from TheTom/turboquant_plus (Apache-2.0, docs/turboquant-recommendations.md,
// "Boundary V (auto-enabled for turbo2-V)"); this is our own implementation on the existing -ctv-first/-ctv-last
// mechanism. OPT-IN here, not the default: LLAMA_TURBO_BOUNDARY_V=1 (TURBO_LAYER_ADAPTIVE=1, the turbo fork's name, is
// honoured too). Measured 2026-10-17 on wikitext-2 c512, -ctk q8_0: it RAISED perplexity on both models tried --
// Olmo-3-7B Q4_K_M 15.4298 -> 15.5017, granite-4.2-8b Q6_K 17.8815 -> 17.9715 -- so the source's default does not carry.

#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

struct llama_turbo_boundary_v {
    ggml_type type_v_first;
    ggml_type type_v_last;
    int32_t   n_v_first;
    int32_t   n_v_last;
    bool      applied;
};

static inline bool llama_turbo_boundary_v_env_on(const char * a, const char * b) {
    return (a && std::strcmp(a, "1") == 0) || (b && std::strcmp(b, "1") == 0);
}

// Pure function so the policy can be tested without a model. env_* are the raw values of the two opt-in variables.
static inline llama_turbo_boundary_v llama_turbo_boundary_v_resolve(
        ggml_type type_v, ggml_type type_v_first, ggml_type type_v_last,
        int32_t n_v_first, int32_t n_v_last, int64_t n_layer,
        const char * env_llama, const char * env_turbo) {
    llama_turbo_boundary_v r = { type_v_first, type_v_last, n_v_first, n_v_last, false };
    const int32_t n_edge = 2;
    const bool user_chose = n_v_first > 0 || n_v_last > 0;
    // Below 4*n_edge layers the boundary would be half the model or more: leave the cache as asked.
    if (type_v != GGML_TYPE_TURBO2_0 || user_chose || n_layer < 4*n_edge ||
        !llama_turbo_boundary_v_env_on(env_llama, env_turbo)) {
        return r;
    }
    r.type_v_first = GGML_TYPE_Q8_0;
    r.type_v_last  = GGML_TYPE_Q8_0;
    r.n_v_first    = n_edge;
    r.n_v_last     = n_edge;
    r.applied      = true;
    return r;
}
