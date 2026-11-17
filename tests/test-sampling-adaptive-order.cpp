// adaptive-p must follow params.adaptive_target, not the presence of ADAPTIVE_P in samplers_sequence
// (upstream ik_llama.cpp #2579). No model needed: common_sampler_init(nullptr, ...) with defaults.
#include "sampling.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

int main() {
    // 1) target >= 0, custom order WITHOUT adaptive-p: the context must still exist (adaptive-p runs last regardless of order)
    {
        common_params_sampling sp;
        sp.samplers_sequence = { llama_sampler_type::TOP_K, llama_sampler_type::TEMPERATURE };
        sp.adaptive_target   = 0.5f;
        common_sampler * s = common_sampler_init(nullptr, sp);
        CHECK(s != nullptr);
        CHECK(s->adapt_p_ctx != nullptr);
        common_sampler_free(s);
    }
    // 2) target < 0 (disabled): no context at init, and none after reset either (reset used to create one from a negative target)
    {
        common_params_sampling sp;
        sp.samplers_sequence = { llama_sampler_type::TOP_K, llama_sampler_type::TEMPERATURE };
        sp.adaptive_target   = -1.0f;
        common_sampler * s = common_sampler_init(nullptr, sp);
        CHECK(s != nullptr);
        CHECK(s->adapt_p_ctx == nullptr);
        common_sampler_reset(s);
        CHECK(s->adapt_p_ctx == nullptr);
        common_sampler_free(s);
    }
    // 3) target >= 0 survives a reset
    {
        common_params_sampling sp;
        sp.samplers_sequence = { llama_sampler_type::TOP_K, llama_sampler_type::ADAPTIVE_P };  // no DRY: it needs a vocab
        sp.adaptive_target = 0.7f;
        common_sampler * s = common_sampler_init(nullptr, sp);
        CHECK(s != nullptr && s->adapt_p_ctx != nullptr);
        common_sampler_reset(s);
        CHECK(s->adapt_p_ctx != nullptr);
        common_sampler_free(s);
    }
    printf("test-sampling-adaptive-order: OK\n");
    return 0;
}
