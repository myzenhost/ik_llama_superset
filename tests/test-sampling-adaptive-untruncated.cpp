// PRIOR-ART-NONE: this file is a unit test for a new sampler option; it switches no capability off (the word "disabled" below is the
// documented off-value of a sampler parameter, which is what the predicate under test reads).
//
// --adaptive-p-untruncated: the predicate that decides whether adaptive-p runs with nothing truncating in front of it, the
// refusal text for REFUSE mode, and the mode-name mapping. The floor itself (min_p applied inside common_sampler_sample) needs a
// model context and is exercised against a live CPU server by tools/server/tests/unit/test_adaptive_untruncated.py.
//
// Measured basis for the feature: temp/research/findings/S69_ADAPTIVE_P_UNTRUNCATED_REFUSE_VS_FLOOR_MEASURED_2026-09-30.md
// (Llama-3.2-1B: untruncated adaptive-p drew 99.8% tokens below 1e-3 probability).
//
// Every assertion has a partner that must come out the other way, so a predicate that is always true or always false fails.

#include "sampling.h"

#include <cstdio>
#include <string>

static int g_fail = 0;

static void expect(const char * what, bool ok) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    g_fail += !ok;
}

// adaptive-p active, every truncating sampler at its documented off value
static common_params_sampling untruncated_params() {
    common_params_sampling p;
    p.adaptive_target = 0.3f;
    p.temp            = 0.8f;
    p.mirostat        = 0;
    p.top_k           = 0;
    p.top_p           = 1.0f;
    p.min_p           = 0.0f;
    p.tfs_z           = 1.0f;
    p.typical_p       = 1.0f;
    p.top_n_sigma     = 0.0f;
    p.top_a           = 0.0f;
    p.epsilon_cutoff  = 0.0f;
    p.eta_cutoff      = 0.0f;
    return p;
}

int main() {
    {
        const auto p = untruncated_params();
        expect("control: all truncators off -> untruncated", common_sampling_adaptive_p_untruncated(p));
        expect("control: all truncators off -> truncates() is false", !common_sampling_truncates(p));
    }
    {
        // each truncating sampler alone flips the answer; the default sequence contains all of them
        struct Case { const char * name; void (*set)(common_params_sampling &); };
        const Case cases[] = {
            { "top_k = 40",            [](common_params_sampling & p) { p.top_k = 40; } },
            { "top_p = 0.95",          [](common_params_sampling & p) { p.top_p = 0.95f; } },
            { "min_p = 0.05",          [](common_params_sampling & p) { p.min_p = 0.05f; } },
            { "tfs_z = 0.9",           [](common_params_sampling & p) { p.tfs_z = 0.9f; } },
            { "typical_p = 0.9",       [](common_params_sampling & p) { p.typical_p = 0.9f; } },
            { "top_n_sigma = 1.0",     [](common_params_sampling & p) { p.top_n_sigma = 1.0f; } },
            { "top_a = 0.1",           [](common_params_sampling & p) { p.top_a = 0.1f; } },
            { "epsilon_cutoff = 0.01", [](common_params_sampling & p) { p.epsilon_cutoff = 0.01f; } },
            { "eta_cutoff = 0.01",     [](common_params_sampling & p) { p.eta_cutoff = 0.01f; } },
        };
        for (const auto & c : cases) {
            auto p = untruncated_params();
            c.set(p);
            expect((std::string(c.name) + " -> truncates, no longer untruncated").c_str(),
                   common_sampling_truncates(p) && !common_sampling_adaptive_p_untruncated(p));
        }
    }
    {
        // a truncator that is set but NOT in the sampler sequence does nothing
        auto p = untruncated_params();
        p.top_k = 40;
        p.samplers_sequence = { llama_sampler_type::TEMPERATURE, llama_sampler_type::ADAPTIVE_P, llama_sampler_type::DIST };
        expect("top_k = 40 but TOP_K not in samplers_sequence -> untruncated", common_sampling_adaptive_p_untruncated(p));
        p.samplers_sequence = { llama_sampler_type::TOP_K, llama_sampler_type::ADAPTIVE_P, llama_sampler_type::DIST };
        expect("control: same top_k = 40 with TOP_K in the sequence -> truncated", !common_sampling_adaptive_p_untruncated(p));
    }
    {
        // adaptive-p must be active at all
        auto p = untruncated_params();
        p.adaptive_target = -1.0f;
        expect("adaptive_target < 0 -> not an adaptive-p case", !common_sampling_adaptive_p_untruncated(p));
        p = untruncated_params();
        p.temp = 0.0f;
        expect("temp == 0 (greedy) -> not an adaptive-p case", !common_sampling_adaptive_p_untruncated(p));
        p = untruncated_params();
        p.mirostat = 2;
        expect("mirostat on -> not an adaptive-p case", !common_sampling_adaptive_p_untruncated(p));
        p = untruncated_params();
        p.samplers_sequence = { llama_sampler_type::TEMPERATURE, llama_sampler_type::DIST };
        expect("ADAPTIVE_P not in the sequence -> not an adaptive-p case", !common_sampling_adaptive_p_untruncated(p));
        expect("control: the baseline case is an adaptive-p case", common_sampling_adaptive_p_untruncated(untruncated_params()));
    }
    {
        // the stock defaults (top_k 40, top_p 0.95, min_p 0.05) truncate: a plain adaptive_target request is NOT touched
        common_params_sampling p;
        p.adaptive_target = 0.3f;
        expect("default sampling params + adaptive_target -> truncated (no floor, no refusal)", !common_sampling_adaptive_p_untruncated(p));
    }
    {
        auto p = untruncated_params();
        p.adaptive_untruncated = COMMON_ADAPTIVE_UNTRUNCATED_REFUSE;
        const std::string msg = common_sampling_adaptive_p_refusal(p);
        expect("REFUSE + untruncated -> a refusal message", !msg.empty());
        expect("refusal names a way out (top_k)", msg.find("top_k") != std::string::npos);
        p.adaptive_untruncated = COMMON_ADAPTIVE_UNTRUNCATED_FLOOR;
        expect("FLOOR + untruncated -> no refusal", common_sampling_adaptive_p_refusal(p).empty());
        p.adaptive_untruncated = COMMON_ADAPTIVE_UNTRUNCATED_ALLOW;
        expect("ALLOW + untruncated -> no refusal", common_sampling_adaptive_p_refusal(p).empty());
        p.adaptive_untruncated = COMMON_ADAPTIVE_UNTRUNCATED_REFUSE;
        p.top_k = 40;
        expect("REFUSE + truncated -> no refusal", common_sampling_adaptive_p_refusal(p).empty());
    }
    {
        expect("name floor  -> FLOOR",  common_adaptive_untruncated_from_name("floor")  == COMMON_ADAPTIVE_UNTRUNCATED_FLOOR);
        expect("name refuse -> REFUSE", common_adaptive_untruncated_from_name("refuse") == COMMON_ADAPTIVE_UNTRUNCATED_REFUSE);
        expect("name allow  -> ALLOW",  common_adaptive_untruncated_from_name("allow")  == COMMON_ADAPTIVE_UNTRUNCATED_ALLOW);
        expect("control: unknown name -> -1", common_adaptive_untruncated_from_name("nope") == -1);
        expect("control: null name -> -1",    common_adaptive_untruncated_from_name(nullptr) == -1);
        expect("value -> name round trip", std::string(common_adaptive_untruncated_name(COMMON_ADAPTIVE_UNTRUNCATED_REFUSE)) == "refuse");
        expect("control: unknown value -> nullptr", common_adaptive_untruncated_name(99) == nullptr);
        common_params_sampling p;
        expect("defaults: mode FLOOR, floor 0.02", p.adaptive_untruncated == COMMON_ADAPTIVE_UNTRUNCATED_FLOOR && p.adaptive_floor == 0.02f);
    }

    printf("%s -- %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
