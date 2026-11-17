// Unit test for common/sampling-family-defaults.h (the table, the name normaliser and the matcher). Model-free.
// RESEARCH-DONE: upstream/LocalAI/core/config/inference_defaults.json + .go read in full; this test pins the matching rules.
#include "sampling-family-defaults.h"

#include <cmath>
#include <cstdio>
#include <set>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static std::string fam(const char * id) {
    const auto * f = common_family_sampling_match(id);
    return f ? f->pattern : "";
}

int main() {
    // ---- the row's examples -------------------------------------------------------------------------------------------
    CHECK(fam("unsloth/Qwen3.5-9B-GGUF") == "qwen3.5");                   // qwen3.5, not qwen3
    CHECK(fam("Qwen3-Coder-30B-A3B-Q4_K_M.gguf") == "qwen3-coder");       // the longer family wins
    CHECK(fam("llama-unknown") == "");                                    // no match
    CHECK(fam("Qwen3-8B-Q4_K_M.gguf") == "qwen3");

    // ---- the longest pattern wins whatever the table order is ----------------------------------------------------------
    CHECK(fam("gemma-3n-E4B-it.gguf") == "gemma-3n");
    CHECK(fam("gemma-3-27b-it.gguf") == "gemma-3");
    CHECK(fam("Meta-Llama-3.1-8B-Instruct-Q8_0.gguf") == "llama-3.1");
    CHECK(fam("Llama-3.2-1B-Instruct.gguf") == "llama-3.2");
    CHECK(fam("Llama-4-Scout-17B.gguf") == "llama-4");
    CHECK(fam("Kimi-K2-Instruct.gguf") == "kimi-k2");
    CHECK(fam("kimi-vl-a3b.gguf") == "kimi");
    CHECK(fam("Qwen2.5-Coder-7B-Instruct.gguf") == "qwen2.5-coder");
    CHECK(fam("Qwen2.5-7B-Instruct.gguf") == "qwen2.5");
    CHECK(fam("deepseek-r1-distill-qwen-7b.gguf") == "deepseek-r1");
    CHECK(fam("MiniMax-M2.5-GGUF") == "minimax-m2.5");

    // ---- normalisation: case, org prefix, extension, underscores, a Windows path --------------------------------------------
    CHECK(common_family_sampling_normalize("Org/Some_Model-Q4.GGUF") == "some-model-q4");
    CHECK(fam("E:\\models\\gguf\\Phi-4-mini-instruct-Q8_0.gguf") == "phi-4");
    CHECK(fam("E:\\models\\gguf\\Phi_4_mini.gguf") == "phi-4");  // '_' -> '-' makes this the same name
    CHECK(fam("hf.co/bartowski/gpt-oss-20b-GGUF") == "gpt-oss");

    // ---- boundaries: a pattern must start a word and not be followed by a digit ----------------------------------------------
    CHECK(fam("qwen35-instruct.gguf") == "");            // "qwen3" must not claim "qwen35" (a Qwen3.5 file written without the dot)
    CHECK(fam("xqwen3-8b.gguf") == "");                  // no left boundary
    CHECK(fam("tseed-model.gguf") == "");                // "seed" inside a word
    CHECK(fam("Seed-OSS-36B-Instruct.gguf") == "seed");
    CHECK(fam("Qwen3.5-9B") == "qwen3.5");               // the dot version is a pattern of its own
    CHECK(fam("SmolLM2-135M-Instruct-Q8_0.gguf") == "smollm");   // a pattern ending in a letter may be followed by a version digit
    CHECK(fam("glm-45-air.gguf") == "");                          // a pattern ending in a digit may not run into another digit
    CHECK(fam("GLM-4.5-Air.gguf") == "glm-4");                    // but a '.' ends it: the family fallback applies

    // ---- candidate ids: the first that matches any family wins ----------------------------------------------------------------
    {
        std::string which;
        const auto * f = common_family_sampling_match_any({"", "my-private-alias", "Qwen3-14B.gguf", "gemma-3-4b"}, &which);
        CHECK(f && std::string(f->pattern) == "qwen3" && which == "Qwen3-14B.gguf");
        CHECK(common_family_sampling_match_any({"nothing", "here"}) == nullptr);
        CHECK(common_family_sampling_match_any({}) == nullptr);
    }

    // ---- table sanity ----------------------------------------------------------------------------------------------------------
    {
        std::set<std::string> seen;
        for (const auto & e : common_family_sampling_table()) {
            CHECK(seen.insert(e.pattern).second);                    // no duplicate patterns
            CHECK(e.temp >= 0.0f && e.top_p > 0.0f && e.top_p <= 1.0f && e.top_k >= 0 && e.min_p >= 0.0f);
            CHECK(std::string(e.pattern) == common_family_sampling_normalize(e.pattern));   // patterns are stored normalised
        }
        CHECK(seen.size() == 60);
        const auto * q = common_family_sampling_match("Qwen3.5-9B");
        CHECK(q && std::fabs(q->temp - 0.7f) < 1e-6 && q->top_k == 20 && std::fabs(q->penalty_present - 1.5f) < 1e-6);
        const auto * g = common_family_sampling_match("gemma-4-31b");
        CHECK(g && g->penalty_present == 0.0f);                      // set to an explicit 0, not "absent"
        const auto * l = common_family_sampling_match("Llama-3.3-70B");
        CHECK(l && l->top_k == 0);                                   // the table's -1 (disabled) is stored as this engine's 0
    }

    if (fails) {
        std::printf("%d check(s) failed\n", fails);
        return 1;
    }
    std::printf("All tests passed\n");
    return 0;
}
