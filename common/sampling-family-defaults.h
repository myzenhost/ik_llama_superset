// Model-family sampling recommendations, used as a FALLBACK for a GGUF that carries no general.sampling.* keys
// (--sampling-family-defaults, default off).
// RESEARCH-DONE: upstream/LocalAI/core/config/inference_defaults.json + inference_defaults.go (MIT) read in full 2026-10-05.
//
// Most downloaded quants predate the general.sampling.* convention, so the server runs the built-in defaults (temp 0.8, top_k 40,
// top_p 0.95, min_p 0.05) even where the vendor publishes something else (Qwen3 thinking: 0.6 / 20 / 0.95 / 0). This table holds the
// vendor-recommended values per family. It is consulted only when (a) the flag is given, (b) the model has NO general.sampling.* key,
// and (c) per value, no explicit flag was passed; every value applied is logged with its source. Reproducible benchmarks must not
// rely on it: it is off unless asked for.
//
// Header-only (the table, the name normaliser and the matcher are pure); the application to common_params_sampling lives in
// common/sampling.cpp. Tested by tests/test-sampling-family-defaults.cpp.
//
// DATA SOURCE: mudler/LocalAI core/config/inference_defaults.json (MIT), itself generated from Unsloth's inference_defaults.json
// (Apache-2.0); carried here as a C++ table, attribution kept. The matching rules are my own:
//   - LocalAI lowercases, strips the "org/" prefix and ".gguf", maps '_' to '-', and takes the FIRST pattern of a hand-ordered list
//     that is a plain substring of the result. This port takes the LONGEST pattern that matches instead, so the answer does not depend
//     on the order of the table ("gemma-3n" over "gemma-3", "qwen3-coder" over "qwen3", "kimi-k2" over "kimi").
//   - a pattern must start at the beginning of the name or after a non-alphanumeric character, and a pattern that ends in a digit must
//     not run into another digit, so "qwen3" does not claim "qwen35-..." (a Qwen3.5 file written without the dot gets nothing rather
//     than the Qwen3 values), "seed" does not claim "...tseed...", while "smollm" still claims "smollm2".
// SOURCE-NUANCE (LocalAI behaviour a description would not say): (1) in LocalAI a repeat_penalty or presence_penalty of 0 counts as
// "unset", so an explicit 0 from the user is overwritten; here the real set of explicit flags (args_seen) decides. (2) top_k -1 in the
// table means "disabled"; this engine's disabled value is 0 (top_k <= 0 is off), so -1 is stored as 0. (3) several rows carry
// presence_penalty 1.5 (qwen3.5 and later); the field is absent in the others, and an absent field leaves the setting alone.
// (4) each candidate model id is tried in turn and the first that matches any family wins (name, alias, file name, repo).
#pragma once

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

struct common_family_sampling {
    const char * pattern;  // normalised substring identifying the family
    float        temp;
    float        top_p;
    int          top_k;    // 0 = disabled (the table's -1 is stored as 0)
    float        min_p;
    float        penalty_repeat;
    float        penalty_present; // < 0 = the family sets none
};

inline const std::vector<common_family_sampling> & common_family_sampling_table() {
    static const std::vector<common_family_sampling> t = {
        { "moss-tts-local-transformer-v1.5", 1.7f, 0.8f,  25, 0.0f,  1.0f, -1.0f },
        { "moss-tts-nano",                   1.7f, 0.8f,  25, 0.0f,  1.0f, -1.0f },
        { "qwen3.8",                         0.7f, 0.8f,  20, 0.0f,  1.0f,  1.5f },
        { "qwen3.6",                         0.7f, 0.8f,  20, 0.0f,  1.0f,  1.5f },
        { "qwen3.5",                         0.7f, 0.8f,  20, 0.0f,  1.0f,  1.5f },
        { "qwen3-coder",                     0.7f, 0.8f,  20, 0.0f,  1.0f, -1.0f },
        { "qwen3-next",                      0.7f, 0.8f,  20, 0.0f,  1.0f, -1.0f },
        { "qwen3-vl",                        0.7f, 0.8f,  20, 0.0f,  1.0f, -1.0f },
        { "qwen3",                           0.6f, 0.95f, 20, 0.0f,  1.0f, -1.0f },
        { "qwen2.5-coder",                   1.5f, 0.95f,  0, 0.1f,  1.0f, -1.0f },
        { "qwen2.5-vl",                      1.5f, 0.95f,  0, 0.1f,  1.0f, -1.0f },
        { "qwen2.5-omni",                    0.7f, 0.8f,  20, 0.0f,  1.0f, -1.0f },
        { "qwen2.5-math",                    0.7f, 0.8f,  20, 0.0f,  1.0f, -1.0f },
        { "qwen2.5",                         0.7f, 0.8f,  20, 0.0f,  1.0f, -1.0f },
        { "qwen2-vl",                        1.5f, 0.95f,  0, 0.1f,  1.0f, -1.0f },
        { "qwen2",                           0.7f, 0.8f,  20, 0.0f,  1.0f, -1.0f },
        { "qwq",                             0.6f, 0.95f, 40, 0.0f,  1.0f, -1.0f },
        { "gemma-4",                         1.0f, 0.95f, 64, 0.0f,  1.0f,  0.0f },
        { "gemma-3n",                        1.0f, 0.95f, 64, 0.0f,  1.0f, -1.0f },
        { "gemma-3",                         1.0f, 0.95f, 64, 0.0f,  1.0f, -1.0f },
        { "medgemma",                        1.0f, 0.95f, 64, 0.0f,  1.0f, -1.0f },
        { "gemma-2",                         1.0f, 0.95f, 64, 0.0f,  1.0f, -1.0f },
        { "muse-glimmer",                    1.0f, 0.95f, 64, 0.0f,  1.0f, -1.0f },
        { "llama-4",                         1.0f, 0.9f,   0, 0.01f, 1.0f, -1.0f },
        { "llama-3.3",                       1.5f, 0.95f,  0, 0.1f,  1.0f, -1.0f },
        { "llama-3.2",                       1.5f, 0.95f,  0, 0.1f,  1.0f, -1.0f },
        { "llama-3.1",                       1.5f, 0.95f,  0, 0.1f,  1.0f, -1.0f },
        { "llama-3",                         1.5f, 0.95f,  0, 0.1f,  1.0f, -1.0f },
        { "phi-4",                           0.8f, 0.95f,  0, 0.0f,  1.0f, -1.0f },
        { "phi-3",                           0.7f, 0.9f,   0, 0.01f, 1.0f, -1.0f },
        { "mistral-nemo",                    0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "mistral-small",                   0.15f, 0.95f, 0, 0.01f, 1.0f, -1.0f },
        { "mistral-large",                   0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "magistral",                       0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "ministral",                       0.15f, 0.95f, 0, 0.01f, 1.0f, -1.0f },
        { "devstral",                        0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "pixtral",                         1.5f, 0.95f,  0, 0.1f,  1.0f, -1.0f },
        { "deepseek-v4",                     1.0f, 1.0f,   0, 0.0f,  1.0f, -1.0f },
        { "deepseek-r1",                     0.6f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "deepseek-v3",                     0.6f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "deepseek-ocr",                    0.0f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "glm-5",                           1.0f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "glm-4",                           1.0f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "nemotron",                        1.0f, 1.0f,   0, 0.01f, 1.0f, -1.0f },
        { "minimax-m2.7",                    1.0f, 0.95f, 40, 0.01f, 1.0f, -1.0f },
        { "minimax-m2.5",                    1.0f, 0.95f, 40, 0.01f, 1.0f, -1.0f },
        { "minimax",                         1.0f, 0.95f, 40, 0.01f, 1.0f, -1.0f },
        { "gpt-oss",                         1.0f, 1.0f,   0, 0.01f, 1.0f, -1.0f },
        { "granite-4",                       0.0f, 1.0f,   0, 0.01f, 1.0f, -1.0f },
        { "kimi-k3",                         1.0f, 0.95f,  0, 0.0f,  1.0f, -1.0f },
        { "kimi-k2",                         0.6f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "kimi",                            0.6f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "lfm2",                            0.1f, 0.1f,  50, 0.15f, 1.05f, -1.0f },
        { "smollm",                          0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "olmo",                            0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "falcon",                          0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "ernie",                           0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "seed",                            0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "grok",                            1.0f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
        { "mimo",                            0.7f, 0.95f,  0, 0.01f, 1.0f, -1.0f },
    };
    return t;
}

// lowercase, drop everything up to the last '/' or '\\', drop a trailing ".gguf", map '_' to '-'
inline std::string common_family_sampling_normalize(const std::string & id) {
    std::string s;
    s.reserve(id.size());
    for (char c : id) {
        s.push_back((char) std::tolower((unsigned char) c));
    }
    const size_t slash = s.find_last_of("/\\");
    if (slash != std::string::npos) {
        s.erase(0, slash + 1);
    }
    const std::string ext = ".gguf";
    if (s.size() >= ext.size() && s.compare(s.size() - ext.size(), ext.size(), ext) == 0) {
        s.erase(s.size() - ext.size());
    }
    for (auto & c : s) {
        if (c == '_') {
            c = '-';
        }
    }
    return s;
}

// the family of one model id, or nullptr
inline const common_family_sampling * common_family_sampling_match(const std::string & id) {
    const std::string s = common_family_sampling_normalize(id);
    const common_family_sampling * best = nullptr;
    size_t best_len = 0;
    for (const auto & f : common_family_sampling_table()) {
        const std::string p = f.pattern;
        size_t pos = s.find(p);
        while (pos != std::string::npos) {
            const bool left_ok  = pos == 0 || !std::isalnum((unsigned char) s[pos - 1]);
            const size_t end    = pos + p.size();
            // a pattern that ends in a digit must not run into another digit ("qwen3" is not "qwen35"); one that ends in a letter may
            // be followed by a version ("smollm" is "smollm2")
            const bool right_ok = end >= s.size() || !std::isdigit((unsigned char) p.back()) || !std::isdigit((unsigned char) s[end]);
            if (left_ok && right_ok) {
                if (p.size() > best_len) {
                    best     = &f;
                    best_len = p.size();
                }
                break;
            }
            pos = s.find(p, pos + 1);
        }
    }
    return best;
}

// the first candidate id that matches a family wins; `which` (optional) receives the id that matched
inline const common_family_sampling * common_family_sampling_match_any(const std::vector<std::string> & ids, std::string * which = nullptr) {
    for (const auto & id : ids) {
        if (id.empty()) {
            continue;
        }
        if (const auto * f = common_family_sampling_match(id)) {
            if (which) {
                *which = id;
            }
            return f;
        }
    }
    return nullptr;
}
