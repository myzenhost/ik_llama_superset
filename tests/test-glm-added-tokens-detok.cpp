// GLM added tokens (<think>, </think>, <tool_call>, </tool_call>, <arg_key>, </arg_key>, <arg_value>, </arg_value>) survive
// tokenize -> detokenize. Row S69-ENGREV-TRNRT-CHECK-GLM-ADDED-TOKENS-DETOKENIZE-2026-11-13-01 (tests/test_glm53_chat.py:219 in the
// reviewed project: added tokens carry special=false, so a detokenizer that skips special tokens drops them and the tool call is lost).
//
// Vocab-only, no weights: models/ggml-vocab-glm4.gguf (the fork's GLM4 tokenizer) with the eight tags appended through
// nolim_write_vocab_gguf. A GLM GGUF stores added tokens with special=false as USER_DEFINED (type 4); the same tags are also checked as
// CONTROL (type 3) so the contract of both kinds is pinned: USER_DEFINED must round-trip with special=false, CONTROL with special=true.
// Run from the source root (CMake sets WORKING_DIRECTORY). argv[1] = A (user-defined) | B (control) runs one case.
#include "common.h"
#include "nolim-synth-vocab.h"
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static const char * BASE = "models/ggml-vocab-glm4.gguf";

static const std::vector<std::string> TAGS = {
    "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
};

static llama_model * load_vocab_model(const char * path) {
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * m = llama_model_load_from_file(path, mparams);
    if (m == nullptr) {
        throw std::runtime_error(std::string("cannot load ") + path);
    }
    return m;
}

// kind 4 = USER_DEFINED (special=false added token), 3 = CONTROL. detok_special = the flag detokenize needs for that kind.
static void run_case(const char * name, int32_t kind, bool parse_special, bool detok_special) {
    const std::string out = std::string("glm_added_tokens_") + name + ".gguf";
    nolim_write_vocab_gguf(BASE, out.c_str(), TAGS, kind);
    llama_model * m = load_vocab_model(out.c_str());
    const llama_vocab * vocab = llama_model_get_vocab(m);
    const llama_token first_new = (llama_token) llama_vocab_n_tokens(vocab) - (llama_token) TAGS.size();

    // every tag alone is exactly one token, the appended id, and detokenizes back to itself
    for (size_t i = 0; i < TAGS.size(); ++i) {
        const std::vector<llama_token> ids = common_tokenize(vocab, TAGS[i], false, parse_special);
        CHECK(ids.size() == 1, "%s: '%s' tokenized to %zu tokens, want 1", name, TAGS[i].c_str(), ids.size());
        if (ids.size() == 1) {
            CHECK(ids[0] == first_new + (llama_token) i, "%s: '%s' id %d, want %d", name, TAGS[i].c_str(), ids[0], first_new + (llama_token) i);
        }
        const std::string back = common_detokenize(vocab, ids, detok_special);
        CHECK(back == TAGS[i], "%s: '%s' detokenized to '%s'", name, TAGS[i].c_str(), back.c_str());
    }

    // a GLM tool-call turn: tags interleaved with ordinary text must round-trip as one string
    const std::string turn =
        "<think>need the weather</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
    const std::vector<llama_token> ids = common_tokenize(vocab, turn, false, parse_special);
    size_t n_tags = 0;
    for (const llama_token t : ids) { n_tags += (t >= first_new) ? 1 : 0; }
    CHECK(n_tags == 8, "%s: turn holds %zu tag tokens, want 8", name, n_tags);
    const std::string back = common_detokenize(vocab, ids, detok_special);
    CHECK(back == turn, "%s: turn detokenized to '%s'", name, back.c_str());
    std::printf("%s: kind=%d tags=%zu turn_tokens=%zu\n", name, (int) kind, TAGS.size(), ids.size());

    llama_model_free(m);
    std::remove(out.c_str());
}

int main(int argc, char ** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    try {
        // A: special=false added tokens (USER_DEFINED). Matched without parse_special, rendered without the special flag.
        if (only.empty() || only == "A") { run_case("user_defined", 4, /*parse_special=*/false, /*detok_special=*/false); }
        // B: CONTROL tokens. Need parse_special on the way in and the special flag on the way out.
        if (only.empty() || only == "B") { run_case("control", 3, /*parse_special=*/true, /*detok_special=*/true); }
    } catch (const std::exception & e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    if (g_fails) {
        std::printf("test-glm-added-tokens-detok: %d FAILED\n", g_fails);
        return 1;
    }
    std::printf("test-glm-added-tokens-detok: ok\n");
    return 0;
}
