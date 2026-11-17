// Mistral FIM markers [PREFIX] / [SUFFIX] / [MIDDLE] must be picked up as the vocab's FIM tokens
// (row S69-ENGINE-MERGE-MISTRAL-COMMON-2026-11-06-01, fork-triage-port-small-01).
//
// Spec read (no code copied): mistral-common src/mistral_common/tokens/tokenizers/base.py:92-94 names the three special tokens
// "[PREFIX]" "[MIDDLE]" "[SUFFIX]"; instruct.py:694-715 InstructTokenizerV2.encode_fim emits BOS, [SUFFIX], suffix, [PREFIX], prefix
// (suffix-first order, i.e. the server's spm_infill path). Before the port the vocab loader knew none of these spellings, so a Mistral/
// Codestral/Devstral tekken vocab loaded with no FIM tokens and /infill refused the model.
//
// Needs models/ggml-vocab-qwen2.gguf (WORKING_DIRECTORY is the source root). The Qwen file already carries its own FIM tokens, which would
// win the first-match scan, so the test renames those to inert names and appends the Mistral spellings. argv[1] = A | B.
//   A: control, same writer with the existing <|fim_prefix|>/<|fim_suffix|>/<|fim_middle|> spellings (the stock Qwen file has none).
//   B: [PREFIX] [SUFFIX] [MIDDLE] appended as control tokens -> pre/suf/mid are exactly those three.
#include "gguf.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static const char * BASE = "models/ggml-vocab-qwen2.gguf";

static llama_model * load_vocab(const char * path) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * m = llama_model_load_from_file(path, mp);
    if (m == nullptr) { throw std::runtime_error(std::string("cannot load ") + path); }
    return m;
}

static std::string tok_text(const llama_vocab * v, llama_token t) {
    return t == LLAMA_TOKEN_NULL ? std::string("<null>") : std::string(llama_vocab_get_text(v, t));
}

// base vocab with every Qwen FIM spelling renamed, then the Mistral markers appended as control tokens (type 3)
static void write_fim_vocab(const char * out_path, const char * pre, const char * suf, const char * mid) {
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(BASE, ip);
    if (g == nullptr) { throw std::runtime_error(std::string("cannot read ") + BASE); }
    const int64_t k_tok = gguf_find_key(g, "tokenizer.ggml.tokens");
    const int64_t k_typ = gguf_find_key(g, "tokenizer.ggml.token_type");
    if (k_tok < 0 || k_typ < 0) { gguf_free(g); throw std::runtime_error("base vocab lacks tokens/token_type"); }
    const size_t n = gguf_get_arr_n(g, k_tok);
    const int32_t * t0 = (const int32_t *) gguf_get_arr_data(g, k_typ);
    std::vector<std::string> toks;
    std::vector<int32_t>     types;
    for (size_t i = 0; i < n; ++i) {
        std::string s = gguf_get_arr_str(g, k_tok, i);
        if (s == "<|fim_prefix|>" || s == "<|fim_suffix|>" || s == "<|fim_middle|>" || s == "<|fim_pad|>" ||
            s == "<|repo_name|>"  || s == "<|file_sep|>") {
            s = "<|inert_" + std::to_string(i) + "|>";
        }
        toks.push_back(s);
        types.push_back(t0[i]);
    }
    for (const char * s : { pre, suf, mid }) { toks.push_back(s); types.push_back(3); }
    std::vector<const char *> cp;
    for (const auto & t : toks) { cp.push_back(t.c_str()); }
    gguf_remove_key(g, "tokenizer.ggml.tokens");
    gguf_remove_key(g, "tokenizer.ggml.token_type");
    gguf_set_arr_str (g, "tokenizer.ggml.tokens",     cp.data(),    cp.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    const bool ok = gguf_write_to_file(g, out_path, false);
    gguf_free(g);
    if (!ok) { throw std::runtime_error(std::string("cannot write ") + out_path); }
}

static void case_a() {
    const std::string out = "mistral_fim_vocab_a.gguf";
    write_fim_vocab(out.c_str(), "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>");
    llama_model * m = load_vocab(out.c_str());
    const llama_vocab * v = llama_model_get_vocab(m);
    CHECK(tok_text(v, llama_vocab_fim_pre(v)) == "<|fim_prefix|>", "A: pre = %s", tok_text(v, llama_vocab_fim_pre(v)).c_str());
    CHECK(tok_text(v, llama_vocab_fim_suf(v)) == "<|fim_suffix|>", "A: suf = %s", tok_text(v, llama_vocab_fim_suf(v)).c_str());
    CHECK(tok_text(v, llama_vocab_fim_mid(v)) == "<|fim_middle|>", "A: mid = %s", tok_text(v, llama_vocab_fim_mid(v)).c_str());
    std::printf("A: pre=%s suf=%s mid=%s\n", tok_text(v, llama_vocab_fim_pre(v)).c_str(),
                tok_text(v, llama_vocab_fim_suf(v)).c_str(), tok_text(v, llama_vocab_fim_mid(v)).c_str());
    llama_model_free(m);
    std::remove(out.c_str());
}

static void case_b() {
    const std::string out = "mistral_fim_vocab_b.gguf";
    write_fim_vocab(out.c_str(), "[PREFIX]", "[SUFFIX]", "[MIDDLE]");
    llama_model * m = load_vocab(out.c_str());
    const llama_vocab * v = llama_model_get_vocab(m);
    CHECK(tok_text(v, llama_vocab_fim_pre(v)) == "[PREFIX]", "B: pre = %s (want [PREFIX])", tok_text(v, llama_vocab_fim_pre(v)).c_str());
    CHECK(tok_text(v, llama_vocab_fim_suf(v)) == "[SUFFIX]", "B: suf = %s (want [SUFFIX])", tok_text(v, llama_vocab_fim_suf(v)).c_str());
    CHECK(tok_text(v, llama_vocab_fim_mid(v)) == "[MIDDLE]", "B: mid = %s (want [MIDDLE])", tok_text(v, llama_vocab_fim_mid(v)).c_str());
    std::printf("B: pre=%s suf=%s mid=%s\n", tok_text(v, llama_vocab_fim_pre(v)).c_str(),
                tok_text(v, llama_vocab_fim_suf(v)).c_str(), tok_text(v, llama_vocab_fim_mid(v)).c_str());
    llama_model_free(m);
    std::remove(out.c_str());
}

int main(int argc, char ** argv) {
    const std::string which = argc > 1 ? argv[1] : "";
    try {
        if (which.empty() || which == "A") { case_a(); }
        if (which.empty() || which == "B") { case_b(); }
    } catch (const std::exception & e) {
        std::printf("FAIL: exception: %s\n", e.what());
        return 1;
    }
    if (g_fails > 0) { std::printf("%d check(s) failed\n", g_fails); return 1; }
    std::printf("PASS\n");
    return 0;
}
