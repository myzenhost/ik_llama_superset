// Test helper (rows S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01, D2 and D7): a vocab-only GGUF made from
// models/ggml-vocab-qwen2.gguf with extra tokens appended. Every other key (architecture, hparams, merges, pre-tokenizer) is copied
// from the real file, so llama_model_load_from_file(vocab_only) takes it exactly like a real vocab; only the token table is grown.
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// token_type values of the GGUF tokenizer (llama_token_type): 1 normal, 3 control, 4 user defined
static inline void nolim_write_vocab_gguf(const char * base_path, const char * out_path,
                                          const std::vector<std::string> & extra_tokens, int32_t extra_type) {
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(base_path, ip);
    if (g == nullptr) {
        throw std::runtime_error(std::string("cannot read ") + base_path);
    }
    const int64_t k_tok = gguf_find_key(g, "tokenizer.ggml.tokens");
    const int64_t k_typ = gguf_find_key(g, "tokenizer.ggml.token_type");
    if (k_tok < 0 || k_typ < 0) {
        gguf_free(g);
        throw std::runtime_error("base vocab lacks tokenizer.ggml.tokens / token_type");
    }
    const size_t n = gguf_get_arr_n(g, k_tok);
    if (gguf_get_arr_n(g, k_typ) != n) {
        gguf_free(g);
        throw std::runtime_error("base vocab: token_type length differs from tokens");
    }
    std::vector<std::string> toks;
    std::vector<int32_t>     types;
    toks.reserve(n + extra_tokens.size());
    types.reserve(n + extra_tokens.size());
    const int32_t * t0 = (const int32_t *) gguf_get_arr_data(g, k_typ);
    for (size_t i = 0; i < n; ++i) {
        toks.emplace_back(gguf_get_arr_str(g, k_tok, i));
        types.push_back(t0[i]);
    }
    for (const auto & t : extra_tokens) {
        toks.push_back(t);
        types.push_back(extra_type);
    }
    std::vector<const char *> cp;
    cp.reserve(toks.size());
    for (const auto & t : toks) { cp.push_back(t.c_str()); }

    gguf_remove_key(g, "tokenizer.ggml.tokens");
    gguf_remove_key(g, "tokenizer.ggml.token_type");
    gguf_set_arr_str (g, "tokenizer.ggml.tokens",     cp.data(),    cp.size());
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    const bool ok = gguf_write_to_file(g, out_path, false);
    gguf_free(g);
    if (!ok) {
        throw std::runtime_error(std::string("cannot write ") + out_path);
    }
}
