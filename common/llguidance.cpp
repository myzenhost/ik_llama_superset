#include "common.h"  // common_vocab_token_table (was missing: a -DLLAMA_LLGUIDANCE=ON build did not compile)
#include "sampling.h"
#include "log.h"
#include "grammar-failclosed.h"
#include "grammar-mask-overlap.h"

#ifdef LLAMA_USE_LLGUIDANCE

#    include "llguidance.h"
#    include <algorithm>
#    include <cmath>
#    include <string>
#    include <vector>

struct llama_sampler_llg {
    const llama_vocab * vocab;
    std::string         grammar_kind;
    std::string         grammar_data;
    LlgTokenizer *      tokenizer;
    LlgMatcher *        grammar;
    common_grammar_failclosed fc;  // sticky: set by the first matcher error, cleared by reset
    // The mask of the NEXT state is computed on a worker thread while the GPU decodes (row S69-EXLLAMAV2-PICK-GRAMMAR-MASK-OVERLAP-MEASURE-2026-10-20-01).
    // mutable: clone() is const but must join a pending job before it copies the matcher.
    mutable common_grammar_mask_overlap overlap;
};

static LlgMatcher * llama_sampler_llg_new(LlgTokenizer * tokenizer, const char * grammar_kind,
                                          const char * grammar_data) {
    LlgConstraintInit cinit;
    llg_constraint_init_set_defaults(&cinit, tokenizer);
    const char * log_level = getenv("LLGUIDANCE_LOG_LEVEL");
    if (log_level && *log_level) {
        cinit.log_stderr_level = atoi(log_level);
    }
    auto c = llg_new_matcher(&cinit, grammar_kind, grammar_data);
    if (llg_matcher_get_error(c)) {
        LOG_ERR("llg error: %s\n", llg_matcher_get_error(c));
        llg_free_matcher(c);
        return nullptr;
    }

    return c;
}

static const char * llama_sampler_llg_name(const llama_sampler * /*smpl*/) {
    return "llguidance";
}

// the token that ends generation; the only one a failed (fail-closed) sampler leaves selectable
static llama_token llama_sampler_llg_eos(const llama_vocab * vocab) {
    llama_token eos = llama_vocab_eot(vocab);
    return eos == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eos;
}

static void llama_sampler_llg_accept_impl(llama_sampler * smpl, llama_token token) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar && !ctx->fc.failed) {
        // a mask still being computed belongs to the state this token leaves, and the worker uses the matcher: join it first
        ctx->overlap.drop();
        // FAIL CLOSED (row S69-ENGREV-APHRODITE-LLG-FAILCLOSED-2026-10-27-01): this result used to be dropped, so a token the
        // matcher refused left it in an unknown state and later tokens ran unconstrained
        if (llg_matcher_consume_token(ctx->grammar, token) != 0) {
            const char * err = llg_matcher_get_error(ctx->grammar);
            LOG_ERR("llg error: token %d rejected: %s -- failing closed (only EOS stays selectable)\n", token, err ? err : "(no message)");
            ctx->fc.fail(err ? err : "llguidance matcher rejected a token");
        } else if (common_grammar_mask_overlap::enabled()) {
            // compute the mask of the new state while the next decode runs; apply() joins it. A failure comes back as the job's
            // result and apply() fails closed on it, exactly as the synchronous path does.
            LlgMatcher * m = ctx->grammar;
            ctx->overlap.start([m]() { return llg_matcher_compute_mask(m); });
        }
    }
}

static void llama_sampler_llg_apply(llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        if (ctx->fc.apply_if_failed(cur_p, llama_sampler_llg_eos(ctx->vocab))) {
            return;
        }
        // join the mask accept() started; a non-zero result is a failed computation, reported below like the synchronous one
        const bool overlap_failed = ctx->overlap.pending() && ctx->overlap.finish(0) != 0;
        const uint32_t * mask = overlap_failed ? nullptr : llg_matcher_get_mask(ctx->grammar);
        if (mask == nullptr) {
            if (!overlap_failed && llg_matcher_compute_mask(ctx->grammar) == 0) {
                mask = llg_matcher_get_mask(ctx->grammar);
            } else {
                // used to free the matcher and return, leaving every later token unmasked; now fail closed
                const char * err = llg_matcher_get_error(ctx->grammar);
                LOG_ERR("llg error: %s -- failing closed (only EOS stays selectable)\n", err ? err : "(no message)");
                ctx->fc.fail(err ? err : "llguidance mask computation failed");
                ctx->fc.apply_if_failed(cur_p, llama_sampler_llg_eos(ctx->vocab));
                return;
            }
        }

        for (size_t i = 0; i < cur_p->size; ++i) {
            auto token = cur_p->data[i].id;
            if ((mask[token / 32] & (1 << (token % 32))) == 0) {
                cur_p->data[i].logit = -INFINITY;
            }
        }
    }
}

static void llama_sampler_llg_reset(llama_sampler * smpl) {
    auto * ctx = (llama_sampler_llg *) smpl->ctx;
    if (ctx->grammar) {
        ctx->overlap.drop();  // the worker uses the matcher
        // llguidance.h: "A matcher in an error state cannot be reset" (returns -1). The result used to be dropped and the fail-closed flag
        // cleared anyway, which would reopen a matcher that is still broken; now the sampler stays failed.
        if (llg_matcher_reset(ctx->grammar) == 0) {
            ctx->fc.clear();
        } else {
            const char * err = llg_matcher_get_error(ctx->grammar);
            LOG_ERR("llg error: reset refused: %s -- staying failed closed\n", err ? err : "(no message)");
            ctx->fc.fail(err ? err : "llguidance matcher cannot be reset");
        }
    }
}

static llama_sampler * llama_sampler_llg_clone(const llama_sampler * smpl) {
    const auto * ctx = (const llama_sampler_llg *) smpl->ctx;

    // grammar_kind == nullptr: the empty-shell path, which never returns nullptr
    auto * result = llama_sampler_init_llg(ctx->vocab, nullptr, nullptr);

    // copy the state
    {
        auto * result_ctx = (llama_sampler_llg *) result->ctx;

        if (ctx->grammar) {
            ctx->overlap.drop();  // do not copy a matcher a worker is still using; the clone starts with no job and apply() computes its mask
            result_ctx->grammar_kind = ctx->grammar_kind;
            result_ctx->grammar_data = ctx->grammar_data;
            result_ctx->grammar      = llg_clone_matcher(ctx->grammar);
            result_ctx->tokenizer    = llg_clone_tokenizer(ctx->tokenizer);
            result_ctx->fc           = ctx->fc;
        }
    }

    return result;
}

static void llama_sampler_llg_free(llama_sampler * smpl) {
    const auto * ctx = (llama_sampler_llg *) smpl->ctx;

    if (ctx->grammar) {
        ctx->overlap.drop();  // join before the matcher is freed
        llg_free_matcher(ctx->grammar);
        llg_free_tokenizer(ctx->tokenizer);
    }

    delete ctx;
}

static llama_sampler_i llama_sampler_llg_i = {
    /* .name              = */ llama_sampler_llg_name,
    /* .accept            = */ llama_sampler_llg_accept_impl,
    /* .apply             = */ llama_sampler_llg_apply,
    /* .reset             = */ llama_sampler_llg_reset,
    /* .clone             = */ llama_sampler_llg_clone,
    /* .free              = */ llama_sampler_llg_free,
    /* .backend_init      = */ NULL,
    /* .backend_accept    = */ NULL,
    /* .backend_apply     = */ NULL,
    /* .backend_set_input = */ NULL,
    /* .backend_reset     = */ NULL,
    /* .copy_state        = */ NULL,
};

static std::vector<llama_token> llama_sampler_llg_tokenize_vec(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> toks(text.size() + 8);
    int32_t                  n = 0;
    try {
        n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
        if (n < 0) {
            toks.resize((size_t) -n);
            n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
        }
    } catch (const std::exception & e) {
        GGML_ABORT("llama_tokenize failed: %s\n", e.what());
    }
    GGML_ASSERT(n >= 0);
    toks.resize((size_t) n);
    return toks;
}

// llguidance's contract (llguidance.h, LlgTokenizerInit::tokenize_fn): tokenize ONLY these bytes. llama_tokenize
// does more on a SentencePiece vocab with add_space_prefix: it prepends "▁", so tokenize("o") returns the
// token for " o". llguidance uses this function for FORCED bytes, so its mask then allowed " o" where the grammar
// forced "o", and the next consume failed ("token \" o\" doesn't satisfy the grammar; forced bytes: got 'o'") --
// after which the matcher was dropped and generation ran unconstrained. MEASURED 2026-09-28 on stories260K (SPM).
// Upstream has the same code; its llguidance test only uses a BPE vocab, which adds no prefix.
//
// Fix: tokenize "\n" + bytes and strip the tokenization of "\n" alone. The prefix absorbs any space the
// tokenizer prepends; a newline does not merge with following text in SPM (byte-fallback token) or byte-level
// BPE (pre-tokenizer splits whitespace). If the result does NOT start with exactly the prefix tokens, that
// assumption failed for this vocab/input, and the plain tokenization is used as before -- never worse.
static size_t llama_sampler_llg_tokenize_fn(const void * user_data, const uint8_t * bytes, size_t bytes_len,
                                            uint32_t * output_tokens, size_t output_tokens_len) {
    const llama_vocab * vocab = (const llama_vocab *) user_data;
    const std::string   text((const char *) bytes, bytes_len);

    const std::vector<llama_token> pre = llama_sampler_llg_tokenize_vec(vocab, "\n");
    const std::vector<llama_token> all = llama_sampler_llg_tokenize_vec(vocab, "\n" + text);

    std::vector<llama_token> toks;
    if (!pre.empty() && all.size() >= pre.size() && std::equal(pre.begin(), pre.end(), all.begin())) {
        toks.assign(all.begin() + pre.size(), all.end());
    } else {
        toks = llama_sampler_llg_tokenize_vec(vocab, text);
    }

    const size_t n = std::min(toks.size(), output_tokens_len);
    for (size_t i = 0; i < n; ++i) {
        output_tokens[i] = (uint32_t) toks[i];
    }
    return toks.size();  // the full count; the caller retries with a larger buffer when it exceeds the length
}

static LlgTokenizer * llama_sampler_llg_new_tokenizer(const llama_vocab * vocab) {
    // TODO store the tokenizer in the vocab somehow
    static const llama_vocab * vocab_cache;
    static LlgTokenizer *      tokenizer_cache;

    if (vocab_cache == vocab) {
        return llg_clone_tokenizer(tokenizer_cache);
    }

    auto tok_eos = llama_vocab_eot(vocab);
    if (tok_eos == LLAMA_TOKEN_NULL) {
        tok_eos = llama_vocab_eos(vocab);
    }

    size_t vocab_size = llama_vocab_n_tokens(vocab);

    // The table must hold the bytes each token contributes MID-SEQUENCE (see common_vocab_token_table).
    std::vector<uint32_t> token_lens;
    std::vector<uint8_t>  token_bytes;
    common_vocab_token_table(vocab, token_lens, token_bytes);

    LlgTokenizerInit tinit = {
        /* .vocab_size                         = */ (uint32_t) vocab_size,
        /* .tok_eos                            = */ (uint32_t) tok_eos,
        /* .token_lens                         = */ token_lens.data(),
        /* .token_bytes                        = */ token_bytes.data(),
        /* .tokenizer_json                     = */ nullptr,
        /* .tokenize_assumes_string            = */ true,
        /* .tokenize_fn                        = */ llama_sampler_llg_tokenize_fn,
        /* .use_approximate_greedy_tokenize_fn = */ false,
        /* .tokenize_user_data                 = */ vocab,
        /* .slices                             = */ nullptr,
    };

    char           error_buffer[1024];
    LlgTokenizer * tokenizer = llg_new_tokenizer(&tinit, error_buffer, sizeof(error_buffer));

    if (tokenizer == nullptr) {
        LOG_ERR("llg tokenizer error: %s\n", error_buffer);
        return tokenizer;
    }

    if (tokenizer_cache) {
        llg_free_tokenizer(tokenizer_cache);
    }
    vocab_cache     = vocab;
    tokenizer_cache = tokenizer;

    return llg_clone_tokenizer(tokenizer_cache);
}

bool llama_sampler_llg_failed(const llama_sampler * smpl, std::string * err) {
    if (smpl == nullptr || smpl->iface != &llama_sampler_llg_i) {
        return false;
    }
    const auto * ctx = (const llama_sampler_llg *) smpl->ctx;
    if (ctx == nullptr || !ctx->fc.failed) {
        return false;
    }
    if (err) {
        *err = ctx->fc.error;
    }
    return true;
}

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab, const char * grammar_kind,
                                       const char * grammar_data) {
    auto * ctx = new llama_sampler_llg;

    if (grammar_kind != nullptr && grammar_kind[0] != '\0') {
        auto tokenizer = llama_sampler_llg_new_tokenizer(vocab);
        *ctx           = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ grammar_kind,
            /* .grammar_data = */ grammar_data,
            /* .tokenizer    = */ tokenizer,
            /* .grammar      = */ llama_sampler_llg_new(tokenizer, grammar_kind, grammar_data),
        };
        if (ctx->grammar) {
            GGML_ASSERT(((size_t) llama_vocab_n_tokens(vocab) + 31) / 32 * 4 ==
                        llg_matcher_get_mask_byte_size(ctx->grammar));
        } else {
            // A grammar was requested and did not compile (llama_sampler_llg_new logged why). Returning a
            // sampler with no matcher, as before, made apply() a no-op: the request ran UNCONSTRAINED and
            // nothing said so. nullptr lets common_sampler_init refuse the request instead.
            if (ctx->tokenizer) {
                llg_free_tokenizer(ctx->tokenizer);
            }
            delete ctx;
            return nullptr;
        }
    } else {
        *ctx = {
            /* .vocab        = */ vocab,
            /* .grammar_kind = */ {},
            /* .grammar_data = */ {},
            /* .tokenizer    = */ nullptr,
            /* .grammar      = */ nullptr,
        };
    }

    return llama_sampler_init(
        /* .iface = */ &llama_sampler_llg_i,
        /* .ctx   = */ ctx);
}

#else

llama_sampler * llama_sampler_init_llg(const llama_vocab *, const char *, const char *) {
    LOG_WRN("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
    return nullptr;
}

bool llama_sampler_llg_failed(const llama_sampler *, std::string *) {
    return false;
}

#endif  // LLAMA_USE_LLGUIDANCE
