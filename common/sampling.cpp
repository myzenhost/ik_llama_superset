#define LLAMA_API_INTERNAL
#include "sampling.h"
#include "sampling-penalty-fade.h"
#include "llama-vocab.h"
#include "common.h"
#include "speculative.h"
#include "reasoning-budget.h"  // was #include "reasoning-budget.cpp" -- that .cpp is also its own TU in common/CMakeLists.txt, so every function was defined twice (MSVC LNK4006 x8)
#include "fit.h"  // decision #5 port: common_memory_breakdown_print, used by ported common_perf_print below
#include "watermark.h"
#include "sampling-skew.h"
#include "topk-scan.h"
#include "sampling-family-defaults.h"

#include <limits>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <random>
#include <unordered_map>

// Keep dflash randomness independent from target
static constexpr uint32_t COMMON_SPECULATIVE_VERIFIER_SEED_XOR = 0x9e3779b9U;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

struct llama_sampler_adaptive_p * llama_clone_adaptive_p(const struct llama_sampler_adaptive_p * adapt_p_ctx);
void llama_free_adaptive_p(struct llama_sampler_adaptive_p * adapt_p_ctx);

struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params_in) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // upstream's `samplers` spelling folds into samplers_sequence once, here, so every reader below and gsmpl->params see one chain
    common_params_sampling params_resolved;
    const bool resolve = !common_sampler_samplers_unset(params_in);
    if (resolve) {
        params_resolved = params_in;
        if (!common_sampler_resolve_samplers(params_resolved)) {
            return nullptr;
        }
    }
    const common_params_sampling & params = resolve ? params_resolved : params_in;

    struct common_sampler * result = new common_sampler();

    result->params  = params;
    result->grammar = nullptr;
    result->rbudget = nullptr;

    struct llama_grammar* grmr = nullptr;
    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        // Same call as upstream common/sampling.cpp:215: the whole string, "%llguidance" header
        // included, is a Lark grammar to llguidance. It lands in the sampler-shaped slot
        // (common_sampler::cnstr), never in `grammar`, which is dereferenced as a llama_grammar.
        result->cnstr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
        if (result->cnstr == nullptr) {
            // llama_sampler_init_llg logged the llguidance error; a request that asked for a
            // constraint must not silently run unconstrained.
            delete result;
            throw std::runtime_error("llguidance: failed to initialise the %llguidance grammar (see log)");
        }
        result->prev.resize(params.n_prev);
        result->n_valid = 0;
        result->grammar_str = grammar_str;
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    }
    else if (grammar_str.compare(0, 9, "%xgrammar") == 0) {
        // XGrammar: "%xgrammar <ebnf|json_schema|json|regex|lark|structural_tag>\n<body>" (common/xgrammar.cpp).
        // Refused, not aborted, when XGrammar is not built in: this is request input, not a build invariant.
        result->cnstr = llama_sampler_init_xgr(vocab, grammar_str.c_str());
        if (result->cnstr == nullptr) {
            delete result;
            throw std::runtime_error("xgrammar: grammar rejected, or this build has no XGrammar (cmake -DLLAMA_XGRAMMAR=ON) -- see log");
        }
        result->prev.resize(params.n_prev);
        result->n_valid = 0;
        result->grammar_str = grammar_str;
    }
    else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
            case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            {
                const auto & word = trigger.value;
                trigger_patterns.push_back(regex_escape(word));
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            {
                trigger_patterns.push_back(trigger.value);
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
            {
                const auto & pattern = trigger.value;
                std::string anchored = "^$";
                if (!pattern.empty()) {
                    anchored = (pattern.front() != '^' ? "^" : "")
                        + pattern
                        + (pattern.back() != '$' ? "$" : "");
                }
                trigger_patterns.push_back(anchored);
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            {
                const auto token = trigger.token;
                trigger_tokens.push_back(token);
                break;
            }
            default:
                GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
            // ADAPTED 2026-09-13. llama_sampler_init_grammar* return a `llama_sampler *`;
            // `grmr` and common_sampler::grammar (sampling.h:272) are `llama_grammar *`, and
            // this fork DEREFERENCES that pointer as a grammar -- ctx->grammar->lazy,
            // ->trigger_patterns, ->trigger_tokens, ->vocab at sampling.cpp:209-216 -- so a
            // sampler handle here would not merely fail to convert, it would be the wrong
            // object for every later line that touches it.
            //
            // Not invented: llama_grammar_reset() twelve lines below builds a grammar with this
            // exact call, from this exact trigger_patterns_c/trigger_tokens pair, and assigns it
            // to this exact field. The lazy flag that upstream encodes by picking one of two
            // functions is just this function's `lazy` argument (src/llama-grammar.h:163-171),
            // and the non-lazy path passes the same empty trigger vectors upstream's
            // llama_sampler_init_grammar would have.
            grmr = llama_grammar_init_impl(vocab, grammar_str.c_str(), "root",
                params.grammar_lazy,
                trigger_patterns_c.data(), trigger_patterns_c.size(),
                trigger_tokens.data(), trigger_tokens.size());
            if (grmr) {
                result->grammar = grmr;
            } else {
                // A grammar was supplied and did not parse (llama_grammar_init_impl logged why). Skipping it ran
                // the request UNCONSTRAINED with a 200 -- tests/unit/test_chat_completion.py
                // test_completion_with_invalid_grammar expects 400, which is what upstream gives (its
                // common_sampler_init returns nullptr there). Throwing reaches the same 400 via the server's
                // catch around common_sampler_init (server-context.cpp "Failed to initialize samplers").
                delete result;
                throw std::runtime_error("failed to parse grammar");
            }
        }
        result->n_valid = 0;
        result->grammar_str = grammar_str;
        result->grammar_root = "root";
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            // unsigned char casts: std::isspace on a negative char (any UTF-8 lead/continuation byte where char is
            // signed, as on MSVC and MinGW) is undefined behaviour -- MSVC debug CRT asserts, release indexes its
            // ctype table out of bounds. A generation prompt or first token starting with non-ASCII text hits it.
            if (i == 0 && !piece.empty() && std::isspace((unsigned char) piece[0])
                    && !std::isspace((unsigned char) params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_grammar_accept_impl(*grmr, vocab, nullptr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        }
        catch (std::exception & e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw;  // rethrow the original object; `throw e;` would slice it to std::exception
        }
    }

    // Same prefill replay for a sampler-shaped constraint (upstream feeds the same tokens to its grmr
    // sampler with llama_sampler_accept).
    if (result->cnstr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(result->cnstr, token);
            LOG_DBG("%s: constraint accepted prefill token (%d)\n", __func__, token);
        }
    }

    // grammar_continue_prefix (koboldcpp grammar_retain_state, adopted per-request -- sampling.h,
    // SOURCE-READ, AGPL-3.0, no code copied): explicit, caller-supplied continuation for a USER
    // GBNF grammar, the one case common_grammar_needs_prefill() deliberately excludes above. Only
    // fires for COMMON_GRAMMAR_TYPE_USER so it can never double-replay alongside the automatic
    // output-format/tool-calls path a few lines up.
    if (grmr && !params.grammar_lazy && params.grammar.type == COMMON_GRAMMAR_TYPE_USER
            && !params.grammar_continue_prefix.empty()) {
        GGML_ASSERT(vocab != nullptr);
        try {
            auto continue_tokens = common_tokenize(vocab, params.grammar_continue_prefix, false, true);
            for (size_t i = 0; i < continue_tokens.size(); i++) {
                std::string piece = common_token_to_piece(vocab, continue_tokens[i], true);
                if (i == 0 && !piece.empty() && std::isspace((unsigned char) piece[0])
                        && !std::isspace((unsigned char) params.grammar_continue_prefix[0])) {
                    // Same tokenizer quirk the generation_prompt replay above works around: some
                    // tokenizers prepend a space token before the first piece of standalone text.
                    continue;
                }
                llama_grammar_accept_impl(*grmr, vocab, nullptr, continue_tokens[i]);
                LOG_DBG("%s: grammar accepted continuation token (%d)\n", __func__, continue_tokens[i]);
            }
        }
        catch (std::exception & e) {
            LOG_ERR("%s: error replaying grammar_continue_prefix for grammar:\n%s\n\nPrefix:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.grammar_continue_prefix.c_str());
            throw;  // rethrow the original object; `throw e;` would slice it to std::exception
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0)) {
        result->rbudget = common_reasoning_budget_init(
            vocab,
            params.reasoning_budget_start,
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            common_reasoning_budget_accept(result->rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    llama_sampling_set_rng_seed(result, params.seed);
    // mirostat state starts at 2 * tau, as upstream's sampler does (llama-sampler.cpp llama_sampler_mirostat[_v2] init and reset).
    // It was never written: the first mirostat token read an uninitialised float, so two identical seeded requests gave different
    // text (MEASURED 2026-11-08, gate b1108g4 tabby e2e + _g4_miro_probe_1108.py: 'mirostat 2' x2 differed, 'none' x2 identical).
    result->mirostat_mu = 2.0f * params.mirostat_tau;
    if (params.adaptive_target >= 0.0f) {
        result->adapt_p_ctx = llama_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.adaptive_updt_w_cur, result->rng());
    }
    for (const auto& cnstr : params.samplers_sequence)
    {
        switch (cnstr)
        {
            case llama_sampler_type::DRY:
            {
                std::vector<const char*> c_breakers;
                c_breakers.reserve(params.dry_sequence_breakers.size());
                for (const auto& str : params.dry_sequence_breakers)
                {
                    c_breakers.push_back(str.c_str());
                }
                result->smpl=llama_sampler_dry_init(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size());

                break;
            }
            case llama_sampler_type::INFILL:
                if (result->infill == nullptr) {
                    result->infill = llama_sampler_init_infill(vocab);
                }
                break;
            default:
                break;
        }
    }

    // FSD: a stateful final selector; reset/clone/accept below keep it in step with the rest of the sampler state
    if (params.fsd_alpha >= 0.0f) {
        result->fsd = llama_sampler_init_fsd(params.fsd_pool_size, params.fsd_k, params.fsd_alpha, params.fsd_n, params.fsd_beta,
                params.fsd_random_first_n, result->rng());
    }

    result->is_decoding = false;

    result->n_rewind = 0;

    result->elb_idx = 0;
    result->elb_search_pos = 0;

    return result;
}

void common_sampler_free(struct common_sampler * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->grammar) {
        llama_grammar_free(ctx->grammar);
    }
    if (ctx->cnstr) {
        llama_sampler_free(ctx->cnstr);
    }
    if (ctx->smpl)
        llama_sampler_dry_free(ctx->smpl);
    if (ctx->adapt_p_ctx)
        llama_free_adaptive_p(ctx->adapt_p_ctx);
    if (ctx->fsd)
        llama_sampler_free(ctx->fsd);
    if (ctx->infill)
        llama_sampler_free(ctx->infill);
    if (ctx->rbudget)
        common_reasoning_budget_free(ctx->rbudget);
    delete ctx;
}

static void llama_grammar_reset(common_sampler * ctx) {
    if (!ctx->grammar) {
        return;
    }
    std::vector<const char*>  trigger_patterns_c;
    trigger_patterns_c.reserve(ctx->grammar->trigger_patterns.size());
    for (auto& trigger_pattern : ctx->grammar->trigger_patterns) {
        trigger_patterns_c.push_back(trigger_pattern.pattern.c_str());
    }

    auto* grammar_new = llama_grammar_init_impl(ctx->grammar->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str(),
        ctx->grammar->lazy, trigger_patterns_c.data(), trigger_patterns_c.size(),
        ctx->grammar->trigger_tokens.data(), ctx->grammar->trigger_tokens.size());

    llama_grammar_free_impl(ctx->grammar);
    ctx->grammar = grammar_new;
}

void common_sampler_reset(common_sampler * ctx) {
    // embedding / rerank slots have no sampler, and server_slot::init_sampler() resets unconditionally,
    // as upstream does; upstream's reset returns on nullptr, this one dereferenced it (0xC0000005 on /rerank)
    if (!ctx) {
        return;
    }
    // llama_grammar_reset(ctx);
    ctx->is_decoding = false;
    // the text the server feeds for expiring logit bias belongs to the run that is ending
    ctx->generated_text = std::string_view();
    ctx->playing_text.clear();
    ctx->n_rewind = 0;
    ctx->rewinded_text.clear();
    ctx->prev.clear();
    ctx->wm_generated.clear();
    ctx->wm_n_marked  = 0;
    ctx->wm_n_skipped = 0;
    // a reused sampler starts the next run with fresh mirostat state (upstream llama_sampler_mirostat_v2_reset: mu = 2 * tau)
    ctx->mirostat_mu = 2.0f * ctx->params.mirostat_tau;
    llama_sampler_dry_reset(ctx->smpl);

    llama_free_adaptive_p(ctx->adapt_p_ctx);
    ctx->adapt_p_ctx = ctx->params.adaptive_target >= 0.0f
        ? llama_init_adaptive_p(ctx->params.adaptive_target, ctx->params.adaptive_decay, ctx->params.adaptive_updt_w_cur, ctx->rng())
        : nullptr;
    if (ctx->fsd) {
        llama_sampler_reset(ctx->fsd);     // drops the n-gram history of the run that is ending
    }
    ctx->speculative_rng.seed(ctx->speculative_seed);
}

void common_sampler_review(common_sampler * ctx, const size_t n_unsent, const bool rewind_status) {
    // add stateful samplers here
    if (ctx->adapt_p_ctx != nullptr) {
        llama_review_adaptive_p(ctx->adapt_p_ctx, n_unsent, rewind_status);
    }

    if (ctx->n_rewind > 0) {
        common_expiring_logit_bias_rewind(ctx);
    }
}

void llama_sampling_set_rng_seed(struct common_sampler * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = std::random_device{}();
    }
    ctx->rng.seed(seed);
    ctx->speculative_seed = seed ^ COMMON_SPECULATIVE_VERIFIER_SEED_XOR;
    ctx->speculative_rng.seed(ctx->speculative_seed);
}

void common_sampler_clone(common_sampler * src, common_sampler * dst) {
    dst->params = src->params;
    dst->mirostat_mu = src->mirostat_mu;
    dst->n_valid = src->n_valid;
    dst->rng = src->rng;
    dst->speculative_seed = src->speculative_seed;
    dst->speculative_rng = src->speculative_rng;
    dst->server_biases = src->server_biases;

    if (dst->grammar) {
        llama_grammar_free(dst->grammar);
        dst->grammar = nullptr;
    }

    if (src->grammar) {
        dst->grammar_root = src->grammar_root;
        dst->grammar_str = src->grammar_str;
        dst->grammar = llama_grammar_copy(src->grammar);
    }

    if (dst->cnstr) {
        llama_sampler_free(dst->cnstr);
        dst->cnstr = nullptr;
    }
    if (src->cnstr) {
        dst->grammar_str = src->grammar_str;
        dst->cnstr = llama_sampler_clone(src->cnstr);  // llguidance: llg_clone_matcher keeps the parse state
    }

    dst->prev = src->prev;
    dst->wm_generated = src->wm_generated;
    dst->wm_n_marked  = src->wm_n_marked;
    dst->wm_n_skipped = src->wm_n_skipped;
    if (dst->smpl) {
        llama_sampler_dry_free(dst->smpl);
        dst->smpl = nullptr;
    }
    if (src->smpl) {
        dst->smpl = llama_sampler_dry_clone(src->smpl);
    }

    if (dst->adapt_p_ctx) {
        llama_free_adaptive_p(dst->adapt_p_ctx);
        dst->adapt_p_ctx = nullptr;
    }
    if (src->adapt_p_ctx) {
        dst->adapt_p_ctx = llama_clone_adaptive_p(src->adapt_p_ctx);
    }

    if (dst->fsd) {
        llama_sampler_free(dst->fsd);
        dst->fsd = nullptr;
    }
    if (src->fsd) {
        dst->fsd = llama_sampler_clone(src->fsd);
    }

    if (dst->infill) {
        llama_sampler_free(dst->infill);
        dst->infill = nullptr;
    }
    if (src->infill) {
        dst->infill = llama_sampler_clone(src->infill);
    }

    if (dst->rbudget) {
        common_reasoning_budget_free(dst->rbudget);
        dst->rbudget = nullptr;
    }
    if (src->rbudget) {
        dst->rbudget = common_reasoning_budget_clone(src->rbudget);
    }

    // expiring logit bias is plain value state; without it a copy restored after a rewind
    // (server banned strings, speculative rollback) resumed the bias schedule where it had
    // advanced to, not where the restored position had it
    dst->elb_states     = src->elb_states;
    dst->elb_idx        = src->elb_idx;
    dst->elb_search_pos = src->elb_search_pos;
}

llama_token llama_sampling_last(common_sampler * ctx) {
    return ctx->prev.back();
}

std::string llama_sampling_prev_str(common_sampler * ctx_sampling, llama_context * ctx_main, int n) {
    const int size = ctx_sampling->prev.size();

    n = std::min(n, size);

    std::string result;

    for (int i = size - n; i < size; i++) {
        result += common_token_to_piece(ctx_main, ctx_sampling->prev[i]);
    }

    return result;
}

std::string llama_sampling_print(const common_params_sampling & params) {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f\n"
            "\txtc_probability = %.3f, xtc_threshold = %.3f, top_n_sigma = %.3f\n"
            "\tadaptive_target = %.2f, adaptive_decay = %.2f",
            params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present,
            params.top_k, params.tfs_z, params.top_p, params.min_p, params.typical_p, params.temp,
            params.mirostat, params.mirostat_eta, params.mirostat_tau,
            params.xtc_probability, params.xtc_threshold, params.top_n_sigma,
            params.adaptive_target, params.adaptive_decay);

    return std::string(result);
}

// Declared at sampling.h:264 as a member of common_params_sampling and never defined, which left
// tools/server/server-context.cpp:1712 with the only undefined symbol in the whole server binary.
// Delegates to the formatter that already exists rather than adding a second one: two functions
// printing the same params is how two output formats drift apart without anyone noticing.
// Same pattern as common_sampler_print, further down this file.
std::string common_params_sampling::print() const {
    return llama_sampling_print(*this);
}

std::string llama_sampling_order_print(const common_params_sampling & params) {
    std::string result = "CFG -> Penalties ";
    if (params.mirostat == 0) {
        for (auto sampler_type : common_sampler_resolved_sequence(params)) {
            const auto sampler_type_name = llama_sampling_type_to_str(sampler_type);
            if (!sampler_type_name.empty()) {
                result += "-> " + sampler_type_name + " ";
            }
        }
    } else {
        result += "-> mirostat ";
    }

    return result;
}

std::string llama_sampling_type_to_str(llama_sampler_type sampler_type) {
    switch (sampler_type) {
        case llama_sampler_type::DRY:         return "dry";
        case llama_sampler_type::TOP_K:       return "top_k";
        case llama_sampler_type::TFS_Z:       return "tfs_z";
        case llama_sampler_type::TYPICAL_P:   return "typical_p";
        case llama_sampler_type::TOP_P:       return "top_p";
        case llama_sampler_type::MIN_P:       return "min_p";
        case llama_sampler_type::TEMPERATURE: return "temperature";
        case llama_sampler_type::XTC        : return "xtc";
        case llama_sampler_type::TOP_N_SIGMA: return "top_n_sigma";
        case llama_sampler_type::ADAPTIVE_P : return "adaptive_p";
        case llama_sampler_type::TOP_A      : return "top_a";
        case llama_sampler_type::EPSILON_CUTOFF: return "epsilon_cutoff";
        case llama_sampler_type::ETA_CUTOFF : return "eta_cutoff";
        case llama_sampler_type::INFILL     : return "infill";
        default : return "";
    }
}

std::vector<llama_sampler_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, llama_sampler_type> sampler_canonical_name_map {
        {"dry",         llama_sampler_type::DRY},
        {"top_k",       llama_sampler_type::TOP_K},
        {"top_p",       llama_sampler_type::TOP_P},
        {"typical_p",   llama_sampler_type::TYPICAL_P},
        {"min_p",       llama_sampler_type::MIN_P},
        {"tfs_z",       llama_sampler_type::TFS_Z},
        {"xtc",         llama_sampler_type::XTC},
        {"top_n_sigma", llama_sampler_type::TOP_N_SIGMA},
        {"temperature", llama_sampler_type::TEMPERATURE},
        {"adaptive_p",  llama_sampler_type::ADAPTIVE_P},
        {"top_a",       llama_sampler_type::TOP_A},
        {"epsilon_cutoff", llama_sampler_type::EPSILON_CUTOFF},
        {"eta_cutoff",  llama_sampler_type::ETA_CUTOFF},
        {"infill",      llama_sampler_type::INFILL},
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llama_sampler_type> sampler_alt_name_map {
        {"dry",         llama_sampler_type::DRY},
        {"top-k",       llama_sampler_type::TOP_K},
        {"top-p",       llama_sampler_type::TOP_P},
        {"nucleus",     llama_sampler_type::TOP_P},
        {"typical-p",   llama_sampler_type::TYPICAL_P},
        {"typical",     llama_sampler_type::TYPICAL_P},
        {"min-p",       llama_sampler_type::MIN_P},
        {"tfs-z",       llama_sampler_type::TFS_Z},
        {"tfs",         llama_sampler_type::TFS_Z},
        {"xtc",         llama_sampler_type::XTC},
        {"top-n-sigma", llama_sampler_type::TOP_N_SIGMA},
        {"temp",        llama_sampler_type::TEMPERATURE},
        {"adaptive-p",  llama_sampler_type::ADAPTIVE_P},
        {"top-a",       llama_sampler_type::TOP_A},
        {"epsilon-cutoff", llama_sampler_type::EPSILON_CUTOFF},
        {"epsilon",     llama_sampler_type::EPSILON_CUTOFF},
        {"eta-cutoff",  llama_sampler_type::ETA_CUTOFF},
        {"eta",         llama_sampler_type::ETA_CUTOFF},
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names.size());
    for (const auto & name : names)
    {
        auto sampler_item = sampler_canonical_name_map.find(name);
        if (sampler_item != sampler_canonical_name_map.end())
        {
            sampler_types.push_back(sampler_item->second);
        } else {
            if (allow_alt_names)
            {
                sampler_item = sampler_alt_name_map.find(name);
                if (sampler_item != sampler_alt_name_map.end())
                {
                    sampler_types.push_back(sampler_item->second);
                }
            }
        }
    }
    return sampler_types;
}

std::vector<llama_sampler_type> llama_sampling_types_from_chars(const std::string & names_string) {
    std::unordered_map<char, llama_sampler_type> sampler_name_map {
        {'d', llama_sampler_type::DRY},
        {'k', llama_sampler_type::TOP_K},
        {'p', llama_sampler_type::TOP_P},
        {'y', llama_sampler_type::TYPICAL_P},
        {'m', llama_sampler_type::MIN_P},
        {'f', llama_sampler_type::TFS_Z},
        {'x', llama_sampler_type::XTC},
        {'n', llama_sampler_type::TOP_N_SIGMA},
        {'t', llama_sampler_type::TEMPERATURE},
        {'w', llama_sampler_type::ADAPTIVE_P},
        {'a', llama_sampler_type::TOP_A},
        {'e', llama_sampler_type::EPSILON_CUTOFF},
        {'h', llama_sampler_type::ETA_CUTOFF},
        {'i', llama_sampler_type::INFILL},
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names_string.size());
    for (const auto & c : names_string) {
        const auto sampler_item = sampler_name_map.find(c);
        if (sampler_item != sampler_name_map.end()) {
            sampler_types.push_back(sampler_item->second);
        }
    }
    return sampler_types;
}

// Top-a (BlinkDL, RWKV): drop every token whose probability is below a * p_max^2. The floor is
// relative to the top token and squared, so it cuts hard when the model is confident and barely
// at all when the distribution is flat. Also offered by koboldcpp (AGPL-3.0): behaviour checked against
// its sample_top_a (gpttype_adapter.cpp:1404, SOURCE-READ 2026-09-26) -- a <= 0 or one candidate is a
// no-op, strict "<" against a * p_max^2, min_keep honoured; no code copied.
static void sample_top_a(struct llama_context * ctx, llama_token_data_array * cur_p, float a, size_t min_keep) {
    if (a <= 0.0f || cur_p->size <= 1) {
        return;
    }
    llama_sample_softmax(ctx, cur_p); // sorts descending and fills p
    const float p_max  = cur_p->data[0].p;
    const float p_min  = a * p_max * p_max;
    size_t keep = cur_p->size;
    for (size_t i = 1; i < cur_p->size; ++i) {
        if (cur_p->data[i].p < p_min) {
            keep = i;
            break;
        }
    }
    cur_p->size = std::max(keep, std::max<size_t>(min_keep, 1));
}

// Epsilon and eta sampling, Hewitt, Manning & Liang 2022, "Truncation Sampling as Language Model
// Desmoothing" (arXiv:2210.15191). Behaviour checked against HF transformers' EpsilonLogitsWarper /
// EtaLogitsWarper (logits_process.py:749-900, Apache-2.0) and Aphrodite's ops epsilon_cutoff.py / eta_cutoff.py
// (SOURCE-READ 2026-09-28; no code copied): probabilities are the softmax over the current candidates, a token
// is dropped when its probability is STRICTLY below the threshold, and the most likely token always survives
// (here max(min_keep, 1) of them, as HF's min_tokens_to_keep). Epsilon's threshold is fixed; eta's is
// min(eta, sqrt(eta) * exp(-H)), H the entropy in nats, so a flat distribution keeps its tail and a peaked one
// loses it. Both cutoffs are raw PROBABILITIES, as in the paper, HF and both Aphrodite ops (the paper
// recommends eta in 3e-4 .. 4e-3).
static void sample_prob_floor(llama_token_data_array * cur_p, float floor_p, size_t min_keep) {
    size_t keep = cur_p->size;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].p < floor_p) {
            keep = i;
            break;
        }
    }
    cur_p->size = std::max(keep, std::max<size_t>(min_keep, 1));
}

static void sample_epsilon_cutoff(struct llama_context * ctx, llama_token_data_array * cur_p, float epsilon, size_t min_keep) {
    if (epsilon <= 0.0f || cur_p->size <= 1) {
        return;
    }
    llama_sample_softmax(ctx, cur_p); // sorts descending and fills p
    sample_prob_floor(cur_p, epsilon, min_keep);
}

static void sample_eta_cutoff(struct llama_context * ctx, llama_token_data_array * cur_p, float eta, size_t min_keep) {
    if (eta <= 0.0f || cur_p->size <= 1) {
        return;
    }
    llama_sample_softmax(ctx, cur_p);
    double entropy = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const double p = cur_p->data[i].p;
        if (p > 0.0) {
            entropy -= p * std::log(p);
        }
    }
    const float floor_p = (float) std::min<double>(eta, std::sqrt((double) eta) * std::exp(-entropy));
    sample_prob_floor(cur_p, floor_p, min_keep);
}

// No-repeat n-gram, the fairseq/HF NoRepeatNGramLogitsProcessor rule (logits_process.py:902-1025) as Aphrodite's
// no_repeat_ngram.py also applies it over prompt + output (SOURCE-READ 2026-09-28; no code copied): with
// n = no_repeat_ngram_size, every token t such that (last n-1 history tokens, t) already occurs as an n-gram in
// the history gets logit -inf. n = 1 bans every token already seen. The history is the sampler's `prev` window,
// prompt tokens included, which common_sampler_accept widens to no_repeat_ngram_last_n (-1 = n_ctx) while
// this is on. Returns the number of distinct banned ids.
static size_t apply_no_repeat_ngram(const std::vector<llama_token> & hist, int32_t n, int32_t last_n,
                                    llama_token_data_array * cur_p) {
    if (n <= 0 || hist.empty()) {
        return 0;
    }
    size_t len = hist.size();
    if (last_n > 0 && (size_t) last_n < len) {
        len = (size_t) last_n;
    }
    if (len + 1 < (size_t) n) {
        return 0; // not enough history for one n-gram ending at the next position
    }
    const llama_token * h = hist.data() + hist.size() - len;
    const size_t ctx_len = (size_t) n - 1;       // tokens before the candidate in each n-gram
    const llama_token * tail = h + len - ctx_len; // the n-1 tokens the next token would follow
    std::unordered_set<llama_token> banned;
    for (size_t s = 0; s + ctx_len < len; ++s) {  // n-gram h[s .. s+n-1], its last token at s+ctx_len
        if (std::equal(tail, tail + ctx_len, h + s)) {
            banned.insert(h[s + ctx_len]);
        }
    }
    if (banned.empty()) {
        return 0;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (banned.count(cur_p->data[i].id)) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
    return banned.size();
}

// Suppress token sequences, CTranslate2's SuppressSequences logits processor (src/decoding_utils.cc:98-150, MIT, behaviour only): a one-token
// sequence is banned at every step; a longer sequence bans its last token when the history ends with all its other tokens. The history is the
// sampler's `prev` window (prompt tokens included, as for no_repeat_ngram). Returns the number of distinct banned ids.
static size_t apply_suppress_sequences(const std::vector<llama_token> & hist, const std::vector<std::vector<llama_token>> & seqs,
                                       llama_token_data_array * cur_p) {
    std::unordered_set<llama_token> banned;
    for (const auto & seq : seqs) {
        if (seq.empty()) {
            continue;
        }
        const size_t ctx_len = seq.size() - 1;
        if (hist.size() >= ctx_len && std::equal(seq.begin(), seq.begin() + ctx_len, hist.end() - ctx_len)) {
            banned.insert(seq.back());
        }
    }
    if (banned.empty()) {
        return 0;
    }
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (banned.count(cur_p->data[i].id)) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
    return banned.size();
}

// Quadratic ("smooth") sampling, kalomaze; formula as in ggml-org/llama.cpp PR #6445 (MIT).
// Each logit is moved by a polynomial in its distance d = logit - max from the top logit:
//     logit' = max - k*f*d^2 + s*f*d^3,   k = (3 - curve)/2,   s = (curve - 1)/2
// Near-tied top tokens barely move while the tail is pushed down, unlike temperature, which
// scales every gap. curve != 1 adds the cubic term, which can reorder tokens, so the array is
// marked unsorted. -inf logits (grammar, logit_bias, banned tokens) stay -inf: with curve < 1
// the cubic term would otherwise turn them into NaN.
static void sample_smoothing(llama_token_data_array * cur_p, float factor, float curve) {
    if (factor <= 0.0f || cur_p->size <= 1) {
        return;
    }
    float h = -INFINITY;
    for (size_t i = 0; i < cur_p->size; ++i) {
        h = std::max(h, cur_p->data[i].logit);
    }
    if (!std::isfinite(h)) {
        return;
    }
    const float k = (3.0f - curve) / 2.0f;
    const float s = (curve - 1.0f) / 2.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
        const float l = cur_p->data[i].logit;
        if (!std::isfinite(l)) {
            continue;
        }
        const float d = l - h;
        cur_p->data[i].logit = h - k * factor * d * d + s * factor * d * d * d;
    }
    cur_p->sorted = false;
}

// A top_k that cuts nothing (k <= 0, or k >= the candidate count) used to sort the whole vocabulary on the spot. That
// sort is now OWED instead: a step that takes unsorted input runs first, and the sort is paid -- the same comparator
// on the same logits, since none of those steps changes a logit -- before the first step that reads the order or
// changes logits, and at the end of the queue (the draw walks the list in order). Every step and the draw therefore
// see the order they saw before; top_p (llama_sample_top_p_impl) sorts only its nucleus, so with top_k <= 0 and top_p < 1
// the full sort is gone. Steps that take unsorted input: top_k itself, top_p, min_p (both of its paths keep the same set),
// and any step its own parameters switch off (each condition below is that sampler's own early return). DRY,
// TEMPERATURE (+smoothing), DIST and any new step: sort first; the adaptive-p step after the loop too.
// SOURCE-READ: mlc-ai/mlc-llm 9fa644f54b04983adea4d0168f49fc6af4a893ba cpp/serve/sampler/cpu_sampler.cc:72-82 (top_p >= 1
// draws from the unsorted distribution: no sort when nothing is cut; Apache-2.0). Fresh code, no code copied.
// SOURCE-NUANCE: MLC never sorts when nothing is cut; here the sort is only moved, because the draw and several
// samplers walk the list in order and the seeded result must not change.
// Row S69-MLC-LLM-PICK-TOPK-TOPP-WITHOUT-FULL-SORT-2026-10-18-01.
static bool sampler_step_takes_unsorted(llama_sampler_type type, const common_params_sampling & params,
                                        const llama_token_data_array & cur_p) {
    switch (type) {
        case llama_sampler_type::TOP_K:          return true;
        case llama_sampler_type::TOP_P:          return true;
        case llama_sampler_type::MIN_P:          return true;
        case llama_sampler_type::ADAPTIVE_P:     return true; // a no-op inside the loop; the step after the loop pays the sort first
        case llama_sampler_type::TOP_A:         return params.top_a <= 0.0f || cur_p.size <= 1;
        case llama_sampler_type::TFS_Z:          return params.tfs_z >= 1.0f || cur_p.size <= 2;
        case llama_sampler_type::TYPICAL_P:      return params.typical_p >= 1.0f;
        case llama_sampler_type::EPSILON_CUTOFF: return params.epsilon_cutoff <= 0.0f || cur_p.size <= 1;
        case llama_sampler_type::ETA_CUTOFF:     return params.eta_cutoff <= 0.0f || cur_p.size <= 1;
        case llama_sampler_type::XTC:            return params.xtc_probability <= 0.0f || params.xtc_threshold > 0.5f || cur_p.size < 2;
        case llama_sampler_type::TOP_N_SIGMA:    return params.top_n_sigma <= 0.0f || cur_p.size < 4;
        default:                                 return false;
    }
}

// no reasons to expose this function in header (common_sampler_queue_apply below is the test entry point)
static void sampler_queue(
    struct llama_context* ctx_main,
    const common_params_sampling& params,
    common_sampler * ctx_sampling,
    llama_token_data_array& cur_p,
    size_t   min_keep) {
    const float         temp = params.temp;
    const float         dynatemp_range = params.dynatemp_range;
    const float         dynatemp_exponent = params.dynatemp_exponent;
    const int32_t       top_k = params.top_k;
    const float         top_p = params.top_p;
    const float         min_p = params.min_p;
    const float         tfs_z = params.tfs_z;
    const float         typical_p = params.typical_p;
    const float         xtc_probability = params.xtc_probability;
    const float         xtc_threshold = params.xtc_threshold;
    const float         top_n_sigma = params.top_n_sigma;

    const std::vector<llama_sampler_type> & samplers_sequence = params.samplers_sequence;
    bool sort_owed = false; // a top_k that cut nothing skipped its sort (see sampler_step_takes_unsorted)
    auto pay_sort = [&]() {
        if (sort_owed && !cur_p.sorted) {
            llama_sample_top_k(ctx_main, &cur_p, 0, min_keep); // k <= 0: sort everything, cut nothing
        }
        sort_owed = false;
    };
    for (auto sampler_type : samplers_sequence) {
        if (sort_owed && !sampler_step_takes_unsorted(sampler_type, params, cur_p)) {
            pay_sort();
        }
        switch (sampler_type) {
            case llama_sampler_type::DRY        :
                if (ctx_sampling != nullptr) {
                    llama_sample_dry(ctx_main, ctx_sampling->smpl, &cur_p);
                }
                break;
            case llama_sampler_type::TOP_K      :
                if (!cur_p.sorted && (top_k <= 0 || top_k >= (int32_t) cur_p.size)) {
                    sort_owed = true;
                    break;
                }
                llama_sample_top_k(ctx_main, &cur_p, top_k, min_keep);
                break;
            case llama_sampler_type::TFS_Z      : llama_sample_tail_free(ctx_main, &cur_p, tfs_z,     min_keep); break;
            case llama_sampler_type::TYPICAL_P  : llama_sample_typical  (ctx_main, &cur_p, typical_p, min_keep); break;
            case llama_sampler_type::TOP_P      : llama_sample_top_p    (ctx_main, &cur_p, top_p,     min_keep); break;
            case llama_sampler_type::MIN_P      : llama_sample_min_p    (ctx_main, &cur_p, min_p,     min_keep); break;
            case llama_sampler_type::XTC        : llama_sample_xtc      (ctx_main, &cur_p, xtc_probability, xtc_threshold, min_keep); break;
            case llama_sampler_type::TOP_N_SIGMA: llama_sample_top_n_sigma(ctx_main, &cur_p, top_n_sigma); break;
            case llama_sampler_type::DIST       : llama_sample_dist  (ctx_main, &cur_p); break;
            case llama_sampler_type::INFILL     :
                if (ctx_sampling != nullptr && ctx_sampling->infill != nullptr) {
                    llama_sampler_apply(ctx_sampling->infill, &cur_p);
                }
                break;
            case llama_sampler_type::TOP_A      : sample_top_a(ctx_main, &cur_p, params.top_a, min_keep); break;
            case llama_sampler_type::EPSILON_CUTOFF: sample_epsilon_cutoff(ctx_main, &cur_p, params.epsilon_cutoff, min_keep); break;
            case llama_sampler_type::ETA_CUTOFF : sample_eta_cutoff(ctx_main, &cur_p, params.eta_cutoff, min_keep); break;
            case llama_sampler_type::TEMPERATURE:
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama_sample_entropy(ctx_main, &cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama_sample_temp(ctx_main, &cur_p, temp);
                }
                // Smoothing runs on the TEMPERATURE-SCALED logits, as koboldcpp does (sample_temperature
                // and sample_entropy in gpttype_adapter.cpp). Order matters: the polynomial does not
                // commute with scaling, f*(d/T)^2 != f*d^2/T, so smoothing first gave different
                // distributions at any T != 1. Greedy (temp <= 0 without dynatemp) skips it there too.
                if (dynatemp_range > 0 || temp > 0.0f) {
                    sample_smoothing(&cur_p, params.smoothing_factor, params.smoothing_curve);
                }
                break;
            default : break;
        }

    }
    pay_sort(); // the draw after the queue walks the list in order
    if (ctx_sampling != nullptr && ctx_sampling->adapt_p_ctx != nullptr) {
        // adaptive p should be put to the last, so we ignore the order in the sampler
        llama_sample_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx);
    }
}

void common_sampler_queue_apply(const common_params_sampling & params, llama_token_data_array & cur_p, size_t min_keep) {
    sampler_queue(nullptr, params, nullptr, cur_p, min_keep);
}

// The output constraint is EITHER the GBNF llama_grammar OR a sampler-shaped constraint
// (common_sampler::cnstr -- llguidance today). These three helpers are the only places that
// dispatch between them, so every grammar site below treats both the same way.
static bool constraint_present(const struct common_sampler * gsmpl) {
    return gsmpl->grammar != nullptr || gsmpl->cnstr != nullptr;
}

static void constraint_apply(struct common_sampler * gsmpl, struct llama_context * ctx_main, llama_token_data_array * cur_p) {
    if (gsmpl->grammar) {
        llama_grammar_apply(gsmpl->grammar, ctx_main, cur_p);
    } else if (gsmpl->cnstr) {
        llama_sampler_apply(gsmpl->cnstr, cur_p);
    }
}

static void constraint_accept(struct common_sampler * gsmpl, struct llama_context * ctx_main, llama_token token) {
    if (gsmpl->grammar) {
        llama_grammar_accept_token(gsmpl->grammar, ctx_main, token);
    } else if (gsmpl->cnstr) {
        llama_sampler_accept(gsmpl->cnstr, token);
    }
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!constraint_present(gsmpl)) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

// Keyed Gumbel-max draw over the candidates sampler_queue() left (logits already temperature-scaled,
// truncation samplers already applied) -- the same point at which vLLM swaps its RNG noise for the keyed
// noise. A context already seen in this generation is drawn with the ordinary RNG instead (see watermark.h).
static llama_token watermark_sample(common_sampler * ctx_sampling, llama_context * ctx_main, llama_token_data_array & cur_p) {
    const common_params_sampling & params = ctx_sampling->params;
    const int width = std::max(1, params.watermark_context_width);
    if (cur_p.size == 0 ||
        common_watermark::next_context_repeated(ctx_sampling->wm_generated, width, params.watermark_dedup_history)) {
        ctx_sampling->wm_n_skipped++;
        return llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);
    }
    std::vector<int32_t> ctx((size_t) width);
    common_watermark::next_context(ctx_sampling->wm_generated, width, ctx.data());
    const auto st = common_watermark::absorb(params.watermark_key, ctx.data(), width);

    std::vector<int32_t> ids(cur_p.size);
    std::vector<float>   logits(cur_p.size);
    for (size_t i = 0; i < cur_p.size; ++i) {
        ids[i]    = cur_p.data[i].id;
        logits[i] = cur_p.data[i].logit;
    }
    const int64_t best = common_watermark::gumbel_select(st, ids.data(), logits.data(), cur_p.size);
    if (best < 0) {
        ctx_sampling->wm_n_skipped++;
        return llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);
    }
    ctx_sampling->wm_n_marked++;
    return ids[(size_t) best];
}

// KGW green list (TGI's scheme): +delta on the green candidates' RAW logits, before temperature and the
// truncation samplers -- TGI applies its WatermarkLogitsProcessor ahead of its warpers. Every position is
// biased, as in TGI; the detector, not the generator, skips repeated (prev, token) pairs.
static void watermark_kgw_bias(common_sampler * ctx_sampling, llama_token_data_array & cur_p) {
    const common_params_sampling & params = ctx_sampling->params;
    const int32_t prev = ctx_sampling->wm_generated.empty() ? -1 : ctx_sampling->wm_generated.back();
    const uint64_t st   = common_watermark::kgw_prev_state(params.watermark_key, prev);
    for (size_t i = 0; i < cur_p.size; ++i) {
        if (common_watermark::kgw_green(st, cur_p.data[i].id, params.watermark_kgw_gamma)) {
            cur_p.data[i].logit += params.watermark_kgw_delta;
        }
    }
    cur_p.sorted = false;
    ctx_sampling->wm_n_marked++;
}

// SynthID tournament over the candidates sampler_queue() left (temperature-scaled logits): softmax, reweight
// by `depth` keyed coin layers, then draw with the ordinary RNG. A repeated context is drawn unmodified.
static llama_token watermark_synthid_sample(common_sampler * ctx_sampling, llama_context * ctx_main, llama_token_data_array & cur_p) {
    const common_params_sampling & params = ctx_sampling->params;
    const int width = std::max(1, params.watermark_context_width);
    if (cur_p.size == 0 ||
        common_watermark::next_context_repeated(ctx_sampling->wm_generated, width, params.watermark_dedup_history)) {
        ctx_sampling->wm_n_skipped++;
        return llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);
    }
    std::vector<int32_t> ctx((size_t) width);
    common_watermark::next_context(ctx_sampling->wm_generated, width, ctx.data());
    const uint64_t ch = common_watermark::synthid_context_hash(ctx.data(), width);

    float mx = -INFINITY;
    for (size_t i = 0; i < cur_p.size; ++i) {
        mx = std::max(mx, cur_p.data[i].logit);
    }
    std::vector<int32_t> ids(cur_p.size);
    std::vector<float>   probs(cur_p.size);
    double sum = 0.0;
    for (size_t i = 0; i < cur_p.size; ++i) {
        ids[i]   = cur_p.data[i].id;
        probs[i] = std::isfinite(cur_p.data[i].logit) ? std::exp(cur_p.data[i].logit - mx) : 0.0f;
        sum += probs[i];
    }
    if (!(sum > 0.0)) {
        ctx_sampling->wm_n_skipped++;
        return llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);
    }
    for (auto & p : probs) {
        p = (float) (p / sum);
    }
    common_watermark::synthid_reweight(params.watermark_key, ch, ids.data(), probs.data(), probs.size(),
                                       std::max(1, params.watermark_synthid_depth));
    for (size_t i = 0; i < cur_p.size; ++i) {
        cur_p.data[i].logit = probs[i] > 0.0f ? std::log(probs[i]) : -INFINITY;
    }
    cur_p.sorted = false;
    ctx_sampling->wm_n_marked++;
    return llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);
}

static llama_token llama_sampling_sample_impl(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first,
                  const int idx_cfg = -1);

static llama_token llama_sampling_sample_impl_inner(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first,
                  const int idx_cfg) {
    const common_params_sampling & params = ctx_sampling->params;

    const float   temp            = params.temp;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;
    const float   adaptive_target = params.adaptive_target;

    std::vector<float> original_logits;
    llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, /* grammar_first= */ grammar_first, &original_logits, idx_cfg);
    llama_token_data_array & cur_p = ctx_sampling->cur_p;
    if (constraint_present(ctx_sampling) && !grammar_first) {
        GGML_ASSERT(!original_logits.empty());
    }
    auto & rbudget = ctx_sampling->rbudget;

    llama_token id = 0;
    float * logits = llama_get_logits_ith(ctx_main, idx);
    // apply reasoning budget first
    common_reasoning_budget_apply(rbudget, &cur_p);
    // Sample grammar first for resampling
    if (constraint_present(ctx_sampling) && grammar_first && grammar_should_apply(ctx_sampling)) {
        // Apply grammar constraints to all candidates
        constraint_apply(ctx_sampling, ctx_main, &cur_p);

        // The caller masked some logits (min_tokens) and the grammar left nothing: put the masked logits back and let the
        // grammar decide again (vLLM restores the stop logits in this case, builtin.py:254-277, Apache-2.0, not copied)
        if (!ctx_sampling->restore_if_empty.empty()) {
            bool any_finite = false;
            for (size_t i = 0; i < cur_p.size && !any_finite; ++i) {
                any_finite = std::isfinite(cur_p.data[i].logit) || cur_p.data[i].logit > 0.0f;
            }
            if (!any_finite) {
                for (const auto & r : ctx_sampling->restore_if_empty) {
                    for (size_t i = 0; i < cur_p.size; ++i) {
                        if (cur_p.data[i].id == r.first) {
                            cur_p.data[i].logit = r.second;
                            break;
                        }
                    }
                }
                constraint_apply(ctx_sampling, ctx_main, &cur_p);
            }
        }

        // The grammar/choice left no candidate at all (e.g. choice=["zzz"] on a vocabulary that cannot spell it).
        // Every sampler below would either abort on the all -inf row or hand back a token the grammar rejects, so
        // report "no token" and let the caller fail this one request.
        bool any_allowed = false;
        for (size_t i = 0; i < cur_p.size && !any_allowed; ++i) {
            any_allowed = cur_p.data[i].logit != -INFINITY;
        }
        if (!any_allowed) {
            LOG_WRN("%s: the grammar/choice allows no token of this vocabulary at this position\n", __func__);
            return LLAMA_TOKEN_NULL;
        }
    }

    // llama_sampler_apply
    if (ctx_sampling->fsd && mirostat == 0) {
        // FSD (S69-INFERFLOW-PICK-FSD-ANTILM-SAMPLER) chooses the token itself. The queue (temperature + the truncation samplers) still
        // runs first when temp > 0; temp <= 0 skips it because its temperature step would divide by it. The anti-LM re-rank then
        // replaces the final draw, so a request that sets fsd_alpha is deterministic at any temperature unless random_first_n is on.
        if (temp > 0.0f) {
            sampler_queue(ctx_main, params, ctx_sampling, cur_p, std::max(1, params.min_keep));
        }
        llama_sampler_apply(ctx_sampling->fsd, &cur_p);
        id = cur_p.data[cur_p.selected].id;
    } else if (temp < 0.0) {
        // greedy sampling, with probs
        llama_sample_softmax(ctx_main, &cur_p);
        id = cur_p.data[0].id;
    } else if (temp == 0.0) {
        // greedy sampling, no probs
        id = llama_sample_token_greedy(ctx_main, &cur_p);
    } else {
        if (mirostat == 1) {
            const int mirostat_m = 100;
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat_with_rng(ctx_main, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_sampling->mirostat_mu,
                                                      ctx_sampling->rng);
        } else if (mirostat == 2) {
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat_v2_with_rng(ctx_main, &cur_p, mirostat_tau, mirostat_eta, &ctx_sampling->mirostat_mu,
                                                         ctx_sampling->rng);
        } else if (adaptive_target >= 0.0f && ctx_sampling->adapt_p_ctx!=nullptr) {
            // adaptive p sampling
            llama_prep_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx);
            // adaptive-p alone draws from the whole vocabulary (see common_adaptive_untruncated): in FLOOR mode apply the
            // floor first, at the same point a user-listed min_p would run inside sampler_queue. REFUSE is enforced where
            // the request/CLI is parsed, ALLOW leaves the candidate list as it is.
            if (params.adaptive_untruncated == COMMON_ADAPTIVE_UNTRUNCATED_FLOOR &&
                common_sampling_adaptive_p_untruncated(params, llama_n_vocab(llama_get_model(ctx_main)))) {
                llama_sample_min_p(ctx_main, &cur_p, params.adaptive_floor, std::max(1, params.min_keep));
            }
            sampler_queue(ctx_main, params, ctx_sampling, cur_p, std::max(1, params.min_keep));
            id = llama_sample_token_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx, ctx_sampling->rng);
        } else {
            // temperature sampling
            size_t min_keep = std::max(1, params.min_keep);

            const bool wm_on = params.watermark && params.watermark_key_set;
            if (wm_on && params.watermark_scheme == common_watermark::SCHEME_KGW) {
                watermark_kgw_bias(ctx_sampling, cur_p);
            }
            sampler_queue(ctx_main, params,ctx_sampling, cur_p, min_keep);
            if (wm_on && params.watermark_scheme == common_watermark::SCHEME_SYNTHID) {
                id = watermark_synthid_sample(ctx_sampling, ctx_main, cur_p);
            } else if (wm_on && params.watermark_scheme == common_watermark::SCHEME_GUMBEL) {
                id = watermark_sample(ctx_sampling, ctx_main, cur_p);
            } else {
                // skew (row S69-APHRODITE-PICK-SKEW-SAMPLING-2026-10-18-01): reshape only what the queue left, right before the
                // draw, as ExLlamaV2 does in its final multinomial. The refusals in arg.cpp / server-schema.cpp keep it off the
                // branches above (mirostat, adaptive-p, FSD) and off the watermark draws, which never reach this line.
                common_sampling_skew_apply(&cur_p, params.skew);
                id = llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);
            }

        }
    }

    if (grammar_first || !grammar_should_apply(ctx_sampling)) {
        return id;
    }

    if (constraint_present(ctx_sampling) && !grammar_first && grammar_should_apply(ctx_sampling)) {
        // Get a pointer to the logits
        float * logits = llama_get_logits_ith(ctx_main, idx);

        // Create an array with a single token data element for the sampled id
        llama_token_data single_token_data = {id, logits[id], 0.0f};
        llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

        // Apply grammar constraints to the single token
        constraint_apply(ctx_sampling, ctx_main, &single_token_data_array);

        // Check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
        bool is_valid = single_token_data_array.data[0].logit != -INFINITY;

        // If the token is not valid according to the grammar, perform resampling
        if (!is_valid) {
            LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, common_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
            std::copy(original_logits.begin(), original_logits.end(), logits);

            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ true, idx_cfg);
        }
    }
    ctx_sampling->n_valid = temp == 0.0f ? 0 : cur_p.size;

    return id;
}

// EOG sampled while a reasoning block is open: close the block instead of stopping with empty content (row
// S69-ENGREV-BEELLAMA-EOG-IN-REASONING-FORCE-CLOSE-2026-10-27-01; idea from BeeLlama common/sampling.cpp, MIT, rewritten).
// The repair moves the budget ctx to FORCING; the second draw then sees every logit but the first end-sequence token at -inf.
// A sampler without reasoning-budget tracking (no rbudget) takes the early return and is unchanged.
static llama_token llama_sampling_sample_impl(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first,
                  const int idx_cfg) {
    llama_token id = llama_sampling_sample_impl_inner(ctx_sampling, ctx_main, ctx_cfg, idx, grammar_first, idx_cfg);
    if (id != LLAMA_TOKEN_NULL && ctx_sampling->rbudget) {
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_main));
        if (common_reasoning_budget_repair_eog(ctx_sampling->rbudget, llama_vocab_is_eog(vocab, id))) {
            LOG_WRN("%s: sampled EOG (%d) while reasoning is open; forcing the end-of-reasoning sequence instead\n", __func__, id);
            id = llama_sampling_sample_impl_inner(ctx_sampling, ctx_main, ctx_cfg, idx, grammar_first, idx_cfg);
        }
    }
    return id;
}

bool common_sampler_topk_prefilter_k(const common_params_sampling & params, bool penalties_active, bool has_cfg, bool has_constraint,
                                     bool has_special_sampler, bool has_server_bias, int32_t n_vocab, int32_t * k_out) {
    if (!params.topk_prefilter || penalties_active || has_cfg || has_constraint || has_special_sampler || has_server_bias) {
        return false;
    }
    if (!(params.temp > 0.0f) || params.mirostat != 0 || params.adaptive_target >= 0.0f || params.fsd_alpha >= 0.0f ||
        params.no_repeat_ngram_size > 0 || !params.suppress_sequences.empty() || params.n_probs > 0 || (params.watermark && params.watermark_key_set)) {
        return false;
    }
    if (params.top_k <= 0 || params.top_k > 1024 || params.top_k >= n_vocab) {
        return false;
    }
    // the first ACTIVE sampler of the chain must be top-k; a leading DRY with multiplier 0 does nothing
    bool first_is_topk = false;
    for (const auto t : common_sampler_resolved_sequence(params)) {
        if (t == llama_sampler_type::DRY && params.dry_multiplier <= 0.0f) {
            continue;
        }
        first_is_topk = t == llama_sampler_type::TOP_K;
        break;
    }
    if (!first_is_topk) {
        return false;
    }
    *k_out = std::min<int32_t>(n_vocab, std::max<int32_t>(params.top_k, std::max<int32_t>(1, params.min_keep)));
    return *k_out <= 1024;
}

bool common_sampler_greedy_prefilter_ok(const common_params_sampling & params, bool penalties_active, bool has_cfg, bool has_constraint,
                                        bool has_special_sampler, bool has_server_bias) {
    return params.topk_prefilter && params.temp == 0.0f && !penalties_active && !has_cfg && !has_constraint && !has_special_sampler &&
           !has_server_bias && params.mirostat == 0 && params.adaptive_target < 0.0f && params.fsd_alpha < 0.0f &&
           params.no_repeat_ngram_size <= 0 && params.suppress_sequences.empty() && params.n_probs <= 0 && !(params.watermark && params.watermark_key_set);
}

static llama_token_data_array llama_sampling_prepare_impl(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first,
                  std::vector<float> * original_logits,
                  const int idx_cfg = -1) {
    const common_params_sampling & params = ctx_sampling->params;

    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    const int32_t penalty_last_n  = params.penalty_last_n < 0 ? params.n_prev : params.penalty_last_n;
    const float   penalty_repeat  = params.penalty_repeat;
    // penalty_scope_generated: the window pass below applies repetition only; frequency/presence are applied
    // afterwards over the generated tokens alone (wm_generated = every token accepted with is_generated).
    const bool    scope_generated = params.penalty_scope_generated;
    const float   penalty_freq    = scope_generated ? 0.0f : params.penalty_freq;
    const float   penalty_present = scope_generated ? 0.0f : params.penalty_present;

    const bool    penalize_nl     = params.penalize_nl;

    auto & prev = ctx_sampling->prev;
    auto & cur  = ctx_sampling->cur;

    // Get a pointer to the logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    // Count non-finite RAW logits before any bias touches the row (see common_sampler::n_nonfinite_last).
    // Logged once per sampler; the caller decides whether to refuse (tools/server fails the slot).
    ctx_sampling->n_nonfinite_last = common_count_nonfinite_logits(logits, n_vocab);
    if (ctx_sampling->n_nonfinite_last > 0) {
        if (ctx_sampling->n_nonfinite_total == 0) {
            LOG_ERR("%s: the model produced %d non-finite logit(s) (NaN/+inf) out of %d at this position\n",
                    __func__, ctx_sampling->n_nonfinite_last, n_vocab);
        }
        ctx_sampling->n_nonfinite_total += (uint64_t) ctx_sampling->n_nonfinite_last;
    }

    if (constraint_present(ctx_sampling) && !grammar_first) {
        GGML_ASSERT(original_logits != NULL);
        // Only make a copy of the original logits if we are not applying grammar checks, not sure if I actually have to do this.
        *original_logits = {logits, logits + n_vocab};
    }

    // apply params.logit_bias -- decision #5 Part C changed the field to
    // std::vector<llama_logit_bias> ({.token,.bias}); was unordered_map<token,float> map syntax.
    for (const auto & lb : params.logit_bias) {
        logits[lb.token] += lb.bias;
    }

    // Model suppress tokens (GGUF tokenizer.ggml.suppress_tokens, written by the gemma/minimax/qwen3tts converters
    // from generation_config.json / the model card) are never sampled: -INFINITY, as upstream does in
    // common/sampling.cpp (it merges them into the logit-bias sampler). The loader read them since the merge
    // (src/llama-vocab.cpp) but nothing called llama_vocab_get_suppress_tokens, so they were silently ignored.
    {
        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(
                llama_model_get_vocab(llama_get_model(ctx_main)), &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            if (suppress[i] >= 0 && suppress[i] < n_vocab) {
                logits[suppress[i]] = -INFINITY;
            }
        }
    }

    // apply params.allowed_token_ids -- a HARD allow-list, applied here (before `cur` is built)
    // so that every sampler downstream sees an already-constrained candidate set.
    if (!params.allowed_token_ids.empty()) {
        std::vector<bool> allowed((size_t) n_vocab, false);
        size_t n_in_range = 0;
        for (const llama_token id : params.allowed_token_ids) {
            // Out-of-range ids are dropped rather than asserted: an id from a different
            // vocabulary must not abort generation.
            if (id >= 0 && id < n_vocab && !allowed[(size_t) id]) {
                allowed[(size_t) id] = true;
                n_in_range++;
            }
        }
        if (n_in_range == 0) {
            // NEITHER of the silent options is taken here, and both are worse than a warning:
            // masking everything leaves zero candidates, and softmax over all -INFINITY is NaN --
            // a failure that surfaces far from its cause; ignoring the list quietly DROPS a
            // constraint the caller asked for, which for an allow-list is the dangerous
            // direction. So: no mask, and say so.
            LOG_WRN("allowed_token_ids: none of the %zu requested id(s) are within "
                        "[0, %d) -- the allow-list is NOT applied\n",
                        params.allowed_token_ids.size(), n_vocab);
        } else {
            for (llama_token id = 0; id < n_vocab; ++id) {
                if (!allowed[(size_t) id]) {
                    logits[id] = -INFINITY;
                }
            }
        }
    }

    if (ctx_cfg) {
        // idx_cfg < 0 (default): guidance row is at the SAME batch index as the main token --
        // the tools/completion CLI shape (separate llama_context, byte-identical to before this
        // param existed). idx_cfg >= 0: guidance row lives at its OWN batch index, for a guidance
        // sequence sharing ctx_main's batch (tools/server's in-context shape).
        float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx_cfg < 0 ? idx : idx_cfg);
        llama_sample_apply_guidance(ctx_main, logits, logits_guidance, params.cfg_scale);
    }

    if (ctx_sampling->elb_states.size() > ctx_sampling->elb_idx) {
        common_expiring_logit_bias_apply(ctx_sampling, logits);
    }

    // S69-MNN-PICK-TOPK-RAW-LOGIT-THRESHOLD-PREFILTER-2026-10-05-01: when top-k leads the chain and nothing else needs the whole vocabulary,
    // build the k survivors straight from the raw logits (exact, same order as llama_sort) instead of a 12-byte x n_vocab array.
    if (params.topk_prefilter) {
        // conservative: any non-neutral penalty setting keeps the full array (they act on arbitrary window tokens)
        const bool pen_active = penalty_repeat != 1.0f || params.penalty_freq != 0.0f || params.penalty_present != 0.0f;
        const bool server_bias = ctx_sampling->server_biases != nullptr && (int) ctx_sampling->server_biases->size() == n_vocab;
        int32_t kk = 0;
        if (common_sampler_topk_prefilter_k(params, pen_active, ctx_cfg != nullptr, constraint_present(ctx_sampling),
                                            ctx_sampling->rbudget != nullptr || ctx_sampling->fsd != nullptr || ctx_sampling->adapt_p_ctx != nullptr,
                                            server_bias, n_vocab, &kk)) {
            cur.resize((size_t) kk);
            const size_t got = common_topk_scan(logits, n_vocab, kk, cur.data());
            cur.resize(got);
            ctx_sampling->cur_p = { cur.data(), cur.size(), -1, true };
            ctx_sampling->n_prefiltered++;
            return ctx_sampling->cur_p;
        }
        // greedy: the first largest raw logit is the whole answer (llama_sample_token_greedy keeps the first max)
        if (common_sampler_greedy_prefilter_ok(params, pen_active, ctx_cfg != nullptr, constraint_present(ctx_sampling),
                                               ctx_sampling->rbudget != nullptr || ctx_sampling->fsd != nullptr || ctx_sampling->adapt_p_ctx != nullptr,
                                               server_bias)) {
            const int32_t best = common_argmax_scan(logits, n_vocab);
            if (best >= 0) {
                cur.resize(1);
                cur[0] = llama_token_data{ best, logits[best], 0.0f };
                ctx_sampling->cur_p = { cur.data(), cur.size(), -1, true };
                ctx_sampling->n_prefiltered++;
                return ctx_sampling->cur_p;
            }
        }
    }

    cur.resize(n_vocab);

    if ((ctx_sampling->server_biases != nullptr) && (ctx_sampling->server_biases->size() == n_vocab)) {
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id] + ctx_sampling->server_biases->at(token_id), 0.0f};
        }
    } else {
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    ctx_sampling->cur_p = { cur.data(), cur.size(), false };

    llama_token_data_array & cur_p = ctx_sampling->cur_p;

    // apply penalties
    const auto& penalty_tokens = params.use_penalty_prompt_tokens ? params.penalty_prompt_tokens : prev;
    // Penalty fade (row S69-EXLLAMAV3-PICK-PENALTY-FADE-2026-10-18-01, see sampling-penalty-fade.h): the history reaches
    // penalty_decay tokens past the penalty_last_n window, where the penalties fade linearly to 0. A disabled window
    // (penalty_last_n == 0) stays disabled.
    const int32_t fade_decay = (params.penalty_decay > 0 && penalty_last_n > 0) ? params.penalty_decay : 0;
    int penalty_tokens_used_size = (int) std::min<int64_t>((int64_t) penalty_tokens.size(),
                                                           (int64_t) penalty_last_n + fade_decay);
    // penalize_prompt = false (unsloth U283, SOURCE-READ tmp/unsloth_pr_1107/U283.diff, MIT): the
    // window is cut to the generated tail of `prev` -- the newest wm_generated.size() entries are exactly this
    // request's generated tokens -- so repeat/frequency/presence (and slope/fade) never see the prompt. Upstream
    // skips feeding the prompt to the sampler at all; this fork keeps `prev` whole because no-repeat n-gram,
    // suppress_sequences and FSD read prompt tokens from it. An explicit penalty_prompt_tokens list is the
    // client's own history and is left alone.
    if (!params.penalize_prompt && !params.use_penalty_prompt_tokens) {
        penalty_tokens_used_size = std::min<int>(penalty_tokens_used_size, (int) ctx_sampling->wm_generated.size());
    }
    if (penalty_tokens_used_size) {
        // llama_token_nl() can legitimately return LLAMA_TOKEN_NULL (-1). Some vocabs (e.g. Falcon3's
        // BPE tokenizer) tokenize "\n" to zero tokens, and the loader's fallback
        // (llama-vocab.cpp, linefeed_id = special_pad_id) runs before LLM_KV_TOKENIZER_PAD_ID is read
        // from the GGUF, so it copies the BPE default, which is itself LLAMA_TOKEN_NULL. Reading
        // logits[-1] is then an out-of-bounds access one float before the current position's logit
        // row, which segfaults when that address happens to be unmapped.
        const llama_token nl_token = llama_token_nl(llama_get_model(ctx_main));
        const float nl_logit = nl_token != LLAMA_TOKEN_NULL ? logits[nl_token] : 0.0f;

        const llama_token * window = penalty_tokens.data() + penalty_tokens.size() - penalty_tokens_used_size;

        // Slope outside (0, 1) is the uniform penalty. koboldcpp resets slope <= 0 to 1 in both of its
        // sampler paths (gpttype_adapter.cpp:6055 and batch_rep_pen_init), so a kobold client sending
        // rep_pen_slope = 0 means "off", never "no penalty on the older half"; this matches it.
        if (fade_decay > 0) {
            // Takes precedence over the slope; the CLI and the server refuse the two together.
            common_penalty_fade_apply(&cur_p, window, (size_t) penalty_tokens_used_size, penalty_last_n, fade_decay,
                    penalty_repeat, penalty_freq, penalty_present);
        } else if (params.penalty_slope > 0.0f && params.penalty_slope < 1.0f && penalty_repeat > 1.0f && penalty_tokens_used_size > 1) {
            // Repetition-penalty slope (KoboldAI/koboldcpp behaviour; koboldcpp is AGPL-3.0, behaviour read
            // from sample_rep_pen / batch_rep_pen_apply, gpttype_adapter.cpp:1747 and :4545, no code copied):
            // tokens seen only in the OLDER half of the window get 1 + (repeat - 1) * slope, tokens in
            // the newer half the full penalty. Frequency/presence stay uniform over the whole window.
            const int split = penalty_tokens_used_size / 2;
            llama_sample_repetition_penalties(ctx_main, &cur_p, window, penalty_tokens_used_size,
                    1.0f, penalty_freq, penalty_present);
            llama_sample_repetition_penalties(ctx_main, &cur_p, window + split, penalty_tokens_used_size - split,
                    penalty_repeat, 0.0f, 0.0f);
            const std::unordered_set<llama_token> newer(window + split, window + penalty_tokens_used_size);
            std::vector<llama_token> older_only;
            for (int i = 0; i < split; ++i) {
                if (!newer.count(window[i])) {
                    older_only.push_back(window[i]);
                }
            }
            if (!older_only.empty()) {
                llama_sample_repetition_penalties(ctx_main, &cur_p, older_only.data(), older_only.size(),
                        1.0f + (penalty_repeat - 1.0f) * params.penalty_slope, 0.0f, 0.0f);
            }
        } else {
            llama_sample_repetition_penalties(ctx_main, &cur_p, window,
                    penalty_tokens_used_size, penalty_repeat, penalty_freq, penalty_present);
        }

        if (scope_generated && !ctx_sampling->wm_generated.empty() &&
                (params.penalty_freq != 0.0f || params.penalty_present != 0.0f)) {
            llama_sample_repetition_penalties(ctx_main, &cur_p, ctx_sampling->wm_generated.data(),
                    ctx_sampling->wm_generated.size(), 1.0f, params.penalty_freq, params.penalty_present);
        }

        if (!penalize_nl && nl_token != LLAMA_TOKEN_NULL) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == nl_token) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    // No-repeat n-gram runs after the penalties (a -inf logit is final whatever came before it) and before
    // the grammar, so a grammar-first caller still sees the ban.
    if (params.no_repeat_ngram_size > 0) {
        const int32_t nr_last_n = params.no_repeat_ngram_last_n < 0 ? (int32_t) llama_n_ctx(ctx_main) : params.no_repeat_ngram_last_n;
        apply_no_repeat_ngram(prev, params.no_repeat_ngram_size, nr_last_n, &cur_p);
    }

    if (!params.suppress_sequences.empty()) {
        apply_suppress_sequences(prev, params.suppress_sequences, &cur_p);
    }

    // apply grammar checks before sampling logic
    if (grammar_first && constraint_present(ctx_sampling)) {
        constraint_apply(ctx_sampling, ctx_main, &cur_p);
    }

    return cur_p;
}

llama_token common_sampler_sample_legacy(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx) {
    // Call the implementation function with is_resampling set to false by default.
    // idx_cfg left at its -1 default: the legacy/CLI path always reads the guidance context at
    // the SAME batch index as the main token, unaffected by the idx_cfg parameter added for
    // tools/server (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01, SOURCE-READ done,
    // RESEARCH-DONE: store.get on that row before this contract was chosen).
    ctx_sampling->last_draw_mask_first = false;
    return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ false);
}

// New trailing params ctx_cfg/idx_cfg (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01,
// RESEARCH-DONE: full row read via store.get before adding this API). ctx_cfg's PRE-EXISTING
// default stays nullptr so every one of this overload's current callers (tools/server,
// tools/mtmd, tools/tts -- grepped before this edit) is byte-for-byte unaffected; this is additive
// plumbing for a CFG feature landing in a later commit on this same branch, not a capability
// regression.
llama_token common_sampler_sample(
    struct common_sampler * ctx_sampling,
    struct llama_context * ctx_main,
    const int idx,
    bool grammar_first,
    struct llama_context * ctx_cfg,
    int idx_cfg) {
    ctx_sampling->last_draw_mask_first = grammar_first;
    // Call the implementation function with is_resampling set to false by default
    return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ grammar_first, idx_cfg);
}

llama_token_data_array llama_sampling_prepare(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool grammar_first,
                  std::vector<float> * original_logits,
                  int idx_cfg) {
    return llama_sampling_prepare_impl(ctx_sampling,ctx_main, ctx_cfg, idx, grammar_first, original_logits, idx_cfg);
}

bool common_sampler_grammar_failed(const struct common_sampler * gsmpl, std::string * err) {
    if (gsmpl == nullptr || gsmpl->cnstr == nullptr) {
        return false;
    }
    return llama_sampler_llg_failed(gsmpl->cnstr, err) || llama_sampler_xgr_failed(gsmpl->cnstr, err);
}

void common_sampler_accept(struct common_sampler * ctx_sampling, llama_token id, bool accept_grammar) {
    common_sampler_accept(ctx_sampling, nullptr, id, accept_grammar);
}

void common_sampler_accept(
        struct common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token token,
        bool is_generated) {
    // LLAMA_TOKEN_NULL is common_sampler_sample's "the grammar/choice allows nothing" answer. Feeding it to the
    // grammar or the history would index out of the vocabulary; every caller that can see it (speculative accept
    // loops included) runs inside a per-request error handler, so refuse it here with the reason.
    if (token == LLAMA_TOKEN_NULL) {
        throw std::invalid_argument("grammar/choice cannot be satisfied by this model's vocabulary (no allowed token)");
    }
    // Bounded window. The old body erased the front whenever prev was non-empty, which keeps the size
    // CONSTANT -- correct only while prev was prefilled to n_prev. common_sampler_reset() clears it,
    // and llama-server resets on every slot launch, so the window collapsed to ONE token: the repeat
    // penalty saw only the last token whatever repeat_last_n said (measured 2026-09-26 via the slope
    // test). Capacity covers penalty_last_n when it exceeds n_prev, so the penalty sees its full window.
    const auto & sp = ctx_sampling->params;
    // No-repeat n-gram needs its own (usually longer) window; it widens the cap only while it is on, so
    // the memory of every other configuration is unchanged. -1 = the context size, as dry_penalty_last_n.
    const int32_t nr_cap = sp.no_repeat_ngram_size <= 0 ? 0
                         : (sp.no_repeat_ngram_last_n < 0 ? (ctx_main ? (int32_t) llama_n_ctx(ctx_main) : sp.n_prev) : sp.no_repeat_ngram_last_n);
    int32_t ss_cap = 0; // suppress_sequences needs the longest banned prefix (all tokens of a banned sequence but its last)
    for (const auto & seq : sp.suppress_sequences) {
        ss_cap = std::max<int32_t>(ss_cap, (int32_t) seq.size());
    }
    // Penalty fade: the decay range lies past the penalty window, so the history must hold both.
    const int32_t fade_cap = sp.penalty_decay <= 0 ? 0
                           : (sp.penalty_last_n < 0 ? sp.n_prev : sp.penalty_last_n) + sp.penalty_decay;
    const size_t cap = (size_t) std::max({ sp.n_prev, sp.penalty_last_n, nr_cap, ss_cap, fade_cap, 1 });
    while (ctx_sampling->prev.size() >= cap) {
        ctx_sampling->prev.erase(ctx_sampling->prev.begin());
    }
    ctx_sampling->prev.push_back(token);
    if (is_generated) {
        ctx_sampling->wm_generated.push_back(token);
    }

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(ctx_sampling);
    if (ctx_sampling->rbudget && is_generated) {
        common_reasoning_budget_accept(ctx_sampling->rbudget, token);
    }

    if (accept_grammar) {
        constraint_accept(ctx_sampling, ctx_main, token);
    }
    // DRY history: prompt tokens only when penalize_prompt (default true, the upstream behaviour)
    if (ctx_sampling->smpl && (is_generated || sp.penalize_prompt)) {
        llama_sampler_dry_accept(ctx_sampling->smpl, token);
    }
    if (ctx_sampling->fsd) {
        llama_sampler_accept(ctx_sampling->fsd, token);     // prompt tokens too: they seed the anti-LM n-gram table
    }

    if (ctx_sampling->is_decoding) {
        ctx_sampling->playing_text += common_token_to_piece(ctx_main, token, true);

        common_expiring_logit_bias_accept(ctx_sampling, ctx_main);
    }
}

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
            });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

std::vector<llama_token> llama_sampling_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<llama_token> & draft) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft);
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const std::vector<llama_token> & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, ctx, id, true);

        result.push_back(id);

        // stop at an EOG: draft tokens after it are never output but would stay in the context. A trailing EOG is still
        // accepted (a replayed draft ends on the target's own last token, which can be EOG). Upstream llama.cpp d280808f
        // (#29638, MIT); row S69-LLAMACPP-PICK-DRAFT-STOP-AT-EOG-2026-10-18-01.
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, ctx, id, true);

        result.push_back(id);
    }

    return result;
}

common_spec_dist common_sampler_sampling_dist(struct common_sampler * gsmpl, llama_token picked) {
    // The distribution the LAST common_sampler_sample() call actually drew `picked` from. The plain temperature path
    // (sampling.cpp: sampler_queue, then llama_sample_token_with_rng) draws from softmax over the LOGITS left in cur_p
    // -- truncation, temperature, penalties and DRY already applied. The candidates' `p` field is NOT that
    // distribution: DIST is softmax(normalize=false) (src/llama.cpp llama_sample_dist), so p = exp(logit - max) and
    // the top token always reads 1.0 (MEASURED 2026-09-28: 'top p=1.000' at every drafted position). Paths that do
    // not draw from softmax(logits) -- greedy (temp <= 0), mirostat, adaptive-p, watermarking, a grammar or constraint
    // that was NOT applied first -- get a one-hot at `picked`: exact, and rejection sampling then reduces to sample-and-match.
    // A grammar applied FIRST (common_sampler_sample grammar_first, the draft model's call) masks cur_p to -inf before the chain, so
    // softmax over the logits left in cur_p is the masked, renormalised distribution the draw came from (the -inf entries drop out below).
    const common_params_sampling & sp = gsmpl->params;
    const bool softmax_draw = sp.temp > 0.0f && sp.mirostat == 0 && !(sp.adaptive_target >= 0.0f && gsmpl->adapt_p_ctx) &&
                              !(sp.watermark && sp.watermark_key_set) && (!constraint_present(gsmpl) || gsmpl->last_draw_mask_first);
    common_spec_dist d;
    const llama_token_data_array * cur_p = &gsmpl->cur_p;
    if (softmax_draw && cur_p->size > 0) {
        float mx = -INFINITY;
        for (size_t k = 0; k < cur_p->size; ++k) {
            mx = std::max(mx, cur_p->data[k].logit);
        }
        double sum = 0.0;
        d.ids.reserve(cur_p->size);
        d.probs.reserve(cur_p->size);
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float e = std::isfinite(cur_p->data[k].logit) ? expf(cur_p->data[k].logit - mx) : 0.0f;
            if (e > 0.0f) {
                d.ids.push_back(cur_p->data[k].id);
                d.probs.push_back(e);
                sum += e;
            }
        }
        if (sum > 0.0 && std::isfinite(sum)) {
            for (auto & pr : d.probs) {
                pr = (float) (pr / sum);
            }
            if (common_spec_dist_prob(d, picked) > 0.0f) {
                return d;
            }
        }
    }
    d.ids.assign(1, picked);
    d.probs.assign(1, 1.0f);
    return d;
}

std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx,
        const std::vector<int> & idxs, const std::vector<llama_token> & draft, const std::vector<common_spec_dist> & dists,
        std::mt19937 & rng, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(dists.size() == draft.size() && "one draft distribution per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // run the target chain at this position (grammar, penalties, temperature ... all applied), then judge the draft
        // token against the resulting p instead of against this one draw
        const llama_token picked = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        const common_spec_dist p = common_sampler_sampling_dist(gsmpl, picked);
        // opt-in LOSSY relaxed acceptance (S69-INFERFLOW-PICK-RELAXED-SPEC-ACCEPT): a draft token inside the target nucleus is kept; the count of
        // positions kept only by that rule is n_spec_relaxed so quality can be audited against the acceptance gain
        const common_spec_accept_result r = (gsmpl->params.spec_relaxed_top_p > 0.0f && gsmpl->params.temp > 0.0f)
            ? common_spec_accept_relaxed(p, dists[i], draft[i], rng, gsmpl->params.spec_relaxed_top_p)
            : common_spec_accept_rejection(p, dists[i], draft[i], rng);
        if (r.relaxed) {
            gsmpl->n_spec_relaxed++;
        }

        // no explicit playing_text append here: common_sampler_accept appends the piece itself while is_decoding is set,
        // so appending first doubled every accepted piece in the text the expiring-logit-bias search reads
        common_sampler_accept(gsmpl, ctx, r.token, true);

        result.push_back(r.token);

        // an accepted EOG ends the run as in common_sampler_sample_and_accept_n (upstream d280808f)
        if (!r.accepted || (llama_vocab_is_eog(vocab, r.token) && i + 1 < draft.size())) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, ctx, id, true);

        result.push_back(id);
    }

    return result;
}

// Upstream spelling of the rejection verify (common/sampling.h): draft_q[i] holds the draft's candidates at draft position i.
// One implementation is kept -- the fork's overload above (common_spec_accept_rejection against the true distribution of the
// target draw) -- so the candidates become a common_spec_dist (entries with p > 0; common_spec_dist_prob renormalises) and the
// draws come from the sampler's own speculative stream, independent of the draft as upstream requires. (merge 2026-11-04)
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");
    std::vector<common_spec_dist> dists(draft_q.size());
    for (size_t i = 0; i < draft_q.size(); ++i) {
        for (const auto & td : draft_q[i]) {
            if (td.p > 0.0f) {
                dists[i].ids.push_back(td.id);
                dists[i].probs.push_back(td.p);
            }
        }
    }
    return common_sampler_sample_and_accept_n_rejection(gsmpl, ctx, idxs, draft, dists, gsmpl->speculative_rng, grammar_first);
}

// upstream spelling: assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }
    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

std::vector<llama_token> common_sampler_sample_and_accept_n(
        struct common_sampler * gsmpl,
        struct llama_context * ctx,
        const std::vector<int> & idxs,
        const std::vector<llama_token> & draft,
        const std::vector<common_speculative_token_dist> & dists,
        bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(dists.size() == draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const auto emit = [&](llama_token id) {
        // accept appends the piece to playing_text itself (is_decoding); an explicit append here doubled it
        common_sampler_accept(gsmpl, ctx, id, true);
        result.push_back(id);
    };
    // Row S69-VLLM-W2-DFLASH2-ACCEPT-USES-CUR-P-2026-10-27-01. This used to judge the draft token against cur_p->data[].p, the
    // candidates' p field after common_sampler_sample(). That field is NOT the distribution the target drew from: the draw is a
    // softmax over the LOGITS left by the sampler chain, and DIST is softmax(normalize=false), so p = exp(logit - max) (the top token
    // reads 1.0, everything else is relative to it) and for greedy / grammar / mirostat / watermark paths it is not a probability at
    // all (MEASURED by tests/test-spec-proposal-accept.cpp, fail-before: at temp 1.0 a draft token with p(x)=0.22 was accepted
    // 100% of the time, and at temp 0.0 the top token was accepted 0% of the time). common_sampler_sampling_dist() returns the true,
    // normalised distribution of that same draw (one-hot for the paths that do not draw from softmax), and
    // common_spec_accept_rejection() is the rule test-spec-accept.cpp checks against a chi-square oracle, so both overloads now
    // verify with the same two pieces.
    // SOURCE-READ: vllm/v1/sample/rejection_sampler.py:45-65 (sha 924707f1bf94) names the same split: accepted tokens judged
    // on the "raw" draft and target probabilities, recovered tokens from the adjusted distribution, bonus token from the target
    // only. common_spec_accept_rejection already implements that rule. Apache-2.0, design only, no code copied.
    size_t i = 0;
    for (; i < draft.size(); ++i) {
        const llama_token picked = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        const common_spec_dist p = common_sampler_sampling_dist(gsmpl, picked);

        const auto & qd = dists[i];
        GGML_ASSERT(qd.ids.size() == qd.probs.size());
        common_spec_dist q;
        q.ids   = qd.ids;
        q.probs = qd.probs;

        const common_spec_accept_result r = common_spec_accept_rejection(p, q, draft[i], gsmpl->speculative_rng);
        emit(r.token);
        // an accepted EOG ends the run as in common_sampler_sample_and_accept_n (upstream d280808f)
        if (!r.accepted || (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx)), r.token) && i + 1 < draft.size())) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);
        emit(id);
    }

    return result;
}


static void elb_print(common_params_sampling& sparams, const common_params_sampling::elb_param::elb_entry& entry) {
    #undef X
    #define X(T, MEMBER, DV, PRECAST) #MEMBER,
    static const std::vector<std::string> names = { X_COMMON_PARAMS_SAMPLING };
    #undef X
    #define X(T, MEMBER, DV, PRECAST) if (std::abs(entry.addsubs[SPARAMS_ ## MEMBER ## _ENUM]) > 0.0f) \
    { LLAMA_LOG_DEBUG("%s: %s = %f\n", __func__, names[SPARAMS_ ## MEMBER ## _ENUM].c_str(), float(A_DOT_B(sparams, MEMBER))); }
    X_COMMON_PARAMS_SAMPLING
}

static void elb_add(common_params_sampling& sparams, const common_params_sampling::elb_param::elb_entry& entry) {
    #undef X
    #define X(T, MEMBER, _, PRECAST) A_DOT_B(sparams, MEMBER) += static_cast<T>(PRECAST(entry.addsubs[SPARAMS_ ## MEMBER ## _ENUM]));
    X_COMMON_PARAMS_SAMPLING
}

static void elb_sub(common_params_sampling& sparams, const common_params_sampling::elb_param::elb_entry& entry) {
    #undef X
    #define X(T, MEMBER, _, PRECAST) A_DOT_B(sparams, MEMBER) -= static_cast<T>(PRECAST(entry.addsubs[SPARAMS_ ## MEMBER ## _ENUM]));
    X_COMMON_PARAMS_SAMPLING
}

void common_expiring_logit_bias_apply(struct common_sampler* ctx_sampling, float* logits) {
    const auto& elb = ctx_sampling->elb_states[ctx_sampling->elb_idx];

    auto index_first_inactive = [&elb](auto& tokens) {
        return std::distance(
            tokens.begin(),
            std::upper_bound(tokens.begin(), tokens.end(), NULL, [&elb](const auto& _, const auto& token) {
                return elb.countup > token.duration;
            })
        );
    };

    auto& search_window = ctx_sampling->scratch;
    string_assign_append(search_window,
        ctx_sampling->generated_text,
        ctx_sampling->playing_text,
        std::max(0, SSIZE(ctx_sampling->generated_text) + SSIZE(ctx_sampling->playing_text) - elb.max_cond_len));

    if (!search_window.empty() && !elb.other_tokens.empty() && (elb.other_tokens.front().duration > elb.countup)) {
        const auto ifi = index_first_inactive(elb.other_tokens);
        for (size_t j = 0; j < ifi; ++j) {
            const auto& [id, bias, _, cond] = elb.other_tokens[j];
            if (string_ends_with(search_window, cond)) {
                // LLAMA_LOG_DEBUG("%s[%d]: %d\n", __func__, __LINE__, id);
                logits[id] += bias;
            }
        }
    }

    if (!elb.first_tokens.empty() && (elb.first_tokens.front().duration > elb.countup)) {
        const auto ifi = index_first_inactive(elb.first_tokens);
        if (search_window.empty()) {
            // empty case here
            for (size_t j = 0; j < ifi; ++j) {
                // LLAMA_LOG_DEBUG("%s[%d]: %d\n", __func__, __LINE__, elb.first_tokens[j].id);
                logits[elb.first_tokens[j].id] += elb.first_tokens[j].bias;
            }
        } else {
            for (size_t j = 0; j < ifi; ++j) {
                const auto& [id, bias, _, cond] = elb.first_tokens[j];
                // no bias if seen (probably too late)
                if (!string_ends_with(search_window, cond)) {
                    // LLAMA_LOG_DEBUG("%s[%d]: %d\n", __func__, __LINE__, id);
                    logits[id] += bias;
                }
            }
        }
    }

    // expiring sparam bias
    for (auto& entry: ctx_sampling->params.elb_params[ctx_sampling->elb_idx].entries) {
        if (!entry.biases.empty()) {
            continue;   // next entry
        }

        const auto& keywords = entry.phrases;
        for (size_t j = 0; j < keywords.size(); ++j) {
            const auto& phrase = keywords[j];
            if (phrase.empty()) {
                // duration bound only
                if (elb.countup == 0) {
                    LLAMA_LOG_DEBUG("%s: before add\n", __func__);
                    elb_print(ctx_sampling->params, entry);

                    elb_add(ctx_sampling->params, entry);
                    entry.addflags[j] = true;

                    LLAMA_LOG_DEBUG("%s: after add\n", __func__);
                    elb_print(ctx_sampling->params, entry);
                } else if (elb.countup == entry.duration) {
                    LLAMA_LOG_DEBUG("%s: before sub\n", __func__);
                    elb_print(ctx_sampling->params, entry);

                    elb_sub(ctx_sampling->params, entry);
                    entry.addflags[j] = false;

                    LLAMA_LOG_DEBUG("%s: after sub\n", __func__);
                    elb_print(ctx_sampling->params, entry);
                }
                continue;   // next entry
            }

            string_assign_append(search_window, ctx_sampling->generated_text, ctx_sampling->playing_text, entry.search_posi[j]);

            size_t count = 0;
            auto pos = search_window.find(phrase);
            while (pos != std::string::npos) {
                LLAMA_LOG_DEBUG("%s: found %s @ %zu\n", __func__, phrase.c_str(), pos);
                ++count;
                pos = search_window.find(phrase, pos + phrase.length());
            }
            entry.search_posi[j] = std::max(0, SSIZE(ctx_sampling->generated_text) + SSIZE(ctx_sampling->playing_text) - SSIZE(phrase) + 1);
            if (count % 2 == 1) {
                // even = no match or cancelled
                LLAMA_LOG_DEBUG("%s: before\n", __func__);
                elb_print(ctx_sampling->params, entry);

                (entry.addflags[j] ? elb_sub : elb_add)(ctx_sampling->params, entry);
                entry.addflags[j] = !entry.addflags[j];

                LLAMA_LOG_DEBUG("%s: after\n", __func__);
                elb_print(ctx_sampling->params, entry);
            }
        }
    }
}

void common_expiring_logit_bias_accept(struct common_sampler* ctx_sampling, struct llama_context * ctx_main) {
    const auto idx = ctx_sampling->elb_idx;
    if (idx >= ctx_sampling->elb_states.size()) {
        ctx_sampling->elb_idx++;
        return;
    }

    auto& elb = ctx_sampling->elb_states[idx];
    if ((elb.delay > ++(elb.countup)) || (elb.search_word_len == 0)) {
        return;
    }

    auto& search_pos = ctx_sampling->elb_search_pos;
    auto& search_window = ctx_sampling->scratch;
    string_assign_append(search_window, ctx_sampling->generated_text, ctx_sampling->playing_text, search_pos);

    size_t pos = 0;
    if (string_is_found(search_window, elb.jumpword, pos)) {
        LLAMA_LOG_DEBUG("%s: found %s in %s @ %zu\n", __func__, string_unescape(elb.jumpword).c_str(), string_unescape(search_window).c_str(), search_pos + pos);
        ctx_sampling->elb_idx = elb.jump_idx;
        search_pos += pos + elb.jumpword.length();
    } else if (string_is_found(search_window, elb.exitword, pos)) {
        LLAMA_LOG_DEBUG("%s: found %s in %s @ %zu\n", __func__, string_unescape(elb.exitword).c_str(), string_unescape(search_window).c_str(), search_pos + pos);
        ctx_sampling->elb_idx++;
        search_pos += pos + elb.exitword.length();
    } else {
        search_pos += std::max(0, SSIZE(search_window) - elb.search_word_len + 1);
        return;
    }

    // undo current sparam bias
    for (auto& entry: ctx_sampling->params.elb_params[idx].entries) {
        for (const auto addflag: entry.addflags) {
            if (addflag) {
                LLAMA_LOG_DEBUG("%s: before\n", __func__);
                elb_print(ctx_sampling->params, entry);

                elb_sub(ctx_sampling->params, entry);

                LLAMA_LOG_DEBUG("%s: after\n", __func__);
                elb_print(ctx_sampling->params, entry);
            }
        }
    }

    // initialize next stage
    ctx_sampling->elb_states[ctx_sampling->elb_idx].init_pos = search_pos;
    for (auto& entry: ctx_sampling->params.elb_params[ctx_sampling->elb_idx].entries) {
        entry.search_posi.assign(entry.search_posi.size(), search_pos);
    }
}

void common_expiring_logit_bias_rewind(struct common_sampler* ctx_sampling) {
    auto& idx = ctx_sampling->elb_idx;
    auto n_rewind = ctx_sampling->n_rewind;

    // LLAMA_LOG_DEBUG("%s[%d]: idx = %d, n_rewind = %d\n", __func__, __LINE__, idx, n_rewind);

    if (idx >= ctx_sampling->elb_states.size()) {
        // entirely expired
        n_rewind -= 1 + idx - ctx_sampling->elb_states.size();
        idx = ctx_sampling->elb_states.size() - 1;
        if (n_rewind <= 0) {
            // not enough rewind to reanimate
            idx -= n_rewind;
            return;
        }
    }

    // LLAMA_LOG_DEBUG("%s[%d]: idx = %d, n_rewind = %d\n", __func__, __LINE__, idx, n_rewind);

    if (n_rewind > ctx_sampling->elb_states[idx].countup) {
        // leaving current state so cleanup
        for (auto& entry: ctx_sampling->params.elb_params[idx].entries) {
            for (const auto addflag: entry.addflags) {
                if (addflag) {
                    LLAMA_LOG_DEBUG("%s: before\n", __func__);
                    elb_print(ctx_sampling->params, entry);

                    elb_sub(ctx_sampling->params, entry);

                    LLAMA_LOG_DEBUG("%s: after\n", __func__);
                    elb_print(ctx_sampling->params, entry);
                }
            }
        }
    }

    // consume n_rewind
    while (n_rewind > 0) {
        while (ctx_sampling->elb_states[idx].countup == 0) {
            --idx;
        }
        auto& countup = ctx_sampling->elb_states[idx].countup;
        LLAMA_LOG_DEBUG("%s[%d]: idx = %d, countup = %zu, n_rewind = %d\n", __func__, __LINE__, idx, countup, n_rewind);
        if (countup > n_rewind) {
            countup -= n_rewind;
            n_rewind = 0;
        } else {
            n_rewind -= countup;
            countup = 0;
        }
        LLAMA_LOG_DEBUG("%s[%d]: idx = %d, countup = %zu, n_rewind = %d\n", __func__, __LINE__, idx, countup, n_rewind);
    }
    LLAMA_LOG_DEBUG("%s[%d]: idx = %d, n_rewind = %d\n", __func__, __LINE__, idx, n_rewind);

    const int32_t net_text_len = SSIZE(ctx_sampling->generated_text) + SSIZE(ctx_sampling->playing_text) - SSIZE(ctx_sampling->rewinded_text);

    auto& elb = ctx_sampling->elb_states[idx];
    auto& window = ctx_sampling->scratch;
    for (auto& entry: ctx_sampling->params.elb_params[idx].entries) {
        if (!entry.biases.empty()) {
            // no need to rewind logit biases
            continue;
        }
        const auto& keywords = entry.phrases;

        const int32_t min_window_pos = net_text_len - entry.max_keyword_len + 1;
        string_assign_append(window, ctx_sampling->generated_text, ctx_sampling->playing_text, min_window_pos);

        for (size_t j = 0; j < keywords.size(); ++j) {
            const auto& keyword = keywords[j];
            if (keyword.empty() && (elb.countup < entry.duration) && !entry.addflags[j]) {
                // redo duration-based bias
                elb_add(ctx_sampling->params, entry);
                entry.addflags[j] = true;
                continue;
            }

            // keyword match within rewinded text?
            size_t count = 0;
            auto pos = window.find(keyword, entry.max_keyword_len - SSIZE(keyword) + std::max(0, elb.init_pos - min_window_pos));
            while (pos != std::string::npos) {
                ++count;
                pos = window.find(keyword, pos + 1);
            }
            if (count % 2 == 1) {
                (entry.addflags[j] ? elb_sub : elb_add)(ctx_sampling->params, entry);
                entry.addflags[j] = !entry.addflags[j];
            }
        }
    }

    // reinitialize
    ctx_sampling->elb_search_pos = std::max(elb.init_pos, net_text_len - elb.search_word_len + 1);
    for (auto& entry: ctx_sampling->params.elb_params[idx].entries) {
        entry.search_posi.assign(entry.search_posi.size(), elb.init_pos);
    }
}


template <>
json common_grammar_trigger::to_json() const {
    json out{
        {"type", (int)type},
        {"value", value},
    };
    if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
        out["token"] = (int)token;
    }
    return out;
}

template <>
common_grammar_trigger common_grammar_trigger::from_json(const json& in) {
    common_grammar_trigger out;
    out.type = (common_grammar_trigger_type)in.at("type").get<int>();
    out.value = in.at("value").get<std::string>();
    if (out.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
        out.token = (llama_token)in.at("token").get<int>();
    }
    return out;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
static bool common_sampler_speculative_top1_avx2(const float * logits, const int n_vocab, int & best_id, float & max_val) {
    if (n_vocab < 8) {
        return false;
    }

    __m256  max_v = _mm256_loadu_ps(logits);
    __m256i id_v  = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    const __m256i step = _mm256_set1_epi32(8);
    __m256i cur_id = _mm256_add_epi32(id_v, step);

    int i = 8;
    for (; i + 7 < n_vocab; i += 8) {
        const __m256 x = _mm256_loadu_ps(logits + i);
        const __m256 gt_max = _mm256_cmp_ps(x, max_v, _CMP_GT_OQ);

        max_v = _mm256_blendv_ps(max_v, x, gt_max);
        id_v  = _mm256_blendv_epi8(id_v, cur_id, _mm256_castps_si256(gt_max));
        cur_id = _mm256_add_epi32(cur_id, step);
    }

    alignas(32) float max_buf[8];
    alignas(32) int id_buf[8];
    _mm256_store_ps(max_buf, max_v);
    _mm256_store_si256((__m256i *) id_buf, id_v);

    best_id = id_buf[0];
    max_val = max_buf[0];

    for (int j = 1; j < 8; ++j) {
        if (max_buf[j] > max_val) {
            max_val = max_buf[j]; best_id = id_buf[j];
        }
    }
    for (; i < n_vocab; ++i) {

        if (logits[i] > max_val) {
            max_val = logits[i]; best_id = i;
        }
    }
    return true;
}
__attribute__((target("avx2,fma")))
static inline __m256 v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}
__attribute__((target("avx2")))
static inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
__attribute__((target("avx2")))
static inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
__attribute__((target("avx2,fma")))
static float prob_avx2(int n, const float * logits, float max_val) {
    float sumf = 0;
    int i = 0;
    if (n >= 8) {
        auto sum_v = _mm256_setzero_ps();
        auto max_v = _mm256_set1_ps(max_val);
        for (; i < n - 7; i += 8) {
            auto x = _mm256_loadu_ps(logits + i);
            auto exp_x = v_expf(_mm256_sub_ps(x, max_v));
            sum_v = _mm256_add_ps(sum_v, exp_x);
        }
        sumf = hsum_float_8(sum_v);
    }
    for (; i < n; ++i) {
        sumf += expf(logits[i] - max_val);
    }
    return 1.0f/sumf;
}
#endif
static float prob_scalar(int n, const float * logits, float max_val) {
    double sum_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_exp += exp((double)(logits[i] - max_val));
    }
    return (float)(1./sum_exp);
}

llama_token common_sampler_sample_speculative(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, float * out_prob) {
    GGML_UNUSED(gsmpl);

    float * logits = llama_get_logits_ith(ctx, idx);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));

    int best_id = 0;
    float max_val = logits[0];
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    static const bool has_avx2 = __builtin_cpu_supports("avx2");
    if (has_avx2 && common_sampler_speculative_top1_avx2(logits, n_vocab, best_id, max_val)) {
        if (out_prob) {
            static const bool has_fma = __builtin_cpu_supports("fma");
            if (has_fma) {
                *out_prob = prob_avx2(n_vocab, logits, max_val);
            } else {
                *out_prob = prob_scalar(n_vocab, logits, max_val);
            }
        }
        return best_id;
    }
#endif
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best_id = i;
        }
    }

    if (out_prob) {
        *out_prob = prob_scalar(n_vocab, logits, max_val);
    }

    return best_id;
}


// ---------------------------------------------------------------------------------------------
// Model-embedded sampling parameters (general.sampling.*), upstream PR #17120.
//
// The flag names below were MEASURED from common.cpp by walking back from each
// `sparams.<member> =` assignment to its enclosing `if (arg == ...)`, not derived from the member
// name. Four of them do not follow the obvious pattern -- mirostat_eta is --mirostat-lr,
// mirostat_tau is --mirostat-ent, tfs_z is --tfs, typical_p is --typical -- and a derived mapping
// would have let the model override a flag the user explicitly passed.
// ---------------------------------------------------------------------------------------------

struct sampling_kv_flag { const char * member; const char * flag; };

static const sampling_kv_flag SAMPLING_KV_FLAGS[] = {
    { "top_k",               "--top-k"              },
    { "top_p",               "--top-p"              },
    { "min_p",               "--min-p"              },
    { "tfs_z",               "--tfs"                },
    { "typical_p",           "--typical"            },
    { "temp",                "--temp"               },
    { "dynatemp_range",      "--dynatemp-range"     },
    { "dynatemp_exponent",   "--dynatemp-exp"       },
    { "penalty_last_n",      "--repeat-last-n"      },
    { "penalty_repeat",      "--repeat-penalty"     },
    { "penalty_freq",        "--frequency-penalty"  },
    { "penalty_present",     "--presence-penalty"   },
    { "dry_multiplier",      "--dry-multiplier"     },
    { "dry_base",            "--dry-base"           },
    { "dry_allowed_length",  "--dry-allowed-length" },
    { "dry_penalty_last_n",  "--dry-penalty-last-n" },
    { "mirostat",            "--mirostat"           },
    { "mirostat_tau",        "--mirostat-ent"       },
    { "mirostat_eta",        "--mirostat-lr"        },
    { "xtc_probability",     "--xtc-probability"    },
    { "xtc_threshold",       "--xtc-threshold"      },
    { "top_n_sigma",         "--top-n-sigma"        },
    { "adaptive_target",     "--adaptive-target"    },
    { "adaptive_decay",      "--adaptive-decay"     },
    { "top_a",               "--top-a"              },
    { "smoothing_factor",    "--smoothing-factor"   },
    { "smoothing_curve",     "--smoothing-curve"    },
    { "penalty_slope",       "--repeat-penalty-slope" },
    { "penalty_decay",       "--repeat-decay" },
    { "skew",                "--skew"         },
    { "epsilon_cutoff",      "--epsilon-cutoff"     },
    { "eta_cutoff",          "--eta-cutoff"         },
    { "no_repeat_ngram_size", "--no-repeat-ngram-size" },
    { "no_repeat_ngram_last_n", "--no-repeat-ngram-last-n" },
};

static const char * sampling_kv_flag_for(const char * member) {
    for (const auto & e : SAMPLING_KV_FLAGS) {
        if (std::strcmp(e.member, member) == 0) {
            return e.flag;
        }
    }
    return nullptr;   // no CLI flag exists for this member; the model may always set it
}

// Read one general.sampling.<name> key. Returns false when the key is absent.
static bool sampling_kv_get(const struct llama_model * model, const char * name, std::string & out) {
    char key[128];
    snprintf(key, sizeof(key), "general.sampling.%s", name);
    char buf[512];
    const int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n < 0) {
        return false;
    }
    out.assign(buf, (size_t) std::min<int32_t>(n, (int32_t) sizeof(buf) - 1));
    return true;
}

// ---------------------------------------------------------------------------------------------
// Model-family sampling recommendations, the fallback for a GGUF with no general.sampling.* key
// (--sampling-family-defaults; table, matcher and the SOURCE-NUANCE notes: sampling-family-defaults.h).
// SOURCE-READ: upstream/LocalAI/core/config/inference_defaults.go:1-128 and inference_defaults.json (MIT).
// SOURCE-NUANCE: LocalAI treats a penalty of 0 as "unset" and overwrites an explicit 0; here args_seen decides.
// ---------------------------------------------------------------------------------------------
int common_params_sampling_apply_family_defaults(const struct llama_model * model,
                                                 common_params_sampling & sparams,
                                                 const std::set<std::string> & args_seen,
                                                 const std::vector<std::string> & model_ids) {
    if (model == nullptr) {
        return 0;
    }

    // the model's own recommendation, when it has one, is the better source: use the family table only when NO key exists
    bool has_model_kv = false;
    {
        std::string raw;
        #undef X
        #define X(T, MEMBER, DV, PRECAST) if (!has_model_kv && sampling_kv_get(model, #MEMBER, raw)) { has_model_kv = true; }
        X_COMMON_PARAMS_SAMPLING
        #undef X
        if (!has_model_kv && sampling_kv_get(model, "sequence", raw)) {
            has_model_kv = true;
        }
    }
    if (has_model_kv) {
        LLAMA_LOG_INFO("sampling: family defaults not used, the model carries its own general.sampling.* values\n");
        return 0;
    }

    std::string which;
    const common_family_sampling * fam = common_family_sampling_match_any(model_ids, &which);
    if (fam == nullptr) {
        LLAMA_LOG_INFO("sampling: no model family matched the %zu candidate name(s); built-in defaults stand\n", model_ids.size());
        return 0;
    }

    int applied = 0;
    auto take = [&](const char * member, auto & field, auto value) {
        const char * flag = sampling_kv_flag_for(member);
        if (flag != nullptr && args_seen.count(flag) > 0) {
            LLAMA_LOG_INFO("sampling: %-18s kept from %s (family '%s' recommends %g)\n", member, flag, fam->pattern, (double) value);
            return;
        }
        field = value;
        applied++;
        LLAMA_LOG_INFO("sampling: %-18s = %g (family '%s', matched on '%s')\n", member, (double) value, fam->pattern, which.c_str());
    };
    take("temp",           sparams.temp,           fam->temp);
    take("top_p",          sparams.top_p,          fam->top_p);
    take("top_k",          sparams.top_k,          fam->top_k);
    take("min_p",          sparams.min_p,          fam->min_p);
    take("penalty_repeat", sparams.penalty_repeat, fam->penalty_repeat);
    if (fam->penalty_present >= 0.0f) {
        take("penalty_present", sparams.penalty_present, fam->penalty_present);
    }
    LLAMA_LOG_INFO("sampling: %d parameter(s) taken from the '%s' family table (--sampling-family-defaults); pass the matching flag to override any of them\n",
                   applied, fam->pattern);
    return applied;
}

int common_params_sampling_apply_model_kv(const struct llama_model * model,
                                          common_params_sampling & sparams,
                                          const std::set<std::string> & args_seen) {
    if (model == nullptr) {
        return 0;
    }

    int applied = 0;

    // The scalar parameters, driven off the same X-macro that declares them, so a parameter added
    // to the struct is covered here automatically instead of being quietly left out.
    #undef X
    #define X(T, MEMBER, DV, PRECAST)                                                              \
    {                                                                                              \
        const char * flag = sampling_kv_flag_for(#MEMBER);                                         \
        const bool user_set = (flag != nullptr) && (args_seen.count(flag) > 0);                    \
        std::string raw;                                                                           \
        if (sampling_kv_get(model, #MEMBER, raw)) {                                                \
            if (user_set) {                                                                        \
                LLAMA_LOG_INFO("sampling: %-18s kept from %s (model recommends %s)\n",             \
                               #MEMBER, flag, raw.c_str());                                        \
            } else {                                                                               \
                try {                                                                              \
                    const double v = std::stod(raw);                                               \
                    sparams.MEMBER = (T) PRECAST(v);                                               \
                    applied++;                                                                     \
                    LLAMA_LOG_INFO("sampling: %-18s = %s (from model KV general.sampling.%s)\n",   \
                                   #MEMBER, raw.c_str(), #MEMBER);                                 \
                } catch (const std::exception & e) {                                               \
                    /* A malformed value is REPORTED, never silently ignored: silence here would   \
                       look identical to the key being absent. */                                  \
                    LLAMA_LOG_WARN("sampling: general.sampling.%s = %s is not a number (%s); "     \
                                   "keeping the default\n", #MEMBER, raw.c_str(), e.what());       \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
    }
    X_COMMON_PARAMS_SAMPLING
    #undef X

    // general.sampling.sequence -- implemented, not stubbed. It routes through the same parser
    // --samplers uses, so there is no second code path to keep in step.
    {
        std::string raw;
        const bool user_set = args_seen.count("--samplers") > 0 || args_seen.count("--sampling-seq") > 0;
        if (sampling_kv_get(model, "sequence", raw)) {
            if (user_set) {
                LLAMA_LOG_INFO("sampling: %-18s kept from --samplers/--sampling-seq "
                               "(model recommends %s)\n", "sequence", raw.c_str());
            } else {
                // string_split<std::string> EXPLICITLY, 2026-09-13. This read `string_split(raw,
                // ';')`, which resolved to a non-template overload declared at common/common.h:1168
                // and defined NOWHERE -- 24 undefined references at link, one per target that pulls
                // in this object. The template specialization at common.h (inline, T=std::string)
                // is the definition that exists and does exactly this: split on the char, keeping
                // empty fields. The dangling declaration is removed in the same commit.
                const auto names = string_split<std::string>(raw, ';');
                const auto seq   = llama_sampling_types_from_names(names, true);
                if (seq.empty()) {
                    LLAMA_LOG_WARN("sampling: general.sampling.sequence = %s named no known "
                                   "sampler; keeping the default sequence\n", raw.c_str());
                } else {
                    sparams.samplers_sequence = seq;
                    applied++;
                    LLAMA_LOG_INFO("sampling: %-18s = %s (from model KV general.sampling.sequence)\n",
                                   "sequence", raw.c_str());
                }
            }
        }
    }

    if (applied > 0) {
        LLAMA_LOG_INFO("sampling: %d parameter(s) taken from the model's own recommendations "
                       "(general.sampling.*); pass the matching flag to override any of them\n",
                       applied);
    }
    return applied;
}


// =================================================================================================
// PORTED FROM theirs' (unsloth/master) chain-based design -- DECISION #5
// SOURCE-READ: unsloth/master:common/sampling.cpp (a branch of this repository; llama.cpp code, MIT)
// (S69-DECISION-5-MADE-OURS-STRUCT-PRIMARY-2026-09-07-01). ours' struct common_sampler (sampling.h)
// is primary and has NO llama_sampler chain / grmr / t_total_us members, so every body below is
// ADAPTED to ours' actual fields, not copied verbatim, except the 4 pure-enum utilities that never
// touch struct common_sampler at all.
// =================================================================================================

// --- verbatim (no struct common_sampler dependency) ---

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

// --- adapted (rewritten against ours' actual struct common_sampler fields) ---

// matches struct field EXACTLY (common_reasoning_budget_ctx * rbudget on both this struct and
// theirs' -- no rewrite needed beyond the body itself, confirmed by field-name+type match).
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// theirs' common_sampler_copy already matched ours' common_sampler_clone(src,dst) IN-PLACE
// semantics (see decision row: theirs' *own* common_sampler_clone allocates+returns instead --
// NOT ported, excluded below, name clash with ours' void clone). Forwarding avoids duplicating
// ours' clone body; const_cast is safe because common_sampler_clone only reads from src.
void common_sampler_copy(const struct common_sampler * src, struct common_sampler * dst) {
    if (!src || !dst) {
        return;
    }
    common_sampler_clone(const_cast<common_sampler *>(src), dst);
}

// ESCALATED (ours' struct has no resolved-seed storage): ours' common_sampler_init only ever
// calls llama_sampling_set_rng_seed(result, params.seed), which seeds ->rng but never writes the
// RESOLVED seed back anywhere -- there is no analogue of theirs' chain-internal dist-sampler seed
// to read back via llama_sampler_get_seed(). params.seed is the best available answer: it is
// wrong (returns the LLAMA_DEFAULT_SEED sentinel) exactly when the caller asked for a random seed,
// which is precisely the case a caller most wants the real resolved value for. Not fixable without
// a new struct field (sampling.h forbidden this pass) -- named, not silently guessed past.
uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return gsmpl ? gsmpl->params.seed : LLAMA_DEFAULT_SEED;
}

// ESCALATED -- MEASURED, not a guess: src/llama-context.cpp:1209 (llama_context::set_sampler)
// requires llama_sampler_chain_n(sampler) > 0 and sampler->iface->backend_init/backend_apply, i.e.
// a REAL backend-offloadable llama_sampler chain, not a "transparent struct, thin passthrough" as
// this function's own header comment (sampling.h:442-444) speculated. That speculation is REFUTED
// by the real caller (tools/server/server-context.cpp:1706, non-UU, already compiles). ours'
// struct has no persistent chain to hand back, so this builds ONE FRESH per call from gsmpl->params,
// mirroring theirs' original common_sampler_init chain-assembly order. COST NAMED: (1) built fresh
// every call -- wasteful if called per-token, acceptable if called only on slot/sampler setup as
// server-context.cpp:1706's call site suggests, NOT verified without a build; (2) ctx->set_sampler
// stores the raw pointer with no visible clone -- lifetime/free-on-replace ownership is the src/
// layer's contract, outside this file; a persistent cached chain would need a new struct field
// (sampling.h forbidden this pass). Flagged for the next pass with a build available.
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    lparams.no_perf = gsmpl->params.no_perf;

    llama_sampler * chain = llama_sampler_chain_init(lparams);

    const auto & params = gsmpl->params; // common_sampler_init already folded upstream's `samplers` into samplers_sequence
    for (const auto & cnstr : params.samplers_sequence) {
        switch (cnstr) {
            case llama_sampler_type::TOP_K:       llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k)); break;
            case llama_sampler_type::TOP_P:       llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1)); break;
            case llama_sampler_type::MIN_P:       llama_sampler_chain_add(chain, llama_sampler_init_min_p(params.min_p, 1)); break;
            case llama_sampler_type::TYPICAL_P:   llama_sampler_chain_add(chain, llama_sampler_init_typical(params.typical_p, 1)); break;
            case llama_sampler_type::TOP_N_SIGMA: llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(params.top_n_sigma)); break;
            case llama_sampler_type::XTC:         llama_sampler_chain_add(chain, llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, 1, params.seed)); break;
            case llama_sampler_type::TEMPERATURE: llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent)); break;
            default: break; // DRY/TFS_Z/ADAPTIVE_P/TOP_A/DIST have no backend-offloadable equivalent here
        }
    }
    // upstream: a chain that ends in temp 0 or top-k 1 is deterministic, so draw greedily -- unless the caller asked for
    // probabilities, which only the distribution sampler fills in (merge 2026-11-04)
    const bool greedy = params.n_probs == 0 && !params.samplers_sequence.empty() &&
        ((params.samplers_sequence.back() == llama_sampler_type::TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
         (params.samplers_sequence.back() == llama_sampler_type::TOP_K && params.top_k == 1));
    llama_sampler_chain_add(chain, greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));

    return chain;
}

bool common_sampler_samplers_unset(const struct common_params_sampling & params) {
    return params.samplers.size() == 1 && params.samplers[0] == COMMON_SAMPLER_TYPE_NONE;
}

// common_sampler_type -> llama_sampler_type. false for a type this fork has no step for; PENALTIES / NONE map to nothing (penalties run
// before the queue here, "CFG -> Penalties"), which `penalties_late` reports when PENALTIES came after another sampler.
static bool common_sampler_type_to_llama(common_sampler_type t, llama_sampler_type & out, bool & skip) {
    skip = false;
    switch (t) {
        case COMMON_SAMPLER_TYPE_DRY:         out = llama_sampler_type::DRY;         return true;
        case COMMON_SAMPLER_TYPE_TOP_K:       out = llama_sampler_type::TOP_K;       return true;
        case COMMON_SAMPLER_TYPE_TOP_P:       out = llama_sampler_type::TOP_P;       return true;
        case COMMON_SAMPLER_TYPE_MIN_P:       out = llama_sampler_type::MIN_P;       return true;
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   out = llama_sampler_type::TYPICAL_P;   return true;
        case COMMON_SAMPLER_TYPE_TEMPERATURE: out = llama_sampler_type::TEMPERATURE; return true;
        case COMMON_SAMPLER_TYPE_XTC:         out = llama_sampler_type::XTC;         return true;
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: out = llama_sampler_type::TOP_N_SIGMA; return true;
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  out = llama_sampler_type::ADAPTIVE_P;  return true;
        case COMMON_SAMPLER_TYPE_INFILL:      out = llama_sampler_type::INFILL;      return true;
        case COMMON_SAMPLER_TYPE_PENALTIES:
        case COMMON_SAMPLER_TYPE_NONE:        skip = true;                           return true;
    }
    return false;
}

std::vector<llama_sampler_type> common_sampler_resolved_sequence(const struct common_params_sampling & params) {
    if (common_sampler_samplers_unset(params)) {
        return params.samplers_sequence;
    }
    std::vector<llama_sampler_type> out;
    out.reserve(params.samplers.size());
    for (const auto t : params.samplers) {
        llama_sampler_type lt;
        bool skip = false;
        if (common_sampler_type_to_llama(t, lt, skip) && !skip) {
            out.push_back(lt);
        }
    }
    return out;
}

bool common_sampler_resolve_samplers(struct common_params_sampling & params) {
    if (common_sampler_samplers_unset(params)) {
        return true;
    }
    std::vector<llama_sampler_type> out;
    out.reserve(params.samplers.size());
    bool seen_other = false;
    for (const auto t : params.samplers) {
        llama_sampler_type lt;
        bool skip = false;
        if (!common_sampler_type_to_llama(t, lt, skip)) {
            LOG_ERR("%s: sampler type %d has no step in this build\n", __func__, (int) t);
            return false;
        }
        if (t == COMMON_SAMPLER_TYPE_PENALTIES && seen_other) {
            LOG_WRN("%s: 'penalties' listed after another sampler; this build applies penalties before the sampler queue\n", __func__);
        }
        if (!skip) {
            out.push_back(lt);
            seen_other = true;
        }
    }
    params.samplers_sequence = std::move(out);
    params.samplers = { COMMON_SAMPLER_TYPE_NONE };
    return true;
}

bool common_sampler_needs_host(const struct common_params_sampling & params) {
    const std::vector<llama_sampler_type> seq = common_sampler_resolved_sequence(params);
    auto in_seq = [&](llama_sampler_type t) {
        return std::find(seq.begin(), seq.end(), t) != seq.end();
    };
    return in_seq(llama_sampler_type::INFILL) || // the backend chain has no vocab handle for the infill sampler
           (in_seq(llama_sampler_type::DRY)        && params.dry_multiplier != 0.0f)  ||
           (in_seq(llama_sampler_type::TFS_Z)      && params.tfs_z < 1.0f)            ||
           (in_seq(llama_sampler_type::ADAPTIVE_P) && params.adaptive_target >= 0.0f) ||
           (in_seq(llama_sampler_type::TOP_A)      && params.top_a > 0.0f)            ||
           (in_seq(llama_sampler_type::EPSILON_CUTOFF) && params.epsilon_cutoff > 0.0f) ||
           (in_seq(llama_sampler_type::ETA_CUTOFF) && params.eta_cutoff > 0.0f)       ||
           // the backend chain's XTC has no vocab handle, so it cannot exempt end-of-generation / newline tokens the way the host
           // llama_sample_xtc now does (SOURCE-READ: ExLlamaV2 generator/sampler.py:277-302, MIT, behaviour only)
           (in_seq(llama_sampler_type::XTC)        && params.xtc_probability > 0.0f && params.xtc_threshold <= 0.5f) ||
           params.no_repeat_ngram_size > 0                                            ||
           !params.suppress_sequences.empty()                                         ||
           // expiring logit bias adds to the host logits and reads the generated text (common_expiring_logit_bias_apply)
           !params.elb_params.empty()                                                 ||
           (in_seq(llama_sampler_type::TEMPERATURE) && params.smoothing_factor > 0.0f) ||
           // the backend chain's penalty sampler is uniform over the window; the slope exists only in
           // llama_sampling_prepare_impl (measured: slope 0 moved 0 of 52 candidates until this line)
           (params.penalty_slope > 0.0f && params.penalty_slope < 1.0f && params.penalty_repeat > 1.0f) ||
           (params.penalty_decay > 0 && params.penalty_last_n != 0 &&
            (params.penalty_repeat != 1.0f || params.penalty_freq != 0.0f || params.penalty_present != 0.0f)) ||
           // skew reshapes the host draw (common/sampling-skew.h); the backend chain's dist sampler has no such step
           params.skew != 0.0f;
}

bool common_sampling_truncates(const struct common_params_sampling & params, int32_t n_vocab) {
    if (n_vocab <= 0) { n_vocab = params.n_vocab_hint; }
    for (const auto t : common_sampler_resolved_sequence(params)) {
        switch (t) {
            // each test is the value at which the sampler function itself returns without dropping anything
            case llama_sampler_type::TOP_K:          if (params.top_k > 0 && (n_vocab <= 0 || params.top_k < n_vocab)) return true; break;
            case llama_sampler_type::TOP_P:          if (params.top_p < 1.0f)          return true; break;
            case llama_sampler_type::MIN_P:          if (params.min_p > 0.0f)          return true; break;
            case llama_sampler_type::TFS_Z:          if (params.tfs_z < 1.0f)          return true; break;
            case llama_sampler_type::TYPICAL_P:      if (params.typical_p < 1.0f)      return true; break;
            case llama_sampler_type::TOP_N_SIGMA:    if (params.top_n_sigma > 0.0f)    return true; break;
            case llama_sampler_type::TOP_A:          if (params.top_a > 0.0f)          return true; break;
            case llama_sampler_type::EPSILON_CUTOFF: if (params.epsilon_cutoff > 0.0f) return true; break;
            case llama_sampler_type::ETA_CUTOFF:     if (params.eta_cutoff > 0.0f)     return true; break;
            default: break; // XTC removes top tokens, not the tail; DRY/TEMPERATURE/DIST do not truncate
        }
    }
    return false;
}

bool common_sampling_adaptive_p_untruncated(const struct common_params_sampling & params, int32_t n_vocab) {
    const std::vector<llama_sampler_type> seq = common_sampler_resolved_sequence(params);
    const bool in_seq = std::find(seq.begin(), seq.end(), llama_sampler_type::ADAPTIVE_P) != seq.end();
    // same gate as the adaptive branch of llama_sampling_sample_impl: temp > 0, no mirostat, target >= 0
    return in_seq && params.adaptive_target >= 0.0f && params.temp > 0.0f && params.mirostat == 0 &&
           !common_sampling_truncates(params, n_vocab);
}

int32_t common_adaptive_untruncated_from_name(const char * name) {
    if (!name) return -1;
    if (!strcmp(name, "floor"))  return COMMON_ADAPTIVE_UNTRUNCATED_FLOOR;
    if (!strcmp(name, "refuse")) return COMMON_ADAPTIVE_UNTRUNCATED_REFUSE;
    if (!strcmp(name, "allow"))  return COMMON_ADAPTIVE_UNTRUNCATED_ALLOW;
    return -1;
}

const char * common_adaptive_untruncated_name(int32_t mode) {
    switch (mode) {
        case COMMON_ADAPTIVE_UNTRUNCATED_FLOOR:  return "floor";
        case COMMON_ADAPTIVE_UNTRUNCATED_REFUSE: return "refuse";
        case COMMON_ADAPTIVE_UNTRUNCATED_ALLOW:  return "allow";
        default: return nullptr;
    }
}

std::string common_sampling_adaptive_p_refusal(const struct common_params_sampling & params, int32_t n_vocab) {
    if (params.adaptive_untruncated != COMMON_ADAPTIVE_UNTRUNCATED_REFUSE || !common_sampling_adaptive_p_untruncated(params, n_vocab)) {
        return std::string();
    }
    return "adaptive_target needs a truncating sampler in front of it (top_k > 0, top_p < 1, min_p > 0, ...). "
           "Alone it draws from the whole vocabulary and mostly picks tail tokens; set one of them, or use "
           "adaptive_untruncated \"floor\" (min_p floor applied for you) or \"allow\"";
}

// ESCALATED (partial loss, named not guessed): ours' struct common_sampler carries no per-instance
// timing accumulator (no analogue of theirs' t_total_us / chain-perf) -- porting the sampling-time
// half of theirs' body verbatim would reference members that do not exist on ours' struct. The
// ctx-side half (load/prompt-eval/eval time, memory breakdown) needs no chain and is ported whole;
// sampling time is reported as 0 rather than fabricated. Adding a real accumulator is a struct
// change (sampling.h forbidden this pass).
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    const double t_sampling_ms = 0.0; // see note above

    // opt-in sampler features report themselves, so a run can show that they fired (and, for the LOSSY relaxed accept, how often)
    if (gsmpl != nullptr && (gsmpl->n_prefiltered > 0 || gsmpl->n_spec_relaxed > 0)) {
        LOG_INF("%s: sampler counters: %llu positions via the top-k prefilter, %llu draft tokens kept only by relaxed accept\n", __func__,
                (unsigned long long) gsmpl->n_prefiltered, (unsigned long long) gsmpl->n_spec_relaxed);
    }

    llama_perf_context_data data_ctx;
    memset(&data_ctx, 0, sizeof(data_ctx));

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        common_memory_breakdown_print(ctx);
    }
}

// --- NOT in the decision's named 5, but PROVEN required by real (non-UU, already-staged) callers
//     -- tools/completion/completion.cpp:454,736,760,778 and tools/server/server-context.cpp:1711
//     call common_sampler_print/_prev_str/_last under theirs' naming. sampling.h does not yet
//     declare these three (ESCALATED separately -- a 3-line header addition, out of scope here).
//     Defined here so the header fix is a pure addition with a body already waiting. Delegate to
//     ours' pre-existing equivalents rather than reimplement chain iteration that no longer exists.

// Allocating clone, for callers that want a new sampler rather than copying into one they own
// (tools/server/server-context.cpp:3795 wraps the result in a common_sampler_ptr).
//
// `new common_sampler()` -- the PARENTHESES ARE LOAD-BEARING. common_sampler carries raw pointers
// (grammar, smpl, adapt_p_ctx, rbudget, server_biases) with no default member initialisers, and
// the two-argument clone below FREES dst's existing pointers before overwriting them. Without
// value-initialisation those pointers would be indeterminate and the clone would free garbage.
// The scratch members it does not copy (cur, cur_p) are rebuilt per call by
// llama_sampling_prepare_impl, so leaving them empty is correct rather than merely harmless.
struct common_sampler * common_sampler_clone(const struct common_sampler * src) {
    if (src == nullptr) {
        return nullptr;
    }

    auto * dst = new common_sampler();
    common_sampler_clone(const_cast<common_sampler *>(src), dst);

    return dst;
}

// --spec-draft-grammar (row S69-LMDEPLOY-PICK-GRAMMAR-MASKED-SPEC-DRAFT-2026-10-18-01). LMDeploy v0.17.0 (Apache-2.0)
// forks the request's grammar matcher for the draft steps and rolls it back afterwards
// (lmdeploy/pytorch/spec_decode/guided_spec_helper.py:69-98, :124-167, :221-231); read, no code copied. Here the draft
// sampler gets a fresh copy of the target's constraint before every draft round, so nothing needs rolling back: the
// target's own state is never advanced by a draft token. The copy uses the same calls as the two-argument clone above.
bool common_sampler_copy_constraint(struct common_sampler * dst, const struct common_sampler * src,
                                    const struct llama_model * model_dst) {
    if (dst == nullptr) {
        return false;
    }
    if (dst->grammar) {
        llama_grammar_free(dst->grammar);
        dst->grammar = nullptr;
    }
    if (dst->cnstr) {
        llama_sampler_free(dst->cnstr);
        dst->cnstr = nullptr;
    }
    // a lazy grammar inside a reasoning budget is not applying on the target, so the draft must not apply it either
    if (src == nullptr || !grammar_should_apply(const_cast<common_sampler *>(src))) {
        return false;
    }
    if (src->grammar) {
        // MEASURED 2026-10-18: a plain llama_grammar_copy accepted through the draft context aborts with
        // GGML_ASSERT(vocab == grammar.vocab) (src/llama-grammar.cpp llama_grammar_accept_impl)
        dst->grammar = model_dst ? llama_grammar_copy_for_model(src->grammar, model_dst) : llama_grammar_copy(src->grammar);
    } else if (src->cnstr && model_dst == nullptr) {
        dst->cnstr = llama_sampler_clone(src->cnstr);
    }
    return constraint_present(dst);
}

// --spec-draft-validate (row S69-VLLM-W2-GRAMMAR-VALIDATED-DRAFTS-2026-10-27-01). vLLM's StructuredOutputManager.validate_tokens
// (vllm 924707f1bf94, vllm/v1/structured_output/__init__.py:294-312; backend_guidance.py:185-200 for the matcher side) returns the longest
// prefix of a draft the grammar accepts WITHOUT advancing the grammar, and the scheduler stores that prefix instead of the draft
// (scheduler.py:2487, :2515), so the verify batch never carries tokens the grammar will refuse. Apache-2.0, read, no code copied.
// Here the walk runs on a private copy of the sampler's GBNF grammar: each draft token is applied as a one-element candidate list (the
// grammar's own would-accept test, EOG included: allowed only when the grammar can end) and, if it survives, accepted on the copy so the
// next token is judged in the next grammar state. The caller's sampler is never advanced.
//   NUANCES THE DESCRIPTION DID NOT SAY: (1) a lazy grammar that has not triggered yet constrains nothing (apply returns early) and its
//   accept only feeds the trigger buffer, so the walk follows the trigger exactly as the target will; (2) nothing is kept after an EOG
//   (the server stops at it; the target would stop); (3) when the reasoning budget says the grammar is not applying now, the draft may
//   cross into the constrained part and we cannot say where, so it is returned whole (vLLM does a reasoning-end search here, :220-292);
//   (4) the llguidance sampler-shaped constraint has no cheap copy+probe on this path and is NOT walked (whole draft returned) -- untested
//   here, so deliberately conservative; (5) an exception from the grammar ends the valid prefix instead of reaching the server loop.
size_t common_sampler_draft_valid_prefix(const struct common_sampler * gsmpl, struct llama_context * ctx,
                                         const std::vector<llama_token> & draft) {
    if (gsmpl == nullptr || ctx == nullptr || draft.empty() || gsmpl->grammar == nullptr ||
        !grammar_should_apply(const_cast<common_sampler *>(gsmpl))) {
        return draft.size();
    }
    llama_grammar * walk = llama_grammar_copy(gsmpl->grammar);
    if (walk == nullptr) {
        return draft.size();
    }
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    size_t n_ok = 0;
    try {
        for (; n_ok < draft.size(); ++n_ok) {
            const llama_token t = draft[n_ok];
            llama_token_data       one_tok = { t, 0.0f, 0.0f };
            llama_token_data_array one     = { &one_tok, 1, false };
            llama_grammar_apply(walk, ctx, &one);
            if (!(one_tok.logit > -INFINITY)) {
                break; // the grammar refuses this token in this state
            }
            llama_grammar_accept_token(walk, ctx, t);
            if (llama_vocab_is_eog(vocab, t)) {
                ++n_ok;
                break;
            }
        }
    } catch (const std::exception & e) {
        LOG_WRN("%s: grammar walk stopped at draft position %zu: %s\n", __func__, n_ok, e.what());
    }
    llama_grammar_free(walk);
    return n_ok;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.back();
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    return llama_sampling_order_print(gsmpl->params);
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    return llama_sampling_prev_str(gsmpl, ctx_main, n);
}

// Mask values are -inf by convention; only NaN and +inf mean the model itself produced garbage.
int32_t common_count_nonfinite_logits(const float * logits, int32_t n) {
    int32_t bad = 0;
    for (int32_t i = 0; i < n; ++i) {
        const float v = logits[i];
        if (std::isnan(v) || v == INFINITY) {
            ++bad;
        }
    }
    return bad;
}

void common_sampler_set_restore_if_empty(struct common_sampler * ctx, std::vector<std::pair<llama_token, float>> restore) {
    if (ctx) {
        ctx->restore_if_empty = std::move(restore);
    }
}

int32_t common_sampler_n_nonfinite_last(const struct common_sampler * ctx) {
    return ctx ? ctx->n_nonfinite_last : 0;
}

std::string common_sampler_forced_string(struct common_sampler * ctx, size_t max_bytes) {
    if (!ctx || !ctx->grammar || ctx->cnstr || ctx->rbudget || ctx->params.grammar_lazy ||
            ctx->params.grammar.type != COMMON_GRAMMAR_TYPE_USER) {
        return std::string();
    }
    return llama_grammar_forced_string(*ctx->grammar, max_bytes);
}
void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
