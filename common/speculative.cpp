#include "speculative.h"
#include "spec-mtp-state.h"
#include "spec-mtp-embd.h"

#include "common.h"
// ADDED 2026-09-13: common_params_parse is declared in arg.h:125, and this file calls it when a
// draft model carries its own flag string (the "llama-server " + params.params splice further
// down). arg.h includes common.h, which is already included here, so this adds the declaration
// without introducing a new dependency direction.
#include "arg.h"
#include "ggml.h"
#include "llama.h"
#include "llama-spec-features-dflash.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"
#include "suffix-automaton.h"
#include "draft-policy.h"
#include "eagle3-d2t.h"
#include "ggml-cpp.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (unsloth/master MTP path; NOT currently called by any code kept in this merge, included for forward-compat since llama-ext.h is a header-only declaration file with no link cost)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
// ADDED 2026-09-13: speculative.cpp uses assert() in common_speculative_impl_ngram_map_k
// ::draft() and nothing here pulled in <cassert>. Kept as assert() rather than promoted to
// GGML_ASSERT so the check keeps exactly the semantics its author wrote -- compiled out
// under NDEBUG -- instead of quietly becoming a release-build abort.
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>


// unsloth/master's per-op-prefixed logging macros, used by the per-sequence-pool code
// kept below (speculative.h's merge-shape note).
#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

void llama_set_mtp_target_context(struct llama_context * ctx, struct llama_context * target_ctx);
void llama_set_mtp_step_idx(struct llama_context * ctx, int32_t mtp_step_idx);
void llama_set_mtp_n_heads(struct llama_context * ctx, int32_t mtp_n_heads);

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_DFLASH,
    COMMON_SPECULATIVE_TYPE_DSPARK,
    COMMON_SPECULATIVE_TYPE_MTP,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_SUFFIX,
    COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH},
    {"dspark",        COMMON_SPECULATIVE_TYPE_DSPARK},
    {"mtp",           COMMON_SPECULATIVE_TYPE_MTP},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX},
    {"suffix_automaton", COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON},
    // Upstream's current spellings (llama.cpp b9590 common/speculative.cpp:25-33: draft-simple,
    // draft-eagle3, draft-mtp, and the draft-dflash/draft-dspark its model-resolution tests use).
    // Both lookups turn '-' into '_' first, so the ngram-* names already matched; these did not,
    // and `--spec-type draft-mtp` was rejected as unknown. Lookup-only aliases: this map is never
    // iterated, and type_to_str still prints the fork's own names.
    {"draft_simple",  COMMON_SPECULATIVE_TYPE_DRAFT},
    {"draft_mtp",     COMMON_SPECULATIVE_TYPE_MTP},
    {"draft_eagle3",  COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"draft_dflash",  COMMON_SPECULATIVE_TYPE_DFLASH},
    {"draft_dspark",  COMMON_SPECULATIVE_TYPE_DSPARK},
};

const llama_ckpt_data spec_ckpt = { LLAMA_CHECKPOINT_SPEC, 0 };

bool common_speculative_needs_checkpoint(const llama_model * model) {
    return model != nullptr &&
        llama_model_has_recurrent(model);
}

void common_speculative_checkpoint::clear() {
    valid = false;
    mode = LLAMA_SPEC_CKPT_NONE;
    n_past = 0;
    sampled = LLAMA_TOKEN_NULL;

    if (sampler != nullptr) {
        common_sampler_free(sampler);
        sampler = nullptr;
    }
}

struct common_speculative_config {
    common_speculative_stage_params stage;
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(
            const common_speculative_stage_params & s,
            const common_params_speculative & p = common_params_speculative{})
        : stage(s), type(s.type), params(p) {}
};

// exported: speculative.h declares it (upstream API, used by tools/imatrix for its NextN draft check)
bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_WRN("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (!llama_model_is_gemma4_mtp_assistant(model_dft) &&
       (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_WRN("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_WRN("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_WRN("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_WRN("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

static bool common_speculative_target_has_appended_mtp_contract(const llama_model * model) {
    return llama_model_is_step35(model) || llama_model_is_deepseek4(model) ||
           llama_model_is_qwen35_family(model) || llama_model_is_qwen4exp(model);
}

static bool common_speculative_has_recognized_mtp_companion(
        const llama_model * target,
        const llama_model * companion) {
    if (target == nullptr || companion == nullptr) {
        return false;
    }
    if (llama_model_is_gemma4_mtp_assistant(companion)) {
        return llama_model_arch_string(target) != nullptr &&
               std::strcmp(llama_model_arch_string(target), "gemma4") == 0;
    }
    if (llama_model_mtp_package(companion) != LLAMA_MTP_PACKAGE_COMPANION) {
        return false;
    }
    if (llama_model_is_qwen4exp(target) || llama_model_is_qwen4exp(companion)) {
        return llama_model_is_qwen4exp(target) && llama_model_is_qwen4exp(companion);
    }
    return common_speculative_target_has_appended_mtp_contract(target);
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // Position zero represents speculative position 1.
    std::vector<uint64_t> drafted_by_position;
    std::vector<uint64_t> accepted_by_position;

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    // Length of the context match behind the latest draft, when the stage knows it (-1 = unknown). Read by the draft policy.
    int last_match_len = -1;

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_pos draft_base_pos,
            llama_seq_id draft_seq_id,
            llama_tokens & result) {
        GGML_UNUSED(draft_base_pos);
        GGML_UNUSED(draft_seq_id);
        draft(params, prompt_tgt, id_last, result);
    }

    virtual void accept(uint16_t n_accepted) = 0;

    // S69 Stage-3: per-decode hook (canonical EAGLE3 process(), llama.cpp PR#18039). Called on EVERY target
    // decode batch so a draft impl can build its KV in lockstep (eagle3 (token[P+1],g_embd[P]) convention +
    // deferred boundary). Default no-op so MTP/DFlash/draft/ngram states are unaffected. Returns false on error.
    virtual bool process(const llama_batch & batch) { GGML_UNUSED(batch); return true; }
};

struct common_speculative_state_mtp;
struct common_speculative_state_dflash;

static common_speculative_state_mtp * common_speculative_get_mtp_state(common_speculative * spec);
static const common_speculative_state_mtp * common_speculative_get_mtp_state(const common_speculative * spec);
static common_speculative_state_dflash * common_speculative_get_dflash_state(common_speculative * spec);
static const common_speculative_state_dflash * common_speculative_get_dflash_state(const common_speculative * spec);

// Normalised probability of the top candidate, softmax over the candidates' post-chain logits. The p_min draft gate
// needs this, and cur_p->data[0].p is NOT it in this fork: DIST is llama_sample_softmax_impl(normalize=false), so p holds
// exp(logit - max) and the top candidate always reads 1.0 -- the gate never fired (MEASURED 2026-09-28, 'top p=1.000' at
// every drafted position of temp/_samp_0928_diag2_server.log while the normalised top probability ranged 0.12-1.00).
// Upstream normalises (engines/llama.cpp src/llama-sampler.cpp:316-318), which is what its identical-looking gate reads. Expects cur_p sorted descending.
static float common_speculative_top_prob(const llama_token_data_array * cur_p) {
    if (cur_p == nullptr || cur_p->size == 0) {
        return 0.0f;
    }
    const float top = cur_p->data[0].logit;
    double sum = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
        sum += std::exp((double) cur_p->data[i].logit - (double) top);
    }
    return sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
}
static int32_t common_speculative_feature_width(const common_speculative * spec);
static void mtp_invalidate_cached_drafts(common_speculative_state_mtp & state);
static bool common_speculative_checkpoint_save(
    common_speculative_checkpoint & ckpt,
    llama_model * model,
    llama_context * ctx,
    common_sampler * sampler_src,
    const common_params_sampling & sparams,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token sampled,
    int max_tokens,
    int ckpt_mode);

static std::vector<llama_token> mtp_speculative_gen_draft(
    common_speculative_state_mtp & state,
    struct common_sampler * smpl,
    struct llama_context * ctx,
    int n_draft,
    float p_min,
    int32_t mtp_heads,
    llama_token id_last,
    llama_pos n_past,
    llama_seq_id seq_id,
    bool constant_draft_positions = false);

static int32_t mtp_update_kv_cache(struct llama_context * ctx, const llama_batch & batch, bool is_prompt_warmup, int32_t mtp_heads);

struct mtp_last_embd {
    std::vector<float> embd;
    float prob = 0.0f;
    int   last_id = -1;
};

struct common_speculative_state_mtp : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_mtp = nullptr;
    int32_t mtp_heads_active = 0;
    // number of NextN heads the model carries, and the minimum head count the committed
    // context has been warmed with since position 0 (deeper heads' caches only hold valid
    // rows for spans warmed with them; a request drafting with MORE heads than the cached
    // prefix was warmed with must reprocess from scratch). Single-sequence by design, like
    // the rest of the openPangu MTP state.
    int32_t n_heads_model = 1;
    int32_t mtp_warmed_heads = 0;
    common_sampler * smpl;

    int32_t resolved_heads() const {
        return mtp_heads_active > 0 ? std::min(mtp_heads_active, n_heads_model) : n_heads_model;
    }
    // For Gemma 4 external MTP assistant: draft positions are held constant
    bool constant_draft_positions = false;
    int n_embd = 0;
    std::unordered_map<llama_seq_id, std::vector<float>> target_hidden_by_seq;
    std::unordered_map<llama_seq_id, mtp_last_embd> draft_cache_by_seq;

    common_speculative_state_mtp(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_mtp,
            bool constant_draft_positions = false)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_mtp(ctx_mtp)
        , constant_draft_positions(constant_draft_positions)
    {
        struct common_params_sampling sparams;
        sparams.samplers_sequence = {
            llama_sampler_type::DIST,
        };
        smpl = common_sampler_init(llama_get_model(ctx_mtp), sparams);
        llama_set_mtp_target_context(ctx_mtp, ctx_tgt);
        n_embd = llama_mtp_state_n_embd(ctx_mtp);
        n_heads_model = std::max(1, llama_model_n_nextn_layer(llama_get_model(ctx_mtp)));

        LOG_INF("%s: MTP context ready (n_ctx=%d, constant_draft_positions=%s)\n", __func__,
                llama_n_ctx(ctx_mtp), constant_draft_positions ? "true" : "false");
        if (n_heads_model > 1) {
            LOG_INF("%s: model carries %d NextN/MTP heads; heads=1 is the default, while "
                    "heads>1 and heads=0 (all model heads) are experimental\n",
                    __func__, n_heads_model);
        }
    }

    ~common_speculative_state_mtp() override {
        common_sampler_free(smpl);
        if (ctx_mtp) {
            llama_free(ctx_mtp);
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        target_hidden_by_seq.clear();
        draft_cache_by_seq.clear();
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        draft(params, prompt_tgt, id_last, -1, 0, result);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_pos draft_base_pos,
            llama_seq_id seq_id,
            llama_tokens & result) override {

        const llama_pos mtp_pos_max = llama_kv_cache_seq_pos_max(ctx_mtp, seq_id);
        const bool has_draft_base_pos = draft_base_pos >= 0;
        // Prefer the target slot position when the caller has it. Gemma4 external MTP reads
        // the target KV cache directly, so ctx_mtp's own KV position is not authoritative.
        const llama_pos n_past = has_draft_base_pos
            ? draft_base_pos
            : (mtp_pos_max >= 0 ? mtp_pos_max + 1 : (llama_pos) prompt_tgt.size());

        if (!has_draft_base_pos && !prompt_tgt.empty() && mtp_pos_max < (llama_pos)prompt_tgt.size() - 1) {
            LOG_WRN("%s: MTP context not fully warmed up: pos_max = %d, expected = %d\n",
                    __func__, (int)mtp_pos_max, (int)prompt_tgt.size() - 1);
        }
        if (has_draft_base_pos && !constant_draft_positions && mtp_pos_max < n_past - 1) {
            LOG_WRN("%s: MTP context not fully warmed up: pos_max = %d, expected >= %d\n",
                    __func__, (int)mtp_pos_max, (int)n_past - 1);
        }

        llama_context * ctx = ctx_mtp;
        mtp_heads_active = std::max<int32_t>(0, params.mtp_heads);

        const auto hidden_it = target_hidden_by_seq.find(seq_id);
        if (hidden_it == target_hidden_by_seq.end() || (int) hidden_it->second.size() != n_embd) {
            LOG_WRN("%s: missing target hidden state for seq_id %d\n", __func__, (int) seq_id);
            result.clear();
            return;
        }

        if (!llama_set_draft_input_hidden_state_copy(ctx, hidden_it->second.data(), hidden_it->second.size())) {
            result.clear();
            return;
        }

        result = mtp_speculative_gen_draft(
            *this,
            smpl,
            ctx,
            params.n_max,
            params.p_min,
            params.mtp_heads,
            id_last,
            n_past,
            seq_id,
            constant_draft_positions
        );
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};

#include "speculative-dflash-impl.h"

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;
        {
            struct common_params_sampling params;
            params.top_k = 10;
            params.samplers_sequence = {
                llama_sampler_type::TOP_K,
                llama_sampler_type::DIST, // needed to get probabilities
            };
            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);



            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_kv_cache_clear(ctx_dft);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_kv_cache_seq_rm (ctx_dft, 0, 0, reuse_i);
                llama_kv_cache_seq_add(ctx_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_kv_cache_seq_rm (ctx_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        //LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, nullptr, id, true);

            // only collect very high-confidence draft tokens
            if (params.p_min_explicit && common_speculative_top_prob(cur_p) < params.p_min) {
                if (i == 0) {
                    result.push_back(id);
                }
                break;
            }

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }


            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

// S69 #33 EAGLE3 separate-draft, hidden-state-conditioned spec state. The target's 3 captured hidden layers
// (set on ctx_tgt via llama_set_eagle3_capture during the target decode) are read here, fed to the draft head
// (build_eagle3 via llama_set_draft_input_hidden_state_copy), which proposes draft tokens.
// CURRENT: K=1 single-token draft (correct + lossless; the target always verifies). The recurrent K>1 chain
// (eagle3_g_direct/eagle3_a feedback) is the next refinement — see DEC-S69-EAGLE3-3LAYER-EXTRACTION-STAYS-CUSTOM.
// Attach an EAGLE3 draft context to its target: the draft reuses the target's token embeddings, the target captures
// its hidden states at the draft's TRAINED layers (gguf eagle3.target_layers, e.g. [2,18,33]; PR #18039 fuses
// early/mid/late features at exactly those layers, and generic defaults hurt the acceptance rate), and the d2t
// (draft->target vocab) map is loaded. The gguf may carry d2t as ABSOLUTE ids (upstream and this tree's converters) or as
// DELTAS (our earlier #29 converter); it is classified once and normalised to absolute (common/eagle3-d2t.h). An unusable
// d2t turns drafting off for this draft, as before (moved here unchanged from the single-context state). RESEARCH-DONE:
// ggml-org/llama.cpp PR #18039 diff, read 2026-10-18 (the converter writes absolute ids and checks them in range).
// Shared by the single-context state and the per-sequence pool impl. Returns the target hidden width.
static int common_eagle3_attach(llama_context * ctx_tgt, llama_context * ctx_dft, std::vector<int32_t> & d2t, bool & d2t_invalid) {
    llama_eagle3_set_target(ctx_dft, ctx_tgt);
    const int n_embd_tgt = (int) llama_model_n_embd(llama_get_model(ctx_tgt));

    const int n_layer_tgt = (int) llama_n_layer(llama_get_model(ctx_tgt));
    int tl[3] = { n_layer_tgt > 4 ? 2 : 0, n_layer_tgt / 2, n_layer_tgt - 1 };  // fallback
    llama_eagle3_target_layers(ctx_dft, tl);  // overwrite with trained layers if the gguf records them
    llama_set_eagle3_capture(ctx_tgt, true, tl[0], tl[1], tl[2]);

    d2t.clear();
    d2t_invalid = false;
    const int32_t nd2t = llama_eagle3_get_d2t(ctx_dft, nullptr);  // size query
    common_eagle3_d2t_form d2t_form = COMMON_EAGLE3_D2T_INVALID;
    if (nd2t > 0) {
        std::vector<int32_t> tmp((size_t) nd2t);
        if (llama_eagle3_get_d2t(ctx_dft, tmp.data()) == nd2t) {
            d2t_form = common_eagle3_d2t_normalize(tmp, llama_n_vocab(llama_get_model(ctx_tgt)));
            if (d2t_form == COMMON_EAGLE3_D2T_INVALID) {
                d2t_invalid = true;
                LOG_ERR("%s: EAGLE3 d2t (%d entries) is neither an absolute nor a delta map into the target vocab "
                        "(%d); drafting disabled\n", __func__, nd2t, llama_n_vocab(llama_get_model(ctx_tgt)));
            } else {
                d2t = std::move(tmp);
            }
        } else {
            d2t_invalid = true;
            LOG_ERR("%s: EAGLE3 d2t present (%d entries) but unreadable (unsupported dtype); drafting disabled\n",
                    __func__, nd2t);
        }
    } else if (nd2t < 0) {
        d2t_invalid = true;
        LOG_ERR("%s: EAGLE3 d2t present but has an unsupported dtype; drafting disabled\n", __func__);
    }
    LOG_INF("%s: EAGLE3 draft ready (n_embd_tgt=%d, capture layers=[%d,%d,%d], d2t=%d, d2t form=%s)\n",
            __func__, n_embd_tgt, tl[0], tl[1], tl[2], nd2t,
            nd2t == 0 ? "none" : common_eagle3_d2t_form_name(d2t_form));
    return n_embd_tgt;
}

struct common_speculative_state_eagle3 : public common_speculative_state {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;   // the EAGLE3 draft-head context
    common_sampler * smpl   = nullptr;
    llama_batch      batch;
    int              n_embd_tgt = 0;
    std::vector<int32_t> d2t;            // draft->target vocab map as ABSOLUTE target ids; empty = identity fallback
    bool                 d2t_invalid = false;  // gguf has a d2t this reader cannot use -> never draft
    // S69 Stage-3/4: deferred boundary state — last (token+feats) pair of the most recent process() batch.
    // The boundary token is unknown until the NEXT process() (its first token) or draft() (id_last) supplies it.
    std::vector<float> pending_g_last;   // [3*n_embd_tgt] feats at the boundary pos
    llama_pos          pending_pos_last = -1;
    bool               have_pending = false;
    llama_token        last_known_tok = -1;  // most recent token in this seq, used to complete prior boundary

    common_speculative_state_eagle3(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        struct common_params_sampling sparams;
        sparams.samplers_sequence = { llama_sampler_type::DIST };
        smpl = common_sampler_init(llama_get_model(ctx_dft), sparams);
        n_embd_tgt = common_eagle3_attach(ctx_tgt, ctx_dft, d2t, d2t_invalid);
    }

    ~common_speculative_state_eagle3() override {
        if (ctx_tgt) llama_set_eagle3_capture(ctx_tgt, false, -1, -1, -1);  // restore clean target serving path
        common_sampler_free(smpl);
        llama_batch_free(batch);
        if (ctx_dft) llama_free(ctx_dft);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        result.clear();
        if (d2t_invalid) return;

        // Canonical EAGLE3 (PR#18039): the boundary deferred by process() is completed HERE — id_last is
        // target's freshest sample, paired with pending_g_last at pending_pos_last to write the missing KV row.
        // Then we sample the first draft token (K=1 baseline; K>1 recurrence added in a follow-up).
        std::vector<float> feats;
        llama_pos draft_pos = -1;

        if (have_pending && pending_pos_last >= 0 && (int) pending_g_last.size() == 3 * n_embd_tgt) {
            feats = pending_g_last;
            draft_pos = pending_pos_last;
            // ensure ctx_dft has no stale rows beyond the boundary
            llama_kv_cache_seq_rm(ctx_dft, 0, pending_pos_last, -1);
        } else {
            // fallback: last-token feats (legacy K=1 path)
            feats.assign((size_t) 3 * n_embd_tgt, 0.0f);
            const int32_t n = llama_get_eagle3_target_feats(ctx_tgt, feats.data());
            if (n <= 0) { LOG_DBG("%s: no EAGLE3 target feats\n", __func__); return; }
            const llama_pos pos_max = llama_kv_cache_seq_pos_max(ctx_dft, 0);
            draft_pos = pos_max >= 0 ? pos_max + 1 : 0;
        }

        if (!llama_set_draft_input_hidden_state_copy(ctx_dft, feats.data(), feats.size())) return;

        common_batch_clear(batch);
        common_batch_add(batch, id_last, draft_pos, { 0 }, true);
        if (llama_decode(ctx_dft, batch) != 0) {
            LOG_WRN("%s: EAGLE3 draft decode failed\n", __func__);
            return;
        }

        // Boundary completed → mark consumed so the next draft() doesn't re-write the same pos.
        have_pending = false;
        llama_token id = common_sampler_sample(smpl, ctx_dft, -1);
        common_sampler_accept(smpl, ctx_dft, id, true);
        // map draft-vocab id -> target-vocab id via d2t, normalised to ABSOLUTE ids at init whichever convention the
        // gguf used (common/eagle3-d2t.h). A draft id outside the map has no target id: draft nothing.
        const llama_token draft_raw = id;
        if (!d2t.empty()) {
            if (id < 0 || id >= (llama_token) d2t.size()) {
                LOG_DBG("%s: EAGLE3 draft id %d outside d2t (%zu)\n", __func__, (int) id, d2t.size());
                return;
            }
            id = (llama_token) d2t[id];
        }
        if (getenv("EAGLE3_DEBUG")) {   // S69 draft-fidelity diagnostic (off by default)
            double s = 0; for (float f : feats) s += (double) f * f;
            LOG_INF("EAGLE3-DBG: feats_n=%d feats_l2=%.4f id_last=%d draft_raw=%d -> target_id=%d\n",
                    (int) feats.size(), (double) std::sqrt(s), (int) id_last, (int) draft_raw, (int) id);
        }
        result.push_back(id);
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }

    // S69 Stage-4: per-decode hook builds draft KV in lockstep with target.
    // Canonical convention (llama.cpp PR#18039): "at memory pos P, decoder input pair = (token[P+1], g_embd[P])".
    // We get g_embd from the target ctx's per-position feats (Stage-2 capture) via the eagle3 fc inside
    // build_eagle3 (we feed feats; the graph computes g internally). For each k in [0, n_tokens-1) of the seq_id 0
    // contiguous range, write decoder KV at pos[k] with (token[k+1], feats[k]). Last pos = DEFERRED boundary
    // (token[k+1] is the NEXT target sample — unknown here; completed by draft()'s seed step or the next process()).
    size_t  n_process_calls = 0;
    bool process(const llama_batch & batch_in) override {
        n_process_calls++;
        if (!batch_in.token || !batch_in.pos || batch_in.n_tokens <= 0) return true;

        // collect seq_id 0 contiguous range (server's default ordering = one seq/batch)
        int beg = -1, end = -1;
        for (int k = 0; k < batch_in.n_tokens; ++k) {
            const bool match = batch_in.n_seq_id && batch_in.n_seq_id[k] >= 1 && batch_in.seq_id && batch_in.seq_id[k][0] == 0;
            if (!match) continue;
            end = k; if (beg < 0) beg = k;
        }
        if (beg < 0) return true;

        const int n_rows_pp = llama_eagle3_get_feats_pp_ith(ctx_tgt, -1, nullptr);
        const int feats_w   = 3 * n_embd_tgt;
        if (n_rows_pp <= 0) {
            if (getenv("EAGLE3_DEBUG")) LOG_INF("EAGLE3-PROCESS: no per-pos feats captured yet\n");
            // Still record boundary intent for the future (using last-token feats fallback).
            return true;
        }

        std::vector<float> feats((size_t) feats_w);
        const int n_pairs = end - beg;   // [beg, end) -> n_pairs writes; pos=end is deferred
        for (int k = beg; k < end; ++k) {
            // pull feats[k]
            const int n = llama_eagle3_get_feats_pp_ith(ctx_tgt, k, feats.data());
            if (n != feats_w) continue;
            if (!llama_set_draft_input_hidden_state_copy(ctx_dft, feats.data(), feats.size())) continue;

            // write KV at pos[k] with token[k+1]
            common_batch_clear(batch);
            common_batch_add(batch, batch_in.token[k+1], batch_in.pos[k], { 0 }, /*logits=*/ false);

            if (llama_decode(ctx_dft, batch) != 0) {
                LOG_WRN("EAGLE3-PROCESS: ctx_dft decode failed at k=%d pos=%d\n", k, (int) batch_in.pos[k]);
                break;
            }
        }

        // stash deferred boundary: feats[end], pos[end]; token at pos[end]+1 will be supplied by draft() (id_last)
        if (end >= 0) {
            pending_g_last.assign((size_t) feats_w, 0.0f);
            if (llama_eagle3_get_feats_pp_ith(ctx_tgt, end, pending_g_last.data()) == feats_w) {
                pending_pos_last = batch_in.pos[end];
                have_pending = true;
                last_known_tok = batch_in.token[end];
            }
        }

        if (getenv("EAGLE3_DEBUG")) {
            LOG_INF("EAGLE3-PROCESS: call #%zu wrote KV pairs=%d, deferred pos=%d (n_rows_pp=%d)\n",
                    n_process_calls, n_pairs, (int) pending_pos_last, n_rows_pp);
        }
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;
        n_low = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;
    common_ngram_cache ngram_cache_learned; // finished requests' context n-grams, merged into the dynamic file at exit
    std::string        path_dynamic;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            // upstream 83c2d012a: a MISSING dynamic file starts empty and is written at destruction; an existing file that
            // cannot be read aborts (writing it at exit would destroy it)
            std::error_code ec;
            const bool exists = std::filesystem::exists(path_dynamic, ec);
            if (!exists && !ec) {
                LOG_WRN("dynamic lookup cache %s does not exist, starting empty (written on exit)\n", path_dynamic.c_str());
            } else {
                try {
                    ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
                } catch (...) {
                    LOG_ERR("failed to open dynamic lookup cache: %s\n", path_dynamic.c_str());
                    GGML_ABORT("Couldn't read dynamic lookup cache");
                }
            }
            this->path_dynamic = path_dynamic;
            this->save_dynamic = true;
        }
    }

    ~common_speculative_state_ngram_cache() override {
        if (!save_dynamic || path_dynamic.empty()) {
            return;
        }
        // upstream 83c2d012a: the dynamic cache plus the context n-grams of every request is written back at destruction
        common_ngram_cache_merge(ngram_cache_learned, ngram_cache_context);
        common_ngram_cache_merge(ngram_cache_dynamic, ngram_cache_learned);
        try {
            common_ngram_cache_save(ngram_cache_dynamic, path_dynamic);
            LOG_INF("saved dynamic lookup cache %s (%zu n-grams)\n", path_dynamic.c_str(), ngram_cache_dynamic.size());
        } catch (const std::exception & e) {
            LOG_ERR("failed to save dynamic lookup cache %s: %s\n", path_dynamic.c_str(), e.what());
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        // upstream 83c2d012a resets cache_size per request; the context cache is append-only (ngram-cache.h), so it is cleared
        // too (a shorter new prompt would otherwise add none of its text) and the previous request's n-grams kept for the file
        if (save_dynamic) {
            common_ngram_cache_merge(ngram_cache_learned, ngram_cache_context);
        }
        ngram_cache_context.clear();
        cache_size = 0;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        if (cache_size > inp.size()) {
            // the text got shorter (context shift / truncation): the append-only cache is rebuilt
            ngram_cache_context.clear();
            cache_size = 0;
        }
        if (cache_size < inp.size()) {
            // upstream 83c2d012a: the FULL history with nnew = the unseen tail, so n-grams ending at the new tokens keep the
            // context before them (passing only the new tokens left the cache holding just the prompt)
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    inp, (int) (inp.size() - cache_size), false);
            cache_size = inp.size();
        }

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_suffix : public common_speculative_state {
    common_suffix_tree tree;
    common_suffix_tree corpus_tree;
    bool has_corpus = false;
    size_t cache_size   = 0;

    // Acceptance feedback
    size_t n_draft_last  = 0;
    bool   had_accept    = false;
    int    n_low         = 0;
    float  base_p_min    = 0.1f;
    float  eff_p_min     = 0.1f;

    common_speculative_state_suffix(
            enum common_speculative_type type,
            int max_depth,
            const std::string & corpus_path,
            const llama_model * model,
            size_t max_nodes = 0)
        : common_speculative_state(type)
        , tree(max_depth)
        , corpus_tree(max_depth)
    {
        // the history tree grows with every request; the corpus tree is read-only after load and bounded by its own insert-work limit
        tree.set_max_nodes(max_nodes);
        if (!corpus_path.empty()) {
            std::function<std::vector<llama_token>(const std::string &)> tokenize_fn;
            if (model) {
                tokenize_fn = [model](const std::string & text) -> std::vector<llama_token> {
                    // ADAPTED 2026-09-13: common_tokenize takes a llama_context or a
                    // llama_vocab (common.h:1424,1430), never a llama_model. The vocab
                    // overload is the one meant here -- there is no context at this point
                    // -- and llama_model_get_vocab is how the rest of this file reaches it.
                    return common_tokenize(llama_model_get_vocab(model), text, false, true);
                };
            }
            has_corpus = corpus_tree.load_corpus(corpus_path, tokenize_fn);
        }
    }

    void begin(const llama_tokens & prompt) override {
        cache_size   = 0;
        n_draft_last = 0;
        had_accept   = false;
        n_low        = 0;
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        base_p_min = params.p_min;
        if (n_draft_last > 0 && !had_accept) {
            if (++n_low >= 3) {
                eff_p_min = std::min(eff_p_min + 0.1f, 0.5f);
                n_low     = 0;
            }
        }
        had_accept = false;

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last);

            tree.extend(tokens_new.data(), (int)tokens_new.size());
            cache_size = prompt_tgt.size() + 1;
        }

        const int ctx_len = std::min((int)(prompt_tgt.size() + 1), tree.max_depth());
        llama_tokens context;
        context.reserve(ctx_len);
        const int ctx_start = (int)prompt_tgt.size() + 1 - ctx_len;
        for (int j = ctx_start; j < (int)prompt_tgt.size(); ++j) {
            context.push_back(prompt_tgt[j]);
        }
        context.push_back(id_last);
        const int min_match_len = std::max(1, params.suffix_min_match_len);

        result = tree.speculate(
            context.data(), (int)context.size(),
            params.n_max,
            eff_p_min,
            1,
            min_match_len);

        if (has_corpus) {
            auto corpus_result = corpus_tree.speculate(
                context.data(), (int)context.size(),
                params.n_max,
                eff_p_min,
                1,
                min_match_len);
            if (corpus_result.size() > result.size()) {
                result = std::move(corpus_result);
            }
        }

        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (n_draft_last == 0) {
            return;
        }
        had_accept = true;
        const double f_acc = (double)n_accepted / (double)n_draft_last;
        if (f_acc < 0.5) {
            if (++n_low >= 3) {
                eff_p_min = std::min(eff_p_min + 0.1f, 0.5f);
                n_low     = 0;
            }
        } else {
            n_low = 0;
            if (eff_p_min > base_p_min) {
                eff_p_min = std::max(eff_p_min - 0.05f, base_p_min);
            }
        }
    }
};

// S69-TRTLLM-PICK-SUFFIXAUTOMATON-2026-09-28-01: longest-suffix-match self-speculation over common_suffix_automaton (suffix-automaton.h).
// Same feeding contract as the suffix-tree stage above (the accepted history, then id_last) with O(n) memory and no depth cap.
struct common_speculative_state_suffix_automaton : public common_speculative_state {
    common_suffix_automaton sa;
    size_t cache_size = 0;

    explicit common_speculative_state_suffix_automaton(enum common_speculative_type type)
        : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        // A fresh sequence starts a fresh history; the automaton is append-only, so a new run must not see the old text.
        sa.clear();
        cache_size = 0;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last);
            sa.extend(tokens_new.data(), (int) tokens_new.size());
            cache_size = prompt_tgt.size() + 1;
        }
        result = sa.speculate(params.n_max, std::max(1, params.suffix_min_match_len));
        last_match_len = sa.match_len();
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};


// --- unsloth/master's per-sequence-pool speculative-decoding design, kept ALONGSIDE
//     the fork's single-companion-context design above (see speculative.h's top-of-file
//     merge-shape note). Only the ngram-map-k/k4v concrete implementation is carried --
//     see common_speculative_init(params, n_seq) below for why the other 7 upstream impl
//     classes are not. ---
using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl

struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}

    // --spec-draft-policy inputs. match_len: length of the context match behind this seq's latest draft (-1 = unknown);
    // stage_n_max: the most tokens this stage drafts per round (-1 = unknown). Read by the pool's draft policy.
    virtual int     match_len(llama_seq_id /*seq_id*/) const { return -1; }
    virtual int32_t stage_n_max() const { return -1; }

    // --spec-async (common_speculative_async_start): only the draft-model impl overrides these
    virtual void async_start(const std::vector<common_speculative_async_req> & /*reqs*/) {}
    virtual void async_cancel() {}
    virtual bool async_stats(common_speculative_async_stats & /*out*/) const { return false; }
};


struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    // the n-gram key is what matched (the single-context policy falls back to ngram_size_n the same way)
    int match_len(llama_seq_id seq_id) const override {
        return seq_id >= 0 && seq_id < (llama_seq_id) config.size() ? (int) config[seq_id].size_key : -1;
    }

    int32_t stage_n_max() const override {
        return config.empty() ? -1 : (int32_t) config[0].size_value;
    }
};


// S69-TRTLLM-PICK-SUFFIXAUTOMATON-2026-09-28-01 follow-up (F1 2026-10-06): the per-sequence pool (the one llama-server builds through
// common_speculative_init(params, n_seq)) carried only ngram_map_k/k4v/draft, so a `--spec-type suffix_automaton` server run got a
// null speculator. This wraps the fork's existing single-context states (suffix tree, suffix automaton) as one state per sequence;
// the states already hold exactly the per-sequence history they need, so no new drafting code is added here.
template <class State>
struct common_speculative_impl_state_per_seq : public common_speculative_impl {
    std::vector<std::unique_ptr<State>> states;
    common_params_speculative           cfg;

    template <class... Args>
    common_speculative_impl_state_per_seq(common_speculative_type type, uint32_t n_seq, const common_params_speculative & cfg, Args &&... args)
        : common_speculative_impl(type, n_seq), cfg(cfg) {
        for (uint32_t i = 0; i < n_seq; i++) {
            states.push_back(std::make_unique<State>(type, args...));
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) n_seq);
        states[seq_id]->begin(prompt);
    }

    bool process(const llama_batch & /*batch*/) override { return true; }

    void draft(common_speculative_draft_params_vec & dparams) override {
        GGML_ASSERT(dparams.size() == n_seq);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }
            common_params_speculative p = cfg;
            if (dp.n_max >= 0) {
                p.n_max = std::min(p.n_max, dp.n_max);
            }
            states[seq_id]->draft(p, *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) n_seq);
        if (is_other) {
            return;
        }
        states[seq_id]->accept(n_accepted);
    }

    int match_len(llama_seq_id seq_id) const override {
        return seq_id >= 0 && seq_id < (llama_seq_id) n_seq ? states[seq_id]->last_match_len : -1;
    }

    int32_t stage_n_max() const override { return cfg.n_max; }
};

// S69-FORK-SERVER-POOL-NGRAM-MOD-SIMPLE-CACHE-DSPARK-IMPLS-2026-10-17-01: ngram_mod for the per-sequence pool (llama-server).
// SOURCE-READ: llama.cpp b9590 common/speculative.cpp:897-1074 (common_speculative_impl_ngram_mod, MIT) and the fork's single-context
// common_speculative_state_ngram_mod above. Written for this tree, no code copied verbatim.
// One hash table for ALL sequences (upstream's choice: a slot drafts from what every slot has seen), per-sequence feed position and
// acceptance streak. NUANCES: (1) a reset of the shared table, by occupancy at begin() or by one slot's low-acceptance streak, empties it
// for every slot, so each slot carries the table generation it last fed and re-feeds its whole history after a reset (upstream leaves
// the other slots' feed position where it was, so in upstream those slots do not re-add their earlier text); (2) since a reset hits
// every slot, the streak rule is upstream's pooled one (5 rounds below 25 %), not the single-context 3 rounds below 50 %; (3) the
// stage's own n_max / n_min come from the stage overrides and the round's n_max (dp.n_max, the server's per-slot budget) caps them.
struct common_speculative_impl_ngram_mod_pool : public common_speculative_impl {
    common_params_speculative          cfg;
    std::shared_ptr<common_ngram_mod>  mod;
    uint64_t                           gen = 0; // bumped on every reset of the shared table
    const bool                         verbose;

    struct seq_info {
        size_t   i_last       = 0; // next prompt position to add to the table
        size_t   n_draft_last = 0; // tokens of the last draft (0 = none)
        int      n_low        = 0; // consecutive low-acceptance rounds
        uint64_t gen_fed      = 0; // table generation this sequence's i_last refers to
    };
    std::vector<seq_info> sinfo;

    common_speculative_impl_ngram_mod_pool(const common_params_speculative & cfg, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq), cfg(cfg), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
        const uint16_t n = cfg.ngram_size_n > 0 ? cfg.ngram_size_n : 12;
        mod = cfg.ngram_mod ? cfg.ngram_mod : std::make_shared<common_ngram_mod>(n, 4*1024*1024);
        sinfo.resize(n_seq);
        LOG_INF("%s: per-sequence ngram_mod: n=%zu, n_max=%d, n_min=%d, one shared table of %zu entries (%.3f MB) for %u sequence(s)\n",
                __func__, mod->get_n(), cfg.n_max, cfg.n_min, mod->size(), (float) mod->size_bytes() / 1024 / 1024, n_seq);
        if (mod->get_n() < 16) {
            LOG_WRN("%s: ngram_mod n=%zu is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n",
                    __func__, mod->get_n());
        }
    }

    void reset_table(const char * why) {
        if (verbose) {
            LOG_WRN("%s: resetting the shared ngram_mod table (%s)\n", __func__, why);
        }
        mod->reset();
        ++gen;
    }

    void feed(seq_info & si, const llama_tokens & tokens, size_t upto) {
        if (si.gen_fed != gen) {
            si.i_last  = 0;
            si.gen_fed = gen;
        }
        for (size_t i = si.i_last; i < upto; ++i) {
            mod->add(tokens.data() + i);
        }
        si.i_last = std::max(si.i_last, upto);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) n_seq);
        auto & si = sinfo[seq_id];
        si = seq_info{};
        si.gen_fed = gen;

        const size_t n = mod->get_n();
        if (prompt.size() < n) {
            return;
        }
        feed(si, prompt, prompt.size() - n);

        const double f = (double) mod->get_used() / (double) mod->size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod->get_used(), mod->size(), f);
        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);
            reset_table("occupancy");
        }
    }

    bool process(const llama_batch & /*batch*/) override { return true; }

    void draft_one(seq_info & si, const common_speculative_draft_params & dp) {
        auto & result = *dp.result;
        const auto & prompt = *dp.prompt;

        si.n_draft_last = 0;

        const size_t n       = mod->get_n();
        const size_t cur_len = prompt.size();
        if (n == 0 || cur_len < n) {
            return;
        }

        // add the new n-grams in chunks of 32 positions (a re-feed after a reset is always taken)
        if (si.gen_fed != gen || si.i_last + 32 < cur_len) {
            feed(si, prompt, cur_len - n);
        }

        int n_max = cfg.n_max;
        if (dp.n_max >= 0) {
            n_max = std::min(n_max, dp.n_max);
        }
        const int n_min = dp.n_min >= 0 ? dp.n_min : cfg.n_min;
        if (n_max <= 0) {
            return;
        }

        // the lookup window: the last n-1 committed tokens, then id_last; drafted tokens are appended after it
        llama_tokens buf(n + n_max);
        for (size_t i = 0; i + 1 < n; ++i) {
            buf[i] = prompt[cur_len - n + 1 + i];
        }
        buf[n - 1] = dp.id_last;

        int n_got = 0;
        for (; n_got < n_max; ++n_got) {
            const llama_token tok = mod->get(buf.data() + n_got);
            if (tok == common_ngram_mod::EMPTY) {
                break;
            }
            buf[n + n_got] = tok;
        }
        if (n_got == 0 || n_got < n_min) {
            return;
        }

        result.assign(buf.begin() + n, buf.begin() + n + n_got);
        si.n_draft_last = result.size();
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        GGML_ASSERT(dparams.size() == n_seq);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }
            draft_one(sinfo[seq_id], dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        auto & si = sinfo[seq_id];
        if (si.n_draft_last == 0) {
            return;
        }
        const double f_acc = (double) n_accepted / (double) si.n_draft_last;
        if (f_acc < 0.25) {
            if (++si.n_low >= 5) {
                reset_table("low acceptance streak");
                si.n_low = 0;
            }
        } else {
            si.n_low = 0;
        }
    }

    int match_len(llama_seq_id /*seq_id*/) const override { return (int) mod->get_n(); }

    int32_t stage_n_max() const override { return cfg.n_max; }
};

// S69-FORK-SERVER-POOL-NGRAM-MOD-SIMPLE-CACHE-DSPARK-IMPLS-2026-10-17-01: ngram_cache (lookup decoding) for the per-sequence pool.
// SOURCE-READ: llama.cpp b9590 common/speculative.cpp:1076-1215 (common_speculative_impl_ngram_cache, MIT) and the fork's single-context
// common_speculative_state_ngram_cache above. Written for this tree, no code copied verbatim.
// NUANCES: (1) the static and dynamic caches are loaded ONCE and shared by every sequence (common_ngram_cache_draft only find()s in
// them, ngram-cache.cpp:67,165) where upstream copies them into each sequence; (2) the context cache is PER SEQUENCE and is cleared at
// begin(): a slot's next request starts a new text, and common_ngram_cache_update may only be APPENDED to (ngram-cache.h:71-72).
// Upstream's begin() is a no-op, so its slot keeps the previous request's cache and fill mark; when the new prompt is shorter than the
// old text, none of the new text is added. Upstream issue ggml-org/llama.cpp#27852 reports acceptance 86 % -> 11 % on slot reuse; that
// this fill mark is its cause is our reading of the code (CAUSE-UNMEASURED upstream), the reuse arm of
// tests/test-server-spec-pool-stages.py tests our side. (3) the update gets the WHOLE text with nnew = the unseen
// tail (ngram-cache.cpp:22 starts at max(size - nnew, ngram_size) and reads ngram_size tokens before each position); the single-context
// state and upstream pass only the new tokens as the whole input, which never counts an n-gram that straddles the previous fill mark.
// (4) n_draft is the stage's n_max (the single-context state hard-codes 8, "TODO get from config"), capped per round by dp.n_max.
struct common_speculative_impl_ngram_cache_pool : public common_speculative_impl {
    common_params_speculative cfg;
    common_ngram_cache        cache_static;
    common_ngram_cache        cache_dynamic;
    common_ngram_cache        cache_learned;        // finished requests' context n-grams, merged into the dynamic file at exit
    bool                      save_dynamic = false; // a dynamic path is set and was loaded or is new (never an unreadable file)

    struct seq_info {
        size_t             cache_size = 0; // tokens of this sequence already in its context cache
        common_ngram_cache cache_context;
    };
    std::vector<seq_info> sinfo;

    common_speculative_impl_ngram_cache_pool(const common_params_speculative & cfg, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq), cfg(cfg) {
        if (!cfg.lookup_cache_static.empty()) {
            try {
                cache_static = common_ngram_cache_load(cfg.lookup_cache_static);
            } catch (...) {
                throw std::runtime_error("failed to open static lookup cache: " + cfg.lookup_cache_static);
            }
        }
        if (!cfg.lookup_cache_dynamic.empty()) {
            // upstream 83c2d012a: a MISSING dynamic file starts an empty cache that is written at destruction; a file that exists
            // but cannot be read is an error (overwriting it at exit would destroy it)
            std::error_code ec;
            const bool exists = std::filesystem::exists(cfg.lookup_cache_dynamic, ec);
            if (!exists && !ec) {
                LOG_WRN("%s: dynamic lookup cache %s does not exist, starting empty (written on exit)\n",
                        __func__, cfg.lookup_cache_dynamic.c_str());
                save_dynamic = true;
            } else {
                try {
                    cache_dynamic = common_ngram_cache_load(cfg.lookup_cache_dynamic);
                    save_dynamic  = true;
                } catch (...) {
                    throw std::runtime_error("failed to open dynamic lookup cache: " + cfg.lookup_cache_dynamic);
                }
            }
        }
        sinfo.resize(n_seq);
        LOG_INF("%s: per-sequence ngram_cache: n_max=%d, static cache %s (%zu n-grams), dynamic cache %s (%zu n-grams), %u sequence(s)\n",
                __func__, cfg.n_max,
                cfg.lookup_cache_static.empty()  ? "none" : cfg.lookup_cache_static.c_str(),  cache_static.size(),
                cfg.lookup_cache_dynamic.empty() ? "none" : cfg.lookup_cache_dynamic.c_str(), cache_dynamic.size(), n_seq);
    }

    ~common_speculative_impl_ngram_cache_pool() override {
        if (!save_dynamic) {
            return;
        }
        // upstream 83c2d012a saves the dynamic cache (plus every sequence's context n-grams) at destruction. Here the finished
        // requests' n-grams were set aside in cache_learned at begin(), so a slot's earlier requests are saved too; drafting
        // keeps reading the dynamic cache as loaded (the learned n-grams only reach it on the next run, as with llama-lookup).
        for (auto & si : sinfo) {
            common_ngram_cache_merge(cache_learned, si.cache_context);
        }
        common_ngram_cache_merge(cache_dynamic, cache_learned);
        try {
            common_ngram_cache_save(cache_dynamic, cfg.lookup_cache_dynamic);
            LOG_INF("%s: saved dynamic lookup cache %s (%zu n-grams)\n", __func__, cfg.lookup_cache_dynamic.c_str(), cache_dynamic.size());
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to save dynamic lookup cache %s: %s\n", __func__, cfg.lookup_cache_dynamic.c_str(), e.what());
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) n_seq);
        // a new request on this slot: its text is rebuilt from the full prompt at the next draft. The previous request's
        // n-grams are kept for the dynamic cache file when one is configured.
        if (save_dynamic) {
            common_ngram_cache_merge(cache_learned, sinfo[seq_id].cache_context);
        }
        sinfo[seq_id].cache_context.clear();
        sinfo[seq_id].cache_size = 0;
    }

    bool process(const llama_batch & /*batch*/) override { return true; }

    void draft(common_speculative_draft_params_vec & dparams) override {
        GGML_ASSERT(dparams.size() == n_seq);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }
            auto & si = sinfo[seq_id];
            const auto & prompt = *dp.prompt;

            int n_draft = cfg.n_max;
            if (dp.n_max >= 0) {
                n_draft = std::min(n_draft, dp.n_max);
            }
            if (n_draft <= 0) {
                continue;
            }

            // the text so far plus id_last; the context cache takes only what it has not seen
            llama_tokens inp(prompt.begin(), prompt.end());
            inp.push_back(dp.id_last);
            if (si.cache_size > inp.size()) {
                // the slot's text got shorter (context shift / truncation): the append-only cache is rebuilt
                si.cache_context.clear();
                si.cache_size = 0;
            }
            if (si.cache_size < inp.size()) {
                // the WHOLE text with nnew = the unseen tail, so n-grams that straddle the old/new boundary are counted too
                common_ngram_cache_update(si.cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, (int) (inp.size() - si.cache_size), false);
                si.cache_size = inp.size();
            }

            llama_tokens out = { dp.id_last };
            common_ngram_cache_draft(inp, out, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, si.cache_context, cache_dynamic, cache_static);
            dp.result->assign(out.begin() + 1, out.end()); // drop the leading id_last

            const int n_min = dp.n_min >= 0 ? dp.n_min : cfg.n_min;
            if ((int) dp.result->size() < n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {}

    int32_t stage_n_max() const override { return cfg.n_max; }
};

struct common_speculative {
    std::vector<common_speculative_config> configs; // resolved stage config for each implementation
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states
    common_speculative_checkpoint checkpoint;
    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
    std::unique_ptr<spec_tuner> tuner;
    int last_n_drafted = 0;
    int64_t t_step_start_us = 0;
    bool last_step_target_only = false;

    // --spec-draft-policy: chooses between a leading self-spec (lookup) stage and the model stage that follows it from learned
    // acceptance and measured round cost. Round bookkeeping is filled by common_speculative_draft, consumed by _accept.
    std::unique_ptr<common_draft_policy> policy;
    bool policy_round_valid  = false;
    bool policy_round_lookup = false;
    int  policy_round_t      = 1;
    int  policy_round_match  = 0;

    // --- unsloth/master's per-sequence-pool fields, UNION-merged in (renamed impls ->
    //     impls_seq to avoid colliding with the fork's `impls` above -- different element
    //     type, common_speculative_impl vs common_speculative_state). Populated only by
    //     common_speculative_init(params, n_seq); empty (default-constructed) when this
    //     object was built via the fork's own common_speculative_init(params, ctx_tgt). ---
    common_speculative_draft_params_vec dparams;
    std::vector<std::unique_ptr<common_speculative_impl>> impls_seq;
    std::vector<common_speculative_impl *> impl_last;
    float draft_temperature = 0.0f;
    uint32_t draft_seed = LLAMA_DEFAULT_SEED;

    // --spec-draft-policy for the per-sequence pool: one policy per sequence (each slot learns its own acceptance and round
    // cost), with that sequence's round bookkeeping. Empty when the policy is off or the chain has no lookup->model shape.
    struct pool_policy_round {
        bool    valid    = false;
        bool    lookup   = false;
        int     t        = 1;
        int     match    = 0;
        int64_t t_start_us = 0;
    };
    std::vector<std::unique_ptr<common_draft_policy>> policy_seq;
    std::vector<pool_policy_round>                    policy_round_seq;

    // synthetic acceptance (upstream), empty = off. LAST member: the fork initialises this struct positionally.
    std::vector<double> synth_probs;
};


// ADAPTED from upstream: theirs took a `common_params_speculative_ngram_map` sub-struct
// (config.params.ngram_simple / .ngram_map_k) that does not exist on the staged common.h's
// common_params_speculative (which has flat ngram_size_n/ngram_size_m/ngram_min_hits fields
// instead, plus a runtime std::shared_ptr<common_ngram_mod> ngram_mod -- not a config struct).
// Rewritten to build the common_ngram_map straight from those flat fields.
static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative & params) {
    uint16_t size_key   = params.ngram_size_n;
    uint16_t size_value = params.ngram_size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}


const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}


std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        // FIX (S69-MERGE-COMMON-CLUSTER): common_speculative_type_from_name (singular, kept
        // from ours) normalizes '-' to '_' before the map lookup; this plural sibling did not
        // (theirs' original had no such normalization, matching its own hyphen-keyed map --
        // but the map kept here is ours', underscore-keyed). Without this, a hyphenated
        // --spec-type value (e.g. "ngram-simple", matching every doc example and every other
        // type string in this codebase) would fail to match ours' "ngram_simple" key. Applied
        // the same normalization here for consistency between the two lookup functions.
        std::string normalized = name;
        std::replace(normalized.begin(), normalized.end(), '-', '_');

        auto type = common_speculative_type_from_name_map.find(normalized);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}


std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}


static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}


struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
};


common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}


common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}



// ADAPTED from upstream: theirs switched over `spec->types` and read
// `spec->draft.n_max` / `spec->ngram_simple.size_m` / `.ngram_map_k.size_m` /
// `.ngram_map_k4v.size_m` / `.ngram_mod.n_max` -- none of which exist on the staged header's
// common_params_speculative. The header already carries a generalized equivalent
// (get_max_stage_n_max(), over the resolved multi-stage chain, common.cpp) -- delegate to it.
int32_t common_speculative_n_max(const common_params_speculative * spec) {
    if (spec == nullptr) {
        return 0;
    }
    return spec->get_max_stage_n_max();
}


// ADAPTED from upstream: theirs read params.speculative.draft.* (a
// common_params_speculative_draft sub-struct) and params.speculative.types (a vector) --
// neither exists as a member on the staged common.h's common_params_speculative (the
// draft sub-struct type is DECLARED there but never wired in as a member; see the
// progress-file escalation). Rewritten against the fork's actual flat fields, which cover
// the same data.
common_params common_base_params_to_speculative(const common_params & params) {
    common_params result = params;
    // upstream's params.speculative.draft.* spelling, folded on this copy (a caller may set it after common_params_parse)
    common_params_speculative_fold_draft(result.speculative);
    const common_params_speculative & spec = result.speculative;

    const bool has_draft = spec.has_dft();

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        result.model.path        = spec.dft_path();
        result.n_gpu_layers  = spec.n_gpu_layers;

        // -devd (upstream: params_spec.devices); empty inherits the target's devices
        if (!spec.devices.empty()) {
            result.devices = spec.devices;
        }
        // upstream 415e909d8: a draft pinned to ONE device does not need the meta (tensor-split) wrapper an inherited -sm tensor
        // would give it. Upstream counts the non-null entries of its parsed device vector; here -devd is still the comma list
        // (parsed at model load), so count its non-empty names, "none" counting as no device as upstream's parse makes it.
        {
            int n_devs = 0;
            size_t pos = 0;
            while (pos <= spec.devices.size()) {
                const size_t end = std::min(spec.devices.find(',', pos), spec.devices.size());
                const std::string name = spec.devices.substr(pos, end - pos);
                if (!name.empty() && name != "none") {
                    ++n_devs;
                }
                pos = end + 1;
            }
            if (n_devs == 1) {
                result.split_mode = LLAMA_SPLIT_MODE_LAYER;
            }
        }
        // --spec-draft-override-tensor / --spec-draft-cpu-moe (upstream: params_spec.tensor_buft_overrides); an empty list keeps the
        // target's list, which is what the draft got before this was applied
        if (!spec.tensor_buft_overrides.empty()) {
            result.tensor_buft_overrides = spec.tensor_buft_overrides;
        }
        // --spec-draft-threads writes cpuparams.n_threads, --draft-params the flat n_threads; same precedence as the stage path below
        const int32_t nt  = spec.n_threads       > 0 ? spec.n_threads       : spec.cpuparams.n_threads;
        const int32_t ntb = spec.n_threads_batch > 0 ? spec.n_threads_batch : spec.cpuparams_batch.n_threads;
        if (nt > 0) {
            result.cpuparams.n_threads       = nt;
            result.cpuparams_batch.n_threads = ntb > 0 ? ntb : nt;
        }
        // upstream draft.lazy_mode (no flat field; kept in `draft` by the fold)
        if (spec.draft.lazy_mode != LLAMA_LAZY_MODE_AUTO) {
            result.lazy_mode = spec.draft.lazy_mode;
        }
    }

    // FIXED 2026-09-14. These two were copied UNCONDITIONALLY, and -- the part that made it bite
    // every run -- from OUTSIDE the `if (has_draft)` block above. common_params_speculative's
    // cache_type_k/_v default to "" (common.h:402-403), where "" is an INHERIT sentinel: the
    // sibling at :2859 in this same file does exactly the check below. The consumer of the result,
    // kv_cache_type_from_str, has no notion of inherit and threw
    //     Invalid cache type:
    // on every llama-cli / llama-server load, draft model or not. Two functions in one file
    // disagreed about what "" means; :2859 had it right, so this now matches it. If that sibling
    // ever changes, this must change with it -- they are the same rule written twice.
    result.cache_type_k  = spec.cache_type_k.empty()
                             ? params.cache_type_k : spec.cache_type_k;
    result.cache_type_v  = spec.cache_type_v.empty()
                             ? params.cache_type_v : spec.cache_type_v;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // --spec-draft-n-ubatch (upstream: params_spec.n_ubatch); 0 inherits the target's. Parsed since the flag was added, applied from 2026-11-05.
    if (spec.n_ubatch > 0) {
        result.n_ubatch = spec.n_ubatch;
    }

    // the MoE cache is only used by the target context (upstream)
    result.moe_cache_size = 0;
    result.moe_cache_auto = false; // --moe-cache-mib auto sizes the target only

    // Classifier-Free Guidance (row S69-ENGINE-MERGE-KOBOLDCPP-CFG-FOLLOWUP-2026-09-26-01):
    // a guidance sequence only ever lives on the MAIN context (ctx_tgt) -- server-context.cpp's
    // cfg_seq_free pool is sized once, for ctx_tgt only, and no draft-context seq id is ever
    // claimed. `result` is a full copy of `params` above, so without this reset the draft
    // context's own n_seq_max (common_context_params_to_llama: n_parallel + n_cfg_guidance_slots)
    // would inflate to match the main context's, while n_outputs_max just above stays fixed at
    // params.n_parallel (deliberately, for the draft's own one-output-per-slot greedy sampling) --
    // src/llama-context.cpp:2847,3022 then asserts n_seq_max <= n_outputs_max and the draft
    // context fails to construct. MEASURED: this was the exact cause of
    // test_cfg_rejected_with_speculative_decoding's server dying at startup (rc=3, zero log
    // output) with --cfg-guidance-slots 1 --parallel 1 -md ...: n_seq_max=2 > n_outputs_max=1.
    result.n_cfg_guidance_slots = 0;

    // dflash/dspark decode the whole noise block in a single pass and sample every block
    // position on the backend. TRANSLATED from theirs' `params.speculative.types` scan to
    // the fork's has_dflash_family_stage() over the resolved stage chain (common.h, already
    // carried into common.cpp).
    if (spec.has_dflash_family_stage()) {
        const int32_t per_seq = std::max(1, spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        result.n_outputs_max_per_seq = per_seq;
    }

    return result;
}


// ADAPTED from upstream: theirs read params.speculative.draft.mparams.path (the missing
// .draft sub-struct) and params.model.path (header keeps the fork's FLAT params.model
// string, not a nested .model.path -- see common.h's own sibling comment on that decision)
// and params.speculative.types (missing vector). Rewritten against the fork's real flat
// fields; also fixes what reads as an upstream bug carried in the original -- it loaded
// params.model.path (the TARGET model) inside the has_draft branch instead of the draft
// model path it had just logged and named model_path.
common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp  = params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (spec_mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    // the draft context holds as many tokens per sequence as the target context
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.dft_path();
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        // a draft head can leave out the embeddings and lm head and use the target's
        mparams.model_shared = model_tgt;

        llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        // a DFlash / DSpark draft needs the same context setup the single-context engine gives it (common_speculative_init,
        // below): the target's IO tensors shared into the draft, and a query block reserved, which is what makes
        // llama_get_dflash_visible_cross_ctx() non-zero for common_speculative_impl_dflash_pool
        // a LiLiCorr draft is the src/models DFlash class: it reads the target's embeddings + head through ctx_other and needs
        // none of the fork DFlash engine's context setup below (common_speculative_impl_lilicorr)
        char lilicorr_meta[32] = {};
        const bool is_lilicorr = llama_model_meta_val_str(model_dft, "lilicorr.candidate_topk", lilicorr_meta, sizeof(lilicorr_meta)) >= 0;
        if (params.speculative.has_dflash_family_stage() && !is_lilicorr) {
            if (!llama_model_share_dflash_io_tensors(model_dft, model_tgt)) {
                LOG_ERR("%s: failed to share target IO tensors with the DFlash draft model\n", __func__);
                return;
            }
            const int32_t block = llama_model_dflash_block_size(model_dft);
            int32_t query_capacity = block;
            if (llama_model_is_deepseek4(model_tgt)) {
                for (const auto & stage : params.speculative.get_resolved_stages()) {
                    if (stage.type == COMMON_SPECULATIVE_TYPE_DSPARK) {
                        query_capacity = std::max(query_capacity, std::max(1, params.speculative.with_stage_overrides(stage).n_max));
                    }
                }
            }
            if (block <= 0 || (int32_t) cparams.n_ctx <= query_capacity) {
                LOG_ERR("%s: invalid DFlash draft setup (block_size=%d, n_ctx=%u, query_capacity=%d)\n", __func__, block,
                        cparams.n_ctx, query_capacity);
                return;
            }
            cparams.dflash_query_capacity = query_capacity;
        }

        llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}


// The draft-model and MTP pool drafters' sampler: TOP_K + DIST (DIST is what fills candidate probabilities in this tree).
static common_params_sampling spec_default_draft_sparams() {
    common_params_sampling sp;
    sp.top_k = 10;
    sp.samplers_sequence = {
        llama_sampler_type::TOP_K,
        llama_sampler_type::DIST,
    };
    return sp;
}

// Plan gap G2 (upstream 1fb7ef3e3 spec_retune, 977c48a25 common/speculative.cpp:37-72; SOURCE-READ there). Under
// --spec-draft-sample a drafter samples at the TARGET request's temperature: the rejection verifier accepts with min(1, p/q),
// and a q drawn at another temperature lowers sum min(p, q). A sequence's sampler is rebuilt only when the request's
// (temp, seed) changes; a temperature-0 request gets the default chain back (upstream's spec_untune). NAN = never tuned.
static void spec_draft_retune(std::vector<common_sampler_ptr> & smpls, std::vector<float> & tuned_temp,
                              std::vector<uint32_t> & tuned_seed, const llama_model * model, llama_seq_id seq_id,
                              float temp, uint32_t seed) {
    const float want = temp > 0.0f ? temp : 0.0f;
    if (tuned_temp[seq_id] == want && tuned_seed[seq_id] == seed) {
        return;
    }
    tuned_temp[seq_id] = want;
    tuned_seed[seq_id] = seed;
    common_params_sampling sp = spec_default_draft_sparams();
    if (want > 0.0f) {
        sp.temp = want;
        sp.samplers_sequence = { llama_sampler_type::TOP_K, llama_sampler_type::TEMPERATURE, llama_sampler_type::DIST };
        // explicit, or the default reseeds at random; mixed so the draft stream differs from the target's
        sp.seed = seed == LLAMA_DEFAULT_SEED ? seed : seed ^ 0x85ebca6bu;
    }
    smpls[seq_id].reset(common_sampler_init(model, sp));
}

// PORTED 2026-09-26 from upstream llama.cpp b9590 common/speculative.cpp:169-376
// (common_speculative_impl_draft_simple), so llama-server's per-sequence speculator can draft with a
// separate draft model -- without it a `draft` stage built no impl and the server discarded the draft
// model it had loaded (row S69-FORK-SERVER-PER-SEQ-SPECULATOR-HAS-NO-DRAFT-IMPL-2026-09-26-01).
// Differences, each forced by this tree, none by preference:
//   * config comes from the fork's FLAT fields (n_max / n_min / p_min, ctx_tgt / ctx_dft) -- upstream's
//     .draft sub-struct is declared here but was never made a member;
//   * the draft sampler chain is TOP_K + DIST, as the fork's own single-context draft state uses: this
//     tree's sampler only fills candidate probabilities with DIST in the chain, and p_min reads them;
//   * common_sampler_accept takes the context in this fork.
// Like upstream, the draft context mirrors the target by decoding every target batch in process(), and
// drafts for all sequences at once from (id_last, n_past).
struct common_speculative_impl_draft : public common_speculative_impl {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    int32_t n_max;
    int32_t n_min;
    float   p_min;
    bool    draft_sample; // --spec-draft-sample: draft the sampler's own draw from q (not the argmax) and record q
    bool    draft_grammar; // --spec-draft-grammar: draft under a copy of the target's grammar (with draft_sample q is the masked dist)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // plan gap G2: see spec_draft_retune above
    std::vector<float>    tuned_temp;
    std::vector<uint32_t> tuned_seed;

    // ---- --spec-async: draft || verify overlap (see common_speculative_async_start in speculative.h) ----
    //
    // THREAD-SAFETY (cited, not assumed). The worker only ever touches ctx_dft, the per-seq samplers `smpls` and `wbatch`, while the main thread
    // runs llama_decode(ctx_tgt). Two llama_contexts share no mutable state in this tree: the per-context KV cache / scheduler / output buffers
    // are context members; ggml's one global mutable (the ggml_cpu_init fp16/quant tables) is initialised under ggml_critical_section at model
    // load (ggml/src/ggml-cpu/ggml-cpu.c:5488-5577), long before the worker exists; include/llama.h:2954 says the one global the API owns, the
    // logger, is NOT thread safe, so the worker never calls llama_log_set (it only logs through common_log, which locks). A threadpool is
    // shared between contexts only when llama_attach_threadpool is called with the same pool (common/common.cpp:3415); the draft context
    // gets its own (none) here. The main thread never calls into ctx_dft or the samplers while the worker runs: every entry point of
    // this impl (process, draft, accept-state, destructor) first joins the worker. Race-checked by tests/test-spec-async.cpp.
    // A draft context on the SAME GPU as the target makes the two decodes contend for the device (PEARL says so, arXiv 2408.11850 sec.
    // 4); the intended configuration is -devd none -ngld 0 (draft on the CPU), and the A/B numbers say whether the other case pays.
    struct async_seq {
        bool        pending = false;  // handed to the worker this round
        bool        step0   = false;  // [id_last, draft] evaluated on ctx_dft by the worker
        bool        valid   = false;  // lookahead usable by the next draft()
        size_t      skipped = 0;      // tokens of the verify batch process() has skipped so far
        llama_pos   pos0    = 0;
        llama_token id_last = 0;
        llama_tokens draft;           // the draft being verified
        llama_tokens la;              // lookahead: c1 (predicted after the last draft token), c2, ...
        // diagnostics (S69-SPECASYNC-DIAG-COUNTERS-2026-11-06-01)
        bool        counted = false;  // handed to the worker and not yet retired into an outcome counter
        bool        gate_la = false;  // this round's worker was allowed to extend the lookahead (w_la at async_start)
        int         n_c1_top = 0;     // the draft's top candidates at c1 (sorted by probability), for the rank counter
        llama_token c1_top[5] = {0, 0, 0, 0, 0};
    };

    bool                   async_on = false;
    int32_t                async_wait_ms = 0;
    bool                   async_informed = false;
    std::vector<async_seq> as;
    // informed join: main-thread-only per-seq notes (never touched by the worker) + the verdict the worker polls
    std::vector<char>      as_greedy;     // the request is greedy (hint from the server)
    std::vector<char>      as_waited;     // the informed join let the worker keep going for this seq
    static constexpr int32_t VD_UNDECIDED = -2; // verify outcome not published yet: the worker keeps going, as it always did
    static constexpr int32_t VD_USELESS   = -1; // the draft was not fully accepted: nobody will use this lookahead
    static constexpr int32_t VD_UNKNOWN   = -3; // main-thread only: the verify rows could not be read
    std::vector<std::atomic<int32_t>> w_verdict; // >= 0: the target's own token after a full accept (the lookahead helps iff c1 equals it)
    int64_t                t_round0_us = 0; // async_start time, for the verify duration EMA
    // worker timing, written by the worker before w_done, read by the main thread after join()
    uint64_t               w_s0_us = 0, w_d_us = 0, w_d_n = 0;
    std::thread            worker;
    std::atomic<bool>      w_stop{false};
    std::atomic<bool>      w_done{false};
    bool                   w_la = true;      // this round's worker extends the lookahead beyond step 0
    // The lookahead only pays when the target accepts the WHOLE draft and its own next token equals c1: P(useful) = p_full * p_c1. When that is
    // low (a weak draft: MEASURED 0.6B vs 8B, 20% per-token acceptance => P(full 8) ~ 0), the worker does step 0 only, so the main thread never
    // waits for a lookahead token that nobody will use. Both are EMAs updated in draft(); every 8th round looks ahead anyway so they keep moving.
    double                 ema_full = 0.5;
    double                 ema_c1   = 0.9;
    static constexpr double LA_MIN_USEFUL = 0.10;
    llama_batch            wbatch{};
    common_speculative_async_stats ast;

    common_speculative_impl_draft(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT, n_seq)
        , ctx_tgt(params.ctx_tgt)
        , ctx_dft(params.ctx_dft)
        , n_max(params.n_max)
        , n_min(params.n_min)
        , p_min(params.p_min_explicit ? params.p_min : 0.0f) // see common_params_speculative::p_min_explicit
        , draft_sample(params.draft_sample)
        , draft_grammar(params.draft_grammar)
    {
        if (params.async) {
            if (params.draft_sample || params.draft_grammar) {
                LOG_WRN("%s: --spec-async needs the greedy, unconstrained draft; --spec-draft-sample / --spec-draft-grammar is on, async stays OFF\n", __func__);
            } else if (common_context_can_seq_rm(ctx_dft) != COMMON_CONTEXT_SEQ_RM_TYPE_PART) {
                LOG_WRN("%s: --spec-async needs a draft context that can roll back partial sequences; async stays OFF\n", __func__);
            } else {
                async_on      = true;
                async_wait_ms = params.async_wait_ms;
                async_informed = params.async_informed;
            }
        }
        GGML_ASSERT(ctx_tgt != nullptr && ctx_dft != nullptr);

        LOG_INF("%s: adding speculative implementation 'draft'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f, n_seq=%u, draft_sample=%d, draft_grammar=%d\n", __func__, n_max, n_min,
                p_min, n_seq, (int) draft_sample, (int) draft_grammar);
        if (params.draft_grammar && params.draft_sample) {
            // grammar-mask renormalisation (row S69-ENGREV-UPSTREAM-MERGE-DRAFT-SAMPLE-EQUIVALENCE-TEST-2026-11-04-01 C2): the draft
            // draws with the constraint applied FIRST (common_sampler_sample(..., grammar_first = true) below), so
            // common_sampler_sampling_dist records q = softmax over the grammar-masked logits, renormalised -- the distribution the
            // token was really drawn from -- and the rejection verifier stays exact. It used to ignore --spec-draft-grammar here.
            LOG_INF("%s: --spec-draft-grammar with --spec-draft-sample: q is the grammar-masked, renormalised draft distribution\n", __func__);
        }

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), spec_default_draft_sparams()));
        }
        tuned_temp.assign(n_seq, NAN);
        tuned_seed.assign(n_seq, LLAMA_DEFAULT_SEED);

        if (!common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft))) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);
            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %u != %u\n", __func__, n_seq, llama_n_seq_max(ctx_dft));
            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }

        if (async_on) {
            wbatch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
            as.resize(n_seq);
            as_greedy.assign(n_seq, 0);
            as_waited.assign(n_seq, 0);
            w_verdict = std::vector<std::atomic<int32_t>>(n_seq);
            for (auto & v : w_verdict) { v.store(VD_UNDECIDED, std::memory_order_relaxed); }
            LOG_INF("%s: --spec-async ON: draft-context evaluation + lookahead overlap the target verify (wait_ms=%d, informed_join=%d)\n", __func__,
                    async_wait_ms, (int) async_informed);
        }
    }

    ~common_speculative_impl_draft() override {
        async_join(false);
        if (async_on) {
            llama_batch_free(wbatch);
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop -- the draft context follows the target through process()
    }

    // ---- --spec-async: control ----

    // exponential moving average (alpha 0.2) of a microsecond measurement; 0 = never measured
    static void ema_us(uint64_t & e, double x) {
        e = e == 0 ? (uint64_t) std::max(1.0, x) : (uint64_t) std::max(1.0, (double) e + 0.2 * (x - (double) e));
    }

    // fold the worker's and the verify's timing of the round that just ended into the EMAs (the worker is joined, so w_* are stable)
    void async_fold_timing(int64_t t_enter_us, bool cut) {
        if (w_s0_us > 0) { ema_us(ast.ema_s0_us, (double) w_s0_us); }
        if (w_d_n > 0)   { ema_us(ast.ema_d_us, (double) w_d_us / (double) w_d_n); }
        if (cut && t_round0_us > 0) { ema_us(ast.ema_v_us, (double) (t_enter_us - t_round0_us)); }
        w_s0_us = w_d_us = w_d_n = 0;
        t_round0_us = 0;
    }

    enum async_out { OUT_HIT, OUT_NOLA, OUT_LAEMPTY, OUT_NOTFULL, OUT_C1MISS, OUT_TRUNC, OUT_GUARD, OUT_NODRAFT, OUT_FALLBACK, OUT_CANCEL, OUT_DROPPED };

    // end of a sequence's round: book exactly one outcome class (once) and forget the state. OUT_HIT is booked at the hit site (n_hit).
    void async_retire(llama_seq_id s, int out) {
        auto & a = as[s];
        if (a.counted) {
            switch (out) {
                case OUT_NOLA:     ast.n_o_nola++;     break;
                case OUT_LAEMPTY:  ast.n_o_laempty++;  break;
                case OUT_NOTFULL:  ast.n_o_notfull++;  break;
                case OUT_C1MISS:   ast.n_o_c1miss++;   break;
                case OUT_TRUNC:    ast.n_o_trunc++;    break;
                case OUT_GUARD:    ast.n_o_guard++;    break;
                case OUT_NODRAFT:  ast.n_o_nodraft++;  break;
                case OUT_FALLBACK: ast.n_o_fallback++; break;
                case OUT_CANCEL:   ast.n_o_cancel++;   break;
                case OUT_DROPPED:  ast.n_o_dropped++;  break;
                default: break;
            }
        }
        a = async_seq{};
    }

    // stop and join the worker. cut == true: the verify is done; wait_us < 0 = the legacy rule (async_wait_ms), >= 0 = wait that long first.
    // t_enter_us = when the caller started its join work (verify duration for the EMA); 0 = now.
    void async_join(bool cut, int64_t wait_us = -1, int64_t t_enter_us = 0) {
        if (!worker.joinable()) {
            return;
        }
        const int64_t t0 = ggml_time_us();
        if (wait_us < 0) {
            wait_us = cut && async_wait_ms > 0 ? (int64_t) async_wait_ms * 1000 : 0;
        }
        if (wait_us > 0) {
            const int64_t t_end = t0 + wait_us;
            while (!w_done.load(std::memory_order_acquire) && ggml_time_us() < t_end) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
        w_stop.store(true, std::memory_order_release);
        worker.join();
        const int64_t t1 = ggml_time_us();
        ast.t_wait_us += (uint64_t) (t1 - t0);
        async_fold_timing(t_enter_us > 0 ? t_enter_us : t0, cut);
    }

    // ---- --spec-async-informed-join ----
    // What the target will do with this seq's draft, read from the verify rows that are already on the host. VD_USELESS: it rejects part of
    // the draft; >= 0: it accepts the whole draft and takes this token next (the lookahead helps iff c1 equals it); VD_UNKNOWN: cannot tell.
    // Greedy requests only (the caller checks): argmax of the row IS the token the target takes. Reads as[s].draft/pos0/id_last, which the
    // worker never writes.
    int32_t async_verdict(llama_seq_id s, const llama_batch & b) {
        const auto & a = as[s];
        const size_t n = a.draft.size();
        if (b.logits == nullptr || b.token == nullptr || llama_output_vocab_subset_used(ctx_tgt) > 0) {
            return VD_UNKNOWN; // last-token-only batch / a subset vocabulary changes the row layout
        }
        std::vector<int32_t> row(n + 1, -1);
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            if (b.n_seq_id[i] != 1 || b.seq_id[i][0] != s) {
                continue;
            }
            const llama_pos off = b.pos[i] - a.pos0;
            if (off < 0 || (size_t) off > n) {
                continue;
            }
            const llama_token expect = off == 0 ? a.id_last : a.draft[(size_t) off - 1];
            if (b.token[i] == expect && b.logits[i]) {
                row[(size_t) off] = i;
            }
        }
        for (const int32_t r : row) {
            if (r < 0) {
                return VD_UNKNOWN; // part of the verify batch is in another view, or a row has no logits
            }
        }
        const int32_t nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_tgt)));
        for (size_t j = 0; j <= n; ++j) {
            llama_token top = LLAMA_TOKEN_NULL;
            const llama_token st = llama_get_sampled_token_ith(ctx_tgt, row[j]); // backend sampling already chose (greedy request: the argmax)
            if (st != LLAMA_TOKEN_NULL) {
                top = st;
            } else {
                if (llama_get_sampled_logits_ith(ctx_tgt, row[j]) != nullptr) {
                    return VD_UNKNOWN; // a backend-sampled candidate subset, not the full row
                }
                const float * l = llama_get_logits_ith(ctx_tgt, row[j]);
                if (l == nullptr) {
                    return VD_UNKNOWN;
                }
                int32_t best = 0;
                for (int32_t i = 1; i < nv; ++i) {
                    if (l[i] > l[best]) {
                        best = i;
                    }
                }
                top = best;
            }
            if (j < n) {
                if (top != a.draft[j]) {
                    return VD_USELESS;
                }
            } else {
                return top;
            }
        }
        return VD_UNKNOWN;
    }

    // The join of process(): decide from the verify result how long the worker may keep going.
    void async_join_informed(const llama_batch & b) {
        if (!worker.joinable()) {
            return;
        }
        const int64_t t_enter = ggml_time_us();
        std::vector<int32_t> vd(n_seq, VD_UNKNOWN);
        bool any_seq = false, all_known = true, any_wanted = false;
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq && all_known; ++s) {
            if (!as[s].counted) {
                continue;
            }
            any_seq = true;
            vd[s] = as_greedy[s] ? async_verdict(s, b) : VD_UNKNOWN;
            all_known = vd[s] != VD_UNKNOWN;
            any_wanted = any_wanted || vd[s] >= 0;
        }
        if (!any_seq || !all_known) {
            ast.n_inf_legacy++;
            async_join(true, -1, t_enter); // the wait_ms rule
            return;
        }
        ast.n_inf_rounds++;
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (as[s].counted) {
                w_verdict[s].store(vd[s], std::memory_order_release); // the worker drops seqs whose lookahead nobody will use
                as_waited[s] = vd[s] >= 0 && as[s].gate_la;
            }
        }
        if (!any_wanted) {
            ast.n_inf_stop++;
            async_join(true, 0, t_enter);
            return;
        }
        // wait for the lookahead, bounded: the time the worker needs for step 0 + the full lookahead, doubled for jitter, counted from the
        // round start; --spec-async-wait-ms (> 0) caps the wait after the verify. Nothing measured yet: 500 ms ceiling.
        const size_t la_max = (size_t) std::max(1, n_max) + 1;
        int64_t bound_us = 500000;
        if (ast.ema_d_us > 0) {
            bound_us = (int64_t) (2.0 * ((double) ast.ema_s0_us + (double) la_max * (double) ast.ema_d_us)) + 2000;
        }
        int64_t wait_us = std::max<int64_t>(1000, t_round0_us + bound_us - t_enter);
        if (async_wait_ms > 0) {
            wait_us = std::min<int64_t>(wait_us, (int64_t) async_wait_ms * 1000);
        }
        ast.n_inf_wait++;
        const int64_t t0 = ggml_time_us();
        async_join(true, wait_us, t_enter);
        ast.t_inf_wait_us += (uint64_t) (ggml_time_us() - t0);
    }

    void async_cancel() override {
        async_join(false);
        // nothing was consumed: trim what the worker evaluated past pos0 and forget every lookahead
        for (llama_seq_id s = 0; s < (llama_seq_id) as.size(); ++s) {
            if (as[s].pending && as[s].step0) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), s, as[s].pos0, -1);
            }
            async_retire(s, OUT_CANCEL);
        }
    }

    bool async_stats(common_speculative_async_stats & out) const override {
        if (!async_on) {
            return false;
        }
        out = ast;
        out.n_open = 0;
        for (const auto & a : as) {
            out.n_open += a.counted ? 1 : 0;
        }
        return true;
    }

    void async_start(const std::vector<common_speculative_async_req> & reqs) override {
        if (!async_on) {
            return;
        }
        async_join(false);
        const int32_t n_batch = (int32_t) llama_n_batch(ctx_dft);
        int32_t n_tok = 0;
        bool any = false;
        for (llama_seq_id s = 0; s < (llama_seq_id) as.size(); ++s) {
            async_retire(s, OUT_DROPPED); // a state that is still here was never consumed by draft()
            as_greedy[s]  = 0;
            as_waited[s]  = 0;
            w_verdict[s].store(VD_UNDECIDED, std::memory_order_relaxed);
        }
        for (const auto & r : reqs) {
            if (r.seq_id < 0 || r.seq_id >= (llama_seq_id) n_seq || r.draft == nullptr || r.draft->empty()) {
                continue;
            }
            const int32_t need = (int32_t) r.draft->size() + 1;
            if (n_tok + need > n_batch) {
                continue; // does not fit the draft batch: this seq takes the serial path
            }
            n_tok += need;
            auto & a  = as[r.seq_id];
            a.pending = true;
            a.counted = true;
            a.pos0    = r.pos0;
            a.id_last = r.id_last;
            a.draft   = *r.draft;
            as_greedy[r.seq_id] = r.greedy ? 1 : 0;
            ast.n_seq_rounds++;
            any = true;
        }
        if (!any) {
            return;
        }
        ast.n_rounds++;
        w_la = ema_full * ema_c1 >= LA_MIN_USEFUL || ast.n_rounds % 8 == 0;
        if (w_la) { ast.n_la_rounds++; }
        for (auto & a : as) {
            a.gate_la = a.counted && w_la;
        }
        w_stop.store(false, std::memory_order_release);
        w_done.store(false, std::memory_order_release);
        w_s0_us = w_d_us = w_d_n = 0;
        t_round0_us = ggml_time_us();
        worker = std::thread([this]() { worker_main(); });
    }

    // Worker body. Step 0 evaluates [id_last, draft] for every pending seq in ONE batch (exactly what process() would do after the verify) and
    // samples c1 from the last row; then it drafts c2.. one batched step at a time until the lookahead is long enough or stop is set.
    // The stop flag is only looked at AFTER step 0, so step 0 always completes.
    void worker_main() {
        const int64_t t0 = ggml_time_us();
        try {
            std::vector<llama_seq_id> act; // seqs still extending their lookahead
            // batch index of each act seq's output token. llama_get_logits_ith / common_sampler_sample take the BATCH position, not an
            // output-row counter: step 0 puts [id_last, draft...] per seq with logits only on the last draft token, so a plain 0,1,2..
            // counter named id_last (logits off) -> "get_logits_ith: invalid logits id 0, reason: batch.logits[0] != true" and a null
            // logits pointer -> 0xC0000005 in test-spec-async (MEASURED 2026-11-06, specasync-test3-1106 stories15M, exit 3221225477).
            std::vector<int32_t> act_idx;
            llama_context * ctx = ctx_dft;

            common_batch_clear(wbatch);
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                auto & a = as[s];
                if (!a.pending) {
                    continue;
                }
                // whatever the serial path left past pos0 (the server trims first; this makes the worker order-independent)
                if (!llama_memory_seq_rm(llama_get_memory(ctx), s, a.pos0, -1)) {
                    a.pending = false;
                    continue;
                }
                common_batch_add(wbatch, a.id_last, a.pos0, { s }, false);
                for (size_t j = 0; j < a.draft.size(); ++j) {
                    common_batch_add(wbatch, a.draft[j], a.pos0 + 1 + (llama_pos) j, { s }, j + 1 == a.draft.size());
                }
                act.push_back(s);
                act_idx.push_back(wbatch.n_tokens - 1); // the last draft token: the only one of this seq with logits on
            }
            const int64_t t_s0 = ggml_time_us();
            const bool s0_failed = !act.empty() && llama_decode(ctx, wbatch) != 0;
            if (!act.empty() && !s0_failed) {
                w_s0_us = (uint64_t) (ggml_time_us() - t_s0);
            }
            if (s0_failed) {
                // leave step0 false: process() falls back to evaluating the tokens itself, from a clean position
                for (auto s : act) {
                    llama_memory_seq_rm(llama_get_memory(ctx), s, as[s].pos0, -1);
                    as[s].pending = false;
                }
                act.clear();
                act_idx.clear();
            }
            for (auto s : act) {
                as[s].step0 = true;
            }
            if (!w_la) {
                act.clear(); // step 0 only: nothing is sampled, nothing for the main thread to wait for
                act_idx.clear();
            }

            const size_t la_max = (size_t) std::max(1, n_max) + 1; // c1 is the verify's own next token; the next draft proper is c2..
            std::vector<llama_seq_id> nxt;
            std::vector<int32_t>      nxt_idx;
            bool first = true;
            while (!act.empty()) {
                // sample one token per active seq at the batch position of its output token in the batch just decoded
                GGML_ASSERT(act_idx.size() == act.size());
                common_batch_clear(wbatch);
                nxt.clear();
                nxt_idx.clear();
                for (size_t k = 0; k < act.size(); ++k) {
                    const llama_seq_id s = act[k];
                    auto & a    = as[s];
                    auto * smpl = smpls[s].get();
                    if (first) {
                        common_sampler_reset(smpl);
                    }
                    common_sampler_sample(smpl, ctx, act_idx[k], true);
                    const auto * cur_p = common_sampler_get_candidates(smpl, true);
                    const llama_token id = cur_p->data[0].id;
                    if (a.la.empty() && a.n_c1_top == 0) {
                        // O5 gate: the draft's own top candidates at c1, to rank the token the target takes (diagnostic only)
                        const size_t nc = std::min<size_t>(5, cur_p->size);
                        for (size_t c = 0; c < nc; ++c) {
                            a.c1_top[c] = cur_p->data[c].id;
                        }
                        a.n_c1_top = (int) nc;
                    }
                    if (common_speculative_top_prob(cur_p) < p_min) {
                        continue; // below the impl p_min: the lookahead ends here (same gate as the serial draft)
                    }
                    common_sampler_accept(smpl, ctx, id, true);
                    a.la.push_back(id);
                    ast.n_la_tokens++;
                    if (a.la.size() >= la_max) {
                        continue;
                    }
                    if (async_informed) {
                        // the join published what the verify decided: drop this seq when the draft was not fully accepted or c1 is not the
                        // token the target took (UNDECIDED = the verify is still running: keep drafting, as the legacy worker does)
                        const int32_t vd = w_verdict[s].load(std::memory_order_acquire);
                        if (vd == VD_USELESS || (vd >= 0 && a.la[0] != vd)) {
                            continue;
                        }
                    }
                    common_batch_add(wbatch, id, a.pos0 + (llama_pos) a.draft.size() + (llama_pos) a.la.size(), { s }, true);
                    nxt.push_back(s);
                    nxt_idx.push_back(wbatch.n_tokens - 1);
                }
                first = false;
                act.swap(nxt);
                act_idx.swap(nxt_idx);
                if (act.empty() || w_stop.load(std::memory_order_acquire)) {
                    break;
                }
                const int64_t t_d = ggml_time_us();
                if (llama_decode(ctx, wbatch) != 0) {
                    break;
                }
                w_d_us += (uint64_t) (ggml_time_us() - t_d);
                w_d_n++;
            }
        } catch (const std::exception & e) {
            LOG_ERR("%s: --spec-async worker failed: %s\n", __func__, e.what());
        }
        ast.t_worker_us += (uint64_t) (ggml_time_us() - t0);
        w_done.store(true, std::memory_order_release);
    }

    bool process(const llama_batch & b) override {
        if (async_on) {
            if (async_informed) {
                async_join_informed(b);
            } else {
                async_join(true);
            }
        }

        // tokens the worker already evaluated (same seq, same position, same token) are skipped; everything else (prompt chunks of other
        // slots, a mismatch) is decoded as before
        bool any_step0 = false;
        if (async_on) {
            for (llama_seq_id s = 0; s < (llama_seq_id) as.size(); ++s) {
                if (as[s].pending && !as[s].step0) {
                    ast.n_fallback++; // the worker never evaluated this seq: it takes the normal decode below
                    async_retire(s, OUT_FALLBACK);
                }
                any_step0 = any_step0 || as[s].pending;
            }
        }
        if (!any_step0 || b.embd != nullptr) {
            const int ret = llama_decode(ctx_dft, b);
            if (ret != 0) {
                LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);
                return false;
            }
            return true;
        }

        std::vector<int32_t> keep;
        keep.reserve((size_t) b.n_tokens);
        for (int32_t i = 0; i < b.n_tokens; ++i) {
            bool skip = false;
            if (b.n_seq_id[i] == 1) {
                const llama_seq_id s = b.seq_id[i][0];
                if (s >= 0 && s < (llama_seq_id) as.size() && as[s].pending) {
                    auto & a = as[s];
                    const llama_pos off = b.pos[i] - a.pos0;
                    if (off >= 0 && (size_t) off <= a.draft.size()) {
                        const llama_token expect = off == 0 ? a.id_last : a.draft[(size_t) off - 1];
                        if (b.token[i] == expect) {
                            skip = true;
                            a.skipped++;
                        } else {
                            // the target batch is not the draft we evaluated: undo it and let the normal decode below redo this seq
                            llama_memory_seq_rm(llama_get_memory(ctx_dft), s, a.pos0, -1);
                            async_retire(s, OUT_FALLBACK);
                            ast.n_fallback++;
                        }
                    }
                }
            }
            if (!skip) {
                keep.push_back(i);
            }
        }
        for (auto & a : as) {
            if (a.pending && a.skipped == a.draft.size() + 1) {
                ast.n_step0++;
                a.valid = !a.la.empty();
            }
        }
        if (keep.empty()) {
            return true;
        }
        llama_batch fb = llama_batch_init((int32_t) keep.size(), 0, 1);
        for (size_t k = 0; k < keep.size(); ++k) {
            const int32_t i = keep[k];
            fb.token[k]     = b.token[i];
            fb.pos[k]       = b.pos[i];
            fb.n_seq_id[k]  = 1;
            fb.seq_id[k][0] = b.seq_id[i][0];
            fb.logits[k]    = b.logits ? b.logits[i] : 0;
        }
        fb.n_tokens = (int32_t) keep.size();
        const int ret = llama_decode(ctx_dft, fb);
        llama_batch_free(fb);
        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);
            return false;
        }
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        async_join(false);

        // --spec-async: serve a seq from the lookahead when the target accepted the whole draft and its own token is the lookahead's c1
        std::vector<bool> from_la(n_seq, false);
        if (async_on) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                auto & a  = as[seq_id];
                auto & dp = dparams[seq_id];
                if (!a.pending) {
                    continue;
                }
                const bool had_la = a.valid;
                bool hit = false;
                bool observed = false, full_acc = false, c1_match = false;
                if (a.counted) {
                    ast.la_hist[std::min<size_t>(a.la.size(), 10)]++;
                }
                if (dp.drafting && dp.prompt != nullptr) {
                    // observe whether the target accepted the whole draft (the next round starts right after it)
                    const size_t n = a.draft.size();
                    const auto & pr = *dp.prompt;
                    const bool full = dp.pos0 == a.pos0 + (llama_pos) n + 1 && pr.size() == (size_t) dp.pos0 && pr.size() >= n + 1 &&
                                      pr[pr.size() - n - 1] == a.id_last && std::equal(a.draft.begin(), a.draft.end(), pr.end() - (long) n);
                    ema_full += 0.1 * ((full ? 1.0 : 0.0) - ema_full);
                    if (full && had_la) {
                        ema_c1 += 0.1 * ((dp.id_last == a.la[0] ? 1.0 : 0.0) - ema_c1);
                    }
                    observed = true;
                    full_acc = full;
                    c1_match = full && had_la && dp.id_last == a.la[0];
                    if (full && a.counted) {
                        ast.n_full++;
                        if (c1_match) {
                            ast.n_c1_match++;
                        }
                        if (a.n_c1_top > 0) {
                            int rank = 5; // O5 gate: where does the token the target took stand in the draft's own top 5 at c1?
                            for (int r = 0; r < a.n_c1_top; ++r) {
                                if (a.c1_top[r] == dp.id_last) {
                                    rank = r;
                                    break;
                                }
                            }
                            ast.c1_rank[rank]++;
                        }
                    }
                }
                if (dp.drafting && had_la && a.la.size() > 1 && dp.p_min < 0.0f && dp.result_q == nullptr && dp.dists == nullptr &&
                        dp.smpl_tgt == nullptr && dp.prompt != nullptr) {
                    const size_t n = a.draft.size();
                    const auto & pr = *dp.prompt;
                    hit = dp.id_last == a.la[0] && dp.pos0 == a.pos0 + (llama_pos) n + 1 && pr.size() == (size_t) dp.pos0 && pr.size() >= n + 1 &&
                          pr[pr.size() - n - 1] == a.id_last &&
                          std::equal(a.draft.begin(), a.draft.end(), pr.end() - (long) n);
                }
                if (hit) {
                    auto & result = *dp.result;
                    result.assign(a.la.begin() + 1, a.la.end());
                    if (dp.n_max > 0 && (int) result.size() > dp.n_max) {
                        result.resize(dp.n_max);
                    }
                    if (n_max > 0 && (int) result.size() > n_max) {
                        result.resize(n_max);
                    }
                    from_la[seq_id] = true;
                    ast.n_hit++;
                    if (as_waited[seq_id]) {
                        ast.n_inf_wait_hit++;
                    }
                }
                // exactly one outcome per round; n_rollback = real mispredictions only (a truncated or guarded lookahead is not one)
                int out = OUT_HIT;
                if (!hit) {
                    if (!observed)               { out = OUT_NODRAFT; }
                    else if (!a.gate_la)         { out = OUT_NOLA; }
                    else if (!had_la)            { out = OUT_LAEMPTY; }
                    else if (!full_acc)          { out = OUT_NOTFULL; }
                    else if (!c1_match)          { out = OUT_C1MISS; }
                    else if (a.la.size() <= 1)   { out = OUT_TRUNC; }
                    else                         { out = OUT_GUARD; }
                    if (out == OUT_NOTFULL || out == OUT_C1MISS) {
                        ast.n_rollback++;
                    }
                    if (as_waited[seq_id]) {
                        ast.n_inf_wait_miss++;
                    }
                }
                as_waited[seq_id] = 0;
                async_retire(seq_id, out);
            }
        }
        const int64_t t_sync0 = ggml_time_us();
        draft_serial(dparams, from_la);
        if (async_on) {
            bool any_sync = false;
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                any_sync = any_sync || (dparams[s].drafting && !from_la[s]);
            }
            if (any_sync) {
                ast.n_sync_drafts++;
                ast.t_sync_draft_us += (uint64_t) (ggml_time_us() - t_sync0);
            }
        }
    }

    // the serial draft (unchanged semantics); seqs in `skip` already have their draft from the lookahead
    void draft_serial(common_speculative_draft_params_vec & dparams, const std::vector<bool> & skip) {
        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting || skip[seq_id]) {
                continue;
            }
            n_drafting++;
            drafting[seq_id] = true;
            if (draft_sample) {
                spec_draft_retune(smpls, tuned_temp, tuned_seed, llama_get_model(ctx_dft), seq_id, dp.temp, dp.seed);
            }
            common_sampler_reset(smpls[seq_id].get());
            if (draft_grammar) {
                // a fresh copy of the target's constraint, which already holds id_last; null smpl_tgt clears it
                common_sampler_copy_constraint(smpls[seq_id].get(), dp.smpl_tgt, llama_get_model(ctx_dft));
            }

            common_batch_add(batch, dp.id_last, dp.pos0, { seq_id }, true);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        // position-gated LoRA (row S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01): the drafted tokens (pos0 + 1 and up)
        // are the draft rows and carry a draft_rows adapter on ctx_dft; id_last (pos0) is an accepted token and stays base.
        // Cleared on every exit so process() (accepted and prompt tokens) never sees the gate. No gated adapter = no effect.
        struct draft_rows_scope {
            llama_context *           ctx;
            std::vector<llama_seq_id> seqs;
            ~draft_rows_scope() {
                for (const llama_seq_id s : seqs) {
                    llama_set_draft_start(ctx, s, -1);
                }
            }
        } draft_rows{ ctx_dft, {} };
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (drafting[seq_id]) {
                llama_set_draft_start(ctx_dft, seq_id, dparams[seq_id].pos0 + 1);
                draft_rows.seqs.push_back(seq_id);
            }
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                const llama_token sampled = common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                if (draft_grammar && sampled == LLAMA_TOKEN_NULL) {
                    // the grammar allows no token of the vocabulary here: nothing to draft (the sample path below would otherwise
                    // accept token -1)
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                // q = the distribution `sampled` was drawn from, taken BEFORE get_candidates re-sorts cur_p. Not
                // cur_p->selected (this fork's token_with_rng never sets it -- MEASURED 'selected idx 0' at every
                // position) and not cur_p->data[k].p (DIST leaves it unnormalised: top always 1.0).
                common_spec_dist q;
                if (draft_sample) {
                    q = common_sampler_sampling_dist(smpl, sampled);
                }

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                // add drafted token for each sequence. Default: the argmax (a one-hot q). --spec-draft-sample: the
                // sampler's own draw, x ~ q.
                const llama_token id = draft_sample ? sampled : cur_p->data[0].id;

                // only collect very high-confidence draft tokens; a request's speculative.p_min overrides
                const float p_min_seq = dparams[seq_id].p_min >= 0.0f ? dparams[seq_id].p_min : p_min;
                if (common_speculative_top_prob(cur_p) < p_min_seq) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (draft_grammar) {
                    // a grammar refuses a token it cannot take by throwing (src/llama-grammar.cpp accept); every
                    // candidate was masked out, so end this draft rather than propose a token the target will refuse
                    try {
                        common_sampler_accept(smpl, ctx_dft, id, true);
                    } catch (const std::exception & e) {
                        LOG_DBG("%s: seq %d: draft grammar stopped the draft: %s\n", __func__, (int) seq_id, e.what());
                        drafting[seq_id] = false;
                        n_drafting--;
                        continue;
                    }
                } else {
                    common_sampler_accept(smpl, ctx_dft, id, true);
                }

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);
                if (draft_sample && dp.dists != nullptr) {
                    LOG_DBG("%s: seq %d pos %zu: drafted %d with q=%.3f over %zu candidates (top %d)\n", __func__,
                            (int) seq_id, result.size() - 1, id, (double) common_spec_dist_prob(q, id), q.ids.size(),
                            cur_p->data[0].id);
                    dp.dists->push_back(std::move(q));
                }

                if ((n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.pos0 + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }
            const int32_t n_min_seq = dp.n_min >= 0 ? dp.n_min : n_min; // a request's speculative.n_min overrides
            if (dp.result->size() < (size_t) n_min_seq) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    int32_t stage_n_max() const override { return n_max; }
};

// llama-server's per-sequence MTP drafter (2026-10-17, row S69-FORK-SERVER-PER-SEQ-SPECULATOR-NO-MTP-NO-DRAFT-POLICY-2026-10-17-01).
// Without it `--spec-type mtp` served with no speculation and `--spec-type suffix,mtp` silently lost its MTP stage.
//
// SOURCE-READ (this tree): the fork's MTP models (qwen35 / qwen35moe / glm4_moe / step35 / ...) decode through the LEGACY graph
// path -- llama_context::decode hands them to llama_decode_internal (src/llama-context.cpp:3125) and the graph is
// llm_build_context::build_qwen35() (src/graphs/build_qwen35.cpp:76). That path never fills upstream's nextn embeddings
// (llama_get_embeddings_nextn returns null), so an impl ported from upstream (llama.cpp @9588757 common/speculative.cpp:1328-1765,
// common_speculative_impl_draft_mtp) failed every request with "the target context produced no nextn embeddings" (measured,
// test_after1/server_mtp.log). This impl uses the fork's own MTP primitives instead, the ones the single-context engine drives:
//   * target hidden rows: with cparams.mtp the target writes one hidden row PER BATCH TOKEN into ctx->embd
//     (src/llama.cpp:8179 n_outputs_embd = n_tokens_all; result_mtp_embd, src/llama.cpp:8519), read through
//     llama_spec_get_hidden_feature_view (src/llama-spec-features.cpp:81);
//   * MTP KV mirror: llama_set_draft_input_hidden_state_copy + mtp_update_kv_cache(..., MTP_OP_WARMUP) on the MTP context, with the
//     rows SHIFTED by one so the token at position p pairs with h(p-1) -- the convention common_speculative_on_target_batch's
//     warmup uses (this file, "row-0 conditioning"), and the input build_qwen35_mtp reads row for row;
//   * drafting: one MTP_OP_DRAFT_GEN decode per drafted token, its result_mtp_embd row feeding the next step, as
//     mtp_speculative_gen_draft does (this file), then the speculative tail is dropped from the MTP KV.
//
// DESIGN: ONE impl for all sequences, its state keyed by seq id, over the ONE MTP context llama-server already creates
// (common_speculative_init_result above: the target model again, ctx_type MTP, one KV sequence per slot). NOT n_seq wrappers of
// common_speculative_state_mtp: that state owns (and frees) its own MTP context and keeps per-CONTEXT carries (mtp_carry,
// draft_cache_by_seq's one-token shortcut, mtp_warmed_heads); n_seq of them would need n_seq MTP contexts over one target.
// Every MTP decode here carries ONE sequence (mtp_update_kv_cache trims and warms a single seq), and drafting runs one sequence
// at a time, so no MTP graph ever mixes slots.
//
// Limits, each stated rather than guessed at:
//   * one MTP head (mtp_n_heads = 1, step 0 for every draft step): the multi-head carry (lctx.mtp_carry, src/llama.cpp:8855) is
//     one buffer per context, so deeper heads cannot be kept per slot; a multi-head model still drafts, with its first head;
//   * an external MTP companion (-md, Gemma4 assistant: the draft model differs from the target and reads the target KV through
//     llama_set_mtp_target_context) is not handled here -- the constructor refuses it, and the pool warns the stage was not built;
//   * embedding-input (vision) batches are not mirrored (as upstream);
//   * the carried-over row is tracked WITH ITS POSITION and the row before it is kept: when the server rolls a sequence back to its
//     speculative checkpoint (hybrid/recurrent target, partial acceptance) and replays the accepted tokens, the replay batch starts
//     at the checkpoint position again, and its first token must pair with the h from BEFORE the rolled-back verify batch.
struct common_speculative_impl_mtp : public common_speculative_impl {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    int32_t n_max;
    int32_t n_min;
    float   p_min;

    llama_batch batch; // token batch, one sequence at a time; the hidden rows go in through llama_set_draft_input_hidden_state_copy

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd       = 0;
    int32_t n_mtp_layers = 1;

    // per-sequence carry-over: h of the last committed position, which pairs with the next token at MTP pos p+1
    std::vector<std::vector<float>> pending_h;     // [n_seq][n_embd]
    // h rows of each sequence in the most recent target batch; row 0 = the sampled token, row N = the Nth accepted draft
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t>            verify_h_rows;
    std::vector<llama_pos>          verify_pos0;   // position of verify row 0
    std::vector<llama_pos>          pending_pos;   // position whose h pending_h holds (-1 = none yet)
    std::vector<std::vector<float>> prev_h;        // the carried-over row before the most recent batch, for a replay
    std::vector<llama_pos>          prev_pos;

    // scratch
    std::vector<float> cond_rows;
    std::vector<float> h_step;

    // h of position `pos` in sequence s from what this impl still holds, or nullptr
    const float * h_at(llama_seq_id s, llama_pos pos) const {
        if (pos < 0) {
            return nullptr;
        }
        if (pending_pos[s] == pos) {
            return pending_h[s].data();
        }
        if (verify_h_rows[s] > 0 && pos >= verify_pos0[s] && pos < verify_pos0[s] + verify_h_rows[s]) {
            return verify_h[s].data() + (size_t) (pos - verify_pos0[s]) * n_embd;
        }
        if (prev_pos[s] == pos) {
            return prev_h[s].data();
        }
        return nullptr;
    }

    bool draft_sample = false; // --spec-draft-sample (plan gap G1)
    std::vector<float>    tuned_temp; // plan gap G2, see spec_draft_retune
    std::vector<uint32_t> tuned_seed;

    common_speculative_impl_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_MTP, n_seq)
        , ctx_tgt(params.ctx_tgt)
        , ctx_dft(params.ctx_dft)
        , n_max(std::max(1, params.n_max))
        , n_min(std::max(0, params.n_min))
        , p_min(params.p_min)
        , draft_sample(params.draft_sample)
    {
        GGML_ASSERT(ctx_tgt != nullptr && ctx_dft != nullptr);

        if (llama_get_model(ctx_dft) != llama_get_model(ctx_tgt)) {
            throw std::runtime_error("the per-sequence pool drafts only with the target's own embedded MTP / NextN head; "
                                     "an external MTP companion model (-md) is handled by the single-context engine");
        }
        n_mtp_layers = llama_model_n_nextn_layer(llama_get_model(ctx_dft));
        if (n_mtp_layers <= 0) {
            throw std::runtime_error("the target model has no MTP / NextN layer");
        }

        n_embd = (int32_t) llama_mtp_state_n_embd(ctx_dft);
        if (n_embd <= 0 || n_embd != (int32_t) llama_mtp_state_n_embd(ctx_tgt)) {
            LOG_ERR("%s: MTP state width %d does not match the target hidden width %d\n", __func__, n_embd,
                    (int) llama_mtp_state_n_embd(ctx_tgt));
            throw std::runtime_error("MTP state width must match the target hidden width");
        }
        if (n_seq > llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %u > %u\n", __func__, n_seq, llama_n_seq_max(ctx_dft));
            throw std::runtime_error("the MTP draft context holds fewer sequences than the speculative n_seq");
        }

        const int32_t n_b = std::max<int32_t>((int32_t) llama_n_batch(ctx_dft), (int32_t) llama_n_batch(ctx_tgt));
        batch = llama_batch_init(n_b, 0, 1);

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), spec_default_draft_sparams()));
        }
        tuned_temp.assign(n_seq, NAN);
        tuned_seed.assign(n_seq, LLAMA_DEFAULT_SEED);

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
        verify_pos0.assign(n_seq, -1);
        pending_pos.assign(n_seq, -1);
        prev_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        prev_pos.assign(n_seq, -1);
        h_step.assign(n_embd, 0.0f);

        LOG_INF("%s: adding speculative implementation 'mtp' (per-sequence, MTP context shared by %u sequence(s), legacy MTP ops)\n", __func__, n_seq);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, nextn_layers=%d (drafting with head 0)\n", __func__,
                n_max, n_min, p_min, n_embd, n_mtp_layers);
    }

    ~common_speculative_impl_mtp() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        const llama_pos pos_max = llama_kv_cache_seq_pos_max(ctx_dft, seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: seq %d: MTP context pos_max=%d < N-1=%d -- process() has not mirrored every prompt token; drafts may degrade\n",
                    __func__, (int) seq_id, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & b) override {
        if (b.n_tokens <= 0) {
            return true;
        }
        const common_spec_mtp_input_kind in_kind = common_spec_mtp_input_kind_of(b.token != nullptr, b.embd != nullptr);
        if (in_kind == COMMON_SPEC_MTP_INPUT_NONE) {
            return true;
        }
        // an embedding-input batch (mtmd image / audio chunk): the rows go to the MTP context as embeddings, paired with the same
        // hidden-state conditioning as token rows (graph side: llm_build_context::build_inp_embd_mtp, embd branch)
        const bool is_embd = in_kind == COMMON_SPEC_MTP_INPUT_EMBD;
        const int32_t n_embd_in = (int32_t) llama_model_n_embd(llama_get_model(ctx_tgt));
        if (is_embd && n_embd_in != n_embd) {
            LOG_WRN("%s: embd batch width %d != MTP width %d; this chunk is not mirrored into the MTP KV\n", __func__, n_embd_in, n_embd);
            return true;
        }

        llama_spec_feature_view view;
        if (!llama_spec_get_hidden_feature_view(ctx_tgt, b, view) || view.width != n_embd || (int32_t) view.rows.size() != b.n_tokens) {
            LOG_ERR("%s: the target context produced no MTP hidden rows for this batch (width %d, rows %zu of %d)\n", __func__,
                    view.width, view.rows.size(), b.n_tokens);
            return false;
        }

        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        const int32_t n_tokens = b.n_tokens;

        auto seq_of = [&](int k) -> llama_seq_id {
            if (b.n_seq_id == nullptr || b.seq_id == nullptr || b.n_seq_id[k] != 1 || b.seq_id[k] == nullptr) {
                return -1;
            }
            const llama_seq_id s = b.seq_id[k][0];
            return s >= 0 && s < (llama_seq_id) n_seq ? s : -1;
        };

        // rows of each sequence in this batch, in batch order
        std::vector<std::vector<int32_t>> rows(n_seq);
        for (int k = 0; k < n_tokens; ++k) {
            const llama_seq_id s = seq_of(k);
            if (s >= 0) {
                rows[s].push_back(k);
            }
        }

        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (rows[s].empty()) {
                continue;
            }
            const int32_t n_rows = (int32_t) rows[s].size();

            // a batch that starts at position 0 is a fresh prompt for this slot: drop the previous occupant's boundary rows
            // (BEELLAMA carry-state row) so no restore / lookup can resurrect them
            if (b.pos[rows[s][0]] == 0) {
                pending_pos[s]   = -1;
                prev_pos[s]      = -1;
                verify_pos0[s]   = -1;
                verify_h_rows[s] = 0;
                std::fill(pending_h[s].begin(), pending_h[s].end(), 0.0f);
            }

            // mirror this sequence's tokens into the MTP KV: token k at pos p pairs with h(p-1) of its own sequence
            common_batch_clear(batch);
            cond_rows.resize((size_t) n_rows * n_embd);
            for (int32_t i = 0; i < n_rows; ++i) {
                const int32_t k = rows[s][i];
                const float * h_prev = nullptr;
                if (i > 0) {
                    h_prev = view.rows[rows[s][i - 1]].data;
                } else if (b.pos[k] > 0) {
                    h_prev = h_at(s, b.pos[k] - 1);
                    if (h_prev == nullptr) {
                        // a jump this impl has no row for (e.g. a prompt-cache restore): the carried-over row stands in;
                        // drafts recover once real rows flow again
                        h_prev = pending_h[s].data();
                    }
                }
                if (h_prev != nullptr) {
                    std::memcpy(cond_rows.data() + (size_t) i * n_embd, h_prev, row_bytes);
                } else {
                    std::fill(cond_rows.begin() + (size_t) i * n_embd, cond_rows.begin() + (size_t) (i + 1) * n_embd, 0.0f);
                }
                if (!is_embd) {
                    common_batch_add(batch, b.token[k], b.pos[k], { s }, false);
                }
            }

            // embd input: a one-sequence embd batch for the MTP context, rows copied from the source batch
            llama_batch ebatch = {};
            if (is_embd) {
                ebatch = llama_batch_init(n_rows, n_embd, 1);
                if (!common_spec_mtp_gather_embd_rows(b.embd, n_embd, rows[s], ebatch.embd)) {
                    llama_batch_free(ebatch);
                    LOG_ERR("%s: seq %d: could not gather %d embedding row(s)\n", __func__, (int) s, n_rows);
                    return false;
                }
                for (int32_t i = 0; i < n_rows; ++i) {
                    ebatch.pos[i]       = b.pos[rows[s][i]];
                    ebatch.n_seq_id[i]  = 1;
                    ebatch.seq_id[i][0] = s;
                    ebatch.logits[i]    = false;
                }
                ebatch.n_tokens = n_rows;
            }

            if (!llama_set_draft_input_hidden_state_copy(ctx_dft, cond_rows.data(), cond_rows.size())) {
                if (is_embd) {
                    llama_batch_free(ebatch);
                }
                LOG_ERR("%s: seq %d: could not hand %d hidden row(s) to the MTP context\n", __func__, (int) s, n_rows);
                return false;
            }
            const int32_t rc = mtp_update_kv_cache(ctx_dft, is_embd ? ebatch : batch, /*is_prompt_warmup=*/ true, /*mtp_heads=*/ 1);
            if (is_embd) {
                llama_batch_free(ebatch);
            }
            if (rc != 0) {
                LOG_ERR("%s: seq %d: MTP KV update failed rc=%d (pos=%d, %d token(s))\n", __func__, (int) s, (int) rc,
                        (int) b.pos[rows[s][0]], n_rows);
                return false;
            }

            const llama_pos pos0 = b.pos[rows[s][0]];

            // keep the row that precedes this batch, for a rollback-and-replay that starts at pos0 again
            if (const float * h_before = h_at(s, pos0 - 1)) {
                if (h_before != prev_h[s].data()) {
                    std::memcpy(prev_h[s].data(), h_before, row_bytes);
                }
                prev_pos[s] = pos0 - 1;
            }

            verify_h_rows[s] = n_rows;
            verify_pos0[s]   = pos0;
            verify_h[s].resize((size_t) n_rows * n_embd);
            for (int32_t i = 0; i < n_rows; ++i) {
                std::memcpy(verify_h[s].data() + (size_t) i * n_embd, view.rows[rows[s][i]].data, row_bytes);
            }
            std::memcpy(pending_h[s].data(), verify_h[s].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
            pending_pos[s] = b.pos[rows[s][n_rows - 1]];
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }
            auto & result = *dp.result;
            if (draft_sample) {
                spec_draft_retune(smpls, tuned_temp, tuned_seed, llama_get_model(ctx_dft), seq_id, dp.temp, dp.seed);
            }
            auto * smpl = smpls[seq_id].get();
            common_sampler_reset(smpl);
            const size_t n_dists0 = dp.dists != nullptr ? dp.dists->size() : 0;

            // id_last sits at n_past, so it pairs with the h of n_past-1 (normally the row accept() carried over)
            const float * h0 = h_at(seq_id, dp.pos0 - 1);
            if (h0 == nullptr) {
                h0 = pending_h[seq_id].data();
            }
            std::memcpy(h_step.data(), h0, row_bytes);

            const int32_t n_max_seq = dp.n_max > 0 ? std::min(n_max, dp.n_max) : n_max;
            const float   p_min_seq = dp.p_min >= 0.0f ? dp.p_min : p_min;

            llama_set_mtp_n_heads(ctx_dft, 1);
            llama_set_mtp_op_type(ctx_dft, MTP_OP_DRAFT_GEN);

            llama_token tok = dp.id_last;
            int n_decode = 0;
            for (int i = 0; i < n_max_seq; ++i) {
                common_batch_clear(batch);
                common_batch_add(batch, tok, dp.pos0 + i, { seq_id }, true);
                if (!llama_set_draft_input_hidden_state_copy(ctx_dft, h_step.data(), h_step.size())) {
                    break;
                }
                llama_set_mtp_step_idx(ctx_dft, 0);

                ++n_decode;
                const int ret = llama_decode(ctx_dft, batch);
                if (ret != 0) {
                    LOG_WRN("%s: seq %d: MTP draft decode[%d] returned %d\n", __func__, (int) seq_id, i, ret);
                    break;
                }

                const llama_token sampled = common_sampler_sample(smpl, ctx_dft, 0, true);
                // plan gap G1 (upstream 1fb7ef3e3 extends probabilistic drafting to MTP): under --spec-draft-sample the
                // draft is the sampler's own draw and q, taken before get_candidates re-sorts, goes to the verifier -- the
                // same path as the draft-model impl above
                common_spec_dist q;
                if (draft_sample) {
                    q = common_sampler_sampling_dist(smpl, sampled);
                }
                const auto * cur_p = common_sampler_get_candidates(smpl, true);
                const llama_token id = draft_sample ? sampled : cur_p->data[0].id;

                // the first draft is always kept; later ones only while the head stays confident (the fork's MTP gate)
                if (!result.empty() && p_min_seq > 0.0f && common_speculative_top_prob(cur_p) < p_min_seq) {
                    break;
                }

                common_sampler_accept(smpl, ctx_dft, id, true);
                result.push_back(id);
                if (draft_sample && dp.dists != nullptr) {
                    dp.dists->push_back(std::move(q));
                }

                const float * emb = llama_get_embeddings_ith(ctx_dft, 0);
                if (emb == nullptr) {
                    break;
                }
                std::memcpy(h_step.data(), emb, row_bytes); // a stable copy: the next decode reuses ctx->embd
                tok = id;
            }

            llama_set_mtp_step_idx(ctx_dft, 0);
            llama_set_mtp_n_heads(ctx_dft, 0);
            llama_set_mtp_op_type(ctx_dft, MTP_OP_NONE);

            // drop the speculative tail from the MTP KV; position n_past (id_last over h(n_past-1)) is committed and stays --
            // the next process() rewrites it from the verify batch anyway
            if (n_decode > 0) {
                llama_kv_cache_seq_rm(ctx_dft, seq_id, dp.pos0 + 1, -1);
            }

            const int32_t n_min_seq = dp.n_min >= 0 ? dp.n_min : n_min;
            if (result.size() < (size_t) n_min_seq) {
                result.clear();
                if (dp.dists != nullptr) {
                    dp.dists->resize(n_dists0); // q stays parallel to the (now empty) draft
                }
            }
        }
    }

    // is_other included: whichever stage drafted, the verified rows decide which h carries over to the next round
    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }
        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, (size_t) n_embd * sizeof(float));
        pending_pos[seq_id] = verify_pos0[seq_id] + i_h;
    }

    // ggml-org/llama.cpp 0f35d6d7b (#30141): checkpoint the boundary hidden row so a slot restore / prompt-cache hit resumes
    // drafting from the right h instead of a stale one. The blob is the versioned, validated layout of common/spec-mtp-state.h
    // (magic, version, width, pos, row): a blob from another model width or a foreign layout is refused and the state is reset,
    // never reinterpreted. The verify / prev rows belong to a batch the restore rolls back, so they are dropped.
    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        common_spec_mtp_state_encode(pending_h[seq_id].data(), (uint32_t) n_embd, (int32_t) pending_pos[seq_id], data);
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        std::vector<float> h((size_t) n_embd);
        int32_t pos = -1;
        const common_spec_mtp_state_status st = common_spec_mtp_state_decode(data, (uint32_t) n_embd, h.data(), pos);
        if (st == COMMON_SPEC_MTP_STATE_OK) {
            std::memcpy(pending_h[seq_id].data(), h.data(), h.size() * sizeof(float));
            pending_pos[seq_id] = pos;
        } else {
            if (st != COMMON_SPEC_MTP_STATE_EMPTY) {
                LOG_WRN("%s: seq %d: MTP state blob refused (status %d, %zu bytes); resetting\n", __func__, (int) seq_id, (int) st, data.size());
            }
            std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
            pending_pos[seq_id] = -1;
        }
        verify_h_rows[seq_id] = 0;
        verify_pos0[seq_id]   = -1;
        prev_pos[seq_id]      = -1;
    }

    int32_t stage_n_max() const override { return n_max; }
};

// S69-FORK-SERVER-POOL-EAGLE3-DFLASH-IMPLS-2026-10-17-01: EAGLE3 for the per-sequence pool (llama-server and llama-speculative-simple,
// which also builds the pool, with n_seq = 1; before this, `--spec-type eagle3` drafted nothing in either of them).
// SOURCE-READ: llama.cpp b9590 common/speculative.cpp:378-408 (pooled eagle3: a "TODO: implement" stub, nothing to port);
// ggml-org/llama.cpp PR #18039 (MIT) for the pairing convention; and the fork's single-context common_speculative_state_eagle3
// above. Written for this tree; no code copied.
// CONVENTION (PR #18039): draft KV row at position P is (token at P+1, target features g at P). The features are the target's
// [lo|mid|hi] hidden rows at the trained layers, captured per batch row (llama_eagle3_get_feats_pp_ith).
// PER SEQUENCE: process() writes the pairs of this seq's rows in the target batch. The last row is a deferred BOUNDARY,
// because the token after it is the target's next sample. draft() completes the boundary with id_last and drafts from its
// logits. accept() moves the boundary to the last verified row and drops the draft KV of rejected rows.
// NUANCES:
//  (1) K = 1 draft per round, as in the single-context state. Steps 2+ would feed the draft's own hidden state through
//      eagle3_g_direct; that path has no setter in this tree yet.
//  (2) The draft is the argmax of the draft logits, mapped through d2t (the single-context state samples DIST).
//  (3) The target captures per UBATCH, so a batch larger than n_ubatch leaves features only for the last ubatch. The
//      rows before it get no draft KV (warned once). Prompts that fit one ubatch, and every verify batch, are complete.
//  (4) The draft context belongs to the caller (common_speculative_init_result), so it is not freed here.
struct common_speculative_impl_eagle3_pool : public common_speculative_impl {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    int32_t n_min;
    int32_t n_embd_tgt = 0;
    int32_t feats_w    = 0;
    int32_t n_vocab_dft = 0;

    std::vector<int32_t> d2t;   // draft -> target ids, absolute; empty = identity
    bool d2t_invalid = false;

    llama_batch batch;

    struct seq_state {
        std::vector<float> verify_g;   // [verify_rows * feats_w], this seq's rows of the latest target batch
        int32_t   verify_rows = 0;
        llama_pos verify_pos0 = -1;
        std::vector<float> pending_g;  // boundary features; pair with the token at pending_pos + 1
        llama_pos pending_pos = -1;
    };
    std::vector<seq_state> seqs;

    std::vector<float>       rows_g;   // scratch: features of the pairs of one decode
    std::vector<float>       row_g;    // scratch: one row
    bool warned_partial = false;

    common_speculative_impl_eagle3_pool(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_EAGLE3, n_seq)
        , ctx_tgt(params.ctx_tgt)
        , ctx_dft(params.ctx_dft)
        , n_min(std::max(0, params.n_min))
    {
        GGML_ASSERT(ctx_tgt != nullptr && ctx_dft != nullptr);
        if (n_seq > llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %u > %u\n", __func__, n_seq, llama_n_seq_max(ctx_dft));
            throw std::runtime_error("the EAGLE3 draft context holds fewer sequences than the speculative n_seq");
        }
        n_embd_tgt  = common_eagle3_attach(ctx_tgt, ctx_dft, d2t, d2t_invalid);
        feats_w     = 3 * n_embd_tgt;
        n_vocab_dft = llama_n_vocab(llama_get_model(ctx_dft));
        if (feats_w <= 0 || n_vocab_dft <= 0) {
            throw std::runtime_error("EAGLE3 draft: no target hidden width or no draft vocab");
        }
        batch = llama_batch_init(std::max<int32_t>(1, (int32_t) llama_n_batch(ctx_dft)), 0, 1);
        seqs.resize(n_seq);
        row_g.assign((size_t) feats_w, 0.0f);
        if (params.n_max > 1) {
            LOG_WRN("%s: EAGLE3 in the per-sequence pool drafts 1 token per round (n_max=%d requested)\n", __func__, params.n_max);
        }
        LOG_INF("%s: adding speculative implementation 'eagle3' (per-sequence, draft context shared by %u sequence(s), "
                "n_embd_tgt=%d, draft vocab=%d, d2t=%zu%s)\n", __func__, n_seq, n_embd_tgt, n_vocab_dft, d2t.size(),
                d2t_invalid ? ", INVALID: drafting off" : "");
    }

    ~common_speculative_impl_eagle3_pool() override {
        llama_set_eagle3_capture(ctx_tgt, false, -1, -1, -1);
        llama_batch_free(batch);
    }

    // the features of batch row k of the latest target decode, or nullptr when that row was not captured
    const float * feats_of(int32_t k, int32_t off, int32_t n_rows_pp) {
        if (k < off || k - off >= n_rows_pp) {
            return nullptr;
        }
        if (llama_eagle3_get_feats_pp_ith(ctx_tgt, k - off, row_g.data()) != feats_w) {
            return nullptr;
        }
        return row_g.data();
    }

    // decode (token, pos, features) pairs of ONE sequence into the draft KV, in chunks of the draft context's n_batch
    bool write_pairs(llama_seq_id s, const std::vector<llama_token> & toks, const std::vector<llama_pos> & poss,
                     const std::vector<float> & g, bool logits_last) {
        const int32_t n = (int32_t) toks.size();
        const int32_t n_b = std::max<int32_t>(1, (int32_t) llama_n_batch(ctx_dft));
        for (int32_t i0 = 0; i0 < n; i0 += n_b) {
            const int32_t m = std::min(n_b, n - i0);
            common_batch_clear(batch);
            for (int32_t i = 0; i < m; ++i) {
                const bool want = logits_last && i0 + i == n - 1;
                common_batch_add(batch, toks[i0 + i], poss[i0 + i], { s }, want);
            }
            if (!llama_set_draft_input_hidden_state_copy(ctx_dft, g.data() + (size_t) i0 * feats_w, (size_t) m * feats_w)) {
                return false;
            }
            const int rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                LOG_WRN("%s: seq %d: EAGLE3 draft decode returned %d (pos %d, %d pair(s))\n", __func__, (int) s, rc,
                        (int) poss[i0], m);
                return false;
            }
        }
        return true;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        // called after the prompt's process(), so the boundary it left must stay
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) n_seq);
    }

    bool process(const llama_batch & b) override {
        if (b.n_tokens <= 0 || b.token == nullptr || b.pos == nullptr || b.embd != nullptr) {
            return true;
        }
        const int32_t n_rows_pp = llama_eagle3_get_feats_pp_ith(ctx_tgt, -1, nullptr);
        if (n_rows_pp <= 0 || n_rows_pp > b.n_tokens) {
            LOG_DBG("%s: no EAGLE3 target features for this batch (%d row(s) captured, %d token(s))\n", __func__,
                    n_rows_pp, b.n_tokens);
            return true;
        }
        const int32_t off = b.n_tokens - n_rows_pp;  // ubatches run in batch order; the capture holds the last one
        if (off > 0 && !warned_partial) {
            warned_partial = true;
            LOG_WRN("%s: a %d-token batch spans several ubatches; EAGLE3 features exist only for its last %d row(s), so "
                    "the rows before get no draft KV (use -ub >= -b for full prompts)\n", __func__, b.n_tokens, n_rows_pp);
        }

        auto seq_of = [&](int k) -> llama_seq_id {
            if (b.n_seq_id == nullptr || b.seq_id == nullptr || b.n_seq_id[k] != 1 || b.seq_id[k] == nullptr) {
                return -1;
            }
            const llama_seq_id s = b.seq_id[k][0];
            return s >= 0 && s < (llama_seq_id) n_seq ? s : -1;
        };
        std::vector<std::vector<int32_t>> rows(n_seq);
        for (int32_t k = 0; k < b.n_tokens; ++k) {
            const llama_seq_id s = seq_of(k);
            if (s >= 0) {
                rows[s].push_back(k);
            }
        }

        std::vector<llama_token> toks;
        std::vector<llama_pos>   poss;
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            const int32_t n = (int32_t) rows[s].size();
            if (n == 0) {
                continue;
            }
            auto & st = seqs[s];
            const llama_pos p0 = b.pos[rows[s][0]];

            toks.clear();
            poss.clear();
            rows_g.clear();
            // the previous boundary, when draft() did not complete it (this seq did not draft last round)
            if (st.pending_pos >= 0 && st.pending_pos == p0 - 1) {
                toks.push_back(b.token[rows[s][0]]);
                poss.push_back(st.pending_pos);
                rows_g.insert(rows_g.end(), st.pending_g.begin(), st.pending_g.end());
            }
            // pairs inside the batch: (token of row i+1, pos of row i, features of row i)
            st.verify_rows = n;
            st.verify_pos0 = p0;
            st.verify_g.assign((size_t) n * feats_w, 0.0f);
            for (int32_t i = 0; i < n; ++i) {
                const float * g = feats_of(rows[s][i], off, n_rows_pp);
                if (g == nullptr) {
                    continue;
                }
                std::memcpy(st.verify_g.data() + (size_t) i * feats_w, g, (size_t) feats_w * sizeof(float));
                if (i + 1 < n) {
                    toks.push_back(b.token[rows[s][i + 1]]);
                    poss.push_back(b.pos[rows[s][i]]);
                    rows_g.insert(rows_g.end(), g, g + feats_w);
                }
            }
            // rewrite from the first pair on, so a replayed or rolled-back range never leaves two rows at one position
            if (!poss.empty()) {
                llama_kv_cache_seq_rm(ctx_dft, s, poss.front(), -1);
                if (!write_pairs(s, toks, poss, rows_g, false)) {
                    return false;
                }
            }
            // the new boundary: the last row; its next token is the target's next sample
            st.pending_g.assign(st.verify_g.end() - feats_w, st.verify_g.end());
            st.pending_pos = b.pos[rows[s][n - 1]];
        }
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        GGML_ASSERT(dparams.size() == n_seq);
        if (d2t_invalid) {
            return;
        }
        std::vector<llama_token> toks(1);
        std::vector<llama_pos>   poss(1);
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            auto & dp = dparams[s];
            if (!dp.drafting) {
                continue;
            }
            auto & st = seqs[s];
            // id_last sits at n_past, so it pairs with the boundary at n_past - 1
            if (st.pending_pos < 0 || st.pending_pos != dp.pos0 - 1) {
                LOG_DBG("%s: seq %d: no EAGLE3 boundary at %d (have %d)\n", __func__, (int) s, (int) dp.pos0 - 1,
                        (int) st.pending_pos);
                continue;
            }
            toks[0] = dp.id_last;
            poss[0] = st.pending_pos;
            llama_kv_cache_seq_rm(ctx_dft, s, st.pending_pos, -1);
            if (!write_pairs(s, toks, poss, st.pending_g, true)) {
                continue;
            }
            // the boundary row is now committed in the draft KV (id_last is the target's own sample)
            st.pending_pos = -1;

            const float * logits = llama_get_logits_ith(ctx_dft, 0);
            if (logits == nullptr) {
                continue;
            }
            llama_token id = 0;
            for (int32_t v = 1; v < n_vocab_dft; ++v) {
                if (logits[v] > logits[id]) {
                    id = v;
                }
            }
            if (!d2t.empty()) {
                if (id < 0 || id >= (llama_token) d2t.size()) {
                    continue;
                }
                id = (llama_token) d2t[id];
            }
            if (n_min > 1 || (dp.n_min > 1)) {
                continue;  // one token per round can never meet a minimum above 1
            }
            dp.result->push_back(id);
        }
    }

    // is_other included: whichever stage drafted, the verified rows decide where the boundary moves
    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        auto & st = seqs[seq_id];
        if (st.verify_rows <= 0) {
            return;
        }
        const int32_t i_b = std::min<int32_t>(n_accepted, st.verify_rows - 1);
        st.pending_g.assign(st.verify_g.begin() + (size_t) i_b * feats_w, st.verify_g.begin() + (size_t) (i_b + 1) * feats_w);
        st.pending_pos = st.verify_pos0 + i_b;
        // the pair at pending_pos carried the first rejected token: drop it and everything after
        llama_kv_cache_seq_rm(ctx_dft, seq_id, st.pending_pos, -1);
        st.verify_rows = 0;
    }

    int32_t stage_n_max() const override { return 1; }
};

// DFlash / DSpark in the per-sequence pool (llama-server, llama-speculative-simple). Row
// S69-ENGREV-UPSTREAM-MERGE-SPECULATIVE-PORT-LIST-2026-11-04-01, approved 2026-11-05 ("port upstream's pooled DFlash family into
// the fork's pool; keep the single-context engine").
//
// SOURCE-READ: upstream 977c48a25 common/speculative.cpp:984-1488 (common_speculative_impl_draft_dflash) and the fork's
// common/speculative-dflash-impl.h:1-750 (common_speculative_state_dflash).
// INVENT-LADDER-OK: adapt route (rule 1.14). Upstream's impl reads its features from llama_get_embeddings_layer_inp(), which is
// filled by class-built src/models graphs (res->t_layer_inp; src/llama-graph.cpp:1460-1466, src/llama-context.cpp:4015) and
// by none of the fork's legacy src/graphs/* targets (qwen3 among them), where upstream would GGML_ABORT. The fork's own capture
// (llama_set_dflash_capture_layers, an eval callback filled on the legacy decode path, src/llama.cpp:8884) serves those, so
// this impl takes upstream's POOL CONTRACT (one impl for n_seq sequences, per-sequence state, process() seeds from every
// target batch, draft() for all drafting sequences, n_max clamped to the trained block) and drafts with the fork's proven
// engine, one common_speculative_state_dflash per sequence:
//   * ONE caller-owned ctx_dft is shared. Each state's draft() clears that context and decodes its own block, so sequences
//     never see each other's KV; the incremental draft-KV plan is per context, so it is reset whenever the drafted sequence
//     changes (a lone active slot keeps the incremental path).
//   * the target capture is installed ONCE, by state 0 (a second install chains the callback to itself), cleared once here.
//   * a state's host feature ring is released after construction and grows on first use, so idle slots cost nothing.
//   * process() sees verify batches whole, rejected draft rows included; draft() drops every row at or past id_last's position
//     (dflash_truncate_target_features) before seeding the block there.
//   * embedding (mtmd) batches are fed like token batches, except a sequence whose rows are all pinned to one position (an
//     M-RoPE image) -- the same skip as upstream fa6769818; the draft then conditions on the rows around the image.
//   * DFlash2 at draft temperature > 0 records the per-position proposal distribution into dp.dists (--spec-draft-sample).
struct common_speculative_impl_dflash_pool : public common_speculative_impl {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    common_params_speculative params;
    int32_t n_draft_max = 0;
    llama_seq_id last_seq = -1; // the sequence whose block is in ctx_dft's incremental draft-KV plan

    std::vector<std::unique_ptr<common_speculative_state_dflash>> states;
    common_speculative_feature_view features;
    llama_tokens out;
    const llama_tokens no_prompt;

    common_speculative_impl_dflash_pool(common_speculative_type type, const common_params_speculative & p, uint32_t n_seq)
        : common_speculative_impl(type, n_seq)
        , ctx_tgt(p.ctx_tgt)
        , ctx_dft(p.ctx_dft)
        , params(p)
    {
        GGML_ASSERT(ctx_tgt != nullptr && ctx_dft != nullptr);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const llama_model * model_dft = llama_get_model(ctx_dft);
        const int32_t cross_ctx = llama_get_dflash_visible_cross_ctx(ctx_dft);
        if (cross_ctx <= 0) {
            throw std::runtime_error("the DFlash draft context exposes no cross-context (it was not created with "
                                     "dflash_query_capacity; see common_speculative_init_result)");
        }
        const bool is_dspark = type == COMMON_SPECULATIVE_TYPE_DSPARK;
        const bool is_dsv4   = is_dspark && llama_model_is_deepseek4(model_tgt);
        const int32_t query_capacity = is_dsv4 ? std::max(1, params.n_max) : 0;
        for (uint32_t s = 0; s < n_seq; ++s) {
            auto st = std::make_unique<common_speculative_state_dflash>(type, ctx_tgt, ctx_dft, cross_ctx, query_capacity,
                                                                        params.n_max, /*install_target_capture=*/ s == 0);
            st->owns_ctx_dft = false;
            if (!st->ready) {
                llama_clear_dflash_capture(ctx_tgt);
                throw std::runtime_error("the DFlash draft is incompatible with the target (see the error above)");
            }
            std::vector<float>().swap(st->target_window_ring);
            states.push_back(std::move(st));
        }
        const int32_t block = llama_model_dflash_block_size(model_dft);
        n_draft_max = is_dsv4 ? states[0]->query_capacity : (is_dspark ? block : block - 1);
        if (params.n_max > n_draft_max) {
            LOG_WRN("%s: requested n_max=%d exceeds what the trained block of %d yields, clamping to %d\n", __func__,
                    params.n_max, block, n_draft_max);
            params.n_max = n_draft_max;
        }
        params.n_min = std::min(params.n_min, params.n_max);
        LOG_INF("%s: adding speculative implementation '%s' (per-sequence, draft context shared by %u sequence(s), block=%d, "
                "n_max=%d, cross_ctx=%d, dflash2=%s)\n", __func__, common_speculative_type_to_str(type).c_str(), n_seq,
                block, params.n_max, cross_ctx, states[0]->is_dflash2 ? "yes" : "no");
    }

    ~common_speculative_impl_dflash_pool() override {
        states.clear();
        llama_clear_dflash_capture(ctx_tgt);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        // the prompt's rows stay: a reused prefix keeps its rows, a new prompt replaces the suffix in process()
        states[seq_id]->begin(prompt);
        last_seq = -1;
    }

    bool process(const llama_batch & b) override {
        if (b.n_tokens <= 0 || b.pos == nullptr) {
            return true;
        }
        if (!llama_spec_get_dflash_feature_view(ctx_tgt, b, features)) {
            LOG_DBG("%s: no DFlash target features for this %d-token batch\n", __func__, b.n_tokens);
            return true;
        }
        std::vector<int32_t>   n_rows(n_seq, 0);
        std::vector<llama_pos> pos_lo(n_seq, 0), pos_hi(n_seq, 0);
        for (const auto & row : features.rows) {
            if (row.seq_id < 0 || row.seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            const llama_seq_id s = row.seq_id;
            pos_lo[s] = n_rows[s] == 0 ? row.pos : std::min(pos_lo[s], row.pos);
            pos_hi[s] = n_rows[s] == 0 ? row.pos : std::max(pos_hi[s], row.pos);
            n_rows[s]++;
        }
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (n_rows[s] == 0) {
                continue;
            }
            if (b.embd != nullptr && n_rows[s] > 1 && pos_lo[s] == pos_hi[s]) {
                continue; // an M-RoPE image pins its rows to one position (upstream fa6769818)
            }
            if (!dflash_append_target_features(*states[s], features, s)) {
                LOG_DBG("%s: seq %d: DFlash feature rows not appended (width %d, positions not increasing?)\n", __func__,
                        (int) s, features.width);
            }
            if (last_seq == s) {
                last_seq = -1; // the shared draft-KV plan refers to this sequence's previous window
            }
        }
        return true;
    }

    bool warned_concurrent = false;

    void draft(common_speculative_draft_params_vec & dparams) override {
        GGML_ASSERT(dparams.size() == n_seq);
        // MITIGATION (row S69-SPEC-POOL-DFLASH-CONCURRENT-VERIFY-CORRUPTS-TARGET-2026-11-05-01): with two sequences in flight
        // and n_max > 1, DFlash verify batches yield target logits the target itself scores ~9 nats lower on a fresh pass
        // (MEASURED 2026-11-05, tests/test-server-spec-pool-dflash.py; the draft-model, suffix and no-spec -np 2 controls are
        // clean, DFlash alone in a slot is clean, round-robin one-drafter-per-round is NOT). Until the cause is found, DFlash
        // drafts only in a round where it is the only drafting sequence. LLAMA_DFLASH_POOL_CONCURRENT=1 restores concurrent
        // drafting (to reproduce the defect).
        static const bool allow_concurrent = [] {
            const char * v = std::getenv("LLAMA_DFLASH_POOL_CONCURRENT");
            return v != nullptr && v[0] == '1';
        }();
        int n_drafting_seqs = 0;
        for (const auto & d : dparams) {
            n_drafting_seqs += d.drafting ? 1 : 0;
        }
        if (n_drafting_seqs > 1 && !allow_concurrent) {
            if (!warned_concurrent) {
                warned_concurrent = true;
                LOG_WRN("%s: %d sequences drafting at once: DFlash drafts only while a single sequence is active (see row "
                        "S69-SPEC-POOL-DFLASH-CONCURRENT-VERIFY-CORRUPTS-TARGET-2026-11-05-01)\n", __func__, n_drafting_seqs);
            }
            return;
        }
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            auto & dp = dparams[s];
            if (!dp.drafting) {
                continue;
            }
            auto & st = *states[s];
            const int32_t n_dropped_rows = dflash_truncate_target_features(st, dp.pos0);
            static const bool round_trace = getenv("DFLASH_ROUND_TRACE") != nullptr; // diagnostic, off by default
            if (round_trace) {
                LOG_INF("%s: ROUND seq=%d pos0=%d dropped=%d rows=%d newest=%d version=%llu synced=%d/%llu\n", __func__, (int) s,
                        (int) dp.pos0, n_dropped_rows, st.target_window_rows, (int) st.last_target_pos,
                        (unsigned long long) st.target_window_version, (int) st.engine_synced,
                        (unsigned long long) st.engine_synced_version);
            }
            if (st.last_target_pos != dp.pos0 - 1) {
                LOG_DBG("%s: seq %d: no DFlash feature row at %d (newest %d)\n", __func__, (int) s, (int) dp.pos0 - 1,
                        (int) st.last_target_pos);
                continue;
            }
            if (last_seq != s) {
                llama_reset_dflash_kv_cache_state(ctx_dft);
                last_seq = s;
            }
            common_params_speculative p = params;
            if (dp.n_max > 0) {
                p.n_max = std::min(p.n_max, dp.n_max);
            }
            out.clear();
            st.draft(p, no_prompt, dp.id_last, out);
            if (round_trace) {
                std::string pieces = common_token_to_piece(ctx_tgt, dp.id_last) + " |";
                for (const llama_token t : out) {
                    pieces += " [" + common_token_to_piece(ctx_tgt, t) + "]";
                }
                LOG_INF("%s: DRAFT seq=%d pos0=%d: %s\n", __func__, (int) s, (int) dp.pos0, pieces.c_str());
            }
            const int32_t n_min = dp.n_min >= 0 ? dp.n_min : params.n_min;
            if (out.empty() || (int32_t) out.size() < n_min) {
                continue;
            }
            dp.result->insert(dp.result->end(), out.begin(), out.end());
            if (dp.dists != nullptr && st.proposal_dists.size() == out.size()) {
                for (const auto & d : st.proposal_dists) {
                    common_spec_dist q;
                    q.ids   = d.ids;
                    q.probs = d.probs;
                    dp.dists->push_back(std::move(q));
                }
            }
        }
    }

    // the window is trimmed lazily in draft(): the next verify batch, or the next draft's truncation, settles it
    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {}

    int32_t stage_n_max() const override { return params.n_max; }
};

// LiLiCorr drafting (--spec-type dflash with a LiLiCorr draft GGUF, i.e. one carrying lilicorr.* keys). Row
// S69-FORK-DFLASH-ENGINE-LACKS-LILICORR-MROPE-DSPARK-NEMOTRON-UPSTREAM-MODEL-UNREACHED-2026-11-05-01.
//
// SOURCE-READ: upstream 977c48a25 common/speculative.cpp:984-1423 (common_speculative_impl_draft_dflash, its is_lilicorr arms:
// :1070-1079 ring sizing, :1145 nextn unmasked, :1184-1285 process() + the projected-context ring, :1287-1423 draft() + the
// greedy lattice walk) and src/models/dflash.cpp:852-955 + :1113 (the class graph this drives: an embd batch fuses the target
// features through fc + enc norm, injects K/V and exposes the fused rows as h_nextn; a token block of exactly block_size per
// sequence with a state row runs build_lilicorr_scores and writes the [top_k ids | top_k x top_k scores] lattice to h_nextn).
// The draft model is the src/models class (llama_model_create_dflash_class), so ctx_dft decodes through upstream's class path.
//
// ADAPTED (the one difference): upstream reads the target features from llama_get_embeddings_layer_inp(), filled only by class
// graphs; the fork's targets (qwen3 among them) decode through the legacy builder, so the features come from the fork's DFlash
// capture instead (llama_set_dflash_capture_layers + llama_spec_get_dflash_feature_view, the same rows the fork DFlash engine
// uses). Upstream's GGUF ids are one-based layer-INPUT ids; the capture takes zero-based layer-OUTPUT ids, hence id - 1, the
// conversion the legacy DFlash loader also applies (src/llama-hparams.cpp, "target_layers must use one-based IDs").
// Draft KV: upstream's server drops ctx_dft rows past the accepted prefix (server-context.cpp:3824); the fork server does not, so
// this impl drops them itself: rows at or past a sequence's first injected position before each injection, and rows at or past
// pos0 before each drafted block.
struct common_speculative_impl_lilicorr : public common_speculative_impl {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    common_params_speculative params;

    common_batch batch;        // drafting blocks
    common_batch batch_inject; // target features for KV cache injection
    std::vector<float> features_buf;
    common_speculative_feature_view features;

    int32_t n_embd_dec = 0;
    int32_t n_embd_enc = 0;
    int32_t block_size = 16;
    llama_token mask_token_id = 0;
    int32_t top_k = 0;
    bool causal_attn = false;

    size_t rows = 0;               // ring rows per sequence (block_size + 1)
    std::vector<float> ring;       // [n_seq * rows, n_embd_dec] projected context rows
    std::vector<llama_pos> ring_pos;

    static bool is_lilicorr(const llama_model * model) {
        char buf[32] = {};
        return model != nullptr && llama_model_meta_val_str(model, "lilicorr.candidate_topk", buf, sizeof(buf)) >= 0 && std::atoi(buf) > 0;
    }

    common_speculative_impl_lilicorr(common_speculative_type type, const common_params_speculative & p, uint32_t n_seq)
        : common_speculative_impl(type, n_seq)
        , ctx_tgt(p.ctx_tgt)
        , ctx_dft(p.ctx_dft)
        , params(p)
    {
        GGML_ASSERT(ctx_tgt != nullptr && ctx_dft != nullptr);
        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        const int32_t * ids = llama_model_target_layer_ids(model_dft);
        const uint32_t n_ids = llama_model_target_layer_ids_n(model_dft);
        if (ids == nullptr || n_ids == 0) {
            throw std::runtime_error("the LiLiCorr draft has no target_layers");
        }
        std::vector<int32_t> capture(n_ids);
        for (uint32_t k = 0; k < n_ids; ++k) {
            if (ids[k] <= 0) {
                throw std::runtime_error("LiLiCorr target_layers must use one-based ids");
            }
            capture[k] = ids[k] - 1;
        }

        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) n_ids * llama_model_n_embd(model_tgt);

        char buf[32] = {};
        if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
            block_size = std::atoi(buf);
        }
        if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
            causal_attn = std::strcmp(buf, "true") == 0;
        }
        if (llama_model_meta_val_str(model_dft, "lilicorr.candidate_topk", buf, sizeof(buf)) >= 0) {
            top_k = std::atoi(buf);
        }
        if (block_size < 2 || top_k <= 0) {
            throw std::runtime_error("invalid LiLiCorr metadata (block_size / candidate_topk)");
        }
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));

        // the lattice yields at most block_size - 1 draft tokens
        const int32_t n_draft_max = block_size - 1;
        if (params.n_max > n_draft_max) {
            LOG_WRN("%s: requested n_max=%d exceeds the trained block of %d, clamping to %d\n", __func__, params.n_max, block_size, n_draft_max);
            params.n_max = n_draft_max;
        }
        params.n_min = std::min(params.n_min, params.n_max);

        rows = (size_t) block_size + 1;
        ring.assign((size_t) n_seq * rows * (size_t) n_embd_dec, 0.0f);
        ring_pos.assign((size_t) n_seq * rows, -1);

        batch        = common_batch(ctx_dft);
        batch_inject = common_batch(ctx_dft);

        if (!llama_set_dflash_capture_layers(ctx_tgt, capture.data(), (int32_t) capture.size())) {
            throw std::runtime_error("failed to configure the target feature capture for LiLiCorr");
        }
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ false);
        llama_set_causal_attn(ctx_dft, causal_attn);

        LOG_INF("%s: adding speculative implementation 'lilicorr' (n_seq=%u, block=%d, top_k=%d, n_max=%d, n_extract=%u, n_embd_enc=%d)\n",
                __func__, n_seq, block_size, top_k, params.n_max, n_ids, n_embd_enc);
    }

    ~common_speculative_impl_lilicorr() override {
        llama_clear_dflash_capture(ctx_tgt);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        std::fill(ring_pos.begin() + (size_t) seq_id * rows, ring_pos.begin() + (size_t) (seq_id + 1) * rows, -1);
    }

    bool process(const llama_batch & b) override {
        if (b.n_tokens <= 0 || b.pos == nullptr) {
            return true;
        }
        if (!llama_spec_get_dflash_feature_view(ctx_tgt, b, features)) {
            LOG_DBG("%s: no LiLiCorr target features for this %d-token batch\n", __func__, b.n_tokens);
            return true;
        }
        if (features.width != n_embd_enc) {
            LOG_ERR("%s: target feature width %d != draft fc input %d\n", __func__, features.width, n_embd_enc);
            return false;
        }
        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            std::vector<const llama_spec_feature_row_view *> sel;
            for (const auto & row : features.rows) {
                if (row.seq_id == s && row.data != nullptr) {
                    sel.push_back(&row);
                }
            }
            if (sel.empty()) {
                continue;
            }
            // an M-RoPE image pins its rows to one position (upstream fa6769818)
            if (b.embd != nullptr && sel.size() > 1 && sel.front()->pos == sel.back()->pos) {
                continue;
            }
            // drop the previous drafted block and any rejected rows before re-injecting from this position
            llama_memory_seq_rm(llama_get_memory(ctx_dft), s, sel.front()->pos, -1);
            for (size_t off = 0; off < sel.size(); off += (size_t) n_ubatch) {
                const size_t n_chunk = std::min((size_t) n_ubatch, sel.size() - off);
                features_buf.resize(n_chunk * (size_t) n_embd_enc);
                batch_inject.clear();
                for (size_t i = 0; i < n_chunk; ++i) {
                    std::memcpy(features_buf.data() + i * (size_t) n_embd_enc, sel[off + i]->data, (size_t) n_embd_enc * sizeof(float));
                }
                for (size_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = sel[off + i]->pos;
                    const llama_pos pos_arr[4] = { p, p, p, 0 };
                    batch_inject.add_embd({ features_buf.data() + i * (size_t) n_embd_enc, 1, (size_t) n_embd_enc }, pos_arr, s, false);
                }
                const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_inject.get());
                if (rc != 0) {
                    LOG_ERR("%s: llama_process(ctx_dft) failed rc=%d (seq %d, n_tokens=%zu)\n", __func__, rc, (int) s, n_chunk);
                    return false;
                }
                // keep the fused context rows of the newest block_size positions (the state row a drafted block reads)
                const float * projected = llama_get_embeddings_nextn(ctx_dft);
                if (projected == nullptr) {
                    continue;
                }
                const llama_pos pos_min = std::max<llama_pos>(0, sel[off + n_chunk - 1]->pos - block_size);
                for (size_t i = 0; i < n_chunk; ++i) {
                    const llama_pos pos = sel[off + i]->pos;
                    if (pos < pos_min) {
                        continue;
                    }
                    const size_t slot = (size_t) s * rows + (size_t) pos % rows;
                    std::memcpy(ring.data() + slot * (size_t) n_embd_dec, projected + i * (size_t) n_embd_dec, (size_t) n_embd_dec * sizeof(float));
                    ring_pos[slot] = pos;
                }
            }
        }
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        GGML_ASSERT(dparams.size() == n_seq);
        batch.clear();
        std::vector<int32_t> beg(n_seq, -1);
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            auto & dp = dparams[s];
            if (!dp.drafting) {
                continue;
            }
            const llama_pos pos_ctx = dp.pos0 - 1;
            const size_t slot = (size_t) s * rows + (size_t) pos_ctx % rows;
            if (pos_ctx < 0 || ring_pos[slot] != pos_ctx) {
                LOG_DBG("%s: seq %d: no projected context row at %d\n", __func__, (int) s, (int) pos_ctx);
                continue;
            }
            // rows at or past pos0 are rejected drafts of the previous round
            llama_memory_seq_rm(llama_get_memory(ctx_dft), s, dp.pos0, -1);
            beg[s] = batch.size();
            for (int32_t i = 0; i < block_size; ++i) {
                const int32_t idx = batch.add(i == 0 ? dp.id_last : mask_token_id, dp.pos0 + i, s, false);
                batch.set_embd_state(idx, { ring.data() + slot * (size_t) n_embd_dec, 1, (size_t) n_embd_dec });
            }
        }
        if (batch.size() == 0) {
            return;
        }
        const int32_t ret = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (ret != 0) {
            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
            return;
        }
        const float * lattice = llama_get_embeddings_nextn(ctx_dft);
        if (lattice == nullptr) {
            LOG_WRN("%s: the LiLiCorr draft produced no lattice\n", __func__);
            return;
        }
        std::vector<float> probs((size_t) top_k);
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (beg[s] < 0) {
                continue;
            }
            auto & dp = dparams[s];
            const int32_t n_max = dp.n_max > 0 ? std::min(params.n_max, dp.n_max) : params.n_max;
            const int32_t n_min = dp.n_min >= 0 ? dp.n_min : params.n_min;
            llama_tokens out;
            int32_t predecessor = 0;
            for (int32_t i = 1; i < n_max + 1; ++i) {
                const float * row    = lattice + (size_t) (beg[s] + i) * (size_t) n_embd_dec;
                const float * scores = row + top_k + (size_t) predecessor * top_k;
                predecessor = (int32_t) std::distance(scores, std::max_element(scores, scores + top_k));
                if (params.p_min > 0.0f) {
                    // softmax(scores) at the argmax: 1 / sum(exp(s_k - s_max))
                    float sum = 0.0f;
                    for (int32_t k = 0; k < top_k; ++k) {
                        sum += std::exp(scores[k] - scores[predecessor]);
                    }
                    if (1.0f / sum < params.p_min) {
                        break;
                    }
                }
                out.push_back((llama_token) row[predecessor]);
            }
            if ((int32_t) out.size() < n_min || out.empty()) {
                continue;
            }
            dp.result->insert(dp.result->end(), out.begin(), out.end());
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {}

    int32_t stage_n_max() const override { return params.n_max; }
};

// initialization of the speculative decoding system (unsloth/master's per-sequence-pool
// entry point) -- the one llama-server uses: ONE speculator for all slots, impls keyed by seq id.
//
// REWRITTEN 2026-10-17 (row S69-FORK-SERVER-PER-SEQ-SPECULATOR-NO-MTP-NO-DRAFT-POLICY-2026-10-17-01). Stages are built in
// CHAIN order (the order --spec-type names them; common_speculative_draft tries impls in this order), and every requested stage
// this pool cannot build is named in a LOG_WRN instead of being dropped silently. What the pool builds:
//   ngram_map_k, ngram_map_k4v            common_speculative_impl_ngram_map_k (upstream's pooled impl)
//   suffix (no corpus), suffix_automaton  the fork's states, one per sequence (common_speculative_impl_state_per_seq)
//   draft                                 common_speculative_impl_draft, given -md (ctx_tgt + ctx_dft)
//   mtp                                   common_speculative_impl_mtp, given the target's NextN head (ctx_tgt + the MTP ctx_dft)
//   ngram_simple                          the fork's state, one per sequence (common_speculative_impl_state_per_seq)
//   ngram_mod                             common_speculative_impl_ngram_mod_pool (one shared table, per-sequence feed/streak)
//   ngram_cache                           common_speculative_impl_ngram_cache_pool (shared static/dynamic, per-sequence context)
// (the three n-gram stages: row S69-FORK-SERVER-POOL-NGRAM-MOD-SIMPLE-CACHE-DSPARK-IMPLS-2026-10-17-01, 2026-10-18)
//   eagle3                                common_speculative_impl_eagle3_pool, given -md <eagle3 draft> (ctx_tgt + ctx_dft); 1 token per round
// (eagle3: row S69-FORK-SERVER-POOL-EAGLE3-DFLASH-IMPLS-2026-10-17-01, 2026-10-18)
//   dflash, dspark                        common_speculative_impl_dflash_pool, given -md <dflash draft> (ctx_tgt + a ctx_dft created
//                                         with dflash_query_capacity by common_speculative_init_result); added 2026-11-05
// What it does not build, warned: suffix with a --spec-suffix-corpus.
//
// --spec-draft-policy: with a self-speculative first stage and a model stage after it (the only 2-stage shape the chain allows),
// one common_draft_policy per sequence chooses per round whether that slot verifies the lookup draft or the model draft
// (common_speculative_draft / common_speculative_accept below). Any other chain: a warning, and the flag is ignored.
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;
    if (spec == nullptr) {
        return n_max;
    }
    for (const auto & cfg : spec->configs) {
        n_max = std::max(n_max, std::max<int32_t>(0, cfg.stage.n_max));
    }
    for (const auto & impl : spec->impls_seq) {
        if (impl) {
            n_max = std::max(n_max, std::max<int32_t>(0, impl->stage_n_max()));
        }
    }
    return n_max;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    static const std::vector<double> none;
    return spec != nullptr ? spec->synth_probs : none;
}

// synthetic acceptance (upstream --spec-synth-len / --spec-synth-rates, MERGE_HEAD speculative.cpp:2904-2925): the conditional
// acceptance probability of each draft position, read by llama-server's verify step instead of the target's verdict
static void common_speculative_synth_init(common_speculative * result, const common_params_speculative & params) {
    if (result == nullptr || !params.has_synth()) {
        return;
    }
    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result);
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);
    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        LOG_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            LOG_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        LOG_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                result->synth_probs.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }
}

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    const bool have_ctx = params.ctx_tgt != nullptr && params.ctx_dft != nullptr;

    const char * fn = __func__; // inside the lambda below __func__ would name the lambda ("operator()")
    for (const auto & st : params.get_resolved_stages()) {
        const std::string name = common_speculative_type_to_str(st.type);
        auto not_built = [&](const char * why) {
            LOG_WRN("%s: speculative stage '%s' requested but not built by the per-sequence speculator: %s\n", fn, name.c_str(), why);
        };

        if (params.draft_sample && st.type != COMMON_SPECULATIVE_TYPE_NONE && !common_speculative_type_draws_from_q(st.type)) {
            LOG_WRN("%s: --spec-draft-sample has no effect on stage '%s' (only the draft-model and MTP stages draw from q); it drafts deterministically\n",
                    fn, name.c_str());
        }

        try {
            switch (st.type) {
                case COMMON_SPECULATIVE_TYPE_NONE:
                    break;
                case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                    const common_speculative_config config(st, params);
                    impls.push_back(std::make_unique<common_speculative_impl_ngram_map_k>(
                                get_common_ngram_map(config.type, config.params), n_seq));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON: {
                    impls.push_back(std::make_unique<common_speculative_impl_state_per_seq<common_speculative_state_suffix_automaton>>(
                            st.type, n_seq, params.with_stage_overrides(st)));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                    if (!params.suffix_corpus.empty()) {
                        // the corpus would be loaded once per sequence
                        not_built("a suffix tree with --spec-suffix-corpus is not pooled (use suffix_automaton, or drop the corpus)");
                        break;
                    }
                    const int depth = params.suffix_max_depth > 0 ? params.suffix_max_depth : 64;
                    const llama_model * model = params.ctx_tgt != nullptr ? llama_get_model(params.ctx_tgt) : nullptr;
                    impls.push_back(std::make_unique<common_speculative_impl_state_per_seq<common_speculative_state_suffix>>(
                            st.type, n_seq, params.with_stage_overrides(st), depth, std::string(), model,
                            (size_t) std::max<int64_t>(0, params.with_stage_overrides(st).suffix_max_nodes)));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_DRAFT: {
                    if (!have_ctx) {
                        not_built("no draft model context (pass -md, and the caller must hand over ctx_tgt + ctx_dft)");
                        break;
                    }
                    impls.push_back(std::make_unique<common_speculative_impl_draft>(params.with_stage_overrides(st), n_seq));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_MTP: {
                    if (!have_ctx) {
                        not_built("no MTP draft context (the model needs an MTP / NextN head, and the caller must hand over ctx_tgt + ctx_dft)");
                        break;
                    }
                    impls.push_back(std::make_unique<common_speculative_impl_mtp>(params.with_stage_overrides(st), n_seq));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                    // stateless lookup in the slot's own text: the single-context state, one per sequence
                    const common_params_speculative sp = params.with_stage_overrides(st);
                    const common_ngram_map map = get_common_ngram_map(st.type, sp);
                    const common_ngram_simple_config cfg_simple = { map.size_key, map.size_value };
                    impls.push_back(std::make_unique<common_speculative_impl_state_per_seq<common_speculative_state_ngram_simple>>(
                            st.type, n_seq, sp, cfg_simple));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                    impls.push_back(std::make_unique<common_speculative_impl_ngram_mod_pool>(params.with_stage_overrides(st), n_seq));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                    impls.push_back(std::make_unique<common_speculative_impl_ngram_cache_pool>(params.with_stage_overrides(st), n_seq));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                    if (!have_ctx) {
                        not_built("no EAGLE3 draft context (pass -md <eagle3 draft gguf>, and the caller must hand over ctx_tgt + ctx_dft)");
                        break;
                    }
                    if (llama_eagle3_get_d2t(params.ctx_dft, nullptr) == 0 &&
                        llama_n_vocab(llama_get_model(params.ctx_dft)) != llama_n_vocab(llama_get_model(params.ctx_tgt))) {
                        not_built("the draft has no d2t map and its vocab differs from the target's");
                        break;
                    }
                    impls.push_back(std::make_unique<common_speculative_impl_eagle3_pool>(params.with_stage_overrides(st), n_seq));
                    break;
                }
                case COMMON_SPECULATIVE_TYPE_DFLASH:
                case COMMON_SPECULATIVE_TYPE_DSPARK: {
                    if (!have_ctx) {
                        not_built("no DFlash draft context (pass -md <dflash draft gguf>, and the caller must hand over ctx_tgt + ctx_dft)");
                        break;
                    }
                    // a LiLiCorr draft (lilicorr.* keys) runs on upstream's class graph, not the fork DFlash engine
                    if (st.type == COMMON_SPECULATIVE_TYPE_DFLASH &&
                            common_speculative_impl_lilicorr::is_lilicorr(llama_get_model(params.ctx_dft))) {
                        impls.push_back(std::make_unique<common_speculative_impl_lilicorr>(st.type, params.with_stage_overrides(st), n_seq));
                        break;
                    }
                    impls.push_back(std::make_unique<common_speculative_impl_dflash_pool>(st.type, params.with_stage_overrides(st), n_seq));
                    break;
                }
                default:
                    not_built("unknown to the per-sequence pool");
                    break;
            }
        } catch (const std::exception & e) {
            LOG_ERR("%s: speculative stage '%s' failed to initialise: %s\n", __func__, name.c_str(), e.what());
            not_built("its initialisation failed (see the error above)");
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    // ADAPTED 2026-09-13. The /* .field = */ comments made this read like designated
    // initialisation, but they are only comments: aggregate initialisation is POSITIONAL, so
    // these three values were landing on `configs`, `impls` and `checkpoint` -- the first three
    // members of struct common_speculative -- and the compiler reported the first
    // mismatch, common_speculative_draft_params_vec into std::vector<common_speculative_config>.
    // Naming the union-merged fields explicitly is both what was intended and immune to
    // anyone adding a member above them.
    auto * result = new common_speculative();

    result->dparams   = common_speculative_draft_params_vec(n_seq);
    result->impls_seq = std::move(impls);
    result->impl_last = std::vector<common_speculative_impl *>(n_seq, nullptr);

    {
        std::string chain;
        for (const auto & impl : result->impls_seq) {
            chain += (chain.empty() ? "" : ",") + common_speculative_type_to_str(impl->type);
        }
        LOG_INF("%s: per-sequence speculator: %u sequence(s), stages in draft order: %s\n", __func__, n_seq, chain.c_str());
    }

    // --spec-draft-policy, per slot (the single-context engine builds the same policy once, see below)
    if (params.draft_policy) {
        const auto & impls_seq = result->impls_seq;
        bool has_model_after = false;
        for (size_t i = 1; i < impls_seq.size(); ++i) {
            has_model_after = has_model_after || !common_speculative_type_is_self_spec(impls_seq[i]->type);
        }
        if (!impls_seq.empty() && common_speculative_type_is_self_spec(impls_seq[0]->type) && has_model_after) {
            const int max_t = params.get_max_stage_n_max() + 1;
            for (uint32_t s = 0; s < n_seq; ++s) {
                result->policy_seq.push_back(std::make_unique<common_draft_policy>(max_t));
            }
            result->policy_round_seq.assign(n_seq, {});
            LOG_INF("%s: draft policy on (per slot, %u slot(s): lookup stage '%s' vs the model stage chosen each round from learned acceptance and measured cost)\n",
                    __func__, n_seq, common_speculative_type_to_str(impls_seq[0]->type).c_str());
        } else {
            LOG_WRN("%s: --spec-draft-policy needs a self-speculative stage followed by a model stage (e.g. --spec-type suffix,mtp) among the BUILT stages; ignored\n", __func__);
        }
    }

    common_speculative_synth_init(result, params);
    return result;
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls_seq) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}


void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    // --spec-draft-policy (per slot): a round starts at the draft and ends at that slot's accept
    const bool use_policy = !spec->policy_seq.empty();
    if (use_policy) {
        const int64_t t_now = ggml_time_us();
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size() && seq_id < (llama_seq_id) spec->policy_round_seq.size(); ++seq_id) {
            if (dparams[seq_id].drafting) {
                spec->policy_round_seq[seq_id] = {};
                spec->policy_round_seq[seq_id].t_start_us = t_now;
            }
        }
    }

    for (size_t i_impl = 0; i_impl < spec->impls_seq.size(); ++i_impl) {
        auto & impl = spec->impls_seq[i_impl];
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        // the first model stage after this lookup stage: the window the lookup draft competes with
        const common_speculative_impl * model_after = nullptr;
        if (use_policy && common_speculative_type_is_self_spec(impl->type)) {
            for (size_t j = i_impl + 1; j < spec->impls_seq.size(); ++j) {
                if (!common_speculative_type_is_self_spec(spec->impls_seq[j]->type)) {
                    model_after = spec->impls_seq[j].get();
                    break;
                }
            }
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // --spec-draft-sample: dp.dists is parallel to result (one q per drafted token), so every cut of result cuts it
            // too -- but only when the drafter recorded any (n-gram / lookup drafters record none, and an empty vector must
            // stay empty so the verifier falls back to sample-and-match). Upstream 8330e9696 fixed the same shape for
            // result_q ("n-gram drafts rejected at temp > 0 after truncation").
            auto trim_dists = [&dp, &result]() {
                if (dp.dists != nullptr && !dp.dists->empty() && dp.dists->size() > result.size()) {
                    dp.dists->resize(result.size());
                }
            };

            if (dp.n_max > 0 && (int) result.size() > dp.n_max) {
                SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                result.resize(dp.n_max);
                trim_dists();
            }

            // --spec-draft-policy: this slot verifies the lookup draft only when its expected committed tokens per ms beat
            // the model stage's window; otherwise the draft is dropped and the slot stays open for the model stage
            if (model_after != nullptr && !result.empty() && seq_id < (llama_seq_id) spec->policy_seq.size()) {
                int n_model = model_after->stage_n_max();
                if (n_model <= 0) {
                    n_model = 1;
                }
                if (dp.n_max > 0) {
                    n_model = std::min(n_model, dp.n_max);
                }
                const int ml    = impl->match_len(seq_id);
                const int match = ml >= 0 ? ml : (int) result.size();
                const auto pk   = spec->policy_seq[seq_id]->choose(n_model + 1, (int) result.size(), match);
                if (!pk.lookup) {
                    result.clear();
                    trim_dists();
                } else {
                    if ((int) result.size() > pk.t - 1) {
                        result.resize(std::max(1, pk.t - 1));
                        trim_dists();
                    }
                    spec->policy_round_seq[seq_id].match = match;
                }
            }

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (use_policy && seq_id < (llama_seq_id) spec->policy_round_seq.size()) {
                    auto & round = spec->policy_round_seq[seq_id];
                    round.valid  = true;
                    round.lookup = common_speculative_type_is_self_spec(impl->type);
                    round.t      = (int) result.size() + 1;
                    if (!round.lookup) {
                        round.match = 0;
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}


void common_speculative_async_start(common_speculative * spec, const std::vector<common_speculative_async_req> & reqs) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls_seq) {
        // only the implementation that produced a seq's draft may evaluate/extend it
        std::vector<common_speculative_async_req> mine;
        for (const auto & r : reqs) {
            if (r.seq_id >= 0 && (size_t) r.seq_id < spec->impl_last.size() && spec->impl_last[r.seq_id] == impl.get()) {
                mine.push_back(r);
            }
        }
        if (!mine.empty()) {
            impl->async_start(mine);
        }
    }
}

void common_speculative_async_cancel(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls_seq) {
        impl->async_cancel();
    }
}

bool common_speculative_async_get_stats(const common_speculative * spec, common_speculative_async_stats & out) {
    if (spec == nullptr) {
        return false;
    }
    for (const auto & impl : spec->impls_seq) {
        if (impl->async_stats(out)) {
            return true;
        }
    }
    return false;
}

std::vector<common_speculative_async_item> common_speculative_async_stats_items(const common_speculative_async_stats & st) {
    std::vector<common_speculative_async_item> v;
    auto c = [&](const char * n, const char * h, uint64_t x) { v.push_back({ n, h, (double) x, false }); };
    auto g = [&](const char * n, const char * h, double x)   { v.push_back({ n, h, x, true }); };
    c("rounds_total",           "--spec-async: sequence-rounds handed to the draft worker", st.n_seq_rounds);
    c("hit_total",              "--spec-async: next draft served from the lookahead", st.n_hit);
    c("nola_total",             "--spec-async: rounds without a lookahead attempt (gate off)", st.n_o_nola);
    c("laempty_total",          "--spec-async: lookahead attempted but empty (p_min / step 0)", st.n_o_laempty);
    c("notfull_total",          "--spec-async: target did not accept the whole draft (misprediction)", st.n_o_notfull);
    c("c1miss_total",           "--spec-async: whole draft accepted but target token != lookahead c1 (misprediction)", st.n_o_c1miss);
    c("trunc_total",            "--spec-async: hit missed only because the join cut the lookahead (full accept, c1 matched, la.size() <= 1)", st.n_o_trunc);
    c("guard_total",            "--spec-async: usable lookahead dropped by a guard in draft()", st.n_o_guard);
    c("nodraft_total",          "--spec-async: rounds that ended without a following draft (request ended)", st.n_o_nodraft);
    c("fallback_total",         "--spec-async: rounds process() re-evaluated itself", st.n_o_fallback);
    c("cancel_total",           "--spec-async: rounds cancelled by a target decode error", st.n_o_cancel);
    c("dropped_total",          "--spec-async: rounds never consumed by draft()", st.n_o_dropped);
    g("open_rounds",            "--spec-async: rounds handed out whose outcome is not decided yet", (double) st.n_open);
    c("full_accept_total",      "--spec-async: rounds where the target accepted the whole draft", st.n_full);
    c("c1_match_total",         "--spec-async: full accepts whose target token equalled the lookahead c1", st.n_c1_match);
    c("rollback_total",         "--spec-async: real mispredictions (notfull + c1miss)", st.n_rollback);
    for (int i = 0; i < 11; ++i) {
        static const char * nm[11] = { "lookahead_len_0_total", "lookahead_len_1_total", "lookahead_len_2_total", "lookahead_len_3_total",
            "lookahead_len_4_total", "lookahead_len_5_total", "lookahead_len_6_total", "lookahead_len_7_total", "lookahead_len_8_total",
            "lookahead_len_9_total", "lookahead_len_10plus_total" };
        c(nm[i], "--spec-async: rounds whose lookahead had this many tokens at the join", st.la_hist[i]);
    }
    for (int i = 0; i < 6; ++i) {
        static const char * nm[6] = { "c1_rank_0_total", "c1_rank_1_total", "c1_rank_2_total", "c1_rank_3_total", "c1_rank_4_total", "c1_rank_absent_total" };
        c(nm[i], "--spec-async: rank of the target's token in the draft top-5 at c1 after a full accept", st.c1_rank[i]);
    }
    g("step0_seconds_ema",      "--spec-async: EMA of the worker's step 0 (draft-context evaluation of [id_last, draft])", st.ema_s0_us / 1e6);
    g("verify_seconds_ema",     "--spec-async: EMA of the verify (async_start to join)", st.ema_v_us / 1e6);
    g("draft_step_seconds_ema", "--spec-async: EMA of one lookahead decode step", st.ema_d_us / 1e6);
    c("informed_joins_total",   "--spec-async-informed-join: joins decided from the verify rows", st.n_inf_rounds);
    c("informed_wait_total",    "--spec-async-informed-join: joins that let the worker keep going", st.n_inf_wait);
    c("informed_stop_total",    "--spec-async-informed-join: joins that stopped the worker at once", st.n_inf_stop);
    c("informed_legacy_total",  "--spec-async-informed-join: joins that used the wait_ms rule (non-greedy / unreadable rows)", st.n_inf_legacy);
    c("informed_wait_hit_total",  "--spec-async-informed-join: waited seqs whose lookahead was used", st.n_inf_wait_hit);
    c("informed_wait_miss_total", "--spec-async-informed-join: waited seqs whose lookahead was not used", st.n_inf_wait_miss);
    g("informed_wait_seconds_total", "--spec-async-informed-join: time blocked in informed waits", st.t_inf_wait_us / 1e6);
    return v;
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    // --spec-draft-policy (per slot): the round that started at this slot's draft ends here
    if (!spec->policy_seq.empty() && seq_id >= 0 && seq_id < (llama_seq_id) spec->policy_round_seq.size()) {
        auto & round = spec->policy_round_seq[seq_id];
        if (round.valid) {
            const double round_ms = (ggml_time_us() - round.t_start_us) / 1000.0;
            spec->policy_seq[seq_id]->observe(round.lookup, round.t, (int) n_accepted, round.match, round_ms);
            round.valid = false;
        }
    }

    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls_seq) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state

bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls_seq) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls_seq) {
        impl->set_state(seq_id, data);
    }
}


void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    // --spec-async trace: one line per request end, cumulative since server start
    {
        common_speculative_async_stats st;
        if (common_speculative_async_get_stats(spec, st)) {
            LOG_INF("spec-async: rounds=%llu lookahead_rounds=%llu step0=%llu hit=%llu rollback=%llu fallback=%llu lookahead_tokens=%llu "
                    "worker_ms=%.1f wait_ms=%.1f sync_drafts=%llu sync_draft_ms=%.1f\n",
                    (unsigned long long) st.n_rounds, (unsigned long long) st.n_la_rounds, (unsigned long long) st.n_step0, (unsigned long long) st.n_hit,
                    (unsigned long long) st.n_rollback, (unsigned long long) st.n_fallback, (unsigned long long) st.n_la_tokens,
                    st.t_worker_us / 1000.0, st.t_wait_us / 1000.0, (unsigned long long) st.n_sync_drafts, st.t_sync_draft_us / 1000.0);
            // S69-SPECASYNC-DIAG-COUNTERS-2026-11-06-01: every sequence-round ends in exactly one of hit/nola/laempty/notfull/c1miss/trunc/guard/
            // nodraft/fallback/cancel/dropped (+ open); rollback = notfull + c1miss only; la_hist = la.size() 0..9,10+ at join;
            // c1_rank = rank of the target's token in the draft top-5 at c1 after a full accept (5 = absent); ms = EMAs of step 0 / verify / draft step.
            LOG_INF("spec-async-diag: seq_rounds=%llu hit=%llu nola=%llu laempty=%llu notfull=%llu c1miss=%llu trunc=%llu guard=%llu nodraft=%llu "
                    "fallback=%llu cancel=%llu dropped=%llu open=%llu full=%llu c1_match=%llu rollback=%llu "
                    "la_hist=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu c1_rank=%llu,%llu,%llu,%llu,%llu,%llu "
                    "s0_ms=%.2f v_ms=%.2f d_ms=%.2f inf_rounds=%llu inf_wait=%llu inf_stop=%llu inf_legacy=%llu inf_wait_hit=%llu inf_wait_miss=%llu inf_wait_ms=%.1f\n",
                    (unsigned long long) st.n_seq_rounds, (unsigned long long) st.n_hit, (unsigned long long) st.n_o_nola, (unsigned long long) st.n_o_laempty,
                    (unsigned long long) st.n_o_notfull, (unsigned long long) st.n_o_c1miss, (unsigned long long) st.n_o_trunc, (unsigned long long) st.n_o_guard,
                    (unsigned long long) st.n_o_nodraft, (unsigned long long) st.n_o_fallback, (unsigned long long) st.n_o_cancel, (unsigned long long) st.n_o_dropped,
                    (unsigned long long) st.n_open, (unsigned long long) st.n_full, (unsigned long long) st.n_c1_match, (unsigned long long) st.n_rollback,
                    (unsigned long long) st.la_hist[0], (unsigned long long) st.la_hist[1], (unsigned long long) st.la_hist[2], (unsigned long long) st.la_hist[3],
                    (unsigned long long) st.la_hist[4], (unsigned long long) st.la_hist[5], (unsigned long long) st.la_hist[6], (unsigned long long) st.la_hist[7],
                    (unsigned long long) st.la_hist[8], (unsigned long long) st.la_hist[9], (unsigned long long) st.la_hist[10],
                    (unsigned long long) st.c1_rank[0], (unsigned long long) st.c1_rank[1], (unsigned long long) st.c1_rank[2], (unsigned long long) st.c1_rank[3],
                    (unsigned long long) st.c1_rank[4], (unsigned long long) st.c1_rank[5],
                    st.ema_s0_us / 1000.0, st.ema_v_us / 1000.0, st.ema_d_us / 1000.0,
                    (unsigned long long) st.n_inf_rounds, (unsigned long long) st.n_inf_wait, (unsigned long long) st.n_inf_stop, (unsigned long long) st.n_inf_legacy,
                    (unsigned long long) st.n_inf_wait_hit, (unsigned long long) st.n_inf_wait_miss, st.t_inf_wait_us / 1000.0);
        }
    }

    for (const auto & impl : spec->impls_seq) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}


static bool common_speculative_stage_chain_matches(
        const std::vector<common_speculative_stage_params> & stages,
        const std::vector<common_speculative_config> & configs) {
    if (stages.size() != configs.size()) {
        return false;
    }

    for (size_t i = 0; i < stages.size(); ++i) {
        if (stages[i].type != configs[i].type) {
            return false;
        }
    }

    return true;
}

static common_params_speculative common_speculative_get_runtime_params(
        const common_speculative_config & config,
        const common_params_speculative & params,
        const common_speculative_stage_params & stage) {
    common_params_speculative result = config.params;

    result.type = config.type;
    result.n_max = stage.has_n_max_override() ? stage.n_max : params.n_max;
    result.n_min = stage.has_n_min_override() ? stage.n_min : params.n_min;
    result.p_min = stage.has_p_min_override() ? stage.p_min : params.p_min;
    result.p_min_explicit = stage.has_p_min_override() || params.p_min_explicit;
    result.mtp_heads = stage.has_mtp_heads_override() ? stage.mtp_heads : params.mtp_heads;
    result.draft_temperature = params.draft_temperature;
    result.draft_seed = params.draft_seed;

    if (config.type == COMMON_SPECULATIVE_TYPE_SUFFIX || config.type == COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON) {
        result.suffix_min_match_len = stage.has_suffix_min_match_len_override()
            ? stage.suffix_min_match_len
            : params.suffix_min_match_len;
    }

    result.n_max = std::max(result.n_max, 0);
    result.n_min = std::max(0, std::min(result.n_min, result.n_max));
    result.mtp_heads = std::max(result.mtp_heads, 0);
    result.stages.clear();

    return result;
}

bool common_speculative_mtp_requires_fresh_warmup(const common_speculative * spec) {
    const auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr || mtp_state->n_heads_model <= 1) {
        return false;
    }

    // drafting with more heads than the cached prefix was warmed with would read
    // never-written deeper-head cache rows; the caller must reprocess from position 0
    return mtp_state->resolved_heads() > mtp_state->mtp_warmed_heads && mtp_state->mtp_warmed_heads > 0;
}

void common_speculative_prepare_request(common_speculative * spec, common_params_speculative & params) {
    if (spec == nullptr) {
        return;
    }

    const auto runtime_stages = params.get_resolved_stages();
    const bool use_runtime_stage_overrides = common_speculative_stage_chain_matches(runtime_stages, spec->configs);

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        auto * mtp_state = dynamic_cast<common_speculative_state_mtp *>(spec->impls[i].get());
        if (mtp_state == nullptr) {
            continue;
        }

        const auto & runtime_stage = use_runtime_stage_overrides ? runtime_stages[i] : spec->configs[i].stage;
        common_params_speculative impl_params = common_speculative_get_runtime_params(spec->configs[i], params, runtime_stage);
        impl_params.draft_temperature = spec->draft_temperature;
        impl_params.draft_seed = spec->draft_seed;
        mtp_state->mtp_heads_active = std::max<int32_t>(0, impl_params.mtp_heads);
    }
}

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // upstream 83c2d012a removed these flags: the dynamic cache is written at destruction whenever a path is set (the state
    // turns save_dynamic on itself then); the static cache is read-only input
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        case COMMON_SPECULATIVE_TYPE_DSPARK:        return "dspark";
        case COMMON_SPECULATIVE_TYPE_MTP:           return "mtp";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_SUFFIX:        return "suffix";
        case COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON: return "suffix_automaton";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    std::string normalized = name;
    std::replace(normalized.begin(), normalized.end(), '-', '_');

    const auto it = common_speculative_type_from_name_map.find(normalized);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    bool res = true;

    llama_kv_cache_clear(ctx_tgt);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_kv_cache_seq_rm(ctx_tgt, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = false;
        goto done;
    }

done:
    llama_kv_cache_clear(ctx_tgt);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    // A draft model whose ARCHITECTURE is eagle3 must run as the EAGLE3 stage, never the generic DRAFT stage: the
    // generic stage prefills the draft with the prompt, and on an EAGLE3 head (a single-layer, feature-driven
    // decoder) that prefill runs build_eagle3 with no target features set -> access violation on the first decode
    // (root-caused via E3-TRACE; DEC-S69-EAGLE3-AUTODETECT-SPECTYPE). HEAD only auto-selects EAGLE3 when a HF
    // sidecar named "eagle3-*" is resolved (arg.cpp); a local `-md eagle3.gguf` with no --spec-type fell through
    // to DRAFT. Decided here by the model's own general.architecture, so it covers every way the draft arrives.
    // Recovered 2026-09-27 from stranded commit 3e367ef6b (it was filed as a "checkpoint").
    if (params.model_dft) {
        char arch[64] = {0};
        if (llama_model_meta_val_str(params.model_dft, "general.architecture", arch, sizeof(arch)) > 0 &&
                std::string(arch) == "eagle3") {
            auto resolved = params.get_resolved_stages();
            bool promoted = false;
            const bool has_eagle3 = std::any_of(resolved.begin(), resolved.end(), [](const common_speculative_stage_params & s) {
                return s.type == COMMON_SPECULATIVE_TYPE_EAGLE3;
            });
            if (!has_eagle3) {
                for (auto & s : resolved) {
                    if (s.type == COMMON_SPECULATIVE_TYPE_DRAFT) {
                        s.type = COMMON_SPECULATIVE_TYPE_EAGLE3;
                        promoted = true;
                    }
                }
            }
            if (promoted) {
                LOG_INF("%s: draft model architecture is eagle3 -> running the EAGLE3 stage instead of generic DRAFT\n", __func__);
                params.stages = resolved;
            }
        }
    }

    std::string chain_error;
    if (!common_speculative_validate_chain(params, &chain_error)) {
        LOG_ERR("%s: invalid speculative stage chain: %s\n", __func__, chain_error.c_str());
        return nullptr;
    }

    const auto stages = params.get_resolved_stages();
    const llama_model * target_model = llama_get_model(ctx_tgt);
    const bool is_dsv4_target = llama_model_is_deepseek4(target_model);
    int32_t dsv4_dspark_query_capacity = 0;
    for (const auto & stage : stages) {
        if (stage.type != COMMON_SPECULATIVE_TYPE_DSPARK || !is_dsv4_target) {
            continue;
        }

        const int32_t stage_n_max = params.with_stage_overrides(stage).n_max;
        dsv4_dspark_query_capacity = std::max(dsv4_dspark_query_capacity, std::max(1, stage_n_max));
    }
    if (params.model_dft && llama_model_is_gemma4_mtp_assistant(params.model_dft)) {
        const bool has_draft_stage = std::any_of(stages.begin(), stages.end(), [](const common_speculative_stage_params & stage) {
            return stage.type == COMMON_SPECULATIVE_TYPE_DRAFT;
        });

        if (has_draft_stage) {
            LOG_ERR("%s: Gemma4 assistant models only support MTP stages; omit -md for self-spec-only runs or use -mtp/--spec-stage mtp for assistant-backed MTP\n", __func__);
            return nullptr;
        }
    }

    const bool has_dflash_stage = std::any_of(stages.begin(), stages.end(), [](const common_speculative_stage_params & stage) {
        return common_speculative_type_is_dflash_family(stage.type);
    });

    const bool needs_draft_ctx = std::any_of(stages.begin(), stages.end(), [&params](const common_speculative_stage_params & stage) {
        return stage.type == COMMON_SPECULATIVE_TYPE_DRAFT ||
               common_speculative_type_is_dflash_family(stage.type) ||
               stage.type == COMMON_SPECULATIVE_TYPE_EAGLE3 ||   // S69 #33: eagle3 draft head needs its own ctx_dft
               (stage.type == COMMON_SPECULATIVE_TYPE_MTP && params.model_dft != nullptr);
    });

    llama_context * ctx_dft = nullptr;
    int32_t dflash_cross_ctx = 0;

    if (needs_draft_ctx) {
        if (!params.model_dft) {
            LOG_ERR("%s: draft speculative stage requires a loaded draft model\n", __func__);
            return nullptr;
        }

        llama_context_params cparams_dft = params.cparams_dft;

        if (has_dflash_stage) {
            if (!llama_model_share_dflash_io_tensors(params.model_dft, llama_get_model(ctx_tgt))) {
                LOG_ERR("%s: failed to share target IO tensors with DFlash draft model\n", __func__);
                return nullptr;
            }

            const int32_t block_size = llama_model_dflash_block_size(params.model_dft);
            if (block_size <= 0) {
                LOG_ERR("%s: invalid DFlash draft block size\n", __func__);
                return nullptr;
            }

            const int32_t query_capacity = std::max(block_size, dsv4_dspark_query_capacity);
            const int32_t target_ctx = std::max<int32_t>(1, llama_n_ctx(ctx_tgt));
            const int32_t requested_draft_ctx = cparams_dft.n_ctx > 0
                    ? (int32_t) cparams_dft.n_ctx : target_ctx;
            const int32_t effective_draft_ctx = std::min(target_ctx, requested_draft_ctx);
            if (requested_draft_ctx > target_ctx) {
                LOG_INF("%s: DFlash draft context %d exceeds target context %d, clamping to target capacity\n",
                        __func__, requested_draft_ctx, target_ctx);
            }
            const int32_t logical_cross_ctx = effective_draft_ctx - query_capacity;
            if (logical_cross_ctx <= 0) {
                LOG_ERR("%s: invalid DFlash draft context size draft=%d target=%d query_capacity=%d, draft context must exceed the query block\n",
                        __func__, requested_draft_ctx, target_ctx, query_capacity);
                return nullptr;
            }

            cparams_dft.n_ctx = (uint32_t) effective_draft_ctx;
            cparams_dft.dflash_query_capacity = query_capacity;
        }

        if (!has_dflash_stage && !llama_model_share_qwen4exp_mtp_tensors(
                    params.model_dft, llama_get_model(ctx_tgt))) {
            LOG_ERR("%s: failed to share target IO tensors with the qwen4exp shared MTP companion\n", __func__);
            return nullptr;
        }

        ctx_dft = llama_init_from_model(params.model_dft, cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
        if (has_dflash_stage) {
            dflash_cross_ctx = llama_get_dflash_visible_cross_ctx(ctx_dft);
            if (dflash_cross_ctx <= 0) {
                LOG_ERR("%s: DFlash context did not expose a valid resolved physical cross-context\n", __func__);
                llama_free(ctx_dft);
                return nullptr;
            }
        }
    }

    // Compute the implementations to use based on the resolved stage chain.
    std::vector<common_speculative_config> configs = {};
    configs.reserve(stages.size());

    for (const auto & stage : stages) {
        common_params_speculative stage_params = params.with_stage_overrides(stage);

        if (stage.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD && !stage_params.ngram_mod) {
            stage_params.ngram_mod = std::make_shared<common_ngram_mod>(stage_params.ngram_size_n, 4*1024*1024);

            LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                    stage_params.ngram_size_n, stage_params.ngram_mod->size(),
                    (float)(stage_params.ngram_mod->size_bytes())/1024/1024);

            if (stage_params.ngram_size_n < 16) {
                LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, stage_params.ngram_size_n);
            }
        }

        configs.push_back(common_speculative_config(stage, stage_params));
    }

    if (!configs.empty() && common_speculative_needs_checkpoint(target_model)) {
        const int ckpt_tokens = std::max(1, params.get_max_stage_n_max() + 1);
        const int actual_mode = llama_spec_ckpt_init(ctx_tgt, spec_ckpt, params.spec_ckpt_mode, ckpt_tokens);
        if (actual_mode == LLAMA_SPEC_CKPT_NONE) {
            LOG_ERR("%s: failed to prepare speculative checkpoint mode '%s' during speculative init (max_tokens=%d)\n",
                    __func__,
                    params.spec_ckpt_mode == LLAMA_SPEC_CKPT_PER_STEP ? "per-step" :
                    params.spec_ckpt_mode == LLAMA_SPEC_CKPT_GPU_FALLBACK ? "gpu-fallback" :
                    params.spec_ckpt_mode == LLAMA_SPEC_CKPT_CPU ? "cpu" : "auto",
                    ckpt_tokens);
            if (ctx_dft != nullptr) {
                llama_free(ctx_dft);
            }
            return nullptr;
        }
        llama_spec_ckpt_discard(ctx_tgt, spec_ckpt);
        params.spec_ckpt_mode = actual_mode;
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        if (params.draft_sample && config.type != COMMON_SPECULATIVE_TYPE_NONE && !common_speculative_type_draws_from_q(config.type)) {
            LOG_WRN("%s: --spec-draft-sample has no effect on stage '%s' (only the draft-model and MTP stages draw from q); it drafts deterministically\n",
                    __func__, common_speculative_type_to_str(config.type).c_str());
        }
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ config.params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DSPARK: {
                const int32_t query_capacity = config.type == COMMON_SPECULATIVE_TYPE_DSPARK && is_dsv4_target
                    ? std::max(1, dsv4_dspark_query_capacity)
                    : 0;
                auto state = std::make_unique<common_speculative_state_dflash>(
                    config.type,
                    ctx_tgt,
                    ctx_dft,
                    dflash_cross_ctx,
                    query_capacity,
                    config.params.n_max);
                if (!state->ready) {
                    LOG_ERR("%s: failed to initialize %s speculative state\n", __func__,
                            common_speculative_type_to_str(config.type).c_str());
                    return nullptr;
                }
                impls.push_back(std::move(state));
                ctx_dft = nullptr;
                break;
            }
            case COMMON_SPECULATIVE_TYPE_MTP: {
                llama_context * ctx_mtp = ctx_dft;
                if (!ctx_mtp) {
                    const llama_model * model = llama_get_model(ctx_tgt);
                    ctx_mtp = llama_init_from_model(const_cast<llama_model *>(model), config.params.cparams_dft);
                    if (!ctx_mtp) {
                        LOG_ERR("%s: failed to create MTP context\n", __func__);
                        return nullptr;
                    }
                }
                ctx_dft = nullptr;

                const bool use_constant_draft_positions = llama_model_is_gemma4_mtp_assistant(llama_get_model(ctx_mtp));
                impls.push_back(std::make_unique<common_speculative_state_mtp>(
                    config.type, ctx_tgt, ctx_mtp, use_constant_draft_positions));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                LOG_INF("%s: EAGLE3 factory case reached (ctx_dft=%p ctx_tgt=%p)\n", __func__,
                        (void*)ctx_dft, (void*)ctx_tgt);
                if (!ctx_dft) {
                    LOG_ERR("%s: EAGLE3 requires a draft model (-md)\n", __func__);
                    return nullptr;
                }
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(
                    config.type, ctx_tgt, ctx_dft));
                ctx_dft = nullptr;  // ownership moved to the state
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config.params.lookup_cache_static, config.params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                int depth = config.params.suffix_max_depth > 0 ? config.params.suffix_max_depth : 64;
                const llama_model * model = llama_get_model(ctx_tgt);
                impls.push_back(std::make_unique<common_speculative_state_suffix>(
                    config.type, depth, config.params.suffix_corpus, model, (size_t) std::max<int64_t>(0, config.params.suffix_max_nodes)));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON: {
                impls.push_back(std::make_unique<common_speculative_state_suffix_automaton>(config.type));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .configs = */ std::move(configs),
        /* .impls = */ std::move(impls)
    };
    common_speculative_prepare_request(result, params);

    // --spec-draft-policy: needs a self-spec stage with a model stage after it, which is the only shape the stage chain allows
    if (params.draft_policy) {
        bool has_lookup = false, has_model = false;
        for (const auto & impl : result->impls) {
            (common_speculative_type_is_self_spec(impl->type) ? has_lookup : has_model) = true;
        }
        if (has_lookup && has_model && common_speculative_type_is_self_spec(result->impls[0]->type)) {
            result->policy = std::make_unique<common_draft_policy>(params.get_max_stage_n_max() + 1);
            LOG_INF("%s: draft policy on (lookup stage vs model stage chosen per round from learned acceptance and measured cost)\n", __func__);
        } else {
            LOG_WRN("%s: --spec-draft-policy needs a self-speculative stage followed by a model stage (e.g. --spec-type suffix,mtp); ignored\n", __func__);
        }
    }

    // initialize autotune if requested
    if (params.autotune && params.has_composite_stage_chain()) {
        LOG_WRN("Autotune disabled — explicit speculative stage chains are not supported yet\n");
    } else if (params.autotune && !result->impls.empty()) {
        auto actual_type = result->impls[0]->type;
        if (actual_type != COMMON_SPECULATIVE_TYPE_NONE &&
            actual_type != COMMON_SPECULATIVE_TYPE_EAGLE3 &&
            actual_type != COMMON_SPECULATIVE_TYPE_DSPARK) {
            result->tuner = std::make_unique<spec_tuner>();
            result->tuner->init(actual_type, result->configs[0].params, llama_get_model(ctx_tgt));
            LOG_DBG("Autotune initialized for %s, tuning %zu parameters\n",
                    common_speculative_type_to_str(actual_type).c_str(),
                    result->tuner->coords.size());
        } else {
            LOG_WRN("Autotune disabled — speculative type %s is not supported for autotuning\n",
                    common_speculative_type_to_str(actual_type).c_str());
        }
    }

    common_speculative_synth_init(result, params);
    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    spec->checkpoint.clear();
    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last,
        llama_pos draft_base_pos,
        llama_seq_id draft_seq_id) {
    llama_tokens result;

    spec->t_step_start_us = ggml_time_us();
    spec->last_step_target_only = false;
    spec->policy_round_valid = false;

    // apply autotune proposal if enabled
    if (spec->tuner && spec->tuner->enabled) {
        spec->tuner->propose(params);
    }

    const auto runtime_stages = params.get_resolved_stages();
    const bool use_runtime_stage_overrides = common_speculative_stage_chain_matches(runtime_stages, spec->configs);

    spec->curr_impl = nullptr; // reset current implementation

    for (size_t i = 0; i < spec->impls.size(); ++i) {
        auto & impl = spec->impls[i];
        const auto & runtime_stage = use_runtime_stage_overrides ? runtime_stages[i] : spec->configs[i].stage;
        common_params_speculative impl_params = common_speculative_get_runtime_params(spec->configs[i], params, runtime_stage);
        impl_params.draft_temperature = spec->draft_temperature;
        impl_params.draft_seed = spec->draft_seed;
        if (spec->tuner && spec->tuner->enabled && impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            impl_params.n_max = params.n_max;
        }
        result.clear();

        if (spec->tuner && spec->tuner->has_dflash_target_only_arm() &&
                impl->type == COMMON_SPECULATIVE_TYPE_DFLASH && impl_params.n_max == 0) {
            spec->curr_impl = impl.get();
            spec->last_step_target_only = true;
            LOG_DBG("%s: selected DFlash target-only arm\n", __func__);
            break;
        }

        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(impl_params, prompt_tgt, id_last, draft_base_pos, draft_seq_id, result);
            impl->n_call_draft++;
        }

        if (result.empty()) {
            continue;
        }

        // --spec-draft-policy: a lookup draft is taken only when its expected committed tokens per ms beat the model stage's
        // window (the match length is the stage's own when it reports one, else the n-gram key size, else the draft length)
        if (spec->policy && common_speculative_type_is_self_spec(impl->type) && i + 1 < spec->impls.size()) {
            int t_model = 1; // the first model stage after this one: its draft count + 1 is the window the lookup competes with
            for (size_t j = i + 1; j < spec->impls.size(); ++j) {
                if (!common_speculative_type_is_self_spec(spec->impls[j]->type)) {
                    const auto & rs_j = use_runtime_stage_overrides ? runtime_stages[j] : spec->configs[j].stage;
                    t_model = common_speculative_get_runtime_params(spec->configs[j], params, rs_j).n_max + 1;
                    break;
                }
            }
            const int match  = impl->last_match_len >= 0 ? impl->last_match_len
                              : impl_params.ngram_size_n > 0 ? (int) impl_params.ngram_size_n : (int) result.size();
            const auto pk = spec->policy->choose(t_model, (int) result.size(), match);
            if (!pk.lookup) {
                result.clear();
                continue;
            }
            if ((int) result.size() > pk.t - 1) {
                result.resize(pk.t - 1);
            }
            spec->policy_round_match = match;
        }

        if (common_speculative_type_is_self_spec(impl->type) && impl_params.n_min > 0 && (int)result.size() < impl_params.n_min) {
            LOG_DBG("%s: impl %s drafted %zu tokens, below fallback threshold %d - trying next implementation\n",
                    __func__, common_speculative_type_to_str(impl->type).c_str(), result.size(), impl_params.n_min);
            result.clear();
            continue;
        }
        LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                impl.get()->n_call_draft, result.size());

        spec->curr_impl = impl.get();
        impl->n_gen_drafts++;
        impl->n_gen_tokens += result.size();
        impl->drafted_by_position.resize(std::max(impl->drafted_by_position.size(), result.size()));
        impl->accepted_by_position.resize(impl->drafted_by_position.size());
        for (size_t i = 0; i < result.size(); ++i) {
            impl->drafted_by_position[i]++;
        }

        break; // We have a draft, so break out of the loop and return it.
    }

    // store draft count for tuner feedback
    if (spec->tuner && spec->tuner->enabled) {
        spec->last_n_drafted = (int)result.size();
    }

    // round bookkeeping for the draft policy: which side drafted, how long the window is, what the match was
    if (spec->policy && spec->curr_impl && !result.empty()) {
        spec->policy_round_valid  = true;
        spec->policy_round_lookup = common_speculative_type_is_self_spec(spec->curr_impl->type);
        spec->policy_round_t      = (int) result.size() + 1;
        if (!spec->policy_round_lookup) {
            spec->policy_round_match = 0;
        }
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (spec->policy && spec->policy_round_valid && spec->t_step_start_us > 0) {
        const double round_ms = (ggml_time_us() - spec->t_step_start_us) / 1000.0;
        spec->policy->observe(spec->policy_round_lookup, spec->policy_round_t, n_accepted, spec->policy_round_match, round_ms);
        spec->policy_round_valid = false;
    }

    if (spec->tuner && spec->tuner->enabled && spec->t_step_start_us > 0) {
        int64_t step_time_us = ggml_time_us() - spec->t_step_start_us;
        double step_tps = (step_time_us > 100)
            ? (n_accepted + 1.0) * 1e6 / (double)step_time_us
            : 0.0;
        spec->tuner->accept_feedback(n_accepted, spec->last_n_drafted, step_tps);
        spec->t_step_start_us = 0;
    }

    common_speculative_state * impl = spec->curr_impl;

    if (!impl) {
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;

            const size_t n_accepted_positions = std::min<size_t>(n_accepted, impl->drafted_by_position.size());
            impl->accepted_by_position.resize(std::max(impl->accepted_by_position.size(), n_accepted_positions));
            for (size_t i = 0; i < n_accepted_positions; ++i) {
                impl->accepted_by_position[i]++;
            }
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }

    if (impl->type != COMMON_SPECULATIVE_TYPE_MTP) {
        if (auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
            mtp_invalidate_cached_drafts(*mtp_state);
        }
    }
}

static bool common_speculative_has_type(const common_speculative * spec, common_speculative_type type) {
    if (spec == nullptr) {
        return false;
    }

    return std::any_of(spec->configs.begin(), spec->configs.end(), [type](const common_speculative_config & config) {
        return config.type == type;
    });
}

static bool common_speculative_has_dflash_family(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    return std::any_of(spec->configs.begin(), spec->configs.end(), [](const common_speculative_config & config) {
        return common_speculative_type_is_dflash_family(config.type);
    });
}

static int common_speculative_ctx_mtp_n_embd(llama_context * ctx) {
    return ctx ? (int) llama_mtp_state_n_embd(ctx) : 0;
}

static bool common_speculative_batch_token_has_seq_id(
        const llama_batch & batch,
        int token_index,
        llama_seq_id seq_id) {
    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.n_seq_id[token_index] <= 0 || batch.seq_id[token_index] == nullptr) {
        return false;
    }

    for (int i = 0; i < batch.n_seq_id[token_index]; ++i) {
        if (batch.seq_id[token_index][i] == seq_id) {
            return true;
        }
    }

    return false;
}

static bool common_speculative_batch_is_exact_single_seq(
        const llama_batch & batch,
        llama_seq_id seq_id) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
            return false;
        }
    }

    return true;
}

static int common_speculative_copy_seq_batch(
        const llama_batch & batch,
        llama_seq_id seq_id,
        llama_batch & seq_batch) {
    if (batch.token == nullptr || batch.pos == nullptr) {
        return -1;
    }

    if (batch.n_tokens < 1) {
        return 0;
    }

    std::vector<int> token_indices;
    token_indices.reserve(batch.n_tokens);
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (common_speculative_batch_token_has_seq_id(batch, i, seq_id)) {
            token_indices.push_back(i);
        }
    }

    if (token_indices.empty()) {
        return 0;
    }

    seq_batch = llama_batch_init((int) token_indices.size(), 0, 1);
    for (const int i : token_indices) {
        common_batch_add(seq_batch, batch.token[i], batch.pos[i], { seq_id }, batch.logits != nullptr && batch.logits[i]);
    }

    return (int) token_indices.size();
}

static bool common_speculative_feature_view_copy_batch_rows(
        const common_speculative_feature_view & view,
        const llama_batch & batch,
        llama_seq_id seq_id,
        std::vector<float> * hidden_rows) {
    if (hidden_rows == nullptr || view.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || view.width <= 0 || batch.n_tokens <= 0 || batch.pos == nullptr) {
        return false;
    }

    std::unordered_map<llama_pos, const float *> rows_by_pos;
    rows_by_pos.reserve(view.rows.size());
    for (const auto & row : view.rows) {
        if (row.seq_id == seq_id && row.data != nullptr) {
            rows_by_pos[row.pos] = row.data;
        }
    }

    hidden_rows->clear();
    hidden_rows->reserve((size_t) batch.n_tokens * view.width);
    for (int i = 0; i < batch.n_tokens; ++i) {
        auto it = rows_by_pos.find(batch.pos[i]);
        if (it == rows_by_pos.end()) {
            hidden_rows->clear();
            return false;
        }

        hidden_rows->insert(hidden_rows->end(), it->second, it->second + view.width);
    }

    return hidden_rows->size() == (size_t) batch.n_tokens * view.width;
}

static bool common_speculative_capture_target_features(
        common_speculative * spec,
        const common_speculative_feature_view & features);

static bool common_speculative_feature_view_from_hidden_rows(
        const std::vector<float> & hidden_rows,
        int32_t width,
        llama_seq_id seq_id,
        llama_pos pos_base,
        common_speculative_feature_view & view) {
    view = {};
    view.kind = COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE;
    view.width = width;

    if (width <= 0 || hidden_rows.empty() || hidden_rows.size() % (size_t) width != 0) {
        return false;
    }

    const size_t n_rows = hidden_rows.size() / (size_t) width;
    view.rows.reserve(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        view.rows.push_back({
            /* .seq_id = */ seq_id,
            /* .pos    = */ pos_base + (llama_pos) i,
            /* .data   = */ hidden_rows.data() + i * (size_t) width,
        });
    }

    return true;
}

static bool common_speculative_collect_target_batch_features(
        const common_speculative * spec,
        llama_context * ctx,
        const llama_batch & batch,
        common_speculative_feature_view & features) {
    features = {};
    if (common_speculative_has_dflash_family(spec)) {
        return llama_spec_get_dflash_feature_view(ctx, batch, features);
    }

    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    if (!llama_spec_get_hidden_feature_view(ctx, batch, features)) {
        return false;
    }

    return true;
}

static bool common_speculative_collect_target_seq_batch_features(
        const common_speculative * spec,
        llama_context * ctx,
        const llama_batch & batch,
        llama_seq_id seq_id,
        common_speculative_feature_view & features) {
    features = {};
    if (common_speculative_has_dflash_family(spec)) {
        return llama_spec_get_dflash_feature_view_for_seq(ctx, batch, seq_id, features);
    }

    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    if (!llama_spec_get_hidden_feature_view_for_seq(ctx, batch, seq_id, features)) {
        return false;
    }

    return true;
}

bool common_speculative_capture_output_hidden(
        common_speculative * spec,
        llama_context * ctx,
        int32_t output_index,
        llama_seq_id seq_id,
        llama_pos pos) {
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    common_speculative_feature_view features;
    if (!llama_spec_get_hidden_feature_view_from_output_index(ctx, output_index, seq_id, pos, features)) {
        return false;
    }

    return common_speculative_capture_target_features(spec, features);
}

bool common_speculative_ensure_sequence_hidden(
        common_speculative * spec,
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos pos) {
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP) || common_speculative_has_sequence_hidden(spec, seq_id)) {
        return true;
    }

    return common_speculative_capture_output_hidden(spec, ctx, -1, seq_id, pos);
}

common_speculative_draft_result common_speculative_draft_ex(
        common_speculative * spec,
        llama_context * ctx,
        common_params_speculative & params,
        const llama_tokens & prompt_tgt,
        llama_token id_last,
        llama_pos draft_base_pos,
        llama_seq_id draft_seq_id,
        const common_params_sampling * sampling) {
    common_speculative_draft_result result = {};

    if (spec != nullptr) {
        spec->draft_temperature = sampling != nullptr ? sampling->temp : 0.0f;
        spec->draft_seed = sampling != nullptr ? sampling->seed : LLAMA_DEFAULT_SEED;
    }

    if (common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        if (!common_speculative_ensure_sequence_hidden(spec, ctx, draft_seq_id, draft_base_pos - 1)) {
            LOG_ERR("%s: seq_id=%d MTP hidden state is empty during speculation\n",
                    __func__, (int) draft_seq_id);
            return result;
        }
    }

    result.tokens = common_speculative_draft(
        spec,
        params,
        prompt_tgt,
        id_last,
        draft_base_pos,
        draft_seq_id);
    result.type = spec != nullptr && spec->curr_impl != nullptr
        ? spec->curr_impl->type
        : COMMON_SPECULATIVE_TYPE_NONE;
    result.target_only = spec != nullptr && spec->last_step_target_only;

    // In-Place TTT target (row S69-FORK-INPLACE-TTT-2026-10-26-01): no draft token may complete a fast-weight chunk,
    // since a rejected draft's update cannot be undone; id_last sits at draft_base_pos (or after the prompt)
    if (ctx != nullptr) {
        const llama_pos pos_last = draft_base_pos >= 0 ? draft_base_pos : (llama_pos) prompt_tgt.size();
        const int32_t   cap      = llama_ttt_draft_cap(llama_get_model(ctx), pos_last);
        if (cap >= 0 && (int64_t) result.tokens.size() > (int64_t) cap) {
            result.tokens.resize((size_t) cap);
        }
    }

    if (spec != nullptr && spec->curr_impl != nullptr &&
            spec->curr_impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
        if (auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
            result.proposal_dists = dflash_state->proposal_dists;
        }
    }
    if (ctx != nullptr && llama_model_ttt_chunk(llama_get_model(ctx)) > 0 &&
            result.proposal_dists.size() > result.tokens.size()) {
        result.proposal_dists.resize(result.tokens.size());   // kept parallel to a TTT-capped draft (above)
    }

    return result;
}

int common_speculative_get_configured_n_max(const common_speculative * spec) {
    if (spec == nullptr || spec->tuner == nullptr || !spec->tuner->has_dflash_target_only_arm()) {
        return 0;
    }
    return spec->tuner->configured_n_max;
}

static bool common_speculative_has_target_features(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    return std::any_of(spec->configs.begin(), spec->configs.end(), [](const common_speculative_config & config) {
        return common_speculative_type_uses_target_features(config.type);
    });
}

// parse --draft-params into draft model/context parameters
static bool common_speculative_parse_params(const std::string & args, gpt_params & out, bool print_usage) {
    // ik #2588 helper, adapted: this tree parses with common_params_parse (arg.h), which takes the
    // llama_example the flags belong to and prints usage/parse errors itself, so print_usage has
    // no separate call to gate here. "llama-server" as argv[0] matches the call this replaced.
    GGML_UNUSED(print_usage);
    auto [argc, argv] = parse_command_line("llama-server " + args);
    const bool ok = common_params_parse(argc, argv, out, LLAMA_EXAMPLE_SERVER);
    free_command_line(argc, argv);
    return ok;
}

// apply -td/--threads-draft to a draft params copy
void common_speculative_apply_draft_threads(const common_params_speculative & spec, gpt_params & out) {
    // Adapted: in this tree -td/-tbd write spec.cpuparams / spec.cpuparams_batch (common/arg.cpp
    // --spec-draft-threads), and the context reads out.cpuparams / out.cpuparams_batch (common.cpp
    // cparams.n_threads), not the flat n_threads fields ik's original wrote. The flat spec fields
    // are still honoured when set, since speculative.cpp's draft-params builder reads them.
    const int32_t nt  = spec.n_threads       > 0 ? spec.n_threads       : spec.cpuparams.n_threads;
    const int32_t ntb = spec.n_threads_batch > 0 ? spec.n_threads_batch : spec.cpuparams_batch.n_threads;
    if (nt > 0) {
        out.n_threads           = nt;
        out.cpuparams.n_threads = nt;
        if (ntb <= 0) {
            out.n_threads_batch           = nt; // follow -td
            out.cpuparams_batch.n_threads = nt;
        }
    }
    if (ntb > 0) {
        out.n_threads_batch           = ntb;
        out.cpuparams_batch.n_threads = ntb;
    }
}

// store the draft affinity from parsed --draft-params
static void common_speculative_apply_draft_affinity(const gpt_params & parsed, common_params_speculative & params) {
    if (!parsed.cpu_affinity_configured) {
        return;
    }
    // --cpu-affinity = hybrid E-cores
    params.cpu_affinity = parsed.cpu_affinity_auto
        ? cpu_affinity_resolve_draft({}, true)
        : cpu_affinity_resolve_draft(parsed.cpu_affinity, false);
    params.cpu_affinity_configured = true;
    if (params.cpu_affinity.empty()) {
        if (parsed.cpu_affinity_auto) {
            LOG_WRN("%s: no efficiency cores available for the draft affinity\n", __func__);
        }
    } else {
        std::string cpus;
        for (size_t i = 0; i < params.cpu_affinity.size(); ++i) {
            cpus += (i == 0 ? "" : ",") + std::to_string(params.cpu_affinity[i]);
        }
        LOG_INF("%s: draft CPU affinity: %s\n", __func__, cpus.c_str());
    }
    const int n_dft_threads = parsed.n_threads > 0 ? parsed.n_threads : (int) params.cpu_affinity.size();
    if (!params.cpu_affinity.empty() && n_dft_threads > (int) params.cpu_affinity.size()) {
        LOG_WRN("%s: draft uses %d threads on %zu pinned CPUs; consider -t %zu in --draft-params\n",
                __func__, n_dft_threads, params.cpu_affinity.size(), params.cpu_affinity.size());
    }
}

// cparams_dft is built from a LOCAL gpt_params (params_dft / params_mtp) that dies when the caller
// returns, but the draft/MTP llama_context is created later (common_speculative_try_init). Every
// pointer common_context_params_to_llama stored into that local is moved into storage owned by
// `params` and rebound here. Same null-for-empty rule as common_context_params_to_llama.
// SOURCE-READ: ikawrakow/ik_llama.cpp 804ee29c (#2715) common/speculative.cpp -- cuda_params +
// offload_policy; the six const char* fields below are fork-only and carried the same bug.
// cpu_affinity is rebound by each caller right after this call (it inherits params_base's).
// A NEW pointer field in common_context_params_to_llama must be added here too: the fork pre-commit
// gate "cparams-dft-ownership" (cortex/scripts/check_cparams_dft_pointer_ownership.py) fails otherwise.
static void common_speculative_own_cparams_dft(common_params_speculative & params, const gpt_params & src) {
    auto & st = params.cparams_dft_store;
    auto & cp = params.cparams_dft;
    st.cuda_params            = src.cuda_params;
    st.offload_policy         = src.offload_policy;
    st.tri_stats_path         = src.triattention_stats;
    st.cats_thresholds_file   = src.cats_thresholds_file;
    st.moe_stats_path         = src.moe_stats_path;
    st.moe_stats_mode         = src.moe_stats_mode;
    st.moe_placement_strategy = src.moe_placement_strategy;
    st.moe_placement_stats    = src.moe_placement_stats;
    st.teal_thresholds_file   = src.teal_thresholds_file;
    const auto cstr_or_null = [](const std::string & s) { return s.empty() ? nullptr : s.c_str(); };
    cp.cuda_params            = st.cuda_params.empty()    ? nullptr : (void *) st.cuda_params.data();
    cp.offload_policy         = st.offload_policy.empty() ? nullptr : (void *) &st.offload_policy;
    cp.tri_stats_path         = cstr_or_null(st.tri_stats_path);
    cp.cats_thresholds_file   = cstr_or_null(st.cats_thresholds_file);
    cp.moe_stats_path         = cstr_or_null(st.moe_stats_path);
    cp.moe_stats_mode         = cstr_or_null(st.moe_stats_mode);
    cp.moe_placement_strategy = cstr_or_null(st.moe_placement_strategy);
    cp.moe_placement_stats    = cstr_or_null(st.moe_placement_stats);
    cp.teal_thresholds_file   = cstr_or_null(st.teal_thresholds_file);
}

bool common_speculative_load_draft_model(
        common_params_speculative & params,
        const gpt_params         & params_base,
        const llama_model        * model_tgt) {
    if (!params.has_dft()) {
        return true;
    }

    // never inherit the target affinity flags into the draft copy
    params.cpu_affinity.clear();
    params.cpu_affinity_configured = false;

    gpt_params params_dft = params_base;
    params_dft.cpu_affinity.clear();
    params_dft.cpu_affinity_auto = false;
    params_dft.cpu_affinity_configured = false;
    if (params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP)) {
        params_dft.has_mtp = true;
    }
    params_dft.devices          = params.devices.empty() ? params_base.devices : params.devices;
    // CORRECTED 2026-09-13: in common_speculative_load_draft_model the local `params` is a
    // common_params_speculative (see its signature at :2842), whose `model` is the FLAT draft
    // string at common.h:400 -- not a common_params_model. params_dft IS a common_params, so
    // only the left side is nested. The nested-model rewrite matched on the receiver NAME and
    // could not see that this one `params` has a different type.
    params_dft.model.path       = params.dft_path();
    params_dft.n_gpu_layers     = params.n_gpu_layers;
    params_dft.cache_type_k     = params.cache_type_k.empty() ? params_base.cache_type_k : params.cache_type_k;
    params_dft.cache_type_v     = params.cache_type_v.empty() ? params_base.cache_type_v : params.cache_type_v;


    // We likely don't want to inherit offload policy for MTP
    if (params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP)) {
        params_dft.ncmoe = 0;
        params_dft.tensor_buft_overrides.clear();
        params_dft.offload_policy.clear();
        LOG_INF("%s: MTP draft ignores target CPU-MoE/tensor placement overrides\n",
                __func__);
    }

    // ik_llama.cpp #2562 (MIT): the draft's `-draft` flag string is parsed AFTER the MTP placement reset, so -ot / --override-kv /
    // -gfm given there win; the sentinel entries a previous parse appended (empty kv key, null tensor pattern, the {-1, 0} fit
    // margin pair) are trimmed first so the second parse appends to a clean list instead of after a terminator.
    if (!params.params.empty()) {
        while (!params_dft.kv_overrides.empty() && params_dft.kv_overrides.back().key[0] == 0) {
            params_dft.kv_overrides.pop_back();
        }
        while (!params_dft.tensor_buft_overrides.empty() && params_dft.tensor_buft_overrides.back().pattern == nullptr) {
            params_dft.tensor_buft_overrides.pop_back();
        }
        while (params_dft.fit_margin_array.size() >= 2 &&
                params_dft.fit_margin_array[params_dft.fit_margin_array.size()-2] == -1 &&
                params_dft.fit_margin_array.back() == 0) {
            params_dft.fit_margin_array.pop_back();
            params_dft.fit_margin_array.pop_back();
        }
        if (!common_speculative_parse_params(params.params, params_dft, true)) {
            return false;
        }
    }
    common_speculative_apply_draft_threads(params, params_dft);
    common_speculative_apply_draft_affinity(params_dft, params);

    LOG_INF("%s: loading draft model '%s'\n", __func__, params_dft.model.path.c_str());

    if (params.has_dflash_family_stage() && params.n_ctx > 0) {
        params_dft.n_ctx = params.n_ctx;
    }
    if (params.has_dflash_family_stage() && params_dft.n_gpu_layers < 0) {
        params_dft.n_gpu_layers = params_base.n_gpu_layers;
    }
    if (params_dft.n_ctx == 0) {
        params_dft.n_ctx = params.n_ctx > 0 ? params.n_ctx : params_base.n_ctx / params_base.n_parallel;
    }
    params_dft.n_parallel = 1;

    params.mparams_dft.path = params_dft.model.path;

    llama_model_params mparams_dft = common_model_params_to_llama(params_dft);
    // a draft head can leave out the embeddings and lm head and use the target's
    // SOURCE-READ: unslothai/llama.cpp f36eda84 common/speculative.cpp (`mparams.model_shared = model_tgt`)
    mparams_dft.model_shared = model_tgt;
    llama_model * loaded_model = llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft);
    if (loaded_model == nullptr) {
        LOG_ERR("%s: failed to load draft model '%s'\n", __func__, params_dft.model.path.c_str());
        return false;
    }

    params.model_dft = loaded_model;
    params.cparams_dft = common_context_params_to_llama(params_dft);
    params.cparams_dft.moe_cache_size = 0;
    common_speculative_own_cparams_dft(params, params_dft); // params_dft dies on return
    // draft affinity; inherits the target one when not requested
    const std::vector<int32_t> & cpu_affinity_dft = params.cpu_affinity_configured
        ? params.cpu_affinity
        : params_base.cpu_affinity;
    params.cparams_dft.cpu_affinity   = cpu_affinity_dft.empty() ? nullptr : cpu_affinity_dft.data();
    params.cparams_dft.n_cpu_affinity = (int32_t) cpu_affinity_dft.size();
    return true;
}

bool common_speculative_prepare_mtp_runtime(
        common_params_speculative & params,
        const gpt_params         & params_base,
        const llama_model        * model,
        bool                       has_external_mtp) {
    if (!params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP)) {
        return false;
    }

    const enum llama_mtp_package target_package = llama_model_mtp_package(model);
    const bool has_embedded_mtp = target_package == LLAMA_MTP_PACKAGE_EMBEDDED;
    if (target_package == LLAMA_MTP_PACKAGE_COMPANION ||
        target_package == LLAMA_MTP_PACKAGE_INVALID) {
        LOG_ERR("%s: the target GGUF is an MTP companion or invalid package, load a complete target model instead\n",
                __func__);
        return false;
    }
    if (target_package == LLAMA_MTP_PACKAGE_TARGET_ONLY && !has_external_mtp) {
        LOG_ERR("%s: target GGUF contains no MTP tail, provide a matching predictor-only companion with -md\n",
                __func__);
        return false;
    }

    if (llama_model_is_step35(model) && has_embedded_mtp &&
        !llama_model_step35_has_nextn_weights(model)) {
        LOG_ERR("%s: Step target is missing one or more complete MTP heads\n", __func__);
        return false;
    }
    if (!has_external_mtp && !has_embedded_mtp) {
        LOG_ERR("%s: MTP speculative stage requested, but the target package is not a complete embedded MTP model\n",
                __func__);
        return false;
    }

    if (!has_external_mtp) {
        gpt_params params_mtp = params_base;
        params_mtp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        common_speculative_apply_draft_threads(params, params_mtp);
        params.cparams_dft = common_context_params_to_llama(params_mtp);
        params.cparams_dft.moe_cache_size = 0;
        common_speculative_own_cparams_dft(params, params_mtp); // params_mtp dies at the end of this block
        // same draft affinity as the external draft path
        const std::vector<int32_t> & cpu_affinity_dft = params.cpu_affinity_configured
            ? params.cpu_affinity
            : params_base.cpu_affinity;
        params.cparams_dft.cpu_affinity   = cpu_affinity_dft.empty() ? nullptr : cpu_affinity_dft.data();
        params.cparams_dft.n_cpu_affinity = (int32_t) cpu_affinity_dft.size();
    }

    params.cparams_dft.mtp         = true;
    params.cparams_dft.mtp_op_type = MTP_OP_WARMUP;
    params.cparams_dft.embeddings  = true;

    return true;
}

common_speculative_init_status common_speculative_try_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        common_speculative      ** out_spec) {
    if (out_spec != nullptr) {
        *out_spec = nullptr;
    }

    if (!params.has_stage_chain()) {
        return COMMON_SPECULATIVE_INIT_SKIPPED;
    }

    common_speculative * spec = common_speculative_init(params, ctx_tgt);
    if (spec != nullptr) {
        if (out_spec != nullptr) {
            *out_spec = spec;
        }
        return COMMON_SPECULATIVE_INIT_READY;
    }

    const llama_model * model = ctx_tgt != nullptr ? llama_get_model(ctx_tgt) : nullptr;
    if (model != nullptr && llama_model_has_recurrent(model)) {
        return COMMON_SPECULATIVE_INIT_ERR_RECURRENT;
    }
    if (params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP)) {
        return COMMON_SPECULATIVE_INIT_ERR_MTP;
    }
    return COMMON_SPECULATIVE_INIT_ERR_GENERIC;
}

bool common_speculative_prepare_startup(
        gpt_params & params_base,
        bool         allow_parallel_mtp) {
    auto & params = params_base.speculative;

    // resolve draft affinity for embedded stages (external: on load)
    params.cpu_affinity.clear();
    params.cpu_affinity_configured = false;
    if (!params.has_dft() && !params.params.empty()) {
        gpt_params params_dft = params_base;
        params_dft.cpu_affinity.clear();
        params_dft.cpu_affinity_auto = false;
        params_dft.cpu_affinity_configured = false;
        if (!common_speculative_parse_params(params.params, params_dft, false)) {
            return false; // invalid --draft-params must abort startup
        }
        common_speculative_apply_draft_threads(params, params_dft);
        common_speculative_apply_draft_affinity(params_dft, params);
    }

    if (!allow_parallel_mtp && params_base.n_parallel > 1 && params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP)) {
        LOG_WRN("%s: MTP is not supported with parallel slots yet, removing the MTP stage to avoid cross-slot corruption. n_parallel=%d, stage_chain=%s\n",
                __func__, params_base.n_parallel, common_speculative_stage_chain_to_str(params).c_str());
        params.remove_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
    }

    if (!params.needs_dft_model()) {
        params.clear_dft();
    }

    params_base.has_mtp = params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);

    return true;
}

bool common_speculative_finalize_startup(
        gpt_params        & params_base,
        const llama_model * model) {
    auto & params = params_base.speculative;

    if (params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) &&
        model != nullptr &&
        llama_model_is_deepseek4(model) &&
        llama_model_n_nextn_layer(model) > 1) {
        LOG_ERR("%s: DeepSeek-V4 MTP supports exactly one NextN predictor layer, got %d.\n",
                __func__, llama_model_n_nextn_layer(model));
        return false;
    }

    if (!params.needs_dft_model()) {
        params.clear_dft();
    }

    if (params.has_dft() && model != nullptr && llama_model_is_openpangu(model)) {
        LOG_ERR("%s: openPangu does not support an external draft model; it has its own MTP heads\n", __func__);
        return false;
    }

    if (params.has_dft()) {
        LLAMA_LOG_INFO("\n\n==================================loading DRAFT model==================================\n\n");
        if (!common_speculative_load_draft_model(params, params_base, model)) {
            return false;
        }

        const bool mtp_requested = params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
        if (mtp_requested && model != nullptr && params.model_dft != nullptr) {
            const llama_model * companion = params.model_dft;
            const bool appended_contract = common_speculative_target_has_appended_mtp_contract(model);

            if (appended_contract &&
                llama_model_mtp_package(companion) != LLAMA_MTP_PACKAGE_COMPANION) {
                LOG_ERR("%s: -md for an MTP stage must be a predictor-only companion GGUF\n", __func__);
                return false;
            }

            if (llama_model_is_step35(model)) {
                if (!llama_model_is_step35(companion)) {
                    LOG_ERR("%s: Step MTP requires a Step companion\n", __func__);
                    return false;
                }

                const int32_t n_heads = llama_model_n_nextn_layer(companion);
                if (n_heads != 3) {
                    LOG_ERR("%s: Step MTP companion requires exactly three predictor heads, got %d\n",
                            __func__, n_heads);
                    return false;
                }

                if (!llama_model_step35_has_nextn_weights(companion)) {
                    LOG_ERR("%s: Step MTP companion is missing required predictor tensors\n", __func__);
                    return false;
                }
            } else if (llama_model_is_deepseek4(model)) {
                if (!llama_model_is_deepseek4(companion)) {
                    LOG_ERR("%s: DeepSeek-V4 MTP requires a DeepSeek-V4 companion\n", __func__);
                    return false;
                }

                const int32_t n_heads = llama_model_n_nextn_layer(companion);
                if (n_heads != 1) {
                    LOG_ERR("%s: DeepSeek-V4 MTP companion requires exactly one predictor layer, got %d\n",
                            __func__, n_heads);
                    return false;
                }
            } else if (llama_model_is_qwen35_family(model)) {
                // dense and MoE are separate architectures, so compare the arch itself
                if (std::strcmp(llama_model_arch_string(model), llama_model_arch_string(companion)) != 0) {
                    LOG_ERR("%s: Qwen3.5 MTP requires a companion of the same architecture, target is %s and companion is %s\n",
                            __func__, llama_model_arch_string(model), llama_model_arch_string(companion));
                    return false;
                }

                const int32_t n_heads = llama_model_n_nextn_layer(companion);
                if (n_heads != 1) {
                    LOG_ERR("%s: Qwen3.5 MTP companion requires exactly one predictor layer, got %d\n",
                            __func__, n_heads);
                    return false;
                }
            } else if (llama_model_is_qwen4exp(model)) {
                if (!llama_model_is_qwen4exp(companion)) {
                    LOG_ERR("%s: Qwen4Exp MTP requires a Qwen4Exp companion\n", __func__);
                    return false;
                }

                const int32_t n_heads = llama_model_n_nextn_layer(companion);
                if (n_heads != 1) {
                    LOG_ERR("%s: Qwen4Exp MTP companion requires exactly one predictor layer, got %d\n",
                            __func__, n_heads);
                    return false;
                }
            }

            if (common_speculative_has_recognized_mtp_companion(model, companion)) {
                if (!common_speculative_are_compatible(model, companion)) {
                    LOG_ERR("%s: MTP target and companion vocabularies are incompatible\n", __func__);
                    return false;
                }

                const uint32_t target_width = llama_model_mtp_feature_width(model);
                const uint32_t companion_width = llama_model_mtp_feature_width(companion);

                if (target_width != companion_width) {
                    LOG_ERR("%s: MTP feature-width mismatch: target=%u companion=%u\n",
                            __func__, target_width, companion_width);
                    return false;
                }
            }
        }
    }

    params_base.has_mtp = params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP);
    const bool has_external_mtp = params_base.has_mtp && params.model_dft &&
        common_speculative_has_recognized_mtp_companion(model, params.model_dft);

    params_base.has_mtp = common_speculative_prepare_mtp_runtime(
        params,
        params_base,
        model,
        has_external_mtp);
    if (params.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) && !params_base.has_mtp) {
        return false;
    }
    if (params_base.has_mtp) {
        params_base.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }

    return true;
}

bool common_speculative_before_draft(
        common_speculative * spec,
        llama_model * model,
        llama_context * ctx,
        common_sampler * sampler_src,
        const common_params_sampling & sparams,
        llama_seq_id seq_id,
        llama_pos n_past,
        llama_token sampled,
        int max_tokens,
        int ckpt_mode) {
    if (spec == nullptr) {
        return false;
    }

    return common_speculative_checkpoint_save(
        spec->checkpoint,
        model,
        ctx,
        sampler_src,
        sparams,
        seq_id,
        n_past,
        sampled,
        max_tokens,
        ckpt_mode);
}

int32_t common_speculative_on_target_seq_batch(
        common_speculative * spec,
        llama_context * ctx_tgt,
        const llama_batch & batch,
        llama_seq_id seq_id,
        bool is_prompt_warmup) {
    if (ctx_tgt == nullptr || batch.n_tokens <= 0) {
        return 0;
    }

    // S69 Stage-3: fire the per-decode process() hook for every impl (eagle3 builds its draft KV
    // in lockstep with the target here). The default impl is a no-op, so MTP/DFlash/draft/ngram are
    // unaffected. This runs ABOVE the target-features guard on purpose: eagle3 does not report
    // target-feature consumption, so a guard placed first would return before its hook ever fired.
    if (spec != nullptr) {
        for (auto & impl : spec->impls) {
            if (impl) { impl->process(batch); }
        }
    }

    // Self-speculative stages do not consume target hidden-state features.
    if (!common_speculative_has_target_features(spec)) {
        return 0;
    }

    if (common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        llama_context * ctx_mtp = common_speculative_get_companion_ctx(spec);
        ctx_mtp = ctx_mtp ? ctx_mtp : ctx_tgt;
        if (ctx_mtp == nullptr) {
            return 0;
        }

        const int n_embd_src = common_speculative_ctx_mtp_n_embd(ctx_tgt);
        const int n_embd_dst = common_speculative_ctx_mtp_n_embd(ctx_mtp);
        if (n_embd_src <= 0 || n_embd_dst <= 0) {
            return -1;
        }

        if (n_embd_src != n_embd_dst) {
            LOG_ERR("MTP warmup hidden state width mismatch: n_embd_src = %d, n_embd_dst = %d\n", n_embd_src, n_embd_dst);
            return -1;
        }
    }

    common_speculative_feature_view feature_view;
    const llama_batch * batch_for_spec = &batch;
    llama_batch seq_batch = {};
    const bool needs_seq_split = is_prompt_warmup && !common_speculative_batch_is_exact_single_seq(batch, seq_id);

    if (needs_seq_split) {
        const int n_seq_tokens = common_speculative_copy_seq_batch(batch, seq_id, seq_batch);
        if (n_seq_tokens <= 0) {
            return n_seq_tokens < 0 ? -1 : 0;
        }

        if (!common_speculative_collect_target_seq_batch_features(spec, ctx_tgt, batch, seq_id, feature_view)) {
            llama_batch_free(seq_batch);
            return -1;
        }

        batch_for_spec = &seq_batch;
    } else {
        if (!common_speculative_collect_target_batch_features(spec, ctx_tgt, batch, feature_view)) {
            return -1;
        }
    }

    const int32_t ret = common_speculative_on_target_batch(spec, *batch_for_spec, feature_view, is_prompt_warmup);
    if (needs_seq_split) {
        llama_batch_free(seq_batch);
    }

    return ret;
}

// S69 Stage-3: canonical per-decode hook wrapper (llama.cpp PR#18039 common_speculative_process). Loops impls'
// process(batch). The serving path already fires the hook via common_speculative_on_target_seq_batch; this is the
// direct-call API for parity/testing.
bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool ok = true;
    if (spec == nullptr) { return ok; }
    for (auto & impl : spec->impls) {
        if (impl) { ok = impl->process(batch) && ok; }
    }
    // unsloth/master's per-sequence-pool implementations (populated only when spec was
    // constructed via common_speculative_init(params, n_seq); empty otherwise, so this is a
    // no-op for spec objects built the fork's way).
    for (auto & impl : spec->impls_seq) {
        ok = impl->process(batch) && ok;
    }
    return ok;
}

bool common_speculative_process(common_speculative * spec, const common_batch & batch) {
    if (spec == nullptr || batch.size() == 0) {
        return true;
    }
    const bool    embd   = batch.has_embd();
    const int32_t n_embd = embd ? (int32_t) batch.tokens[0].embd.n_embd : 0;
    int32_t n_seq_max = 1;
    for (const auto & t : batch.tokens) {
        n_seq_max = std::max<int32_t>(n_seq_max, 1 + (int32_t) t.seq_ids_extra.size());
    }
    llama_batch b = llama_batch_init(batch.size(), n_embd, n_seq_max);
    for (int32_t i = 0; i < batch.size(); ++i) {
        const auto & t = batch.tokens[(size_t) i];
        if (embd) {
            memcpy(b.embd + (size_t) i * n_embd, t.embd.data, (size_t) n_embd * sizeof(float));
        } else {
            b.token[i] = t.id;
        }
        b.pos[i]       = t.pos[0];
        b.n_seq_id[i]  = 1 + (int32_t) t.seq_ids_extra.size();
        b.seq_id[i][0] = t.seq_id;
        for (size_t k = 0; k < t.seq_ids_extra.size(); ++k) {
            b.seq_id[i][1 + k] = t.seq_ids_extra[k];
        }
        b.logits[i] = t.output;
    }
    b.n_tokens = batch.size();
    const bool ok = common_speculative_process(spec, b);
    llama_batch_free(b);
    return ok;
}

bool common_speculative_copy_output_hidden_rows(
        const common_speculative * spec,
        llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows) {
    hidden_rows.clear();
    if (common_speculative_has_dflash_family(spec)) {
        return llama_spec_copy_dflash_rows_from_output_indices(ctx, output_indices, hidden_rows);
    }

    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    return llama_spec_copy_hidden_rows_from_output_indices(ctx, output_indices, hidden_rows);
}

static bool common_speculative_build_commit_tokens(
        common_speculative_type spec_type_used,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        std::vector<llama_token> & commit_tokens) {
    commit_tokens.clear();
    if (ids.empty()) {
        return true;
    }

    if (spec_type_used == COMMON_SPECULATIVE_TYPE_MTP) {
        commit_tokens = ids;
        return true;
    }

    commit_tokens.reserve(ids.size());
    commit_tokens.push_back(sampled_before);
    if (ids.size() > 1) {
        commit_tokens.insert(commit_tokens.end(), ids.begin(), ids.end() - 1);
    }

    return commit_tokens.size() == ids.size();
}

static bool common_speculative_apply_hidden_rows(
        common_speculative * spec,
        llama_seq_id seq_id,
        llama_pos pos_base,
        const std::vector<llama_token> & ids,
        const std::vector<float> & hidden_rows) {
    const int32_t feature_width = common_speculative_feature_width(spec);
    if (feature_width <= 0 || ids.empty()) {
        return true;
    }

    const size_t expected_floats = ids.size() * (size_t) feature_width;
    if (hidden_rows.size() != expected_floats) {
        return false;
    }

    llama_batch accepted_batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(accepted_batch, ids[i], pos_base + (llama_pos) i, { seq_id }, true);
    }

    common_speculative_feature_view feature_view;
    const bool have_feature_view = common_speculative_feature_view_from_hidden_rows(
        hidden_rows, feature_width, seq_id, pos_base, feature_view);
    const int32_t ret = have_feature_view
        ? common_speculative_on_target_batch(spec, accepted_batch, feature_view, false)
        : -1;

    llama_batch_free(accepted_batch);
    return ret == 0;
}

bool common_speculative_commit_accepted_hidden_rows(
        common_speculative * spec,
        common_speculative_type spec_type_used,
        llama_seq_id seq_id,
        llama_pos pos_base,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        const std::vector<float> & hidden_rows) {
    if (common_speculative_feature_width(spec) <= 0 || ids.empty()) {
        return true;
    }

    std::vector<llama_token> commit_tokens;
    if (!common_speculative_build_commit_tokens(spec_type_used, sampled_before, ids, commit_tokens)) {
        return false;
    }

    return common_speculative_apply_hidden_rows(spec, seq_id, pos_base, commit_tokens, hidden_rows);
}

bool common_speculative_commit_accepted_output(
        common_speculative * spec,
        llama_context * ctx,
        common_speculative_type spec_type_used,
        llama_seq_id seq_id,
        llama_pos pos_base,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        const std::vector<int32_t> & output_indices) {
    if (common_speculative_feature_width(spec) <= 0 || ids.empty()) {
        return true;
    }

    std::vector<float> hidden_rows;
    if (!common_speculative_copy_output_hidden_rows(spec, ctx, output_indices, hidden_rows)) {
        return false;
    }

    return common_speculative_commit_accepted_hidden_rows(
        spec,
        spec_type_used,
        seq_id,
        pos_base,
        sampled_before,
        ids,
        hidden_rows);
}

static bool common_speculative_checkpoint_save(
        common_speculative_checkpoint & ckpt,
        llama_model * model,
        llama_context * ctx,
        common_sampler * sampler_src,
        const common_params_sampling & sparams,
        llama_seq_id seq_id,
        llama_pos n_past,
        llama_token sampled,
        int max_tokens,
        int ckpt_mode) {
    ckpt.clear();
    ckpt.n_past = n_past;
    ckpt.sampled = sampled;

    const int actual_mode = llama_spec_ckpt_init(ctx, spec_ckpt, ckpt_mode, max_tokens);
    if (actual_mode == LLAMA_SPEC_CKPT_NONE) {
        return false;
    }
    ckpt.mode = actual_mode;

    ckpt.valid = llama_spec_ckpt_save(ctx, spec_ckpt, seq_id);
    if (!ckpt.valid) {
        llama_spec_ckpt_discard(ctx, spec_ckpt);
        return false;
    }

    ckpt.sampler = common_sampler_init(model, sparams);
    if (ckpt.sampler == nullptr) {
        common_speculative_checkpoint_discard(ckpt, ctx);
        return false;
    }

    if (sampler_src != nullptr) {
        common_sampler_clone(sampler_src, ckpt.sampler);
    }

    return true;
}

const common_speculative_checkpoint * common_speculative_get_checkpoint(const common_speculative * spec) {
    return spec != nullptr ? &spec->checkpoint : nullptr;
}

void common_speculative_checkpoint_discard(
        common_speculative_checkpoint & ckpt,
        llama_context * ctx) {
    ckpt.clear();
    llama_spec_ckpt_discard(ctx, spec_ckpt);
}

bool common_speculative_checkpoint_restore(
        common_speculative_checkpoint & ckpt,
        common_speculative * spec,
        llama_context * ctx,
        common_sampler * sampler_dst,
        llama_seq_id seq_id,
        common_speculative_type spec_type_used,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        int n_draft,
        const std::vector<float> & mtp_hidden_state_pre,
        int32_t mtp_n_past_base) {
    if (!ckpt.valid) {
        return true;
    }

    const int step = (int) ids.size() - 1;
    const enum llama_spec_ckpt_restore_result restore_result = llama_spec_ckpt_restore_ex(
            ctx, spec_ckpt, seq_id, ckpt.n_past, ckpt.mode == LLAMA_SPEC_CKPT_PER_STEP ? step : 0);
    if (restore_result == LLAMA_SPEC_CKPT_RESTORE_FAILED) {
        LOG_ERR("%s: seq_id=%d speculative checkpoint restore failed\n", __func__, (int) seq_id);
        common_speculative_checkpoint_discard(ckpt, ctx);
        return false;
    }

    if (restore_result == LLAMA_SPEC_CKPT_RESTORE_DIRECT) {
        if (ckpt.sampler != nullptr && sampler_dst != nullptr) {
            common_sampler_clone(ckpt.sampler, sampler_dst);
        }
        if (sampler_dst != nullptr) {
            for (llama_token id : ids) {
                common_sampler_accept(sampler_dst, ctx, id, true);
            }
        }

        if (common_speculative_has_target_features(spec) && !mtp_hidden_state_pre.empty()) {
            if (!common_speculative_commit_accepted_hidden_rows(
                    spec,
                    spec_type_used,
                    seq_id,
                    mtp_n_past_base,
                    sampled_before,
                    ids,
                    mtp_hidden_state_pre)) {
                common_speculative_clear_sequence_hidden(spec, seq_id);
            } else if (spec_type_used != COMMON_SPECULATIVE_TYPE_MTP) {
                LOG_DBG("%s: seq_id=%d synced MTP target hidden state from accepted-prefix rows after per-step restore\n",
                        __func__, (int) seq_id);
            }
        }

        LOG_DBG("%s: seq_id=%d per-step restore: step=%d (rejected %d drafts)\n",
                __func__, (int) seq_id, step, (int) (n_draft - (ids.size() - 1)));
    } else {
        if (ckpt.sampler != nullptr && sampler_dst != nullptr) {
            common_sampler_clone(ckpt.sampler, sampler_dst);
        }

        if (!ids.empty()) {
            const int n_re = (int) ids.size();
            std::vector<float> redecoded_hidden;
            llama_batch re_batch = llama_batch_init(n_re, 0, 1);
            common_batch_add(re_batch, ckpt.sampled, ckpt.n_past, { seq_id }, n_re == 1);
            for (int j = 0; j < n_re - 1; ++j) {
                common_batch_add(re_batch, ids[j], ckpt.n_past + 1 + j, { seq_id }, j == n_re - 2);
            }

            if (common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
                for (int j = 0; j < re_batch.n_tokens; ++j) {
                    re_batch.logits[j] = true;
                }
                llama_set_embeddings(ctx, true);
            }

            const int ret = llama_decode(ctx, re_batch);
            if (ret != 0) {
                LOG_ERR("%s: seq_id=%d failed to re-decode accepted tokens after checkpoint restore: %d\n",
                        __func__, (int) seq_id, ret);
                llama_batch_free(re_batch);
                common_speculative_checkpoint_discard(ckpt, ctx);
                return false;
            }

            if (common_speculative_has_target_features(spec)) {
                std::vector<int32_t> redecoded_indices(n_re);
                for (int j = 0; j < n_re; ++j) {
                    redecoded_indices[j] = j;
                }

                if (!common_speculative_commit_accepted_output(
                        spec,
                        ctx,
                        spec_type_used,
                        seq_id,
                        ckpt.n_past,
                        sampled_before,
                        ids,
                        redecoded_indices)) {
                    common_speculative_clear_sequence_hidden(spec, seq_id);
                }
            }
            llama_batch_free(re_batch);

            if (sampler_dst != nullptr) {
                for (llama_token id : ids) {
                    common_sampler_accept(sampler_dst, ctx, id, true);
                }
            }

            LOG_DBG("%s: seq_id=%d spec checkpoint restored: re-decoded %d tokens (rejected %d drafts)\n",
                    __func__, (int) seq_id, n_re, (int) (n_draft - (ids.size() - 1)));
        }
    }

    common_speculative_checkpoint_discard(ckpt, ctx);
    return true;
}

bool common_speculative_commit(
        common_speculative * spec,
        llama_context * ctx,
        common_sampler * sampler_dst,
        llama_seq_id seq_id,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        int n_draft,
        llama_pos pos_base,
        const std::vector<int32_t> & accepted_output_indices) {
    GGML_ASSERT(spec != nullptr);
    GGML_ASSERT(!ids.empty());

    common_speculative_checkpoint & ckpt = spec->checkpoint;
    const common_speculative_type spec_type_used = spec->curr_impl != nullptr
        ? spec->curr_impl->type
        : COMMON_SPECULATIVE_TYPE_NONE;

    const int n_accepted = (int) ids.size() - 1;
    const bool any_rejected = n_accepted < n_draft;
    std::vector<float> mtp_hidden_state_pre;

    common_speculative_accept(spec, n_accepted);

    if (common_speculative_has_target_features(spec) &&
            any_rejected &&
            ckpt.valid &&
            !accepted_output_indices.empty()) {
        if (!common_speculative_copy_output_hidden_rows(spec, ctx, accepted_output_indices, mtp_hidden_state_pre)) {
            mtp_hidden_state_pre.clear();
        }
    }

    if (any_rejected && ckpt.valid) {
        const bool restored = common_speculative_checkpoint_restore(
            ckpt,
            spec,
            ctx,
            sampler_dst,
            seq_id,
            spec_type_used,
            sampled_before,
            ids,
            n_draft,
            mtp_hidden_state_pre,
            pos_base);
        return restored;
    }

    if (common_speculative_has_target_features(spec) && !accepted_output_indices.empty()) {
        if (!common_speculative_commit_accepted_output(
                spec,
                ctx,
                spec_type_used,
                seq_id,
                pos_base,
                sampled_before,
                ids,
                accepted_output_indices)) {
            common_speculative_clear_sequence_hidden(spec, seq_id);
        } else if (spec_type_used != COMMON_SPECULATIVE_TYPE_MTP) {
            LOG_DBG("%s: seq_id=%d synced MTP target hidden state from accepted-prefix rows\n",
                    __func__, (int) seq_id);
        }
    }

    llama_kv_cache_seq_rm(ctx, seq_id, pos_base + (llama_pos) (ids.size() - 1), -1);
    common_speculative_checkpoint_discard(ckpt, ctx);
    return true;
}

void common_speculative_print_stats(const common_speculative * spec, double slot_tps, int n_decoded, int n_past, common_params_speculative * active_params) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());

    }

    if (spec->tuner && spec->tuner->enabled && slot_tps > 0.0 && n_decoded > 0) {
        auto * mutable_spec = const_cast<common_speculative *>(spec);
        if (active_params) {
            mutable_spec->tuner->end_of_request(slot_tps, n_past, *active_params);
        } else {
            common_params_speculative tmp_params;
            mutable_spec->tuner->end_of_request(slot_tps, n_past, tmp_params);
        }
    }
}

// ----------------------------------------------------------------------------
// MTP
// ----------------------------------------------------------------------------

static common_speculative_state_mtp * common_speculative_get_mtp_state(common_speculative * spec) {
    if (!spec) {
        return nullptr;
    }

    for (auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_MTP) {
            continue;
        }

        if (auto * mtp_state = dynamic_cast<common_speculative_state_mtp *>(impl.get())) {
            return mtp_state;
        }
    }

    return nullptr;
}

static const common_speculative_state_mtp * common_speculative_get_mtp_state(const common_speculative * spec) {
    return common_speculative_get_mtp_state(const_cast<common_speculative *>(spec));
}

static common_speculative_state_dflash * common_speculative_get_dflash_state(common_speculative * spec) {
    if (!spec) {
        return nullptr;
    }

    for (auto & impl : spec->impls) {
        if (!common_speculative_type_is_dflash_family(impl->type)) {
            continue;
        }

        if (auto * dflash_state = dynamic_cast<common_speculative_state_dflash *>(impl.get())) {
            return dflash_state;
        }
    }

    return nullptr;
}

static const common_speculative_state_dflash * common_speculative_get_dflash_state(const common_speculative * spec) {
    return common_speculative_get_dflash_state(const_cast<common_speculative *>(spec));
}

static int32_t common_speculative_feature_width(const common_speculative * spec) {
    if (const auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        return dflash_state->n_target_features;
    }

    if (const auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
        return mtp_state->n_embd;
    }

    return 0;
}

static mtp_last_embd & mtp_get_last_embd(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    auto & last = state.draft_cache_by_seq[seq_id];
    if ((int) last.embd.size() != state.n_embd) {
        last.embd.resize(state.n_embd);
    }
    return last;
}

static void mtp_invalidate_cached_draft(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    auto it = state.draft_cache_by_seq.find(seq_id);
    if (it == state.draft_cache_by_seq.end()) {
        return;
    }

    it->second.last_id = -1;
    it->second.prob = 0.0f;
}

static void mtp_invalidate_cached_drafts(common_speculative_state_mtp & state) {
    for (auto & entry : state.draft_cache_by_seq) {
        entry.second.last_id = -1;
        entry.second.prob = 0.0f;
    }
}

static void mtp_store_target_hidden(
        common_speculative_state_mtp & state,
        llama_seq_id seq_id,
        const float * hidden,
        int32_t width) {
    if (hidden == nullptr || width <= 0) {
        return;
    }

    auto & stored = state.target_hidden_by_seq[seq_id];
    stored.assign(hidden, hidden + width);
}

static bool mtp_model_uses_recurrent_conditioning(const common_speculative_state_mtp & state) {
    if (state.ctx_mtp == nullptr) {
        return false;
    }
    return true;

    const llama_model * model = llama_get_model(state.ctx_mtp);
    if (!llama_model_has_recurrent(model)) {
        return false;
    }

    std::string arch{llama_model_arch_string(model)};
    return arch == "qwen35" || arch == "qwen35moe";
}

static void mtp_clear_target_hidden(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    state.target_hidden_by_seq.erase(seq_id);
    state.draft_cache_by_seq.erase(seq_id);
}

static bool common_speculative_capture_target_features(common_speculative * spec, const common_speculative_feature_view & features) {
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr || features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || features.width <= 0) {
        return false;
    }

    bool captured = false;
    for (const auto & row : features.rows) {
        if (row.data == nullptr) {
            continue;
        }

        mtp_store_target_hidden(*mtp_state, row.seq_id, row.data, features.width);
        mtp_invalidate_cached_draft(*mtp_state, row.seq_id);
        captured = true;
    }

    return captured;
}

bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id) {
    const auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return false;
    }

    auto it = mtp_state->target_hidden_by_seq.find(seq_id);
    return it != mtp_state->target_hidden_by_seq.end() && !it->second.empty();
}

void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id) {
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state != nullptr) {
        mtp_clear_target_hidden(*mtp_state, seq_id);
    }

    if (auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        dflash_clear_target_features(*dflash_state);
    }
}

void common_speculative_clear_sequence(
        common_speculative * spec,
        llama_seq_id seq_id,
        bool clear_companion_ctx) {
    if (spec != nullptr) {
        spec->checkpoint.clear();
        spec->curr_impl = nullptr;
        spec->last_n_drafted = 0;
        spec->t_step_start_us = 0;
        spec->last_step_target_only = false;
    }

    common_speculative_clear_sequence_hidden(spec, seq_id);

    if (clear_companion_ctx) {
        if (auto * ctx_mtp = common_speculative_get_companion_ctx(spec); ctx_mtp != nullptr) {
            llama_kv_cache_clear(ctx_mtp);
        }
    }
}

bool common_speculative_trim_sequence(
        common_speculative * spec,
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos pos_begin) {
    const bool target_trimmed = llama_kv_cache_seq_rm(ctx, seq_id, pos_begin, -1);
    if (auto * ctx_mtp = common_speculative_get_companion_ctx(spec); ctx_mtp != nullptr) {
        return target_trimmed && llama_kv_cache_seq_rm(ctx_mtp, seq_id, pos_begin, -1);
    }

    return target_trimmed;
}

void common_speculative_clear_sequence_kv(
        common_speculative * spec,
        llama_context * ctx,
        llama_seq_id seq_id) {
    common_speculative_clear_sequence(spec, seq_id);
    llama_kv_cache_seq_rm(ctx, seq_id, -1, -1);
    if (auto * ctx_mtp = common_speculative_get_companion_ctx(spec); ctx_mtp != nullptr) {
        llama_kv_cache_seq_rm(ctx_mtp, seq_id, -1, -1);
    }
}

llama_context * common_speculative_get_companion_ctx(common_speculative * spec) {
    if (auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
        return mtp_state->ctx_mtp;
    }

    if (auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        return dflash_state->ctx_dft;
    }

    return nullptr;
}

static int32_t mtp_accept_batch(
        common_speculative_state_mtp & state,
        const llama_batch & accepted_batch,
        llama_seq_id seq_id,
        const float * hidden_rows) {
    if (accepted_batch.n_tokens == 0 || hidden_rows == nullptr) {
        return 0;
    }

    const size_t hidden_rows_floats = (size_t) accepted_batch.n_tokens * state.n_embd;
    if (!llama_set_draft_input_hidden_state_copy(state.ctx_mtp, hidden_rows, hidden_rows_floats)) {
        return -1;
    }
    if (mtp_update_kv_cache(state.ctx_mtp, accepted_batch, false, state.mtp_heads_active) != 0) {
        return -1;
    }

    if (llama_model_is_openpangu(llama_get_model(state.ctx_mtp))) {
        // The one-token draft shortcut re-seeded below would skip re-decoding the last
        // sampled token next round, leaving a hole at its position. openPangu's KV cache
        // is position-addressed append-only (cell == position), so draft decodes must be
        // position-contiguous; decline the shortcut and let the next round decode the
        // sampled token normally.
        mtp_invalidate_cached_draft(state, seq_id);
        return 0;
    }

    auto & last = mtp_get_last_embd(state, seq_id);
    const float * embd = llama_get_embeddings_ith(state.ctx_mtp, accepted_batch.n_tokens - 1);
    if (embd != nullptr) {
        std::memcpy(last.embd.data(), embd, last.embd.size() * sizeof(float));
        if (!llama_set_draft_input_hidden_state_copy(state.ctx_mtp, last.embd.data(), last.embd.size())) {
            return -1;
        }
        last.last_id = common_sampler_sample_speculative(nullptr, state.ctx_mtp, accepted_batch.n_tokens - 1, &last.prob);
    }

    return 0;
}

int32_t common_speculative_on_target_batch(
        common_speculative * spec,
        const llama_batch & batch,
        const common_speculative_feature_view & features,
    bool is_prompt_warmup) {
    if (auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        if (features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || batch.n_tokens <= 0) {
            return 0;
        }

        if (features.width != dflash_state->n_target_features) {
            LOG_ERR("%s: DFlash feature width mismatch: got %d expected %d\n",
                    __func__, features.width, dflash_state->n_target_features);
            return -1;
        }

        if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.n_seq_id[0] <= 0 || batch.seq_id[0] == nullptr) {
            return -1;
        }

        const llama_seq_id seq_id = batch.seq_id[0][0];
        for (int i = 0; i < batch.n_tokens; ++i) {
            if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
                return -1;
            }
        }

        if (!dflash_append_target_features(*dflash_state, features, seq_id)) {
            return -1;
        }
        return 0;
    }

    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return 0;
    }

    if (features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || features.width <= 0 || batch.n_tokens <= 0) {
        return 0;
    }

    if (features.width != mtp_state->n_embd) {
        LOG_ERR("%s: MTP feature width mismatch: got %d expected %d\n",
                __func__, features.width, mtp_state->n_embd);
        return -1;
    }

    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.n_seq_id[0] <= 0 || batch.seq_id[0] == nullptr) {
        return -1;
    }

    const llama_seq_id seq_id = batch.seq_id[0][0];
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
            return -1;
        }
    }

    std::vector<float> hidden_rows_storage;
    if (!common_speculative_feature_view_copy_batch_rows(features, batch, seq_id, &hidden_rows_storage)) {
        return -1;
    }

    // snapshot the previous call's last hidden before the store below overwrites it (row-0 conditioning needs it)
    std::vector<float> prev_call_last_hidden;
    {
        const auto prev_it = mtp_state->target_hidden_by_seq.find(seq_id);
        if (prev_it != mtp_state->target_hidden_by_seq.end()) {
            prev_call_last_hidden = prev_it->second;
        }
    }

    const float * last_hidden = hidden_rows_storage.data() + (size_t) (batch.n_tokens - 1) * features.width;
    mtp_store_target_hidden(*mtp_state, seq_id, last_hidden, features.width);

    // track the minimum head count the committed context has been warmed with: a fresh
    // position-0 warmup resets it, everything after can only narrow it
    {
        const int32_t resolved = mtp_state->resolved_heads();
        if (is_prompt_warmup && batch.pos != nullptr && batch.n_tokens > 0 && batch.pos[0] == 0) {
            mtp_state->mtp_warmed_heads = resolved;
        } else {
            mtp_state->mtp_warmed_heads = mtp_state->mtp_warmed_heads > 0
                ? std::min(mtp_state->mtp_warmed_heads, resolved) : resolved;
        }
    }

    if (mtp_state->constant_draft_positions) {
        mtp_invalidate_cached_draft(*mtp_state, seq_id);
        return 0;
    }

    if (!is_prompt_warmup) {
        return mtp_accept_batch(*mtp_state, batch, seq_id, hidden_rows_storage.data());
    }

    const bool uses_shifted_hidden_rows = mtp_model_uses_recurrent_conditioning(*mtp_state);
    std::vector<float> previous_hidden_storage;
    if (uses_shifted_hidden_rows) {
        if ((int32_t) prev_call_last_hidden.size() == features.width) {
            previous_hidden_storage = std::move(prev_call_last_hidden);
        } else {
            previous_hidden_storage.assign(features.width, 0.0f);
        }
    }

    const float * conditioned_hidden_rows = hidden_rows_storage.data();
    std::vector<float> conditioned_hidden_storage;
    if (uses_shifted_hidden_rows) {
        conditioned_hidden_storage.resize(hidden_rows_storage.size());
        std::copy(previous_hidden_storage.begin(), previous_hidden_storage.end(), conditioned_hidden_storage.begin());
        if (batch.n_tokens > 1) {
            std::copy(
                hidden_rows_storage.begin(),
                hidden_rows_storage.begin() + (size_t) (batch.n_tokens - 1) * features.width,
                conditioned_hidden_storage.begin() + features.width);
        }
        conditioned_hidden_rows = conditioned_hidden_storage.data();
    }

    if (!llama_set_draft_input_hidden_state_copy(mtp_state->ctx_mtp, conditioned_hidden_rows, hidden_rows_storage.size())) {
        return -1;
    }
    const int32_t ret = mtp_update_kv_cache(mtp_state->ctx_mtp, batch, true, mtp_state->mtp_heads_active);
    mtp_invalidate_cached_draft(*mtp_state, seq_id);
    return ret;
}

common_speculative_type common_speculative_current_type(const common_speculative * spec) {
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return COMMON_SPECULATIVE_TYPE_NONE;
    }

    return spec->curr_impl->type;
}

common_speculative_metrics_snapshot common_speculative_get_metrics_snapshot(const common_speculative * spec) {
    common_speculative_metrics_snapshot snapshot;
    if (spec == nullptr) {
        return snapshot;
    }

    snapshot.stages.reserve(spec->impls.size());
    for (const auto & impl : spec->impls) {
        common_speculative_metrics_stage_snapshot stage;
        stage.type = impl->type;
        stage.n_call_begin = impl->n_call_begin;
        stage.n_call_draft = impl->n_call_draft;
        stage.n_call_accept = impl->n_call_accept;
        stage.n_gen_drafts = impl->n_gen_drafts;
        stage.n_acc_drafts = impl->n_acc_drafts;
        stage.n_gen_tokens = impl->n_gen_tokens;
        stage.n_acc_tokens = impl->n_acc_tokens;
        stage.drafted_by_position = impl->drafted_by_position;
        stage.accepted_by_position = impl->accepted_by_position;
        stage.t_begin_us = impl->t_begin_us;
        stage.t_draft_us = impl->t_draft_us;
        stage.t_accept_us = impl->t_accept_us;
        snapshot.stages.push_back(std::move(stage));
    }

    return snapshot;
}

void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past) {
    if (spec != nullptr) {
        spec->last_step_target_only = false;
        spec->t_step_start_us = 0;
    }
    auto * dflash_state = common_speculative_get_dflash_state(spec);
    if (dflash_state == nullptr) {
        auto * ctx_mtp = common_speculative_get_companion_ctx(spec);
        if (ctx_mtp == nullptr) {
            return;
        }
        llama_kv_cache_seq_rm (ctx_mtp, seq_id, kv_keep, kv_keep + kv_discard);
        llama_kv_cache_seq_add(ctx_mtp, seq_id, kv_keep + kv_discard, kv_past, -kv_discard);
    }

    if (dflash_state != nullptr) {
        dflash_context_shift(*dflash_state, kv_keep, kv_discard, kv_past);
    }
}

std::vector<llama_token> mtp_speculative_gen_draft(
    common_speculative_state_mtp & state,
    struct common_sampler * smpl,
    struct llama_context * ctx,
    int n_draft,
    float p_min,
    int32_t mtp_heads,
    llama_token id_last,
    llama_pos n_past,
    llama_seq_id seq_id,
    bool constant_draft_positions) {

    llama_tokens drafts;
    drafts.reserve(n_draft);

    if (!smpl) return drafts;

    if (n_draft <= 0) {
        mtp_invalidate_cached_draft(state, seq_id);
        return drafts;
    }

    common_sampler_reset(smpl);

    if (llama_model_is_openpangu(llama_get_model(ctx)) &&
        llama_kv_cache_seq_pos_max(ctx, seq_id) >= n_past) {
        // Position-addressed cache: drafting restarts at n_past, so any rows at or beyond
        // it (the accepted-update writes one row past the accepted prefix) must be dropped
        // first to keep the draft decode position-contiguous with the cache head.
        llama_kv_cache_seq_rm(ctx, seq_id, n_past, -1);
    }

    const int n_embd = llama_mtp_state_n_embd(ctx);
    const int n_mtp_heads_model = std::max(1, llama_model_n_nextn_layer(llama_get_model(ctx)));
    const int n_mtp_heads = mtp_heads > 0
        ? std::max(1, std::min((int) mtp_heads, n_mtp_heads_model))
        : n_mtp_heads_model;

    llama_batch mtp_batch = llama_batch_init(1, 0, 1);
    llama_set_mtp_n_heads(ctx, n_mtp_heads);
    llama_set_mtp_op_type(ctx, MTP_OP_DRAFT_GEN);

    float prob;
    auto prob_ptr = p_min > 0 ? &prob : nullptr;

    llama_token current_input_id = id_last;
    llama_pos current_n_past = n_past;

    auto & last = mtp_get_last_embd(state, seq_id);
    int i0 = 0;
    if (last.last_id >= 0) {
        if (last.prob < p_min) {
            n_draft = 1;
        }
        current_input_id = last.last_id;
        last.last_id = -1;
        drafts.push_back(current_input_id);
        current_n_past++;
        if (!llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size())) {
            llama_batch_free(mtp_batch);
            llama_set_mtp_step_idx(ctx, 0);
            llama_set_mtp_n_heads(ctx, 0);
            llama_set_mtp_op_type(ctx, MTP_OP_NONE);
            return drafts;
        }
        i0 = 1;
    }

    int n_decode = 0;
    for (int i = i0; i < n_draft; ++i) {
        mtp_batch.n_tokens = 0;
        const llama_pos draft_pos = constant_draft_positions ? n_past : current_n_past;
        common_batch_add(mtp_batch, current_input_id, draft_pos, {seq_id}, true);
        llama_set_mtp_step_idx(ctx, std::min(i, n_mtp_heads - 1));

        ++n_decode;
        if (llama_decode(ctx, mtp_batch) != 0) {
            break;
        }

        llama_token id_next = common_sampler_sample_speculative(smpl, ctx, 0, prob_ptr);

        if (i > 0 && prob_ptr && prob < p_min) {
            break;
        }

        drafts.push_back(id_next);

        const float * emb = llama_get_embeddings_ith(ctx, 0);
        if (!emb) {
            break;
        }

        // Keep a stable copy because later decode steps reuse ctx->embd storage.
        memcpy(last.embd.data(), emb, n_embd * sizeof(float));
        if (!llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size())) {
            break;
        }

        current_input_id = id_next;
        current_n_past++;

        if (prob_ptr && prob < p_min) {
            break;
        }
    }
    llama_batch_free(mtp_batch);
    llama_set_mtp_step_idx(ctx, 0);
    llama_set_mtp_n_heads(ctx, 0);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);

    // Purge the metadata for the draft tokens.
    // This prevents cache state corruption where two cells map to the same logical position.
    // If the state contained in `last` had a valid token id and probability, it means that we
    // have previously run an "accept" batch, where the token sampled from the main model was included.
    // Even in that case, the token at `n_past` is already committed and must remain in the KV cache,
    // so we only discard the speculative tail starting at `n_past + 1`.
    if (n_decode > 0) {
        llama_kv_cache_seq_rm(ctx, seq_id, n_past + 1, n_past + n_decode + 2);
    }

    return drafts;
}


int32_t mtp_update_kv_cache(struct llama_context * ctx, const llama_batch& batch, bool is_prompt_warmup, int32_t mtp_heads) {
    if (batch.n_tokens == 0) {
        return 0;
    }

    llama_seq_id seq_id    = batch.seq_id[0][0];
    llama_pos    start_pos = batch.pos[0];

    if (llama_kv_cache_seq_pos_max(ctx, seq_id) >= start_pos) {
        llama_kv_cache_seq_rm(ctx, seq_id, start_pos, -1);
    }

    LOG_DBG("[MTP-UPDATE|%s] Updating %d tokens for seq_id %d from pos %d...\n",
            is_prompt_warmup ? "PROMPT_WARMUP" : "GEN_ACCEPTED", batch.n_tokens, seq_id, (int)start_pos);

    // Warmup only updates MTP state and does not need logits.
    // Accepted-token updates need only the final logit for the next draft.
    llama_batch mtp_batch = batch;
    for (int i = 0; i < mtp_batch.n_tokens; ++i) {
        mtp_batch.logits[i] = false;
    }
    if (!is_prompt_warmup) {
        mtp_batch.logits[mtp_batch.n_tokens-1] = true;
    }
    if (is_prompt_warmup) {
        llama_set_mtp_n_heads(ctx, mtp_heads);
        llama_set_mtp_step_idx(ctx, 0);
        llama_set_mtp_op_type(ctx, MTP_OP_WARMUP);
    } else {
        llama_set_mtp_n_heads(ctx, mtp_heads);
        llama_set_mtp_step_idx(ctx, 0);
        llama_set_mtp_op_type(ctx, MTP_OP_UPDATE_ACCEPTED);
    }

    const int32_t ret = llama_decode(ctx, mtp_batch);
    llama_set_mtp_step_idx(ctx, 0);
    llama_set_mtp_n_heads(ctx, 0);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);
    return ret;
}

common_speculative_round_result common_speculative_run_round(
        common_speculative * spec,
        llama_model * model,
        llama_context * ctx,
        common_sampler * sampler,
        llama_context * ctx_guidance,
        common_params_speculative params,
        const common_params_sampling & sparams,
        llama_seq_id seq_id,
        llama_pos n_past,
        int n_predict_budget,
        bool have_carry,
        const llama_tokens & draft_history,
        llama_token carry_token) {
    common_speculative_round_result result;

    if (spec == nullptr || n_predict_budget == 1) {
        return result;
    }

    const int n_ctx = llama_n_ctx(ctx);
    const int n_batch = llama_n_batch(ctx);
    int max_usable_draft = params.get_max_stage_n_max();
    if (max_usable_draft <= 0) {
        max_usable_draft = params.n_max;
    }
    const int configured_n_max = common_speculative_get_configured_n_max(spec);
    if (configured_n_max > 0) {
        max_usable_draft = std::min(max_usable_draft, configured_n_max);
    }
    if (n_predict_budget >= 0) {
        max_usable_draft = std::min(max_usable_draft, n_predict_budget - 2);
    }
    max_usable_draft = std::min(max_usable_draft, n_ctx - (int) n_past - 2);
    max_usable_draft = std::min(max_usable_draft, n_batch - 1);

    // A normal speculative round needs room for the sampled token, at least one
    // draft position, and the verification carry.
    if (max_usable_draft <= 0) {
        return result;
    }

    params.n_max = std::max(0, max_usable_draft);
    params.n_min = std::min(std::max(0, params.n_min), params.n_max);
    for (auto & stage : params.stages) {
        if (stage.has_n_max_override()) {
            stage.n_max = std::min(stage.n_max, params.n_max);
        }
        if (stage.has_n_min_override()) {
            const int stage_max = stage.has_n_max_override() ? stage.n_max : params.n_max;
            stage.n_min = std::min(stage.n_min, stage_max);
        }
    }

    result.attempted = true;
    result.sampled_before_from_carry = have_carry;
    if (have_carry) {
        result.sampled_before = carry_token;
    } else {
        result.sampled_before = common_sampler_sample_legacy(sampler, ctx, ctx_guidance);
        common_sampler_accept(sampler, ctx, result.sampled_before, true);
    }
    result.sampled_before_ready = true;

    auto draft_result = common_speculative_draft_ex(
        spec,
        ctx,
        params,
        draft_history,
        result.sampled_before,
        n_past,
        seq_id,
        &sparams);
    auto & draft = draft_result.tokens;
    auto & proposal_dists = draft_result.proposal_dists;
    const int min_usable_draft = params.get_min_usable_stage_n_min();
    if ((int) draft.size() < min_usable_draft || (draft.empty() && !draft_result.target_only)) {
        return result;
    }

    if (!proposal_dists.empty() && proposal_dists.size() != draft.size()) {
        result.failed = true;
        result.error = "DFlash2 proposal distribution count does not match draft";
        return result;
    }

    if (common_speculative_needs_checkpoint(model)) {
        if (!common_speculative_before_draft(
                spec,
                model,
                ctx,
                sampler,
                sparams,
                seq_id,
                n_past,
                result.sampled_before,
                (int) draft.size() + 1,
                params.spec_ckpt_mode)) {
            draft.clear();
        }
    }

    if (draft.empty() && !draft_result.target_only) {
        return result;
    }

    llama_batch verify_batch = llama_batch_init((int) draft.size() + 1, 0, 1);
    std::vector<int> verify_indices;
    verify_indices.reserve(draft.size() + 1);

    common_batch_add(verify_batch, result.sampled_before, n_past, { seq_id }, true);
    verify_indices.push_back(0);
    for (size_t i = 0; i < draft.size(); ++i) {
        common_batch_add(verify_batch, draft[i], n_past + 1 + (llama_pos) i, { seq_id }, true);
        verify_indices.push_back((int) i + 1);
    }

    if (llama_decode(ctx, verify_batch) != 0) {
        llama_batch_free(verify_batch);
        result.failed = true;
        result.error = "speculative verify decode failed";
        return result;
    }
    std::vector<llama_token> ids;
    try {
        ids = proposal_dists.empty()
            ? common_sampler_sample_and_accept_n(sampler, ctx, verify_indices, draft)
            : common_sampler_sample_and_accept_n(sampler, ctx, verify_indices, draft, proposal_dists);
    } catch (const std::exception & e) {
        llama_batch_free(verify_batch);
        result.failed = true;
        result.error = e.what();
        return result;
    }

    std::vector<int32_t> accepted_output_indices;
    if (!ids.empty()) {
        accepted_output_indices.assign(verify_indices.begin(), verify_indices.begin() + ids.size());
    }

    if (!ids.empty()) {
        common_speculative_commit(
            spec,
            ctx,
            sampler,
            seq_id,
            result.sampled_before,
            ids,
            (int) draft.size(),
            n_past + 1,
            accepted_output_indices);
        result.ids = std::move(ids);
        result.used_speculative = true;
    }

    llama_batch_free(verify_batch);
    return result;
}
