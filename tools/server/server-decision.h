#pragma once

#include "server-common.h"
#include "server-task.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

// typed decision models (TypeSafe /v1/systemone API)
// the model answers each question in one forward pass, no token is generated

// Token limits that belong to the model, not to the server: a GGUF carries them as "<arch>.decision.<key>"
// (max_state_tokens, max_tokens, max_tokens_image, max_tokens_audio, max_option_tokens, n_options_max). A key that is absent,
// not a number or zero keeps the value below, which are the limits of the models that were converted before the
// keys existed, so an old GGUF behaves as before.
struct server_decision_limits {
    size_t h2o_max_state_tokens     = 32000; // h2o-lightning serve_config.json prompt.max_state_tokens
    size_t d1omni_max_tokens        = 16384; // lfm2-d1-omni config max_length
    size_t d1omni_max_tokens_image  = 896;   // config image_text_length
    size_t d1omni_max_tokens_audio  = 15360; // config audio_text_length
    size_t max_option_tokens        = 48;    // PROVENANCE: reference-code literal, NO config key exists (2026-11-09 audit of the HF configs and reference code of all six decision types: laya rl_common.py:60 and Julia-1 julia/data.py:86 hardcode x[:48]; kai/d1/d1-omni/lev/kev have no per-option cap). The reader accepts "max_option_tokens" if a converter ever writes it; none does, so this 48 is the value in use. Used by the laya marker layout (model and N budgets) only
    size_t n_options_max            = 255;   // PROVENANCE: model-metadata (key n_options_max, 255 = fallback). options one question may carry (marker-token models: kev, laya, clef, vllm-decision2, lfm2-d1, lfm2-d1-omni); 255 = the uint8 marker capacity of the converter format

    // get(key) returns the value of "<arch>.decision.<key>" or "" if the model has none
    static server_decision_limits from_meta(const std::function<std::string(const std::string &)> & get);
};

// options one question may carry, for the model types whose count is not set by their label list (0 = the label list sets it)
size_t server_decision_n_options_max(common_decision_type type, const server_decision_limits & limits);

enum server_decision_question_type {
    SERVER_DECISION_QUESTION_CHOICE,
    SERVER_DECISION_QUESTION_SCORE,
    SERVER_DECISION_QUESTION_NOUL,
};

struct server_decision_option {
    std::string key;
    json description; // null if not provided
};

struct server_decision_question {
    std::string id;
    server_decision_question_type type;
    json instructions;
    std::vector<server_decision_option> options; // in the order of the model outputs
    // per-request overrides (body "decision_budget", "decision_window_agg"), empty = the server default
    std::string budget;
    std::string window_agg;
    std::string window;
};

struct server_decision_context {
    common_decision_type type = COMMON_DECISION_TYPE_NONE;

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
    void init(const llama_model * model);

    // tests only (tests/test-server-systemone.cpp): a decision context without a model -- labels are token ids
    // 0..n_labels-1, no template, no vocab. Fork port (server-systemone.cpp:139-148 before the 2026-11-04 merge).
    void init_for_test(common_decision_type t, size_t n_labels, bool true_first, const std::map<std::string, float> & temps);

    // true if the questions of a request start with the same tokens, and the model can continue from them
    bool can_share_prompt() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_GRANITE_SWITCH:
            case COMMON_DECISION_TYPE_LEV:
            case COMMON_DECISION_TYPE_KEV:
            case COMMON_DECISION_TYPE_NIMBLE:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_VLLM_DECISION2:
            case COMMON_DECISION_TYPE_LFM2_D1:
            case COMMON_DECISION_TYPE_H2O_LIGHTNING:
                return true;
            default:
                return false;
        }
    }

    // true if all the questions of a request go in one prompt, see fill_task_joint()
    bool is_joint() const {
        return type == COMMON_DECISION_TYPE_CLEF;
    }

    // true if the prompt of the model has a place for images
    bool can_use_images() const {
        switch (type) {
            case COMMON_DECISION_TYPE_OPENJEV:
            case COMMON_DECISION_TYPE_CLEF:
            case COMMON_DECISION_TYPE_PPLX_DECIDER:
            case COMMON_DECISION_TYPE_VLLM_D3:
            case COMMON_DECISION_TYPE_LFM2_D1:
            case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
            case COMMON_DECISION_TYPE_H2O_LIGHTNING:
                return true;
            default:
                return false;
        }
    }

    // --decision-budget / --decision-window-agg, and the largest prompt the context takes in one pass (0 = not known):
    // the smaller of the slot context and, for a non-causal model, the physical batch
    void set_serving(const std::string & budget, const std::string & window_agg, const std::string & window, size_t n_fit, size_t n_ctx_train);
    size_t get_n_fit() const { return n_fit; }
    const server_decision_limits & get_limits() const { return limits; }
    void set_limits_for_test(const server_decision_limits & l) { limits = l; }
    // the aggregation of the windows of this question
    std::string get_window_agg(const server_decision_question & question) const;

    // throw std::invalid_argument on bad input
    std::vector<server_decision_question> parse_questions(const json & body) const;

    // the state as the model reads it, done once per request; returns the state for most models
    json prepare_state(const json & state) const;

    // returns the state without its images, they are appended to files in order
    // images come from "files" (alias "images") and from the image_url and input_audio parts of a state made of chat messages
    // an image can be an audio clip if the model supports it
    json parse_state(const json & body, std::vector<raw_buffer> & files) const;

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const server_decision_question & question) const;

    // set the prompt of one variant of this question, and where to read its result
    // mctx is only used if there are files
    void fill_task(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task,
            size_t window = 0) const; // window: which window of a long state (window 0 sets task.decision.n_windows)

    // set the prompt of all the questions, the result has the scores of all their options, in order
    // mctx is only used if there are files
    void fill_task_joint(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    // scores: the raw model outputs of each variant
    json format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores, bool has_media = false) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    std::map<size_t, std::vector<float>> score_bias; // added to the scores of a score question, by number of levels
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]
    bool   noul_criteria_order = false; // noul options are in the order of the criteria keys, if given
    bool   choice_sorted   = false; // choice options are in the order of their keys
    bool   strict_payload  = false; // state, instructions and descriptions must be text, an object or an array
    bool   canonical_legend = false; // non-string score descriptions are returned as canonical JSON text

    // OPENJEV, LEV, NIMBLE, PPLX_DECIDER, VLLM_D3, H2O_LIGHTNING
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    // LAYA, KEV
    llama_token token_marker      = LLAMA_TOKEN_NULL;
    llama_token token_sep         = LLAMA_TOKEN_NULL;
    std::string text_marker;
    size_t      max_head_tokens   = 0; // question + options
    server_decision_limits limits; // from the GGUF, see init()

    // --decision-budget
    std::string budget_default   = "auto";
    std::string agg_default      = "max";
    std::string window_default   = "auto";
    size_t      n_fit            = 0; // largest prompt of one pass
    size_t      n_ctx_train      = 0;

    // H2O_LIGHTNING
    std::vector<std::string> quoted_spellings; // texts of the special tokens, quoted in the input, longest first
    std::string              quoted_first;     // the first bytes of these texts

    std::string render(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            size_t n_images,
            bool is_audio = false) const;
    json render_options(const server_decision_question & question, size_t variant) const;
    size_t n_outputs(const server_decision_question & question) const;
    // LFM2_D1: label text and tokens of each option
    void d1_labels(const server_decision_question & question, std::vector<std::string> & texts, std::vector<llama_tokens> & groups) const;
    // H2O_LIGHTNING: the input of the template, its texts are rendered as the reference shim does
    json render_h2o(const json & state, const server_decision_question & question) const;
    std::string h2o_cut_state(const std::string & text) const;
    std::string h2o_quote(const std::string & text) const;
    void fill_task_laya(llama_tokens & tokens, const server_decision_question & question, server_task & task, size_t window) const;
    void fill_task_d1omni(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            const std::vector<raw_buffer> & files,
            mtmd_context * mctx,
            const mtmd_helper_init_opt & init_opt,
            server_task & task) const;

    float get_temperature(const server_decision_question & question) const;
    double get_decision_confidence(server_decision_question_type type, const std::vector<double> & probs) const;
};

// combine the per-option logits of the windows of a long state: max (MaxP), mean (AvgP), lse (log-sum-exp) or first
std::vector<float> server_decision_aggregate(const std::vector<std::vector<float>> & windows, const std::string & agg);

// group the tasks so that the common prefix of their prompts is evaluated only once
// each group is one parent and its children, it takes at most n_slots slots
// note: the order of the tasks is preserved
std::vector<server_task> server_decision_group_tasks(std::vector<server_task> && tasks, size_t n_slots);

// write "<|name|>" with U+00A6 in place of "|", so text from the request is not parsed as special tokens
std::string server_decision_escape_special_tokens(const std::string & text);
