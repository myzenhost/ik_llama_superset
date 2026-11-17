#pragma once

#include "common.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "subproc.h"

#include "json.h"
#include "server-effort.h"
#include "server-answer-reserve.h"
#include "server-latency.h"
#include "server-request-log.h"
#include "server-budget-group.h"
#include "server-retry.h"
#include "server-load-fail.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <mutex>
#include <thread>
#include <queue>
#include <string>
#include <vector>

using json = common_json;

#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_TRC(slot, fmt, ...) LOG_TRC("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT(""                                 fmt,                                                                __VA_ARGS__)

#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_TRC(fmt, ...) LOG_TRC("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key).get<T>();
        } catch (const common_json_error & err) {
            LOG_WRN("Wrong type supplied for parameter '%s', using default value: %s\n", key.c_str(), err.what());
            return default_value;
        }
    } else {
        return default_value;
    }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_EXCEED_CONTEXT_SIZE, // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}
    server_grammar_trigger(const json & in) {
        value.type = (common_grammar_trigger_type) in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token) in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out {
            {"type", (int) value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int) value.token;
        }
        return out;
    }
};

json format_error_response(const std::string & message, const enum error_type type);

// HTTP status of an error body made by format_error_response: error.code when it is a number, else error.status
// (the context_length_exceeded error carries the OpenAI string in "code" and the number in "status"), else fallback
int error_http_status(const json & error_data, int fallback = 500);
// path of the running llama-server binary, used to spawn siblings. throws on failure
std::filesystem::path get_server_exec_path();

//
// random string / id
//

std::string random_string();
std::string gen_chatcmplid();
std::string gen_tool_call_id();

// get a random marker; note: each time the server restarts, the marker will be different
const char * get_media_marker();

//
// model output modalities
//

// output modalities for architecture.output_modalities in GET /models
// has_cap_chat: false for a tts-only model, has_cap_tts: the model can generate audio
std::vector<std::string> server_model_output_modalities(common_decision_type decision_type, bool has_cap_chat, bool has_cap_tts);
// same, with the decision type read from a loaded model
std::vector<std::string> server_model_output_modalities(const llama_model * model, bool has_cap_chat, bool has_cap_tts);

// architecture object of GET /models; shared by the direct server and the router
json server_model_architecture_json(
        bool inp_image,
        bool inp_audio,
        bool inp_video,
        const std::vector<std::string> & output_modalities);

//
// lora utils
//

// check whether the given lora set has only aloras activated (empty => false)
bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras);

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next);

std::map<int, float> parse_lora_request(const json & data);

// Shape check of a request's "lora" list: an array of objects, each with an integer "id" of an adapter this server loaded
// (0 <= id < n_server_adapters) and an optional numeric "scale". Throws std::runtime_error with the message the completion
// schema has always given, so a wrong id is an error instead of a silent no-op.
void validate_lora_request(const json & lora, size_t n_server_adapters);

// The per-request adapter fields of a request body, for handlers that do not go through the completion schema
// (/v1/embeddings, /rerank): "lora" (list of {id, scale}), "lora_adapter" (a name in --lora-pool-dir) and
// "lora_adapter_scale". Same fields, same validation and same messages as the completion schema. A field that is absent
// leaves its output untouched. Throws std::runtime_error.
void parse_request_lora_fields(const json & body, size_t n_server_adapters,
                               std::map<int, float> & lora, std::string & lora_adapter, float & lora_adapter_scale);

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2);

// get the ids of all enabled loras
std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras);

struct server_tokens; // defined below

// Fingerprint of the LoRA configuration that shapes the KV of a prompt: a hash over (adapter pointer, scale) of every
// adapter whose scale is not 0, in list order. 0 = no adapter active -- the key every server without a LoRA has
// always used, so callers that never pass a fingerprint keep their historical cache behaviour. Nonzero otherwise.
// The pointer is a process-lifetime identity, which is all the in-memory caches that consume this need.
// Row S69-LORAX-PICK-LORA-CONFIG-IN-PROMPT-CACHE-AND-RADIX-KEY-2026-09-29-01.
uint64_t lora_config_fingerprint(const std::vector<common_adapter_lora_info> & loras);

// Start index of the LAST occurrence of `lora`'s aLoRA invocation sequence in `tokens`; tokens.size() when the
// adapter has no invocation sequence or it does not occur. (launch_slot_with_task's search, shared so the cache
// lookup and the launch resolve a request's aLoRA state identically.)
size_t lora_find_alora_invocation_start(const struct llama_adapter_lora * lora, const server_tokens & tokens);

// Fingerprint of the configuration a request with prompt `tokens` and adapter list `loras` actually RUNS under: an
// aLoRA-only list whose invocation sequence is absent from the prompt is deactivated by launch_slot_with_task, so
// it runs on the base weights and gets fingerprint 0, exactly like a request that named no adapter.
uint64_t lora_effective_fingerprint(std::vector<common_adapter_lora_info> loras, const server_tokens & tokens);

// Fingerprint of the control vector currently added to the residual stream: FNV-1a over the combined (already
// scaled) data, n_embd and the layer range. 0 when `data` is empty, which is the no-vector state. A control vector
// changes every KV cell computed under it exactly like an adapter does, so the prompt caches key on it too.
// Row S69-VLLM-PICK-CONTROL-VECTOR-CACHE-INVALIDATION-2026-09-29-01.
uint64_t control_vector_fingerprint(const float * data, size_t n, int32_t n_embd, int32_t layer_start, int32_t layer_end);

// The key the prompt caches use: the LoRA fingerprint with the control-vector fingerprint folded in. Returns
// `lora_fp` unchanged when `cv_fp` is 0, so a server without a control vector keeps its historical keys.
uint64_t kv_config_fingerprint(uint64_t lora_fp, uint64_t cv_fp);
// Folds the request's cache namespace (cache_salt, extra_key) into a configuration fingerprint. Both empty -> fp unchanged.
uint64_t kv_namespace_fingerprint(uint64_t fp, const std::string & cache_salt, const std::string & extra_key);

//
// server_tokens
//

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
    bool has_mtmd = false;

private: // disallow accessing these members directly, risking out-of-sync

    // map a **start** index in tokens to the image chunk
    // note: the order need to be in-sync with tokens
    std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;

    // list of tokens
    //   if the token is LLAMA_TOKEN_NULL, it indicates that this position is occupied by media chunk
    //   otherwise, it is a normal text token
    // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
    // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
    llama_tokens tokens;

    // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
    //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
    // idx  0   1   2   3   4   5      6      7      8      9      10
    // pos  0   1   2   3   4   5      5      5      7      7      7
    // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // Prevent copying
    // TODO: server_tokens should be copyable - remove this:
    server_tokens(const server_tokens&) = delete;
    server_tokens& operator=(const server_tokens&) = delete;

    // Allow moving (usually implicitly generated if members are movable)
    server_tokens(server_tokens&&) = default;
    server_tokens& operator=(server_tokens&&) = default;

    // Allow accessing elements using [] operator
    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token& operator[](size_t index) const { return tokens[index]; }

    server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd);
    server_tokens(const llama_tokens & tokens, bool has_mtmd);

    // for debugging
    std::string str() const;

    // the next position after n_tokens. if n_tokens < 0, return the next position after all tokens.
    llama_pos pos_next(int64_t n_tokens = -1) const;

    // number of tokens with position < max_pos
    size_t size_up_to_pos(llama_pos max_pos) const;

    const mtmd::input_chunk_ptr & find_chunk(size_t idx) const;

    // find next media chunk after idx
    // returns a pair of pointer to the chunk (nullptr if not found) and its start index in tokens
    std::pair<const mtmd::input_chunk_ptr *, size_t> find_next_media_chunk(size_t idx) const;

    void push_back(llama_token tok);

    // will create a copy of the chunk if it contains non-text data
    void push_back(const mtmd_input_chunk * chunk);

    // same as push_back, but media chunks are stored as placeholders (no image/audio data)
    // only use this if the chunk will never be encoded again (e.g. it is already in the KV cache)
    void push_back_placeholder(const mtmd_input_chunk * chunk);

    // appends server tokens, updates the media map. copies media chunks.
    void push_back(server_tokens & tokens);

    // for compatibility with context shift and prompt truncation
    void insert(const llama_tokens & inp_tokens);

    // for compatibility with speculative decoding, ctx shift
    const llama_tokens & get_tokens() const;

    llama_tokens get_text_tokens() const;

    std::vector<char> serialize() const;
    static server_tokens deserialize(const llama_tokens & packed, bool has_mtmd);

    // for compatibility with speculative decoding
    void set_token(llama_pos pos, llama_token id);

    size_t size() const { return tokens.size(); }

    bool empty() const { return tokens.empty(); }

    void clear() {
        map_idx_to_media.clear();
        tokens.clear();
    }

    void keep_first(size_t n);

    std::string detokenize(const llama_context * ctx, bool special) const;

    size_t get_common_prefix(const server_tokens & b) const;

    // CAPABILITY-GENUINELY-NA: this "reasoning"/"think" token exclusion is about the SERVED model's
    // own <think> reasoning-trace tags in llama-server output, unrelated to Claude's own reasoning or
    // any harness capability.
    // Ported from the pre-merge fork's examples/server/server-common.h:460 @ 35f0cac75 (the fork's
    // own MIT code) -- row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01,
    // item 2. Returns a copy with any think_token.begin/end-delimited reasoning-trace span removed,
    // so slot-similarity and prompt-cache comparisons are not thrown off by reasoning content that
    // differs between turns even when the visible prefix is identical. When think_token.exclude is
    // false, behaves like clone().
    server_tokens get_tokens_exclude_think(const llama_context * ctx, const thinking_tokens & think_token) const;

    // split the tokens into message spans, skipping over media chunks
    common_chat_msg_spans find_message_spans(const common_chat_msg_delimiters & delims) const;

    // check text token IDs and the mapping between media chunks and token ranges
    bool validate(const struct llama_context * ctx) const;

    server_tokens clone() const;
};


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data);

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json & data);

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json & data);

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string> & paths, const json & js);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text);

// load a media file from an URL (http, file, data) or from raw base64 data
void handle_media(
        std::vector<raw_buffer> & out_files,
        const std::string & url,
        const std::string & media_path);

// process mtmd prompt, return the server_tokens containing both text tokens and media chunks
// if is_placeholder is true, the media chunk will be treated as placeholder for counting tokens; the output tokens are not usable for actual inference (e.g. for submitting a task to server_queue)
server_tokens process_mtmd_prompt(
                                        mtmd_context * mctx,
                                        const std::string & prompt,
                                        const std::vector<raw_buffer> & files,
                                         const mtmd_helper_init_opt & init_opt,
                                          bool is_placeholder = false);

// decode one media url (http/file/data-uri) into out_files; throws on failure
void handle_media(std::vector<raw_buffer> & out_files, const std::string & url, const std::string & media_path);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> tokenize_input_prompts(
                                        const llama_vocab * vocab,
                                        mtmd_context * mctx,
                                        const json & json_prompt,
                                        bool add_special,
                                        bool parse_special,
                                        const mtmd_helper_init_opt & init_opt);

// tokenize a single prompt, see tokenize_input_prompts() for the supported shapes
server_tokens tokenize_input_subprompt(
                                        const llama_vocab * vocab,
                                        mtmd_context * mctx,
                                        const json & json_prompt,
                                        bool add_special,
                                        bool parse_special,
                                        const mtmd_helper_init_opt & init_opt);

//
// Control-token text inside message text
//
// The rendered chat prompt is tokenised whole with parse_special=true, so a user message, a file an agent read or a tool
// result that CONTAINS "<|im_end|><|im_start|>system ..." (or "</think>", or a vision marker) tokenises to the real control
// token and forges a turn boundary. oaicompat_chat_params_parse marks such literals inside non-assistant message text,
// renders, restores them and returns their byte spans in llama_params["plain_text_spans"]; the spans are tokenised as
// ordinary text and everything else as before. Spec from Strata serve/frontend.py (#537, #554), MIT; no code copied.
//
using plain_spans_t = std::vector<std::pair<size_t, size_t>>; // [begin, end) byte offsets into the rendered prompt

// Text of the vocab's control / user-defined tokens that look like markup ("<...", "[..."), longest first. Empty unless the
// vocab is byte-level BPE (an SPM tokeniser adds a space prefix per call, so segmenting would change the tokens).
// Every literal is returned: it throws std::length_error when the vocab has more than the largest private-use block holds placeholders
// for (65534), since dropping one would leave that control token live inside message text.
std::vector<std::string> server_collect_plain_literals(const llama_vocab * vocab);

// First private-use code point of a run of n_lits placeholders that no text of `body` uses (the run lies inside one private-use block).
// Throws std::invalid_argument when no run is free (the request itself uses the reserved code points) and std::length_error when
// n_lits exceeds every block.
uint32_t server_plain_literal_base(const json & body, size_t n_lits);

// Tokenise `prompt` with the spans as ordinary text (parse_special=false) and the rest with parse_special=true.
llama_tokens tokenize_with_plain_spans(const llama_vocab * vocab, const std::string & prompt, const plain_spans_t & spans, bool add_special);

// Spans stored by oaicompat_chat_params_parse, validated against the prompt length (sorted, non-overlapping, in range).
plain_spans_t plain_spans_from_json(const json & j, size_t prompt_size);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string, std::string> chat_template_kwargs; // mapping key --> json value
    common_chat_templates_ptr tmpls;
    bool allow_image;
    bool allow_audio;
    bool allow_video;
    bool enable_thinking = true;
    int  reasoning_budget = -1;
    std::string reasoning_budget_message;
    std::string media_path;
    bool force_pure_content = false;
    // fork -ptcall / --parallel-tool-calls: server-side DEFAULT when a request omits "parallel_tool_calls"
    // (the request field still wins). Restored 2026-09-27; the upstream merge 279b4bfaf dropped the flag.
    bool parallel_tool_calls_default = false;
    // --system-prompt / --system-prompt-file / fork -spf: inserted as the first message of a chat request that
    // has no system message of its own. The old fork server prefixed it to every slot (system_prompt_set);
    // with chat templates a system MESSAGE is the equivalent that templates render correctly.
    std::string default_system_prompt;
    // --tool-arg-coercion: server-side DEFAULT for repairing tool-call argument types to the request's tool schemas
    // (the request field "tool_arg_coercion" wins). Applies to non-streaming responses only.
    bool tool_arg_coercion = true;
    // --tool-call-format-json / --tool-call-format-preset: the JSON of a common_chat_xml_tool_format, "" = none (validated at startup).
    std::string tool_call_format;
    // server_collect_plain_literals(vocab): control-token text neutralised inside message text. Empty = feature off.
    std::vector<std::string> plain_literals;
    // --reasoning-effort-budget: OpenAI reasoning_effort -> thinking-token budget (server-effort.h); empty = not in use.
    // n_predict_default is the server's own generation cap (--n-predict), -1 = none.
    server_effort_table effort_budget;
    int n_predict_default = -1;
    // --reasoning-answer-reserve: thinking budget = generation limit - reserve when a request sets no budget
    // (server-answer-reserve.h); inactive = off. The request field "reasoning_answer_reserve" overrides it.
    server_answer_reserve answer_reserve;
    // --reasoning-effort-high-text (empty = the Qwen3.8 sentence): the chat parser marks a request whose user text holds it
    // (reasoning_loop_user_has_sentence), so reasoning_loop_recovery never splices inside a user's words
    std::string rlr_high_text;
    // --chat-template-kwargs-force: keys that replace the request's chat_template_kwargs (server-template-kwargs-force.h)
    std::map<std::string, std::string> chat_template_kwargs_force;
};

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body);

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    const llama_vocab * vocab,
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files,
    common_chat_session & out_session);

// used by /embeddings endpoint, content has the same format as a chat message content array
server_tokens tokenize_oai_content_array(
    const llama_vocab * vocab,
    mtmd_context * mctx,
    const server_chat_params & opt,
    json content,
    bool add_special,
    bool parse_special,
    const mtmd_helper_init_opt & init_opt);

// TODO: move it to server-task.cpp
json format_embeddings_response_oaicompat(
    const json & request,
    const std::string & model_name,
    const json & embeddings,
    bool use_base64 = false,
    const std::string & quant_name = "");   // "binary" | "ubinary" | "int8": the embeddings already hold the packed integers

// TODO: move it to server-task.cpp
json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n);

//
// stats and metrics
//

// shared between server_slot and server_task_result_*
struct server_slot_stats {
    uint64_t n_prompt_cached    = 0;
    uint64_t n_prompt_processed = 0;
    uint64_t n_gen              = 0;

    // speculative decoding stats
    // note: the per-position breakdown lives in server_slot, it is not needed in a task result
    uint64_t n_draft_tokens      = 0;
    uint64_t n_draft_accepted    = 0;
    uint64_t n_draft_verif_steps = 0;

    // grammar jump-forward: tokens appended without their own sampling step
    uint64_t n_jump_forward_tokens = 0;

    // reasoning_loop_recovery (server-loop-recovery.h): retries that ran (0 or 1), and the last coverage score the word-level
    // detector computed (-1 = it never ran). Reported in timings when the request turned the feature on.
    bool    rlr_on                 = false;
    int32_t n_reasoning_recoveries = 0;
    float   reasoning_coverage     = -1.0f;
    std::string rlr_note;              // why a detected loop was NOT recovered (empty when it was, or none was found)

    // these are absolute timestamps (in us)
    // note: must be signed - they are subtracted before the later ones are set
    int64_t t_start       = 0;
    int64_t t_prompt_last = 0;
    int64_t t_gen_last    = 0;

    // latency instruments (server-latency.h). t_submit is when the request entered the server, so t_start - t_submit
    // is the queue time; 0 = unknown, in which case TTFT is measured from t_start
    int64_t t_submit      = 0;
    int64_t t_first_token = 0;
    // inter-token latency: the gap between consecutive token-EMITTING steps of this request. A speculative step
    // that accepts several tokens is one emission, so it contributes one gap, not several zero gaps.
    int64_t  itl_sum_us = 0;
    int64_t  itl_min_us = 0;
    int64_t  itl_max_us = 0;
    uint64_t n_itl      = 0;
    int64_t  itl_last_us = -1; // the gap the latest update_gen_last() added; -1 when it was the first token

    // can only move one direction: start -> prompt -> gen
    void update_prompt_start(int64_t t_submit_us = 0) {
        GGML_ASSERT(t_start == 0);
        t_start  = ggml_time_us();
        t_submit = t_submit_us > 0 && t_submit_us <= t_start ? t_submit_us : t_start;
    }
    void set_prompt_last(int64_t t_us) {
        GGML_ASSERT(t_start > 0);
        t_prompt_last = t_us;
    }
    void update_prompt_last() {
        set_prompt_last(ggml_time_us());
    }
    void update_gen_last() {
        GGML_ASSERT(t_prompt_last > 0);
        const int64_t t = ggml_time_us();
        if (t_first_token == 0) {
            t_first_token = t;
            itl_last_us   = -1;
        } else {
            const int64_t gap = std::max<int64_t>(0, t - t_gen_last);
            itl_last_us = gap;
            itl_sum_us += gap;
            itl_min_us  = n_itl == 0 ? gap : std::min(itl_min_us, gap);
            itl_max_us  = std::max(itl_max_us, gap);
            n_itl++;
        }
        t_gen_last = t;
    }

    // time to first token, from submission (or from the slot start when the submit time is unknown); 0 = no token yet
    double ttft_ms() const {
        if (t_first_token == 0) {
            return 0.0;
        }
        return (t_first_token - (t_submit > 0 ? t_submit : t_start)) / 1000.0;
    }
    // mean inter-token latency over the tokens after the first (a speculative burst spreads its gap over its tokens)
    double itl_mean_ms() const {
        return n_gen > 1 ? itl_sum_us / 1000.0 / (double) (n_gen - 1) : 0.0;
    }

    // these are time durations
    int64_t t_elapsed_us() const {
        return ggml_time_us() - t_start;
    }
    double t_prompt_ms() const {
        if (t_prompt_last == 0) {
            return 0.0; // the prompt is not processed yet
        }
        return (t_prompt_last - t_start) / 1000.0;
    }
    int64_t t_gen_us() const {
        if (t_gen_last == 0) {
            return 0; // the generation is not started yet
        }
        // clamp to 1 us, the first token can land in the same us as t_prompt_last
        return std::max<int64_t>(1, t_gen_last - t_prompt_last);
    }
    double t_gen_ms() const {
        return t_gen_us() / 1000.0;
    }

    // number of decode steps spent on generation
    // the first token is free, it comes from the logits of the last prompt batch
    uint64_t n_gen_steps() const {
        return n_gen > 0 ? n_gen - 1 : 0;
    }

    // other derived metrics
    // note: all of them return 0.0 if the divisor is not known yet
    double t_prompt_per_token_ms() const {
        return n_prompt_processed > 0 ? t_prompt_ms() / n_prompt_processed : 0.0;
    }
    double t_gen_per_token_ms() const {
        return n_gen_steps() > 0 ? t_gen_ms() / n_gen_steps() : 0.0;
    }
    double n_prompt_tps() const {
        const double t_ms = t_prompt_ms();
        return t_ms > 0.0 ? 1e3 / t_ms * n_prompt_processed : 0.0;
    }
    double n_gen_tps() const {
        const double t_ms = t_gen_ms();
        return t_ms > 0.0 ? 1e3 / t_ms * n_gen_steps() : 0.0;
    }

    // false if the slot never started, i.e. the task result carries no stats
    bool is_set() const {
        return t_start > 0;
    }

    json to_json() const;
};

// shared between server_context_impl and server_task_result_*
// unlike server_slot_stats, server_metrics is server-global and cumulative, not tied to a slot
struct server_metrics {
    int64_t t_start = 0;

    struct bucket {
        uint64_t count = 0; // number of tokens
        uint64_t steps = 0; // number of decode steps,
                            // this excludes first generated token (logits from prompt batch)
        uint64_t time  = 0; // in microseconds

        // the rate uses the decode steps, so that "free" tokens do not inflate it
        double n_per_second() const {
            return time > 0 ? (double) steps / (double) time * 1e6 : 0.0;
        }

        void add(uint64_t n, uint64_t n_steps, uint64_t t_us) {
            count += n;
            steps += n_steps;
            time  += t_us;
        }
    };

    // these are reset by reset_bucket(), only the rate is read from them
    bucket prompt_bucket;
    bucket predict_bucket;

    // metrics below are cumulative since the server started
    bucket prompt; // only processed tokens, cached ones are counted separately below
    bucket predict;

    // tokens reused from the cache need no decode, so they only have a count
    uint64_t n_prompt_cached = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_decode     = 0;
    uint64_t n_busy_slots = 0;

    uint64_t n_draft_tokens      = 0; // Total draft tokens generated
    uint64_t n_draft_accepted    = 0; // Draft tokens actually accepted
    uint64_t n_draft_verif_steps = 0; // Total draft token verification steps by the target model
    uint64_t n_jump_forward_tokens = 0; // Total tokens appended by grammar jump-forward
    uint64_t n_kv_preempted      = 0; // In-flight requests preempted because the --kv-unified pool ran out of cells
    uint64_t n_kv_requeued       = 0; // Preempted requests put back in the queue to be recomputed (--kv-preempt-requeue) instead of answered 503
    uint64_t n_kv_preempt_cells  = 0; // KV cells held by preemption victims when they were evicted (computed work that a recompute repeats, --kv-preempt-victim)
    uint64_t n_kv_idle_purged    = 0; // Idle slots whose cached prompt was purged to free cells under --kv-unified
    std::vector<uint64_t> n_accepted_per_pos; // Accepted tokens per draft position

    // --spec-async counters (cumulative since the server started; snapshot of the drafter's own stats, refreshed when a prediction ends)
    bool                           spec_async_on = false;
    common_speculative_async_stats spec_async;

    // latency histograms, step composition and finish reasons (server-latency.h)
    server_latency_metrics lat;

    // --request-log: JSONL line per finished request + request_log_count counter (server-request-log.h). Null when the flag is off;
    // shared so the copy of this struct that /metrics reads sees the same counters.
    std::shared_ptr<server_request_logger> reqlog;

    void init() {
        t_start = ggml_time_us();
    }

    void reset_bucket() {
        prompt_bucket  = {};
        predict_bucket = {};
    }

    void add_prompt(uint64_t n_tokens, uint64_t t_us) {
        prompt       .add(n_tokens, n_tokens, t_us);
        prompt_bucket.add(n_tokens, n_tokens, t_us);
    }

    void add_prompt_cached(uint64_t n_tokens) {
        n_prompt_cached += n_tokens;
    }
};

//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx, size_t n_top);

std::string safe_json_to_str(const json & data);

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens);
std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens);

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data);

std::string format_oai_resp_sse(const json & data);

// format Anthropic-style SSE with event types
std::string format_anthropic_sse(const json & data);

bool is_valid_utf8(const std::string & str);

// /tokenize and /chat_tokenize token list: {id} plus "piece" (with_pieces) and "text"/"start"/"stop"
// (with_offsets; code-point offsets into the concatenated pieces, see server-token-offsets.h)
json tokens_to_json_detailed(const llama_vocab * vocab, const llama_tokens & tokens, bool with_pieces, bool with_offsets);

//
// formatting output responses
// TODO: move these to server-task.cpp
//

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt);

std::string format_rerank_template(
        const std::string & tmpl,
        const std::string & query,
        const std::string & doc);

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
// query_files/doc_files carry decoded media for multimodal (Qwen) inputs; when non-empty the
// prompt is built with get_media_marker() sentinels and spliced via process_mtmd_prompt.
server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc,
        const std::vector<raw_buffer> & query_files,
        const std::vector<raw_buffer> & doc_files,
        const mtmd_helper_init_opt & init_opt);

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct server_pipe {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};

    // 0 = unbounded (default)
    // > 0, write() drops the oldest item once the queue is full
    size_t max_size = 0;

    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    // close_on_stop = true: should_stop means the reader is gone for good, so the writer is told the pipe is broken.
    // close_on_stop = false: should_stop is a per-read deadline and further reads still come, so the pipe stays usable.
    bool read(T & output, const std::function<bool()> & should_stop, bool close_on_stop = true) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop && should_stop()) { // a null should_stop means "never stop"
                if (close_on_stop) {
                    close_read(); // signal broken pipe to writer
                }
                return false; // cancelled / deadline reached
            }
            cv.wait_for(lk, poll_interval);
        }
    }

    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        if (max_size > 0) {
            while (queue.size() >= max_size) {
                queue.pop(); // drop oldest to stay bounded
            }
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

// a child server writes its state commands to stdout and its logs to stderr
enum server_subproc_stream {
    SERVER_SUBPROC_STDOUT,
    SERVER_SUBPROC_STDERR,
    SERVER_SUBPROC_STREAMS,
};

// gives the current stdout to the caller as a stream of its own, and sends everything else written to stdout to stderr
FILE * server_reserve_stdout();

// wrapper around common_subproc to manage a child server process
// mainly used by router mode
struct server_subproc {
    common_subproc sproc;
    std::atomic<bool> stopped{false}; // set by the monitor once the process exited and was reaped
    std::atomic<bool> oom_seen{false}; // the child logged an out-of-memory error (fork: load-OOM retry)

    // fork: the error lines the child logged, newest last, capped to the last 8 KiB at a line boundary (ollama llm/status.go
    // StatusWriter.AppendError), so a failed load can say why instead of only "failed to load". Thread-safe.
    void        append_error(const std::string & line);
    std::string last_error() const;

    // fork: the last few raw stderr lines (any kind) and the text a finished child leaves behind: the matched error lines, else
    // "exit code X; last output: ..." so a segfault or silent exit(3) still says something (server-load-fail.h).
    server_output_ring recent_output;
    std::string        failure_detail(int exit_code) const { return server_load_failure_detail(last_error(), recent_output.text(), exit_code); }

    // fork, Windows: put the child in the router's process-lifetime job object with KILL_ON_JOB_CLOSE, so the kernel
    // terminates it when the router dies for any reason (crash, TerminateProcess) -- the stdin-EOF exit only works for a
    // child that is still reading stdin. Returns false + err on failure (the stdin-EOF path still applies). No-op elsewhere.
    bool assign_to_kill_on_close_job(std::string & err);

    bool is_alive() { return sproc.alive(); }
    void terminate() { sproc.terminate(); }
    int  join() { return sproc.join(); }

    // true if both output pipes of the child are available (call after create())
    bool has_output();

    // true once both output pipes are closed
    bool output_closed() const;

    // non-blocking read from one output pipe
    // returns the number of bytes read, 0 when nothing is available, -1 when the pipe is closed or broken
    int read_output(server_subproc_stream stream, char * buf, size_t len);

    // wait until one of a set of children has output, wake() is called, or a timeout passes
    struct waiter {
        waiter();
        ~waiter();

        // thread-safe; on Windows this is a no-op, wait() returns within 50 ms anyway
        void wake();

        // timeout_ms < 0 waits until data or wake(); ready[i] is set for each child with data on an open pipe (or a broken pipe)
        void wait(const std::vector<server_subproc *> & procs, std::vector<bool> & ready, int64_t timeout_ms);

    private:
#ifndef _WIN32
        intptr_t wake_fd[2] = { -1, -1 }; // POSIX self-pipe
#endif
    };

private:
    intptr_t out_handles[SERVER_SUBPROC_STREAMS] = { -1, -1 }; // fd on POSIX, HANDLE on Windows; taken lazily from sproc
    bool     out_closed [SERVER_SUBPROC_STREAMS] = { false, false };
    mutable std::mutex err_mu;
    std::string        err_tail;
};

//
// router <-> child protocol
//

// what this llama-server process does. a router spawns the other roles as child processes
// and talks to them over their stdin/stdout
enum server_role_type {
    SERVER_ROLE_ROUTER,   // no model: lists the models, spawns instances on demand and proxies requests to them
    SERVER_ROLE_INSTANCE, // serves one model, standalone or as a child of a router
    SERVER_ROLE_DOWNLOAD, // child of a router: download a model, report the progress, exit
};

static inline const char * server_role_type_to_str(server_role_type role) {
    switch (role) {
        case SERVER_ROLE_ROUTER:   return "router";
        case SERVER_ROLE_INSTANCE: return "instance";
        case SERVER_ROLE_DOWNLOAD: return "download";
        default: GGML_ASSERT(false && "invalid server_role_type");
    }
}

// the role a router asks of a child; anything it does not name is an instance
static inline server_role_type server_role_type_from_str(const std::string & str) {
    if (str == "download") {
        return SERVER_ROLE_DOWNLOAD;
    }
    return SERVER_ROLE_INSTANCE;
}

// set by the router in the environment of every child
#define SERVER_ENV_ROUTER_PORT "LLAMA_SERVER_ROUTER_PORT" // its presence makes the process a child
#define SERVER_ENV_CHILD_ROLE  "LLAMA_SERVER_CHILD_ROLE"  // server_role_type_to_str(), an instance when unset

// lines on the child's stdin (router -> child) and stdout (child -> router),
// see server_role::setup() and server_role::notify_to_router()
#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_STATE "cmd_child_to_router:state:" // followed by json string

// address a child binds to: the router may listen on 0.0.0.0, but reaches its children locally
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
#define CHILD_ADDR "127.0.0.1"

// the states a child reports to the router
enum server_state {
    SERVER_STATE_DOWNLOADING,
    SERVER_STATE_LOADING,
    SERVER_STATE_READY,
    SERVER_STATE_SLEEPING,
};

static inline std::string server_state_to_str(server_state state) {
    switch (state) {
        case SERVER_STATE_DOWNLOADING: return "downloading";
        case SERVER_STATE_LOADING:     return "loading";
        case SERVER_STATE_READY:       return "ready";
        case SERVER_STATE_SLEEPING:    return "sleeping";
        default: GGML_ASSERT(false && "invalid server_state");
    }
}

static inline server_state server_state_from_str(const std::string & str) {
    if (str == "downloading") return SERVER_STATE_DOWNLOADING;
    if (str == "loading")     return SERVER_STATE_LOADING;
    if (str == "ready")       return SERVER_STATE_READY;
    if (str == "sleeping")    return SERVER_STATE_SLEEPING;
    GGML_ASSERT(false && "invalid server_state string");
}

// the role of this process and, when it is a child of a router, its side of the protocol
struct server_role {
    // serializes the notify_to_router writes
    std::mutex mtx_stdout;

    // in a child, keeps stdout for the commands to the router, so it is created before anything is written;
    // everything else written to stdout goes to stderr with the logs
    server_role();
    ~server_role();

    // true if the current process was spawned by a router
    bool is_managed_by_router();
    server_role_type get_role();

    // register the shutdown_handler to be called by the router
    // return the monitoring thread (to be joined by the caller)
    std::thread setup(const std::function<void(int)> & shutdown_handler);

    // notify router server for status changes (e.g. loading, downloading, sleeping, etc.)
    // message will be handled by server_models::handle_child_state() on the router side
    void notify_to_router(const std::string & state_name, const json & payload);

private:
    FILE * cmd_out = nullptr; // the stdout the router reads the commands from
};
