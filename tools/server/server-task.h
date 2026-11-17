#pragma once

#include "common.h"
#include "disk-tier.h"
#include "llama.h"

#include <memory>
#include <regex>
#include <string>
#include <unordered_set>
#include <list>
#include <map>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"
#include "server-asr-format.h"
#include "mtmd-helper.h"


enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_DECISION,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_TRANSCRIBE_RT,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_GET,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
    // ported from the pre-merge fork's examples/server/server-task.h (commit 35f0cac75, before
    // merge 279b4bfaf replaced the monolithic server with the split tools/server) --
    // row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G1.
    // SOURCE-READ: examples/server/server-task.h:36-38 @ 35f0cac75 (local worktree copy).
    SERVER_TASK_TYPE_GET_CONTROL_VECTORS,
    SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR,
    SERVER_TASK_TYPE_UNLOAD_CONTROL_VECTOR,
    SERVER_TASK_TYPE_SET_CONTROL_VECTOR,
    // ported from the pre-merge fork's examples/server/server.cpp (commit 35f0cac75) --
    // row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G2.
    // SOURCE-READ: examples/server/server.cpp:1845-1855 @ 35f0cac75 (local worktree copy).
    SERVER_TASK_TYPE_SLOT_PROMPTS,
    // SOURCE-READ: examples/server/server.cpp:1786-1843 @ 35f0cac75 (list_saved_prompts, local worktree
    // copy). Needs the task queue, not the HTTP thread: decoding a saved file's packed server_tokens
    // blob into preview text calls server_tokens::detokenize(llama_context*), and ctx_tgt is private
    // to server_context_impl (main-loop-only) -- see the "GET /list" build failure this replaced.
    SERVER_TASK_TYPE_LIST_SAVED_PROMPTS,
    // POST /v1/decision (--decision-seqs): parallel constrained decisions, tools/parallel-decision.
    // Ported from thecodacus/llama.cpp @ b9244f893 (MIT). Runs on the main loop: it uses ctx_tgt.
    // RENAMED in the 2026-11-04 upstream merge (was SERVER_TASK_TYPE_DECISION): upstream's SERVER_TASK_TYPE_DECISION is
    // its typed decision models, a different feature
    SERVER_TASK_TYPE_PARALLEL_DECISION,
    // POST /v1/score: one prompt, N candidate continuations. The prompt is decoded ONCE, then each candidate is decoded from the
    // prompt boundary, and the per-token log-probabilities of every candidate come back. Main-loop task on an idle slot's
    // sequence (deferred while every slot is busy). Row S69-ENGREV-LOCALAI-SHARED-PREFIX-CANDIDATE-SCORING-2026-10-27-01.
    SERVER_TASK_TYPE_SCORE,
    // POST /v1/logits: one prefill-only pass, the raw LM-head logits back (top-N pairs per row, or the whole vocabulary). Main-loop
    // task on an idle slot's sequence (deferred while every slot is busy); the request body rides in score_request.
    // Row S69-ENGREV-NEXA-V1-LOGITS-PREFILL-ENDPOINT-2026-11-15-01.
    SERVER_TASK_TYPE_LOGITS,
    // LLAMA_HOTSWAP_ENABLED: llama_reload_changed_tensors() on the main loop, between requests (deferred while a slot is
    // processing, or skipped when hotswap_if_idle is set), then every KV cache and cached prompt is dropped. Ports the
    // unbuilt examples/server/server.cpp:803-817 /health hook (ik_llama #2131, Thireus, MIT) and adds the file-signalled
    // mode of docs/development/on-demand-tensor-reload.md. Row S69-ENGINE-REVIEW-1016-SERVER-HOTSWAP-HOOK-01.
    SERVER_TASK_TYPE_HOTSWAP_RELOAD,
    SERVER_TASK_TYPE_TTS,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
    STOP_TYPE_REPETITION, // repetition_detection found the output looping; reported as finish_reason "stop"
    STOP_TYPE_DEADLINE,   // the request's end-to-end budget (t_deadline_ms) ran out while generating; finish_reason "deadline"
    STOP_TYPE_GRAMMAR_ERROR, // the llguidance / xgrammar matcher failed (fail-closed sampler); the slot ends here, finish_reason "grammar_error"
    STOP_TYPE_BUDGET_GROUP,  // the shared token / wall-time pool of the request's budget group ran out (server-budget-group.h); finish_reason "budget_group"
};

enum server_control_action {
    SERVER_CONTROL_ACTION_NONE,
    SERVER_CONTROL_ACTION_REASONING_END,
    SERVER_CONTROL_ACTION_CANCEL,       // release the slot of a live completion found by its completion id
    SERVER_CONTROL_ACTION_AUDIO_APPEND, // push control_audio into the slot's audio stream
    SERVER_CONTROL_ACTION_AUDIO_COMMIT, // flush the audio stream, more audio can follow
    SERVER_CONTROL_ACTION_AUDIO_END,    // flush the audio stream and end it
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    // return_prompt_token_ids: the final (non-stream) reply carries `prompt_token_ids`, the exact token ids the model was given after chat
    // templating and tokenisation (media positions are null), so a client that continues or scores the conversation can send the ids back
    // instead of re-rendering text. SOURCE-READ: sglang/python/sglang/srt/entrypoints/openai/protocol.py:900 (chat request field), :1553,
    // serving_chat.py:1227-1230 (refused with stream), :2435 (returned with the response). Apache-2.0, no code copied.
    bool return_prompt_token_ids = false;

    // skip_special_tokens (default true = unchanged): false renders special/control tokens as their piece text in this request's output
    // even when the server runs without --special. SOURCE-READ: sglang/python/sglang/srt/sampling/sampling_params.py:146 (field, default
    // True), managers/detokenizer_manager.py:246-291 (decode grouped by skip_special_tokens). Apache-2.0, no code copied.
    bool skip_special_tokens = true;

    // token healing (row S69-tabbyapi-PICK-TOKEN-HEALING-2026-09-28-01): the prompt's last token is removed and the first sampled
    // token is constrained to start with its text, so a prompt cut mid-word can continue across the tokenizer boundary.
    // token_healing is the request flag; heal_prefix_token is set by the request handler ONLY when healing was applied.
    bool        token_healing     = false;

    // set by the request handler only when the request carries the server's startup pin token (the --warm-prefix-file warm-up):
    // the slot's prompt state is saved to the host prompt cache and pinned when the task ends
    bool        cache_pin         = false;
    llama_token heal_prefix_token = LLAMA_TOKEN_NULL;

    // request log (server-request-log.h): the ingested W3C trace id (empty when the request had no valid traceparent) and the id the log line
    // is keyed by (X-Request-Id, else the trace id, else the completion id). Set by the HTTP handler; empty for internal tasks.
    std::string trace_id;
    std::string request_id;

    // budget group (server-budget-group.h): the id of the recursion tree this request belongs to (X-Budget-Group header or body field
    // budget_group; empty = none). The pool's limits live in the server's table, fixed by the first request of the group.
    std::string budget_group;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // if positive, the prompt phase (slot start to prompt done) may take this long, else the request gets an error
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    // t_deadline_ms: ONE budget for the whole request, counted from its arrival: queue wait + prompt + generation (0 = none; the server
    // default is --request-deadline-ms). Still queued or reading its prompt when it runs out -> error 503; generating -> stopped, finish_reason
    // "deadline". SOURCE-READ: tf-serving@4438d84 tensorflow_serving/batching/batching_session.cc:746-783 (Apache-2.0, no code copied).
    int64_t t_deadline_ms = 0;

    std::map<int, float> lora; // mapping adapter ID -> scale

    // --lora-pool-dir: name of an adapter in the on-demand pool (<dir>/<name>.gguf) this request runs with, and its scale.
    // Empty = the request names none (the field is accepted but ignored when the server has no pool).
    std::string lora_adapter;
    float       lora_adapter_scale = 1.0f;

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    // /v1/audio/transcriptions response only (set by the --asr-quality retry loop): add the decode statistics of the attempt
    // (mean token logprob, tail entropy, token count) to the reply
    bool asr_stats = false;

    // json_schema_strict request field (server-schema.cpp; read there from the request, default --json-schema-strict)
    bool json_schema_strict = false;

    // banned strings (ik_llama.cpp): hold back output, and when it contains a banned string or regex
    // match, rewind to where the match starts, ban that token at that position and resample
    std::vector<std::string> ban_phrases;      // lower-cased, matched case-insensitively
    std::vector<std::string> ban_regex;        // ECMAScript
    std::vector<std::string> ban_regex_ci;     // ECMAScript, case-insensitive
    int32_t banned_n         = 1;              // tokens banned at the rewind position (-1 = whole match)
    float   banned_bias      = -999.0f;        // logit bias applied to a banned token at its position
    int32_t rewind_count_max = -1;             // -1 = max(20, 2 * number of patterns), 0 = unlimited
    int32_t banbuffer_size   = 0;              // tokens held back; 0 = longest pattern length + 1
    bool    saturate_predict = false;          // rewound tokens do not count against n_predict

    // stop controls (vLLM SamplingParams.min_tokens / stop_token_ids; row S69-VLLM-PICK-REQUEST-STOP-CONTROLS-2026-09-29-01)
    // min_tokens: the end-of-generation tokens are masked until this many tokens have been generated. Not applied to a
    //             request that carries a grammar (the grammar may allow nothing else; masking would leave no candidate).
    // stop_token_ids: generation stops when one of these token ids is sampled; the token is not part of the output.
    int32_t                  min_tokens = 0;
    std::vector<llama_token> stop_token_ids;

    // grammar jump-forward (row S69-MLC-LLM-PICK-GRAMMAR-JUMP-FORWARD-2026-10-15-01): max tokens appended per step when a user grammar forces
    // the next text; -1 = the server's --grammar-jump-forward, 0 = off for this request
    int32_t                  grammar_jump_forward = -1;

    // prompt scoring (vLLM SamplingParams.prompt_logprobs; row S69-NEXA-PICK-SERVER-PROMPT-LOGPROB-SCORING-2026-09-29-01):
    // the response carries, for every prompt token after the first, the log-probability the model gave it given the
    // tokens before it, and whether it was the model's own top choice. The prompt is evaluated without any cache reuse
    // (a reused prefix was not scored in this request), so this costs a full prefill.
    bool prompt_logprobs = false;

    // logprob_start_len: with prompt_logprobs, score only the prompt tokens from this index on (entries before it come back null). The prefix that
    // is not scored may then be served from the slot / prompt cache / radix cache, so scoring N continuations of one long prompt (a multiple-choice
    // "select", a log-likelihood benchmark) re-evaluates the shared prefix once, not N times. 0 (default) = score the whole prompt, which still
    // runs with no cache reuse. At most logprob_start_len - 1 tokens are reused: the logits that score token `start` come from evaluating token
    // start - 1.
    // SOURCE-READ: sglang/python/sglang/srt/managers/io_struct.py:232 (request field), schedule_batch.py:1665-1675 (the matched prefix is capped at
    // logprob_start_len), :2272-2288 (scoring begins at max(logprob_start_len, prefix_len)), lang/backend/runtime_endpoint.py:246-313 (select():
    // one request primes the shared prefix with max_new_tokens 0, then one batch scores prefix+choice from logprob_start_len). Apache-2.0, no code copied.
    int32_t logprob_start_len = 0;

    // include_stop_str_in_output: a generation that ends on a "stop" string keeps that string at the end of the output
    // (default: the string is cut off). Only text is affected; stop_token_ids and EOS never appear in the output.
    bool include_stop_str_in_output = false;

    // stop_regex: generation ends when one of these patterns matches the decoded text (earliest match wins, also against "stop"
    // strings); the matched text is cut like a stop string and kept with include_stop_str_in_output. A regex has no partial-match
    // mode, so matching looks only at the last STOP_REGEX_WINDOW_BYTES bytes and, while streaming, that many bytes are held back
    // from the client until the generation ends or they are old enough. Patterns are ECMAScript, at most STOP_REGEX_MAX_PATTERN
    // characters, and must not match the empty string.
    // SOURCE-READ: engines/sglang/python/sglang/srt/sampling/sampling_params.py:339-351 (MAX_STOP_REGEX_LEN 256). Apache-2.0, no code copied.
    static constexpr size_t STOP_REGEX_MAX_PATTERN     = 256;
    static constexpr size_t STOP_REGEX_WINDOW_BYTES    = 512;
    std::vector<std::string> stop_regex;
    std::vector<std::regex>  stop_regex_compiled;

    // logprob_token_ids: for every generated position also return the log-probability of each listed token id, taken from the
    // raw (pre-sampling) distribution. At most 128 ids, all inside the vocabulary.
    // SOURCE-READ: engines/vllm/vllm/sampling_params.py:297-309 (Apache-2.0, no code copied).
    // Rows S69-VLLM-PICK-LOGPROB-TOKEN-IDS-2026-09-29-01 and S69-aphrodite-PICK-LOGPROB-FOR-SPECIFIC-TOKEN-IDS-2026-09-29-01.
    std::vector<llama_token> logprob_token_ids;

    // tool_call_fix: ids of every vocab token whose text contains ']'. They are masked only for the first 3 generated tokens and
    // only while the output so far holds at most one '[' (so a reply cannot open as an empty array "[]"); after that they are free.
    // SOURCE-READ: engines/koboldcpp/gpttype_adapter.cpp:159 (list), :5783-5794 (filled), :7023-7030 (applied). AGPL-3.0: behaviour
    // only, no code copied. Row S69-KOBOLDCPP-PICK-TOOL-CALL-FIX-FIRST-3-TOKENS-ONLY-2026-09-29-01.
    std::vector<llama_token> tool_call_fix_ids;

    // priority: when every slot is busy, deferred requests are started lowest value first, then in arrival order (default 0; an
    // interactive request sent with a lower value is not stuck behind a batch job that arrived earlier). Does not interrupt a request
    // that is already running.
    // SOURCE-READ: engines/vllm/vllm/v1/core/sched/request_queue.py:100-178 (heap on (priority, arrival_time), lower first),
    // engines/vllm/vllm/entrypoints/openai/completion/protocol.py:134-140 (request field), engines/sglang/python/sglang/srt/managers/
    // schedule_policy.py:561-572 (sort by priority then FCFS). Apache-2.0, no code copied.
    int32_t priority = 0;

    // cache_salt / extra_key: caller-chosen cache namespace. Two requests with different values never share KV: not through the slot's
    // own leftover cache, the host prompt cache, the disk tier or the radix cache (all of them already key on the configuration
    // fingerprint, which now includes these two strings). Empty (the default) = the shared namespace, bit-identical to before.
    // SOURCE-READ: sglang/python/sglang/srt/mem_cache/radix_cache.py:62-77 (RadixKey carries extra_key + cache_salt), :245-247 (both
    // go into the child-dict key, so differing values never share a node), entrypoints/openai/protocol.py:410-412,970-972,1841-1846
    // (request fields; an empty salt counts as none, radix_cache.py:77). Apache-2.0, no code copied.
    // Also vLLM's cache_salt, same field name: engines/vllm/vllm/v1/core/kv_cache_utils.py:630-640 (salt hashed into the first
    // block's extra keys, chained into every later block hash). Apache-2.0, no code copied.
    std::string cache_salt;
    std::string extra_key;

    // max_queue_wait_ms: a request that is still waiting for a slot this long after it was first deferred is answered 503 and dropped
    // (0 = wait as long as it takes; the server default is --max-queue-wait-ms). Never stops a request that already has a slot.
    int32_t max_queue_wait_ms = 0;

    // stream_interval: a streaming response sends a chunk only once this many generated tokens are unsent (1 = every token). The first
    // token and the last chunk are always sent. Ignored with n_probs / return_tokens / stop_regex (they need one chunk per token). The
    // server default is --stream-interval. SOURCE-READ: sglang output_streamer.py:413-445; Apache-2.0, no code copied.
    int32_t stream_interval = 1;

    // repetition_detection: the generation ends when the tail of its output is one pattern of rep_max_pattern_size tokens or fewer
    // (at least rep_min_pattern_size) repeated rep_min_count times back to back. Only output tokens count, and min_tokens is honoured.
    // rep_max_pattern_size == 0 means the check is off. The finished response reports stop_type "repetition".
    // SOURCE-READ: engines/vllm/vllm/v1/core/sched/utils.py:10-62 (the tail test) and :126-138 (order: after min_tokens),
    // engines/vllm/vllm/sampling_params.py:165-191 (parameter rules). Apache-2.0, no code copied.
    // echo (OpenAI /v1/completions): the returned text starts with the prompt. With logprobs also requested the prompt tokens are scored
    // (prompt_logprobs is switched on for the request) and choices[].logprobs carries the legacy arrays tokens / token_logprobs /
    // top_logprobs / text_offset with one entry per prompt token followed by the generated ones - what lm-eval's completions client sends
    // (payload: prompt, max_tokens 1, logprobs 1, echo true, temperature 0) and parses.
    // SOURCE-READ: portable/envs/lm_eval/Lib/site-packages/lm_eval/models/openai_completions.py:61-122 (MIT, contract only).
    bool echo = false;

    int32_t rep_max_pattern_size = 0;
    int32_t rep_min_pattern_size = 1;
    int32_t rep_min_count        = 0;
    // repetition_detection span_min_len / span_min_count: the tail span (span_min_len tokens, doubling) appears span_min_count times
    // ANYWHERE in the recent output, not only back to back. span_min_count == 0 means this half is off. See server-repetition.h.
    int32_t rep_span_min_len     = 0;
    int32_t rep_span_min_count   = 0;
    // repetition_detection min_duplicate_tokens: the back-to-back pattern must also span at least this many tokens (length x copies),
    // so a short run such as "=====" is not a loop. 0 = no floor (the old behaviour). rtp-llm min_duplicate_tokens.
    int32_t rep_min_dup_tokens   = 0;
    // repetition_detection token_guard: the model-free periodic-tail / n-gram / low-entropy detectors of server-repetition.h
    // (BeeLlama loop guard contract), judged on accepted tokens every 64 tokens. false = off.
    bool    rep_token_guard      = false;
    // repetition_detection tool_call_markers {begin, end, min_count}: token ids that open and close a tool call. The generation ends
    // when the call it just completed is byte-identical to the rep_tool_min_count - 1 calls right before it across [prompt, output].
    // rep_tool_min_count == 0 means this half is off. See server-toolloop.h.
    int32_t rep_tool_begin       = -1;
    int32_t rep_tool_end         = -1;
    int32_t rep_tool_min_count   = 0;
    // repetition_detection coverage_threshold / coverage_window / coverage_every: the word-level coverage detector of
    // server-loop-recovery.h, run on the output text while the reasoning block is open. 0 = off unless
    // reasoning_loop_recovery is stop or recover, which turn it on at 0.25.
    float   rep_cov_threshold    = 0.0f;
    int32_t rep_cov_window       = 2000;
    int32_t rep_cov_every        = 512;

    // reasoning_loop_recovery (server_loop_recovery::mode): 0 off, 1 stop, 2 recover. Default --reasoning-loop-recovery.
    int32_t     rlr_mode = 0;
    std::string rlr_high_text;              // empty = --reasoning-effort-high-text, else the Qwen3.8 sentence
    std::string rlr_low_text;               // empty = --reasoning-effort-low-text, else the Qwen3.8 sentence
    // set by the chat-completions parser: the first message is not a system/developer message and a user message contains the
    // high sentence, so the first turn may be the user's text and the splice is not allowed (recover falls back to stop)
    bool        rlr_user_has_sentence = false;

    bool has_bans() const { return !ban_phrases.empty() || !ban_regex.empty() || !ban_regex_ci.empty(); }

    // allowlist sampler (fork, pre-merge examples/server): the rule sets and keywords are server-wide
    // (--allowlist-unicode-rule / --allowlist-keyword); these two a request may override
    std::vector<std::string> allow_pieces;   // allowlist_piece_array
    size_t                   allow_kw_delay = 0; // allowlist_keyword_delay

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    json               responses_tool_map; // /v1/responses: namespace / custom tool names to report calls under
    bool               responses_reasoning_blob = false; // /v1/responses include=["reasoning.encrypted_content"]
    bool               responses_reasoning_summary = false; // /v1/responses reasoning.summary requested

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    server_control_action control_action = SERVER_CONTROL_ACTION_NONE;
    std::string           control_cmpl_id;
    std::vector<float>    control_audio; // PCM F32 samples for SERVER_CONTROL_ACTION_AUDIO_APPEND

    // reported in generation_settings, parsing itself is owned by the chat session
    common_chat_format      chat_format      = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;
    // the fork's per-request parse options (server-schema.cpp fills them); applied onto the chat session by create_state()
    common_chat_parser_params chat_parser_params;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_session chat_session; // owns all parsing for this generation
    common_chat_msg chat_msg;
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta
    bool oai_resp_fc_custom = false; // the call being streamed is a custom tool: its input goes out whole, at the end
    json oai_resp_tool_map;          // flat tool name -> {namespace, name, custom}; see oai_resp_tool_call_item()
    bool oai_resp_reasoning_blob = false; // emit encrypted_content (server-reasoning-blob.h)
    bool oai_resp_reasoning_summary = false; // emit reasoning_summary_* events

    // /v1/audio/transcriptions stream=true: holds the Qwen3-ASR "language X<asr_text>" tag out of the deltas
    asrfmt::language_tag_stream asr_tag;

    task_result_state(common_chat_session session = {});

    // How a tool call is reported on /v1/responses: a call to a tool the client declared inside a `namespace` carries that
    // namespace and its own name; a call to a `custom` tool is a custom_tool_call item holding the raw input text. `tool_map`
    // is what server_chat_convert_responses_to_chatcmpl recorded (empty: plain function tools only).
    static json oai_resp_tool_call_item(const json & tool_map, const std::string & id, const std::string & flat_name,
                                        const std::string & arguments, const char * status);
    static bool oai_resp_tool_is_custom(const json & tool_map, const std::string & flat_name);
    // An error AFTER the stream started: Codex reads a bare {"error":...} chunk as a dropped connection and retries the whole
    // turn. This is the event list (one response.failed with the stream's response id) for format_oai_resp_sse().
    static json oai_resp_failed_events(const std::string & resp_id, const json & error);

    // OpenAI Responses stream, silent stretch (a long prompt being read, a long think): the events to send instead of a
    // bare SSE comment. Codex's idle timeout counts events, not comments. Emits response.created first when no real event
    // has gone out yet (and marks it sent so the first token does not repeat it), then response.in_progress.
    json oai_resp_idle_events();

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // ggml_time_ms() when the task was first deferred for lack of a slot (0 = never deferred); set by server_queue::defer
    int64_t t_first_deferred_ms = 0;

    // --sched-policy hrrn: the queue's prefill-token counter (server_queue::add_prefill_tokens) when the task was first deferred; set by
    // server_queue::defer. The counter minus this = prefill tokens the server processed while the task waited (its age, in tokens).
    uint64_t sched_arrival_tokens = 0;

    // ggml_time_us() when the task object was created, i.e. when the request reached the server (0 = unknown, a
    // default-constructed task). The queue and TTFT histograms measure from here.
    int64_t t_submit_us = 0;

    // ggml_time_us() at which params.t_deadline_ms runs out, counted from the arrival of the HTTP request (0 = no deadline)
    int64_t t_deadline_us = 0;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;
    // if set on a parent, the children have their own prompt and only share its first n_tokens_shared tokens
    int32_t n_tokens_shared = 0;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;
    // used by SERVER_TASK_TYPE_METRICS: also collect the per-device memory rows (GET /memory, memory_*_bytes gauges)
    bool metrics_memory = false;

    // used by SERVER_TASK_TYPE_PARALLEL_DECISION: the request body
    json decision_request;

    // used by SERVER_TASK_TYPE_SCORE: the request body
    json score_request;

    // used by SERVER_TASK_TYPE_HOTSWAP_RELOAD: true = answer "skipped" at once when a slot is processing (the /health hook,
    // which must not block); false = defer until no slot is processing (the control-file driver, which waits)
    bool hotswap_if_idle = false;

    // used by SERVER_TASK_TYPE_DECISION
    // where to read the model output of each option, exactly one of the two lists is used
    struct decision {
        std::vector<llama_token> labels;       // logits of these tokens, at the last prompt token
        std::vector<int32_t>     label_groups; // if set, number of labels per output, the output is their max
        std::vector<int32_t>     markers;      // embeddings[column] at these prompt positions
        int32_t                  column = 0;
        // if set, embeddings is [q | k], and the output is instead the scaled dot product of q[pointer] and k[marker]
        int32_t                  pointer = -1;

        // --decision-budget auto: a state that does not fit the context is read in this many overlapping windows, each is its own task
        int32_t                  n_windows = 1;
        // --decision-budget model|N: tokens of the question and options that were cut to the head budget
        int32_t                  n_cut = 0;

        // first prompt position that is read, -1 if none
        int32_t pos_first() const {
            int32_t pos = pointer;
            for (const int32_t marker : markers) {
                pos = pos < 0 ? marker : std::min(pos, marker);
            }
            for (size_t i = 0; i < order.size(); i++) {
                if (order[i] != 0) {
                    return pos < 0 ? (int32_t) i : std::min(pos, (int32_t) i);
                }
            }
            return pos;
        }

        // for a joint head: one value per prompt token, see llama_batch_ext_set_decision_order()
        // the scores are the first n_scores rows of the embeddings
        std::vector<int32_t> order;
        int32_t              n_scores = 0;
    };
    decision decision;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    // used by SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR. SOURCE-READ: examples/server/server-context.cpp
    // :3128-3131 @ 35f0cac75 (local worktree copy) -- layer_end == -1 means "default to
    // llama_n_layer(model)", resolved in process_single_task where the model is in scope.
    struct control_vector_load_params {
        std::string path;
        float       scale       = 1.0f;
        int32_t     layer_start = 1;
        int32_t     layer_end   = -1;
    };
    control_vector_load_params control_vector_load;

    // used by SERVER_TASK_TYPE_UNLOAD_CONTROL_VECTOR
    int control_vector_unload_id = -1;

    // used by SERVER_TASK_TYPE_SET_CONTROL_VECTOR (the /control-vectors/apply route).
    // SOURCE-READ: examples/server/server.cpp:1743-1784 @ c3d1c7480 (this worktree's own base
    // commit -- the handler itself, still on disk, unbuilt; the task-processing side this task
    // posts to is ported separately, cited at its own site in server-context.cpp).
    struct control_vector_scale_update {
        int     id = -1;
        float   scale = 0.0f;
        bool    has_layer_start = false;
        int32_t layer_start = 0;
        bool    has_layer_end = false;
        int32_t layer_end = 0;
    };
    std::vector<control_vector_scale_update> control_vector_apply;

    // used by SERVER_TASK_TYPE_TTS
    mtmd_helper::gen_audio::inp tts_inp;

    server_task() = default;

    server_task(server_task_type type) : type(type) { t_submit_us = ggml_time_us(); }

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            case SERVER_TASK_TYPE_DECISION:
                return !decision.markers.empty() || !decision.order.empty();
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
                return true;
            case SERVER_TASK_TYPE_DECISION:
                return !decision.labels.empty();
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TTS:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void apply_chat_session(const common_chat_session & session) {
        if (!session.has_template()) {
            return;
        }

        session.apply_sampling(params.sampling);
        params.chat_format = session.format();
        params.antiprompt.insert(params.antiprompt.end(), session.additional_stops().begin(), session.additional_stops().end());
        params.message_spans = tokens.find_message_spans(session.message_delimiters());
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.t_deadline_us = t_deadline_us; // the same end-to-end budget as the parent
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // --kv-preempt-requeue (row S69-VLLM-W2-RECOMPUTE-PREEMPTION-INSTEAD-OF-503-2026-10-27-01): how many times this request was
    // preempted on KV exhaustion and put back in the queue to be recomputed from its prompt
    int32_t n_preempt_requeues = 0;

    // --auto-truncate (row S69-ENGREV-SGLANG-AUTO-TRUNCATE-2026-10-27-01): prompt tokens cut away to fit the slot context (0 = none)
    int32_t n_prompt_truncated = 0;

    // a fresh copy of a plain completion task (no children, not a child, not CLI) for the queue, with the requeue count raised. The
    // same id / index, so the HTTP thread that waits on this id receives the answer of the recomputation. Only the fields a completion
    // uses are carried over, the rest keeps its default.
    server_task requeue_copy() const {
        server_task copy;

        copy.id            = id;
        copy.index         = index;
        copy.t_submit_us   = t_submit_us;
        copy.t_deadline_us = t_deadline_us;
        copy.params        = params;
        copy.tokens        = tokens.clone();
        copy.type          = type;
        copy.id_slot       = -1; // the slot it had is the one just freed; let the queue pick any
        copy.n_preempt_requeues = n_preempt_requeues + 1;
        copy.n_prompt_truncated = n_prompt_truncated;

        return copy;
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    // built on the chat session (upstream's owner of all parsing) with the request's parse options applied
    task_result_state create_state(const common_chat_session & session = {}) const {
        task_result_state state(session);
        state.chat_session.apply_parser_options(params.chat_parser_params);
        state.oai_resp_tool_map = params.responses_tool_map;
        state.oai_resp_reasoning_blob = params.responses_reasoning_blob;
        state.oai_resp_reasoning_summary = params.responses_reasoning_summary;
        return state;
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }

    // embedding and rerank tasks always process their whole prompt, their state is never reused
    bool is_stateless() const {
        return type == SERVER_TASK_TYPE_EMBEDDING || type == SERVER_TASK_TYPE_RERANK;
    }
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    // A finished completion the server answers with a retryable error instead (an empty reply after thinking): nullptr = no.
    // Checked once, right after update(); only server_task_result_cmpl_final overrides it.
    virtual std::unique_ptr<server_task_result> to_retryable_error() {
        return nullptr;
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;
    // raw-distribution log-probability of each requested logprob_token_ids entry (empty unless asked)
    std::vector<std::pair<llama_token, float>> id_logprobs;
    // FIRST generated position only (-1 elsewhere): summed probability of every end-of-generation token in the raw
    // distribution there. /v1/audio/transcriptions reports it as verbose_json's no_speech_prob proxy: an LLM audio
    // decoder has no no-speech head, but a model that ends the reply at once is saying it heard nothing to write.
    float eog_prob = -1.0f;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    common_chat_input content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    server_slot_stats stats;
    std::string prompt;
    std::vector<llama_token> prompt_token_ids; // filled only when the request set return_prompt_token_ids
    bool return_prompt_token_ids = false;

    bool truncated;
    int32_t n_decoded;
    int32_t n_reasoning_tokens = 0; // generated inside the thinking block (server-reasoning-count.h)
    int32_t n_preempt_requeues = 0; // times this request was preempted on KV exhaustion and recomputed (--kv-preempt-requeue)
    int32_t n_prompt_truncated = 0; // REPORT ONLY, opt-in --auto-truncate: prompt tokens cut to fit the slot ctx (0 = none)
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    std::string grammar_error; // STOP_TYPE_GRAMMAR_ERROR: the first matcher error, reported next to finish_reason "grammar_error"
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    bool asr_stats = false;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    // prompt_logprobs (request field): per prompt token, the log-probability it had given the tokens before it, and
    // whether it was the model's top choice. NaN = no entry (the first token). Empty when not requested.
    std::vector<float>   prompt_logprobs;
    std::vector<uint8_t> prompt_greedy;
    // echo only: per prompt token its text, and for each position the model's top token id and that token's log-probability
    std::vector<std::string> prompt_pieces;
    std::vector<int32_t>     prompt_arg;
    std::vector<std::string> prompt_arg_pieces;
    std::vector<float>       prompt_arg_lp;
    bool                     echo = false;
    // the prompt was scored token by token and its pieces are present: the legacy logprobs arrays can be built
    bool echo_prompt_scored() const {
        return echo && !prompt_logprobs.empty() && prompt_pieces.size() == prompt_logprobs.size() &&
               prompt_arg.size() == prompt_logprobs.size() && prompt_arg_pieces.size() == prompt_logprobs.size();
    }
    json prompt_logprobs_json() const;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        asr_flush; // TASK_RESPONSE_TYPE_OAI_ASR stream: text the delta filter still held at the end
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    json        oai_resp_tool_map;
    bool        oai_resp_reasoning_blob = false;
    bool        oai_resp_reasoning_summary = false;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        if (res_type == TASK_RESPONSE_TYPE_OAI_ASR) {
            asr_flush = state.asr_tag.flush(); // undecided pieces (the text never became speech or a tag) go out as a last delta
        }
        // a call stranded in unclosed reasoning is delivered only when the model stopped on its own; a turn cut by max_tokens
        // or the context may end in the middle of a command
        if (stop == STOP_TYPE_LIMIT || stop == STOP_TYPE_NONE || stop == STOP_TYPE_DEADLINE || stop == STOP_TYPE_GRAMMAR_ERROR || stop == STOP_TYPE_BUDGET_GROUP) {
            state.chat_session.parser_options().rescue_reasoning_calls = false;
            state.chat_session.parser_options().repair_malformed_calls = false; // a cut call is withheld, not rebuilt
        }
        // withhold_incomplete_tool_calls: a call the text ended inside is withheld when the turn was cut (max_tokens, the
        // context, a deadline, the repetition stop). A turn the model ended itself (EOS, a stop word) keeps the lenient parse
        // that upstream has (common/chat.cpp parses the final message with COMMON_PEG_PARSE_FLAG_LENIENT): a format may
        // close its call with the end-of-generation token or a stop string, which never reaches the text.
        if (stop == STOP_TYPE_EOS || stop == STOP_TYPE_WORD) {
            state.chat_session.parser_options().drop_incomplete_tool_calls = false;
        }
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);
        // a turn the model ended on its own inside or right after its reasoning, with nothing for the client (fastllm-1027 C8 (d))
        empty_thinking_reply = (stop == STOP_TYPE_EOS || stop == STOP_TYPE_WORD) &&
                               common_chat_is_empty_thinking_reply(oaicompat_msg, state.chat_session.parser_options());

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
        oai_resp_tool_map = state.oai_resp_tool_map;
        oai_resp_reasoning_blob = state.oai_resp_reasoning_blob;
        oai_resp_reasoning_summary = state.oai_resp_reasoning_summary;
    }

    bool empty_thinking_reply = false; // set by update(), see to_retryable_error

    virtual std::unique_ptr<server_task_result> to_retryable_error() override;

    json to_json_by_type();
    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    common_chat_input content;
    llama_tokens      tokens;

    int32_t n_decoded;
    int32_t n_reasoning_tokens = 0;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    bool audio_commit_drained = false; // realtime: non-final commit has fully decoded
    completion_token_output prob_output;
    server_slot_stats stats;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;
    bool oai_resp_fc_custom = false;
    json oai_resp_tool_map;
    bool oai_resp_reasoning_summary = false;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    // TASK_RESPONSE_TYPE_OAI_ASR: this chunk's text with the language tag held back (task_result_state::asr_tag)
    std::string asr_delta;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_tts : server_task_result {
    std::string audio; // raw bytes for this chunk (WAV or PCM, per request's out_type)
    int32_t     sample_rate = 0;
    bool        final = false; // true for the last chunk of a request

    virtual bool is_stop() override { return final; }

    virtual json to_json() override;
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;
    // all n_cls_out raw outputs of the sequence; /rerank ignores them (it keeps score = [0]), /classify returns them all
    // (row S69-ENGREV-LORAX-CLASSIFY-ROUTE-MULTILABEL-2026-10-27-01)
    std::vector<float> scores;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_decision : server_task_result {
    std::vector<float> scores; // one raw model output per option

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

// used by /metrics API
// snapshot of the NVMe prompt-cache tier (--cache-disk-dir) for /metrics; `enabled` false = the lines are not emitted
struct server_prompt_cache_disk_stats {
    bool     enabled = false;
    uint64_t n_spilled = 0, n_restored = 0, n_restore_failed = 0, n_dropped_small = 0, n_dropped_refused = 0, n_disk_evicted = 0;
    uint64_t bytes_written = 0, bytes_read = 0, n_refused_cap = 0, n_refused_budget = 0, n_io_failed = 0, n_checksum_failed = 0;
    uint64_t used_bytes = 0, limit_bytes = 0, n_entries = 0;
    // writes the tier's free-space guard refused: the volume was (nearly) full (common_disk_tier_stats::n_refused_space,
    // row S69-FORK-DISK-TIER-REFUSED-SPACE-SERVER-METRICS-2026-10-19-01)
    uint64_t n_refused_space = 0;
    // context checkpoints of spilled entries (row S69-TIERED-CKPT-BLOBS-INTO-DISK-TIER-2026-10-05-01)
    uint64_t n_ckpt_spilled = 0, n_ckpt_refused = 0, n_ckpt_restored = 0, n_ckpt_restore_failed = 0, n_ckpt_adopted = 0;
    uint64_t n_ckpt_slot_spilled = 0, n_ckpt_slot_refused = 0;
    uint64_t n_dropped_slow = 0, n_ckpt_dropped_slow = 0, n_restore_skipped = 0;
    double   prefill_tps = 0, restore_Bps = 0, min_speedup = 0;
    uint64_t bytes_written_ckpt = 0, bytes_read_ckpt = 0;
    uint64_t ram_bytes = 0, ram_ckpt_bytes = 0; // what the RAM cache holds now, and how much of it is checkpoints
};

// the fork's parallel /v1/decision (thecodacus b9244f893): the whole answer as one JSON body
struct server_task_result_parallel_decision : server_task_result {
    json data;

    virtual json to_json() override {
        return data;
    }
};

struct server_task_result_hotswap : server_task_result {
    bool   reloaded               = false; // at least one tensor was re-read from disk
    bool   skipped_busy           = false; // hotswap_if_idle and a slot was processing: nothing was tried
    int    n_slots_cleared        = 0;
    size_t n_prompt_cache_dropped = 0;
    bool   radix_cleared          = false;

    virtual json to_json() override {
        return json {
            { "reloaded",               reloaded },
            { "skipped_busy",           skipped_busy },
            { "n_slots_cleared",        n_slots_cleared },
            { "n_prompt_cache_dropped", n_prompt_cache_dropped },
            { "radix_cleared",          radix_cleared },
        };
    }
};

// the slot-derived gauges of /metrics; server_routes also snapshots them when the server enters sleep, since the slots are
// rebuilt on wake and /metrics during sleep must report what they held right before (unit/test_sleep.py)
struct server_slot_gauges {
    int      n_processing_slots   = 0;
    uint64_t n_kv_tokens_active   = 0;
    uint64_t n_kv_tokens_held     = 0;
    uint64_t n_kv_tokens_capacity = 0;
};

struct server_task_result_metrics : server_task_result {
    // these are immediate stats, not accumulated (server_metrics is cumulative)
    int n_processing_slots = 0;
    int n_tasks_deferred = 0;

    // KV occupancy in tokens: held by processing slots / held by all slots (idle slots keep their prompt for reuse) / capacity
    uint64_t n_kv_tokens_active   = 0;
    uint64_t n_kv_tokens_held     = 0;
    uint64_t n_kv_tokens_capacity = 0;

    server_metrics metrics;

    server_prompt_cache_disk_stats disk;

    // --mmproj-embd-cache: vision-encoder output cache (mtmd_embd_cache_info); has_mmproj = false on a server without a vision model
    bool     has_mmproj = false;
    uint64_t mmproj_encodes = 0, mmproj_hits = 0, mmproj_misses = 0, mmproj_evictions = 0, mmproj_not_stored = 0;
    uint64_t mmproj_bytes = 0, mmproj_entries = 0, mmproj_budget = 0;

    // --radix-cache counters (server_radix_cache::stats_t); has_radix = false when the tree is off, so such a server exports exactly
    // the metrics it always did
    bool     has_radix = false;
    uint64_t radix_hits = 0, radix_misses = 0, radix_hit_tokens = 0, radix_evictions = 0, radix_nodes = 0;

    // per-device memory rows (ik_llama.cpp 55813a5d7): a JSON array of {name, model, context, compute[, mmproj][, total, free]},
    // null when the task did not ask for it; n_layer is the target model's layer count (GET /memory prints it)
    json memory = nullptr;
    int  n_layer = 0;

    virtual json to_json() override;

    struct metric_item {
        std::string name;
        std::string description;
        double value; // prometheus values are always float64
    };
    std::string to_metrics();
};

// used by /slots API
struct server_task_result_slots : server_task_result {
    int n_idle_slots = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

// used by SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR / UNLOAD_CONTROL_VECTOR / SET_CONTROL_VECTOR.
// SOURCE-READ: examples/server/server.cpp:1709-1784 @ c3d1c7480 (old handlers' {success,error[,id]}
// shape, ported verbatim rather than reusing server_task_result_control's {success,message} so any
// pre-existing client of this historical API keeps the field names it expects).
struct server_task_result_control_vector : server_task_result {
    bool        success = false;
    std::string error;
    bool        has_cv_id = false;
    int         cv_id     = -1; // NOT server_task_result::id -- that one routes the result back to
                                 // the waiting HTTP thread and must stay == task.id (own name to
                                 // avoid shadowing it, a bug caught while writing this).

    virtual json to_json() override {
        json j = { { "success", success } };
        if (!success && !error.empty()) {
            j["error"] = error;
        }
        if (has_cv_id) {
            j["id"] = cv_id;
        }
        return j;
    }
};

// used by SERVER_TASK_TYPE_GET_CONTROL_VECTORS (GET /control-vectors). Read on the main loop
// thread (mirrors SERVER_TASK_TYPE_GET_LORA), never off a raw HTTP-thread read of server state.
struct server_task_result_control_vectors_list : server_task_result {
    json data = json::array();

    virtual json to_json() override {
        return data;
    }
};

// A checkpoint's CONTENT never changes after it is created; spill / cram / reload only move its bytes between tiers. So the slot
// prompt, its clones (child slots, server_prompt::clone) and the prompt-cache entries SHARE one object instead of deep-copying
// every checkpoint (the state of a hybrid / recurrent model is tens to hundreds of MiB per checkpoint).
// Upstream llama.cpp e152332a1 (#30137, MIT) shares std::shared_ptr<const common_prompt_checkpoint>; the fork keeps it non-const
// because the tier moves (spill_tier / spill / cram_compress / reload / adopt_spill_file) rewrite the object's own storage.
// Composition with the fork's spill files: a spill file belongs to ONE object, so it is deleted by server_ckpt_release() only
// when the last owner lets go (a bare remove_spill_file() on a shared object would pull the file out from under the other owners).
using server_prompt_checkpoint_ptr = std::shared_ptr<common_prompt_checkpoint>;
using server_prompt_checkpoints    = std::list<server_prompt_checkpoint_ptr>;

// drop one owner; the last owner also deletes the spill file, if the object has one
inline void server_ckpt_release(server_prompt_checkpoint_ptr & p) {
    if (p && p.use_count() == 1) {
        p->remove_spill_file();
    }
    p.reset();
}

// The object this owner may reload() into memory. A spilled object shared with other owners is copied first (cheap: its bytes are
// not resident, only the positions and the tier / file reference are copied), so a slot restoring a spilled cache entry holds
// its own resident bytes and the entry stays spilled and holds nothing -- what the deep copy gave before sharing.
inline common_prompt_checkpoint & server_ckpt_private(server_prompt_checkpoint_ptr & p) {
    if (p.use_count() > 1 && p->spilled()) {
        p = std::make_shared<common_prompt_checkpoint>(*p);
    }
    return *p;
}

struct server_prompt {
    server_tokens tokens;

    server_prompt_checkpoints checkpoints;

    // Fingerprint of the LoRA configuration the KV behind `tokens` was computed under (lora_config_fingerprint(),
    // server-common.h). 0 = no adapter active, which is what every server without a LoRA has always been, so a
    // request without adapters keys the caches exactly as before this field existed. Set when a task is launched on
    // the slot (launch_slot_with_task); carried into and out of server_prompt_cache entries.
    // Row S69-LORAX-PICK-LORA-CONFIG-IN-PROMPT-CACHE-AND-RADIX-KEY-2026-09-29-01.
    uint64_t lora_fp = 0;

    void clear() {
        tokens.clear();
        // a spilled checkpoint owns a file on disk (--ctx-ckpt-spill-dir); a plain clear() would leave it behind
        for (auto & ckpt : checkpoints) {
            server_ckpt_release(ckpt);
        }
        checkpoints.clear();
        lora_fp = 0;
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            lora_fp,
        };
    }
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;
    // Speculative-decoder carry state (MTP boundary hidden row, common/spec-mtp-state.h): a few KiB, always resident, never
    // spilled. Row S69-ENGREV-BEELLAMA-MTP-CARRY-STATE-CKPT-CACHE-2026-10-27-01.
    std::vector<uint8_t> spec;

    size_t size() const {
        return main.size() + drft.size() + spec.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    // times load() restored this entry (segmented-LRU: 0 = probation, >0 = protected; only read when server_prompt_cache::slru)
    uint32_t n_hits = 0;

    // Pinned (--warm-prefix-file): never chosen for eviction, never dropped as "contained in a longer prompt". Bounded by
    // server_prompt_cache::pinned_share of the byte limit, which the warm-up checks. Row S69-rtp-llm-PICK-STARTUP-WARMED-PINNED-PREFIX-CACHE-2026-09-29-01.
    bool pinned = false;

    // NVMe tier (--cache-disk-dir). GATE-CLAIM-OK: design comment, claims no resource is unavailable.
    // Non-zero: the main/draft state bytes live in the server_prompt_cache's common_disk_tier under this blob id and
    // `data` is empty; the tokens, LoRA fingerprint and checkpoints stay here so prefix matching works unchanged.
    // 0 = the state is held in memory, which is every entry on a server without the flag.
    uint64_t disk_id      = 0;
    size_t   disk_n_bytes = 0; // main + draft bytes held by the blob

    bool spilled() const { return disk_id != 0; }

    // resident bytes: what this entry costs in memory (--cache-ram counts exactly this)
    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // Scan-resistant eviction (--cache-ram-slru, default off). Plain LRU lets a burst of one-shot long prompts push out a state that
    // was restored many times (a shared system prompt). With this on, entries never restored (probation) are evicted first, oldest
    // first; restored entries (protected) are only evicted when no probation entry is left or when they hold more than 80 percent
    // of the byte limit. With no restores at all the order is exactly LRU.
    // SOURCE-READ: engines/vllm/vllm/v1/kv_offload/cpu/policies/arc.py (recent/frequent split) and engines/sglang/python/sglang/srt/
    // mem_cache/utils.py:48-84 (SLRU among the offered policies); Apache-2.0, behaviour only, no code copied.
    bool slru = false;

    // Mark `it` as just used: one more hit, and most-recent position. load() calls this.
    void touch(std::list<server_prompt_cache_state>::iterator it);

    // Free memory by one entry: the one the eviction policy names among those still held in memory (the front one unless
    // slru is on). With the NVMe tier on, that entry is written to disk and keeps its place in `states` (see enable_disk());
    // without it, or when the tier refuses, the entry is removed as it always was. Makes progress on every call.
    // Returns false when nothing could be freed (every entry held in memory is pinned). The make-room loops stop on false,
    // and alloc() then declines to store the newcomer instead of exceeding the limit.
    bool evict_one();

    // Pinned entries (see server_prompt_cache_state::pinned). pinned_share is the largest fraction of limit_size they may hold.
    float  pinned_share = 0.5f;
    size_t pinned_size() const;   // resident bytes of the pinned entries
    size_t n_pinned() const;
    // Pin every entry computed under lora_fp whose tokens are a prefix of (or equal to) `tokens`. Returns how many were pinned.
    size_t pin_prefix_of(const server_tokens & tokens, uint64_t lora_fp);
    // True when some pinned entry computed under lora_fp is a prefix of (or equal to) `tokens` (also when it was pinned earlier).
    bool   has_pinned_prefix_of(const server_tokens & tokens, uint64_t lora_fp) const;
    // Unpin and delete every pinned entry (used when the pinned bytes exceed the share). Returns how many were deleted.
    size_t drop_pinned();
    bool   pinned_over_share() const { return limit_size > 0 && (double) pinned_size() > (double) pinned_share * (double) limit_size; }

    // --- NVMe tier (--cache-disk-dir), row S69-TIERED-KV-DISK-TIER-SLICE1-2026-10-05-01 -----------------------------------
    // Off until enable_disk() succeeds; with it off every code path below is the pre-tier behaviour.
    // limit_bytes: cap on the bytes in `dir`; min_tokens: prompts shorter than this are dropped, not spilled (the disk
    // round trip has a fixed cost that a short prefill may not repay; MEASURED: temp/_c2_1005_rb_*.txt, restoring beats
    // recomputing from 3 to 20 tokens on four real models on this CPU host, so the --cache-disk-min-tokens default is 32);
    // write_budget: SSD wear budget in bytes/hour, 0 = unlimited.
    bool enable_disk(const std::string & dir, size_t limit_bytes, size_t min_tokens, size_t write_budget_bytes_per_hour,
                     std::string & err, common_disk_tier::clock_fn clock = nullptr);
    bool disk_enabled() const { return disk != nullptr; }

    // Write one in-memory entry to the tier and free its bytes. False (entry untouched) when the tier is off, the prompt is
    // below min_tokens, or the tier refused it (cap, write budget, I/O error); evict_one() then removes the entry as before.
    bool spill_state(std::list<server_prompt_cache_state>::iterator it);

    // The state bytes of a spilled entry, read back and verified. False (and the entry is removed from the cache, blob
    // included) when the blob is missing, truncated or fails its checksum: the caller treats it as a miss.
    bool read_spilled(std::list<server_prompt_cache_state>::iterator it, server_prompt_data & out);

    // Remove an entry and, if it is spilled, its blob. Returns the iterator after it.
    std::list<server_prompt_cache_state>::iterator drop_state(std::list<server_prompt_cache_state>::iterator it);

    // Make `file_bytes` of room in the tier by removing the oldest spilled entries (never `keep`), but only if the write budget
    // will accept the write: otherwise already-paid-for entries would be thrown away for nothing. True when it now fits.
    bool make_disk_room(size_t file_bytes, std::list<server_prompt_cache_state>::iterator keep);

    // A slot reloaded (or failed to reload) a checkpoint that lives in the tier; the slot code reports it so /metrics counts it.
    void note_ckpt_reload(bool ok) { (ok ? ctr.n_ckpt_restored : ctr.n_ckpt_restore_failed)++; }

    // --- restore versus recompute (row S69-TIERED-RESTORE-VS-RECOMPUTE-POLICY-2026-10-05-01) ---------------------------------
    // A spilled state is worth keeping and restoring only while reading it back takes less time than computing it again:
    //     tokens_saved / prefill_tokens_per_s  >=  min_speedup * bytes / restore_bytes_per_s
    // Both rates are measured on the running server (EWMA): the prefill rate from the server's own prompt processing, the restore
    // rate from the restores it has done (read + checksum + loading the state into the context). Until the first restore the
    // tier's own write throughput stands in for it (writes were slower than reads on the drive measured, so the stand-in
    // errs towards not spilling). With either rate unknown, or min_speedup = 0, the rule has no opinion and the other limits
    // (--cache-disk-min-tokens, size cap, write budget) decide alone.
    // MEASURED: temp/_c2_1005_rb_granite.txt, granite-4.0-h-350m on this CPU: restoring a 23-39 MB state took 14-46 ms
    // while recomputing 64..4096 tokens took 107..5616 ms (crossover about 10-30 tokens); the rule exists for hosts where prefill is
    // much quicker than here, which this host cannot show.
    double min_speedup = 1.0;
    void   note_prefill(size_t n_tokens, double seconds);
    void   note_restore(size_t bytes, double seconds);
    double prefill_tps() const { return prefill_tps_; }
    double restore_Bps() const;
    bool   restore_pays(size_t tokens_saved, size_t bytes) const;
    double prefill_tps_ = 0;
    double restore_Bps_ = 0;

    // With --ctx-ckpt-spill-dir AND the tier on, the checkpoints a slot has to move out of memory (--ctx-ckpt-live-n) go to the tier
    // instead of the slot's own files, so they share the tier's size cap, SSD write budget and checksum (the files have none of the
    // three). Makes room like a spill does; false (the checkpoint stays in memory) when the tier refuses.
    bool spill_slot_checkpoint(common_prompt_checkpoint & ck); // ck may be shared with cache entries (see server_prompt_checkpoint_ptr)

    struct disk_counters {
        uint64_t n_spilled         = 0; // entries moved from memory to disk
        uint64_t n_restored        = 0; // spilled entries read back and used by load()
        uint64_t n_restore_failed  = 0; // spilled entries that could not be read back (dropped, request recomputes)
        uint64_t n_dropped_small   = 0; // evicted without a spill: prompt below min_tokens
        uint64_t n_dropped_refused = 0; // evicted without a spill: tier full, over the write budget, volume low on free space or an I/O error
        uint64_t n_disk_evicted    = 0; // spilled entries removed to make room for newer ones
        uint64_t n_ckpt_spilled         = 0; // context checkpoints of a spilled entry written to the tier
        uint64_t n_ckpt_refused         = 0; // ... that the tier refused (they stay in memory)
        uint64_t n_ckpt_restored        = 0; // tier checkpoints a slot read back and used
        uint64_t n_ckpt_restore_failed  = 0; // tier checkpoints that failed their checks (dropped, the prompt is recomputed)
        uint64_t n_ckpt_adopted         = 0; // checkpoints a slot had spilled to a --ctx-ckpt-spill-dir file, copied into the entry
        uint64_t n_ckpt_slot_spilled    = 0; // a slot's live checkpoints moved to the tier (--ctx-ckpt-spill-dir with the tier on)
        uint64_t n_ckpt_slot_refused    = 0; // ... that the tier refused (they stay in the slot's memory)
        uint64_t n_dropped_slow         = 0; // evicted without a spill: restoring it would take longer than recomputing it (restore_pays)
        uint64_t n_ckpt_dropped_slow    = 0; // checkpoints of a spilled entry kept in memory for the same reason
        uint64_t n_restore_skipped      = 0; // a spilled entry matched but was not restored: recomputing the matched prefix is quicker
    };
    disk_counters ctr;

    size_t size_disk() const;      // bytes held on disk (file sizes)
    size_t n_spilled() const;      // entries on disk
    server_prompt_cache_disk_stats disk_stats() const; // for /metrics; enabled=false when the tier is off

    // shared with every context checkpoint that has a blob in it (common_disk_blob keeps the tier alive until the last one goes)
    std::shared_ptr<common_disk_tier> disk;
    size_t disk_min_tokens = 0;

    size_t size_ckpt() const;      // resident bytes that are context checkpoints

    size_t size() const;           // resident bytes

    size_t n_tokens() const;       // tokens of every entry, spilled or not
    size_t n_tokens_resident() const; // tokens of the entries held in memory (what the token limit bounds)

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // Pick the cached entry load() would restore for `tokens_new` -- states.end() when none is better than the slot's
    // own `prompt`. Only entries whose lora_fp equals `lora_fp` (the request's LoRA configuration) are candidates:
    // KV computed under another adapter configuration is not the KV of the same tokens under this one. A slot prompt
    // held under a different fingerprint is not a baseline an entry has to beat (the caller drops it on launch).
    // Split out of load() so the selection can be tested without a llama_context.
    std::list<server_prompt_cache_state>::iterator find_best(const server_prompt & prompt, const server_tokens & tokens_new, const llama_context * ctx_tgt, const thinking_tokens & think_tokens, uint64_t lora_fp);

    // lora_fp: fingerprint of the request's LoRA configuration; default 0 = no adapters (the historical behaviour).
    // spec_out / hit (optional): on a cache hit, *hit = true and *spec_out receives the entry's speculative carry-state bytes
    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, const thinking_tokens & think_tokens, uint64_t lora_fp = 0,
              std::vector<uint8_t> * spec_out = nullptr, bool * hit = nullptr);

    // Remove every cached prompt whose tokens are a prefix of (or equal to) `tokens`, i.e. every entry that could
    // have been published from a slot holding `tokens`. Used by slot ERASE so erased content is gone, not merely
    // evicted from the slot. -> number of entries removed.
    size_t remove_prefixes_of(const server_tokens & tokens);

    // Drop every entry (resident or spilled to the disk tier, pinned or not). Used after a tensor hot-swap: every cached
    // state was computed with the previous weights. -> number of entries removed.
    size_t clear_all();

    void update();
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
