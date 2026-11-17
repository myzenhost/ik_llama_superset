#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "json.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <set>

struct server_context_impl; // private implementation
namespace respcompact { struct plan; } // server-responses-compact.h

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    std::vector<std::string> model_output_modalities; // output modalities for GET /models
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    bool has_inp_video;
    bool has_cap_chat;
    bool has_cap_tts;
    bool has_audio_stream; // realtime transcription, see mtmd_support_audio_stream()
    json json_ui_settings;
    int slot_n_ctx;
    uint64_t mem_host_bytes;    // model + KV + compute buffers in host memory (common_memory_totals)
    uint64_t mem_device_bytes;  // ... in device (GPU) buffers
    int32_t n_ubatch;   // physical batch: the most tokens one non-splittable (embedding) request can hold
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
    std::string model_ftype;
    // GGUF "general.architecture" (e.g. "llama", "qwen3", "gemma3"), read via
    // llama_model_meta_val_str. Added for the ollama-native API shim (row
    // S69-SERVING-SURFACE-OLLAMA-NATIVE-API-SHIM-2026-09-27-01), which needs it for
    // /api/tags,/api/show,/api/ps "details.family" -- empty string if the GGUF has no such key.
    std::string model_arch;
    // per-device / host / other-buffer memory in use by this model and context (bytes and MiB), the structured form of
    // the "memory breakdown [MiB]" log table; a supervisor reads this instead of parsing the log. Taken when get_meta()
    // is called, which is after load: model / context / compute do not move afterwards, the device free figure is
    // reported as free_at_load for that reason. Row S69-OLLAMA-PICK-EXPOSE-MEMORY-BREAKDOWN-JSON-2026-09-29-01.
    json memory_breakdown;
};

using server_state_callback_t = std::function<void(server_state, json /* payload */)>;

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // note: must be set before load_model() is called
    void set_state_callback(server_state_callback_t callback);

    // called with the per-slot context (n_ctx_slot) after EVERY model (re)load: the first load and each wake from sleep
    // (load_model() runs again, and the context may differ). The tools' output budget follows it. Same-thread as load_model().
    // note: must be set before load_model() is called
    void set_model_loaded_callback(std::function<void(int n_ctx_slot)> callback);

    // --warm-prefix-file: the secret only the warm-up thread knows; a request that carries it as "cache_pin_token" gets its prompt state pinned
    void   set_pin_token(const std::string & token);
    // thread-safe counters of the pinned prefixes (updated by the slot loop)
    size_t pinned_bytes() const;
    size_t pinned_entries() const;
    bool   pin_overflowed() const; // pinned bytes crossed the share: the pins were deleted
    // one count per finished warm-up pin pass (whatever its outcome), and whether the LAST pass left its prefix pinned (false when the
    // prompt state was too large to be kept in the host prompt cache at all)
    size_t pin_passes() const;
    bool   last_pin_covered() const;
};


// forward declarations
struct server_res_generator;

// G3 SQL session store (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01).
// Opaque here on purpose: the real definition (server-context.cpp) wraps a sqlite_modern_cpp
// sqlite::database and only exists when this server build defines SQLITE3_MODERN_CPP_SUPPORT
// (CMake option LLAMA_SERVER_SQLITE3, default OFF). This forward declaration -- and the
// server_routes member below that uses it -- are unconditional either way, so the 7 routes can
// always be mounted and always report a clear reason when the feature is off, instead of a bare
// 404 (see server_routes::sql_unavailable_reason(), server-context.cpp).
struct sql_session_db;

// Admission ceilings for completion requests (--max-queued-requests / --max-queued-tokens). Without them the task queue
// grows without bound under overload: every request is accepted and waits behind all the others, so latency and
// memory grow with the backlog. With them, a request that would cross a ceiling is answered 503 at once, so the
// client can retry or go elsewhere. Behaviour follows Aphrodite's scheduler options max_num_queued_reqs /
// max_num_queued_tokens (aphrodite/config/scheduler.py:74-103, AGPL-3.0, read for behaviour; no code copied):
//   * requests: counted from admission until the response object is destroyed (done, failed or disconnected);
//   * tokens:   a request's WHOLE prompt counts until it produces its first result (streaming) or completes
//               (non-streaming) -- an overestimate of the real prefill backlog, which is the safe direction for a
//               latency guard. One deliberate difference: when nothing is in prefill, a request is admitted
//               whatever its size, so a prompt longer than the ceiling is slow rather than refused forever.
struct server_admission {
    int32_t max_reqs   = 0; // 0 = no limit
    int64_t max_tokens = 0; // 0 = no limit

    std::mutex mtx;
    int32_t  n_reqs       = 0;
    int64_t  n_tokens     = 0;
    uint64_t n_admitted   = 0;
    uint64_t n_rejected   = 0;
    uint64_t n_evicted    = 0; // queued requests that gave their place to a higher-priority arrival (--evict-low-priority)

    bool enabled() const { return max_reqs > 0 || max_tokens > 0; }
    // true = admitted and counted; false = refused, `why` says which ceiling
    bool try_admit(int64_t prompt_tokens, std::string & why);
    // count a request in without checking the ceilings: used when it has just taken the place of an evicted one (--evict-low-priority).
    // The evicted request's own ticket is released when its response object is destroyed, so the counters sit above the ceiling for
    // that short moment and then settle back
    void force_admit(int64_t prompt_tokens);
    void release(int64_t prompt_tokens, bool request);
    json to_json();
};

// Held by the response object of an admitted request; gives its share back exactly once.
struct server_admission_ticket {
    std::shared_ptr<server_admission> adm;
    int64_t tokens  = 0;     // still counted as prefill
    bool    request = true;  // still counted as in flight

    server_admission_ticket(std::shared_ptr<server_admission> a, int64_t t) : adm(std::move(a)), tokens(t) {}
    server_admission_ticket(const server_admission_ticket &) = delete;
    server_admission_ticket & operator=(const server_admission_ticket &) = delete;
    void end_prefill() { if (tokens > 0) { adm->release(tokens, false); tokens = 0; } }
    ~server_admission_ticket() { adm->release(tokens, request); }
};

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // LLAMA_HOTSWAP_ENABLED: GET /health also reloads the tensors that changed on disk when no slot is processing. Set by
    // server.cpp in single-model mode only -- a router has no task loop to run the reload on.
    std::atomic<bool> hotswap_on_health{false};

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_memory;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    // Ported from the pre-merge fork's examples/server/server.cpp (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G2).
    server_http_context::handler_t get_slots_list;      // G2: GET /slots/list (always available)
    server_http_context::handler_t get_saved_prompts;   // G2: GET /list (--slot-save-path)
    server_http_context::handler_t post_delete_prompt;  // G2: POST /delete_prompt (--slot-save-path)
    server_http_context::handler_t post_rename_prompt;  // G2: POST /rename_prompt (--slot-save-path)
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_chat_completions_tok;
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_responses_tok_oai;
    server_http_context::handler_t post_responses_compact;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::ws_handler_t ws_realtime;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t post_chat_tokenize;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_watermark_detect;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t post_classify;
    server_http_context::handler_t post_decision;
    server_http_context::handler_t post_score; // shared-prefix candidate scoring
    server_http_context::handler_t post_logits; // prefill-only raw LM-head logits (POST /v1/logits)
    server_http_context::handler_t post_systemone; // typed decision models (ik c069e89e1)
    server_http_context::handler_t post_tts;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;
    server_http_context::handler_t get_sampler_overrides;
    server_http_context::handler_t post_sampler_overrides;

    // Ported from the pre-merge fork's examples/server/server.cpp (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, groups G5 + G1).
    server_http_context::handler_t get_version;             // G5: GET /version
    server_http_context::handler_t get_props_v1;             // G5: GET /v1/props
    server_http_context::handler_t get_control_vectors;      // G1: GET /control-vectors
    server_http_context::handler_t post_control_vectors_load;   // G1: POST /control-vectors/load
    server_http_context::handler_t post_control_vectors_unload; // G1: POST /control-vectors/unload
    server_http_context::handler_t post_control_vectors_apply;  // G1: POST /control-vectors/apply

    // Ported from the pre-merge fork's examples/server/server.cpp (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G3).
    // A tiny SQLite-backed key/value store (3 tables: sessions/templates/names) that clients (the
    // bundled web UIs) use to persist named sessions server-side. Opt-in on two axes: compiled in
    // only with -DLLAMA_SERVER_SQLITE3=ON, and active only when --sql-save-file <path> is passed.
    // Always mounted (no #ifdef in server.cpp); each handler reports a clear reason when either
    // axis is off instead of a bare 404 -- same pattern as get_saved_prompts above for
    // --slot-save-path. get_sql_sessions answers both GET and POST /sessions (the fork's own
    // handle_sessions ignored the request body either way).
    server_http_context::handler_t post_sql_load;      // G3: POST /load
    server_http_context::handler_t post_sql_save;       // G3: POST /save
    server_http_context::handler_t post_sql_rename;     // G3: POST /rename
    server_http_context::handler_t post_sql_all;        // G3: POST /all
    server_http_context::handler_t get_sql_sessions;    // G3: GET /sessions
    server_http_context::handler_t post_sql_sessions;   // G3: POST /sessions (identical handler,
                                                          // split in two so router mode can proxy
                                                          // each verb through its own proxy_get/proxy_post)
    server_http_context::handler_t post_sql_delete;     // G3: POST /delete
    server_http_context::handler_t get_sql_vacuum;      // G3: GET /vacuum
    // G4: sqlite-zstd transparent compression over the G3 store (needs --sqlite-zstd-ext-file)
    server_http_context::handler_t get_sql_zstd_configs;      // GET  /zstd_get_configs
    server_http_context::handler_t post_sql_zstd_maintenance; // POST /zstd_incremental_maintenance
    server_http_context::handler_t post_sql_zstd_enable;      // POST /zstd_enable_transparent
    server_http_context::handler_t post_sql_zstd_update;      // POST /zstd_update_transparent

    // to be used in router mode
    json get_model_info() const;

    // Ollama shim load / unload (an /api/chat with empty messages, an /api/generate with an empty prompt; row
    // S69-ENGREV-OLLAMA-SHIM-EMPTY-MESSAGES-LOAD-UNLOAD-2026-10-27-01, SOURCE-READ ollama server/routes.go:480,563,3098,3241).
    // Router mode: server.cpp sets these two to the router's handlers (server_models_routes::ollama_model_load / _unload). Body
    // {"model": name, "keep_alive": ...}; answer 200 {"success":true} or an error object. Empty in single-model mode.
    server_http_context::handler_t ollama_model_load;   // load the model if it is not running, then apply keep_alive (expiry timer)
    server_http_context::handler_t ollama_model_unload; // expire the model now (keep_alive 0): idle -> stops, busy -> stops when idle
    // Single-model mode: the model "unloads" by entering the sleeping state (--sleep-idle-seconds >= 0), which frees it and reloads it on
    // the next request. Without --sleep-idle-seconds the server cannot unload its own model.
    bool model_sleep_enabled() const;
    bool model_is_sleeping() const;
    // keep_alive 0: request the sleeping state and wait up to wait_ms for it; false = sleeping is not enabled. A busy slot defers it.
    bool sleep_model(int64_t wait_ms) const;
    // empty-request load: leave the sleeping state (no-op when awake)
    void wake_model() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type,
            const common_chat_session & chat_session = {});
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_rerank_impl(const server_http_req & req);
    std::unique_ptr<server_res_generator> handle_classify_impl(const server_http_req & req);
    std::unique_ptr<server_res_generator> handle_count_tokens(const server_http_req & req, task_response_type res_type);
    // Responses compaction: runs the forced create_summary call; nullptr = success (item + usage filled), else the error reply.
    std::unique_ptr<server_res_generator> run_compaction(const server_http_req & req, respcompact::plan & plan, json & item, json & usage);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);

    std::shared_ptr<server_admission> admission = std::make_shared<server_admission>();

    // G3 SQL session store (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01).
    // sql_db is null unless --sql-save-file was given AND opening/creating it and its schema
    // succeeded AND SQLITE3_MODERN_CPP_SUPPORT is compiled in; sql_db_error explains which of
    // those is false, filled in by the constructor, read by every G3 route handler.
    std::shared_ptr<sql_session_db> sql_db;
    std::string sql_db_error;
    std::string sql_unavailable_reason() const;
    // G4: true only when --sqlite-zstd-ext-file loaded into sql_db's connection; sql_zstd_error says why not.
    bool        sql_zstd_loaded = false;
    std::string sql_zstd_error;
    std::string sql_zstd_unavailable_reason() const;

    // cached responses, to be used during sleep
    std::mutex     mutex_cache;
    json           cached_models  = nullptr;
    json           cached_props   = nullptr;
    server_metrics cached_metrics;
    server_slot_gauges cached_gauges;
    // per-device memory rows (memory_*_bytes gauges), snapshotted on entering sleep next to cached_metrics: the context is gone while
    // asleep, so the scrape reports the rows it had right before, and /metrics keeps the same line set awake and asleep
    json           cached_memory = nullptr;
    // set when a scrape during sleep already reported the throughput buckets
    bool           should_reset_buckets = false;
    // call right before sleep to update the cached responses
    void update_cached_responses(bool is_sleeping);
};
