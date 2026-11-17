#pragma once

#include "common.h"
#include "download.h"
#include "preset.h"
#include "server-common.h"
#include "server-http.h"
#include "server-model-events.h"
#include "server-queue.h"
#include "server-retry.h"

#include <mutex>
#include <condition_variable>
#include <thread>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

/**
 * state diagram:
 *
 * DOWNLOADING ──► DOWNLOADED ──► (replaced by new instance)
 *
 * UNLOADED ──► LOADING ──► LOADED ◄──── SLEEPING
 *  ▲            │            │               ▲
 *  └───failed───┘            │               │
 *  ▲                         └──sleeping─────┘
 *  └────────unloaded─────────┘
 */
enum server_model_status {
    // TODO: also add downloading state when the logic is added
    SERVER_MODEL_STATUS_DOWNLOADING,
    SERVER_MODEL_STATUS_INCOMPLETE, // a download that never finished, listed without an instance
    SERVER_MODEL_STATUS_DOWNLOADED,
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED,
    SERVER_MODEL_STATUS_SLEEPING
};

enum server_model_source {
    SERVER_MODEL_SOURCE_PRESET,
    SERVER_MODEL_SOURCE_MODELS_DIR,
    SERVER_MODEL_SOURCE_CACHE,
};

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_DOWNLOADING: return "downloading";
        case SERVER_MODEL_STATUS_INCOMPLETE:  return "incomplete";
        case SERVER_MODEL_STATUS_DOWNLOADED:  return "downloaded";
        case SERVER_MODEL_STATUS_UNLOADED:    return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:     return "loading";
        case SERVER_MODEL_STATUS_LOADED:      return "loaded";
        case SERVER_MODEL_STATUS_SLEEPING:    return "sleeping";
        default:                              return "unknown";
    }
}

static std::string server_model_source_to_string(server_model_source source) {
    switch (source) {
        case SERVER_MODEL_SOURCE_PRESET:     return "preset";
        case SERVER_MODEL_SOURCE_MODELS_DIR: return "models_dir";
        case SERVER_MODEL_SOURCE_CACHE:      return "cache";
        default:                             return "unknown";
    }
}

struct server_model_meta {
    server_model_source source = SERVER_MODEL_SOURCE_CACHE;
    common_preset preset;
    std::string name;
    std::set<std::string> aliases; // additional names that resolve to this model
    std::set<std::string> tags;    // informational tags, not used for routing
    int port = 0;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0; // for LRU unloading
    std::vector<std::string> args; // args passed to the model instance, will be populated by render_args()
    json loaded_info; // info to be reflected via /v1/models endpoint ; if in DOWNLOADING state, it should contain download progress info
    json progress; // reflect load or download progress info, if any
    int exit_code = 0; // exit code of the model instance process (only valid if status == FAILED)
    int stop_timeout = 0; // seconds to wait before force-killing the model instance during shutdown
    std::string stop_reason; // why the last child stopped (server-model-events.h): idle-ttl, lru-evict, manual, reload, shutdown, load-oom, crash, exited; empty while running or never run
    std::string last_error;  // the error lines the last failed child logged (copied from the instance), empty when it did not fail
    bool hidden = false; // hidden from GET /models, but still accept if requested
    json architecture = server_model_architecture_json(false, false, false, {"text"});
    uint32_t n_ctx_train = 0; // trained context, read from the GGUF metadata; 0 when unknown

    bool is_ready() const {
        return status == SERVER_MODEL_STATUS_LOADED;
    }

    bool is_running() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING || status == SERVER_MODEL_STATUS_SLEEPING;
    }

    bool is_ready_or_sleep() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_SLEEPING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }

    void update_args(common_preset_context & ctx_presets, std::string bin_path);
    void update_caps(const common_params & base);
};

// a download that never finished: the tag it was asked under, the sidecar kinds it
// pulls and the partial files its plan writes
struct server_unfinished_download {
    std::string                          model;
    std::vector<std::string>             sidecars;
    std::vector<common_download_partial> partials;
};

// the downloads that never finished, recorded next to the model cache. The cache names
// partial files after their content hash, so this record is the only trace of what is
// still incomplete: it lets a restarted router list, resume or remove them
// a partial another recorded download also writes is never deleted
// these functions are thread-safe
struct server_unfinished_downloads {
    // read the record an earlier run left
    void load();

    // record a download that starts, replacing an earlier record of the same model,
    // and delete the partials of that record the new plan no longer writes
    void add(const server_unfinished_download & download);

    // forget a download whose files are complete
    void forget(const std::string & model);

    // forget a download and delete its partials; returns false when it is not recorded
    bool remove(const std::string & model);

    // drop the sidecar `kind` from a download and delete its partials; returns false
    // when the download is not recorded with that kind
    bool remove_sidecar(const std::string & model, const std::string & kind);

    // every recorded download
    std::vector<server_unfinished_download> get_all();

private:
    std::mutex mutex;
    std::vector<server_unfinished_download> downloads;

    // write the record back; callers hold the mutex
    void save();

    // delete the partials of `parts` no recorded download writes; callers hold the mutex
    void delete_partials(const std::vector<common_download_partial> & parts);
};

struct server_models_routes;
struct server_lru_sched; // defined in server-models.cpp
struct server_monitor;   // defined in server-models.cpp

// thrown by load() when models_max is already taken. A request that meets it after passing
// the capacity check has lost a check-then-act race and waits in the queue instead of failing
// ref: https://github.com/ggml-org/llama.cpp/issues/28774
struct server_models_limit_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct server_models {
    friend struct server_models_routes;
    friend struct server_lru_sched;
    friend struct server_monitor;

private:
    struct instance_t {
        std::shared_ptr<server_subproc> subproc; // shared with the monitor thread
        server_model_meta meta;
        int req_count = 0; // number of active proxy requests
        // requests between admission and req_count++ (waiting for the load, or about to proxy).
        // a model with any is never picked for eviction or keep_alive expiry: without this a
        // freshly loaded model is idle for a moment and its own waiter loses it
        int n_pending = 0;
        // a stop was requested and the child has not exited yet; its status still reads
        // loaded/sleeping, so this is what keeps a new request from being sent to it
        bool exiting = false;
        // a preset reload wanted to stop this model (preset changed or removed) while it was serving a request; the stop is
        // deferred (row S69-ENGREV-JAN-PRESET-RELOAD-KILLS-BUSY-MODEL-2026-10-27-01) and the reload is redone once it is idle
        bool reload_deferred = false;
        int64_t keep_alive_ms = -1; // how long to stay loaded once idle, -1 = forever
        int64_t expire_at_ms  = 0;  // when the idle model is unloaded, 0 = not scheduled
        // --models-vram-admit: bytes this child was predicted to need, and the free device memory
        // read just before it was spawned (-1 = not read), so its real footprint can be measured
        // when it reports loaded
        int64_t vram_predicted   = 0;
        int64_t vram_free_before = -1;
        // load-OOM retry: the last child died while LOADING after printing an out-of-memory error,
        // and one waiter is currently running the single retry (the others keep waiting for it)
        bool load_oom     = false;
        bool oom_retrying = false;
        // fork: the error lines the last child of this name logged before it exited (server_subproc::last_error), appended
        // to the load-failure message so the client sees why (missing tensor, bad mmproj, OOM)
        std::string last_error = {};
        // why a stop was requested (request_stop_locked); NONE = nobody asked, so an exit is a crash. Consumed by update_status(UNLOADED)
        server_stop_reason stop_asked = SERVER_STOP_NONE;
        // this spawn's failed load has been counted for --model-load-cooldown (many waiters see one failure)
        bool failure_booked = false;

        // ask the child to exit (it handles the command on its stdin, see server_role::setup)
        void request_exit() const;
    };

    // load-OOM retry, reimplemented from ollama's scheduler (sched.go load(): oomRetryAttempted,
    // reduceAutoNumCtxForLoadOOM, evictAllAndWait). Evicts every idle model and, when the context
    // size was left automatic, steps it down (>32k -> 32k, >4k -> 4k); then loads once more.
    // Returns false when there is nothing to change, so the caller reports the failure instead.
    // caller must NOT hold mutex
    bool retry_after_load_oom(const std::string & name);

    // --models-vram-admit: footprint measured at the last load of each model (free before spawn
    // minus free once loaded). Preferred over the GGUF-size guess for the next prediction.
    std::map<std::string, int64_t> vram_measured;

    // evicts idle LRU models until name's predicted memory fits; returns the prediction (0 = off/unknown)
    // caller must NOT hold mutex
    int64_t ensure_vram_for(const std::string & name);
    // free device memory, or the simulated value under LLAMA_SERVER_VRAM_TOTAL_MB; caller holds mutex
    int64_t vram_free_locked();

    std::mutex mutex;
    std::mutex num_ctx_mutex; // serialises apply_num_ctx reloads; never taken while `mutex` is held
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    // models asked to stop, still counted as running until the monitor records their exit
    std::set<std::string> stopping_models;

    // keep_alive: unloads idle models whose expire_at_ms has passed
    std::thread keep_alive_th;
    std::condition_variable cv_keep_alive;
    bool keep_alive_stop = false;
    void keep_alive_loop();

    // caller must hold mutex; asks a running child to exit and marks it exiting
    void request_stop_locked(const std::string & name, instance_t & inst, server_stop_reason reason);
    // caller holds mutex: when `inst` has gone idle and a reload was deferred for it, ask for the reload again
    void redo_deferred_reload_locked(instance_t & inst);
    // downloads that never finished, with the sidecar kinds they pull
    server_unfinished_downloads unfinished_downloads;

    // set to true while load_models() is executing a reload; load() will wait until clear
    bool is_reloading = false;

    // if true, the next get_meta() will trigger a reload of model list
    bool need_reload = false;

    // models marked with load-on-startup, unset once load_startup_models() drains it
    // no value means the startup phase is over, so a reload must not queue anything
    std::optional<std::vector<std::string>> startup_models{std::in_place};

    // --models-snapshot state: seq/frozen are guarded by mutex; snapshot_mutex serialises the file writes
    uint64_t   snapshot_seq         = 0;
    uint64_t   snapshot_written_seq = 0;
    bool       snapshot_frozen      = false;
    std::mutex snapshot_mutex;

    // conv_id -> model name that currently serves its stream session, lets the resumable stream
    // routes go straight to the owning child instead of polling every one. populated when
    // proxy_request forwards a POST carrying an X-Conversation-Id. best effort: a stale entry just
    // makes the child answer not found and the client recovers. owns its lock, one mutex per struct
    struct conv_model_tracker {
        // returns the ticket of this registration, 0 when nothing was registered. erasing or
        // replacing the entry invalidates the ticket, which is how a stop cancels a request
        // parked in the model load wait
        uint64_t remember(const std::string & conv_id, const std::string & model) {
            if (conv_id.empty() || model.empty()) {
                return 0;
            }
            std::lock_guard<std::mutex> lock(mu);
            uint64_t ticket = next_ticket++;
            map[conv_id] = { model, ticket };
            return ticket;
        }

        // false means a stop erased the entry or a newer request replaced it
        bool alive(const std::string & conv_id, uint64_t ticket) {
            std::lock_guard<std::mutex> lock(mu);
            auto it = map.find(conv_id);
            return it != map.end() && it->second.ticket == ticket;
        }

        std::optional<std::string> lookup(const std::string & conv_id) {
            if (conv_id.empty()) {
                return std::nullopt;
            }
            std::lock_guard<std::mutex> lock(mu);
            auto it = map.find(conv_id);
            if (it == map.end()) {
                return std::nullopt;
            }
            return it->second.model;
        }

        void forget(const std::string & conv_id) {
            if (conv_id.empty()) {
                return;
            }
            std::lock_guard<std::mutex> lock(mu);
            map.erase(conv_id);
        }

      private:
        struct entry_t {
            std::string model;
            uint64_t    ticket;
        };
        std::mutex                               mu;
        uint64_t                                 next_ticket = 1;
        std::unordered_map<std::string, entry_t> map;
    };

    common_preset_context ctx_preset;

    common_params base_params;
    std::string bin_path;
    std::vector<std::string> base_env;
    common_preset base_preset; // base preset from llama-server CLI args

    // queue of requests waiting for a models_max slot
    std::unique_ptr<server_lru_sched> sched;

    // if true, add some delay to simulate works (useful for testing)
    bool debug_fake_timing = false;

    void update_meta(const std::string & name, const server_model_meta & meta);

    // unload least recently used models if the limit is reached
    void unload_lru();

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

    // ask the monitor to stop a running instance; send_exit is false for a child that was already force-killed
    // not thread-safe, caller must hold mutex
    void request_stop(const std::string & name, bool send_exit = true, server_stop_reason reason = SERVER_STOP_MANUAL);

    // called by the monitor once a child exited and was reaped
    void on_child_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, server_role_type role, int exit_code);

    // notify SSE clients
    void notify_sse(const std::string & event, const std::string & model_id, const json & data = nullptr);

public:
    // conv_id -> model tracker for the resumable stream routes, owns its lock
    conv_model_tracker conv_models;

    // models whose last loads failed are refused with 503 + Retry-After for an exponentially growing time, so a client
    // that polls a model with a corrupt file does not respawn a loader process on every request (--model-load-cooldown)
    server_load_cooldown load_cooldown;

    server_models(const common_params & params, int argc, char ** argv);
    ~server_models();

    server_response sse; // for real-time updates via SSE endpoint

    // bounded ring (256) of recent status changes with the stop reason, served by GET /models/events (server-model-events.h);
    // unlike sse it is kept, so a client that connects late can ask what happened
    server_model_event_ring events;

    // (re-)load the list of models from various sources and prepare the metadata mapping
    // - if this is called the first time, simply populate the metadata
    // - if this is called subsequently (e.g. when refreshing from disk):
    //   - if a model is running but updated or removed from the source, it will be unloaded
    //   - if a model is not running, it will be added or updated according to the source
    void load_models();

    // lazy-load startup_models, to be called after main() setup phase
    void load_startup_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // downloads that never finished, with their sidecar kinds (thread-safe)
    std::vector<server_unfinished_download> unfinished_downloads_get();

    // drop a download that will never be resumed, its record and its partial files;
    // throws for a name that is not recorded (thread-safe)
    void unfinished_downloads_remove(const std::string & name);

    // remove the sidecar `kind` of a model that is not in use, its files and partials (thread-safe)
    bool remove_sidecar(const std::string & name, const std::string & kind);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta> get_all_meta();

    // per-slot context the server tools' output budget is taken from in the router (which loads no model): the running child
    // named `name` (or one of its aliases), else -- no route known -- the largest running child; read live from each child's
    // /v1/props, so a num_ctx reload or a swap to another model shows up on the next call. 0 = no running child. (thread-safe)
    int64_t tools_slot_ctx(const std::string & name);

    // the models that may serve a request for `name`, in order: `name` itself first, then its fallback chain (the `fallback-model` key of
    // its preset section, else the router's --fallback-model), unknown names dropped, capped by --failover-max-attempts (server-failover.h)
    std::vector<std::string> failover_chain(const std::string & name);

    struct load_options {
        server_role_type role = SERVER_ROLE_INSTANCE;
        // used for spawning a downloading child process
        std::optional<server_model_meta> custom_meta = std::nullopt;
        // > 0: spawn the child with this context size, without changing the stored preset
        int32_t ctx_override = 0;
        // > 0: spawn the child with this many tokens PER SLOT (-c = slot_ctx * n_parallel, --kv-unified-per-slot = slot_ctx), this spawn only;
        // used by apply_num_ctx (ollama options.num_ctx). Wins over ctx_override.
        int32_t slot_ctx = 0;
        // download mode: the sidecar kinds pulled with the model, as -hf would, and
        // the partial files the download plan writes
        std::vector<std::string>             sidecars;
        std::vector<common_download_partial> partials;
    };

    // load and unload model instances
    // these functions are thread-safe
    void load(const std::string & name);
    void load(const std::string & name, const load_options & opts);
    void unload(const std::string & name);
    void unload_all();

    struct update_status_args {
        server_model_status status;
        int exit_code = 0; // only valid if status == UNLOADED
        json loaded_info = nullptr;
        json progress = nullptr;
    };
    // update the status of a model instance (thread-safe)
    // also send SSE notification to /models/sse endpoint
    void update_status(const std::string & name, const update_status_args & args);
    void update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok = true);

    // remove a cache model from disk and update the list (thread-safe)
    // note: only cache models can be removed; returns false if the model doesn't exist or is not a cache model
    bool remove(const std::string & name);

    // wait until the model instance is fully loaded (thread-safe)
    // note: predicate is called while holding the lock
    // return when the model no longer in "loading" state
    void wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate);
    void wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate);

    // ensure the model is in ready state (thread-safe)
    // return false if model is ready
    // otherwise, load the model and blocking wait until it's ready, then return true (meta may need to be refreshed)
    // if models_max is reached, the request waits in a queue until a slot frees up
    // a model whose child is still exiting is waited out and loaded again
    // throws if the load fails, or if should_stop fires while waiting
    bool ensure_model_ready(const std::string & name, const std::function<bool()> & should_stop = nullptr);

    // Ollama keep_alive on a request that runs no inference (empty /api/chat messages, empty /api/generate prompt; row
    // S69-ENGREV-OLLAMA-SHIM-EMPTY-MESSAGES-LOAD-UNLOAD-2026-10-27-01).
    // SOURCE-READ: tmp/ollload_1108/src/routes.go:480-491 (generate) and :3098-3107 (chat), ollama 3fbc609be4b3, MIT, no code copied.
    // SOURCE-NUANCE: unload is `expireRunner(m)` -- a no-op when the model is not loaded -- and still answers done_reason "unload" with
    // 200; it needs an EMPTY prompt/messages AND keep_alive exactly 0, so keep_alive 0 on a real request still generates.
    // The latest keep_alive wins, as in proxy_request(). 0 = unload now (an idle model is stopped at once, a busy one when its last
    // request ends, which is when proxy_request's cleanup schedules the expiry); > 0 restarts the idle timer; < 0 = stay loaded.
    // Returns -1 when the model is not running (nothing applied), 1 when an idle model was told to stop now, else 0.
    int apply_keep_alive(const std::string & name, int64_t keep_alive_ms);

    // per-request options.num_ctx (row S69-ENGREV-OLLAMA-ROUTER-PER-REQUEST-NUM-CTX-RELOAD-2026-10-27-01; design and source cites at the
    // definition in server-models.cpp): call after ensure_model_ready() and before proxy_request(), holding the pending_acquire.
    // req_ctx <= 0 (auto) or equal to the loaded per-slot context = nothing happens. Otherwise the request is clamped (train ctx cap,
    // floor 4 / 2048 with a projector), the busy model is waited out, and the child is reloaded at that context through load()
    // (models_max, --models-vram-admit eviction and the memory estimate apply). Returns "" on success, else the reason for an HTTP 400
    // (the previous context is restored). Never loads a smaller context silently.
    std::string apply_num_ctx(const std::string & name, int64_t req_ctx, const std::function<bool()> & should_stop = nullptr);

    // marks one request as pending on a model from admission until it is proxied, so the model
    // is not evicted or expired under it. hold it across ensure_model_ready() + proxy_request()
    struct pending_ref {
        pending_ref() = default;
        pending_ref(server_models * models, std::string name) : models(models), name(std::move(name)) {}
        pending_ref(pending_ref && other) noexcept : models(other.models), name(std::move(other.name)) { other.models = nullptr; }
        pending_ref(const pending_ref &) = delete;
        pending_ref & operator=(const pending_ref &) = delete;
        pending_ref & operator=(pending_ref &&) = delete;
        ~pending_ref() { if (models) { models->pending_release(name); } }
      private:
        server_models * models = nullptr;
        std::string name;
    };
    pending_ref pending_acquire(const std::string & name);
    void pending_release(const std::string & name);

    // proxy an HTTP request to the model instance
    // keep_alive_ms: the request's keep_alive (-1 = forever), nullopt leaves the model's value as is
    server_http_res_ptr proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used, bool detached = false,
                                      std::optional<int64_t> keep_alive_ms = std::nullopt);
    // forward a websocket connection to the child, returns when either side closes
    void proxy_ws(const server_http_req & req, server_http_context::ws_conn & conn, const std::string & name);

    // handle message sent from server_role::notify_to_router()
    // raw input must starts with CMD_CHILD_TO_ROUTER_STATE, followed by a JSON string
    // called from the monitor thread
    // payload per state:
    //     state = loading     -> payload = {} (TODO: add progress info)
    //     state = ready       -> payload = model_info (json), or {} if wakeup from sleeping
    //     state = sleeping    -> payload = {}
    void handle_child_state(const std::string & name, const std::string & raw_input);

private:
    // one thread watching every child; keep last, the destructor joins the thread
    std::unique_ptr<server_monitor> monitor;
};

struct server_models_routes {
    common_params params;
    json ui_settings = json::object();     // Primary: new name
    std::atomic<bool> stopping = false;    // for graceful disconnecting SSE clients during shutdown
    server_models models;
    server_models_routes(const common_params & params, int argc, char ** argv)
            : params(params), models(params, argc, argv) {
        const std::string & cfg = this->params.ui_config_json;
        if (!cfg.empty()) {
            try {
                json json_settings = json::parse(cfg);
                ui_settings = json_settings;
            } catch (const std::exception & e) {
                LOG_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                throw;
            }
        }
        init_routes();
    }

    void init_routes();
    // handlers using lambda function, so that they can capture `this` without `std::bind`
    server_http_context::handler_t get_router_props;
    server_http_context::handler_t proxy_get;
    server_http_context::handler_t proxy_post;
    server_http_context::ws_handler_t proxy_ws;

    server_http_context::handler_t get_router_models;
    server_http_context::handler_t post_router_models_load;
    server_http_context::handler_t post_router_models_unload;
    server_http_context::handler_t post_router_models_estimate;
    // Ollama shim: empty-request load / keep_alive 0 unload (server_routes::ollama_model_load / ollama_model_unload)
    server_http_context::handler_t ollama_model_load;
    server_http_context::handler_t ollama_model_unload;
    // management API
    server_http_context::handler_t get_router_models_sse;
    server_http_context::handler_t get_router_models_events; // GET /models/events?since=N: the lifecycle ring
    server_http_context::handler_t post_router_models;
    server_http_context::handler_t del_router_models;

    // router side handlers for the resumable streaming routes. each resolves the child that owns
    // a conversation through the conv_id -> model map, no probing or fan out
    server_http_context::handler_t router_stream_get;
    server_http_context::handler_t router_streams_lookup;
    server_http_context::handler_t router_stream_delete;
};

/**
 * A simple HTTP proxy that forwards requests to another server
 * and relays the responses back.
 */
struct server_http_proxy : server_http_res {
    std::function<void()> cleanup = nullptr;
    server_http_proxy(const std::string & method,
                      const std::string & scheme,
                      const std::string & host,
                      int port,
                      const std::string & path,
                      const std::map<std::string, std::string> & headers,
                      const std::string & body,
                      const std::map<std::string, uploaded_file> & files,
                      const std::function<bool()> should_stop,
                      int32_t timeout_read,
                      int32_t timeout_write
                      );
    ~server_http_proxy() {
        if (cleanup_pipes) {
            cleanup_pipes();
        }
        if (cleanup) {
            cleanup();
        }
    }
private:
    std::function<void()> cleanup_pipes = nullptr;
    std::thread thread;
    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
        std::string content_type;
    };
};
