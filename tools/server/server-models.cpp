#include "server-common.h"
#include "http.h"
#include "server-models.h"
#include "server-models-snapshot.h"
#include "server-models-reload-guard.h"
#include "server-vram.h"
#include "server-failover.h"
#include "server-oom-line.h"
#include "server-context.h"
#include "server-stream.h"
#include "server-auth.h"
#include "server-tools.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "hf-cache.h"
#include "http.h"
#include "subproc.h"
#include "gguf.h"
#include "fit.h"
#include "ggml-backend.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <optional>

#include <functional>
#include <optional>
#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <cstring>

#ifndef _WIN32
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif
#include "log-upstream.h" // upstream LOG() semantics -- see common/log-upstream.h

#define DEFAULT_STOP_TIMEOUT 10 // seconds

// note: SIGPIPE is ignored by the server
static void request_child_exit(server_subproc & proc) {
    FILE * stdin_file = proc.sproc.stdin_file();
    if (stdin_file) {
        fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
        fflush(stdin_file);
    }
}

static bool line_reports_oom(const std::string & line); // fork: load-OOM retry, defined below

// fork: the error text of one child log line, "" when it is not an error line.
// SOURCE-READ: tmp/ollama_src_1107/llm_status.go (ollama 3fbc609be4b3, MIT; no code copied; errorPrefixes at
//   lines 67-83, statusErrorLine at 170-193)
// SOURCE-NUANCE: ollama matches the EARLIEST of a fixed prefix list anywhere in the line and keeps the text from that prefix on
//   (statusErrorLine), else the whole line when it reads as out-of-memory; it never keeps plain info lines. Added for this fork's
//   own loader wording, which none of ollama's prefixes match: "failed to load", "exiting due to".
static std::string status_error_line(const std::string & raw) {
    std::string line = raw;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
        line.pop_back();
    }
    if (line.empty()) {
        return "";
    }
    static const char * prefixes[] = {
        "panic:", "fatal error:", "error:", "Error:", "CUDA error", "ROCm error", "cudaMalloc failed", "\"ERR\"",
        "error loading model", "GGML_ASSERT", "llama_init_from_model:",
        "failed to load", "exiting due to", // fork loader wording (llama.cpp / tools/server)
    };
    size_t best = std::string::npos;
    for (const char * p : prefixes) {
        const size_t i = line.find(p);
        if (i != std::string::npos && (best == std::string::npos || i < best)) {
            best = i;
        }
    }
    if (best != std::string::npos) {
        return line.substr(best);
    }
    return line_reports_oom(line) ? line : "";
}

// single-threaded, watching all child processes at once
struct server_monitor {
    server_monitor(server_models & models) : models(models) {
        th = std::thread([this]() { run(); });
    }

    ~server_monitor() {
        push({ cmd_t::QUIT, {}, "", 0, false });
        th.join();
    }

    // thread-safe
    void watch(const std::string & name, std::shared_ptr<server_subproc> proc, server_role_type role, int port) {
        child_t c;
        c.name = name;
        c.proc = std::move(proc);
        c.role = role;
        c.port = port;
        if (!c.proc->has_output()) {
            SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            c.eof = true;
        }
        push({ cmd_t::WATCH, std::move(c), "", 0, false });
    }

    // thread-safe
    void stop(const std::string & name, int stop_timeout, bool send_exit) {
        push({ cmd_t::STOP, {}, name, stop_timeout, send_exit });
    }

private:
    struct child_t {
        std::string name;
        std::shared_ptr<server_subproc> proc;
        server_role_type role = SERVER_ROLE_INSTANCE;
        int port = 0;
        std::string buf[SERVER_SUBPROC_STREAMS]; // partial line of each pipe
        bool eof = false;     // output closed, waiting for the process to be reaped
        int64_t deadline = 0; // force-kill time in ms, 0 when no stop is pending
    };

    struct cmd_t {
        enum { WATCH, STOP, QUIT } type;
        child_t child;
        std::string name;
        int  stop_timeout;
        bool send_exit;
    };

    void push(cmd_t && cmd) {
        {
            std::lock_guard<std::mutex> lk(mu);
            cmds.push_back(std::move(cmd));
        }
        waiter.wake();
    }

    // returns true if the loop should exit
    bool handle_commands() {
        std::deque<cmd_t> batch;
        {
            std::lock_guard<std::mutex> lk(mu);
            batch.swap(cmds);
        }
        for (auto & cmd : batch) {
            switch (cmd.type) {
                case cmd_t::WATCH:
                    children.push_back(std::move(cmd.child));
                    break;
                case cmd_t::STOP:
                    // the newest child with this name is the one the registry knows
                    for (auto it = children.rbegin(); it != children.rend(); ++it) {
                        if (it->name != cmd.name) {
                            continue;
                        }
                        if (cmd.send_exit && !it->eof) {
                            request_child_exit(*it->proc);
                        }
                        it->deadline = ggml_time_ms() + (int64_t) cmd.stop_timeout * 1000;
                        break;
                    }
                    break;
                case cmd_t::QUIT:
                    return true;
            }
        }
        return false;
    }

    // read what the child wrote, handle its commands and forward its logs, line by line
    void read_output(child_t & c) {
        for (int i = 0; i < SERVER_SUBPROC_STREAMS; i++) {
            read_stream(c, (server_subproc_stream) i);
        }
        c.eof = c.proc->output_closed();
    }

    void read_stream(child_t & c, server_subproc_stream stream) {
        char chunk[4096];
        std::string & buf = c.buf[stream];
        bool closed = false;
        while (true) {
            int n = c.proc->read_output(stream, chunk, sizeof(chunk));
            if (n < 0) {
                closed = true;
                break;
            }
            if (n == 0) {
                break;
            }
            buf.append(chunk, (size_t) n);
            size_t start = 0;
            while (true) {
                size_t nl = buf.find('\n', start);
                if (nl == std::string::npos) {
                    break;
                }
                on_line(c, stream, buf.substr(start, nl + 1 - start));
                start = nl + 1;
            }
            buf.erase(0, start);
            if (buf.size() > max_line) {
                buf.clear(); // a child that never writes a newline must not grow this without bound
            }
        }
        if (closed && !buf.empty()) {
            on_line(c, stream, buf);
            buf.clear();
        }
    }

    void on_line(child_t & c, server_subproc_stream stream, const std::string & line) {
        if (stream == SERVER_SUBPROC_STDERR) {
            LOG("[%5d] %s", c.port, line.c_str()); // forward log
            if (!c.proc->oom_seen.load(std::memory_order_relaxed) && line_reports_oom(line)) {
                c.proc->oom_seen.store(true, std::memory_order_relaxed); // fork: load-OOM retry
            }
            c.proc->recent_output.push(line); // fork: last-lines fallback for a child that dies without an error-prefix line
            const std::string em = status_error_line(line);
            if (!em.empty()) {
                c.proc->append_error(em); // fork: surfaced in the load-failure error
            }
        } else if (string_starts_with(line, CMD_CHILD_TO_ROUTER_STATE)) {
            LOG_DBG("[%5d] %s", c.port, line.c_str()); // prevent spamming the log
            models.handle_child_state(c.name, line);
        } else {
            SRV_WRN("[%5d] unexpected output on the command pipe: %s", c.port, line.c_str());
        }
    }

    void run() {
        while (true) {
            if (handle_commands()) {
                return;
            }

            // wait for output, a wakeup, or the next deadline;
            // a child whose output closed is polled for its exit every 50 ms
            int64_t now     = ggml_time_ms();
            int64_t timeout = -1;
            for (const auto & c : children) {
                if (c.eof) {
                    timeout = timeout < 0 ? 50 : std::min<int64_t>(timeout, 50);
                }
                if (c.deadline) {
                    int64_t d = std::max<int64_t>(0, c.deadline - now);
                    timeout = timeout < 0 ? d : std::min(timeout, d);
                }
            }
            std::vector<server_subproc *> procs;
            std::vector<child_t *>        owners;
            for (auto & c : children) {
                if (!c.eof) {
                    procs.push_back(c.proc.get());
                    owners.push_back(&c);
                }
            }
            std::vector<bool> ready;
            waiter.wait(procs, ready, timeout);
            for (size_t i = 0; i < owners.size(); i++) {
                if (ready[i]) {
                    read_output(*owners[i]);
                }
            }

            // deadlines and exits
            now = ggml_time_ms();
            for (auto it = children.begin(); it != children.end();) {
                if (it->deadline && now >= it->deadline && !it->proc->stopped.load(std::memory_order_acquire)) {
                    SRV_WRN("force-killing model instance name=%s after timeout\n", it->name.c_str());
                    it->proc->terminate();
                    it->deadline = 0;
                }
                if (it->eof && !it->proc->is_alive()) {
                    int exit_code = it->proc->join();
                    it->proc->stopped.store(true, std::memory_order_release);
                    models.on_child_exit(it->name, it->proc, it->role, exit_code);
                    SRV_INF("instance name=%s exited with status %d\n", it->name.c_str(), exit_code);
                    it = children.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    static constexpr size_t max_line = 1024 * 1024;

    server_models & models;
    std::mutex mu;
    std::deque<cmd_t> cmds;
    std::vector<child_t> children; // monitor thread only
    server_subproc::waiter waiter;
    std::thread th;
};

struct server_lru_sched {
    server_lru_sched(server_models & models) : models(models) {}

    bool has_capacity(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        return models.base_params.models_max <= 0
            || count_running() < (size_t) models.base_params.models_max;
    }

    // returns "" if no model can be given up
    std::string pick_victim(std::unique_lock<std::mutex> & lk, const std::string & exclude = "") {
        check_lock(lk);
        std::string victim;
        int64_t victim_last_used = 0;
        for (const auto & m : models.mapping) {
            // a busy model is mid-request, one still coming up has no request to finish
            if (m.second.req_count != 0 || !m.second.meta.is_ready_or_sleep()) {
                continue;
            }
            // a request is about to use it, it is already on its way out, or a queued request wants it
            if (m.first == exclude || m.second.n_pending != 0 || m.second.exiting ||
                models.stopping_models.count(m.first) || find(m.first)) {
                continue;
            }
            if (victim.empty() || m.second.meta.last_used < victim_last_used) {
                victim           = m.first;
                victim_last_used = m.second.meta.last_used;
            }
        }
        return victim;
    }

    // requests wanting the same model share one entry, so they all need only one slot
    // and all get unblocked by the single load that entry performs
    void join(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (entry_t * e = find(model_id)) {
            e->n_waiters++;
            SRV_INF("request for name=%s joined the queue, %d waiting\n", model_id.c_str(), e->n_waiters);
            return;
        }
        queue.push_back({ model_id, 1, false });
        SRV_INF("request for name=%s queued at position %zu\n",
                model_id.c_str(), queue.size());
    }

    void leave(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (it->model_id == model_id) {
                if (--it->n_waiters <= 0) {
                    queue.erase(it); // last one waiting for this model went away
                }
                return;
            }
        }
    }

    bool queue_empty(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        return queue.empty();
    }

    // true if it is this model's turn to load, and nobody is loading it yet
    bool try_claim(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (queue.empty() || queue.front().model_id != model_id || queue.front().loading) {
            return false;
        }
        if (!has_capacity(lk)) {
            return false;
        }
        queue.front().loading = true;
        return true;
    }

    // on failure the entry is back in line; on success it stays until its waiters leave,
    // so the model coming up is never picked as a victim before they use it
    void claim_done(std::unique_lock<std::mutex> & lk, const std::string & model_id, bool ok) {
        check_lock(lk);
        if (ok) {
            return;
        }
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (it->model_id == model_id) {
                it->loading = false;
                return;
            }
        }
    }

    // evict idle models while queued requests outnumber the slots that are free or being freed
    // caller must hold models.mutex; never blocks, so it is safe from any thread
    void tick(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        if (models.base_params.models_max <= 0 || queue.empty()) {
            return;
        }
        int n_running  = 0;
        int n_stopping = 0;
        for (const auto & m : models.mapping) {
            if (m.second.meta.is_running()) {
                n_running++;
                if (models.stopping_models.count(m.first)) {
                    n_stopping++;
                }
            }
        }
        int n_needed  = 0;
        int n_claimed = 0; // claimed the slot, but load() has not spawned yet
        for (const auto & e : queue) {
            if (!e.loading) {
                n_needed++;
                continue;
            }
            auto it = models.mapping.find(e.model_id);
            if (it != models.mapping.end() && !it->second.meta.is_running()) {
                n_claimed++;
            }
        }
        int n_free = models.base_params.models_max - n_running + n_stopping - n_claimed;
        while (n_free < n_needed) {
            std::string victim = pick_victim(lk);
            if (victim.empty()) {
                return; // all remaining models are busy, wait for a request to end
            }
            SRV_INF("evicting idle LRU name=%s for a queued request\n", victim.c_str());
            models.request_stop(victim, true, SERVER_STOP_LRU_EVICT);
            n_free++;
        }
    }

  private:
    struct entry_t {
        std::string model_id;
        int  n_waiters; // requests waiting for this model
        bool loading;   // one of the waiters is doing the load right now
    };

    entry_t * find(const std::string & model_id) {
        for (auto & e : queue) {
            if (e.model_id == model_id) {
                return &e;
            }
        }
        return nullptr;
    }

    void check_lock(std::unique_lock<std::mutex> & lk) {
        GGML_ASSERT(lk.owns_lock() && lk.mutex() == &models.mutex);
    }

    size_t count_running() {
        size_t count = 0;
        for (const auto & m : models.mapping) {
            if (m.second.meta.is_running()) {
                count++;
            }
        }
        return count;
    }

    server_models & models;
    std::deque<entry_t> queue;
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.timeout_read/write which only applies to the generation proxy
static constexpr int STREAM_LOOKUP_TIMEOUT_MS = 250;

static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    // the router enforces --rate-limit; a child sees every request from the router's own address, so a limit there would
    // become one budget shared by all clients
    preset.unset_option("LLAMA_ARG_RATE_LIMIT");
    preset.unset_option("LLAMA_ARG_RATE_LIMIT_KEY");
    preset.unset_option("LLAMA_ARG_API_KEY_FILE");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_MEMORY_MARGIN");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    preset.unset_option("LLAMA_ARG_MODELS_KEEP_ALIVE");
    preset.unset_option("LLAMA_ARG_MODELS_SNAPSHOT"); // the router alone owns the snapshot file
    preset.unset_option("LLAMA_ARG_CONNECT");
    preset.unset_option("LLAMA_ARG_CONNECT_CODE");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_ALIAS");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wstring_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);
    preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port));
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", name);
    // the child output goes through the router to its terminal, so it follows the router colors
    preset.set_option(ctx_preset, "LLAMA_ARG_LOG_COLORS", common_log_get_colors(common_log_main()) ? "on" : "off");
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LLAMA_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps(const common_params & base) {
    // reset to the default so a failed refresh cannot keep old values
    architecture = server_model_architecture_json(false, false, false, {"text"});
    n_ctx_train  = 0;

    // resolve the model file offline; do not download
    common_params params;
    params.model = base.model;
    // --no-mmproj applies to child models and blocks auto-attached projectors
    params.no_mmproj = base.no_mmproj;
    try {
        preset.apply_to_params(params, {
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_MMPROJ_AUTO",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_FILE",
        });
        params.offline = true;
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        common_models_handler_apply(handler, params);
    } catch (const std::exception & e) {
        LOG_WRN("failed to resolve the model of '%s': %s\n", name.c_str(), e.what());
        return;
    }

    bool inp_image = false;
    bool inp_audio = false;
    bool gen_audio = false;
    try {
        if (!params.no_mmproj && !params.mmproj.path.empty()) {
            mtmd_caps caps = mtmd_get_cap_from_file(params.mmproj.path.c_str());
            inp_image = caps.inp_vision;
            inp_audio = caps.inp_audio;
            gen_audio = caps.gen_audio;
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to read the multimodal capabilities of '%s': %s\n", name.c_str(), e.what());
        // keep the output modalities from the GGUF metadata
    }

    // read the output modalities and the trained context from the GGUF metadata
    std::vector<std::string> output_modalities = {"text"};
    if (!params.model.path.empty()) {
        const common_gguf_info info = common_get_gguf_info(params.model.path);
        // same rule as mtmd_helper_get_capabilities()
        const bool can_chat = !gen_audio || info.has_chat_tmpl;
        output_modalities = server_model_output_modalities(info.decision_type, can_chat, gen_audio);
        n_ctx_train       = info.n_ctx_train;
    }

    // offline discovery cannot see video; a loaded model reports it
    architecture = server_model_architecture_json(inp_image, inp_audio, false, output_modalities);
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)),
              sched(std::make_unique<server_lru_sched>(*this)),
              monitor(std::make_unique<server_monitor>(*this)) {
    // propagate base params to child
    unset_reserved_args(base_preset, true);

    // do not propagate these options, but allow preset to explicitly set them
    base_preset.unset_option("LLAMA_ARG_LOG_FILE");

    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_cooldown.base_ms = params.models_load_cooldown_ms;
    load_cooldown.max_ms  = std::max<int64_t>(params.models_load_cooldown_max_ms, params.models_load_cooldown_ms);
    unfinished_downloads.load();
    load_models();
    debug_fake_timing = !common_get_env("LLAMA_SERVER_DEBUG_FAKE_TIMING").empty();
    // started last: a throw above must not leave a joinable thread behind
    keep_alive_th = std::thread([this]() { keep_alive_loop(); });
}

server_models::~server_models() {
    {
        std::lock_guard<std::mutex> lk(mutex);
        keep_alive_stop = true;
    }
    cv_keep_alive.notify_all();
    if (keep_alive_th.joinable()) {
        keep_alive_th.join();
    }
}

void server_models::request_stop_locked(const std::string & name, instance_t & inst, server_stop_reason reason) {
    SRV_INF("stopping model instance name=%s\n", name.c_str());
    const bool loading = inst.meta.status == SERVER_MODEL_STATUS_LOADING;
    if (loading) {
        // special case: if model is in loading state, unloading means force-killing it
        SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
        inst.subproc->terminate();
    }
    request_stop(name, !loading, reason); // marks it exiting; the monitor records the exit
}

void server_models::redo_deferred_reload_locked(instance_t & inst) {
    if (inst.reload_deferred && inst.req_count == 0 && inst.n_pending == 0) {
        inst.reload_deferred = false;
        need_reload = true; // the next get_meta()/get_all_meta() runs load_models() again, now with the model idle
    }
}

void server_models::keep_alive_loop() {
    std::unique_lock<std::mutex> lk(mutex);
    while (!keep_alive_stop) {
        const int64_t now  = ggml_time_ms();
        int64_t       next = 0;
        for (auto & [name, inst] : mapping) {
            if (inst.expire_at_ms <= 0) {
                continue;
            }
            if (!inst.meta.is_ready_or_sleep() || inst.exiting) {
                inst.expire_at_ms = 0; // went down some other way
                continue;
            }
            if (inst.req_count != 0 || inst.n_pending != 0) {
                continue; // in use again; the next idle reschedules it
            }
            if (now >= inst.expire_at_ms) {
                // decided and marked under one lock, so no request can slip in between
                SRV_INF("keep_alive of %" PRId64 " ms expired for model name=%s, unloading\n", inst.keep_alive_ms, name.c_str());
                request_stop_locked(name, inst, SERVER_STOP_IDLE_TTL);
            } else if (next == 0 || inst.expire_at_ms < next) {
                next = inst.expire_at_ms;
            }
        }
        if (next == 0) {
            cv_keep_alive.wait(lk);
        } else {
            cv_keep_alive.wait_for(lk, std::chrono::milliseconds(next - now));
        }
    }
}

server_models::pending_ref server_models::pending_acquire(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it == mapping.end()) {
        return pending_ref(); // unknown name, the handler reports it
    }
    it->second.n_pending++;
    return pending_ref(this, name);
}

void server_models::pending_release(const std::string & name) {
    bool wake = false;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        // the entry may have been replaced by a reload; never go negative
        if (it != mapping.end() && it->second.n_pending > 0) {
            it->second.n_pending--;
            // an expiry skipped while this request was pending is due again
            wake = it->second.n_pending == 0 && it->second.expire_at_ms > 0;
            redo_deferred_reload_locked(it->second);
        }
    }
    if (wake) {
        cv_keep_alive.notify_all();
    }
}

// Ollama keep_alive on a request that runs no inference (row S69-ENGREV-OLLAMA-SHIM-EMPTY-MESSAGES-LOAD-UNLOAD-2026-10-27-01).
// SOURCE-READ: tmp/ollload_1108/src/routes.go:480-491 (generate) and :3098-3107 (chat), ollama 3fbc609be4b3, MIT, no code copied.
// SOURCE-NUANCE: unload is `expireRunner(m)` -- a no-op when the model is not loaded -- and still answers done_reason "unload" with
// 200; the fork's equivalent is the keep_alive expiry keep_alive_loop() already applies, driven to "now".
int server_models::apply_keep_alive(const std::string & name, int64_t keep_alive_ms) {
    bool wake = false;
    int  rc   = 0;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end() || it->second.exiting || !it->second.meta.is_running()) {
            return -1;
        }
        instance_t & inst = it->second;
        inst.keep_alive_ms = keep_alive_ms; // the latest request decides, as proxy_request() does
        if (inst.req_count == 0) {
            // idle: the expiry starts now (a busy model gets it from proxy_request's cleanup when its last request ends)
            if (keep_alive_ms < 0) {
                inst.expire_at_ms = 0; // stay loaded
            } else if (keep_alive_ms == 0 && inst.n_pending == 0) {
                // decided and marked under one lock, so no request can slip in between (same as keep_alive_loop)
                SRV_INF("keep_alive 0 for model name=%s, unloading\n", name.c_str());
                request_stop_locked(name, inst, SERVER_STOP_IDLE_TTL);
                rc = 1;
            } else {
                inst.expire_at_ms = ggml_time_ms() + keep_alive_ms; // a pending request postpones it, pending_release() wakes the loop
                wake = true;
            }
        }
    }
    if (wake) {
        cv_keep_alive.notify_all();
    }
    return rc;
}

void server_models::instance_t::request_exit() const {
    request_child_exit(*subproc);
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    meta.update_args(ctx_preset, bin_path); // render args
    meta.update_caps(base_params);
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<server_subproc>(),
        /* meta    */ std::move(meta)
    };
}

static bool has_sidecar(const std::vector<std::string> & sidecars, const std::string & kind) {
    return std::find(sidecars.begin(), sidecars.end(), kind) != sidecars.end();
}

// reject a sidecar kind no download can pull
static void check_sidecar(const std::string & kind) {
    std::vector<std::string> kinds;
    for (const auto & s : common_download_sidecars()) {
        kinds.emplace_back(s.name);
    }
    if (!has_sidecar(kinds, kind)) {
        throw std::invalid_argument("unknown sidecar '" + kind + "', expected one of " + string_join(kinds, ", "));
    }
}

// the sidecars a download asks for; by default the ones -hf pulls without a draft type
static std::vector<std::string> parse_sidecars(const json & body) {
    std::vector<std::string> defaults;
    for (const auto & s : common_download_sidecars()) {
        if (!s.draft) {
            defaults.emplace_back(s.name);
        }
    }
    std::vector<std::string> sidecars = json_value(body, "sidecars", defaults);
    for (const auto & kind : sidecars) {
        check_sidecar(kind);
    }
    return sidecars;
}

// the download options of a model with its sidecars, the plan a removal purges partials from
static common_download_opts sidecar_download_opts(const common_params & params, const std::vector<std::string> & sidecars) {
    common_download_opts opts;
    opts.bearer_token = params.hf_token;
    for (const auto & kind : sidecars) {
        common_download_opts_add_sidecar(opts, kind);
    }
    return opts;
}

// the HF repo a model resolves from: a preset names its own, a cached model is
// listed under it, and a local file or a models dir entry has none
static std::string model_hf_repo(const server_model_meta & meta) {
    std::string hf_repo;
    if (meta.preset.get_option("LLAMA_ARG_HF_REPO", hf_repo) && !hf_repo.empty()) {
        return hf_repo;
    }
    return meta.source == SERVER_MODEL_SOURCE_CACHE ? meta.name : "";
}

// the sidecar kinds of a HF repo present on disk, as the offline plan resolves them
static json cached_sidecars(const std::string & hf_repo) {
    common_params_model model;
    model.hf_repo = hf_repo;

    common_download_opts opts;
    opts.offline = true;
    for (const auto & s : common_download_sidecars()) {
        opts.*s.opt = true;
    }

    return common_download_plan_sidecars(common_download_get_hf_plan(model, opts));
}

// replace any inherited value of `key` in a child environment
static void set_child_env(std::vector<std::string> & env, const std::string & key, const std::string & value) {
    env.erase(std::remove_if(env.begin(), env.end(), [&](const std::string & e) {
        return e.rfind(key + "=", 0) == 0;
    }), env.end());
    env.push_back(key + "=" + value);
}

// where the unfinished downloads are remembered, inside the model cache
static std::string unfinished_downloads_path() {
    return hf_cache::get_cache_path() + "/unfinished-downloads.json";
}

void server_unfinished_downloads::load() {
    std::ifstream in(unfinished_downloads_path());
    if (!in) {
        return;
    }
    std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::lock_guard<std::mutex> lk(mutex);
    try {
        json data = json::parse(raw);
        for (const auto & d : data.value("downloads", json::array())) {
            server_unfinished_download download;
            download.model    = json_value(d, "model", std::string());
            download.sidecars = json_value(d, "sidecars", std::vector<std::string>());
            for (const auto & p : d.value("partials", json::array())) {
                download.partials.push_back({ json_value(p, "kind", std::string()), json_value(p, "path", std::string()) });
            }
            if (!download.model.empty()) {
                downloads.push_back(std::move(download));
            }
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to read unfinished downloads: %s\n", e.what());
    }
}

void server_unfinished_downloads::save() {
    json data;
    data["downloads"] = json::array();
    for (const auto & d : downloads) {
        json partials = json::array();
        for (const auto & p : d.partials) {
            partials.push_back({ {"kind", p.kind}, {"path", p.path} });
        }
        data["downloads"].push_back({ {"model", d.model}, {"sidecars", d.sidecars}, {"partials", partials} });
    }
    try {
        fs_write_atomic(unfinished_downloads_path(), data.dump());
    } catch (const std::exception & e) {
        LOG_WRN("failed to record unfinished downloads: %s\n", e.what());
    }
}

void server_unfinished_downloads::delete_partials(const std::vector<common_download_partial> & parts) {
    for (const auto & p : parts) {
        const bool shared = std::any_of(downloads.begin(), downloads.end(), [&](const server_unfinished_download & d) {
            return std::any_of(d.partials.begin(), d.partials.end(), [&](const common_download_partial & q) {
                return q.path == p.path;
            });
        });
        if (shared) {
            continue;
        }
        std::error_code ec;
        std::filesystem::remove(p.path, ec);
    }
}

void server_unfinished_downloads::add(const server_unfinished_download & download) {
    std::lock_guard<std::mutex> lk(mutex);
    std::vector<common_download_partial> stale;
    auto it = std::find_if(downloads.begin(), downloads.end(), [&](const auto & d) { return d.model == download.model; });
    if (it != downloads.end()) {
        stale = std::move(it->partials);
        *it = download;
    } else {
        downloads.push_back(download);
    }
    save();
    delete_partials(stale);
}

void server_unfinished_downloads::forget(const std::string & model) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = std::find_if(downloads.begin(), downloads.end(), [&](const auto & d) { return d.model == model; });
    if (it != downloads.end()) {
        downloads.erase(it);
        save();
    }
}

bool server_unfinished_downloads::remove(const std::string & model) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = std::find_if(downloads.begin(), downloads.end(), [&](const auto & d) { return d.model == model; });
    if (it == downloads.end()) {
        return false;
    }
    auto partials = std::move(it->partials);
    downloads.erase(it);
    save();
    delete_partials(partials);
    return true;
}

bool server_unfinished_downloads::remove_sidecar(const std::string & model, const std::string & kind) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = std::find_if(downloads.begin(), downloads.end(), [&](const auto & d) { return d.model == model; });
    if (it == downloads.end() || !has_sidecar(it->sidecars, kind)) {
        return false;
    }
    it->sidecars.erase(std::remove(it->sidecars.begin(), it->sidecars.end(), kind), it->sidecars.end());

    std::vector<common_download_partial> dropped;
    auto & partials = it->partials;
    auto pos = std::stable_partition(partials.begin(), partials.end(), [&](const common_download_partial & p) {
        return p.kind != kind;
    });
    dropped.assign(pos, partials.end());
    partials.erase(pos, partials.end());

    save();
    delete_partials(dropped);
    return true;
}

std::vector<server_unfinished_download> server_unfinished_downloads::get_all() {
    std::lock_guard<std::mutex> lk(mutex);
    return downloads;
}

std::vector<server_unfinished_download> server_models::unfinished_downloads_get() {
    return unfinished_downloads.get_all();
}

void server_models::unfinished_downloads_remove(const std::string & name) {
    // under the lock, so a download of the same tag cannot start while its files go
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        throw std::invalid_argument("model '" + name + "' is downloading");
    }
    // a name with no record is no model: its files belong to the cached tags of the repo
    if (!unfinished_downloads.remove(name)) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    bool ok = common_download_remove(name);
    SRV_INF("removing unfinished download name=%s (%s)\n", name.c_str(), ok ? "succeeded" : "nothing cached");
}

bool server_models::remove_sidecar(const std::string & name, const std::string & kind) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end() && (it->second.meta.is_running() || it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING)) {
        throw std::invalid_argument("model '" + name + "' is in use, unload it first");
    }
    // a sidecar that only ever landed as a partial is still something removed
    bool ok = unfinished_downloads.remove_sidecar(name, kind);
    ok = common_download_remove_sidecars(name, sidecar_download_opts(base_params, { kind })) || ok;
    SRV_INF("removing sidecar %s of model name=%s (%s)\n", kind.c_str(), name.c_str(), ok ? "succeeded" : "nothing cached");
    return ok;
}

void server_models::notify_sse(const std::string & event, const std::string & model_id, const json & data) {
    std::unique_ptr<server_task_result_router> result = std::make_unique<server_task_result_router>();
    result->data = {
        {"model", model_id},
        {"event", event},
    };
    if (!data.is_null()) {
        result->data["data"] = data;
    }
    SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", event.c_str(), model_id.c_str(), safe_json_to_str(result->data).c_str());
    sse.broadcast(std::move(result));
}

void server_models::load_models() {
    // Phase 1: load presets from all sources - pure I/O, no lock needed
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_TRC("Loaded %zu cached model presets from %s\n", cached_models.size(), hf_cache::get_cache_path().c_str());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_TRC("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_TRC("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    std::unordered_map<std::string, server_model_source> source_map;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_CACHE;
    }
    for (const auto & [name, preset] : local_models)  {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_MODELS_DIR;
    }
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);
        } else {
            final_presets[name] = custom;
        }
        source_map[name] = SERVER_MODEL_SOURCE_PRESET;
    }

    // overlay router's own CLI args on top of every model preset so that
    // e.g. `llama-server --temp 0` is honoured by all child processes
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    auto get_source = [&](const std::string & name) {
        return source_map.count(name) ? source_map.at(name) : SERVER_MODEL_SOURCE_PRESET;
    };

    // hide cache models whose resolved file is already used by a preset with dedup-cache-models enabled
    std::set<std::string> hidden_models;
    {
        std::set<std::string> preset_paths;
        auto add_hf_path = [&preset_paths](const common_preset & preset, const char * repo_key, const char * file_key) {
            std::string hf_repo;
            if (!preset.get_option(repo_key, hf_repo) || hf_repo.empty()) {
                return;
            }
            std::string hf_file;
            preset.get_option(file_key, hf_file);
            std::string path = common_download_resolve_path(hf_repo, hf_file);
            if (!path.empty()) {
                preset_paths.insert(path);
            }
        };
        for (const auto & [name, preset] : custom_presets) {
            std::string val;
            if (!preset.get_option(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS, val) || !common_arg_utils::is_truthy(val)) {
                continue;
            }
            add_hf_path(preset, "LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_FILE");
            add_hf_path(preset, "LLAMA_ARG_SPEC_DRAFT_HF_REPO", "LLAMA_ARG_SPEC_DRAFT_MODEL");
        }
        if (!preset_paths.empty()) {
            for (const auto & [name, preset] : cached_models) {
                if (get_source(name) != SERVER_MODEL_SOURCE_CACHE) {
                    continue; // merged with another source, not a pure cache entry
                }
                std::string path = common_download_resolve_path(name);
                if (!path.empty() && preset_paths.count(path)) {
                    SRV_INF("hiding cache model name=%s (deduplicated by a preset)\n", name.c_str());
                    hidden_models.insert(name);
                }
            }
        }
    }

    // Helpers that read `mapping` - must be called while holding the lock.
    auto join_set = [](const std::set<std::string> & s) {
        std::string result;
        for (const auto & v : s) {
            if (!result.empty()) result += ", ";
            result += v;
        }
        return result;
    };
    auto log_available_models = [&]() {
        SRV_INF("Available models (%zu):\n", mapping.size());
        if (mapping.empty()) {
            SRV_INF("%s", "  no models found on the system (visit https://llama.app/models for suggestions)\n");
        } else {
            for (const auto & [name, inst] : mapping) {
                const std::string source = server_model_source_to_string(inst.meta.source);

                std::string info;
                if (!inst.meta.aliases.empty()) info += " (aliases: " + join_set(inst.meta.aliases) + ")";
                if (!inst.meta.tags.empty())    info += " [tags: "    + join_set(inst.meta.tags)    + "]";

                SRV_INF("  [%10s] %s%s\n", source.c_str(), name.c_str(), info.c_str());
            }
        }
    };
    auto apply_stop_timeout = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
                try {
                    inst.meta.stop_timeout = std::stoi(val);
                } catch (...) {
                    SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                        val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                    inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
                }
            }
        }
    };
    auto apply_hidden = [&]() {
        for (auto & [name, inst] : mapping) {
            inst.meta.hidden = hidden_models.count(name) > 0;
        }
    };
    // update_args() injects HOST/PORT/ALIAS/LOG_COLORS, so strip them before comparing presets
    auto preset_options_for_compare = [](common_preset p) {
        p.unset_option("LLAMA_ARG_HOST");
        p.unset_option("LLAMA_ARG_PORT");
        p.unset_option("LLAMA_ARG_ALIAS");
        p.unset_option("LLAMA_ARG_LOG_COLORS");
        return p.options;
    };

    // Phase 2: acquire the lock once for all mapping mutations.
    // We temporarily release it only when calling functions that acquire it internally (unload)
    std::unique_lock<std::mutex> lk(mutex);

    need_reload = false;
    bool is_first_load = mapping.empty();

    if (is_first_load) {
        // FIRST LOAD: add all models, then unlock for autoloading
        for (const auto & [name, preset] : final_presets) {
            server_model_meta meta{
                /* source        */ get_source(name),
                /* preset        */ preset,
                /* name          */ name,
                /* aliases       */ {},
                /* tags          */ {},
                /* port          */ 0,
                /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used     */ 0,
                /* args          */ std::vector<std::string>(),
                /* loaded_info   */ {},
                /* progress      */ {},
                /* exit_code     */ 0,
                /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                // /* need_download */ false,
            };
            add_model(std::move(meta));
        }
        apply_stop_timeout();
        apply_hidden();
        log_available_models();

        // skipped on reload, see startup_models
        if (startup_models.has_value()) {
            std::vector<std::string> models_to_load;
            for (const auto & [name, inst] : mapping) {
                std::string val;
                if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                    models_to_load.push_back(name);
                }
            }
            if ((int)models_to_load.size() > base_params.models_max) {
                throw std::runtime_error(string_format(
                    "number of models to load on startup (%zu) exceeds models_max (%d)",
                    models_to_load.size(), base_params.models_max));
            }

            // --models-snapshot: add the models that were loaded when the router last ran (TorchServe snapshot)
            if (!base_params.models_snapshot.empty()) {
                std::vector<std::string> snap;
                std::string err;
                if (!server_models_snapshot_read(base_params.models_snapshot, snap, err)) {
                    SRV_WRN("models snapshot %s not restored: %s\n", base_params.models_snapshot.c_str(), err.c_str());
                } else {
                    std::set<std::string> known;
                    std::set<std::string> explicit_names;
                    for (const auto & [name, inst] : mapping) {
                        known.insert(name);
                        std::string val;
                        if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val)) {
                            explicit_names.insert(name);
                        }
                    }
                    auto plan = server_models_snapshot_merge(models_to_load, snap, known, explicit_names, base_params.models_max);
                    for (const auto & n : plan.restored) {
                        SRV_INF("models snapshot: restoring %s\n", n.c_str());
                    }
                    for (const auto & n : plan.skipped_explicit) {
                        SRV_INF("models snapshot: %s skipped, its preset sets %s and wins\n", n.c_str(), COMMON_ARG_PRESET_LOAD_ON_STARTUP);
                    }
                    for (const auto & n : plan.skipped_unknown) {
                        SRV_WRN("models snapshot: %s skipped, no such model is configured now\n", n.c_str());
                    }
                    for (const auto & n : plan.skipped_cap) {
                        SRV_WRN("models snapshot: %s skipped, --models-max %d reached\n", n.c_str(), base_params.models_max);
                    }
                    models_to_load = std::move(plan.load);
                }
            }

            // to be lazy-loaded after main() setup phase is completed
            startup_models = std::move(models_to_load);
        }

        lk.unlock();
    } else {
        // RELOAD: diff the new preset list against the current mapping and reconcile
        {
            // an empty new set (bad edit, unreadable source) would unload every running model: keep the old set
            size_t n_running = 0;
            for (const auto & [name, inst] : mapping) {
                n_running += inst.meta.is_running() ? 1 : 0;
            }
            if (server_models_reload_refuse_empty(final_presets.size(), n_running, base_params.models_reload_allow_empty)) {
                SRV_WRN("(reload) new preset set is empty, refusing to unload %zu running model(s); keeping the current set (--reload-allow-empty to allow)\n", n_running);
                return;
            }
        }
        is_reloading = true;

        // find running models whose source was removed or whose preset changed
        std::vector<std::string> to_unload;
        for (auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            const bool removed = it == final_presets.end();
            if (!removed && preset_options_for_compare(inst.meta.preset) == preset_options_for_compare(it->second)) {
                continue; // unchanged
            }
            if (inst.req_count != 0 || inst.n_pending != 0) {
                // busy: stopping it now would cut the request mid-stream. Keep it (old preset) and redo the reload when it
                // goes idle, see redo_deferred_reload_locked(). Jan's registry defers while an engine lease is held.
                SRV_WRN("(reload) model name=%s is serving %d request(s) (+%d pending), unloading deferred until it is idle\n",
                    name.c_str(), inst.req_count, inst.n_pending);
                inst.reload_deferred = true;
                continue;
            }
            inst.reload_deferred = false;
            to_unload.push_back(name); // removed from source, or preset changed
        }

        // unload() acquires the lock internally, so release before each call
        for (const auto & name : to_unload) {
            SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
            lk.unlock();
            unload(name);
            lk.lock();
        }

        // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
        cv.wait(lk, [&]() {
            for (const auto & name : to_unload) {
                auto it = mapping.find(name);
                if (it != mapping.end() && it->second.meta.is_running()) return false;
            }
            return true;
        });

        // erase models no longer in any source
        for (auto it = mapping.begin(); it != mapping.end(); ) {
            if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                ++it; // download thread is still busy, skip
            } else if (it->second.reload_deferred && it->second.meta.is_running()) {
                ++it; // removed from source while serving a request: erased by the reload that runs once it is idle
            } else if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // download finished, safe to erase
                it = mapping.erase(it);
            } else if (final_presets.find(it->first) == final_presets.end()) {
                SRV_INF("(reload) removing model name=%s (no longer in source)\n", it->first.c_str());
                it = mapping.erase(it);
            } else {
                ++it;
            }
        }

        // update presets for non-running models still in source
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) continue; // erased above

            inst.meta.preset = it->second;

            // re-parse aliases, then validate against other models
            std::set<std::string> new_aliases;
            std::string alias_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
                for (auto & alias : string_split<std::string>(alias_str, ',')) {
                    alias = string_strip(alias);
                    if (!alias.empty()) new_aliases.insert(alias);
                }
            }
            inst.meta.aliases.clear();
            for (const auto & alias : new_aliases) {
                bool conflict = false;
                for (const auto & [other_name, other_inst] : mapping) {
                    if (other_name == name) continue;
                    if (other_name == alias || other_inst.meta.aliases.count(alias)) {
                        SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n",
                            alias.c_str(), name.c_str(), other_name.c_str());
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) inst.meta.aliases.insert(alias);
            }

            // re-parse tags
            inst.meta.tags.clear();
            std::string tags_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
                for (auto & tag : string_split<std::string>(tags_str, ',')) {
                    tag = string_strip(tag);
                    if (!tag.empty()) inst.meta.tags.insert(tag);
                }
            }

            inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
            inst.meta.update_args(ctx_preset, bin_path);
            inst.meta.update_caps(base_params);
        }

        // add models that are new in this reload, load-on-startup is not honored here since a
        // reload never spawns an instance
        for (const auto & [name, preset] : final_presets) {
            if (mapping.find(name) == mapping.end()) {
                server_model_meta meta{
                    /* source        */ get_source(name),
                    /* preset        */ preset,
                    /* name          */ name,
                    /* aliases       */ {},
                    /* tags          */ {},
                    /* port          */ 0,
                    /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                    /* last_used     */ 0,
                    /* args          */ std::vector<std::string>(),
                    /* loaded_info   */ {},
                    /* progress      */ {},
                    /* exit_code     */ 0,
                    /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                    // /* need_download */ false,
                };
                add_model(std::move(meta));
            }
        }

        apply_stop_timeout();
        apply_hidden();

        // clear reload flag under the lock, this releases the load() calls waiting on !is_reloading
        is_reloading = false;
        cv.notify_all();

        log_available_models();

        lk.unlock();

        notify_sse("models_reload", "*");
    }
}

void server_models::load_startup_models() {
    std::vector<std::string> to_load;
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (!startup_models.has_value()) {
            return; // already drained
        }
        to_load = std::move(*startup_models);
        startup_models.reset();
    }
    for (const auto & name : to_load) {
        SRV_INF("(startup) loading model %s\n", name.c_str());
        load(name);
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
    }
    return std::nullopt;
}

std::vector<std::string> server_models::failover_chain(const std::string & name) {
    std::string list;
    if (const auto meta = get_meta(name); meta.has_value()) {
        meta->preset.get_option("LLAMA_ARG_FALLBACK_MODEL", list); // the model's own section, or the router-wide option merged into it
    }
    if (list.empty()) {
        list = base_params.fallback_model;
    }
    if (list.empty()) {
        return {name};
    }
    return server_failover_chain(name, list, [this](const std::string & n) {
        const auto m = get_meta(n);
        return m.has_value() ? m->name : std::string(); // an alias resolves to the model name, an unknown name is dropped
    }, base_params.failover_max_attempts);
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

//
// VRAM-aware admission (--models-vram-admit), reimplemented from ollama's scheduler
// (server/sched.go: load() pre-flight, updateFreeSpace, findRunnerToUnload; MIT, Go -- not copied).
// models_max bounds the NUMBER of children; this bounds their device-memory total, which is what
// actually fails. CAPABILITY-GENUINELY-NA: process/GPU accounting in the router, not a model capability.
//

// bytes of a GGUF on disk; a split model (name-00001-of-0000N.gguf) counts every part
static int64_t gguf_size_on_disk(const std::string & path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (path.empty() || !fs::exists(path, ec)) {
        return 0;
    }
    const std::string fname = fs::path(path).filename().string();
    const size_t pos = fname.find("-00001-of-");
    if (pos == std::string::npos) {
        const auto sz = fs::file_size(path, ec);
        return ec ? 0 : (int64_t) sz;
    }
    const std::string stem = fname.substr(0, pos + 1); // "name-"
    int64_t total = 0;
    for (const auto & e : fs::directory_iterator(fs::path(path).parent_path(), ec)) {
        const std::string n = e.path().filename().string();
        if (n.rfind(stem, 0) == 0 && n.find("-of-") != std::string::npos) {
            const auto sz = fs::file_size(e.path(), ec);
            total += ec ? 0 : (int64_t) sz;
        }
    }
    return total;
}

// the model (and projector, "" when disabled) files a preset resolves to, offline -- no download
static void resolve_model_files(const server_model_meta & meta, std::string & model_path, std::string & mmproj_path) {
    common_params params;
    meta.preset.apply_to_params(params, { "LLAMA_ARG_MODEL", "LLAMA_ARG_MMPROJ", "LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_REPO_FILE" });
    params.offline = true;
    common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
    common_models_handler_apply(handler, params);
    model_path  = params.model.path;
    mmproj_path = params.no_mmproj ? std::string() : params.mmproj.path;
}

// What a GGUF header says about a model, read WITHOUT loading it (metadata only, no tensor data, no backend).
// SOURCE-READ: gpt4all-backend/src/llamamodel.cpp:719-758 (maxContextLength / layerCount / chatTemplate probe the GGUF
// key-values directly; MIT). Written fresh against ggml's gguf API, no code copied.
// SOURCE-NUANCE: the probe must be keyed by the file's architecture prefix (<arch>.block_count), and a model with no
// chat template is a normal answer, not an error -- hence has_chat_template instead of an empty string that could also mean
// "the key was present and empty".
struct gguf_light_info_t {
    bool        ok = false;
    std::string arch;
    std::string name;
    int64_t     context_length   = 0;
    int64_t     block_count      = 0;
    int64_t     embedding_length = 0;
    int64_t     expert_count     = 0;
    bool        has_chat_template = false;
    std::string chat_template;
    int64_t     file_bytes = 0;
};

static int64_t gguf_int_kv(const gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        return 0;
    }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(g, id);
        default: return 0;
    }
}

static gguf_light_info_t gguf_light_info(const std::string & path) {
    gguf_light_info_t info;
    if (path.empty()) {
        return info;
    }
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        return info;
    }
    const int64_t arch_id = gguf_find_key(g, "general.architecture");
    if (arch_id >= 0 && gguf_get_kv_type(g, arch_id) == GGUF_TYPE_STRING) {
        info.arch = gguf_get_val_str(g, arch_id);
        info.context_length   = gguf_int_kv(g, info.arch + ".context_length");
        info.block_count      = gguf_int_kv(g, info.arch + ".block_count");
        info.embedding_length = gguf_int_kv(g, info.arch + ".embedding_length");
        info.expert_count     = gguf_int_kv(g, info.arch + ".expert_count");
    }
    const int64_t name_id = gguf_find_key(g, "general.name");
    if (name_id >= 0 && gguf_get_kv_type(g, name_id) == GGUF_TYPE_STRING) {
        info.name = gguf_get_val_str(g, name_id);
    }
    const int64_t tmpl_id = gguf_find_key(g, "tokenizer.chat_template");
    if (tmpl_id >= 0 && gguf_get_kv_type(g, tmpl_id) == GGUF_TYPE_STRING) {
        info.has_chat_template = true;
        info.chat_template     = gguf_get_val_str(g, tmpl_id);
    }
    gguf_free(g);
    info.file_bytes = gguf_size_on_disk(path);
    info.ok = true;
    return info;
}

// the model listing asks for every model on every refresh: keep the answer until the file changes
static gguf_light_info_t gguf_light_info_cached(const std::string & path) {
    namespace fs = std::filesystem;
    static std::mutex mu;
    static std::map<std::string, std::pair<std::pair<int64_t, int64_t>, gguf_light_info_t>> cache;
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) {
        return gguf_light_info_t();
    }
    const auto mtime = (int64_t) fs::last_write_time(path, ec).time_since_epoch().count();
    const std::pair<int64_t, int64_t> stamp = { (int64_t) size, mtime };
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(path);
        if (it != cache.end() && it->second.first == stamp) {
            return it->second.second;
        }
    }
    gguf_light_info_t info = gguf_light_info(path);
    std::lock_guard<std::mutex> lk(mu);
    cache[path] = { stamp, info };
    return info;
}

static json gguf_light_info_to_json(const gguf_light_info_t & i, bool with_template) {
    json j {
        {"architecture",        i.arch},
        {"name",                i.name},
        {"context_length",      i.context_length},
        {"block_count",         i.block_count},
        {"embedding_length",    i.embedding_length},
        {"expert_count",        i.expert_count},
        {"has_chat_template",   i.has_chat_template},
        {"chat_template_bytes", (int64_t) i.chat_template.size()},
        {"file_bytes",          i.file_bytes},
    };
    if (with_template && i.has_chat_template) {
        j["chat_template"] = i.chat_template;
    }
    return j;
}

// <arch>.context_length from the GGUF header (metadata only, no tensor data); 0 if absent
static int64_t gguf_train_ctx(const std::string & path) {
    return gguf_light_info(path).context_length;
}

// Pre-load memory-fit estimate: what a model's preset would need on each device and on the host, WITHOUT spawning a child.
// SOURCE-READ: lmstudio-ai/lms src/subcommands/load.ts:302-307 and :768-799 (--estimate-only, printEstimatedResourceUsage) and
// get.ts:541-559 (createArtifactDownloadOptionTag fit tags), MIT, described in the row; the estimate itself is OUR common/fit.cpp
// (common_get_device_memory_data loads the model with no_alloc, so only the header and the tensor table are read).
// SOURCE-NUANCE: (1) the tags depend on what is free NOW, so the answer is only as good as the free-memory reading and says so in
// "confidence"; (2) the same preset can be loaded with --fit, which lowers context or offload to fit, so with fit on the numbers
// reported are the FITTED ones and the as-requested outcome is reported next to them; (3) an unset context is not "the model's
// maximum" here: 4096 is assumed and flagged, because the maximum of a long-context model needs more memory than any card has.
// Fit measurement swaps the global llama log callback, so one estimate runs at a time.
static json estimate_model_memory(const server_model_meta & meta, int32_t n_ctx_override, int64_t sim_vram_free) {
    static std::mutex estimate_mutex;
    std::lock_guard<std::mutex> lk(estimate_mutex);

    common_params params;
    meta.preset.apply_to_params(params);
    params.offline = true;
    common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
    common_models_handler_apply(handler, params); // offline: resolves paths, never downloads
    if (params.model.path.empty() || gguf_size_on_disk(params.model.path) <= 0) {
        throw std::runtime_error("the model file is not on disk (not downloaded yet?), nothing to estimate");
    }

    bool n_ctx_assumed = false;
    if (n_ctx_override > 0) {
        params.n_ctx = n_ctx_override;
    }
    if (params.n_ctx == 0) {
        params.n_ctx  = 4096;
        n_ctx_assumed = true;
    }

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0;
    auto before = common_get_device_memory_data(params.model.path.c_str(), &mparams, &cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

    json fit_info = json::object();
    auto dmd = before;
    if (params.fit_params) {
        const common_params_fit_status st = common_fit_params(params.model.path.c_str(), &mparams, &cparams,
                params.tensor_split, params.tensor_buft_overrides.data(), params.fit_params_target.data(), params.fit_params_min_ctx,
                nullptr, GGML_LOG_LEVEL_ERROR);
        fit_info = json {
            {"enabled", true},
            {"status",  st == COMMON_PARAMS_FIT_STATUS_SUCCESS ? "success" : (st == COMMON_PARAMS_FIT_STATUS_FAILURE ? "failure" : "error")},
            {"n_ctx",   (int64_t) cparams.n_ctx},
            {"n_gpu_layers", (int64_t) mparams.n_gpu_layers},
        };
        if (st == COMMON_PARAMS_FIT_STATUS_SUCCESS) {
            std::vector<ggml_backend_dev_t> devs2;
            dmd = common_get_device_memory_data(params.model.path.c_str(), &mparams, &cparams, devs2, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
        }
    } else {
        fit_info = json {{"enabled", false}};
    }

    // free memory readings: the test hooks replace them so the tags can be checked on any machine
    int64_t host_free = dmd.back().free;
    if (const char * sim = std::getenv("LLAMA_SERVER_HOST_FREE_MB")) {
        host_free = (int64_t) std::atoll(sim) << 20;
    }

    auto need_of = [](const common_device_memory_data & d) { return (int64_t) (d.model + d.context + d.compute); };
    auto row = [&](const common_device_memory_data & d, int64_t free_b, bool fits) {
        return json {
            {"total_bytes",   d.total},
            {"free_bytes",    free_b},
            {"model_bytes",   (int64_t) d.model},
            {"context_bytes", (int64_t) d.context},
            {"compute_bytes", (int64_t) d.compute},
            {"needed_bytes",  need_of(d)},
            {"fits",          fits},
        };
    };

    json devices = json::array();
    bool all_dev_fit = true;
    bool free_unknown = false;
    int64_t gpu_model_bytes = 0;
    for (size_t i = 0; i < devs.size() && i + 1 < dmd.size(); i++) {
        int64_t free_b = dmd[i].free;
        if (sim_vram_free >= 0 && i == 0) {
            free_b = sim_vram_free;
        }
        const bool unreported = dmd[i].free == 0 && dmd[i].total == 0 && !(sim_vram_free >= 0 && i == 0);
        const bool fits_i = unreported || need_of(dmd[i]) <= free_b; // unreported = unknown, assumed to fit and flagged below
        if (unreported) {
            free_unknown = true;
        } else if (!fits_i) {
            all_dev_fit = false;
        }
        json r = row(dmd[i], free_b, fits_i);
        r["name"] = std::string(ggml_backend_dev_name(devs[i]));
        gpu_model_bytes += (int64_t) dmd[i].model;
        devices.push_back(r);
    }
    const bool host_fits = need_of(dmd.back()) <= host_free;
    json host = row(dmd.back(), host_free, host_fits);
    const int64_t host_model_bytes = (int64_t) dmd.back().model;
    const int64_t all_model_bytes  = gpu_model_bytes + host_model_bytes;

    std::string tag;
    if (!host_fits || !all_dev_fit) {
        tag = "will_not_fit";
    } else if (devs.empty() || gpu_model_bytes == 0) {
        tag = "cpu_only";
    } else if (host_model_bytes * 50 > all_model_bytes) {
        tag = "partial_gpu_offload"; // more than 2% of the weights stay on the host
    } else {
        tag = "full_gpu_offload";
    }

    std::string confidence = "free memory read from the backends";
    if (free_unknown) {
        confidence = "low: a device did not report its free memory, it is assumed to fit";
    }
    if (sim_vram_free >= 0 || std::getenv("LLAMA_SERVER_HOST_FREE_MB")) {
        confidence = "simulated free memory (test hook)";
    }

    std::string mmproj_note;
    int64_t mmproj_bytes = 0;
    if (!params.no_mmproj && !params.mmproj.path.empty()) {
        mmproj_bytes = gguf_size_on_disk(params.mmproj.path);
        mmproj_note  = "mmproj_bytes is the projector file size on disk; it is not part of the device figures above";
    }

    return json {
        {"model",           meta.name},
        {"path",            params.model.path},
        {"fit",             tag},
        {"confidence",      confidence},
        {"n_ctx",           (int64_t) cparams.n_ctx},
        {"n_ctx_assumed",   n_ctx_assumed},
        {"n_ctx_train",     (int64_t) hp_nct},
        {"n_layer",         (int64_t) hp_ngl},
        {"n_gpu_layers",    (int64_t) mparams.n_gpu_layers},
        {"fit_params",      fit_info},
        {"devices",         devices},
        {"host",            host},
        {"mmproj_bytes",    mmproj_bytes},
        {"note",            mmproj_note},
    };
}

// a child log line that reports an allocation failure -- the backends' own wording; the rules live in server-oom-line.h
static bool line_reports_oom(const std::string & line) {
    return server_line_reports_oom(line);
}

// ollama nextLowerAutoNumCtx: >32k -> 32k, >4k -> 4k, else nothing lower to try
static int32_t next_lower_auto_ctx(int64_t n_ctx) {
    if (n_ctx > 32768) {
        return 32768;
    }
    if (n_ctx > 4096) {
        return 4096;
    }
    return 0;
}

// RESEARCH-DONE: ollama server/sched.go read in full 2026-09-26
// (https://raw.githubusercontent.com/ollama/ollama/main/server/sched.go, lines 300-360, 630-790, 922-931;
// saved as temp/research/src_2026-09-26/ollama_sched_go.txt). MIT, Go: reimplemented, not copied.
bool server_models::retry_after_load_oom(const std::string & name) {
    server_model_meta meta;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end()) {
            return false;
        }
        meta = it->second.meta;
    }

    // context: only when the user left it automatic (ollama retries only numCtxAuto). An explicit
    // -c is a request, and silently loading with less would hide the failure behind a smaller window.
    int32_t ctx_override = 0;
    std::string ctx_val;
    const bool ctx_auto = !meta.preset.get_option("LLAMA_ARG_CTX_SIZE", ctx_val) || ctx_val.empty() || ctx_val == "0";
    if (ctx_auto) {
        try {
            std::string model_path, mmproj_path;
            resolve_model_files(meta, model_path, mmproj_path);
            ctx_override = next_lower_auto_ctx(gguf_train_ctx(model_path));
        } catch (const std::exception & e) {
            SRV_WRN("load-OOM retry: could not read the training context of name=%s (%s)\n", name.c_str(), e.what());
        }
    }

    // evict every idle model (ollama evictAllAndWait); busy ones keep serving their requests
    std::vector<std::string> evicted;
    {
        std::unique_lock<std::mutex> lk(mutex);
        for (std::string victim; !(victim = sched->pick_victim(lk, name)).empty(); ) {
            request_stop_locked(victim, mapping[victim], SERVER_STOP_LRU_EVICT);
            evicted.push_back(victim);
        }
        cv.wait(lk, [this, &evicted]() {
            for (const auto & v : evicted) {
                if (mapping[v].meta.status != SERVER_MODEL_STATUS_UNLOADED) {
                    return false;
                }
            }
            return true;
        });
    }

    if (evicted.empty() && ctx_override == 0) {
        // same outcome as ollama: nothing would differ on a second attempt, so the error goes out
        SRV_WRN("load-OOM retry: name=%s -- no idle model to evict and no automatic context to lower; reporting the failure\n", name.c_str());
        return false;
    }
    std::string list;
    for (const auto & v : evicted) {
        list += (list.empty() ? "" : ",") + v;
    }
    SRV_WRN("load-OOM retry: name=%s evicted %zu idle model(s) [%s], context %s -- retrying once\n",
            name.c_str(), evicted.size(), list.c_str(),
            ctx_override > 0 ? std::to_string(ctx_override).c_str() : "unchanged");

    load_options opts;
    opts.ctx_override = ctx_override;
    load(name, opts);
    return true;
}

int64_t server_models::vram_free_locked() {
    // LLAMA_SERVER_VRAM_TOTAL_MB: test hook. Free device memory is simulated as that total minus
    // every running child's prediction, so admission and eviction order can be tested
    // deterministically with small models, independent of anything else on the GPU.
    if (const char * sim = std::getenv("LLAMA_SERVER_VRAM_TOTAL_MB")) {
        int64_t free_b = (int64_t) std::atoll(sim) << 20;
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                free_b -= m.second.vram_predicted;
            }
        }
        return std::max<int64_t>(free_b, 0);
    }
    return server_vram_free_bytes();
}

int64_t server_models::ensure_vram_for(const std::string & name) {
    // --models-vram-admit F: need <= F x free. --models-memory-margin N (ggml-org/llama.cpp #21231, 85cc432e2): MiB of
    // device memory that must stay free after the load, need + N <= free. Either switch turns admission on; both must hold.
    const int64_t margin_b = (int64_t) std::max(base_params.models_memory_margin, 0) << 20;
    if (base_params.models_vram_admit <= 0.0f && margin_b <= 0) {
        return 0;
    }
    // prediction: the footprint measured at this model's last load, else weights (+ projector) on disk
    server_model_meta meta;
    int64_t need = 0;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end()) {
            return 0;
        }
        meta = it->second.meta;
        if (auto m = vram_measured.find(name); m != vram_measured.end()) {
            need = m->second;
        }
    }
    if (need <= 0) {
        try {
            std::string model_path, mmproj_path;
            resolve_model_files(meta, model_path, mmproj_path);
            need = gguf_size_on_disk(model_path) + gguf_size_on_disk(mmproj_path);
        } catch (const std::exception & e) {
            SRV_WRN("vram admission: could not resolve the model file of name=%s (%s), admitting without a check\n", name.c_str(), e.what());
            return 0;
        }
    }
    if (need <= 0) {
        return 0;
    }

    for (int round = 0; round < 64; ++round) {
        std::string victim;
        int64_t victim_bytes = 0;
        int64_t free_b = 0;
        {
            std::unique_lock<std::mutex> lk(mutex);
            free_b = vram_free_locked();
            if (free_b < 0) {
                return need; // no reading: admit
            }
            const bool fits_frac   = base_params.models_vram_admit <= 0.0f || (double) need <= (double) base_params.models_vram_admit * (double) free_b;
            const bool fits_margin = margin_b <= 0 || need + margin_b <= free_b;
            if (fits_frac && fits_margin) {
                SRV_INF("vram admission: name=%s needs %.2f GiB, %.2f GiB free (x%.2f, margin %d MiB) -- admitted\n",
                        name.c_str(), need / 1073741824.0, free_b / 1073741824.0, base_params.models_vram_admit, base_params.models_memory_margin);
                return need;
            }
            victim = sched->pick_victim(lk, name);
            if (victim.empty()) {
                SRV_WRN("vram admission: name=%s needs %.2f GiB, %.2f GiB free (x%.2f, margin %d MiB), and no idle model can be evicted -- loading anyway\n",
                        name.c_str(), need / 1073741824.0, free_b / 1073741824.0, base_params.models_vram_admit, base_params.models_memory_margin);
                return need;
            }
            victim_bytes = mapping[victim].vram_predicted;
            SRV_INF("vram admission: name=%s needs %.2f GiB, %.2f GiB free (x%.2f, margin %d MiB) -- evicting idle LRU name=%s\n",
                    name.c_str(), need / 1073741824.0, free_b / 1073741824.0, base_params.models_vram_admit, base_params.models_memory_margin, victim.c_str());
            request_stop_locked(victim, mapping[victim], SERVER_STOP_LRU_EVICT); // decided and stopped under one lock
            cv.wait(lk, [this, &victim]() {
                return mapping[victim].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
        // The child has exited but the driver can report its allocation for a moment longer: poll
        // (up to 5 s, ollama's waitForVRAMRecovery) until 75% of what it held reads as free, so the
        // next round does not evict a second model for bytes that are already being released.
        if (!std::getenv("LLAMA_SERVER_VRAM_TOTAL_MB") && victim_bytes > 0) {
            const int64_t target = free_b + victim_bytes * 3 / 4;
            for (int i = 0; i < 20; ++i) {
                const int64_t now = server_vram_free_bytes();
                if (now < 0 || now >= target) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        }
    }
    return need;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    std::string lru_model_name;
    {
        std::unique_lock<std::mutex> lk(mutex);
        if (sched->has_capacity(lk)) {
            return;
        }
        lru_model_name = sched->pick_victim(lk, "");
        if (!lru_model_name.empty()) {
            SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
            request_stop_locked(lru_model_name, mapping[lru_model_name], SERVER_STOP_LRU_EVICT); // decided and stopped under one lock
        }
    }
    if (!lru_model_name.empty()) {
        // wait for unload to complete
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this, &lru_model_name]() {
                return mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
    }
}

void server_models::load(const std::string & name) {
    load(name, load_options{});
}

static int32_t preset_n_parallel(const server_model_meta & meta); // defined with apply_num_ctx

void server_models::load(const std::string & name, const load_options & opts) {
    // the fake load time of a model; a download worker loads none
    if (debug_fake_timing && opts.role == SERVER_ROLE_INSTANCE) {
        // do not hold the mutex here, other requests must keep making progress
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    int64_t vram_need = 0;
    if (!opts.custom_meta.has_value()) {
        if (!has_model(name)) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        unload_lru();
        vram_need = ensure_vram_for(name); // --models-vram-admit; 0 when off
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    // a download over an existing entry is decided under the lock, so a load of the
    // same model cannot start in between: a paused download or an idle cached model
    // is downloaded again with its files kept, a download on its way and a running
    // model are refused, and a preset or local model is not a download
    if (opts.role == SERVER_ROLE_DOWNLOAD) {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            const auto & cur = it->second.meta;
            if (cur.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                throw std::invalid_argument("model '" + name + "' is already downloading");
            }
            if (cur.is_running()) {
                throw std::invalid_argument("model '" + name + "' is running, unload it first");
            }
            if (cur.source != SERVER_MODEL_SOURCE_CACHE) {
                throw std::invalid_argument("model '" + name + "' already exists");
            }
            SRV_INF("resuming download for model '%s'\n", name.c_str());
        }

        // two downloads writing the same partial would interleave their bytes
        for (const auto & d : unfinished_downloads.get_all()) {
            auto other = mapping.find(d.model);
            if (d.model == name || other == mapping.end() || other->second.meta.status != SERVER_MODEL_STATUS_DOWNLOADING) {
                continue;
            }
            for (const auto & p : d.partials) {
                const bool shared = std::any_of(opts.partials.begin(), opts.partials.end(), [&](const common_download_partial & q) {
                    return q.path == p.path;
                });
                if (shared) {
                    throw std::invalid_argument("model '" + name + "' shares files with the download of '" + d.model + "' on its way, retry once it is done");
                }
            }
        }
    }

    auto meta = opts.custom_meta.has_value() ? *opts.custom_meta : mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // Re-check capacity under the lock to prevent concurrent loads from
    // exceeding models_max. Without this, the window between unload_lru()
    // releasing its lock and this lock_guard acquiring allows multiple
    // threads to each observe capacity and all proceed to load.
    // Download workers do not use models_max slots.
    if (opts.role == SERVER_ROLE_INSTANCE && base_params.models_max > 0) {
        size_t count_active = 0;
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
            }
        }
        if (count_active >= (size_t)base_params.models_max) {
            throw server_models_limit_error("model limit reached, try again later");
        }
    }

    // prepare new instance info
    instance_t inst;
    inst.meta             = meta;
    inst.meta.port        = common_http_get_free_port();
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();
    inst.keep_alive_ms    = base_params.models_keep_alive_ms;
    inst.vram_predicted   = vram_need;
    inst.vram_free_before = (vram_need > 0 && !std::getenv("LLAMA_SERVER_VRAM_TOTAL_MB")) ? server_vram_free_bytes() : -1;
    // requests already waiting for this model keep protecting the instance that replaces the old one
    if (auto old = mapping.find(name); old != mapping.end()) {
        inst.n_pending = old->second.n_pending;
    }

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<server_subproc>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        inst.meta.update_args(ctx_preset, bin_path); // render args
        if (opts.ctx_override > 0) {
            // load-OOM retry: this spawn only. The stored preset keeps the user's setting, so a
            // reload does not see a changed preset and the next load starts from it again.
            server_model_meta tmp = inst.meta;
            tmp.preset.set_option(ctx_preset, "LLAMA_ARG_CTX_SIZE", std::to_string(opts.ctx_override));
            tmp.update_args(ctx_preset, bin_path);
            inst.meta.args = tmp.args;
        }
        if (opts.slot_ctx > 0) {
            // ollama options.num_ctx: this many tokens per slot. The pool is slot_ctx * n_parallel and the per-slot cap pins every slot to
            // slot_ctx (unified KV would otherwise let one slot take the whole pool). This spawn only; the stored preset is untouched.
            server_model_meta tmp = inst.meta;
            const int64_t pool = std::min<int64_t>((int64_t) opts.slot_ctx * preset_n_parallel(tmp), INT32_MAX);
            tmp.preset.set_option(ctx_preset, "LLAMA_ARG_CTX_SIZE", std::to_string(pool));
            tmp.preset.set_option(ctx_preset, "LLAMA_ARG_KV_UNIFIED_PER_SLOT", std::to_string(opts.slot_ctx));
            tmp.update_args(ctx_preset, bin_path);
            inst.meta.args = tmp.args;
        }

        std::vector<std::string> child_args = inst.meta.args; // copy
        std::vector<std::string> child_env  = base_env; // copy
        child_env.push_back(std::string(SERVER_ENV_ROUTER_PORT) + "=" + std::to_string(base_params.port));

        if (opts.role == SERVER_ROLE_DOWNLOAD) {
            inst.meta.status = SERVER_MODEL_STATUS_DOWNLOADING;
            child_env.push_back(std::string(SERVER_ENV_CHILD_ROLE) + "=" + server_role_type_to_str(SERVER_ROLE_DOWNLOAD));
            child_env.push_back("LLAMA_ARG_HF_REPO=" + name);

            // the sidecars as -hf would pull them: the draft types, and the mmproj switch
            // for the one that is not a draft
            bool        mmproj = false;
            std::string spec_types;
            for (const auto & s : common_download_sidecars()) {
                if (!has_sidecar(opts.sidecars, s.name)) {
                    continue;
                }
                if (s.draft) {
                    spec_types += (spec_types.empty() ? "" : ",") + std::string("draft-") + s.name;
                } else {
                    mmproj = true;
                }
            }
            set_child_env(child_env, "LLAMA_ARG_MMPROJ_AUTO", mmproj ? "1" : "0");
            set_child_env(child_env, "LLAMA_ARG_SPEC_TYPE", spec_types.empty() ? "none" : spec_types);
        }

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        // the child writes its commands to stdout and its logs to stderr
        int options = subprocess_option_no_window;
        if (!inst.subproc->sproc.create(child_args, options, child_env)) {
            throw std::runtime_error("failed to spawn server instance");
        }
        // fork (ollama llm/runner_job_windows.go): the kernel kills the child when the router dies, even if the child is wedged and
        // never reads its stdin EOF; a failure here leaves the stdin-EOF exit as the only path, so it is logged, not fatal
        std::string job_err;
        if (!inst.subproc->assign_to_kill_on_close_job(job_err)) {
            SRV_WRN("model name=%s: could not put the child in the kill-on-close job (%s); it may outlive a crashed router\n",
                    name.c_str(), job_err.c_str());
        }
    }

    // remember the download: the partial files carry a content hash, not the name
    if (opts.role == SERVER_ROLE_DOWNLOAD) {
        unfinished_downloads.add({ name, opts.sidecars, opts.partials });
    }

    // old process should have exited already, but just in case, we clean it up here
    {
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.subproc && it->second.subproc->is_alive()) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            it->second.subproc->terminate(); // force kill
        }
    }

    notify_sse("model_status", name, {
        {"status", server_model_status_to_string(inst.meta.status)},
    });

    auto proc = inst.subproc;
    int  port = inst.meta.port;
    mapping[name] = std::move(inst);
    monitor->watch(name, proc, opts.role, port);
    cv.notify_all();
}

void server_models::request_stop(const std::string & name, bool send_exit, server_stop_reason reason) {
    auto it = mapping.find(name);
    if (it == mapping.end() || stopping_models.count(name)) {
        return;
    }
    stopping_models.insert(name);
    it->second.stop_asked = reason; // the first stop wins, the one above returns early for a second
    // fork: an exiting instance takes no new request (proxy_request) and keep_alive no longer schedules it;
    // update_status(UNLOADED) clears it once the monitor has reaped the child
    it->second.exiting      = true;
    it->second.expire_at_ms = 0;
    monitor->stop(name, it->second.meta.stop_timeout, send_exit);
}

void server_models::on_child_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, server_role_type role, int exit_code) {
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end() || it->second.subproc != proc) {
            stopping_models.erase(name);
            return; // entry erased, or a newer instance took the name
        }
    }
    if (role == SERVER_ROLE_DOWNLOAD) {
        // instance will be cleaned up on next load_models() call
        std::lock_guard<std::mutex> lk(mutex);
        stopping_models.erase(name);
        cv.notify_all();
    } else {
        {
            // fork: died while still LOADING after an out-of-memory error -> eligible for the one retry
            std::lock_guard<std::mutex> lk(mutex);
            auto it = mapping.find(name);
            if (it != mapping.end()) {
                // fork: a clean exit (0) keeps "" as before; a non-zero exit with no matched line gets "exit code X; last output: ..."
                it->second.last_error = proc->failure_detail(exit_code);
                it->second.load_oom = exit_code != 0
                    && it->second.meta.status == SERVER_MODEL_STATUS_LOADING
                    && proc->oom_seen.load(std::memory_order_relaxed);
                if (it->second.load_oom) {
                    SRV_WRN("instance name=%s ran out of memory while loading\n", name.c_str());
                }
            }
        }
        update_status(name, {
            SERVER_MODEL_STATUS_UNLOADED,
            exit_code
        });
    }
}

void server_models::unload(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            it->second.request_exit();
            // for convenience, we wait the status change here
            wait(lk, name, [](const server_model_meta & new_meta) {
                return new_meta.status != SERVER_MODEL_STATUS_DOWNLOADING;
            });
        } else if (it->second.meta.is_running()) {
            request_stop_locked(name, it->second, SERVER_STOP_MANUAL);
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::unique_lock<std::mutex> lk(mutex);
    snapshot_frozen = true; // shutdown: keep the snapshot of what was loaded, do not record these unloads
    for (auto & [name, inst] : mapping) {
        if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            inst.request_exit();
        } else if (inst.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            bool loading = inst.meta.status == SERVER_MODEL_STATUS_LOADING;
            if (loading) {
                inst.subproc->terminate();
            }
            request_stop(name, !loading, SERVER_STOP_SHUTDOWN);
        }
    }
    // wait for every child to exit, the monitor force-kills the ones that ignore the exit command
    cv.wait(lk, [this]() {
        for (const auto & [name, inst] : mapping) {
            if (inst.meta.is_running() || inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                return false;
            }
        }
        return true;
    });
}

void server_models::update_status(const std::string & name, const update_status_args & args) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        const std::string prev_status = server_model_status_to_string(meta.status);
        meta.status      = args.status;
        meta.exit_code   = args.exit_code;
        // fork: row S69-ENGREV-TFSERVING-LIFECYCLE-EVENT-RING-2026-10-27-01 -- say WHY it went down and keep the transition
        std::string ev_reason;
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            ev_reason       = server_exit_reason(it->second.stop_asked, it->second.load_oom, args.exit_code);
            meta.stop_reason = ev_reason;
            meta.last_error  = args.exit_code != 0 ? it->second.last_error : std::string();
            it->second.stop_asked = SERVER_STOP_NONE;
        } else if (args.status == SERVER_MODEL_STATUS_LOADING || args.status == SERVER_MODEL_STATUS_LOADED) {
            meta.stop_reason.clear(); // a new run: the last one's reason no longer applies
            meta.last_error.clear();
        }
        if (prev_status != server_model_status_to_string(args.status)) {
            events.record((int64_t) std::time(nullptr) * 1000, name, prev_status, server_model_status_to_string(args.status), ev_reason, args.exit_code);
        }
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            stopping_models.erase(name);
            // the child is gone: a request waiting on the exit may load it again
            it->second.exiting      = false;
            it->second.expire_at_ms = 0;
        }
        if (args.status == SERVER_MODEL_STATUS_LOADED) {
            load_cooldown.on_success(name); // a successful load ends the failure streak
        }
        if (args.status == SERVER_MODEL_STATUS_LOADED && it->second.vram_free_before >= 0) {
            // --models-vram-admit: the child's real footprint is how far free device bytes fell
            // while it loaded. Another process allocating in the same window inflates it, which
            // errs toward evicting sooner, the safe direction. Measured once per load.
            // CAPABILITY-GENUINELY-NA: router GPU accounting, not a model capability.
            const int64_t now = server_vram_free_bytes();
            const int64_t used = now >= 0 ? it->second.vram_free_before - now : -1;
            it->second.vram_free_before = -1;
            if (used > 0) {
                SRV_INF("vram admission: name=%s measured %.2f GiB (predicted %.2f GiB)\n",
                        name.c_str(), used / 1073741824.0, it->second.vram_predicted / 1073741824.0);
                vram_measured[name]        = used;
                it->second.vram_predicted  = used;
            }
        }
        if (!args.loaded_info.is_null()) {
            meta.loaded_info = args.loaded_info;
            // the child replaces both arrays in full; a bad or missing value changes nothing
            if (args.loaded_info.contains("architecture") && args.loaded_info.at("architecture").is_object()) {
                const json & child_arch = args.loaded_info.at("architecture");
                for (const char * key : { "input_modalities", "output_modalities" }) {
                    if (!child_arch.contains(key) || !child_arch.at(key).is_array()) {
                        continue;
                    }
                    std::vector<std::string> modalities;
                    bool valid = true;
                    for (const auto & m : child_arch.at(key)) {
                        if (!m.is_string()) {
                            valid = false;
                            break;
                        }
                        modalities.push_back(m.get<std::string>());
                    }
                    if (valid) {
                        meta.architecture[key] = std::move(modalities);
                    }
                }
            }
        }
        if (!args.progress.is_null()) {
            meta.progress = args.progress;
        }
        // a model that comes up idle or goes down changes the slot count for queued requests
        sched->tick(lk);
    }
    // broadcast status change to SSE
    {
        json data = {
            {"status", server_model_status_to_string(args.status)},
        };
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            data["exit_code"] = args.exit_code;
        }
        if (!args.loaded_info.is_null()) {
            data["info"] = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            data["progress"] = args.progress;
        }
        // note: notify_sse doesn't acquire the lock, so no deadlock here
        notify_sse("status_change", name, data);
    }
    // --models-snapshot: the loaded set changed, so rewrite the file. The set is taken under the lock with a sequence
    // number; the write happens outside it, and a write older than the last one written is dropped, so two status
    // changes racing to the file cannot leave the older set on disk. unload_all() (shutdown) freezes the snapshot
    // first, so stopping the router does not erase what it is to restore next time.
    std::vector<std::string> snap_names;
    uint64_t snap_seq = 0;
    const bool snap_write = !base_params.models_snapshot.empty() && !snapshot_frozen &&
        (args.status == SERVER_MODEL_STATUS_LOADED || args.status == SERVER_MODEL_STATUS_UNLOADED);
    if (snap_write) {
        for (const auto & [n, inst] : mapping) {
            // a sleeping model is still one the router serves (it wakes on the next request)
            if (inst.meta.status == SERVER_MODEL_STATUS_LOADED || inst.meta.status == SERVER_MODEL_STATUS_SLEEPING) {
                snap_names.push_back(n);
            }
        }
        snap_seq = ++snapshot_seq;
    }
    cv.notify_all();
    lk.unlock();
    if (snap_write) {
        std::lock_guard<std::mutex> slk(snapshot_mutex);
        if (snap_seq > snapshot_written_seq) {
            std::string err;
            if (server_models_snapshot_write(base_params.models_snapshot, snap_names, err)) {
                snapshot_written_seq = snap_seq;
                SRV_DBG("models snapshot: %zu model(s) -> %s\n", snap_names.size(), base_params.models_snapshot.c_str());
            } else {
                SRV_WRN("models snapshot not written: %s\n", err.c_str());
            }
        }
    }
}

void server_models::update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok) {
    json curr;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            if (done) {
                // mark the instance to be erased on next load_models() call
                it->second.meta.status = SERVER_MODEL_STATUS_DOWNLOADED;
                need_reload = true;

                // a download that failed keeps its tag: its partial files are still
                // there and re-posting the tag resumes them
                if (ok) {
                    unfinished_downloads.forget(name);
                }
            } else {
                json & info = it->second.meta.loaded_info;
                if (!info.contains("progress")) {
                    info["progress"] = json{};
                }
                info["progress"][progress.url] = {
                    {"done",  progress.downloaded},
                    {"total", progress.total},
                };
                curr = it->second.meta.loaded_info; // copy
            }
        }
    }
    if (done) {
        cv.notify_all(); // notify in case unload() is waiting for download to be cancelled
        notify_sse(ok ? "download_finished" : "download_failed", name, {});
    } else {
        notify_sse("download_progress", name, curr);
    }
}

bool server_models::remove(const std::string & name) {
    // do everything under one lock acquisition; avoid get_meta() /
    // unload() because they can trigger load_models() which erases
    // transient DOWNLOADING / DOWNLOADED entries as a side-effect
    std::unique_lock<std::mutex> lk(mutex);

    auto it = mapping.find(name);
    if (it == mapping.end()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (it->second.meta.source != SERVER_MODEL_SOURCE_CACHE) {
        throw std::runtime_error("model name=" + name + " is not removable (not from cache)");
    }

    if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
        // cancel in-flight download
        SRV_INF("cancelling download for model name=%s\n", name.c_str());
        it->second.request_exit();
    } else if (it->second.meta.is_running()) {
        // stop running instance
        SRV_INF("stopping model instance name=%s\n", name.c_str());
        bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
        if (loading) {
            it->second.subproc->terminate();
        }
        request_stop(name, !loading);
    }

    // wait until the child is gone
    wait(lk, name, [](const server_model_meta & meta) {
        return meta.status == SERVER_MODEL_STATUS_UNLOADED
            || meta.status == SERVER_MODEL_STATUS_DOWNLOADED;
    });

    // load_models() may have erased the entry during the wait; the files go under the
    // lock, so a download of the same tag cannot start meanwhile
    mapping.erase(name);
    unfinished_downloads.remove(name);

    // best-effort: a cancelled download may have no cached files
    bool ok = common_download_remove(name);
    SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "nothing cached");
    notify_sse("model_remove", name, {});
    return true;
}

void server_models::wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    std::unique_lock<std::mutex> lk(mutex);
    wait(lk, name, predicate);
}

void server_models::wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    cv.wait(lk, [this, &name, &predicate]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return predicate(it->second.meta);

        }
        // model was removed from mapping by another code path (e.g. load_models()).
        // nothing left to wait for - tell the caller to proceed.
        return true;
    });
}

bool server_models::ensure_model_ready(const std::string & name, const std::function<bool()> & should_stop) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    {
        // a child told to exit (keep_alive expiry, eviction, unload) still reports its old status
        // until it is gone: wait for that, then take the normal path, which loads it again
        std::unique_lock<std::mutex> lk(mutex);
        bool waited_exit = false;
        while (true) {
            auto it = mapping.find(name);
            if (it == mapping.end() || !it->second.exiting) {
                break;
            }
            if (!waited_exit) {
                SRV_INF("model name=%s is exiting, waiting for it before loading it again\n", name.c_str());
                waited_exit = true;
            }
            if (should_stop && should_stop()) {
                throw std::runtime_error("request cancelled while waiting for model name=" + name + " to exit");
            }
            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
        if (waited_exit) {
            lk.unlock();
            meta = get_meta(name);
            if (!meta.has_value()) {
                throw std::runtime_error("model name=" + name + " is not found");
            }
        }
    }
    bool stopping;
    {
        std::lock_guard<std::mutex> lk(mutex);
        stopping = stopping_models.count(name) > 0;
    }
    if (!stopping && meta->is_ready()) {
        return false; // ready for taking requests
    }
    if (!stopping && meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // child is sleeping but still running; new request will wake it up
    }

    // a model whose recent loads failed is refused for a while instead of respawning a loader per request
    if (const int64_t left_ms = load_cooldown.remaining_ms(name, ggml_time_ms()); left_ms > 0) {
        SRV_WRN("model name=%s failed to load %d time(s) in a row, refusing it for another %lld ms\n",
                name.c_str(), load_cooldown.n_failures(name), (long long) left_ms);
        throw server_unavailable_error(
                "model name=" + name + " failed to load " + std::to_string(load_cooldown.n_failures(name)) +
                " time(s) in a row; not trying again for " + std::to_string((left_ms + 999) / 1000) + " s",
                server_retry_after_s(left_ms));
    }

    bool queued   = false;
    bool did_load = false;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.meta.status == SERVER_MODEL_STATUS_UNLOADED) {
            // the queue entry protects the model from eviction until its waiters leave
            sched->join(lk, name);
            sched->tick(lk);
            queued = true;
        }
    }

    // while queued, this is also where the load happens: the head of the queue does it
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    std::unique_lock<std::mutex> lk(mutex);
    auto leave_queue = [this, &queued, &lk, &name]() {
        if (queued) {
            sched->leave(lk, name);
            queued = false;
        }
    };

    // count one failed load once, however many waiters observe it; caller holds mutex
    auto book_failure = [this, &name]() {
        if (auto f = mapping.find(name); f != mapping.end() && !f->second.failure_booked) {
            f->second.failure_booked = true;
            const int64_t armed_ms = load_cooldown.on_failure(name, ggml_time_ms());
            if (armed_ms > 0) {
                SRV_WRN("model name=%s failed to load (%d in a row); further requests are refused for %lld ms\n",
                        name.c_str(), load_cooldown.n_failures(name), (long long) armed_ms);
            }
        }
    };

    try {
        bool saw_loading = false;
        bool oom_retry_attempted = false;
        while (true) {
            auto it = mapping.find(name);
            if (it == mapping.end()) {
                break; // removed by another code path, nothing to wait for
            }
            if (stopping_models.count(name)) {
                // a stopping instance takes no new request, the next instance serves it
                if (!queued) {
                    sched->join(lk, name);
                    sched->tick(lk);
                    queued = true;
                }
                if (should_stop && should_stop()) {
                    throw std::runtime_error("request cancelled while waiting for model name=" + name);
                }
                cv.wait_for(lk, std::chrono::milliseconds(200));
                continue;
            }
            const server_model_status status = it->second.meta.status;

            if (status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_SLEEPING) {
                break;
            }
            if (status == SERVER_MODEL_STATUS_DOWNLOADING || status == SERVER_MODEL_STATUS_DOWNLOADED) {
                break; // do not wait on a download child
            }
            if (status == SERVER_MODEL_STATUS_LOADING) {
                saw_loading = true;
            } else if (status == SERVER_MODEL_STATUS_UNLOADED) {
                if (did_load || saw_loading) {
                    // a spawn happened and the instance came back down
                    if (it->second.oom_retrying) {
                        // another waiter is running the load-OOM retry; its new spawn is ours too
                        cv.wait_for(lk, std::chrono::milliseconds(200));
                        continue;
                    }
                    if (it->second.meta.is_failed() && it->second.load_oom && !oom_retry_attempted) {
                        // ONE retry per waiter (ollama's oomRetryAttempted): a second OOM is reported
                        oom_retry_attempted = true;
                        it->second.load_oom     = false; // consumed: no other waiter retries this failure
                        it->second.oom_retrying = true;
                        lk.unlock();
                        bool retried = false;
                        try {
                            retried = retry_after_load_oom(name);
                        } catch (const std::exception & e) {
                            SRV_WRN("load-OOM retry of name=%s failed: %s\n", name.c_str(), e.what());
                        }
                        lk.lock();
                        if (auto again = mapping.find(name); again != mapping.end()) {
                            again->second.oom_retrying = false;
                        }
                        cv.notify_all();
                        if (retried) {
                            saw_loading = false;
                            did_load    = true;
                            continue;
                        }
                        book_failure();
                        throw std::runtime_error("model name=" + name + " ran out of memory while loading" + (it->second.last_error.empty() ? std::string() : ": " + it->second.last_error));
                    }
                    if (it->second.meta.is_failed()) {
                        book_failure();
                        throw std::runtime_error("model name=" + name + (it->second.load_oom
                            ? " ran out of memory while loading, again after the one retry"
                            : " failed to load") + (it->second.last_error.empty() ? std::string() : ": " + it->second.last_error));
                    }
                    break; // unloaded by another code path, caller reports "not running"
                }
                if (!queued) {
                    break; // not queued, and the load someone else started fell over
                }
            }

            if (should_stop && should_stop()) {
                // if a model was evicted for us, the free slot goes to the next waiter
                throw std::runtime_error("request cancelled while waiting for model name=" + name);
            }

            // our turn: our model is at the head, and a slot really did free up
            if (status == SERVER_MODEL_STATUS_UNLOADED && sched->try_claim(lk, name)) {
                lk.unlock();
                bool ok = true;
                try {
                    SRV_INF("slot available, loading queued model name=%s\n", name.c_str());
                    load(name);
                    did_load = true;
                } catch (const std::exception & e) {
                    // lost a race for the slot, stay in line and retry
                    SRV_WRN("queued load of name=%s did not go through: %s\n", name.c_str(), e.what());
                    ok = false;
                }
                lk.lock();
                sched->claim_done(lk, name, ok);
                sched->tick(lk);
                continue;
            }

            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
    } catch (...) {
        leave_queue();
        sched->tick(lk); // a slot freed for this waiter goes to the next one
        throw;
    }
    leave_queue();

    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Ollama options.num_ctx -> reload the child at that context. Row S69-ENGREV-OLLAMA-ROUTER-PER-REQUEST-NUM-CTX-RELOAD-2026-10-27-01.
// SOURCE-READ: ollama server/sched.go (3fbc609be4b3, MIT, Go; reimplemented, no code copied) -- effectiveContext :204-210 (requested ctx
// capped at the model train ctx), getRunner :213-220 (floor 4; floor 2048 for a vision/projector model), needsReload :1423-1474 (a request
// whose options differ from the loaded runner's reloads it; both num_ctx auto-sized keeps the loaded one, :1441-1443), server ctx = per-seq
// ctx * numParallel (effectiveLlamaServerContext :847).
// SOURCE-NUANCE (deliberate difference): ollama also reloads when the NEW request carries no num_ctx but the loaded runner was started
// with one (the options differ). Here a request without num_ctx uses whatever is loaded: two clients alternating "with" and "without"
// would otherwise reload the model on every request. Equal-or-better for the case the row names; the cost is that an unset num_ctx
// never shrinks a context a previous client asked for (an unload, keep_alive expiry or /models/unload resets it to the preset).
// Per-slot context is what ollama's num_ctx means, so the child is spawned with -c = ctx * n_parallel and --kv-unified-per-slot = ctx:
// every slot then has exactly ctx, in unified and in split KV mode alike. The stored preset is never changed (this spawn only).
// ---------------------------------------------------------------------------------------------------------------------

// per-slot context of a running child, read from its own /v1/props; 0 when it cannot be read
static int64_t child_slot_ctx(int port) {
    httplib::Client cli(CHILD_ADDR, port);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(10, 0);
    cli.set_write_timeout(5, 0);
    auto res = cli.Get("/v1/props");
    if (!res || res->status != 200) {
        return 0;
    }
    try {
        return json::parse(res->body).value("n_ctx", (int64_t) 0);
    } catch (const std::exception &) {
        return 0;
    }
}

int64_t server_models::tools_slot_ctx(const std::string & name) {
    std::vector<std::pair<std::string, int64_t>> running;
    std::string route; // the canonical name the caller's name resolves to ("" = unknown)
    for (const auto & m : get_all_meta()) {
        if (!m.is_ready_or_sleep() || m.port <= 0) {
            continue;
        }
        if (!name.empty() && (m.name == name || m.aliases.count(name))) {
            route = m.name;
        }
        running.emplace_back(m.name, child_slot_ctx(m.port));
    }
    return server_tools_pick_ctx(running, route);
}

// the parallel slot count the child will run with: the preset's value, else the server default (auto = 4, server.cpp)
static int32_t preset_n_parallel(const server_model_meta & meta) {
    std::string v;
    if (meta.preset.get_option("LLAMA_ARG_N_PARALLEL", v) && !v.empty()) {
        try {
            const int n = std::stoi(v);
            if (n >= 1) {
                return n;
            }
        } catch (const std::exception &) {}
    }
    return 4;
}

std::string server_models::apply_num_ctx(const std::string & name, int64_t req_ctx, const std::function<bool()> & should_stop) {
    if (req_ctx <= 0) {
        return ""; // auto-sized: keep whatever is loaded
    }
    // one context reload at a time, and a reload waiting for a busy model holds back the next num_ctx request for that model
    std::lock_guard<std::mutex> serial(num_ctx_mutex);

    server_model_meta meta;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end() || !it->second.meta.is_ready_or_sleep()) {
            return ""; // nothing running to compare with; the caller's proxy_request reports it
        }
        meta = it->second.meta;
    }

    int64_t train  = meta.n_ctx_train;
    bool    vision = false;
    try {
        std::string model_path, mmproj_path;
        resolve_model_files(meta, model_path, mmproj_path);
        if (train <= 0) {
            train = gguf_train_ctx(model_path);
        }
        vision = !mmproj_path.empty();
    } catch (const std::exception & e) {
        SRV_WRN("num_ctx: could not resolve the files of name=%s (%s); the train-context cap and the vision floor are skipped\n", name.c_str(), e.what());
    }
    int64_t want = std::max<int64_t>(req_ctx, 4);   // sched.go:213
    if (vision) {
        want = std::max<int64_t>(want, 2048);       // sched.go:216-220
    }
    if (train > 0 && want > train) {
        SRV_WRN("num_ctx: name=%s requested %lld, capped at the model training context %lld\n", name.c_str(), (long long) req_ctx, (long long) train);
        want = train;                               // sched.go:204-210
    }

    const int64_t have = child_slot_ctx(meta.port);
    if (have <= 0) {
        SRV_WRN("num_ctx: name=%s could not read the loaded per-slot context from the child; num_ctx=%lld NOT applied\n", name.c_str(), (long long) req_ctx);
        return "";
    }
    if (have == want) {
        return ""; // already there (sched.go:1441 / DeepEqual): no reload
    }

    const int32_t n_par = preset_n_parallel(meta);
    SRV_INF("num_ctx: name=%s per-slot context %lld -> %lld (requested %lld, n_parallel %d): reloading the child\n",
            name.c_str(), (long long) have, (long long) want, (long long) req_ctx, (int) n_par);

    // the busy guard: never stop a model that is serving a request (same rule as the preset reload). n_pending counts this caller too.
    {
        std::unique_lock<std::mutex> lk(mutex);
        bool announced = false;
        while (true) {
            auto it = mapping.find(name);
            if (it == mapping.end() || !it->second.meta.is_ready_or_sleep() || it->second.exiting) {
                return "model '" + name + "' went away while num_ctx " + std::to_string(req_ctx) + " was waiting to be applied; retry the request";
            }
            if (it->second.req_count == 0 && it->second.n_pending <= 1) {
                request_stop_locked(name, it->second, SERVER_STOP_RELOAD); // decided and stopped under one lock, nothing can slip in between
                break;
            }
            if (!announced) {
                SRV_WRN("num_ctx: name=%s is serving %d request(s) (+%d pending), the reload to %lld waits until it is idle\n",
                        name.c_str(), it->second.req_count, it->second.n_pending - 1, (long long) want);
                announced = true;
            }
            if (should_stop && should_stop()) {
                return "request cancelled while waiting for model '" + name + "' to go idle for a context reload";
            }
            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
        cv.wait(lk, [&]() {
            auto it = mapping.find(name);
            return it == mapping.end() || (it->second.meta.status == SERVER_MODEL_STATUS_UNLOADED && !it->second.exiting);
        });
    }

    // spawn with `slot_ctx` per slot (0 = the preset as stored) and wait until it is up; false + err on failure
    auto spawn = [&](int32_t slot_ctx, std::string & err) -> bool {
        load_options opts;
        opts.slot_ctx = slot_ctx;
        try {
            load(name, opts); // unload_lru + ensure_vram_for (--models-vram-admit) run inside, evicting idle models as for any load
        } catch (const std::exception & e) {
            err = e.what();
            return false;
        }
        std::unique_lock<std::mutex> lk(mutex);
        while (true) {
            auto it = mapping.find(name);
            if (it == mapping.end()) {
                err = "the model disappeared while loading";
                return false;
            }
            const auto st = it->second.meta.status;
            if (st == SERVER_MODEL_STATUS_LOADED || st == SERVER_MODEL_STATUS_SLEEPING) {
                return true;
            }
            if (st == SERVER_MODEL_STATUS_UNLOADED) {
                err = std::string(it->second.load_oom ? "ran out of memory while loading" : "failed to load") +
                      (it->second.last_error.empty() ? std::string() : ": " + it->second.last_error);
                return false;
            }
            if (should_stop && should_stop()) {
                err = "request cancelled while the model was loading";
                return false;
            }
            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
    };
    auto port_of = [&]() -> int {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        return it == mapping.end() ? 0 : it->second.meta.port;
    };
    // the old context comes back: the preset as stored (this reload never changed it)
    auto restore = [&]() -> std::string {
        std::string rerr;
        if (spawn(0, rerr)) {
            return "the model stays loaded at its previous context";
        }
        return "reloading the previous context failed too (" + rerr + "); the model is unloaded and loads again on the next request";
    };

    // memory check before spawning, now that the old child memory is released (--models-vram-admit on: the router admission switch)
    if (base_params.models_vram_admit > 0.0f) {
        try {
            // same free-VRAM source as the router admission (ensure_vram_for) and the /estimate route: the
            // LLAMA_SERVER_VRAM_TOTAL_MB test hook when set, the backends otherwise. With -1 here the reload ignored the hook
            // the admission honoured, so on a CUDA build the two disagreed (gate b1107g2b 2026-11-08: num_ctx 32768 of a 0.5B
            // model was admitted against the real 32 GB while the admission path ran on the simulated figure).
            int64_t sim_vram_free = -1;
            if (std::getenv("LLAMA_SERVER_VRAM_TOTAL_MB") != nullptr) {
                std::lock_guard<std::mutex> lk(mutex);
                sim_vram_free = vram_free_locked();
            }
            const json est = estimate_model_memory(meta, (int32_t) std::min<int64_t>(want * n_par, INT32_MAX), sim_vram_free);
            if (json_value(est, "fit", std::string()) == "will_not_fit") {
                SRV_WRN("num_ctx: name=%s per-slot context %lld refused by the memory estimate (will_not_fit)\n", name.c_str(), (long long) want);
                return "num_ctx " + std::to_string(req_ctx) + (want != req_ctx ? " (effective " + std::to_string(want) + ")" : std::string()) +
                       " does not fit in memory for model '" + name + "' (" + std::to_string(n_par) + " slot(s)); " + restore();
            }
        } catch (const std::exception & e) {
            SRV_WRN("num_ctx: name=%s memory estimate failed (%s), loading and letting the child decide\n", name.c_str(), e.what());
        }
    }

    std::string err;
    if (!spawn((int32_t) want, err)) {
        SRV_WRN("num_ctx: name=%s reload at %lld failed: %s\n", name.c_str(), (long long) want, err.c_str());
        return "num_ctx " + std::to_string(req_ctx) + " could not be applied to model '" + name + "': " + err + "; " + restore();
    }
    const int64_t got = child_slot_ctx(port_of());
    if (got != want) {
        // never serve at a silently different context (a fit adjustment, or a racing loader that won)
        SRV_WRN("num_ctx: name=%s came up with per-slot context %lld, wanted %lld\n", name.c_str(), (long long) got, (long long) want);
        return "num_ctx " + std::to_string(req_ctx) + " could not be applied to model '" + name + "': the child came up with " +
               std::to_string(got) + " tokens per slot, not " + std::to_string(want);
    }
    SRV_INF("num_ctx: name=%s now runs %lld tokens per slot\n", name.c_str(), (long long) got);
    return "";
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used, bool detached,
                                                 std::optional<int64_t> keep_alive_ms) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (!meta->is_running()) {
        throw std::invalid_argument("model name=" + name + " is not running");
    }
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto & inst = mapping[name];
        if (inst.exiting) {
            // told to exit after this request was admitted, it would die mid-request
            throw std::invalid_argument("model name=" + name + " is being unloaded");
        }
        if (update_last_used) {
            inst.meta.last_used = ggml_time_ms();
        }
        inst.req_count++;
        inst.expire_at_ms = 0; // busy again, the next idle reschedules it
        if (keep_alive_ms.has_value()) {
            inst.keep_alive_ms = *keep_alive_ms; // the latest request decides, as in ollama
        }
    }
    if (debug_fake_timing) {
        // sleep after req_count++, so the model counts as busy while we wait here
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    SRV_TRC("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            "http",
            CHILD_ADDR,
            meta->port,
            proxy_path,
            req.headers,
            req.body,
            req.files,
            // a detached request belongs to a replay session
            detached
                ? std::function<bool()>([]() { return false; })
                : req.should_stop,
            base_params.timeout_read,
            base_params.timeout_write
            );

    proxy->cleanup = [this, name]() {
        bool schedule = false;
        {
            std::unique_lock<std::mutex> lk(mutex);
            auto it = mapping.find(name);
            if (it != mapping.end() && it->second.req_count > 0) {
                it->second.req_count--;
                if (it->second.req_count == 0) {
                    redo_deferred_reload_locked(it->second);
                    if (it->second.keep_alive_ms >= 0 && !it->second.exiting) {
                        it->second.expire_at_ms = ggml_time_ms() + it->second.keep_alive_ms;
                        schedule = true;
                    }
                    // a model that went idle may give its slot to a queued request
                    sched->tick(lk);
                }
            }
        }
        if (schedule) {
            cv_keep_alive.notify_all();
        }
    };

    return proxy;
}

void server_models::proxy_ws(const server_http_req & req, server_http_context::ws_conn & conn, const std::string & name) {
    auto send_error = [&conn](const std::string & message) {
        conn.send(safe_json_to_str(json {
            {"type", "error"},
            {"error", {{"type", "server_error"}, {"code", nullptr}, {"message", message}, {"param", nullptr}}},
        }));
        conn.close();
    };

    auto meta = get_meta(name);
    if (!meta.has_value() || !meta->is_running()) {
        send_error("model name=" + name + " is not running");
        return;
    }
    {
        std::unique_lock<std::mutex> lk(mutex);
        mapping[name].meta.last_used = ggml_time_ms();
        mapping[name].req_count++;
    }
    SRV_TRC("proxying websocket to model %s on port %d\n", name.c_str(), meta->port);

    std::string url = std::string("ws://") + CHILD_ADDR + ":" + std::to_string(meta->port) + req.path;
    if (!req.query_string.empty()) {
        url += '?' + req.query_string;
    }
    // the client library adds its own upgrade headers
    httplib::Headers headers;
    for (const auto & [key, value] : req.headers) {
        std::string lower = key;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == "host" || lower == "connection" || lower == "upgrade" || lower.compare(0, 14, "sec-websocket-") == 0) {
            continue;
        }
        headers.emplace(key, value);
    }

    {
        httplib::ws::WebSocketClient upstream(url, headers);
        if (!upstream.connect()) {
            send_error("failed to connect to model name=" + name);
        } else {
            // client to child
            std::thread pump([&]() {
                std::string msg;
                while (true) {
                    const int res = conn.read(msg);
                    if (res == 0 || !(res == 1 ? upstream.send(msg) : upstream.send(msg.data(), msg.size()))) {
                        break;
                    }
                }
                upstream.close();
            });
            // child to client
            std::string msg;
            while (true) {
                const auto res = upstream.read(msg);
                if (res == httplib::ws::Fail || !(res == httplib::ws::Text ? conn.send(msg) : conn.send_binary(msg))) {
                    break;
                }
            }
            conn.close();
            pump.join();
        }
    }

    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end() && it->second.req_count > 0) {
        it->second.req_count--;
        if (it->second.req_count == 0) {
            redo_deferred_reload_locked(it->second);
            sched->tick(lk);
        }
    }
}

void server_models::handle_child_state(const std::string & name, const std::string & raw_input) {
    server_state state;
    json payload;

    try {
        json data = json::parse(raw_input.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
        state = server_state_from_str(json_value(data, "state", std::string()));
        payload = json_value(data, "payload", json{});
    } catch (const std::exception & e) {
        SRV_ERR("failed to parse child state update for name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    switch (state) {
        case SERVER_STATE_DOWNLOADING:
            {
                std::string result = json_value(payload, "result", std::string());
                std::string url    = json_value(payload, "url",    std::string());
                auto request_exit = [&]() {
                    std::lock_guard<std::mutex> lk(mutex);
                    auto it = mapping.find(name);
                    if (it != mapping.end()) {
                        return it->second.request_exit();
                    }
                };
                if (result == "download_finished") {
                    update_download_progress(name, {}, true, true);
                    request_exit();
                } else if (result == "download_failed") {
                    update_download_progress(name, {}, true, false);
                    request_exit();
                } else if (!url.empty()) {
                    common_download_progress p;
                    p.url        = url;
                    p.downloaded = json_value(payload, "downloaded", (size_t)0);
                    p.total      = json_value(payload, "total", (size_t)0);
                    update_download_progress(name, p, false);
                }
            } break;
        case SERVER_STATE_LOADING:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADING,
                    0,
                    nullptr, // no loaded_info yet
                    payload,
                });
            } break;
        case SERVER_STATE_READY:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADED,
                    0,
                    // note: payload can be empty if this is a wakeup from sleep
                    payload.size() > 0 ? payload : nullptr,
                    {}, // reset progress info
                });
            } break;
        case SERVER_STATE_SLEEPING:
            {
                update_status(name, { SERVER_MODEL_STATUS_SLEEPING });
            } break;
        default:
            // should never happen, but just in case
            GGML_ASSERT(false && "unexpected state from child server");
    }
}

//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> sse_client_id_counter = 0;
struct server_models_sse_client {
    server_response & queue_results;
    int client_id;
    server_models_sse_client(server_response & q)
            : queue_results(q), client_id(sse_client_id_counter.fetch_add(1, std::memory_order_relaxed)) {
        SRV_DBG("new SSE client connected, assigned client_id=%d\n", client_id);
        queue_results.add_waiting_task_id(client_id);
    }
    ~server_models_sse_client() {
        SRV_DBG("SSE client disconnected, removing client_id=%d\n", client_id);
        queue_results.remove_waiting_task_id(client_id);
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            static const int http_polling_seconds = 1; // check should_stop every 1 second
            server_task_result_ptr result = queue_results.recv_with_timeout({client_id}, http_polling_seconds);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    return nullptr;
                }
                // continue waiting otherwise
            } else {
                SRV_DBG("recv result for client_id=%d: %s\n", client_id, safe_json_to_str(result->to_json()).c_str());
                return result;
            }
        }
        // should not reach here
    }
};

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = error_http_status(error_data, 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // resolve alias to canonical model name
    name = meta->name;
    if (!models_autoload && !meta->is_running()) {
        res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string encode_qs(const std::string & in) {
    std::string out;
    out.reserve(in.size() * 3);
    for (unsigned char c : in) {
        bool safe = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                 || c == '-' || c == '_' || c == '.' || c == '~';
        if (safe) {
            out.push_back(char(c));
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out.append(buf, 3);
        }
    }
    return out;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> resolve_child_for_conv(
        server_models & models, const std::string & conversation_id) {
    if (conversation_id.empty()) {
        return std::nullopt;
    }
    auto tracked = models.conv_models.lookup(conversation_id);
    if (!tracked.has_value()) {
        return std::nullopt;
    }
    auto meta = models.get_meta(*tracked);
    if (meta.has_value() && meta->is_ready()) {
        return meta;
    }
    return std::nullopt;
}

void server_models_routes::init_routes() {
    if (!common_subproc::is_supported()) {
        throw std::runtime_error("subprocess is not enabled on this build");
    }

    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            json props = {
                // TODO: add support for this on web UI
                {"role",                 "router"},
                {"max_instances",        params.models_max},
                {"models_autoload",      params.models_autoload},
                // this is a dummy response to make sure the UI doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
                // New key
                {"ui_settings",          ui_settings},
                {"build_info",           std::string(llama_build_info())},
                {"cors_proxy_enabled",   params.ui_mcp_proxy},
            };
            if (!params.server_connect_code.empty()) {
                props["connect_code"] = params.server_connect_code;
            }
            res_ok(res, props);
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        auto pending = models.pending_acquire(name); // released after proxy_request counted it
        if (autoload) {
            models.ensure_model_ready(name, req.should_stop);
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_ws = [this](const server_http_req & req, server_http_context::ws_conn & conn) {
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            // error_res holds {"error": {...}}, send it as a websocket error event
            json err = json::parse(error_res->data);
            err["type"] = "error";
            conn.send(safe_json_to_str(err));
            conn.close();
            return;
        }
        if (autoload) {
            models.ensure_model_ready(name, req.should_stop);
        }
        models.proxy_ws(req, conn, name);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        // ollama keep_alive: seconds as a number, or a duration string; negative = forever
        int64_t keep_alive_ms = params.models_keep_alive_ms;
        if (body.contains("keep_alive") && !body.at("keep_alive").is_null()) {
            const json & ka = body.at("keep_alive");
            bool ok = true;
            if (ka.is_number()) {
                const double sec = ka.get<double>();
                keep_alive_ms = sec < 0 ? -1 : (int64_t) (sec * 1000.0);
            } else if (ka.is_string()) {
                ok = string_parse_duration_ms(ka.get<std::string>(), keep_alive_ms);
            } else {
                ok = false;
            }
            if (!ok) {
                res_err(error_res, format_error_response(
                        "invalid keep_alive: " + ka.dump() + " (seconds, or a duration such as \"5m\")", ERROR_TYPE_INVALID_REQUEST));
                return error_res;
            }
        }
        // One attempt on one model: wait for it to be ready, then proxy the request to it. Throws when the model cannot be made ready.
        bool stopped = false; // a stop issued by the user (not a model failure): never retried on another model
        auto attempt = [&](const std::string & target, bool is_fallback) -> server_http_res_ptr {
            if (is_fallback) {
                // DISABLE-REVERSIBLE: --no-models-autoload is the operator's existing switch; this only honours it for a fallback model
                const auto fb = models.get_meta(target);
                if (!fb.has_value()) {
                    throw std::runtime_error("model name=" + target + " is not found");
                }
                if (!autoload && !fb->is_running()) {
                    throw std::invalid_argument("model name=" + target + " is not running");
                }
            }
            auto pending = models.pending_acquire(target); // released after proxy_request counted it
            // remember which child serves this conversation so the stream routes can route straight
            // to it without polling, keyed on the exact conv id from the header. registered before
            // the load wait so a stop issued while the model loads can erase the entry and cancel
            // this request instead of leaving an orphan generation
            std::string conv_id = server_stream_conv_id_from_headers(req.headers);
            uint64_t ticket = models.conv_models.remember(conv_id, target);
            // a dead socket must not cancel a session request, only a stop does (checked right below)
            auto should_stop = ticket == 0 ? req.should_stop : nullptr;
            bool waited = autoload && models.ensure_model_ready(target, should_stop);
            if (ticket != 0 && !models.conv_models.alive(conv_id, ticket)) {
                SRV_INF("request for conv_id=%s cancelled while model name=%s was loading\n",
                        conv_id.c_str(), target.c_str());
                res_err(error_res, format_error_response(
                        "request cancelled by a stop while the model was loading", ERROR_TYPE_INVALID_REQUEST));
                stopped = true;
                return std::move(error_res);
            }
            // options.num_ctx from the Ollama shim (body field "ollama_num_ctx", set by server-ollama.cpp only in router mode): reload the
            // child at that per-slot context when it differs, or answer 400 with the reason; never serve at a silently different context
            if (const int64_t want_ctx = body.contains("ollama_num_ctx") && body.at("ollama_num_ctx").is_number_integer()
                                             ? body.at("ollama_num_ctx").get<int64_t>() : 0; want_ctx > 0) {
                if (!autoload) {
                    SRV_WRN("num_ctx %lld for model name=%s not applied: --no-models-autoload forbids the reload it needs\n",
                            (long long) want_ctx, target.c_str());
                } else {
                    const std::string nerr = models.apply_num_ctx(target, want_ctx, should_stop);
                    if (!nerr.empty()) {
                        res_err(error_res, format_error_response(nerr, ERROR_TYPE_INVALID_REQUEST));
                        stopped = true; // a refusal is not a model failure: no failover
                        return std::move(error_res);
                    }
                }
            }
            // a session request that waited for a load detaches from the client socket: the
            // client may have dropped during the wait (page reload) and the session buffer must
            // still receive the generation for a later resume
            return models.proxy_request(req, method, target, true, waited && ticket != 0, keep_alive_ms); // update last usage for POST request only
        };

        // Failover chain (--fallback-model, MLflow FallbackProvider): without one this is the single attempt above, unchanged.
        const std::vector<std::string> chain = models.failover_chain(name);
        if (chain.size() <= 1) {
            return attempt(name, false);
        }
        for (size_t i = 0; i < chain.size(); ++i) {
            const bool last = i + 1 == chain.size();
            try {
                auto res = attempt(chain[i], i > 0);
                // The router answers from the response HEADERS: a child sends them only once its first result exists, and nothing of the
                // body has been handed to the client yet, so a server error here can still go to the next model. A response that is
                // already flowing is never reached by this check (MLflow base.py:824-825 `if yielded: raise`).
                if (stopped || last || !server_failover_status(res->status)) {
                    res->headers["X-Model-Served"]      = chain[i];
                    res->headers["X-Failover-Attempts"] = std::to_string(i + 1);
                    return res;
                }
                SRV_WRN("failover: model %s answered HTTP %d, trying %s next (attempt %zu of %zu)\n",
                        chain[i].c_str(), res->status, chain[i + 1].c_str(), i + 2, chain.size());
            } catch (const std::exception & e) {
                if (last || (req.should_stop && req.should_stop())) {
                    throw; // the final error is the last model's own error (MLflow base.py:831-839)
                }
                SRV_WRN("failover: model %s could not serve the request (%s), trying %s next (attempt %zu of %zu)\n",
                        chain[i].c_str(), e.what(), chain[i + 1].c_str(), i + 2, chain.size());
            }
        }
        throw std::runtime_error("failover chain exhausted"); // not reachable: the last attempt returns or throws
    };

    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->is_running()) {
            res_err(res, format_error_response("model is already running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load_cooldown.clear(meta->name); // an explicit operator action bypasses and resets the failure cooldown
        models.load(meta->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_http_req & req) {
        bool reload = !req.get_param("reload", "").empty();
        if (reload) {
            models.load_models();
        }
        const std::string info_mode = req.get_param("info", "");
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        // a download on its way has no files to resolve yet, only the kinds it pulls
        const auto unfinished = models.unfinished_downloads_get();

        for (const auto & meta : all_models) {
            if (meta.hidden) {
                continue; // cache model deduplicated by a preset
            }
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }
            if (!meta.stop_reason.empty()) {
                status["stop_reason"] = meta.stop_reason; // why the last child stopped: idle-ttl, lru-evict, manual, reload, shutdown, load-oom, crash
            }
            if (!meta.last_error.empty()) {
                status["last_error"] = meta.last_error;
            }

            json model_info = json {
                {"id",            meta.name},
                {"aliases",       meta.aliases},
                {"tags",          meta.tags},
                {"object",        "model"},    // for OAI-compat
                {"owned_by",      "llamacpp"}, // for OAI-compat
                {"created",       t},          // for OAI-compat
                {"status",        status},
                {"architecture",  meta.architecture},
                {"source",        server_model_source_to_string(meta.source)},
                {"can_remove",    meta.source == SERVER_MODEL_SOURCE_CACHE},
                // {"need_download", meta.need_download},
                // TODO: add other fields from the GGUF metadata
            };

            // ?info=1 (or ?info=full for the chat template text): what each model's GGUF header says, read without loading it
            if (!info_mode.empty()) {
                try {
                    std::string model_path, mmproj_path;
                    resolve_model_files(meta, model_path, mmproj_path);
                    const gguf_light_info_t gi = gguf_light_info_cached(model_path);
                    if (gi.ok) {
                        model_info["gguf"] = gguf_light_info_to_json(gi, info_mode == "full");
                    }
                } catch (const std::exception & e) {
                    SRV_WRN("model listing: could not read the GGUF header of name=%s (%s)\n", meta.name.c_str(), e.what());
                }
            }

            if (meta.n_ctx_train > 0) {
                model_info["context_length"] = meta.n_ctx_train;
            }

            // the sidecars a model has on disk, resolved as -hf would load them, for a
            // model that resolves from a HF repo. A download on its way reports the
            // kinds it pulls, as an unfinished one does
            const auto it_unfinished = std::find_if(unfinished.begin(), unfinished.end(),
                [&](const server_unfinished_download & d) { return d.model == meta.name; });
            const std::string hf_repo = model_hf_repo(meta);

            if (it_unfinished != unfinished.end() && meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                model_info["sidecars"] = it_unfinished->sidecars;
            } else if (!hf_repo.empty()) {
                const json cached = cached_sidecars(hf_repo);
                model_info["sidecars"] = cached;

                // the sidecars of an unfinished download of this model that are not on
                // disk yet: a POST with the kind resumes one, a DELETE with it drops it
                if (it_unfinished != unfinished.end()) {
                    std::vector<std::string> incomplete;
                    for (const auto & p : it_unfinished->partials) {
                        if (!p.kind.empty() && !has_sidecar(incomplete, p.kind) &&
                            std::find(cached.begin(), cached.end(), p.kind) == cached.end()) {
                            incomplete.push_back(p.kind);
                        }
                    }
                    if (!incomplete.empty()) {
                        model_info["incomplete_sidecars"] = incomplete;
                    }
                }
            }

            // merge with loaded_info from the child process if available
            if (meta.is_running()) {
                for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                    if (!model_info.contains(it.key())) {
                        model_info[it.key()] = it.value();
                    }
                }
            }
            models_json.push_back(model_info);
        }

        const json architecture_default = json {
            {"input_modalities",  json::array({"text"})},
            {"output_modalities", json::array({"text"})},
        };

        // a download that never finished has no instance: report it as incomplete so
        // the UI can offer to resume it, without starting anything
        for (const auto & d : unfinished) {
            if (models.has_model(d.model)) {
                continue; // complete, or downloading right now
            }
            models_json.push_back(json {
                {"id",           d.model},
                {"aliases",      json::array()},
                {"tags",         json::array()},
                {"object",       "model"},
                {"owned_by",     "llamacpp"},
                {"created",      t},
                {"status",       json { {"value", server_model_status_to_string(SERVER_MODEL_STATUS_INCOMPLETE)}, {"args", json::array()} }},
                {"architecture", architecture_default},
                {"source",       server_model_source_to_string(SERVER_MODEL_SOURCE_CACHE)},
                {"can_remove",   true},
                {"sidecars",     d.sidecars},
            });
        }

        res_ok(res, {
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    // POST /models/estimate {"model": name, "n_ctx": optional}: per-device memory the preset would need, without spawning a child
    this->post_router_models_estimate = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        const int32_t n_ctx = json_value(body, "n_ctx", (int32_t) 0);
        if (n_ctx < 0) {
            res_err(res, format_error_response("n_ctx must be 0 (use the preset) or a positive number", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        int64_t sim_vram_free = -1;
        if (std::getenv("LLAMA_SERVER_VRAM_TOTAL_MB") != nullptr) {
            std::lock_guard<std::mutex> lk(models.mutex);
            sim_vram_free = models.vram_free_locked();
        }
        try {
            res_ok(res, estimate_model_memory(*meta, n_ctx, sim_vram_free));
        } catch (const std::exception & e) {
            res_err(res, format_error_response(std::string("estimate failed: ") + e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_running() && model->status != SERVER_MODEL_STATUS_DOWNLOADING) {
            res_err(res, format_error_response("model is not running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    // Ollama shim, empty-request load / keep_alive 0 unload (called by server-ollama.cpp through server_routes::ollama_model_load/_unload).
    // SOURCE-READ: tmp/ollload_1108/src/routes.go:563-571 (generate load), :3241-3248 (chat load), :480-491 / :3098-3107 (unload).
    // SOURCE-NUANCE: both resolve the model first (404 for an unknown name) and reply 200 whether or not anything was loaded; unload of a
    // model that is not loaded still answers "unload". The name may arrive as "name:latest" (the tag Ollama clients add).
    auto resolve_ollama_name = [this](const std::string & in) -> std::string {
        if (models.get_meta(in).has_value()) {
            return in;
        }
        static const std::string latest = ":latest";
        if (in.size() > latest.size() && in.compare(in.size() - latest.size(), latest.size(), latest) == 0) {
            const std::string base = in.substr(0, in.size() - latest.size());
            if (models.get_meta(base).has_value()) {
                return base;
            }
        }
        return in;
    };
    this->ollama_model_load = [this, resolve_ollama_name](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = resolve_ollama_name(json_value(body, "model", std::string()));
        const bool autoload = is_autoload(params, req);
        if (!name.empty() && !models.get_meta(name).has_value()) {
            // the caller's Ollama client expects 404 for a model the server does not hold (the shared validator answers 400)
            res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (!router_validate_model(name, models, autoload, res)) {
            return res;
        }
        int64_t keep_alive_ms = params.models_keep_alive_ms;
        if (body.contains("keep_alive") && !body.at("keep_alive").is_null()) {
            const json & ka = body.at("keep_alive");
            bool ok = true;
            if (ka.is_number()) {
                const double sec = ka.get<double>();
                keep_alive_ms = sec < 0 ? -1 : (int64_t) (sec * 1000.0);
            } else if (ka.is_string()) {
                ok = string_parse_duration_ms(ka.get<std::string>(), keep_alive_ms);
            } else {
                ok = false;
            }
            if (!ok) {
                res_err(res, format_error_response("invalid keep_alive: " + ka.dump() + " (seconds, or a duration such as \"5m\")", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        auto pending = models.pending_acquire(name); // keeps the model from expiring between the load and the keep_alive below
        models.ensure_model_ready(name, req.should_stop);
        if (const int64_t want_ctx = body.contains("ollama_num_ctx") && body.at("ollama_num_ctx").is_number_integer()
                                         ? body.at("ollama_num_ctx").get<int64_t>() : 0; want_ctx > 0 && autoload) {
            const std::string nerr = models.apply_num_ctx(name, want_ctx, req.should_stop);
            if (!nerr.empty()) {
                res_err(res, format_error_response(nerr, ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        models.apply_keep_alive(name, keep_alive_ms);
        res_ok(res, {{"success", true}, {"model", name}});
        return res;
    };
    this->ollama_model_unload = [this, resolve_ollama_name](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        const std::string given = json_value(body, "model", std::string());
        if (given.empty()) {
            res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const std::string name = resolve_ollama_name(given);
        const auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response(string_format("model '%s' not found", given.c_str()), ERROR_TYPE_NOT_FOUND));
            return res;
        }
        // not running: nothing to expire, still a success (expireRunner is a no-op for a model that is not loaded)
        const int rc = models.apply_keep_alive(meta->name, 0);
        if (rc == 1) {
            // an idle model was told to stop: wait for the child to be gone so /api/ps right after this reply is truthful; a busy model
            // stops when its last request ends, which this reply does not wait for
            const int64_t t_end = ggml_time_ms() + 30000;
            while (ggml_time_ms() < t_end && !req.should_stop()) {
                const auto m = models.get_meta(meta->name);
                if (!m.has_value() || !m->is_running()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        }
        res_ok(res, {{"success", true}, {"model", meta->name}});
        return res;
    };

    this->get_router_models_sse = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        auto sse_client = std::make_shared<server_models_sse_client>(models.sse);
        res->next = [this, sse_client, &req](std::string & output) -> bool {
            auto result = sse_client->next([&]() {
                return stopping.load(std::memory_order_relaxed) || req.should_stop();
            });
            if (result == nullptr) {
                return false; // client disconnected or should_stop
            }
            output = "data: " + safe_json_to_str(result->to_json()) + "\n\n";
            return true; // listen for the next event
        };
        return res;
    };

    // GET /models/events?since=N -- the retained lifecycle ring (server-model-events.h); unlike /models/sse a late client can read it
    this->get_router_models_events = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        uint64_t since = 0;
        const std::string since_str = req.get_param("since", "");
        if (!since_str.empty()) {
            if (since_str.find_first_not_of("0123456789") != std::string::npos || since_str.size() > 18) {
                res_err(res, format_error_response("since must be a non-negative integer", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            since = std::stoull(since_str);
        }
        json items = json::array();
        for (const auto & ev : models.events.since(since)) {
            items.push_back({
                {"seq",       ev.seq},
                {"ts_ms",     ev.ts_ms},
                {"model",     ev.model},
                {"from",      ev.from},
                {"to",        ev.to},
                {"reason",    ev.reason},
                {"exit_code", ev.exit_code},
            });
        }
        res_ok(res, {
            {"object",     "list"},
            {"newest_seq", models.events.newest_seq()},
            {"capacity",   server_model_event_ring::CAPACITY},
            {"data",       items},
        });
        return res;
    };

    this->post_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        // DISABLE-REVERSIBLE: --models-register (default on), see common/arg.cpp
        if (!params.models_register) {
            res_err(res, format_error_response("model registration is disabled on this server (--no-models-register)", ERROR_TYPE_PERMISSION));
            return res;
        }

        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }
        // --models-register-allow (server-auth.h): refused before any metadata fetch, so a refused repo costs no network call
        if (const std::string why = server_models_register_refusal(params.models_register_allow, name); !why.empty()) {
            res_err(res, format_error_response(why, ERROR_TYPE_PERMISSION));
            return res;
        }
        const auto sidecars = parse_sidecars(body);

        common_params_model model;
        model.hf_repo = name;

        // validate by fetching metadata, through the plan the download follows
        common_download_hf_plan plan;
        try {
            plan = common_download_get_hf_plan(model, sidecar_download_opts(params, sidecars));
        } catch (...) {
            SRV_ERR("unknown error while validating model '%s'\n", name.c_str());
            // other exceptions will be handled by the outer ex_wrapper()
            throw;
        }

        // a tag that resolves no file has nothing to download
        if (plan.preset.path.empty() && plan.primary.path.empty() && common_download_plan_sidecars(plan).empty()) {
            throw std::invalid_argument("model '" + name + "' resolves no file to download");
        }

        // then, proceed with the actual download; load() decides over an existing entry
        SRV_INF("starting download for model '%s'\n", name.c_str());
        {
            server_models::load_options load_opts;
            load_opts.role = SERVER_ROLE_DOWNLOAD;
            load_opts.custom_meta = server_model_meta{};
            load_opts.custom_meta->source = SERVER_MODEL_SOURCE_CACHE;
            load_opts.custom_meta->name   = name;
            load_opts.sidecars            = sidecars;
            load_opts.partials            = common_download_plan_partials(plan);
            models.load(name, load_opts);
        }

        res_ok(res, {{"success", true}});
        return res;
    };

    this->del_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        // DISABLE-REVERSIBLE: --models-register (default on), see common/arg.cpp
        if (!params.models_register) {
            res_err(res, format_error_response("model removal is disabled on this server (--no-models-register)", ERROR_TYPE_PERMISSION));
            return res;
        }

        std::string name = req.get_param("model");
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        // a sidecar alone goes, the model keeps its files
        std::string sidecar = req.get_param("sidecar");
        if (!sidecar.empty()) {
            check_sidecar(sidecar);
            res_ok(res, {{"success", models.remove_sidecar(name, sidecar)}});
            return res;
        }

        // a download listed from an earlier run has no instance, only its tag and
        // its partial files
        if (!models.has_model(name)) {
            models.unfinished_downloads_remove(name);
            res_ok(res, {{"success", true}});
            return res;
        }

        models.remove(name); // throws on error

        res_ok(res, {{"success", true}});
        return res;
    };

    this->router_stream_get = [this](const server_http_req & req) {
        // GET /v1/stream?conv_id=<id>&from=N. resolve the owning child from the conv_id -> model
        // map, 404 when nothing maps
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::optional<server_model_meta> owner = resolve_child_for_conv(models, conv_id);
        if (!owner.has_value()) {
            // a registered conv whose model is still loading earns a retry: the session appears
            // once the load ends and the pending request reaches the child
            auto tracked = models.conv_models.lookup(conv_id);
            auto meta = tracked.has_value() ? models.get_meta(*tracked) : std::nullopt;
            bool transient = meta.has_value() && (meta->status == SERVER_MODEL_STATUS_LOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADED);
            if (transient) {
                res_err(res, format_error_response("Stream owner model is loading, retry later", ERROR_TYPE_UNAVAILABLE));
            } else {
                res_err(res, format_error_response("Stream not found or expired", ERROR_TYPE_NOT_FOUND));
            }
            return res;
        }
        std::string from = req.get_param("from");
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        if (!from.empty()) {
            child_path += "&from=" + from;
        }
        SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n",
                owner->name.c_str(), owner->port, child_path.c_str());
        auto proxy = std::make_unique<server_http_proxy>(
                "GET",
                "http",
                CHILD_ADDR,
                owner->port,
                child_path,
                req.headers,
                req.body,
                req.files,
                req.should_stop,
                params.timeout_read,
                params.timeout_write);
        return std::unique_ptr<server_http_res>(std::move(proxy));
    };

    this->router_streams_lookup = [this](const server_http_req & req) {
        // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
        // map, group the ids per child, and query only the children that actually own some of
        // them instead of fanning out to every ready child. a child only answers for the ids
        // it owns, never lists anything else
        auto res = std::make_unique<server_http_res>();
        std::vector<std::string> requested;
        try {
            json body = json::parse(req.body);
            if (body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
                for (const auto & v : body["conversation_ids"]) {
                    if (v.is_string() && !v.get<std::string>().empty()) {
                        requested.push_back(v.get<std::string>());
                    }
                }
            }
        } catch (const std::exception &) {
            res_ok(res, json::array());
            return res;
        }

        // group requested ids by the child port that owns them, drop ids that map to nothing
        std::unordered_map<int, json> per_child;
        for (const auto & cid : requested) {
            auto owner = resolve_child_for_conv(models, cid);
            if (!owner.has_value()) {
                continue;
            }
            per_child[owner->port].push_back(cid);
        }

        json aggregated = json::array();
        for (auto & [port, ids] : per_child) {
            json child_body = {{"conversation_ids", ids}};
            httplib::Client cli(CHILD_ADDR, port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Post("/v1/streams/lookup", child_body.dump(), "application/json");
            if (!resp || resp->status != 200) {
                continue;
            }
            try {
                json child_arr = json::parse(resp->body);
                if (!child_arr.is_array()) {
                    continue;
                }
                for (auto & entry : child_arr) {
                    if (entry.is_object()) {
                        aggregated.push_back(entry);
                    }
                }
            } catch (const std::exception &) {
                continue;
            }
        }
        res_ok(res, aggregated);
        return res;
    };

    this->router_stream_delete = [this](const server_http_req & req) {
        // DELETE /v1/stream?conv_id=<id>. resolve the owning child via the map and forward only to
        // it, evict_and_cancel is idempotent on the child
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        auto owner = resolve_child_for_conv(models, conv_id);
        if (owner.has_value()) {
            httplib::Client cli(CHILD_ADDR, owner->port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Delete(child_path.c_str());
            (void) resp; // the child logs its own miss when the session is unknown there
        } else if (auto tracked = models.conv_models.lookup(conv_id); tracked.has_value()) {
            // the entry exists but its model is still loading: the forget below erases it,
            // which cancels the request parked in proxy_post before the generation starts
            SRV_INF("router stop for conv_id=%s while model name=%s is loading, cancelling the pending request\n",
                    conv_id.c_str(), tracked->c_str());
        } else {
            SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n",
                    conv_id.c_str());
        }
        // drop the tracking entry, the session is being torn down
        models.conv_models.forget(conv_id);
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}



//
// server_http_proxy
//

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static std::string generate_multipart_boundary() {
    thread_local std::mt19937 gen(std::random_device{}());
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string boundary = "----llama-cpp-proxy-";
    for (int i = 0; i < 16; i++) {
        boundary += chars[dis(gen)];
    }
    return boundary;
}

static std::string build_multipart_body(
        const json & form_fields,
        const std::map<std::string, uploaded_file> & files,
        const std::string & boundary) {
    static auto sanitize_field = [](const std::string & text) {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    };

    std::ostringstream body;

    for (const auto & [key, value] : form_fields.items()) {
        if (value.is_array()) {
            for (const auto & item : value) {
                body << "--" << boundary << "\r\n";
                body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
                body << "\r\n";
                if (!item.is_string()) {
                    throw std::invalid_argument("expected string");
                }
                body << item.get<std::string>() << "\r\n";
            }
        } else {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
            body << "\r\n";
            if (!value.is_string()) {
                throw std::invalid_argument("expected string");
            }
            body << value.get<std::string>() << "\r\n";
        }
    }

    for (const auto & [key, file] : files) {
        body << "--" << boundary << "\r\n";
        body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"";
        if (!file.filename.empty()) {
            body << "; filename=\"" << sanitize_field(file.filename) << "\"";
        }
        body << "\r\n";
        if (!file.content_type.empty()) {
            body << "Content-Type: " << sanitize_field(file.content_type) << "\r\n";
        } else {
            body << "Content-Type: application/octet-stream\r\n";
        }
        body << "\r\n";
        body.write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        body << "\r\n";
    }

    body << "--" << boundary << "--\r\n";
    return body.str();
}

server_http_proxy::server_http_proxy(
        const std::string & method,
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
        ) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::ClientImpl>(host, port);
    auto pipe = std::make_shared<server_pipe<msg_t>>();

    if (scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.reset(new httplib::SSLClient(host, port));
#else
        throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
    }

    // setup Client
    cli->set_follow_location(true);
    cli->set_connection_timeout(timeout_read, 0); // use --timeout value instead of hardcoded 5 s
    cli->set_write_timeout(timeout_read, 0); // reversed for cli (client) vs srv (server)
    cli->set_read_timeout(timeout_write, 0);
    this->status = 500; // to be overwritten upon response
    this->cleanup_pipes = [pipe, cli]() {
        pipe->close_read();
        pipe->close_write();
        cli->stop();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // build the header message forwarded to the reader thread, stripping internal proxy headers
    auto make_header_msg = [](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return msg;
    };

    // true once response_handler has already forwarded the headers
    auto headers_sent = std::make_shared<std::atomic<bool>>(false);

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, headers_sent, make_header_msg](const httplib::Response & response) {
        headers_sent->store(true);
        return pipe->write(make_header_msg(response)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    // when files are present, the body was converted from multipart form data to JSON
    // we need to reconstruct the multipart body for the downstream server
    std::string effective_body = body;
    std::string override_content_type;
    bool has_files = !files.empty();

    if (has_files) {
        json form_fields = json::parse_no_throw(body);
        if (!form_fields.is_discarded()) {
            auto boundary = generate_multipart_boundary();
            effective_body = build_multipart_body(form_fields, files, boundary);
            override_content_type = "multipart/form-data; boundary=" + boundary;
        } else {
            throw std::runtime_error("failed to parse multipart form fields JSON");
        }
    }

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding") {
                // disable Accept-Encoding to avoid compressed responses
                continue;
            }
            if (lowered == "transfer-encoding") {
                // the body is already decoded
                continue;
            }
            if (lowered == "content-length") {
                // let httplib calculate Content-Length from the actual body
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    // we set our own Content-Type with the new boundary
                    continue;
                }
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
                const std::string url_host = common_http_format_host(host);
                req.set_header(key, is_default_port ? url_host : url_host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = effective_body;
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req, headers_sent, make_header_msg]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, "", ""}); // header
            pipe->write({{}, 0, "proxy error: " + err_str, ""}); // body
        } else if (!headers_sent->load()) {
            // httplib skips response_handler for bodyless statuses like 204, send headers here instead
            pipe->write(make_header_msg(*result));
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            SRV_DBG("%s", "received response headers\n");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            SRV_DBG("%s", "no response headers received (request cancelled?)\n");
        }
    }
}
