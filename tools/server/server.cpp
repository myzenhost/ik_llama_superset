#include "server-connect.h"
#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"
#include "server-ollama.h"
#include "server-tgi.h"
#include "server-schema.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <exception>
#include <fstream>
#include <random>
#include <chrono>
#include <signal.h>
#include <sstream>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

// A native crash (access violation, illegal instruction, ...) used to end llama-server on Windows
// with no output at all: ggml_print_backtrace() is "platform not supported" there, and GGML_ABORT
// is the only path that prints. MEASURED 2026-09-26: a qwen35 -np 4 slot relaunch died with
// 0xC0000005 and an empty log. This filter prints the exception, the faulting address and a stack
// walked from the fault's OWN context (not the handler's), as module+offset and, when a PDB is
// present, symbol and line. It then returns EXCEPTION_CONTINUE_SEARCH so the process still ends
// exactly as it would have (same exit code, WER unchanged).
static LONG WINAPI server_unhandled_exception_filter(EXCEPTION_POINTERS * ep) {
    static volatile LONG entered = 0;
    if (InterlockedExchange(&entered, 1) != 0) {
        return EXCEPTION_CONTINUE_SEARCH; // a second fault while reporting: do not recurse
    }
    const EXCEPTION_RECORD * er = ep->ExceptionRecord;
    fprintf(stderr, "\n*** llama-server: unhandled exception 0x%08lX at %p (thread %lu)",
            (unsigned long) er->ExceptionCode, er->ExceptionAddress, (unsigned long) GetCurrentThreadId());
    if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2) {
        const ULONG_PTR op = er->ExceptionInformation[0];
        fprintf(stderr, ", %s address %p", op == 0 ? "reading" : op == 1 ? "writing" : "executing",
                (void *) er->ExceptionInformation[1]);
    }
    fprintf(stderr, "\n");

    HANDLE process = GetCurrentProcess();
    HANDLE thread  = GetCurrentThread();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS);
    const bool have_syms = SymInitialize(process, nullptr, TRUE) != FALSE;

    CONTEXT ctx = *ep->ContextRecord; // StackWalk64 modifies it
    STACKFRAME64 frame = {};
    frame.AddrPC.Offset    = ctx.Rip; frame.AddrPC.Mode    = AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp; frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp; frame.AddrStack.Mode = AddrModeFlat;

    alignas(SYMBOL_INFO) char sym_buf[sizeof(SYMBOL_INFO) + 512];
    for (int i = 0; i < 64; ++i) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &ctx, nullptr,
                         SymFunctionTableAccess64, SymGetModuleBase64, nullptr) || frame.AddrPC.Offset == 0) {
            break;
        }
        const DWORD64 pc   = frame.AddrPC.Offset;
        const DWORD64 base = SymGetModuleBase64(process, pc);
        char module[MAX_PATH] = "?";
        if (base) {
            GetModuleFileNameA((HMODULE) base, module, sizeof(module));
        }
        const char * mod_name = strrchr(module, '\\') ? strrchr(module, '\\') + 1 : module;
        fprintf(stderr, "  #%-2d %s+0x%llx", i, mod_name, (unsigned long long) (base ? pc - base : pc));
        if (have_syms) {
            auto * sym = (SYMBOL_INFO *) sym_buf;
            sym->SizeOfStruct = sizeof(SYMBOL_INFO);
            sym->MaxNameLen   = 511;
            DWORD64 disp = 0;
            if (SymFromAddr(process, pc, &disp, sym)) {
                fprintf(stderr, "  %s+0x%llx", sym->Name, (unsigned long long) disp);
            }
            IMAGEHLP_LINE64 line = {};
            line.SizeOfStruct = sizeof(line);
            DWORD ldisp = 0;
            if (SymGetLineFromAddr64(process, pc, &ldisp, &line)) {
                fprintf(stderr, "  [%s:%lu]", line.FileName, (unsigned long) line.LineNumber);
            }
        }
        fprintf(stderr, "\n");
    }
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

// ------------------------------------------------------------------
// Tensor hot-swap (docs/development/on-demand-tensor-reload.md), row S69-ENGINE-REVIEW-1016-SERVER-HOTSWAP-HOOK-01.
//   LLAMA_HOTSWAP_ENABLED       snapshot tensor sources at load (src/llama.cpp); GET /health reloads what changed on disk
//                               when no slot is processing (the hook of the unbuilt examples/server/server.cpp:803-817,
//                               ik_llama #2131 by Thireus, MIT).
//   LLAMA_HOTSWAP_CONTROL_FILE  with LLAMA_HOTSWAP_ENABLED: the file-signalled protocol of the same document, served from a
//   LLAMA_HOTSWAP_STATUS_FILE   helper thread. Status (ours, tmp + rename): "<state> <iter> <seq>", state = done |
//                               reload_failed | exiting; "done 1 0" once the model is loaded. Control (the driver's, read
//                               and deleted): "<command> [<seq>]". reload: reload on the main loop once no slot is
//                               processing, then drop every KV cache and cached prompt -> "done <iter+1> <seq>", or
//                               "reload_failed <iter> <seq>" when nothing changed. compute: a server evaluates on request,
//                               so only acknowledged -> "done <iter+1> <seq>". exit (quit, stop): "exiting <iter> <seq>",
//                               then the ordinary shutdown. A seq not above the last accepted one is ignored; a command
//                               without one always runs; a file with no newline yet is left for the next poll.
// ------------------------------------------------------------------
static bool hotswap_write_status(const std::string & path, const char * state, int iter, int seq, const std::atomic<bool> & stop) {
    // the driver's only view of our progress: retry until published (a reader must never see a partial line, so tmp +
    // rename; remove first because rename onto an existing file fails on Windows)
    const std::string tmp = path + ".tmp";
    for (long attempt = 0; ; ++attempt) {
        FILE * f = fopen(tmp.c_str(), "wb"); // binary: LF line ending on Windows too
        if (f) {
            // a short write (disk full) must not be published: check the write AND the close, and on failure drop the
            // temp and retry instead of renaming a partial line over the driver's view
            const bool wrote  = fprintf(f, "%s %d %d\n", state, iter, seq) > 0 && fflush(f) == 0 && !ferror(f);
            const bool closed = fclose(f) == 0;
            if (!wrote || !closed) {
                std::remove(tmp.c_str());
            } else {
                std::remove(path.c_str());
                if (std::rename(tmp.c_str(), path.c_str()) == 0) {
                    return true;
                }
            }
        }
        if (stop.load()) {
            return false;
        }
        if (attempt > 0 && attempt % 500 == 0) { // about every 10 s
            SRV_WRN("hotswap: still trying to publish status '%s' to '%s'\n", state, path.c_str());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

// true when a complete command was read (and the control file removed)
static bool hotswap_read_command(const std::string & path, std::string & cmd, int & seq) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    if (content.find('\n') == std::string::npos) {
        return false; // still being written
    }
    std::istringstream parser(content);
    cmd.clear();
    parser >> cmd;
    if (!(parser >> seq)) {
        seq = -1;
    }
    // remove it so it never runs twice; the writer may still hold it open for a moment on Windows
    for (int attempt = 0; attempt < 50 && std::remove(path.c_str()) != 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return !cmd.empty();
}

// post a reload to the main loop and wait for it: 1 = reloaded, 0 = nothing changed, -1 = stopped or failed
static int hotswap_reload_on_main_loop(server_context & ctx_server, const std::atomic<bool> & stop) {
    server_response_reader rd = ctx_server.get_response_reader();
    server_task task(SERVER_TASK_TYPE_HOTSWAP_RELOAD);
    task.id              = rd.get_new_id();
    task.hotswap_if_idle = false; // wait for the running requests to finish
    rd.post_task(std::move(task));
    auto result = rd.next([&stop]() { return stop.load(); });
    if (!result || result->is_error()) {
        return -1;
    }
    return result->to_json().value("reloaded", false) ? 1 : 0;
}

static void hotswap_signal_loop(server_context & ctx_server, const std::string & ctrl, const std::string & stat, std::atomic<bool> & stop) {
    int iter = 1; // advances on every accepted reload (that changed something) or compute
    int seq  = 0; // seq of the last accepted command
    std::remove(ctrl.c_str()); // drop a stale command from an earlier run
    SRV_INF("hotswap: signal mode enabled (control='%s', status='%s')\n", ctrl.c_str(), stat.c_str());
    hotswap_write_status(stat, "done", iter, seq, stop);
    while (!stop.load()) {
        std::string cmd;
        int cseq = -1;
        if (!hotswap_read_command(ctrl, cmd, cseq)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            continue;
        }
        if (cseq >= 0 && cseq <= seq) {
            SRV_INF("hotswap: ignoring stale command '%s' (seq %d <= %d)\n", cmd.c_str(), cseq, seq);
            continue;
        }
        if (cseq >= 0) {
            seq = cseq;
        }
        SRV_INF("hotswap: command received: %s\n", cmd.c_str());
        if (cmd == "exit" || cmd == "quit" || cmd == "stop") {
            SRV_INF("%s", "hotswap: exit requested, shutting down\n");
            hotswap_write_status(stat, "exiting", iter, seq, stop);
            if (shutdown_handler) {
                shutdown_handler(0);
            }
            return;
        }
        if (cmd == "reload") {
            const int r = hotswap_reload_on_main_loop(ctx_server, stop);
            if (r < 0) {
                return; // shutting down
            }
            if (r == 0) {
                SRV_WRN("%s", "hotswap: reload requested, but no tensor has changed on disk\n");
                hotswap_write_status(stat, "reload_failed", iter, seq, stop);
                continue;
            }
            hotswap_write_status(stat, "done", ++iter, seq, stop);
            continue;
        }
        if (cmd == "compute") {
            hotswap_write_status(stat, "done", ++iter, seq, stop);
            continue;
        }
        SRV_WRN("hotswap: unknown command '%s' ignored\n", cmd.c_str());
    }
}

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// how the server is run; everything is on when it is the llama-server binary
struct server_run_opts {
    bool install_signal_handlers = true; // off when embedded in llama-cli, which handles them itself
    bool resolve_model           = true; // download / resolve -hf and docker models; off when the caller did it already
};

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}


// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        int retry_after_s = 0; // > 0: answer 503 and tell the client when to come back
        try {
            return func(req);
        } catch (const server_unavailable_error & e) {
            error         = ERROR_TYPE_UNAVAILABLE;
            message       = e.what();
            retry_after_s = e.retry_after_s;
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const common_json_error & e) {
            // a body that is not valid JSON, or a field of the wrong type (common_json_error is a runtime_error, so it used to
            // fall through to the 500 below). SOURCE-READ: vllm/entrypoints/openai/chat_completion/api_router.py:41 (400 on a bad body)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = error_http_status(error_data, 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            if (retry_after_s > 0) {
                res->headers["Retry-After"] = std::to_string(retry_after_s);
            }
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

//
// shared by every role
//

static server_role_type server_role_detect(const common_params & params, server_role & role) {
    if (role.is_managed_by_router()) {
        return role.get_role();
    }
    // a server given no model is a router
    const bool has_model = !params.model.path.empty()
                        || !params.model.hf_repo.empty()
                        || !params.model.docker_repo.empty();
    return has_model ? SERVER_ROLE_INSTANCE : SERVER_ROLE_ROUTER;
}

// adjust the params for the role; returns false on a configuration error
static bool server_prepare_params(common_params & params, server_role_type role_type) {
    if (role_type != SERVER_ROLE_ROUTER) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }

        // decision branches fork from the trunk with llama_memory_seq_cp; a unified KV cache lets them share the trunk's
        // cells instead of copying them between per-sequence streams (fork, --decision-seqs)
        if (params.n_seq_decision > 0 && !params.kv_unified) {
            SRV_INF("--decision-seqs %d: enabling the unified KV cache\n", params.n_seq_decision);
            params.kv_unified = true;
        }
    }

    // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
    // or with -c 0 for max context
    const bool ctx_pool_auto_sized = params.kv_unified_per_slot > 0 &&
                                     params.n_ctx == 0 &&
                                     (uint32_t) params.fit_params_min_ctx != UINT32_MAX;

    if (ctx_pool_auto_sized) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
        SRV_INF("--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n", params.n_parallel,
                params.kv_unified_per_slot, params.n_ctx);
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // check early, so a missing llama-connect fails-fast
    if (params.server_connect) {
        const std::string reason = server_connect::unavailable_reason(params);
        if (!reason.empty()) {
            SRV_ERR("--connect is not available: %s\n", reason.c_str());
            return false;
        }
        params.server_connect_code = server_connect::resolve_code(params.server_connect_code);
    } else if (!params.server_connect_code.empty()) {
        // the code is only meaningful while the tunnel runs, and /props keys off it being set
        SRV_WRN("%s", "--connect-code is ignored without --connect\n");
        params.server_connect_code.clear();
    }

    return true;
}

// --disable-management-routes (fork): the routes that CHANGE the server or its files (model load/unload, control vectors,
// LoRA hotswap, slot save/restore/erase, saved prompts, the SQL session store, runtime props) answer 403 instead.
// Read-only introspection (/slots, /props, /metrics, /health, /control-vectors listing) stays open. A route that takes
// a server-side path is a way for anyone holding the API key (or anyone at all, when no key is set) to make the
// server read or write a file; on a shared host that is worth refusing outright. The switch is off by default, so
// nothing changes unless the operator asks.
// DISABLE-REVERSIBLE: --disable-management-routes (default off), row S69-Q9-PICK-MANAGEMENT-PLANE-DEFAULT-OFF-CONTROL-VECTOR-PATH-2026-09-30-01
// RESEARCH-DONE: pytorch/serve README ("token authorization enabled and model API control disabled by default")
static server_http_context::handler_t server_mgmt_route(const common_params & params, const server_http_context::handler_t & h) {
    if (!params.server_disable_management_routes) {
        return h;
    }
    return [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "management routes are disabled on this server (--disable-management-routes)"},
                {"type", "feature_disabled"},
                {"code", 403},
            }}
        });
        return res;
    };
}

// the API of every server; a router points the handlers in routes at its proxies before calling this
static void server_register_routes(server_http_context & ctx_http, server_routes & routes, const common_params & params,
                                   server_routes_ollama & routes_ollama, server_routes_tgi & routes_tgi) {
    auto mgmt = [&params](const server_http_context::handler_t & h) { return server_mgmt_route(params, h); };
    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/memory",                   ex_wrapper(routes.get_memory));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(mgmt(routes.post_props)));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/cancel",                   ex_wrapper(routes.post_control)); // body {"id": <cmpl id>, "action": "cancel"}
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/responses/compact",     ex_wrapper(routes.post_responses_compact));
    ctx_http.post("/responses/compact",        ex_wrapper(routes.post_responses_compact));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/tts",                      ex_wrapper(routes.post_tts));
    ctx_http.ws  ("/v1/realtime",              routes.ws_realtime);
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    // Ollama-native API shim (row S69-SERVING-SURFACE-OLLAMA-NATIVE-API-SHIM-2026-09-27-01):
    // registry CRUD (/api/create|pull|push|blobs) is deliberately NOT registered here, see server-ollama.h for why.
    ctx_http.post("/api/chat",                 ex_wrapper(routes_ollama.post_api_chat));
    ctx_http.post("/api/generate",             ex_wrapper(routes_ollama.post_api_generate));
    ctx_http.post("/api/embed",                ex_wrapper(routes_ollama.post_api_embed));
    ctx_http.post("/api/embeddings",           ex_wrapper(routes_ollama.post_api_embeddings_legacy));
    ctx_http.get ("/api/tags",                 ex_wrapper(routes_ollama.get_api_tags));
    ctx_http.post("/api/show",                 ex_wrapper(routes_ollama.post_api_show));
    ctx_http.get ("/api/ps",                   ex_wrapper(routes_ollama.get_api_ps));
    ctx_http.get ("/api/version",              ex_wrapper(routes_ollama.get_api_version));
    ctx_http.post("/api/copy",                 ex_wrapper(routes_ollama.post_api_copy));
    ctx_http.del ("/api/delete",               ex_wrapper(routes_ollama.del_api_delete));
    // TGI native API (server-tgi.h). GET "/" stays the web UI; only POST "/" is TGI's compat route.
    ctx_http.post("/generate",                 ex_wrapper(routes_tgi.post_generate));
    ctx_http.post("/generate_stream",          ex_wrapper(routes_tgi.post_generate_stream));
    ctx_http.post("/",                         ex_wrapper(routes_tgi.post_compat));
    ctx_http.get ("/info",                     ex_wrapper(routes_tgi.get_info));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/classify",                 ex_wrapper(routes.post_classify));
    ctx_http.post("/v1/classify",              ex_wrapper(routes.post_classify));
    // parallel constrained decisions (fork, --decision-seqs; SERVER_TASK_TYPE_PARALLEL_DECISION)
    ctx_http.post("/decision",                 ex_wrapper(routes.post_decision));
    ctx_http.post("/v1/decision",              ex_wrapper(routes.post_decision));
    // shared-prefix multi-candidate scoring (SERVER_TASK_TYPE_SCORE)
    ctx_http.post("/score",                    ex_wrapper(routes.post_score));
    ctx_http.post("/v1/score",                 ex_wrapper(routes.post_score));
    ctx_http.post("/v1/logits",                ex_wrapper(routes.post_logits));
    ctx_http.post("/v1/systemone",             ex_wrapper(routes.post_systemone));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/watermark/detect",         ex_wrapper(routes.post_watermark_detect));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    ctx_http.post("/chat_tokenize",            ex_wrapper(routes.post_chat_tokenize));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(mgmt(routes.post_lora_adapters)));
    // operator sampler overrides (server-overrides.h): GET shows the layers, POST replaces them (admin tier)
    ctx_http.get ("/sampler-overrides",        ex_wrapper(routes.get_sampler_overrides));
    ctx_http.post("/sampler-overrides",        ex_wrapper(mgmt(routes.post_sampler_overrides)));
    // Ported from the pre-merge fork's examples/server (row
    // S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01): G5 info + G1 runtime control vectors.
    ctx_http.get ("/version",                  ex_wrapper(routes.get_version));
    ctx_http.get ("/v1/props",                 ex_wrapper(routes.get_props_v1));
    ctx_http.get ("/control-vectors",          ex_wrapper(routes.get_control_vectors));
    ctx_http.post("/control-vectors/load",     ex_wrapper(mgmt(routes.post_control_vectors_load)));
    ctx_http.post("/control-vectors/unload",   ex_wrapper(mgmt(routes.post_control_vectors_unload)));
    ctx_http.post("/control-vectors/apply",    ex_wrapper(mgmt(routes.post_control_vectors_apply)));
    // G3: SQL session store (same row). Always registered; every handler reports a clear reason via
    // sql_unavailable_reason() when the feature was not compiled in, not requested, or failed to open.
    ctx_http.post("/load",                     ex_wrapper(mgmt(routes.post_sql_load)));
    ctx_http.post("/save",                     ex_wrapper(mgmt(routes.post_sql_save)));
    ctx_http.post("/rename",                   ex_wrapper(mgmt(routes.post_sql_rename)));
    ctx_http.post("/all",                      ex_wrapper(mgmt(routes.post_sql_all)));
    ctx_http.get ("/sessions",                 ex_wrapper(routes.get_sql_sessions));
    ctx_http.post("/sessions",                 ex_wrapper(mgmt(routes.post_sql_sessions)));
    ctx_http.post("/delete",                   ex_wrapper(mgmt(routes.post_sql_delete)));
    ctx_http.get ("/vacuum",                   ex_wrapper(mgmt(routes.get_sql_vacuum)));
    // G4 transparent zstd compression over the G3 store (--sqlite-zstd-ext-file)
    ctx_http.get ("/zstd_get_configs",             ex_wrapper(mgmt(routes.get_sql_zstd_configs)));
    ctx_http.post("/zstd_incremental_maintenance", ex_wrapper(mgmt(routes.post_sql_zstd_maintenance)));
    ctx_http.post("/zstd_enable_transparent",      ex_wrapper(mgmt(routes.post_sql_zstd_enable)));
    ctx_http.post("/zstd_update_transparent",      ex_wrapper(mgmt(routes.post_sql_zstd_update)));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(mgmt(routes.post_slots)));
    // G2: saved slot prompts (same row). /slots/list is always registered; the other three are gated inside their
    // handlers on params.slot_save_path, same as post_slots above.
    ctx_http.get ("/slots/list",               ex_wrapper(routes.get_slots_list));
    ctx_http.get ("/list",                     ex_wrapper(routes.get_saved_prompts));
    ctx_http.post("/delete_prompt",            ex_wrapper(mgmt(routes.post_delete_prompt)));
    ctx_http.post("/rename_prompt",            ex_wrapper(mgmt(routes.post_rename_prompt)));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();
}

// resumable streaming, see server-stream.h: an instance binds the local session factories,
// a router binds proxies that resolve the owning child
static void server_register_stream_routes(
        server_http_context & ctx_http,
        const server_http_context::handler_t & stream_get,
        const server_http_context::handler_t & streams_lookup,
        const server_http_context::handler_t & stream_delete) {
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete));
}

// the features behind a flag, and the warning about the ones that must not face untrusted clients
// returns false when one of them fails to start
static bool server_setup_features(
        server_http_context & ctx_http,
        const common_params & params,
        server_mcp & mcp_mgr,
        server_tools & tools,
        server_role_type role_type) {
    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)\n");
    }

    std::vector<std::string> warn_names;
    if (role_type == SERVER_ROLE_ROUTER) {
        warn_names.push_back("router mode");
    }

    if (params.server_connect) {
        warn_names.push_back("peer-to-peer tunnel (llama-connect, experimental)");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return false;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return false;
        }
        tools.output_budget_flag.store(params.server_tools_max_output);
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("server tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    if (!warn_names.empty()) {
        std::string features;
        for (const auto & name : warn_names) {
            if (!features.empty()) features += ", ";
            features += name;
        }
        SRV_WRN("security: %s enabled - do not expose to untrusted environments\n", features.c_str());
    }

    return true;
}

static void server_install_signal_handlers() {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    // CTRL_BREAK too: it is the only console event that can be aimed at one process group
    // (GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid)), so it is how a supervisor stops a server it
    // started with CREATE_NEW_PROCESS_GROUP -- a group created that way has Ctrl-C switched off. Without
    // this the default handler ran ExitProcess and skipped the clean-up (router: unload_all, snapshot).
    // MEASURED 2026-10-18 (row S69-MSNAP-E2E-CONSOLE-HOST-IGNORED-CTRL-C-2026-10-18-01).
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
}

// once the HTTP server listens: log the addresses and spawn llama-connect if asked, so the port is
// bound and the child startup health check reaches us. the destructor of connect_proc stops it
static bool server_on_listening(server_http_context & ctx_http, const common_params & params, server_connect & connect_proc) {
    for (const auto & address : ctx_http.listening_addresses) {
        SRV_INF("listening on %s\n", address.c_str());
    }

    if (params.server_connect && !connect_proc.start(params, ctx_http.port)) {
        SRV_ERR("%s", "exiting due to llama-connect error\n");
        return false;
    }
    return true;
}

//
// download worker: a child of the router that pulls a model into the cache and reports the progress
//

struct server_download_state : public common_download_callback {
    server_role * role;
    std::function<bool()> should_stop;
    std::atomic<int64_t> last_progress_time{0}; // multiple files downloading in different threads
    bool is_ok = false;

    server_download_state(server_role * r) : role(r) {}

    bool run(common_params & params) {
        try {
            common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            common_models_handler_apply(handler, params, this);
            is_ok = true;
        } catch (const std::exception & e) {
            auto model_name = params.model.get_name();
            SRV_ERR("download failed for model name=%s: %s\n", model_name.c_str(), e.what());
            is_ok = false;
        }
        return is_ok;
    }
    void on_progress(const common_download_progress & p) {
        json data = {
            {"url", p.url},
            {"downloaded", p.downloaded},
            {"total", p.total},
        };
        role->notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), data);
    }
    void on_start(const common_download_progress & p) override {
        on_progress(p);
    }
    void on_update(const common_download_progress & p) override {
        int64_t now = ggml_time_ms();
        // throttle progress updates to avoid flooding logs
        if (now - last_progress_time.load(std::memory_order_relaxed) >= 100) {
            on_progress(p);
            last_progress_time.store(now, std::memory_order_relaxed);
        }
    }
    void on_done(const common_download_progress & p, bool) override {
        on_progress(p);
    }
    bool is_cancelled() const override {
        return should_stop ? should_stop() : false;
    }
};

static int server_run_download(common_params & params, server_role & role) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    // monitor stdin for cancellation command from the router
    std::thread signal_thread = role.setup([cancelled](int) {
        cancelled->store(true, std::memory_order_relaxed);
    });

    server_download_state dl(&role);
    dl.should_stop = [cancelled]() {
        return cancelled->load(std::memory_order_relaxed);
    };

    bool ok = dl.run(params);

    role.notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), {
        {"result", ok ? "download_finished" : "download_failed"},
    });

    // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
    if (signal_thread.joinable()) {
        signal_thread.join();
    }

    SRV_INF("download completed %s\n", ok ? "successfully" : "with errors");
    return 0;
}

//
// router
//

static int server_run_router(common_params & params, int argc, char ** argv, const server_run_opts & opts) {
    SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // the router loads no model, its context stays empty; the routes it does not proxy still need one
    server_context ctx_server;
    server_routes  routes(params, ctx_server);
    server_tools   tools;
    const bool is_router_server = true; // read by the fork blocks below (the router resolves model names itself)
    server_routes_ollama routes_ollama(routes); // ollama-native API shim, row
                                                 // S69-SERVING-SURFACE-OLLAMA-NATIVE-API-SHIM-2026-09-27-01
    {
        server_ollama_config ocfg;
        ocfg.modelfile_path = params.ollama_modelfile;
        ocfg.strict_names   = params.ollama_strict_names && !is_router_server; // router mode resolves names itself
        ocfg.router_mode    = true;  // options.num_ctx -> per-request child reload (server-models.cpp apply_num_ctx)
        ocfg.store_dir      = params.ollama_models_dir;
        ocfg.store_model    = params.ollama_model;
        ocfg.store_write    = params.ollama_store_write;
        try {
            routes_ollama.configure(ocfg);
        } catch (const std::exception & e) {
            SRV_ERR("--modelfile: %s\n", e.what());
            return 1;
        }
    }
    server_routes_tgi routes_tgi(routes); // TGI native API, row S69-TGI-PICK-NATIVE-GENERATE-ROUTES-2026-10-18-01

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    // setup server instances manager
    std::optional<server_models_routes> models_routes;
    try {
        models_routes.emplace(params, argc, argv);
    } catch (const std::exception & e) {
        SRV_ERR("failed to initialize router models: %s\n", e.what());
        return 1;
    }

    // proxy handlers
    // note: routes.get_health stays the same
    routes.get_metrics                 = models_routes->proxy_get;
    routes.get_memory                  = models_routes->proxy_get;
    routes.post_props                  = models_routes->proxy_post;
    routes.post_completions            = models_routes->proxy_post;
    routes.post_completions_oai        = models_routes->proxy_post;
    routes.post_chat_completions       = models_routes->proxy_post;
    routes.post_control                = models_routes->proxy_post;
    routes.post_responses_oai          = models_routes->proxy_post;
    routes.post_responses_compact      = models_routes->proxy_post;
    routes.post_transcriptions_oai     = models_routes->proxy_post;
    routes.post_tts                    = models_routes->proxy_post;
    routes.ws_realtime                 = models_routes->proxy_ws;
    routes.post_anthropic_messages     = models_routes->proxy_post;
    routes.post_anthropic_count_tokens = models_routes->proxy_post;
    routes.post_infill                 = models_routes->proxy_post;
    routes.post_embeddings             = models_routes->proxy_post;
    routes.post_embeddings_oai         = models_routes->proxy_post;
    routes.post_rerank                 = models_routes->proxy_post;
    routes.post_classify               = models_routes->proxy_post;
    routes.post_decision               = models_routes->proxy_post;
    routes.post_score                  = models_routes->proxy_post;
    routes.post_logits                 = models_routes->proxy_post;
    routes.post_systemone              = models_routes->proxy_post;
    routes.post_tokenize               = models_routes->proxy_post;
    routes.post_detokenize             = models_routes->proxy_post;
    routes.post_watermark_detect       = models_routes->proxy_post;
    routes.post_apply_template         = models_routes->proxy_post;
    routes.post_chat_tokenize          = models_routes->proxy_post;
    routes.post_chat_completions_tok   = models_routes->proxy_post;
    routes.post_responses_tok_oai      = models_routes->proxy_post;
    routes.get_lora_adapters           = models_routes->proxy_get;
    routes.post_lora_adapters          = models_routes->proxy_post;
    routes.get_sampler_overrides       = models_routes->proxy_get;
    routes.post_sampler_overrides      = models_routes->proxy_post;
    routes.get_slots                   = models_routes->proxy_get;
    routes.post_slots                  = models_routes->proxy_post;
    routes.get_slots_list              = models_routes->proxy_get;
    routes.get_saved_prompts           = models_routes->proxy_get;
    routes.post_delete_prompt          = models_routes->proxy_post;
    routes.post_rename_prompt          = models_routes->proxy_post;
    // Ported routes (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01)
    routes.get_version                 = models_routes->proxy_get;
    routes.get_props_v1                = models_routes->proxy_get;
    routes.get_control_vectors         = models_routes->proxy_get;
    routes.post_control_vectors_load   = models_routes->proxy_post;
    routes.post_control_vectors_unload = models_routes->proxy_post;
    routes.post_control_vectors_apply  = models_routes->proxy_post;
    // G3: SQL session store.
    routes.post_sql_load               = models_routes->proxy_post;
    routes.post_sql_save               = models_routes->proxy_post;
    routes.post_sql_rename             = models_routes->proxy_post;
    routes.post_sql_all                = models_routes->proxy_post;
    routes.get_sql_sessions            = models_routes->proxy_get;
    routes.post_sql_sessions           = models_routes->proxy_post;
    routes.post_sql_delete             = models_routes->proxy_post;
    routes.get_sql_vacuum              = models_routes->proxy_get;
    routes.get_sql_zstd_configs        = models_routes->proxy_get;
    routes.post_sql_zstd_maintenance   = models_routes->proxy_post;
    routes.post_sql_zstd_enable        = models_routes->proxy_post;
    routes.post_sql_zstd_update        = models_routes->proxy_post;

    // custom routes for router
    routes.get_props                   = models_routes->get_router_props;
    routes.get_models                  = models_routes->get_router_models;
    // Ollama shim: empty /api/chat messages or /api/generate prompt = load, keep_alive 0 = unload (the router owns the children)
    routes.ollama_model_load           = models_routes->ollama_model_load;
    routes.ollama_model_unload         = models_routes->ollama_model_unload;

    auto mgmt = [&params](const server_http_context::handler_t & h) { return server_mgmt_route(params, h); };
    ctx_http.post("/models",               ex_wrapper(mgmt(models_routes->post_router_models)));
    ctx_http.post("/models/load",          ex_wrapper(mgmt(models_routes->post_router_models_load)));
    ctx_http.post("/models/unload",        ex_wrapper(mgmt(models_routes->post_router_models_unload)));
    ctx_http.post("/models/estimate",      ex_wrapper(models_routes->post_router_models_estimate)); // read-only: loads headers, never weights
    ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
    ctx_http.get ("/models/events",        ex_wrapper(models_routes->get_router_models_events));
    ctx_http.del ("/models",               ex_wrapper(mgmt(models_routes->del_router_models)));

    server_register_routes(ctx_http, routes, params, routes_ollama, routes_tgi);
    server_register_stream_routes(ctx_http,
        models_routes->router_stream_get,
        models_routes->router_streams_lookup,
        models_routes->router_stream_delete);

    if (!server_setup_features(ctx_http, params, mcp_mgr, tools, SERVER_ROLE_ROUTER)) {
        return 1;
    }
    // The router runs /tools itself (server_setup_features registers handle_post above; nothing forwards it to a child) and loads no
    // model, so the output budget cannot come from a loaded context: ask the children per call -- the one the body's optional "model"
    // names, else the largest running one (server_models::tools_slot_ctx). --tools-max-output still wins.
    tools.n_ctx_provider = [&models_routes](const std::string & model) -> int64_t {
        return models_routes->models.tools_slot_ctx(model);
    };

    auto clean_up = [&models_routes, &mcp_mgr]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        // stop the session GC first, it finalizes live sessions and wakes pending readers
        server_stream_session_manager_stop();
        if (models_routes.has_value()) {
            models_routes->stopping.store(true); // maybe redundant, but just to be safe
            models_routes->models.unload_all();
        }
        mcp_mgr.shutdown();
        llama_backend_free();
    };

    if (!ctx_http.start()) {
        clean_up();
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }
    ctx_http.is_ready.store(true);

    shutdown_handler = [&](int sig) {
        // each step is logged: a router that did not stop on Ctrl-C left no line saying how far it got
        // (row S69-MSNAP-E2E-CONSOLE-HOST-IGNORED-CTRL-C-2026-10-18-01)
        SRV_INF("router: shutdown requested (signal %d)\n", sig);
        ctx_http.drain.begin();
        if (models_routes.has_value()) {
            // important to disconnect any SSE clients
            models_routes->stopping.store(true);
        }
        mcp_mgr.shutdown();
        SRV_INF("%s", "router: MCP stopped, stopping the HTTP server\n");
        ctx_http.stop();
        SRV_INF("%s", "router: HTTP server stop requested\n");
    };

    try {
        models_routes->models.load_startup_models();
    } catch (const std::exception & e) {
        SRV_ERR("failed to load models on startup: %s\n", e.what());
        ctx_http.stop();
        ctx_http.join();
        clean_up();
        return 1;
    }

    if (opts.install_signal_handlers) {
        server_install_signal_handlers();
    }

    server_connect connect_proc;
    if (!server_on_listening(ctx_http, params, connect_proc)) {
        ctx_http.stop();
        ctx_http.join();
        clean_up();
        return 1;
    }

    if (!params.models_preset_hf.empty()) {
        SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
        SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
    }

    ctx_http.join(); // keep the main thread alive
    SRV_INF("%s", "router: HTTP server thread ended\n");

    // when the HTTP server stops, clean up and exit
    clean_up();

    return 0;
}

//
// instance
//

static int server_run_instance(common_params & params, server_role & role, common_models_handler & models_handler, const server_run_opts & opts) {
    // -hf / docker: download if needed and resolve the local path. a child resolves what the router downloaded
    if (opts.resolve_model) {
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;
    server_routes  routes(params, ctx_server);
    server_tools   tools;
    const bool is_router_server = false; // read by the fork blocks below
    server_routes_ollama routes_ollama(routes); // ollama-native API shim, row
                                                 // S69-SERVING-SURFACE-OLLAMA-NATIVE-API-SHIM-2026-09-27-01
    {
        server_ollama_config ocfg;
        ocfg.modelfile_path = params.ollama_modelfile;
        ocfg.strict_names   = params.ollama_strict_names && !is_router_server; // router mode resolves names itself
        ocfg.store_dir      = params.ollama_models_dir;
        ocfg.store_model    = params.ollama_model;
        ocfg.store_write    = params.ollama_store_write;
        try {
            routes_ollama.configure(ocfg);
        } catch (const std::exception & e) {
            SRV_ERR("--modelfile: %s\n", e.what());
            return 1;
        }
    }
    server_routes_tgi routes_tgi(routes); // TGI native API, row S69-TGI-PICK-NATIVE-GENERATE-ROUTES-2026-10-18-01

    // --sampler-overrides / --sampler-overrides-model: operator layers over request fields (server-overrides.h). A router does
    // not apply them itself; each model instance it spawns gets the flags and loads the files. A bad file stops start-up: a
    // server that silently runs without the limits its operator wrote is worse than one that does not start.
    if (!is_router_server) {
        const std::pair<const std::string *, bool> files[] = {{&params.sampler_overrides_file, false}, {&params.sampler_overrides_model_file, true}};
        for (const auto & f : files) {
            if (f.first->empty()) {
                continue;
            }
            std::ifstream in(*f.first, std::ios::binary);
            if (!in) {
                SRV_ERR("cannot read the sampler overrides file '%s'\n", f.first->c_str());
                return 1;
            }
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            const json spec = json::parse_no_throw(text);
            std::vector<std::string> warnings;
            std::string err;
            server_override_layer layer;
            if (spec.is_discarded() || !server_override_parse(spec, server_schema::override_groups(), "'" + *f.first + "'", layer, err, warnings)) {
                SRV_ERR("sampler overrides file '%s': %s\n", f.first->c_str(), spec.is_discarded() ? "not valid JSON" : err.c_str());
                return 1;
            }
            for (const auto & w : warnings) {
                SRV_WRN("%s\n", w.c_str());
            }
            const size_t n_layer = layer.size();
            if (f.second) { server_override_instance().set_model(std::move(layer)); }
            else          { server_override_instance().set_global(std::move(layer)); }
            SRV_INF("sampler overrides (%s layer) loaded from '%s': %zu entr%s\n", f.second ? "model" : "global", f.first->c_str(), n_layer, n_layer == 1 ? "y" : "ies");
        }
    }

    // --warm-prefix-file (row S69-rtp-llm-PICK-STARTUP-WARMED-PINNED-PREFIX-CACHE-2026-09-29-01): read and checked BEFORE the model
    // loads, so a bad file or a missing cache stops start-up at once (an error, not a quiet downgrade to a cold start).
    struct warm_item {
        std::string id;
        std::string path;
        json        body;
    };
    std::vector<warm_item> warm_items;
    std::string warm_token;
    if (!is_router_server && !params.warm_prefix_file.empty()) {
        if (params.cache_ram_mib == 0) {
            SRV_ERR("%s", "--warm-prefix-file needs the host prompt cache: --cache-ram 0 disables it\n");
            return 1;
        }
        std::ifstream in(params.warm_prefix_file, std::ios::binary);
        if (!in) {
            SRV_ERR("cannot read the warm prefix file '%s'\n", params.warm_prefix_file.c_str());
            return 1;
        }
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const json spec = json::parse_no_throw(text);
        if (spec.is_discarded() || !spec.is_array() || spec.begin() == spec.end()) {
            SRV_ERR("warm prefix file '%s': expected a non-empty JSON list of {\"id\", \"messages\" | \"prompt\"}\n", params.warm_prefix_file.c_str());
            return 1;
        }
        size_t idx = 0;
        for (const auto & e : spec) {
            const std::string where = "warm prefix file '" + params.warm_prefix_file + "' entry " + std::to_string(idx++);
            const bool has_msgs = e.is_object() && e.contains("messages");
            const bool has_prompt = e.is_object() && e.contains("prompt");
            if (!e.is_object() || has_msgs == has_prompt) {
                SRV_ERR("%s: needs exactly one of \"messages\" (a list) or \"prompt\" (a string)\n", where.c_str());
                return 1;
            }
            warm_item w;
            w.id = json_value(e, "id", std::string("entry-") + std::to_string(idx - 1));
            w.body = json::object();
            if (has_msgs) {
                if (!e.at("messages").is_array() || e.at("messages").begin() == e.at("messages").end()) {
                    SRV_ERR("%s: \"messages\" must be a non-empty list\n", where.c_str());
                    return 1;
                }
                w.path = "/v1/chat/completions";
                w.body["messages"] = e.at("messages");
                w.body["max_tokens"] = 1;
            } else {
                if (!e.at("prompt").is_string() || e.at("prompt").get<std::string>().empty()) {
                    SRV_ERR("%s: \"prompt\" must be a non-empty string\n", where.c_str());
                    return 1;
                }
                w.path = "/completion";
                w.body["prompt"] = e.at("prompt");
                w.body["n_predict"] = 1;
            }
            if (e.contains("tools")) {
                w.body["tools"] = e.at("tools");
            }
            w.body["temperature"] = 0;
            w.body["stream"] = false;
            w.body["cache_prompt"] = true;
            warm_items.push_back(std::move(w));
        }
        std::random_device rd;
        std::mt19937_64 gen(((uint64_t) rd() << 32) ^ rd());
        char tok[40];
        snprintf(tok, sizeof(tok), "%016llx%016llx", (unsigned long long) gen(), (unsigned long long) gen());
        warm_token = tok;
        for (auto & w : warm_items) {
            w.body["cache_pin_token"] = warm_token;
        }
        SRV_INF("warm prefix file '%s': %zu entr%s, pinned prefixes may hold %.0f%% of --cache-ram\n", params.warm_prefix_file.c_str(),
                warm_items.size(), warm_items.size() == 1 ? "y" : "ies", 100.0 * (double) params.warm_prefix_share);
    }
    std::atomic<int> warm_exit_code{0};
    std::thread warm_thread;


    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    server_register_routes(ctx_http, routes, params, routes_ollama, routes_tgi);
    server_register_stream_routes(ctx_http,
        server_stream_make_get_handler(),
        server_stream_make_lookup_handler(),
        server_stream_make_delete_handler());

    if (!server_setup_features(ctx_http, params, mcp_mgr, tools, SERVER_ROLE_INSTANCE)) {
        return 1;
    }

    // setup clean up function, to be called before exit
    auto clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        // stop the session GC first, it finalizes live sessions and wakes pending readers
        server_stream_session_manager_stop();
        ctx_http.stop();
        ctx_server.terminate();
        mcp_mgr.shutdown();
        llama_backend_free();
    };

    // hot-swap signal mode needs both files (hotswap_signal_loop); refuse before anything starts
    {
        const char * hs_ctrl = std::getenv("LLAMA_HOTSWAP_CONTROL_FILE");
        const char * hs_stat = std::getenv("LLAMA_HOTSWAP_STATUS_FILE");
        if (std::getenv("LLAMA_HOTSWAP_ENABLED") && hs_ctrl && *hs_ctrl && !(hs_stat && *hs_stat)) {
            SRV_ERR("%s", "LLAMA_HOTSWAP_CONTROL_FILE requires LLAMA_HOTSWAP_STATUS_FILE to be set as well\n");
            clean_up();
            return 1;
        }
    }

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!ctx_http.start()) {
        clean_up();
        SRV_ERR("%s", "exiting due to HTTP server error\n");
        return 1;
    }

    // setup communication child --> router if necessary
    if (role.is_managed_by_router()) {
        ctx_server.set_state_callback([&](server_state state, json payload) {
            role.notify_to_router(server_state_to_str(state), payload);
        });
    }

    // the tools' output budget is a share of the loaded context (server-tools.cpp, server_tools_output_budget); load_model() runs
    // again on every wake from --sleep-idle-seconds, so the budget is recomputed from the callback, not once after the first load.
    // (A LLAMA_HOTSWAP tensor reload rewrites weights in place and keeps the context, so it needs no update.)
    ctx_server.set_model_loaded_callback([&tools](int n_ctx_slot) {
        tools.n_ctx_seq.store(n_ctx_slot);
    });

    // load_model() can throw (e.g. common_model_params_to_llama rejects a quantized V cache without flash attention);
    // an escaped exception killed the process with no message. Report what() and take the ordinary error exit.
    bool model_loaded = false;
    try {
        model_loaded = ctx_server.load_model(params);
    } catch (const std::exception & e) {
        SRV_ERR("model loading raised an exception: %s\n", e.what());
    }
    if (!model_loaded) {
        clean_up();
        ctx_http.join();
        SRV_ERR("%s", "exiting due to model loading error\n");
        return 1;
    }

    if (!warm_token.empty()) {
        ctx_server.set_pin_token(warm_token);
    }

    routes.update_meta(ctx_server);
    routes.hotswap_on_health.store(std::getenv("LLAMA_HOTSWAP_ENABLED") != nullptr);
    ctx_http.is_ready.store(true);

    SRV_INF("%s", "model loaded\n");

    shutdown_handler = [&](int sig) {
        if (ctx_http.drain.begin()) {
            SRV_INF("shutdown requested (signal %d): draining, new requests and /health now answer 503\n", sig);
        }
        mcp_mgr.shutdown();
        // this will unblock start_loop()
        ctx_server.terminate();
    };

    if (opts.install_signal_handlers) {
        server_install_signal_handlers();
    }

    server_connect connect_proc;
    if (!server_on_listening(ctx_http, params, connect_proc)) {
        ctx_http.stop();
        ctx_http.join();
        clean_up();
        return 1;
    }

    // optionally, notify router server that this instance is ready
    std::thread monitor_thread;
    if (role.is_managed_by_router()) {
        monitor_thread = role.setup(shutdown_handler);
        role.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
    }

    // --warm-prefix-file: run each prefix through the normal request path (chat template, tools and all) from a helper thread
    // while the main thread serves the queue; a request that carries the secret token gets its prompt state pinned
    if (!warm_items.empty()) {
        warm_thread = std::thread([&]() {
            const std::function<bool()> never = []() { return false; };
            size_t n_done = 0;
            for (const auto & w : warm_items) {
                const size_t before_passes = ctx_server.pin_passes();
                server_http_res_ptr res;
                {
                    server_http_req req { {}, {}, w.path, "", w.body.dump(), {}, never };
                    const auto & handler = w.path == "/completion" ? routes.post_completions : routes.post_chat_completions;
                    res = ex_wrapper(handler)(req);
                }
                if (!res || res->status != 200) {
                    SRV_ERR("warm prefix '%s' failed (HTTP %d): %s\n", w.id.c_str(), res ? res->status : 0, res ? res->data.substr(0, 300).c_str() : "no response");
                    warm_exit_code.store(1);
                    shutdown_handler(0);
                    return;
                }
                // the slot loop pins after the task ends (also when the server goes idle at once); wait for that pin pass to finish -- it
                // counts passes, so an already-pinned prefix (which changes no byte count) is not mistaken for a pass that never ran
                for (int i = 0; i < 400 && ctx_server.pin_passes() == before_passes; ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (ctx_server.pin_passes() == before_passes) {
                    SRV_ERR("warm prefix '%s': no pin pass ran within 20 s of its request ending\n", w.id.c_str());
                    warm_exit_code.store(1);
                    shutdown_handler(0);
                    return;
                }
                if (ctx_server.pin_overflowed() || !ctx_server.last_pin_covered()) {
                    SRV_ERR("warm prefix '%s': the pinned prefixes do not fit in %.0f%% of --cache-ram%s; raise --cache-ram or --warm-prefix-share, or shorten the prefixes\n",
                            w.id.c_str(), 100.0 * (double) params.warm_prefix_share,
                            ctx_server.pin_overflowed() ? "" : " (this prefix's prompt state is larger than the whole host prompt cache and was not kept)");
                    warm_exit_code.store(1);
                    shutdown_handler(0);
                    return;
                }
                n_done++;
                SRV_INF("warm prefix '%s' done (%zu of %zu): %zu pinned entr%s, %.1f MiB\n", w.id.c_str(), n_done, warm_items.size(),
                        ctx_server.pinned_entries(), ctx_server.pinned_entries() == 1 ? "y" : "ies", ctx_server.pinned_bytes() / (1024.0 * 1024.0));
            }
            SRV_INF("warm prefixes: all %zu done\n", n_done);
        });
    }

    // tensor hot-swap, file-signalled mode (see hotswap_signal_loop); the env pairing was checked before the model load
    std::atomic<bool> hotswap_stop{false};
    std::thread hotswap_thread;
    {
        const char * hs_ctrl = std::getenv("LLAMA_HOTSWAP_CONTROL_FILE");
        const char * hs_stat = std::getenv("LLAMA_HOTSWAP_STATUS_FILE");
        if (std::getenv("LLAMA_HOTSWAP_ENABLED") && hs_ctrl && *hs_ctrl && hs_stat && *hs_stat) {
            hotswap_thread = std::thread(hotswap_signal_loop, std::ref(ctx_server), std::string(hs_ctrl), std::string(hs_stat),
                                         std::ref(hotswap_stop));
        }
    }

    // this call blocks the main thread until queue_tasks.terminate() is called
    ctx_server.start_loop();

    hotswap_stop.store(true);
    if (hotswap_thread.joinable()) {
        hotswap_thread.join();
    }
    if (warm_thread.joinable()) {
        warm_thread.join();
    }

    clean_up();
    ctx_http.join();
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }

    auto * ll_ctx = ctx_server.get_llama_context();
    if (ll_ctx != nullptr) {
        common_memory_breakdown_print(ll_ctx);
    }

    return warm_exit_code.load();
}

//
// entry points
//

static int server_run(common_params & params, int argc, char ** argv, server_role & role, const server_run_opts & opts) {
    common_models_handler models_handler;

    // -hf may name a preset repo, which makes this a router: settle that before deciding the role
    if (opts.resolve_model && !params.model.hf_repo.empty()) {
        try {
            models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            if (common_models_handler_is_preset_repo(models_handler)) {
                // apply the preset and start the server in router mode
                common_models_handler_apply(models_handler, params);
            }
        } catch (const std::exception & e) {
            SRV_ERR("failed to fetch model metadata: %s\n", e.what());
            return 1;
        }
    }

    const server_role_type role_type = server_role_detect(params, role);

    // only an instance loads a model; the others must not touch the GPU, so skip the
    // device enumeration for them and the CUDA primary context stays uncreated
    common_params_print_info(params, role_type == SERVER_ROLE_INSTANCE);

    if (!server_prepare_params(params, role_type)) {
        return 1;
    }

    switch (role_type) {
        case SERVER_ROLE_DOWNLOAD: return server_run_download(params, role);
        case SERVER_ROLE_ROUTER:   return server_run_router(params, argc, argv, opts);
        case SERVER_ROLE_INSTANCE: return server_run_instance(params, role, models_handler, opts);
    }

    GGML_ABORT("invalid server role");
}

int llama_server(int argc, char ** argv) {
    server_role role;

    std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
    // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
    signal(SIGPIPE, SIG_IGN);
#else
    // report native crashes instead of dying silently -- see server_unhandled_exception_filter
    SetUnhandledExceptionFilter(server_unhandled_exception_filter);
#endif

    // own arguments required by this example
    common_params params;

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    server_stream_session_manager_start();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    SRV_INF("%s", "initializing ...\n");

    llama_backend_init();
    llama_numa_init(params.numa);

    const int result = server_run(params, argc, argv, role, server_run_opts{});
    common_log_flush(common_log_main());
    return result;
}

int llama_server(common_params & params, int argc, char ** argv) {
    // embedded in llama-cli: it handles the signals, and the model is resolved already
    server_run_opts opts;
    opts.install_signal_handlers = false;
    opts.resolve_model           = false;

    server_role role;
    return server_run(params, argc, argv, role, opts);
}
