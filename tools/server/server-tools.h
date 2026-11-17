#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"
#include "server-mcp.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct server_tool {
    std::string name;
    std::string display_name;
    bool permission_write = false;
    bool support_stream = false; // if true, output can be streamed
    bool uses_cwd = false;       // if true, the tool resolves paths and runs against the working directory

    virtual ~server_tool() = default;
    virtual json get_definition() const = 0;
    virtual std::string type() const { return "server"; }

    struct stream {
        server_response & qr;
        int id;
        std::function<bool()> alive;
        void push(const std::string & chunk);
    };
    virtual json invoke(json params, stream * st = nullptr) const = 0;

    json to_json() const;
};

struct server_tools_runtime; // impl detail, defined in server-tools.cpp

// bytes one tool call may hand back to the model (read_file, exec_shell_command, get_info output; the
// file_glob_search / grep_search listings). `flag_bytes` is --tools-max-output (0 = unset); `n_ctx_seq` is the
// per-slot context the model was loaded with (0 = no model reachable, e.g. the router process)
size_t server_tools_output_budget(int n_ctx_seq, size_t flag_bytes);
#define SERVER_TOOLS_HAS_CTX_PROVIDER 1 // tests/test-server-tools-budget-swap.cpp keys its old-logic build on this

// router process: pick the per-slot context a tool call's budget comes from out of the running children's contexts
// (name, n_ctx_seq). `model` names the child the call routes to (body field "model"); an empty or unknown name means no route
// is known and the LARGEST running context is used (so a call is never cut shorter than the roomiest child could take).
// 0 = no running child (the 16 KiB fallback applies).
int64_t server_tools_pick_ctx(const std::vector<std::pair<std::string, int64_t>> & running, const std::string & model);

struct server_tools {
    std::vector<std::unique_ptr<server_tool>> tools;

    // for streaming
    server_response queue_res;
    std::atomic<int> res_id{0};

    // set when --tools-runtime is configured; routes every tool call through an isolate
    std::unique_ptr<server_tools_runtime> runtime;

    // output budget inputs: set once the model is loaded (n_ctx_seq) and from --tools-max-output (flag)
    std::atomic<int>    n_ctx_seq{0};
    std::atomic<size_t> output_budget_flag{0};
    // router: no model is loaded here, so the context is asked per call (the child the call routes to, else the largest running
    // child). Empty = use n_ctx_seq. Called from request threads; the owner makes it thread-safe.
    std::function<int64_t(const std::string & model)> n_ctx_provider;
    size_t output_budget(const std::string & model = std::string()) const {
        int64_t n = n_ctx_provider ? n_ctx_provider(model) : 0;
        if (n <= 0) {
            n = n_ctx_seq.load();
        }
        return server_tools_output_budget((int) std::min<int64_t>(n, INT32_MAX), output_budget_flag.load());
    }

    void setup(const std::vector<std::string> & enabled_tools,
               server_mcp & mcp_mgr,
               const std::string & tools_runtime);

    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;

    server_tools();
    ~server_tools();
};
