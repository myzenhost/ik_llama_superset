// Child log line -> "this is an out-of-memory failure" (tools/server/server-oom-line.h), model-free, header-only.
// Row S69-ENGREV-JAN-OOM-LINE-BARE-FAILED-TO-ALLOCATE-2026-10-27-01. Build by hand: g++ -std=c++17 -I.. test_server_oom_line.cpp
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-oom-line.h"

#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    n_fail += ok ? 0 : 1;
}

// the rule line_reports_oom() had before: a bare substring list. Kept only to show the cases below tell the two rules apart.
static bool old_rule(const std::string & line) {
    std::string s = line;
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    for (const char * n : { "out of memory", "cudamalloc failed", "failed to allocate", "unable to allocate", "outofdevicememory", "out of device memory" }) {
        if (s.find(n) != std::string::npos) return true;
    }
    return false;
}

int main() {
    // lines this tree really prints for an exhausted allocation: every one must stay an OOM
    const std::vector<std::string> oom = {
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 12000.00 MiB on device 0: cudaMalloc failed: out of memory",
        "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 1073741824",
        "alloc_tensor_range: failed to allocate CUDA0 buffer of size 4294967296",
        "llama_init_from_model: failed to allocate compute buffers",
        "llama_kv_cache_init: failed to allocate buffer for kv cache",
        "graph_reserve: failed to allocate compute pp buffers",
        "llama_new_context_with_model: failed to allocate graph",
        "ggml_backend_sched_alloc_splits: failed to allocate 65536 bytes",
        "llama_model_load: error loading model: unable to allocate backend buffer",
        "ggml_vulkan: Device memory allocation of size 123 failed. vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
        "CUDA error: out of memory",
        "output buffer: failed to allocate output buffer of size 12.00 MiB",
    };
    for (const auto & l : oom) {
        check(server_line_reports_oom(l), "OOM kept: " + l.substr(0, 70));
    }
    // lines that mention the phrase but are not memory exhaustion: must not arm the OOM retry
    const std::vector<std::string> not_oom = {
        "ggml_cuda_host_malloc: failed to allocate 512.00 MiB of pinned memory: invalid argument",       // non-fatal WDDM pinned-memory limit
        "ggml_cann_host_malloc: failed to allocate 64.00 MiB of pinned memory: no memory",                // non-fatal, falls back to pageable
        "llama_init_from_model: failed to allocate context for kv cache",                                  // ggml_init context object, not a buffer
        "load_tensors: failed to allocate context for control vector",
        "srv  load: model file not found",
        "llama_model_load: error loading model: missing tensor 'blk.0.attn_q.weight'",
    };
    for (const auto & l : not_oom) {
        check(!server_line_reports_oom(l), "not OOM: " + l.substr(0, 70));
    }
    // the cases separate the rules: the old rule calls the pinned-memory and context lines OOM
    check(old_rule(not_oom[0]) && old_rule(not_oom[2]), "scenario discriminates: the old rule counted the pinned-memory and context lines as OOM");
    printf("%s\n", n_fail ? "FAILED" : "ALL PASSED");
    return n_fail ? 1 : 0;
}
