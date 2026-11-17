#pragma once

// Does a child's log line report that an allocation failed because memory ran out? Header-only so a model-free test can reach it
// (tests_cpp/test_server_oom_line.cpp). Used by the router to arm the single load-OOM retry (evict idle models, step the context
// down), which is the wrong remedy for a load that failed for another reason.
//
// Row S69-ENGREV-JAN-OOM-LINE-BARE-FAILED-TO-ALLOCATE-2026-10-27-01. Before: a bare "failed to allocate" matched, including the
// non-fatal ggml-cuda warning "failed to allocate N MiB of pinned memory" (WDDM pinned-memory limits, the load goes on without it),
// so a child that later died of something else (bad tensor, bad mmproj) was retried as an OOM. Jan (Apache-2.0) requires the
// "buffer of size" context for this phrase (worker.rs is_oom_line); ollama's llm_status.go lists the bare phrase. Both read, no code
// copied; this is the intersection kept: every allocation-failure spelling of this tree stays, the non-fatal ones do not.

#include <algorithm>
#include <cctype>
#include <string>

static inline bool server_line_reports_oom(const std::string & line) {
    std::string s = line;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    auto has = [&s](const char * n) { return s.find(n) != std::string::npos; };

    // spellings that name memory exhaustion by themselves (CUDA/HIP/Vulkan/Metal/ggml wording)
    static const char * needles[] = {
        "out of memory", "cudamalloc failed", "unable to allocate", "outofdevicememory", "out of device memory",
    };
    for (const char * n : needles) {
        if (has(n)) {
            return true;
        }
    }
    if (has("failed to allocate")) {
        // a pinned (page-locked) host staging buffer that could not be registered is a warning, the backend falls back to pageable
        // memory; and a ggml/llama *context* that could not be created ("failed to allocate context for ...") is not a buffer
        if (has("pinned memory")) {
            return false;
        }
        // the real ones: "failed to allocate CUDA0 buffer of size N", "failed to allocate compute buffers", "failed to allocate buffer
        // for kv cache", "failed to allocate graph", "failed to allocate N bytes"
        return has("buffer") || has("graph") || has(" bytes");
    }
    return false;
}
