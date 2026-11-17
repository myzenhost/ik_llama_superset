// Draining state for a stopping server: after the first SIGTERM / SIGINT / CTRL_BREAK, /health and every other route
// answer 503 "draining" with a Retry-After, so a load balancer takes the instance out of rotation instead of sending new
// work into a process that is about to exit. Header-only, no llama / http dependencies, unit-tested
// (tools/server/tests_cpp/test_server_drain.cpp).
// Row S69-ENGREV-RTPLLM-SERVER-SIGTERM-DRAIN-2026-10-27-01. SOURCE-READ: tools/server/server.cpp:239-248 (signal_handler
// went straight to shutdown_handler: no state a health probe could see) and server-http.cpp:440-463 (the only 503 was
// "Loading model"). Idea from rtp-llm's drain-on-SIGTERM; description-level, no code copied.
// Limit: this answers new requests only. Requests already routed run to the end of the existing terminate path; no
// in-flight wait is added here (the stream tail is not visible to the HTTP pre/post routing hooks).
#pragma once

#include <atomic>
#include <string>

struct server_drain_state {
    std::atomic<bool> draining{false};

    // true the first time it is called (the caller that wins logs it), false after
    bool begin() {
        bool expected = false;
        return draining.compare_exchange_strong(expected, true);
    }
    bool is_draining() const { return draining.load(); }

    static int retry_after_s() { return 5; } // a drained instance does not come back; the balancer should pick another
    static const char * message() { return "Server is draining: shutting down, retry on another instance"; }
};
