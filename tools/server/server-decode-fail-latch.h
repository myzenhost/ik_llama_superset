#pragma once

// /health decode-failure latch (S69-ENGREV-TGI-HEALTH-DECODE-FAIL-LATCH-2026-10-27-01).
//
// A "Compute error." from the decode loop (llama_decode ret < -1) means the backend
// context may be wedged (e.g. a CUDA fault). /health used to keep answering 200, so a
// watchdog never saw it. The latch is set on that failure and cleared by the next
// successful decode; while set, /health answers 503. Context-exceeded and invalid
// batch are request-caused and never latch.
//
// Design only from the public behaviour of a health check that fails after a failed
// forward pass (Apache-2.0 router); no code copied.

#include <atomic>

struct server_decode_fail_latch {
    std::atomic<bool> failed{false};

    // ret is the llama_decode / llama_process return code of one decode attempt
    void note_decode(int ret) {
        if (ret < -1) {
            failed.store(true);
        } else if (ret == 0) {
            failed.store(false);
        }
        // ret == 1 (no KV slot) and ret == -1 (invalid batch) are caused by the request
    }

    bool is_failed() const { return failed.load(); }
};

// one server context per process
static inline server_decode_fail_latch & server_decode_fail_latch_global() {
    static server_decode_fail_latch l;
    return l;
}
