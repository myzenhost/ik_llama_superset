#pragma once

// Overlap of the grammar mask computation with the next decode (row S69-EXLLAMAV2-PICK-GRAMMAR-MASK-OVERLAP-MEASURE-2026-10-20-01).
// Spec only, no code copied: ExLlamaV2 generator/filters/base.py:53-119 and dynamic.py:1213-1220 run the filter's next() on a worker
// thread while the forward pass of the following step is queued, and join it when the mask is needed. Before this header the llguidance
// sampler computed the mask lazily inside apply() (common/llguidance.cpp), so the whole matcher step sat between the decode and the
// sample on the decode thread. Now accept() starts the job for the NEXT state right after it consumes the token, apply() joins it.
//
// Rules the owner must keep (they are what tests/test-grammar-mask-overlap.cpp checks):
//   - one job at a time; start() joins a pending job first (its result belongs to a state that is gone);
//   - anything that touches the matcher (consume, reset, clone, free) calls drop() first, so the worker never races the caller;
//   - an exception inside the job comes back as a non-zero result, never as a terminate().
// LLAMA_GRAMMAR_MASK_SYNC=1 switches the overlap off (the old synchronous path), so a suspected race can be ruled in or out at once.
// Header-only and free of llguidance so a CPU unit test covers it.

#include <cstdlib>
#include <functional>
#include <future>
#include <utility>

class common_grammar_mask_overlap {
public:
    common_grammar_mask_overlap() = default;
    common_grammar_mask_overlap(common_grammar_mask_overlap && o) noexcept : fut_(std::move(o.fut_)) {}
    common_grammar_mask_overlap & operator=(common_grammar_mask_overlap && o) noexcept {
        if (this != &o) {
            drop();
            fut_ = std::move(o.fut_);
        }
        return *this;
    }
    common_grammar_mask_overlap(const common_grammar_mask_overlap &)             = delete;
    common_grammar_mask_overlap & operator=(const common_grammar_mask_overlap &) = delete;
    ~common_grammar_mask_overlap() { drop(); }

    // Run job on a worker thread. A pending job is joined and its result dropped first.
    void start(std::function<int()> job) {
        drop();
        fut_ = std::async(std::launch::async, std::move(job));
    }

    bool pending() const { return fut_.valid(); }

    // Join the pending job and return its result (non-zero when the job threw). Nothing pending: returns idle_rc without waiting.
    int finish(int idle_rc = 0) {
        if (!fut_.valid()) {
            return idle_rc;
        }
        int rc = -1;
        try {
            rc = fut_.get();
        } catch (...) {  // SWALLOW-OK: the job's failure is reported through the return value, the caller fails closed on it
            rc = -1;
        }
        return rc;
    }

    // Join and discard. Call before anything else touches the object the job works on.
    void drop() { (void) finish(0); }

    // On unless LLAMA_GRAMMAR_MASK_SYNC is set to something other than "" or "0".
    static bool enabled() {
        const char * v = std::getenv("LLAMA_GRAMMAR_MASK_SYNC");
        return v == nullptr || v[0] == '\0' || (v[0] == '0' && v[1] == '\0');
    }

private:
    std::future<int> fut_;
};
