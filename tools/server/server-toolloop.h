#pragma once

// Tool-call loop guard for repetition_detection (tool_call_markers {begin, end, min_count}).
//
// An agent loop that keeps issuing the SAME tool call produces the same marker-delimited span again and again across
// [history, output]. This finds completed spans (begin marker .. end marker, inclusive) left to right and reports
// true when the longest run of byte-identical CONSECUTIVE spans is at least min_count and that run ends on a span that
// is completed inside the current output (so a loop already in the history alone does not stop a fresh answer).
//
// Same contract as rtp-llm TokenToolCallLoopGuard (completed spans found left to right, longest run of identical
// consecutive spans ending on a current output span); SOURCE-READ rtp-llm TokenToolCallLoopGuard.cc:96-141,:167
// (Apache-2.0), no code copied. Header-only and free of llama.h so it can be unit-tested with a plain compiler.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace server_toolloop {

// `all` is history followed by output; `n_history` is where the output starts.
template <typename Tok>
inline bool toolcall_loop_detected(const std::vector<Tok> & all, size_t n_history, Tok begin_tok, Tok end_tok, int32_t min_count) {
    if (min_count < 2 || begin_tok == end_tok) {
        return false;
    }
    struct span { size_t s, e; }; // [s, e) including both markers
    std::vector<span> spans;
    size_t open = (size_t) -1;
    for (size_t i = 0; i < all.size(); ++i) {
        if (all[i] == begin_tok) {
            open = i;                     // a new begin restarts an unterminated span
        } else if (all[i] == end_tok && open != (size_t) -1) {
            spans.push_back({open, i + 1});
            open = (size_t) -1;
        }
    }
    if (spans.size() < (size_t) min_count) {
        return false;
    }
    auto same = [&](const span & a, const span & b) {
        if (a.e - a.s != b.e - b.s) {
            return false;
        }
        for (size_t i = 0; i < a.e - a.s; ++i) {
            if (all[a.s + i] != all[b.s + i]) {
                return false;
            }
        }
        return true;
    };
    // the run must END on the last completed span, and that span must have completed in the output
    const size_t last = spans.size() - 1;
    if (spans[last].e <= n_history) {
        return false;
    }
    size_t run = 1;
    for (size_t k = last; k > 0 && same(spans[k - 1], spans[last]); --k) {
        ++run;
    }
    return run >= (size_t) min_count;
}

} // namespace server_toolloop
