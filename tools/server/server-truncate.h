// --auto-truncate: a text prompt longer than the slot context is cut to fit instead of failing with HTTP 400
// "exceeds the available context size". Header-only, no llama / http dependencies, unit-tested
// (tools/server/tests_cpp/test_server_truncate.cpp).
// Row S69-ENGREV-SGLANG-AUTO-TRUNCATE-2026-10-27-01. SOURCE-READ: tools/server/server-context.cpp:7648-7667 (the two
// over-context refusals) and :10299-10307 (where a text prompt is tokenised, before any slot is chosen). sglang's
// auto_truncate idea (an over-long prompt is shortened rather than refused); description-level, no code copied.
// Policy: keep the first n_keep tokens (a system prompt) and the LAST tokens, drop the middle; leave room for the
// generation (min(n_predict, n_ctx / 2)) so the reply is not cut at once. The response says so (prompt_truncated).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

struct server_truncate_plan {
    bool   truncate  = false;
    size_t keep_head = 0;
    size_t keep_tail = 0;
    size_t n_dropped = 0;
};

// n_ctx: tokens a slot can hold. n_predict <= 0 means "unbounded": reserve nothing beyond the one token the slot needs.
static inline server_truncate_plan server_truncate_plan_for(size_t n_tokens, size_t n_ctx, int64_t n_predict, int64_t n_keep) {
    server_truncate_plan p;
    if (n_ctx < 2) {
        return p; // nothing sensible to keep
    }
    const size_t reserve = n_predict > 0 ? std::min<size_t>((size_t) n_predict, n_ctx / 2) : 0;
    const size_t budget  = n_ctx - 1 - reserve; // the slot needs the prompt strictly below n_ctx
    if (n_tokens <= budget) {
        return p;
    }
    p.truncate  = true;
    p.keep_head = n_keep > 0 ? std::min<size_t>((size_t) n_keep, budget / 2) : 0; // a huge n_keep must not eat the tail
    p.keep_tail = budget - p.keep_head;
    p.n_dropped = n_tokens - p.keep_head - p.keep_tail;
    return p;
}
