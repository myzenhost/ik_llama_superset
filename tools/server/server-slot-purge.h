#pragma once

// Which idle slot try_clear_idle_slots() purges when a --kv-unified decode runs out of cells
// (row S69-VLLM-W2-IDLE-SLOT-LRU-PURGE-2026-10-27-01).
//
// The old rule took the first idle slot, by slot id, that still held tokens. Slot 0 was therefore purged over and over while a
// slot that had sat unused for minutes kept its cache. vLLM reuses cached blocks in least-recently-used order: its free-block queue
// "the least recent used block is at the front (LRU)" (vllm 924707f1bf94, vllm/v1/core/kv_cache_utils.py:247-263, class
// FreeKVCacheBlockQueue) and cached blocks are appended at the end on free so they are reused last (vllm/v1/core/block_pool.py:798-805;
// Apache-2.0, read, no code copied). Here the unit is a whole llama-server slot, so the rule is: among idle slots that hold tokens,
// purge the one used longest ago.
//
// t_last_used[c]  last-use time of candidate c (ggml_time_us(); -1 = never used)
// n_tokens[c]     tokens cached in candidate c (idle candidates only; a slot with no tokens frees nothing and is never picked)
// Returns the index of the slot to purge, -1 when no candidate holds tokens. A tie goes to the lower index (deterministic).

#include <cstddef>
#include <cstdint>
#include <vector>

inline int server_slot_pick_purge(const std::vector<int64_t> & t_last_used, const std::vector<size_t> & n_tokens) {
    int     best   = -1;
    int64_t t_best = 0;
    const size_t n = t_last_used.size() < n_tokens.size() ? t_last_used.size() : n_tokens.size();
    for (size_t c = 0; c < n; ++c) {
        if (n_tokens[c] == 0) {
            continue;
        }
        if (best < 0 || t_last_used[c] < t_best) {
            best   = (int) c;
            t_best = t_last_used[c];
        }
    }
    return best;
}
