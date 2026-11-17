#pragma once

// The pos_min threshold for reusing a slot's state (or a context checkpoint) when the prompt is resumed at pos_next.
// Row S69-LLAMACPP-PICK-SWA-REUSE-THRESHOLD-OFF-BY-TWO-2026-10-18-01; behaviour from upstream llama.cpp 61ad80bf (#30580,
// MIT), read whole; no code copied.
//
// The first evaluated token is at pos_eval = pos_next (new tokens follow) or pos_next - 1 (the last prompt token is
// evaluated again for its logits). With a sliding window of n_swa it attends to the positions > pos_eval - n_swa, so a
// SWA state that starts at pos_eval - n_swa + 1 holds the whole window it needs. A state without SWA must hold every
// position before pos_eval. A state is usable when its pos_min is BELOW the returned value.
//
// The old threshold was pos_eval - n_swa. It asked for two more cells than the window, so a state that holds exactly
// its window was refused. That is what the memory-module SWA cache saves: state_write skips SWA-masked cells
// (src/llama-kv-cache.cpp:2134-2139, llama_kv_cache::state_write) and seq_pos_min reads the SWA cache
// (src/llama-kv-cache-iswa.cpp:144-146). On this fork's legacy KV path (src/llama.cpp:11956-11967) the cache is not
// trimmed, pos_min stays 0, and both thresholds accept (MEASURED 2026-10-18 with tinygemma3, window 64).

#include "llama.h"

#include <algorithm>

inline llama_pos server_swa_reuse_pos_min_thold(llama_pos pos_next, bool has_new_tokens, int32_t n_swa) {
    const llama_pos pos_eval = pos_next - (has_new_tokens ? 0 : 1);
    return std::max<llama_pos>(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);
}

// Whether the slot's state can be cut back to the first evaluated position. With --swa-compress the SWA layers hold
// one rolled window of rows, and llama_kv_cache_seq_rm (src/llama.cpp:2809-2832) refuses a p0 inside
// (pos_base_swa, pos_base_swa + window) unless p0 is 0 or past the stored end; the server's seq_rm then aborts
// (common_context_seq_rm, common/common.cpp:2668-2672). rewind_floor is llama_kv_cache_swa_rewind_floor (0 when nothing
// is compacted or the window never rolled), pos_end is the slot's pos_max + 1.
// The threshold above accepts pos_min = pos_eval - n_swa + 1, while the compacted floor is pos_base_swa + n_swa: a
// state whose window rolled to pos_eval - n_swa + 1 passes the threshold and then cannot be rewound (MEASURED
// 2026-10-19: tinygemma3 --swa-compress, window 64, sharing 255 tokens killed llama-server). Such a state is
// handled as one that fails the threshold: a checkpoint, or a full reprocess. Row
// S69-FORK-SWA-REUSE-E2E-REACH-VIA-SWA-COMPRESS-2026-10-19-01.
inline bool server_swa_reuse_rewind_ok(llama_pos pos_next, bool has_new_tokens, llama_pos rewind_floor, llama_pos pos_end) {
    const llama_pos p0 = pos_next - (has_new_tokens ? 0 : 1);
    return rewind_floor <= 0 || p0 <= 0 || p0 >= pos_end || p0 >= rewind_floor;
}
