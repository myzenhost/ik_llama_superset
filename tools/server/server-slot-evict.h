#pragma once

// Queue-aware choice of the slot to evict in get_available_slot's least-recently-used branch
// (row S69-NANOVLLM-PICK-QUEUE-AWARE-SLOT-EVICTION-2026-10-18-01, --slot-evict-lookahead N, default 0 = plain LRU).
//
// Plain LRU takes the idle slot used longest ago, even when the next request in the queue would have picked that slot
// for its cached prefix: the request then lands elsewhere and prefills the whole prefix again (with --cache-ram the
// evicted prompt is saved and reloaded, which softens the loss but still copies the state twice).
// With a lookahead of N, the first N waiting requests are compared against the idle slots: an idle slot is RETAINED when
// it is the slot one of those requests would select by LCP similarity (best similarity among the idle slots, above
// --slot-prompt-similarity). The least recently used NON-retained slot is evicted; when every idle slot is retained the
// choice falls back to plain LRU, so the policy never refuses a slot.
//
// RESEARCH-DONE: nano-vLLM PR #278 (MIT, closed unmerged; github.com/GeeeekExplorer/nano-vllm/pull/278, opt-in
// prefix_cache_lookahead=16, which prefers evicting cached KV blocks no request in the first N waiting requests will
// reuse). Only the PR description was read; it works on KV blocks inside one cache, this works on whole llama-server
// slots. No code copied.

#include <cstddef>
#include <cstdint>
#include <vector>

struct server_slot_evict_result {
    int                pick       = -1;    // index into the candidates, -1 when there are none
    int                lru_pick   = -1;    // what plain LRU would have picked
    std::vector<bool>  retained;           // per candidate: a waiting request would select it by similarity
    size_t             n_retained = 0;
};

// t_last_used[c]   last-use time of idle candidate c (the caller passes idle slots only)
// n_waiting        how many waiting requests were sampled (already capped at the lookahead)
// sim(c, w)        similarity of waiting request w to candidate c, as get_available_slot computes it (LCP / prompt length)
// sim_thold        --slot-prompt-similarity; a request selects a slot by similarity only above it, so with 0 no slot is
//                  retained (similarity selection is off and a waiting request takes an LRU slot anyway)
//
// Plain LRU keeps get_available_slot's tie rule: `t_last_used <= t_last` lets the LATER of equal candidates win.
template <typename Sim>
server_slot_evict_result server_slot_pick_evict(const std::vector<int64_t> & t_last_used, size_t n_waiting, Sim && sim, float sim_thold) {
    server_slot_evict_result res;
    const size_t n = t_last_used.size();
    res.retained.assign(n, false);
    if (n == 0) {
        return res;
    }

    const auto lru = [&](bool skip_retained) {
        int     best   = -1;
        int64_t t_best = -1;
        for (size_t c = 0; c < n; ++c) {
            if (skip_retained && res.retained[c]) {
                continue;
            }
            if (best < 0 || t_last_used[c] <= t_best) {
                t_best = t_last_used[c];
                best   = (int) c;
            }
        }
        return best;
    };

    res.lru_pick = lru(false);

    if (sim_thold > 0.0f && n > 1) {
        for (size_t w = 0; w < n_waiting; ++w) {
            // the slot this waiting request would select: the first candidate with the strictly best similarity above the
            // threshold (get_available_slot's `f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity`)
            int   best   = -1;
            float f_best = 0.0f;
            for (size_t c = 0; c < n; ++c) {
                const float f = sim(c, w);
                if (f > f_best && f > sim_thold) {
                    f_best = f;
                    best   = (int) c;
                }
            }
            if (best >= 0 && !res.retained[best]) {
                res.retained[best] = true;
                res.n_retained++;
            }
        }
    }

    res.pick = res.n_retained > 0 && res.n_retained < n ? lru(true) : res.lru_pick;
    return res;
}
