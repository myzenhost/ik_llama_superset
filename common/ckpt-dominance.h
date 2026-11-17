#pragma once
// Dominated-checkpoint freeing for the per-slot context checkpoint list (--ctx-ckpt-dominance).
//
// Rule, reimplemented in C++ from trnrt radix_cache.py:363-399 (RadixCache.free_dominated; Apache-2.0, clone
// upstream/trnrt, SOURCE-READ: upstream/trnrt/trnrt/engine/radix_cache.py). No code
// copied: their data structure is a radix tree walked through parents, ours is one slot's list ordered by n_tokens.
// SOURCE-NUANCE: the walk stops at the first ancestor that has another child and keeps that ancestor; the nearest
// prompt-end anchor is kept only once per path (an older anchor is freed like any other); a locked node is skipped.
//   * all checkpoints of a slot lie on ONE token path; the deepest (largest n_tokens) one serves every request that
//     continues past an older one, so an older checkpoint serves only a request that BRANCHES in between;
//   * walking from the checkpoint below the deepest toward the start, the walk STOPS at the first entry flagged
//     `branch` (another continuation is known from there: the LCP / branch-point checkpoint); that entry and
//     everything below it are kept;
//   * the nearest prompt-end checkpoint (`anchor`) at or below the deepest one is kept: the next turn replays this
//     path's reply and, where a re-rendered tool call differs from the cached tokens, resumes from it.
// Online prior art (ledger): Marconi, arXiv 2411.19379 (admission and eviction policy for hybrid prefix caches).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

struct common_ckpt_dom_entry {
    int64_t n_tokens = 0;
    bool    anchor   = false; // created at the prompt end (the checkpoint the next turn replays from)
    bool    branch   = false; // created at a branch point (LCP snapshot): another continuation exists from here
};

// Returns the positions (indices into `e`, ascending) of the entries to free.
inline std::vector<size_t> common_ckpt_dominated(const std::vector<common_ckpt_dom_entry> & e) {
    std::vector<size_t> out;
    if (e.size() < 2) {
        return out;
    }
    std::vector<size_t> ord(e.size());
    std::iota(ord.begin(), ord.end(), (size_t) 0);
    std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return e[a].n_tokens < e[b].n_tokens; });

    bool kept = e[ord.back()].anchor;
    for (size_t k = ord.size() - 1; k-- > 0; ) {
        const auto & c = e[ord[k]];
        if (c.branch) {
            break;
        }
        if (c.anchor && !kept) {
            kept = true;
            continue;
        }
        out.push_back(ord[k]);
    }
    std::sort(out.begin(), out.end());
    return out;
}
