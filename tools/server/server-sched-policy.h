// Which deferred request starts next when a slot frees up (--sched-policy). Pure functions, no llama context, so the ordering is testable
// on its own (tests/test-server-sglang-picks.cpp).
//
//   fcfs  lowest `priority` value first, then arrival order -- exactly what the queue always did.
//   lpm   lowest `priority` value first; within one priority the request with the LONGEST prefix already held by a cache (a slot's own
//         leftover prompt, the host prompt cache, the radix tree) first; a request whose prompt shares >= depri_threshold tokens with an
//         EARLIER waiting request (and has little cached itself) goes last, so the earlier one runs first and the later one finds the
//         shared prefix in the cache instead of recomputing it next to it.
//   spf   shortest-prefill-first: lowest `priority` first; within one priority the request with the fewest UNCACHED prompt tokens
//         (max(1, n_prompt - cached_prefix)) first; the same in-batch deprioritisation as lpm; ties keep arrival order.
//         SOURCE-READ: sglang/python/sglang/srt/managers/schedule_policy.py:177 (CacheAwarePolicy.SHORTEST_PREFILL_FIRST), :395-399
//         (_shortest_prefill_work), :402-412 (_sort_by_shortest_prefill: key = deprioritized, work, queue-entry time). Apache-2.0, no code copied.
//   hrrn  highest-response-ratio-next: lowest `priority` first; within one priority the request with the highest 1 + waited / uncached
//         first, where `waited` = prefill tokens the server processed since the request began waiting (token-based aging, NOT wall clock: at
//         constant throughput it equals 1 + wait_time / prefill_time) and uncached = n_prompt - cached_prefix. A request with 0 uncached
//         tokens has no prefill work and drains first; same in-batch deprioritisation as lpm; ties keep arrival order.
//         SOURCE-READ: sglang/python/sglang/srt/managers/schedule_policy.py:488-528 (_uncached_len, _sort_by_hrrn: key = deprioritized -> +inf,
//         uncached<=0 -> -inf, else -(max(0, processed - arrival_processed) / uncached), then rid), :330-341 (off above 128 queued). Apache-2.0,
//         behaviour only, no code copied. The fork's tie-break is arrival order (SGLang's is rid string order).
//
// SOURCE-READ: sglang/python/sglang/srt/managers/schedule_policy.py:162-168 (policy names), :202-269 (calc_priority), :271-282 (LPM/HRRN fall
// back to FCFS when more than 128 requests wait: the matching is the expensive part), :313-364 (_compute_prefix_matches: in-batch prefix
// check for requests with prefix_len <= 32; one sharing >= 32 tokens with an earlier waiting request is "temporarily deprioritized", else it is
// inserted into the waiting-queue tree), :409-420 (_sort_by_longest_prefix: key = -matched, deprioritized = +inf; Python's stable sort keeps
// arrival order among equals), :105-115 (the two 32-token thresholds, -1 disables). Apache-2.0, behaviour only, no code copied.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class server_sched_policy {
    FCFS,
    LPM,
    SPF,
    HRRN,
};

inline bool server_sched_policy_parse(const std::string & name, server_sched_policy & out) {
    if (name == "fcfs") { out = server_sched_policy::FCFS; return true; }
    if (name == "lpm")  { out = server_sched_policy::LPM;  return true; }
    if (name == "spf" || name == "shortest-prefill-first") { out = server_sched_policy::SPF; return true; }
    if (name == "hrrn") { out = server_sched_policy::HRRN; return true; }
    return false;
}

inline const char * server_sched_policy_name(server_sched_policy p) {
    return p == server_sched_policy::LPM ? "lpm" : p == server_sched_policy::SPF ? "spf" : p == server_sched_policy::HRRN ? "hrrn" : "fcfs";
}

struct server_sched_cand {
    int32_t priority      = 0;        // the request's `priority` field (lower = earlier)
    size_t  n_tokens      = 0;        // prompt length (spf)
    size_t  cached_prefix = 0;        // tokens of this prompt already resident in some cache (0 when unknown)
    uint64_t waited_tokens = 0;       // hrrn: prefill tokens the server processed while this request waited (aging, see above)
    const std::vector<int32_t> * tokens = nullptr; // the prompt, for the in-batch comparison; may be null (then no comparison)
};

struct server_sched_params {
    server_sched_policy policy = server_sched_policy::FCFS;
    // A request is compared against the earlier waiting ones only when its own cached prefix is at most this long (-1 = never compare).
    int64_t check_threshold   = 32;
    // ...and goes last when it shares at least this many tokens with an earlier waiting request (-1 = never).
    int64_t depri_threshold   = 32;
    // More waiting requests than this: plain fcfs (the comparison is quadratic in the queue length).
    size_t  max_queue         = 128;
};

struct server_sched_result {
    size_t pick = 0;               // index into the candidates (arrival order) of the request to start
    size_t fcfs_pick = 0;          // what fcfs would have started
    size_t n_deprioritized = 0;    // candidates pushed to the back by the in-batch rule
    bool   used_lpm = false;       // false when the policy is fcfs or the queue was longer than max_queue
};

inline size_t server_sched_common_prefix(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

// `cands` in arrival order, non-empty.
inline server_sched_result server_sched_pick(const server_sched_params & sp, const std::vector<server_sched_cand> & cands) {
    server_sched_result r;
    // fcfs: the first candidate with the smallest priority value (strict '<' keeps the earliest among equals)
    for (size_t i = 1; i < cands.size(); ++i) {
        if (cands[i].priority < cands[r.fcfs_pick].priority) {
            r.fcfs_pick = i;
        }
    }
    r.pick = r.fcfs_pick;
    if (sp.policy == server_sched_policy::FCFS || cands.size() < 2 || cands.size() > sp.max_queue) {
        return r;
    }
    r.used_lpm = true;

    std::vector<char> depri(cands.size(), 0);
    if (sp.check_threshold >= 0 && sp.depri_threshold >= 0) {
        std::vector<size_t> kept; // earlier requests that stay in line, to compare later ones against
        for (size_t i = 0; i < cands.size(); ++i) {
            if (cands[i].tokens == nullptr || (int64_t) cands[i].cached_prefix > sp.check_threshold) {
                continue;
            }
            size_t best = 0;
            for (size_t k : kept) {
                // only requests that run in the same priority class are each other's "earlier one"
                if (cands[k].priority != cands[i].priority) {
                    continue;
                }
                best = std::max(best, server_sched_common_prefix(*cands[k].tokens, *cands[i].tokens));
            }
            if ((int64_t) best >= sp.depri_threshold) {
                depri[i] = 1;
                r.n_deprioritized++;
            } else {
                kept.push_back(i);
            }
        }
    }

    size_t pick = 0;
    const auto better = [&](size_t a, size_t b) { // is a strictly before b
        if (cands[a].priority != cands[b].priority) {
            return cands[a].priority < cands[b].priority;
        }
        if (depri[a] != depri[b]) {
            return depri[a] < depri[b];
        }
        if (sp.policy == server_sched_policy::SPF) {
            const size_t wa = std::max<size_t>(1, cands[a].n_tokens - std::min(cands[a].n_tokens, cands[a].cached_prefix));
            const size_t wb = std::max<size_t>(1, cands[b].n_tokens - std::min(cands[b].n_tokens, cands[b].cached_prefix));
            return wa < wb;
        }
        if (sp.policy == server_sched_policy::HRRN) {
            const double ua = (double) (cands[a].n_tokens - std::min(cands[a].n_tokens, cands[a].cached_prefix));
            const double ub = (double) (cands[b].n_tokens - std::min(cands[b].n_tokens, cands[b].cached_prefix));
            if (ua == 0 || ub == 0) {
                return ua == 0 && ub != 0; // no prefill work drains first; two of them keep arrival order
            }
            // waited_a / ua > waited_b / ub, cross-multiplied (no division)
            return (double) cands[a].waited_tokens * ub > (double) cands[b].waited_tokens * ua;
        }
        return cands[a].cached_prefix > cands[b].cached_prefix; // equal: the earlier one stays (arrival order)
    };
    for (size_t i = 1; i < cands.size(); ++i) {
        if (better(i, pick)) {
            pick = i;
        }
    }
    r.pick = pick;
    return r;
}
