#pragma once

// Which in-flight request a --kv-unified KV-exhaustion preemption evicts (preempt_largest_slot() in server-context.cpp)
// (row S69-VLLM-W2-PREEMPT-VICTIM-LEAST-PROGRESS-AND-E2E-2026-10-27-01, --kv-preempt-victim).
//
// What vLLM does (vllm 924707f1bf94, vllm/v1/core/sched/scheduler.py:759-767; Apache-2.0, read, no code copied): when a running
// request cannot get a new KV block, the victim is the LAST request in the running list under first-come-first-served scheduling
// (the newest admission) and the request with the greatest (priority value, arrival time) under the priority policy, i.e. the least
// important and, among equals, the newest. The loop evicts one victim per failed allocation and retries the same request, so the
// number of evictions adapts to how much room each one frees; a request evicted for itself ends the loop (:802-809).
// NUANCES THE DESCRIPTION DID NOT SAY: (1) vLLM compares priority FIRST and arrival only as the tie-break, so an important newcomer
// never evicts an unimportant old request; (2) the victim choice does not look at how big the victim is or how much it already
// computed, so the "least work lost" outcome is a side effect of newest-first, not a rule; (3) a request that was evicted keeps its
// arrival time, so a recomputed request is NOT the newest again when it re-runs (here: t_submit_us survives requeue_copy()).
//
// Policies (common_params::kv_preempt_victim):
//   0 largest        highest priority value, then most cells held. The behaviour before this row; one eviction frees the most room.
//   1 least-progress highest priority value, then a request that can be requeued (a streaming one, or one at the requeue limit, can
//                    only be answered 503), then FEWEST computed cells (least recompute thrown away), then the newest arrival.
//   2 newest         highest priority value, then the newest arrival (vLLM's FCFS rule).
// Every policy ends on the lower candidate index for a full tie, so the choice is deterministic.
//
// Near-finished exemption (row S69-ENGREV-PADDLENLP-PREEMPT-NEAR-FINISHED-EXEMPT-2026-10-27-01, --kv-preempt-near-finished N): PaddleNLP
// speculate_step.cu:114-120 (3f87dac9b719, Apache-2.0, read, no code copied) excludes a request on its last possible decode block from the
// victim scan because it is about to finish and free its blocks, and falls back to the full scan when nobody else qualifies (:125-126).
// Here the trigger is tokens remaining (exact) instead of blocks: a candidate with remaining <= N loses to every candidate above N;
// when ALL are near-finished the normal order picks among them. N = 0 (default) disables the exemption, so today's order is unchanged.

#include <climits>
#include <cstdint>
#include <vector>

struct server_victim_cand {
    int32_t priority    = 0;     // request priority value (higher value = less important)
    int64_t cells       = 0;     // KV cells the request holds (main sequence + CFG guidance sequence)
    int64_t t_arrival   = 0;     // when the request arrived (microseconds); a requeued request keeps its original value
    int64_t id          = 0;     // task id (monotonic with arrival) for a tie on t_arrival
    bool    requeueable = false; // preempting it can end in a recompute instead of a 503
    int64_t remaining   = INT64_MAX; // tokens it may still generate (n_predict - generated); INT64_MAX = limitless / unknown
};

// true when `a` is a better victim than `b` under `policy`
inline bool server_victim_better(const server_victim_cand & a, const server_victim_cand & b, int policy) {
    if (a.priority != b.priority) {
        return a.priority > b.priority;
    }
    switch (policy) {
        case 1:
            if (a.requeueable != b.requeueable) {
                return a.requeueable;
            }
            if (a.cells != b.cells) {
                return a.cells < b.cells;
            }
            if (a.t_arrival != b.t_arrival) {
                return a.t_arrival > b.t_arrival;
            }
            return a.id > b.id;
        case 2:
            if (a.t_arrival != b.t_arrival) {
                return a.t_arrival > b.t_arrival;
            }
            return a.id > b.id;
        default:
            return a.cells > b.cells;
    }
}

// index of the victim among `c`, -1 when `c` is empty; near_finished_n > 0 exempts candidates with remaining <= near_finished_n
// (unless every candidate is near-finished)
inline int server_pick_victim(const std::vector<server_victim_cand> & c, int policy, int64_t near_finished_n = 0) {
    auto near = [near_finished_n](const server_victim_cand & x) { return near_finished_n > 0 && x.remaining <= near_finished_n; };
    bool any_far = false;
    for (const auto & x : c) {
        any_far = any_far || !near(x);
    }
    int best = -1;
    for (size_t i = 0; i < c.size(); ++i) {
        if (any_far && near(c[i])) {
            continue;
        }
        if (best < 0 || server_victim_better(c[i], c[(size_t) best], policy)) {
            best = (int) i;
        }
    }
    return best;
}
