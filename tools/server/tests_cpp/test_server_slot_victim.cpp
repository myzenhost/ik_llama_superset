// KV-exhaustion preemption victim policies for --kv-unified (tools/server/server-slot-victim.h), model-free.
// Row S69-VLLM-W2-PREEMPT-VICTIM-LEAST-PROGRESS-AND-E2E-2026-10-27-01; vLLM scheduler.py:759-767 victim choice, read, no code copied.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-slot-victim.h"

#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    n_fail += ok ? 0 : 1;
}

static server_victim_cand mk(int32_t prio, int64_t cells, int64_t t_arr, int64_t id, bool rq) {
    server_victim_cand c;
    c.priority = prio; c.cells = cells; c.t_arrival = t_arr; c.id = id; c.requeueable = rq;
    return c;
}

// The rule preempt_largest_slot() had before the policy flag (highest priority value, then most cells, first wins a full tie). Kept
// here only to show that the default policy is the same function, and that the scenarios below tell the policies apart.
static int old_rule(const std::vector<server_victim_cand> & c) {
    int best = -1;
    for (size_t i = 0; i < c.size(); ++i) {
        if (best < 0 || c[i].priority > c[(size_t) best].priority ||
            (c[i].priority == c[(size_t) best].priority && c[i].cells > c[(size_t) best].cells)) {
            best = (int) i;
        }
    }
    return best;
}

int main() {
    // four requests: 0 is the big old one, 3 the small newest one
    const std::vector<server_victim_cand> v = {
        mk(0, 180, 100, 1, true), mk(0, 120, 200, 2, true), mk(0, 90, 300, 3, true), mk(0, 40, 400, 4, true),
    };
    check(old_rule(v) == 0, "scenario: the old rule evicts the largest (0)");
    check(server_pick_victim(v, 0) == 0, "largest (default) == old rule: evicts the largest (0)");
    check(server_pick_victim(v, 1) == 3, "least-progress evicts the fewest cells (3)");
    check(server_pick_victim(v, 2) == 3, "newest evicts the latest arrival (3)");

    // newest and least-progress differ when the newest is not the smallest
    const std::vector<server_victim_cand> w = {
        mk(0, 60, 100, 1, true), mk(0, 150, 300, 2, true), mk(0, 90, 200, 3, true),
    };
    check(server_pick_victim(w, 1) == 0, "least-progress ignores arrival order: smallest (0)");
    check(server_pick_victim(w, 2) == 1, "newest ignores size: latest arrival (1)");
    check(server_pick_victim(w, 0) == 1, "largest: 150 cells (1)");

    // priority beats every policy (a higher value = less important is evicted first), as vLLM compares priority before arrival
    const std::vector<server_victim_cand> p = {
        mk(0, 10, 100, 1, true), mk(5, 200, 50, 2, true), mk(5, 300, 60, 3, true),
    };
    check(server_pick_victim(p, 0) == 2, "priority 5 first, then the largest of them (2)");
    check(server_pick_victim(p, 1) == 1, "priority 5 first, then the fewest cells of them (1)");
    check(server_pick_victim(p, 2) == 2, "priority 5 first, then the latest arrival of them (2)");

    // least-progress prefers a victim that can be requeued over a streaming one, even if the streaming one is smaller
    const std::vector<server_victim_cand> s = {
        mk(0, 30, 100, 1, false), mk(0, 100, 200, 2, true), mk(0, 70, 300, 3, true),
    };
    check(server_pick_victim(s, 1) == 2, "least-progress: a requeueable victim (70 cells) before a smaller streaming one");
    check(server_pick_victim(s, 2) == 2, "newest: latest arrival (2)");
    check(server_pick_victim(s, 0) == 1, "largest: unchanged, ignores requeueable (1)");
    // none requeueable: the flag is off or every request streams -> plain fewest cells
    const std::vector<server_victim_cand> n = {
        mk(0, 80, 100, 1, false), mk(0, 30, 200, 2, false),
    };
    check(server_pick_victim(n, 1) == 1, "least-progress with nobody requeueable: fewest cells (1)");

    // full ties: the arrival decides, then the id, then the lower index
    const std::vector<server_victim_cand> t1 = { mk(0, 50, 100, 1, true), mk(0, 50, 200, 2, true) };
    check(server_pick_victim(t1, 1) == 1, "equal cells: the newer arrival (1)");
    const std::vector<server_victim_cand> t2 = { mk(0, 50, 100, 7, true), mk(0, 50, 100, 9, true) };
    check(server_pick_victim(t2, 1) == 1, "equal cells and arrival: the higher id (1)");
    check(server_pick_victim(t2, 2) == 1, "newest: equal arrival, the higher id (1)");
    const std::vector<server_victim_cand> t3 = { mk(0, 50, 100, 7, true), mk(0, 50, 100, 7, true) };
    check(server_pick_victim(t3, 1) == 0 && server_pick_victim(t3, 2) == 0 && server_pick_victim(t3, 0) == 0, "total tie: the lower index, every policy");

    // a requeued request keeps its arrival time, so newest does not re-select it merely for having restarted: modelled by equal arrival
    // times here (the recompute is not newer than the request that was running beside it)
    const std::vector<server_victim_cand> r = { mk(0, 20, 100, 1, true), mk(0, 90, 200, 2, true) };
    check(server_pick_victim(r, 2) == 1, "newest uses the ORIGINAL arrival of a requeued request");

    // empty / single
    check(server_pick_victim({}, 1) == -1, "no candidate: -1");
    check(server_pick_victim({ mk(0, 5, 1, 1, false) }, 1) == 0, "one candidate: 0");
    // unknown policy value falls back to the default rule
    check(server_pick_victim(v, 99) == 0, "unknown policy: largest");

    // near-finished exemption (row S69-ENGREV-PADDLENLP-PREEMPT-NEAR-FINISHED-EXEMPT-2026-10-27-01): remaining <= N loses to any candidate above N
    auto rem = [](server_victim_cand c, int64_t r) { c.remaining = r; return c; };
    const std::vector<server_victim_cand> nf = { rem(mk(0, 200, 100, 1, true), 3), rem(mk(0, 50, 200, 2, true), 500) };
    check(server_pick_victim(nf, 0) == 0, "near-finished off (N=0): old order, the largest (0) even though it has 3 tokens left");
    check(server_pick_victim(nf, 0, 8) == 1, "near-finished N=8: the request with 3 tokens left is never picked while another exists (1)");
    check(server_pick_victim(nf, 1, 8) == 1 && server_pick_victim(nf, 2, 8) == 1, "near-finished exemption holds under least-progress and newest");
    const std::vector<server_victim_cand> nf_all = { rem(mk(0, 200, 100, 1, true), 3), rem(mk(0, 50, 200, 2, true), 5) };
    check(server_pick_victim(nf_all, 0, 8) == 0, "all near-finished: falls back to the old order (largest, 0)");
    check(server_pick_victim(nf_all, 1, 8) == 1, "all near-finished: falls back to the policy (least-progress, 1)");
    const std::vector<server_victim_cand> nf_edge = { rem(mk(0, 200, 100, 1, true), 8), rem(mk(0, 50, 200, 2, true), 9) };
    check(server_pick_victim(nf_edge, 0, 8) == 1, "boundary: remaining == N is near-finished, N+1 is not (1)");
    // the exemption is applied before priority: a near-finished high-value (unimportant) request is spared
    const std::vector<server_victim_cand> nf_p = { rem(mk(5, 10, 100, 1, true), 2), rem(mk(0, 10, 200, 2, true), 900) };
    check(server_pick_victim(nf_p, 0, 8) == 1, "exemption outranks priority: the near-finished priority-5 request is spared (1)");
    check(server_pick_victim(nf_p, 0, 0) == 0, "exemption off: priority 5 evicted (0)");
    // limitless generation (INT64_MAX default) is never near-finished
    check(server_pick_victim({ mk(0, 200, 100, 1, true), rem(mk(0, 50, 200, 2, true), 1) }, 0, 8) == 0, "default remaining = limitless: not exempt (0)");

    printf("%s (%d failing)\n", n_fail == 0 ? "OK" : "FAILED", n_fail);
    return n_fail == 0 ? 0 : 1;
}
