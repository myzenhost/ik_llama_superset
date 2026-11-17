// Queue-aware slot eviction (tools/server/server-slot-evict.h, --slot-evict-lookahead), model-free.
// Row S69-NANOVLLM-PICK-QUEUE-AWARE-SLOT-EVICTION-2026-10-18-01; idea from nano-vLLM PR #278 (MIT), description read, no code copied.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-slot-evict.h"

#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    n_fail += ok ? 0 : 1;
}

// sims[w][c]
static server_slot_evict_result pick(const std::vector<int64_t> & t, const std::vector<std::vector<float>> & sims, float thold) {
    return server_slot_pick_evict(t, sims.size(), [&](size_t c, size_t w) { return sims[w][c]; }, thold);
}

int main() {
    // slot 0 is the least recently used and holds the prefix the next waiting request continues
    {
        const auto r = pick({ 10, 20, 30 }, { { 0.9f, 0.0f, 0.0f } }, 0.1f);
        check(r.lru_pick == 0, "plain LRU would evict slot 0");
        check(r.pick == 1 && r.retained[0] && r.n_retained == 1, "slot 0 is kept for the waiting request, the next LRU slot (1) is evicted");
    }
    // control: no waiting request -> plain LRU
    {
        const auto r = pick({ 10, 20, 30 }, {}, 0.1f);
        check(r.pick == 0 && r.n_retained == 0, "no waiting request: plain LRU");
    }
    // control: the waiting request matches nothing above the threshold -> plain LRU
    {
        const auto r = pick({ 10, 20, 30 }, { { 0.05f, 0.1f, 0.0f } }, 0.1f);
        check(r.pick == 0 && r.n_retained == 0, "similarity at or below the threshold retains nothing");
    }
    // control: similarity selection off (threshold 0) -> plain LRU even with a perfect match
    {
        const auto r = pick({ 10, 20, 30 }, { { 1.0f, 0.0f, 0.0f } }, 0.0f);
        check(r.pick == 0 && r.n_retained == 0, "--slot-prompt-similarity 0: plain LRU");
    }
    // every idle slot is wanted -> fall back to plain LRU (never refuse a slot)
    {
        const auto r = pick({ 10, 20 }, { { 0.9f, 0.0f }, { 0.0f, 0.8f } }, 0.1f);
        check(r.n_retained == 2 && r.pick == 0, "all idle slots retained: plain LRU fallback");
    }
    // a waiting request retains only its BEST slot: slot 0 (0.3) loses to slot 2 (0.7), so slot 0 stays evictable
    {
        const auto r = pick({ 10, 20, 30 }, { { 0.3f, 0.0f, 0.7f } }, 0.1f);
        check(r.retained[2] && !r.retained[0] && r.pick == 0, "only the best-matching slot is retained");
    }
    // ties among equal similarity: the first candidate wins (get_available_slot's strict '>')
    {
        const auto r = pick({ 10, 20, 30 }, { { 0.5f, 0.5f, 0.0f } }, 0.1f);
        check(r.retained[0] && !r.retained[1] && r.pick == 1, "equal similarity retains the first slot");
    }
    // plain LRU tie rule kept: equal t_last_used -> the LATER candidate (`<=`)
    {
        const auto r = pick({ 10, 10, 30 }, {}, 0.1f);
        check(r.pick == 1, "equal last-use time: the later slot, as get_available_slot's '<='");
    }
    // one idle slot: nothing to choose
    {
        const auto r = pick({ 10 }, { { 1.0f } }, 0.1f);
        check(r.pick == 0 && r.n_retained == 0, "a single idle slot is taken");
    }
    // no candidates
    {
        const auto r = pick({}, {}, 0.1f);
        check(r.pick == -1, "no idle slot: -1");
    }
    printf("%s (%d failing)\n", n_fail == 0 ? "OK" : "FAILED", n_fail);
    return n_fail == 0 ? 0 : 1;
}
