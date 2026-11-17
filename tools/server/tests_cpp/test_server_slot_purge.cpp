// Idle-slot purge order for --kv-unified (tools/server/server-slot-purge.h), model-free.
// Row S69-VLLM-W2-IDLE-SLOT-LRU-PURGE-2026-10-27-01; vLLM least-recently-used block freeing, read, no code copied.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "server-slot-purge.h"

#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    n_fail += ok ? 0 : 1;
}

// The rule try_clear_idle_slots() had before: the first idle slot by id that holds tokens. Kept here only to show that the scenarios
// below tell the two rules apart (a test both rules pass proves nothing).
static int old_first_fit(const std::vector<size_t> & n_tokens) {
    for (size_t c = 0; c < n_tokens.size(); ++c) {
        if (n_tokens[c] > 0) {
            return (int) c;
        }
    }
    return -1;
}

int main() {
    // three idle slots, all cached; slot 2 was used longest ago, slot 0 most recently
    {
        const std::vector<int64_t> t = { 300, 200, 100 };
        const std::vector<size_t>  n = { 50, 60, 70 };
        check(old_first_fit(n) == 0, "scenario discriminates: the old rule would purge slot 0 (most recently used)");
        check(server_slot_pick_purge(t, n) == 2, "the oldest-used slot (2) is purged");
    }
    // the oldest slot holds nothing: it frees nothing, so the oldest slot that DOES hold tokens goes
    {
        const std::vector<int64_t> t = { 300, 200, 100 };
        const std::vector<size_t>  n = { 50, 60, 0 };
        check(server_slot_pick_purge(t, n) == 1, "an empty slot is skipped, the next-oldest cached slot (1) is purged");
    }
    // nothing cached -> nothing to purge
    {
        check(server_slot_pick_purge({ 1, 2 }, { 0, 0 }) == -1, "no cached idle slot: -1");
        check(server_slot_pick_purge({}, {}) == -1, "no idle slot: -1");
    }
    // never-used marker (-1) is the oldest possible
    {
        check(server_slot_pick_purge({ 500, -1, 400 }, { 5, 5, 5 }) == 1, "t_last_used -1 (never used) sorts oldest");
    }
    // tie: the lower index, deterministic
    {
        check(server_slot_pick_purge({ 100, 100, 200 }, { 5, 5, 5 }) == 0, "equal last-use time: the lower index");
    }
    // single candidate
    {
        check(server_slot_pick_purge({ 100 }, { 9 }) == 0, "a single cached idle slot is purged");
    }
    printf("%s (%d failing)\n", n_fail == 0 ? "OK" : "FAILED", n_fail);
    return n_fail == 0 ? 0 : 1;
}
