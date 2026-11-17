// Router model lifecycle events (tools/server/server-model-events.h), model-free, header-only.
// Row S69-ENGREV-TFSERVING-LIFECYCLE-EVENT-RING-2026-10-27-01. Build by hand: g++ -std=c++17 -I.. test_server_model_events.cpp
// GATE-CLAIM-OK: unit test, no resource claims.
// PRIOR-ART-NONE: grep of tools/server and tests for stop_reason|event_ring|models/events found only the Anthropic API stop_reason field, no router lifecycle ring.
#include "server-model-events.h"

#include <cstdio>
#include <string>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    n_fail += ok ? 0 : 1;
}

int main() {
    // the four stop paths of the row's test plan give four distinct reasons, and a crash is none of them
    check(server_exit_reason(SERVER_STOP_IDLE_TTL,  false, 0)   == "idle-ttl",   "keep-alive expiry -> idle-ttl");
    check(server_exit_reason(SERVER_STOP_LRU_EVICT, false, 0)   == "lru-evict",  "eviction -> lru-evict");
    check(server_exit_reason(SERVER_STOP_MANUAL,    false, 0)   == "manual",     "POST unload -> manual");
    check(server_exit_reason(SERVER_STOP_RELOAD,    false, 0)   == "reload",     "reload -> reload");
    check(server_exit_reason(SERVER_STOP_SHUTDOWN,  false, 0)   == "shutdown",   "router exit -> shutdown");
    check(server_exit_reason(SERVER_STOP_NONE,      false, 139) == "crash",      "killed child (non-zero, nobody asked) -> crash");
    check(server_exit_reason(SERVER_STOP_NONE,      true,  1)   == "load-oom",   "OOM while loading -> load-oom");
    check(server_exit_reason(SERVER_STOP_NONE,      false, 0)   == "exited",     "clean exit nobody asked for -> exited");
    // a stop we asked for keeps its reason even when the child then exits non-zero (a force-killed loader)
    check(server_exit_reason(SERVER_STOP_MANUAL,    false, 1)   == "manual",     "asked stop outranks a non-zero exit code");

    // ring: ordered, resumable by seq
    server_model_event_ring ring;
    check(ring.size() == 0 && ring.newest_seq() == 0, "empty ring");
    const uint64_t s1 = ring.record(1000, "a", "loading", "loaded", "", 0);
    const uint64_t s2 = ring.record(2000, "a", "loaded", "unloaded", "idle-ttl", 0);
    const uint64_t s3 = ring.record(3000, "b", "loading", "unloaded", "crash", 139);
    check(s1 == 1 && s2 == 2 && s3 == 3, "seq is 1-based and increasing");
    const auto all = ring.since(0);
    check(all.size() == 3 && all[0].model == "a" && all[2].reason == "crash" && all[2].exit_code == 139, "since(0) returns all, oldest first");
    const auto tail = ring.since(s2);
    check(tail.size() == 1 && tail[0].seq == s3, "since(seq) returns only newer events");
    check(ring.since(s3).empty(), "since(newest) is empty");

    // bounded: 1000 events never exceed the capacity, the oldest drop, seq keeps counting
    server_model_event_ring big;
    for (int i = 0; i < 1000; i++) {
        big.record(i, "m", "loaded", "unloaded", "lru-evict", 0);
    }
    check(big.size() == server_model_event_ring::CAPACITY && big.size() == 256, "ring never exceeds 256");
    const auto held = big.since(0);
    check(held.front().seq == 1000 - 256 + 1 && held.back().seq == 1000, "oldest events dropped, newest kept");
    check(big.since(990).size() == 10, "resume past the drop point still works");

    // capacity 0 would hold nothing and divide a client's cursor logic: it clamps to 1
    server_model_event_ring one(0);
    one.record(1, "m", "a", "b", "", 0);
    one.record(2, "m", "b", "c", "", 0);
    check(one.size() == 1 && one.since(0)[0].seq == 2, "capacity 0 clamps to 1");

    printf("%s\n", n_fail == 0 ? "ALL PASS" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
