// Picks merged from SGLang (tools/server/server-radix-cache.{h,cpp}, server-sched-policy.h, server-queue.cpp, server-common.cpp):
//   1. cache namespace    -- kv_namespace_fingerprint() (request fields cache_salt / extra_key)
//   2. radix eviction     -- server_radix_evict_before() and the real tree under every --radix-evict-policy
//   3. scheduling policy  -- server_sched_pick() and the real server_queue::pop_deferred_task() under --sched-policy
// Parts 1, 2a, 3 need no model. Part 2b (the real tree) runs when a tiny GGUF is given as argv[1] or env SGLANG_TEST_MODEL
// (tests/make_tiny_llama_gguf.py writes one); without it that part prints SKIP, a path that fails to load is a failure.
//
// Row S69-ENGINE-MERGE-PROGRAMME-ALL-ENGINES-2026-09-09-01. SGLang is Apache-2.0; behaviour only, no code copied.

#include "server-task.h"
#include "server-common.h"
#include "server-queue.h"
#include "server-radix-cache.h"
#include "server-sched-policy.h"
#include "common.h"
#include "sampling.h"
#include "suffix-tree.h"
#include "llama.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) {                                              \
        printf("  FAIL: %s -- ", #cond);                        \
        printf(__VA_ARGS__);                                    \
        printf("\n");                                           \
        g_fail++;                                               \
    } else {                                                    \
        printf("  ok:   %s\n", #cond);                          \
    }                                                           \
} while (0)

// ---------------------------------------------------------------------------------------------------------------------------------
static void part1_namespace() {
    printf("part 1: cache namespace fingerprint\n");
    const uint64_t base = 0x1234;
    CHECK(kv_namespace_fingerprint(base, "", "") == base, "no salt, no key: the fingerprint is unchanged (flag-off identical)");
    CHECK(kv_namespace_fingerprint(0, "", "") == 0, "no salt on the no-adapter fingerprint 0 stays 0");
    const uint64_t a = kv_namespace_fingerprint(base, "tenant-a", "");
    const uint64_t b = kv_namespace_fingerprint(base, "tenant-b", "");
    CHECK(a != base && b != base && a != b, "different salts give different fingerprints");
    CHECK(a == kv_namespace_fingerprint(base, "tenant-a", ""), "same salt, same fingerprint");
    CHECK(kv_namespace_fingerprint(base, "x", "") != kv_namespace_fingerprint(base, "", "x"), "salt 'x' and extra_key 'x' are different namespaces");
    CHECK(kv_namespace_fingerprint(base, "ab", "c") != kv_namespace_fingerprint(base, "a", "bc"), "the boundary between the two strings matters");
    CHECK(kv_namespace_fingerprint(0, "s", "") != kv_namespace_fingerprint(1, "s", ""), "the same salt on different adapter fingerprints differs");
    CHECK(kv_namespace_fingerprint(0, "s", "") != 0, "a salted namespace never collides with the unsalted fingerprint 0");
}

// ---------------------------------------------------------------------------------------------------------------------------------
static server_radix_evict_info ei(uint64_t last_use, uint64_t created, uint64_t hits, int32_t prio) {
    server_radix_evict_info i;
    i.last_use = last_use; i.created = created; i.hits = hits; i.priority = prio;
    return i;
}

static void part2a_evict_order() {
    printf("part 2a: eviction order of each policy\n");
    using P = server_radix_evict_policy;
    // three nodes, none ties with another on its policy's primary key
    const auto A = ei(/*lu*/6, /*cr*/1, /*hits*/2, /*prio*/9);
    const auto B = ei(4, 2, 3, 0);
    const auto C = ei(5, 3, 1, 4);
    auto first = [&](P p) {
        const server_radix_evict_info * n[3] = { &A, &B, &C };
        int best = 0;
        for (int i = 1; i < 3; i++) {
            if (server_radix_evict_before(p, 2, *n[i], *n[best])) {
                best = i;
            }
        }
        return "ABC"[best];
    };
    CHECK(first(P::LRU)      == 'B', "lru evicts the least recently used (B)");
    CHECK(first(P::LFU)      == 'C', "lfu evicts the fewest hits (C)");
    CHECK(first(P::FIFO)     == 'A', "fifo evicts the oldest creation (A)");
    CHECK(first(P::MRU)      == 'A', "mru evicts the most recently used (A)");
    CHECK(first(P::FILO)     == 'C', "filo evicts the newest creation (C)");
    CHECK(first(P::SLRU)     == 'C', "slru (threshold 2) evicts the only probationary node (C)");
    CHECK(first(P::PRIORITY) == 'A', "priority evicts the largest priority value, the least important (A)");
    // ties fall back to recency, then creation
    CHECK(server_radix_evict_before(P::LFU, 2, ei(1, 1, 5, 0), ei(2, 2, 5, 0)), "lfu tie on hits: older last use first");
    CHECK(!server_radix_evict_before(P::LRU, 2, ei(1, 1, 5, 0), ei(1, 1, 5, 0)), "an identical pair is not 'before' itself");
    // slru: two protected nodes order by recency, and protected comes after probationary even when older
    CHECK(server_radix_evict_before(P::SLRU, 2, ei(9, 9, 1, 0), ei(1, 1, 7, 0)), "slru: a recent probationary node goes before an old protected one");
    server_radix_evict_policy pp = P::LRU;
    CHECK(server_radix_evict_policy_parse("slru", pp) && pp == P::SLRU, "parse slru");
    CHECK(!server_radix_evict_policy_parse("bogus", pp) && pp == P::SLRU, "an unknown name is refused and leaves the value alone");
    CHECK(std::string(server_radix_evict_policy_name(P::FILO)) == "filo", "name round trip");
}

// ---------------------------------------------------------------------------------------------------------------------------------
struct tiny {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
};

static bool tiny_open(tiny & t, const char * path) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    t.model = llama_model_load_from_file(path, mp);
    if (!t.model) {
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 256;
    cp.n_batch    = 64;
    cp.n_ubatch   = 64;
    cp.n_seq_max  = 8;
    cp.kv_unified = true;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    t.ctx = llama_init_from_model(t.model, cp);
    return t.ctx != nullptr;
}

static bool decode_seq(llama_context * ctx, const llama_tokens & toks, llama_seq_id seq) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        common_batch_add(b, toks[i], (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static llama_tokens range(llama_token from, int n) {
    llama_tokens v;
    for (int i = 0; i < n; i++) {
        v.push_back(from + i);
    }
    return v;
}

// Fill a 3-id tree with A, B, C and give them the hit/recency/priority history from part 2a, then insert D and report which of A/B/C is gone.
static char evict_victim(tiny & t, server_radix_evict_policy policy, uint32_t slru_hits, bool & ok, uint64_t & evicted_tokens) {
    const llama_tokens A = range(5, 8), B = range(20, 8), C = range(30, 8), D = range(40, 8);
    llama_memory_t mem = llama_get_memory(t.ctx);
    llama_memory_seq_rm(mem, -1, -1, -1);
    server_radix_cache radix(t.ctx, /*seq_id_base*/ 1, /*count*/ 3);
    radix.set_evict_policy(policy, slru_hits);
    auto put = [&](const llama_tokens & toks, int32_t prio) {
        llama_memory_seq_rm(mem, 0, -1, -1);
        ok = ok && decode_seq(t.ctx, toks, 0);
        radix.insert(toks, 0, 0, prio);
    };
    put(A, 9); put(B, 0); put(B, 0); put(B, 0); put(C, 4); put(A, 9);   // created A<B<C; last use B<C<A; hits A2 B3 C1; prio A9 B0 C4
    put(D, 0);                                                            // needs a fresh id: one idle leaf goes
    evicted_tokens = radix.get_stats().evict_tokens;
    char gone = '?';
    const std::pair<char, const llama_tokens *> all[] = { {'A', &A}, {'B', &B}, {'C', &C} };
    int n_gone = 0;
    for (const auto & p : all) {
        if (radix.peek_prefix(*p.second, 0) == 0) {
            gone = p.first;
            n_gone++;
        }
    }
    ok = ok && n_gone == 1 && radix.peek_prefix(D, 0) == D.size() && radix.get_stats().evictions == 1;
    return gone;
}

static void part2b_tree(tiny & t) {
    printf("part 2b: the real radix tree under each policy (kv_unified context)\n");
    using P = server_radix_evict_policy;
    struct { P p; char victim; const char * name; } cases[] = {
        {P::LRU, 'B', "lru"}, {P::LFU, 'C', "lfu"}, {P::FIFO, 'A', "fifo"}, {P::MRU, 'A', "mru"},
        {P::FILO, 'C', "filo"}, {P::SLRU, 'C', "slru"}, {P::PRIORITY, 'A', "priority"},
    };
    for (const auto & c : cases) {
        bool ok = true;
        uint64_t evicted_tokens = 0;
        const char got = evict_victim(t, c.p, 2, ok, evicted_tokens);
        CHECK(ok, "%s: setup decoded, exactly one of A/B/C gone, D resident, one eviction counted", c.name);
        CHECK(got == c.victim, "%s evicts %c (got %c)", c.name, c.victim, got);
        CHECK(evicted_tokens == 8, "%s: evict_tokens counts the 8 tokens of the evicted leaf (got %llu)", c.name, (unsigned long long) evicted_tokens);
    }
    // default is lru: a tree that never had set_evict_policy() called behaves as the historical cache
    {
        const llama_tokens A = range(5, 8), B = range(20, 8), C = range(30, 8), D = range(40, 8);
        llama_memory_t mem = llama_get_memory(t.ctx);
        llama_memory_seq_rm(mem, -1, -1, -1);
        server_radix_cache radix(t.ctx, 1, 2);
        CHECK(radix.get_evict_policy() == P::LRU, "a new tree defaults to lru");
        for (const auto * toks : { &A, &B, &C }) {
            llama_memory_seq_rm(mem, 0, -1, -1);
            decode_seq(t.ctx, *toks, 0);
            radix.insert(*toks, 0);
        }
        CHECK(radix.peek_prefix(A, 0) == 0 && radix.peek_prefix(B, 0) == 8 && radix.peek_prefix(C, 0) == 8, "default tree evicted the oldest (A) when C arrived");
        (void) D;
    }
    // peek_prefix is read-only: no locks, no stats
    {
        const llama_tokens A = range(5, 8);
        llama_memory_t mem = llama_get_memory(t.ctx);
        llama_memory_seq_rm(mem, -1, -1, -1);
        server_radix_cache radix(t.ctx, 1, 2);
        decode_seq(t.ctx, A, 0);
        radix.insert(A, 0);
        const auto before = radix.get_stats();
        CHECK(radix.peek_prefix(A, 0) == 8, "peek finds the inserted prompt");
        CHECK(radix.peek_prefix(range(5, 3), 0) == 3, "peek finds a partial edge");
        CHECK(radix.peek_prefix(A, 77) == 0, "peek under another fingerprint finds nothing");
        const auto after = radix.get_stats();
        CHECK(before.hits == after.hits && before.misses == after.misses, "peek leaves the hit/miss counters alone");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
static std::vector<int32_t> toks(int32_t from, int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; i++) v.push_back(from + i);
    return v;
}

// the loop the queue used before --sched-policy existed
static size_t legacy_pick(const std::vector<server_sched_cand> & c) {
    size_t p = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        if (c[i].priority < c[p].priority) p = i;
    }
    return p;
}

static void part3a_sched() {
    printf("part 3a: server_sched_pick\n");
    // fcfs == the legacy loop on random queues
    std::mt19937 rng(7);
    bool same = true;
    for (int it = 0; it < 2000; it++) {
        std::vector<server_sched_cand> c(1 + rng() % 12);
        for (auto & x : c) { x.priority = (int) (rng() % 4) - 1; x.cached_prefix = rng() % 100; }
        server_sched_params sp; // fcfs
        if (server_sched_pick(sp, c).pick != legacy_pick(c)) same = false;
        // lpm with a queue over max_queue is fcfs too
        server_sched_params big; big.policy = server_sched_policy::LPM; big.max_queue = 0;
        if (c.size() > 0 && server_sched_pick(big, c).pick != legacy_pick(c)) same = false;
    }
    CHECK(same, "fcfs (and lpm past max_queue) pick exactly what the old loop picked, 2000 random queues");

    server_sched_params lpm; lpm.policy = server_sched_policy::LPM;
    {
        std::vector<server_sched_cand> c(2);
        c[0].cached_prefix = 0; c[1].cached_prefix = 50;
        CHECK(server_sched_pick(lpm, c).pick == 1, "lpm: the request with the longer cached prefix goes first");
        CHECK(server_sched_pick(lpm, c).fcfs_pick == 0, "...where fcfs would have started the first");
    }
    {
        std::vector<server_sched_cand> c(2);
        c[0].priority = 5; c[0].cached_prefix = 100;
        c[1].priority = 0; c[1].cached_prefix = 0;
        CHECK(server_sched_pick(lpm, c).pick == 1, "lpm: priority still comes first");
    }
    {
        std::vector<server_sched_cand> c(3);
        CHECK(server_sched_pick(lpm, c).pick == 0, "lpm: equal requests keep arrival order");
    }
    {
        // r0 unique, nothing cached; r1 has 30 tokens cached but shares 40 tokens with r0 -> held back behind r0
        auto p0 = toks(100, 60), p1 = toks(100, 40); p1.insert(p1.end(), 500, 500 + 20); // p1 = 40 shared + 20 own
        std::vector<server_sched_cand> c(2);
        c[0].tokens = &p0; c[0].cached_prefix = 0;
        c[1].tokens = &p1; c[1].cached_prefix = 30;
        auto r = server_sched_pick(lpm, c);
        CHECK(r.pick == 0 && r.n_deprioritized == 1, "in-batch rule: r1 shares 40 tokens with the earlier r0 and has <=32 cached, so r0 runs first (pick %zu, depri %zu)", r.pick, r.n_deprioritized);
        server_sched_params off = lpm; off.depri_threshold = -1;
        CHECK(server_sched_pick(off, c).pick == 1, "with the threshold -1 the longer cached prefix wins again");
        // a request with a big cached prefix is not compared at all
        c[1].cached_prefix = 33;
        CHECK(server_sched_pick(lpm, c).pick == 1, "cached prefix 33 > 32: not compared, goes by its cached prefix");
        // different priorities are different classes: r1 is not held back by an earlier request that runs in another class
        c[1].cached_prefix = 30; c[0].priority = 3;
        auto r2 = server_sched_pick(lpm, c);
        CHECK(r2.pick == 1 && r2.n_deprioritized == 0, "r0 (priority 3) is no 'earlier request' for r1 (priority 0): r1 starts, nobody held back (pick %zu, depri %zu)", r2.pick, r2.n_deprioritized);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
static server_task make_task(int id, int32_t prio, const std::vector<int32_t> & prompt) {
    server_task t(SERVER_TASK_TYPE_COMPLETION);
    t.id = id;
    t.params.priority = prio;
    t.tokens = server_tokens(llama_tokens(prompt.begin(), prompt.end()), false);
    return t;
}

// Runs the real queue loop on a thread, defers `tasks`, frees one slot (pop_deferred_task) and returns the id the loop got.
static int pop_first(const server_sched_params & sp, std::function<size_t(const server_task &)> cached, std::vector<server_task> tasks) {
    server_queue q;
    std::mutex m;
    std::atomic<int> got{-1};
    q.on_new_task([&](server_task && t, bool) { int exp = -1; got.compare_exchange_strong(exp, t.id); return true; });
    q.on_update_slots([]() {});
    q.set_sched_policy(sp, std::move(cached));
    std::thread loop([&]() { q.start_loop(); });
    for (auto & t : tasks) {
        q.defer(std::move(t));
    }
    q.pop_deferred_task(0);
    for (int i = 0; i < 400 && got.load() < 0; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    q.terminate();
    loop.join();
    return got.load();
}

static void part3b_queue() {
    printf("part 3b: the real server_queue under --sched-policy\n");
    const auto shared = toks(100, 40);
    auto make_set = [&]() {
        std::vector<server_task> v;
        v.push_back(make_task(10, 0, toks(100, 60)));     // task 10: first in line, nothing cached
        v.push_back(make_task(11, 0, [&]{ auto p = shared; p.insert(p.end(), {500, 501, 502}); return p; }())); // task 11: a cache holds 30 of it
        v.push_back(make_task(12, 0, toks(300, 20)));     // task 12: unrelated
        return v;
    };
    auto cached = [&](const server_task & t) -> size_t { return t.id == 11 ? 30 : 0; };

    server_sched_params fcfs;
    CHECK(pop_first(fcfs, cached, make_set()) == 10, "fcfs starts the first deferred task (10)");

    server_sched_params lpm; lpm.policy = server_sched_policy::LPM;
    // task 11 shares 40 tokens with task 10, so lpm holds it behind task 10 although it has 30 cached tokens; 12 has none
    CHECK(pop_first(lpm, cached, make_set()) == 10, "lpm holds task 11 behind the earlier task 10 that shares its prefix");
    server_sched_params lpm_off = lpm; lpm_off.depri_threshold = -1;
    CHECK(pop_first(lpm_off, cached, make_set()) == 11, "lpm without the hold-back starts task 11, which has the longest cached prefix");
    CHECK(pop_first(lpm, nullptr, make_set()) == 10, "lpm without a cache probe is fcfs");
    // priority first
    auto prio = make_set(); prio[2].params.priority = -1;
    CHECK(pop_first(lpm, cached, std::move(prio)) == 12, "lpm: the lower priority value (task 12) still starts first");
}

static void part3c_spf() {
    printf("part 3c: --sched-policy spf (shortest-prefill-first)\n");
    server_sched_params spf; spf.policy = server_sched_policy::SPF;
    {
        std::vector<server_sched_cand> c(3);
        c[0].n_tokens = 500; c[1].n_tokens = 50; c[2].n_tokens = 200;
        CHECK(server_sched_pick(spf, c).pick == 1, "spf: the shortest uncached prefill (50 tokens) starts first");
        CHECK(server_sched_pick(spf, c).fcfs_pick == 0, "...where fcfs would have started the first");
    }
    {
        std::vector<server_sched_cand> c(2);
        c[0].n_tokens = 500; c[0].cached_prefix = 480; // 20 uncached
        c[1].n_tokens = 50;                            // 50 uncached
        CHECK(server_sched_pick(spf, c).pick == 0, "spf counts only UNCACHED tokens: long prompt with a big cached prefix beats a short cold one");
    }
    {
        std::vector<server_sched_cand> c(2);
        c[0].n_tokens = 500; c[0].priority = 0;
        c[1].n_tokens = 10;  c[1].priority = 5;
        CHECK(server_sched_pick(spf, c).pick == 0, "spf: priority still comes first");
    }
    {
        std::vector<server_sched_cand> c(3);
        c[0].n_tokens = c[1].n_tokens = c[2].n_tokens = 80;
        CHECK(server_sched_pick(spf, c).pick == 0, "spf: equal work keeps arrival order");
    }
    {
        server_sched_params big = spf; big.max_queue = 0;
        std::vector<server_sched_cand> c(2); c[0].n_tokens = 500; c[1].n_tokens = 5;
        CHECK(server_sched_pick(big, c).pick == 0 && !server_sched_pick(big, c).used_lpm, "spf: a queue longer than max_queue is plain fcfs");
    }
    // the real queue
    server_sched_params sp = spf;
    auto set = [&]() {
        std::vector<server_task> v;
        v.push_back(make_task(20, 0, toks(100, 90)));
        v.push_back(make_task(21, 0, toks(200, 30)));
        v.push_back(make_task(22, 0, toks(300, 60)));
        return v;
    };
    CHECK(pop_first(sp, [](const server_task &) -> size_t { return 0; }, set()) == 21, "real server_queue under spf starts the 30-token task (21)");
    CHECK(pop_first(sp, [](const server_task & t) -> size_t { return t.id == 20 ? 80 : 0; }, set()) == 20, "...unless task 20 has 80 tokens cached (10 uncached)");
    server_sched_params fcfs;
    CHECK(pop_first(fcfs, [](const server_task &) -> size_t { return 0; }, set()) == 20, "control: fcfs on the same queue starts task 20");
    server_sched_policy p;
    CHECK(server_sched_policy_parse("shortest-prefill-first", p) && p == server_sched_policy::SPF, "the SGLang name parses to spf");
}

// pop_first variant that lets the test advance the queue's prefill-token counter AFTER the tasks are deferred (aging)
static int pop_first_aged(const server_sched_params & sp, std::vector<server_task> tasks, std::vector<uint64_t> age_after) {
    server_queue q;
    std::atomic<int> got{-1};
    q.on_new_task([&](server_task && t, bool) { int exp = -1; got.compare_exchange_strong(exp, t.id); return true; });
    q.on_update_slots([]() {});
    q.set_sched_policy(sp, [](const server_task &) -> size_t { return 0; });
    std::thread loop([&]() { q.start_loop(); });
    // task i is deferred, then the counter moves by age_after[i]: later arrivals are younger (they waited through fewer tokens)
    size_t i_task = 0;
    for (auto & t : tasks) {
        q.defer(std::move(t));
        if (i_task < age_after.size()) { q.add_prefill_tokens(age_after[i_task]); }
        i_task++;
    }
    q.pop_deferred_task(0);
    for (int i = 0; i < 400 && got.load() < 0; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    q.terminate();
    loop.join();
    return got.load();
}

static void part3d_hrrn() {
    printf("part 3d: --sched-policy hrrn (highest response ratio next, token-aged)\n");
    server_sched_policy p;
    CHECK(server_sched_policy_parse("hrrn", p) && p == server_sched_policy::HRRN, "hrrn parses");
    CHECK(std::string(server_sched_policy_name(server_sched_policy::HRRN)) == "hrrn", "hrrn has a name");
    server_sched_params hr; hr.policy = server_sched_policy::HRRN;
    {   // same wait, shorter uncached work has the higher ratio
        std::vector<server_sched_cand> c(2);
        c[0].n_tokens = 400; c[0].waited_tokens = 1000;  // ratio 2.5
        c[1].n_tokens = 100; c[1].waited_tokens = 1000;  // ratio 10
        CHECK(server_sched_pick(hr, c).pick == 1, "hrrn: equal wait, the shorter uncached prefill has the higher ratio");
        CHECK(server_sched_pick(hr, c).fcfs_pick == 0, "...where fcfs would have started the first");
    }
    {   // aging: a long request that has waited long enough overtakes a fresh short one (the anti-starvation point of hrrn vs spf)
        std::vector<server_sched_cand> c(2);
        c[0].n_tokens = 1000; c[0].waited_tokens = 50000; // ratio 50
        c[1].n_tokens = 10;   c[1].waited_tokens = 100;   // ratio 10
        CHECK(server_sched_pick(hr, c).pick == 0, "hrrn: a long request that waited 50x its own prefill beats a fresh short one");
        server_sched_params spf; spf.policy = server_sched_policy::SPF;
        CHECK(server_sched_pick(spf, c).pick == 1, "control: spf would starve it (starts the short one)");
    }
    {   // nothing uncached: no prefill work, drains first
        std::vector<server_sched_cand> c(2);
        c[0].n_tokens = 50;  c[0].waited_tokens = 100000;
        c[1].n_tokens = 80;  c[1].cached_prefix = 80;
        CHECK(server_sched_pick(hr, c).pick == 1, "hrrn: a request with 0 uncached tokens drains first");
    }
    {   // nothing has waited: ratio 0 for all, arrival order (fcfs)
        std::vector<server_sched_cand> c(3);
        c[0].n_tokens = 500; c[1].n_tokens = 10; c[2].n_tokens = 90;
        CHECK(server_sched_pick(hr, c).pick == 0, "hrrn: nothing waited yet -> arrival order");
    }
    {   // priority first
        std::vector<server_sched_cand> c(2);
        c[0].n_tokens = 500; c[0].waited_tokens = 100; c[0].priority = 0;
        c[1].n_tokens = 10;  c[1].waited_tokens = 100000; c[1].priority = 5;
        CHECK(server_sched_pick(hr, c).pick == 0, "hrrn: priority still comes first");
    }
    {   // above max_queue: plain fcfs
        server_sched_params big = hr; big.max_queue = 0;
        std::vector<server_sched_cand> c(2); c[0].n_tokens = 500; c[1].n_tokens = 5; c[1].waited_tokens = 999;
        CHECK(server_sched_pick(big, c).pick == 0 && !server_sched_pick(big, c).used_lpm, "hrrn: a queue longer than max_queue is plain fcfs");
    }
    // the real queue. The counter moves after each deferral, so an older task has waited through more prefill tokens than a younger one.
    // Case 1: task 40 (1000 tokens) waits through 100+500 tokens (ratio 0.6), younger task 41 (100 tokens) through 500 (ratio 5): hrrn
    // starts 41 although fcfs would start 40 (the "REORDERED" log line).
    auto set1 = [&]() {
        std::vector<server_task> v;
        v.push_back(make_task(40, 0, toks(100, 1000)));
        v.push_back(make_task(41, 0, toks(5000, 100)));
        return v;
    };
    server_sched_params fcfs;
    CHECK(pop_first_aged(hr, set1(), {100, 500}) == 41, "real server_queue under hrrn starts the short younger task (41), ratio 5 vs 0.6");
    CHECK(pop_first_aged(fcfs, set1(), {100, 500}) == 40, "control: fcfs on the same queue starts task 40");
    // Case 2 (what separates hrrn from spf): task 42 (1000 tokens) waits through 50000+100 tokens (ratio 50.1), task 43 (10 tokens) through
    // 100 (ratio 10): hrrn starts 42, spf starts 43 (the short one would starve an old long request forever).
    auto set2 = [&]() {
        std::vector<server_task> v;
        v.push_back(make_task(42, 0, toks(100, 1000)));
        v.push_back(make_task(43, 0, toks(5000, 10)));
        return v;
    };
    server_sched_params spf; spf.policy = server_sched_policy::SPF;
    CHECK(pop_first_aged(hr, set2(), {50000, 100}) == 42, "real server_queue under hrrn: the long task that waited 50000 tokens beats the short one");
    CHECK(pop_first_aged(spf, set2(), {50000, 100}) == 43, "control: spf on the same queue starts the short task (43)");
    // the age is the counter delta since the FIRST deferral: a task deferred when the counter was already high is not 'old'
    {
        server_queue q;
        q.add_prefill_tokens(7000);
        server_task t = make_task(50, 0, toks(100, 10));
        q.defer(std::move(t));
        q.add_prefill_tokens(5);
        CHECK(q.prefill_tokens() == 7005, "the prefill counter accumulates");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Part 4: the stop-token set handed to XGrammar is every EOG token. Needs a real vocabulary with several EOG ids (a chat model such as
// qwen2.5: <|endoftext|> eos, <|im_end|> eot, and more flagged by name); pass it as env SGLANG_TEST_VOCAB_MODEL. Loads the vocab only.
static void part4_xgrammar_stop(const char * vocab_path) {
    printf("part 4: XGrammar stop-token set = every EOG token\n");
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path, mp);
    if (!model) {
        printf("  FAIL: could not open vocab model %s\n", vocab_path);
        g_fail++;
        return;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // the rule before the fix: eos and eot only
    std::set<int32_t> old_rule;
    if (llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL) old_rule.insert(llama_vocab_eos(vocab));
    if (llama_vocab_eot(vocab) != LLAMA_TOKEN_NULL) old_rule.insert(llama_vocab_eot(vocab));
    std::set<int32_t> eog;
    for (llama_token t = 0; t < llama_vocab_n_tokens(vocab); ++t) {
        if (llama_vocab_is_eog(vocab, t)) eog.insert(t);
    }
    printf("  vocab has %zu EOG token(s); the old rule (eos+eot) names %zu\n", eog.size(), old_rule.size());
    CHECK(eog.size() >= 3, "control: this vocab has several EOG ids (the test is meaningful)");
    CHECK(old_rule.size() < eog.size(), "the gap: the eos+eot rule misses %zu EOG token(s)", eog.size() - old_rule.size());
    const std::vector<int32_t> got = common_xgrammar_stop_tokens(vocab);
    const std::set<int32_t> got_set(got.begin(), got.end());
    CHECK(got_set == eog, "common_xgrammar_stop_tokens returns exactly the EOG tokens (%zu)", got.size());
    CHECK(got.size() == got_set.size(), "no id twice");
    bool has_all_old = true;
    for (int32_t t : old_rule) has_all_old = has_all_old && got_set.count(t) > 0;
    CHECK(has_all_old, "eos and eot are still in the set (existing behaviour unchanged)");
    llama_model_free(model);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Part 5: the suffix-decoding history tree has a node cap (row S69-ENGREV-SGLANG2-SUFFIX-TREE-UNBOUNDED-2026-10-27-01).
// DEGRADED PROXY: depth 8 and 2000-token requests instead of depth 64 / 8k, so the run takes about a second; the growth law (nodes per
// history token ~ depth for fresh text, unbounded without a cap) is the same, only the constants differ. Needs no model.
static llama_token lcg_tok(uint64_t & s) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return (llama_token) (4 + (s >> 33) % 30000);
}

static void part5_suffix_cap() {
    printf("part 5: common_suffix_tree node cap\n");
    const int depth = 8, n_req = 20, req_len = 2000;
    // Arm A (control, the old behaviour): no cap. Arm B: cap 60000 nodes.
    common_suffix_tree unb(depth), cap(depth);
    cap.set_max_nodes(60000);
    uint64_t seed = 12345;
    size_t unb_at_5 = 0, unb_at_20 = 0, cap_max = 0;
    for (int r = 0; r < n_req; ++r) {
        std::vector<llama_token> req(req_len);
        for (auto & t : req) t = lcg_tok(seed);
        // fed in 250-token steps, the way draft() feeds the new accepted tokens
        for (int off = 0; off < req_len; off += 250) {
            unb.extend(req.data() + off, 250);
            cap.extend(req.data() + off, 250);
            cap_max = std::max(cap_max, cap.n_nodes());
        }
        if (r == 4)  unb_at_5  = unb.n_nodes();
        if (r == 19) unb_at_20 = unb.n_nodes();
    }
    printf("  uncapped: %zu nodes after 5 requests, %zu after 20 (%d history tokens); capped: max %zu nodes, %zu trims\n",
           unb_at_5, unb_at_20, unb.token_count(), cap_max, cap.n_trims());
    CHECK(unb_at_5 > 0 && unb_at_20 >= 3 * unb_at_5, "control: without a cap the tree grows linearly with the requests (x%.1f from 5 to 20 requests)", (double) unb_at_20 / std::max<size_t>(unb_at_5, 1));
    CHECK(unb.n_trims() == 0 && unb.token_count() == n_req * req_len, "control: max_nodes 0 keeps the whole history (old behaviour)");
    CHECK(cap_max <= 60000, "capped: the tree never exceeds the cap after an extend (max %zu)", cap_max);
    CHECK(cap.n_trims() >= 1, "capped: the history was cut back (%zu trims)", cap.n_trims());
    CHECK(cap.token_count() <= 60000, "capped: the stored history is bounded too (%d tokens)", cap.token_count());
    // a repeat-heavy tail survives the cut: a period-37 pattern fed last must still be speculated after many trims
    std::vector<llama_token> pat(37);
    for (auto & t : pat) t = lcg_tok(seed);
    for (int rep = 0; rep < 60; ++rep) {
        cap.extend(pat.data(), (int) pat.size());
    }
    std::vector<llama_token> ctx(pat.begin(), pat.begin() + 6);
    const auto draft = cap.speculate(ctx.data(), (int) ctx.size(), 8, 0.1f, 1, 3);
    bool right = draft.size() >= 4;
    for (size_t i = 0; i < draft.size() && i < 8; ++i) right = right && draft[i] == pat[6 + i];
    CHECK(right, "capped: a recent repeating pattern is still drafted correctly (%zu tokens)", draft.size());
    CHECK(cap.n_nodes() <= 60000, "capped: still within the cap after the repeat-heavy tail (%zu)", cap.n_nodes());
    // the same pattern in the uncapped tree gives the same draft (accepted length not worse)
    for (int rep = 0; rep < 60; ++rep) {
        unb.extend(pat.data(), (int) pat.size());
    }
    const auto draft_unb = unb.speculate(ctx.data(), (int) ctx.size(), 8, 0.1f, 1, 3);
    CHECK(draft_unb.size() == draft.size(), "uncapped and capped trees draft the same length on the repeat-heavy tail (%zu vs %zu)", draft_unb.size(), draft.size());
    // a tiny cap never loops or crashes
    common_suffix_tree tiny_cap(depth);
    tiny_cap.set_max_nodes(10);
    std::vector<llama_token> junk(5000);
    for (auto & t : junk) t = lcg_tok(seed);
    tiny_cap.extend(junk.data(), (int) junk.size());
    CHECK(tiny_cap.n_nodes() > 0 && tiny_cap.token_count() > 0, "a tiny cap (10 nodes) still works: keeps a 2*depth window");
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : std::getenv("SGLANG_TEST_MODEL");
    printf("test-server-sglang-picks\n");
    part1_namespace();
    part2a_evict_order();
    part3a_sched();
    part3b_queue();
    part3c_spf();
    part3d_hrrn();
    part5_suffix_cap();
    if (const char * vp = std::getenv("SGLANG_TEST_VOCAB_MODEL")) {
        if (*vp) {
            part4_xgrammar_stop(vp);
        }
    } else {
        printf("part 4: SKIP (set SGLANG_TEST_VOCAB_MODEL to a chat model gguf with several EOG ids)\n");
    }
    if (!path || !*path) {
        printf("part 2b: SKIP (no model path: pass argv[1] or set SGLANG_TEST_MODEL)\n");
    } else {
        tiny t;
        if (!tiny_open(t, path)) {
            printf("  FAIL: could not open model %s\n", path);
            g_fail++;
        } else {
            part2b_tree(t);
            llama_free(t.ctx);
            llama_model_free(t.model);
        }
    }
    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
