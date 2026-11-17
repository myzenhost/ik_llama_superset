// common_lora_pool: the row's scenario at the policy level.  Row S69-LORAX-PICK-ON-DEMAND-ADAPTER-POOL-2026-09-29-01 (step 1).   GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: the class did not exist; the fork's server can only use adapters named at start-up (server-context.cpp construct_lora_list, row body), so "load a third adapter into a
// pool sized for two" had no implementation to call.  The scenario is the one the row specifies (3 adapters, a pool for 2):
//   A  load 1 and 2, touch 1, plan 3 -> evicts 2 (the LRU idle one), after the swap exactly {1,3} are READY; re-planning 2 evicts the then-LRU one and succeeds again;
//   B  an in-flight adapter is never an eviction candidate: with 1 in flight and 3 idle-LRU... plan evicts the idle one; with BOTH in flight the plan is BUSY, nothing is evicted;
//   C  a corrupt adapter: the first load fails, the memo makes the second plan PLAN_ERRORED with the same message (fast), and clear_error retries;
//   D  explicit unload is refused while a request is in flight, allowed after release;
//   E  an adapter larger than the whole budget is TOO_BIG for ever; unknown ids are UNKNOWN; re-registering a loaded adapter is refused.
#include "lora-pool.h"

#include <cstdio>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static void load(common_lora_pool & p, const std::string & id) {
    const common_lora_pool::plan pl = p.plan_load(id);
    CHECK(pl.status == common_lora_pool::PLAN_LOAD);
    for (const auto & e : pl.evict) {
        p.mark_evicted(e);
    }
    p.mark_loaded(id);
}

int main() {
    using P = common_lora_pool;
    // ---- A
    {
        P p(200);                         // room for two 100-byte adapters
        p.add("a1", "a1.gguf", 100);
        p.add("a2", "a2.gguf", 100);
        p.add("a3", "a3.gguf", 100);
        load(p, "a1");
        load(p, "a2");
        CHECK(p.used() == 200);
        CHECK(p.plan_load("a1").status == P::PLAN_READY);
        CHECK(p.acquire("a1"));           // touch a1 (now more recent than a2)
        p.release("a1");
        const P::plan pl = p.plan_load("a3");
        CHECK(pl.status == P::PLAN_LOAD && pl.evict.size() == 1 && pl.evict[0] == "a2");
        load(p, "a3");
        CHECK((p.ready_ids() == std::vector<std::string>{ "a1", "a3" }));
        CHECK(p.used() == 200);
        // a2 comes back: now a1 is the LRU idle one (a3 was loaded after the touch)
        const P::plan pl2 = p.plan_load("a2");
        CHECK(pl2.status == P::PLAN_LOAD && pl2.evict.size() == 1 && pl2.evict[0] == "a1");
        load(p, "a2");
        CHECK((p.ready_ids() == std::vector<std::string>{ "a2", "a3" }));
    }
    // ---- B
    {
        P p(200);
        p.add("x", "x", 100); p.add("y", "y", 100); p.add("z", "z", 100);
        load(p, "x"); load(p, "y");
        CHECK(p.acquire("x"));            // x in flight AND the older of the two
        const P::plan pl = p.plan_load("z");
        CHECK(pl.status == P::PLAN_LOAD && pl.evict.size() == 1 && pl.evict[0] == "y");   // y is idle: x is protected although it is LRU by load order
        CHECK(p.acquire("y"));            // both in flight
        const P::plan pb = p.plan_load("z");
        CHECK(pb.status == P::PLAN_BUSY && pb.evict.empty());
        CHECK(p.used() == 200);           // nothing was touched by planning
        p.release("y");
        CHECK(p.plan_load("z").status == P::PLAN_LOAD);
    }
    // a plan that needs two evictions
    {
        P p(300);
        p.add("s1", "s", 100); p.add("s2", "s", 100); p.add("s3", "s", 100); p.add("big", "b", 150);
        load(p, "s1"); load(p, "s2"); load(p, "s3");
        CHECK(p.acquire("s2"));
        p.release("s2");                  // recency order, oldest first: s1 (loaded 1st), s3 (loaded 3rd), s2 (touched last)
        const P::plan pl = p.plan_load("big");
        // 300 + 150 = 450 > 300: free s1 (350 still > 300), free s3 (250 <= 300) -> two evictions, s2 survives
        CHECK(pl.status == P::PLAN_LOAD && pl.evict.size() == 2 && pl.evict[0] == "s1" && pl.evict[1] == "s3");
    }
    // ---- C
    {
        P p(200);
        p.add("bad", "bad.gguf", 50);
        P::plan pl = p.plan_load("bad");
        CHECK(pl.status == P::PLAN_LOAD);
        p.mark_failed("bad", "not a LoRA GGUF");
        pl = p.plan_load("bad");
        CHECK(pl.status == P::PLAN_ERRORED && pl.error == "not a LoRA GGUF");
        pl = p.plan_load("bad");          // the second request fails fast with the same message
        CHECK(pl.status == P::PLAN_ERRORED && pl.error == "not a LoRA GGUF");
        CHECK(!p.acquire("bad"));
        p.clear_error("bad");
        CHECK(p.plan_load("bad").status == P::PLAN_LOAD);
    }
    // ---- D
    {
        P p(200);
        p.add("u", "u", 100);
        load(p, "u");
        CHECK(p.can_unload("u"));
        CHECK(p.acquire("u"));
        CHECK(!p.can_unload("u"));
        p.release("u");
        CHECK(p.can_unload("u"));
        p.mark_evicted("u");
        CHECK(p.used() == 0 && p.plan_load("u").status == P::PLAN_LOAD);
        // mark_evicted must not free an in-flight adapter
        load(p, "u");
        CHECK(p.acquire("u"));
        p.mark_evicted("u");
        CHECK(p.find("u")->st == P::STATE_READY);
    }
    // ---- E
    {
        P p(100);
        p.add("huge", "h", 101);
        CHECK(p.plan_load("huge").status == P::PLAN_TOO_BIG);
        CHECK(p.plan_load("nope").status == P::PLAN_UNKNOWN);
        p.add("m", "m", 40);
        load(p, "m");
        CHECK(!p.add("m", "other", 10));  // loaded: refused
        CHECK(p.find("m")->bytes == 40);
        CHECK(p.add("n", "n", 10) && p.add("n", "n2", 20) && p.find("n")->bytes == 20);   // unloaded: replaced
    }
    if (n_fail) {
        fprintf(stderr, "test-lora-pool: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-lora-pool: all passed\n");
    return 0;
}
