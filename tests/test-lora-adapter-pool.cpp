// Model-free test for common/lora-adapter-pool.h (S69-LORAX-PICK-ON-DEMAND-ADAPTER-POOL-2026-09-29-01).
// The loader is a fake: "adapters" are ints, sizes come from a table.

#include "lora-adapter-pool.h"

#include <atomic>
#include <cstdio>
#include <map>
#include <thread>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

struct fake_store {
    std::map<std::string, size_t> sizes;
    std::atomic<int>              loads{0};
    std::vector<std::string>      evicted;
    lora_adapter_pool::loader_fn loader() {
        return [this](const std::string & id, lora_adapter_pool::loaded & out) {
            auto it = sizes.find(id);
            if (it == sizes.end()) return false;
            loads++;
            out.handle = std::make_shared<int>((int) id.size());
            out.bytes  = it->second;
            return true;
        };
    }
};

static void test_eviction_order_count() {
    fake_store s; for (auto n : {"a","b","c","d"}) s.sizes[n] = 10;
    lora_adapter_pool p(3, 0, s.loader(), [&](const std::string & id) { s.evicted.push_back(id); });
    for (auto n : {"a","b","c"}) { CHECK(p.acquire(n).st == lora_adapter_pool::status::ok); p.release(n); }
    // touch a -> LRU order is now b, c, a
    CHECK(p.acquire("a").loaded_now == false); p.release("a");
    CHECK((p.lru_order() == std::vector<std::string>{"b","c","a"}));
    CHECK(p.acquire("d").st == lora_adapter_pool::status::ok);
    CHECK((s.evicted == std::vector<std::string>{"b"}));
    CHECK(p.size() == 3 && !p.contains("b"));
    p.release("d");
    // exactly at the limit does not evict anything
    CHECK(s.loads == 4);
}

static void test_byte_budget_and_pinning() {
    fake_store s; s.sizes = {{"a",40},{"b",40},{"c",40},{"big",101},{"x",20}};
    lora_adapter_pool p(0, 100, s.loader(), [&](const std::string & id) { s.evicted.push_back(id); });
    CHECK(p.acquire("a").st == lora_adapter_pool::status::ok);   // pinned
    CHECK(p.acquire("b").st == lora_adapter_pool::status::ok);   // pinned, 80 bytes
    CHECK(p.bytes() == 80);
    CHECK(p.acquire("x").st == lora_adapter_pool::status::ok);   // exactly 100: fits, no eviction
    CHECK(p.bytes() == 100 && s.evicted.empty());
    // c needs 40, everything is pinned -> refused, nothing evicted
    CHECK(p.acquire("c").st == lora_adapter_pool::status::no_room);
    CHECK(p.contains("a") && p.contains("b") && p.contains("x") && s.evicted.empty());
    // bigger than the whole budget -> over_budget even with nothing pinned
    CHECK(p.acquire("big").st == lora_adapter_pool::status::over_budget);
    // unknown id -> load_failed
    CHECK(p.acquire("nope").st == lora_adapter_pool::status::load_failed);
    // unpin b only: c (40) can evict b (40): 100-40+40 = 100
    CHECK(p.release("b"));
    CHECK(p.acquire("c").st == lora_adapter_pool::status::ok);
    CHECK((s.evicted == std::vector<std::string>{"b"}));
    CHECK(p.contains("a") && p.refs("a") == 1);   // pinned a survived
    // double release is refused
    CHECK(p.release("x")); CHECK(!p.release("x"));
    // a second acquire of a pinned adapter stacks refs; one release leaves it pinned
    CHECK(p.acquire("a").st == lora_adapter_pool::status::ok);
    CHECK(p.refs("a") == 2); p.release("a"); CHECK(p.refs("a") == 1);
}

static void test_concurrent() {
    fake_store s;
    const int N = 6;
    for (int i = 0; i < N; i++) s.sizes["m" + std::to_string(i)] = 10;
    lora_adapter_pool p(3, 0, s.loader());
    std::atomic<int> ok{0}, refused{0}, bad{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 8; t++) {
        th.emplace_back([&, t] {
            unsigned x = 12345u + t;
            for (int i = 0; i < 4000; i++) {
                x = x * 1664525u + 1013904223u;
                std::string id = "m" + std::to_string((x >> 16) % N);
                auto a = p.acquire(id);
                if (a.st == lora_adapter_pool::status::ok) {
                    ok++;
                    // while pinned the handle must stay alive and the id must stay resident
                    if (!a.handle || !p.contains(id) || p.refs(id) == 0) bad++;
                    if (p.size() > 3) bad++;
                    p.release(id);
                } else {
                    refused++;
                }
            }
        });
    }
    for (auto & t : th) t.join();
    printf("concurrent: ok=%d refused=%d bad=%d resident=%zu loads=%d\n", ok.load(), refused.load(), bad.load(), p.size(), s.loads.load());
    CHECK(bad == 0);
    CHECK(ok + refused == 8 * 4000);
    CHECK(p.size() <= 3);
    for (int i = 0; i < N; i++) CHECK(p.refs("m" + std::to_string(i)) == 0);
}

static void test_scheduler() {
    // starvation: adapter A has 100 waiting requests, B and C one each; slice = 3
    lora_fair_scheduler f(3);
    for (uint64_t i = 0; i < 100; i++) f.enqueue("A", i);
    f.enqueue("B", 1000);
    f.enqueue("C", 2000);
    std::string order;
    for (int i = 0; i < 8; i++) { auto r = f.pick(); CHECK(r.has_value()); order += r->first; }
    CHECK(order == "AAABCAAA");
    // FIFO within an adapter
    lora_fair_scheduler g(2);
    g.enqueue("X", 1); g.enqueue("X", 2); g.enqueue("X", 3);
    CHECK(g.pick()->second == 1 && g.pick()->second == 2 && g.pick()->second == 3);
    CHECK(!g.pick().has_value());
    // only the current adapter has work: it continues across slices
    lora_fair_scheduler h(1);
    for (uint64_t i = 0; i < 4; i++) h.enqueue("S", i);
    for (uint64_t i = 0; i < 4; i++) CHECK(h.pick()->second == i);
    // time-based slice via injected clock, quanta effectively unlimited
    uint64_t now = 0;
    lora_fair_scheduler t(1000, 50, [&] { return now; });
    for (uint64_t i = 0; i < 10; i++) { t.enqueue("P", i); t.enqueue("Q", 100 + i); }
    std::string o2;
    for (int i = 0; i < 6; i++) { o2 += t.pick()->first; now += 20; }
    // P at t=0,20,40 (start 0); at t=60 >=50 elapsed -> Q at 60,80,100 ; slice start 60
    CHECK(o2 == "PPPQQQ");
    CHECK(f.waiting() == 94);   // 102 enqueued, 8 picked
}

int main() {
    test_eviction_order_count();
    test_byte_budget_and_pinning();
    test_concurrent();
    test_scheduler();
    printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
