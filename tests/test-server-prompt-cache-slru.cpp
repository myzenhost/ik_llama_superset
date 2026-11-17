// Segmented-LRU eviction of the host prompt cache (--cache-ram-slru; row S69-VLLM-PICK-PROMPT-CACHE-SLRU-EVICTION-2026-09-29-01).
//
// SOURCE-READ: engines/vllm/vllm/v1/kv_offload/cpu/policies/arc.py (recent / frequent split) and
// engines/sglang/python/sglang/srt/mem_cache/utils.py:48-84 (SLRU among the offered policies). Apache-2.0, behaviour only.
//
// No model is needed: entries are synthetic (distinct first tokens so none is a prefix of another) of 256 KiB each in a 1 MiB cache, so
// the cache holds 4. A "hit" is what load() does to an entry: touch(). Every expectation has a partner that shows the switch is what
// changed the outcome.

#include "server-task.h"
#include "server-common.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <string>
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

static const size_t ENTRY = 256 * 1024;   // bytes per entry; the cache limit of 1 MiB holds 4

// prompt number k: its first token is k, so no prompt is a prefix of another
static server_prompt make_prompt(llama_token k) {
    llama_tokens toks = { k, 7, 8, 9 };
    server_prompt p { server_tokens(toks, false), {} };
    p.lora_fp = 0;
    return p;
}

static bool has(const server_prompt_cache & c, llama_token k) {
    for (const auto & s : c.states) {
        if (s.prompt.tokens[0] == k) {
            return true;
        }
    }
    return false;
}

static std::list<server_prompt_cache_state>::iterator find_it(server_prompt_cache & c, llama_token k) {
    for (auto it = c.states.begin(); it != c.states.end(); ++it) {
        if (it->prompt.tokens[0] == k) {
            return it;
        }
    }
    return c.states.end();
}

// H (=1) is inserted and restored twice, then 10 one-shot prompts arrive. Returns the cache.
static void run_burst(server_prompt_cache & c) {
    c.alloc(make_prompt(1), ENTRY, 0);
    c.touch(find_it(c, 1));
    c.touch(find_it(c, 1));
    for (llama_token k = 100; k < 110; ++k) {
        c.alloc(make_prompt(k), ENTRY, 0);
    }
}

static void test_burst_does_not_evict_the_hot_entry() {
    printf("test 1: a burst of one-shot prompts vs an entry restored twice\n");
    server_prompt_cache lru(1, 0);
    run_burst(lru);
    CHECK(lru.states.size() == 4, "cache holds 4 entries, got %zu", lru.states.size());
    CHECK(!has(lru, 1), "control (slru off): plain LRU evicted the hot entry");

    server_prompt_cache slru(1, 0);
    slru.slru = true;
    run_burst(slru);
    CHECK(slru.states.size() == 4, "cache holds 4 entries, got %zu", slru.states.size());
    CHECK(has(slru, 1), "slru on: the hot entry survived the burst");
    CHECK(has(slru, 109) && has(slru, 108) && has(slru, 107), "slru on: the newest one-shot entries are the ones kept besides it");
    CHECK(!has(slru, 100), "slru on: the oldest one-shot entry went first");
}

static void test_no_hits_is_plain_lru() {
    printf("test 2: with no restores the order is exactly LRU\n");
    server_prompt_cache c(1, 0);
    c.slru = true;
    for (llama_token k = 100; k < 106; ++k) {
        c.alloc(make_prompt(k), ENTRY, 0);
    }
    CHECK(c.states.size() == 4, "cache holds 4 entries, got %zu", c.states.size());
    CHECK(!has(c, 100) && !has(c, 101), "the two oldest were evicted");
    CHECK(has(c, 102) && has(c, 103) && has(c, 104) && has(c, 105), "the four newest are kept");
    CHECK(c.states.front().prompt.tokens[0] == 102, "front is the oldest survivor");
}

static void test_protected_segment_is_capped() {
    printf("test 3: entries that are all protected still turn over (80 percent cap)\n");
    server_prompt_cache c(1, 0);
    c.slru = true;
    for (llama_token k = 1; k <= 4; ++k) {
        c.alloc(make_prompt(k), ENTRY, 0);
        c.touch(find_it(c, k));
    }
    // all four are protected (100 percent of the limit): a newcomer must still get in, at the cost of the oldest protected one
    c.alloc(make_prompt(50), ENTRY, 0);
    CHECK(has(c, 50), "the newcomer was stored");
    CHECK(!has(c, 1), "the oldest protected entry was evicted");
    CHECK(has(c, 2) && has(c, 3) && has(c, 4), "the other protected entries stay");
}

static void test_probation_goes_before_protected() {
    printf("test 4: a never-restored entry goes before an older restored one\n");
    // entry 1 is the OLDEST and restored; entries 2-4 are newer and never restored. The fifth insert needs room.
    auto fill = [](server_prompt_cache & c) {
        c.alloc(make_prompt(1), ENTRY, 0);
        c.touch(find_it(c, 1));
        c.alloc(make_prompt(2), ENTRY, 0);
        c.alloc(make_prompt(3), ENTRY, 0);
        c.alloc(make_prompt(4), ENTRY, 0);
        c.alloc(make_prompt(5), ENTRY, 0);
    };
    server_prompt_cache slru(1, 0);
    slru.slru = true;
    fill(slru);
    CHECK(!has(slru, 2), "slru on: entry 2 (never restored, oldest probation) was evicted");
    CHECK(has(slru, 1), "slru on: entry 1 (older but restored) stays");

    server_prompt_cache lru(1, 0);
    fill(lru);
    CHECK(!has(lru, 1), "control (slru off): plain LRU evicted the older entry 1");
    CHECK(has(lru, 2), "control (slru off): entry 2 stays");
}

int main() {
    test_burst_does_not_evict_the_hot_entry();
    test_no_hits_is_plain_lru();
    test_protected_segment_is_capped();
    test_probation_goes_before_protected();
    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
