// Pinned prefixes of the host prompt cache (--warm-prefix-file; row S69-rtp-llm-PICK-STARTUP-WARMED-PINNED-PREFIX-CACHE-2026-09-29-01).
//
// SOURCE-READ: engines/rtp-llm/docs/backend/reuse_kv_cache.md (MultiTaskPrompt: startup prompts stay resident; an unmet limit is an
// error, not a silent downgrade). Apache-2.0, behaviour only, no code copied.
//
// No model is needed: entries are synthetic of 256 KiB each in a 1 MiB cache, so the cache holds 4. Every expectation has a control that
// shows the pin is what changed the outcome.

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

static const size_t ENTRY = 256 * 1024;

static server_prompt make_prompt(llama_token k, size_t extra = 0, uint64_t lora_fp = 0) {
    llama_tokens toks = { k, 7, 8, 9 };
    for (size_t i = 0; i < extra; ++i) {
        toks.push_back(10 + (llama_token) i);
    }
    server_prompt p { server_tokens(toks, false), {} };
    p.lora_fp = lora_fp;
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

static void pin(server_prompt_cache & c, llama_token k) {
    c.pin_prefix_of(make_prompt(k).tokens, 0);
}

static void test_pinned_entry_survives_a_flood() {
    printf("test 1: a pinned entry survives a flood of one-shot prompts; the same entry unpinned does not\n");
    server_prompt_cache pinned(1, 0);
    pinned.alloc(make_prompt(1), ENTRY, 0);
    pin(pinned, 1);
    for (llama_token k = 100; k < 112; ++k) {
        pinned.alloc(make_prompt(k), ENTRY, 0);
    }
    CHECK(has(pinned, 1), "pinned entry is still there after 12 newcomers");
    CHECK(pinned.states.size() == 4, "cache still holds 4 entries, got %zu", pinned.states.size());
    CHECK(pinned.n_pinned() == 1 && pinned.pinned_size() == ENTRY, "one pinned entry of %zu bytes, got %zu / %zu", ENTRY, pinned.n_pinned(), pinned.pinned_size());

    server_prompt_cache control(1, 0);
    control.alloc(make_prompt(1), ENTRY, 0);
    for (llama_token k = 100; k < 112; ++k) {
        control.alloc(make_prompt(k), ENTRY, 0);
    }
    CHECK(!has(control, 1), "control (not pinned): the entry was evicted");
}

static void test_slru_never_picks_a_pinned_entry() {
    printf("test 2: with slru on, a pinned entry is not a victim, restored or not\n");
    server_prompt_cache c(1, 0);
    c.slru = true;
    c.alloc(make_prompt(1), ENTRY, 0);
    pin(c, 1);
    for (llama_token k = 100; k < 110; ++k) {
        c.alloc(make_prompt(k), ENTRY, 0);
    }
    CHECK(has(c, 1), "pinned entry survived under slru");
}

static void test_a_longer_prompt_does_not_replace_a_pinned_prefix() {
    printf("test 3: a longer prompt that contains a pinned prefix does not delete it\n");
    server_prompt_cache c(1, 0);
    c.alloc(make_prompt(1), ENTRY, 0);
    pin(c, 1);
    c.alloc(make_prompt(1, 2), ENTRY, 0);               // {1,7,8,9,10,11} contains {1,7,8,9}
    CHECK(c.states.size() == 2, "both the pinned prefix and the longer prompt are held, got %zu", c.states.size());

    server_prompt_cache control(1, 0);
    control.alloc(make_prompt(1), ENTRY, 0);
    control.alloc(make_prompt(1, 2), ENTRY, 0);
    CHECK(control.states.size() == 1, "control (not pinned): the shorter prompt was replaced by the longer one, got %zu", control.states.size());
}

static void test_nothing_to_evict_declines_the_newcomer() {
    printf("test 4: when only pinned entries are left, a newcomer is declined and the limit holds\n");
    server_prompt_cache c(1, 0);
    for (llama_token k = 1; k <= 4; ++k) {
        c.alloc(make_prompt(k), ENTRY, 0);
        pin(c, k);
    }
    CHECK(c.pinned_size() == 4 * ENTRY, "the cache is full of pinned bytes");
    auto * st = c.alloc(make_prompt(50), ENTRY, 0);
    CHECK(st == nullptr, "newcomer declined");
    CHECK(c.size() <= c.limit_size, "limit holds: %zu <= %zu", c.size(), c.limit_size);
    CHECK(!c.evict_one(), "evict_one() reports that nothing could be freed");
    c.update();                                          // must terminate, not spin
    CHECK(c.states.size() == 4, "update() with only pinned entries left leaves them alone");
}

static void test_pin_prefix_of_and_share() {
    printf("test 5: pin_prefix_of matches prefixes under the same LoRA fingerprint only; the share check and drop_pinned\n");
    server_prompt_cache c(1, 0);
    c.alloc(make_prompt(1, 0, 0), ENTRY, 0);
    c.alloc(make_prompt(2, 0, 77), ENTRY, 0);
    c.alloc(make_prompt(3), ENTRY, 0);
    CHECK(c.pin_prefix_of(make_prompt(1, 3).tokens, 0) == 1, "the entry that is a prefix of the longer tokens was pinned");
    CHECK(c.pin_prefix_of(make_prompt(2, 3).tokens, 0) == 0, "an entry computed under another LoRA configuration was not");
    CHECK(c.pin_prefix_of(make_prompt(2, 3).tokens, 77) == 1, "...but is pinned under its own fingerprint");
    CHECK(c.pin_prefix_of(make_prompt(3, 0).tokens, 0) == 1, "an identical prompt counts as a prefix");
    CHECK(c.pin_prefix_of(make_prompt(3, 0).tokens, 0) == 0, "pinning twice pins nothing new");
    CHECK(c.n_pinned() == 3, "three pinned, got %zu", c.n_pinned());
    CHECK(c.pinned_over_share(), "3 of 4 slots pinned is over the default share of 0.5");
    c.pinned_share = 0.9f;
    CHECK(!c.pinned_over_share(), "under a 0.9 share it is not");
    c.pinned_share = 0.5f;
    CHECK(c.drop_pinned() == 3 && c.states.empty(), "drop_pinned() deletes the pins and their entries");
}

int main() {
    test_pinned_entry_survives_a_flood();
    test_slru_never_picks_a_pinned_entry();
    test_a_longer_prompt_does_not_replace_a_pinned_prefix();
    test_nothing_to_evict_declines_the_newcomer();
    test_pin_prefix_of_and_share();
    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
