// Randomized invariant test for server_prompt_cache (tools/server/server-task.{h,cpp}).
//
// Method from TGI's radix allocator property test (text-generation-inference backends/v3/src/radix.rs:928-1040,
// Apache-2.0): many randomized rounds, every invariant re-checked after every operation. Written against OUR data
// structure -- a whole-prompt state cache, not a block allocator -- so no code is shared with it.
// Row S69-tgi-PICK-PROMPTCACHE-RANDTEST-2026-09-28-01.
//
// Covered: alloc(), update(), remove_prefixes_of(), size(), n_tokens(). NOT covered here: load(), which restores
// into a live llama_context; its non-consuming contract is exercised end to end by
// tools/server/tests/unit/test_cross_slot_prefix_cache.py.

#include "server-task.h"
#include "server-common.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_round = -1;

#define CHECK(cond, ...) do {                                                           \
    if (!(cond)) {                                                                      \
        if (g_fail < 20) {                                                              \
            printf("  FAIL round %d: %s -- ", g_round, #cond);                          \
            printf(__VA_ARGS__);                                                        \
            printf("\n");                                                               \
        }                                                                               \
        g_fail++;                                                                       \
    }                                                                                   \
} while (0)

static bool is_prefix(const llama_tokens & a, const llama_tokens & b) {  // a is a prefix of (or equal to) b
    if (a.size() > b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

static llama_tokens tokens_of(const server_prompt_cache_state & s) {
    return s.prompt.tokens.get_tokens();  // text-only prompts here, so this is the whole sequence
}

static void check_invariants(const server_prompt_cache & c) {
    std::vector<llama_tokens> all;
    std::set<const uint8_t *> buffers;
    for (const auto & s : c.states) {
        all.push_back(tokens_of(s));
        if (!s.data.main.empty()) {
            CHECK(buffers.insert(s.data.main.data()).second, "two entries share one main buffer");
        }
    }
    for (size_t i = 0; i < all.size(); i++) {
        for (size_t j = 0; j < all.size(); j++) {
            if (i != j) {
                CHECK(!is_prefix(all[i], all[j]), "entry %zu (len %zu) is a prefix of entry %zu (len %zu)",
                      i, all[i].size(), j, all[j].size());
            }
        }
    }
    if (c.limit_size > 0) {
        CHECK(c.size() <= c.limit_size, "size %zu > limit %zu", c.size(), c.limit_size);
    }
}

int main(int argc, char ** argv) {
    const uint32_t seed   = argc > 1 ? (uint32_t) std::strtoul(argv[1], nullptr, 10) : 20260928u;
    const int      rounds = argc > 2 ? std::atoi(argv[2]) : 20000;
    printf("test-server-prompt-cache: seed %u, %d rounds\n", seed, rounds);
    std::mt19937 rng(seed);
    auto rnd = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    // limit_size is in MiB; state sizes below are ~0.1-0.6 MiB so a 2 MiB cache holds a handful and must evict.
    server_prompt_cache cache(2, 0);
    server_prompt_cache token_capped(0, 64);  // token-limit-only cache, exercised through update()
    std::vector<llama_tokens> history;        // prompts seen, to generate shared prefixes

    size_t n_stored = 0, n_skipped = 0, n_removed_total = 0;
    for (g_round = 0; g_round < rounds; g_round++) {
        const int op = rnd(0, 9);
        if (op <= 6) {
            // ---- alloc: half the time extend or cut an earlier prompt, so prefix relations are common ----
            llama_tokens toks;
            if (!history.empty() && rnd(0, 1)) {
                toks = history[rnd(0, (int) history.size() - 1)];
                if (rnd(0, 1) && toks.size() > 1) {
                    toks.resize(rnd(1, (int) toks.size() - 1));
                } else {
                    for (int k = rnd(1, 4); k > 0; k--) toks.push_back(rnd(1, 4));
                }
            } else {
                for (int k = rnd(1, 12); k > 0; k--) toks.push_back(rnd(1, 4));
            }
            history.push_back(toks);
            if (history.size() > 64) history.erase(history.begin());

            const size_t sz = (size_t) rnd(100, 600) * 1024;
            std::vector<const server_prompt_cache_state *> before;
            for (const auto & s : cache.states) before.push_back(&s);
            const bool had_superset = [&] {
                for (const auto & s : cache.states) if (is_prefix(toks, tokens_of(s))) return true;
                return false;
            }();

            server_prompt p { server_tokens(toks, false), {} };
            server_prompt_cache_state * got = cache.alloc(p, sz, 0);
            if (got) {
                n_stored++;
                CHECK(!had_superset, "stored although an entry already contained the prompt");
                CHECK(got == &cache.states.back(), "alloc did not return the newest entry");
                CHECK(tokens_of(*got) == toks, "stored tokens differ from the prompt");
                CHECK(got->data.main.size() == sz, "stored %zu bytes, asked %zu", got->data.main.size(), sz);
                for (const auto & s : cache.states) {
                    if (&s != got) CHECK(!is_prefix(tokens_of(s), toks), "a prefix of the new prompt survived alloc");
                }
            } else {
                n_skipped++;
                CHECK(had_superset || sz > cache.limit_size, "alloc refused a prompt with no reason");
                CHECK(cache.states.size() == before.size(), "a refused alloc changed the cache");
            }
            // survivors keep their relative order (eviction is from the front, entries are never reordered here)
            size_t pos = 0;
            for (const auto & s : cache.states) {
                if (&s == got) continue;
                while (pos < before.size() && before[pos] != &s) pos++;
                CHECK(pos < before.size(), "an entry appeared that was neither old nor new");
                pos++;
            }
        } else if (op <= 8) {
            // ---- remove_prefixes_of: erase with a prompt that may or may not cover cached entries ----
            llama_tokens t = history.empty() ? llama_tokens{1} : history[rnd(0, (int) history.size() - 1)];
            if (rnd(0, 2) == 0) t.push_back(rnd(1, 4));
            size_t expect_removed = 0;
            for (const auto & s : cache.states) {
                const auto st = tokens_of(s);
                if (!st.empty() && is_prefix(st, t)) expect_removed++;
            }
            const size_t n_before = cache.states.size();
            const size_t n = cache.remove_prefixes_of(server_tokens(t, false));
            n_removed_total += n;
            CHECK(n == expect_removed, "removed %zu, expected %zu", n, expect_removed);
            CHECK(cache.states.size() == n_before - n, "count after erase is wrong");
            for (const auto & s : cache.states) CHECK(!is_prefix(tokens_of(s), t), "a prefix of the erased tokens survived");
        } else {
            // ---- update() on both caches ----
            cache.update();
            llama_tokens t;
            for (int k = rnd(1, 30); k > 0; k--) t.push_back(rnd(1, 50));
            server_prompt p { server_tokens(t, false), {} };
            token_capped.alloc(p, 1024, 0);
            token_capped.update();
            CHECK(token_capped.n_tokens() <= 64, "token-capped cache holds %zu tokens", token_capped.n_tokens());
        }
        check_invariants(cache);
    }
    printf("  stored %zu, refused %zu, erased %zu, final entries %zu, final size %zu of %zu bytes\n",
           n_stored, n_skipped, n_removed_total, cache.states.size(), cache.size(), cache.limit_size);
    // the test must have actually exercised every path it claims to check
    CHECK(n_stored > 100 && n_skipped > 100 && n_removed_total > 100, "a path was barely exercised");
    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
