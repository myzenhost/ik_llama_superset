// Standalone test for the token loop guard in server-repetition.h (row S69-ENGREV-BEELLAMA-LOOPGUARD-TOKEN-DETECTORS-2026-10-27-01).
// Build: g++ -std=c++17 -I.. test_server_loop_guard.cpp -o t && ./t
#include "server-repetition.h"
#include <cstdio>
#include <vector>

using namespace server_repetition;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static loop_guard_result run(const std::vector<int> & toks, bool reasoning, loop_guard_params p = {}) {
    token_loop_guard<int> g(p);
    loop_guard_result last;
    for (int t : toks) {
        g.accept(t, reasoning);
        if (g.should_check(reasoning, false)) {
            last = g.check(reasoning);
            if (last.triggered) {
                return last;
            }
        }
    }
    return last;
}

// deterministic varied stream: no repeats inside 8 tokens, no dominance
static std::vector<int> varied(size_t n, uint32_t seed) {
    std::vector<int> v;
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        v.push_back((int) (s >> 8) % 5000);
    }
    return v;
}

int main() {
    // 1. '}\n' x200 -> periodic tail, period 2 (opposite: the varied stream of the same length does not trigger)
    {
        std::vector<int> t = varied(600, 7);
        for (int i = 0; i < 600; ++i) { t.push_back(125); t.push_back(10); } // fills the 1024-token window
        CHECK(run(t, false).triggered); // while the window still holds a few pre-loop tokens low_entropy may fire first
        std::vector<int> pure;
        for (int i = 0; i < 300; ++i) { pure.push_back(125); pure.push_back(10); }
        const auto r = run(pure, false);
        CHECK(r.triggered && r.kind == "periodic_tail" && r.period == 2);
        CHECK(!run(varied(1000, 3), false).triggered);
    }
    // 2. period 40 copy of a paragraph, fuzzy: one mismatch per 400 tokens is still a loop
    {
        std::vector<int> para = varied(40, 11), t = varied(100, 5);
        for (int i = 0; i < 30; ++i) { t.insert(t.end(), para.begin(), para.end()); }
        t[t.size() - 130] = -1; // a single changed token
        const auto r = run(t, false);
        CHECK(r.triggered && r.kind == "periodic_tail" && r.period == 40);
    }
    // 3. n-gram dominance is for the reasoning region only: scattered repeated 16-grams covering >= half of the window
    {
        std::vector<int> block = varied(32, 21), t;
        for (int i = 0; i < 16; ++i) { auto f = varied(8, 100 + i); t.insert(t.end(), block.begin(), block.end()); t.insert(t.end(), f.begin(), f.end()); }
        loop_guard_params p;
        p.max_period = 4; // the periodic detector cannot see period 40: only dominance can
        CHECK(run(t, true, p).triggered && run(t, true, p).kind == "ngram_dominance");
        CHECK(run(t, false, p).kind != "ngram_dominance"); // visible region: no dominance check
    }
    // 4. low entropy: 1100 tokens drawn from 6 distinct ids, no periodicity at max_period 4
    {
        std::vector<int> t;
        uint32_t s = 99;
        for (int i = 0; i < 1100; ++i) { s = s * 1664525u + 1013904223u; t.push_back((int) (s >> 16) % 6); }
        loop_guard_params p;
        p.max_period = 4;
        p.min_repeated_coverage = 256;
        const auto r = run(t, false, p);
        CHECK(r.triggered && r.kind == "low_entropy");
    }
    // 5. 20 distinct legal-style paragraphs: zero triggers in either region
    {
        std::vector<int> t;
        for (int i = 0; i < 20; ++i) { auto para = varied(120, 1000 + i * 17); t.insert(t.end(), para.begin(), para.end()); }
        CHECK(!run(t, true).triggered);
        CHECK(!run(t, false).triggered);
    }
    // 6. reasoning region honours min_reasoning_tokens: the same loop shorter than the minimum is not judged
    {
        std::vector<int> t;
        for (int i = 0; i < 100; ++i) { t.push_back(1); t.push_back(2); }
        CHECK(!run(t, true).triggered);   // 200 tokens < 512
        for (int i = 0; i < 200; ++i) { t.push_back(1); t.push_back(2); }
        CHECK(run(t, true).triggered);
    }
    // 7. state rewind: a copy taken before speculative tokens restores exactly
    {
        token_loop_guard<int> g;
        for (int i = 0; i < 700; ++i) { g.accept(varied(1, 500 + i)[0], true); }
        const token_loop_guard<int> saved = g;
        for (int i = 0; i < 600; ++i) { g.accept(1, true); g.accept(2, true); }
        CHECK(g.check(true).triggered);
        g = saved;
        CHECK(!g.check(true).triggered);
        CHECK(g.seen(true) == 700);
    }
    // 8. contiguous floor (rtp-llm min_duplicate_tokens): '=====' (1 token x 5) is dropped, a long loop is kept
    {
        std::vector<int> bar(5, 61); // five '=' tokens in a row
        CHECK(contiguous_pattern_repeated(bar, 1, 4, 3, 0));        // no floor: flagged (the false positive)
        CHECK(!contiguous_pattern_repeated(bar, 1, 4, 3, 32));      // floor 32: 3 copies of <= 4 tokens is < 32 duplicated tokens
        std::vector<int> loop;
        for (int i = 0; i < 40; ++i) { loop.push_back(125); loop.push_back(10); }
        CHECK(contiguous_pattern_repeated(loop, 1, 4, 3, 32));      // '}\n' x40 = 80 tokens >= 32
        std::vector<int> mid;
        for (int i = 0; i < 8; ++i) { mid.push_back(7); mid.push_back(8); mid.push_back(9); mid.push_back(10); }
        CHECK(contiguous_pattern_repeated(mid, 1, 4, 3, 32));       // 4-token pattern x 8 copies; copies>=3 and len*copies>=32
    }
    std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
