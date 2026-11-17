// Standalone test for server-loop-recovery.h. Build: g++ -std=c++17 -O2 -I.. test_server_loop_recovery.cpp -o t && ./t
//
// Two parts.
//  1. Unit checks with opposite-answer controls: the coverage score, the token-id effort splice and the mode parser.
//  2. MEASUREMENT (printed, and asserted where the result decides something): the coverage detector against the
//     existing non-contiguous span detector (server-repetition.h, 4096-token window) on synthetic reasoning streams.
//     One synthetic word = one token, which FAVOURS the span detector (real text runs about 1.3 tokens per word, so
//     its 4096-token window covers fewer words than here). The question it answers: does the coverage detector catch a
//     loop the span detector misses? Scenario A says yes once the loop period P satisfies (min_count - 1) * P + span_min_len
//     > 4096 (P > ~2040 for min_count 3, > ~4080 for 2); below that the span detector fires first, and both stay silent on
//     fresh text (C).
#include "server-loop-recovery.h"
#include "server-repetition.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace server_loop_recovery;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

// deterministic word source
struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 2654435761ULL + 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t) (s >> 11); }
};

static std::string join(const std::vector<int32_t> & ids) {
    std::string t;
    for (const int32_t id : ids) {
        t += "w" + std::to_string(id) + " ";
    }
    return t;
}

struct first_fire {
    long span   = -1;  // index of the output token at which the span detector first fired
    long cov    = -1;  // index at which the coverage detector first fired (checked every p.every tokens)
    float cov_at = 0;  // coverage score at that check
};

// Feed a stream token by token, the way process_token sees it.
static first_fire run_stream(const std::vector<int32_t> & stream, int32_t span_len, int32_t span_count, const coverage_params & p,
                             bool want_span = true) {
    first_fire f;
    std::vector<int32_t> out;
    out.reserve(stream.size());
    std::vector<uint64_t> words;
    for (size_t i = 0; i < stream.size(); ++i) {
        out.push_back(stream[i]);
        if (want_span && f.span < 0 && server_repetition::noncontiguous_span_repeated(out, span_len, span_count)) {
            f.span = (long) i;
        }
        if (f.cov < 0 && p.threshold > 0 && (long) (i + 1) % p.every == 0) {
            const float c = reasoning_coverage(join(out), p);
            if (c >= p.threshold) {
                f.cov    = (long) i;
                f.cov_at = c;
            }
        }
        if (f.cov >= 0 && (f.span >= 0 || !want_span)) {
            break;
        }
    }
    return f;
}

static std::vector<int32_t> random_words(rng & r, size_t n, int32_t vocab = 50000) {
    std::vector<int32_t> v(n);
    for (auto & x : v) {
        x = (int32_t) (r.next() % (uint32_t) vocab);
    }
    return v;
}

int main() {
    const coverage_params P; // Strata's defaults: 12-grams, >= 3 times, last 2000 of 30000 words, 0.25, every 512

    // ---- 1a. tokenizer of the coverage score ------------------------------------------------------------------------
    {
        const auto a = words_hashed("Hello, World!  hello world", 0);
        CHECK(a.size() == 6);              // hello , world ! hello world
        CHECK(a[0] == a[4]);               // lowercased
        CHECK(a[2] == a[5]);
        CHECK(a[1] != a[0] && a[1] != a[3]);
        const auto b = words_hashed("a b c d e", 2);
        CHECK(b.size() == 2 && b[0] == words_hashed("d", 0)[0]);
        const auto u = words_hashed("caf\xc3\xa9 caf", 0); // UTF-8 bytes stay inside the word
        CHECK(u.size() == 2 && u[0] != u[1]);
    }

    // ---- 1b. coverage score: controls --------------------------------------------------------------------------------
    {
        rng r(1);
        // shorter than the window -> 0 whatever it contains
        const auto loop = random_words(r, 100);
        std::vector<int32_t> s;
        for (int k = 0; k < 15; ++k) s.insert(s.end(), loop.begin(), loop.end());   // 1500 words < 2000
        CHECK(reasoning_coverage(join(s), P) == 0.0f);
        // a 100-word paragraph repeated 30 times: everything is covered
        for (int k = 0; k < 15; ++k) s.insert(s.end(), loop.begin(), loop.end());
        CHECK(reasoning_coverage(join(s), P) > 0.99f);
        // opposite control: 30,000 random words are not a loop
        const auto fresh = random_words(r, 30000);
        CHECK(reasoning_coverage(join(fresh), P) == 0.0f);
        // a phrase seen only twice does not count (min_count 3)
        std::vector<int32_t> two = random_words(r, 1500);
        two.insert(two.end(), loop.begin(), loop.end());
        const auto mid = random_words(r, 300);
        two.insert(two.end(), mid.begin(), mid.end());
        two.insert(two.end(), loop.begin(), loop.end());
        CHECK(reasoning_coverage(join(two), P) == 0.0f);
        // ... and a third copy makes exactly that paragraph count: 300/2000 = 0.15 of the window (two copies inside it)
        two.insert(two.end(), loop.begin(), loop.end());
        const float c3 = reasoning_coverage(join(two), P);
        CHECK(c3 > 0.149f && c3 < 0.151f);
        // threshold <= 0 is "off" for the caller; the score itself still computes
        coverage_params off = P;
        off.window = 0;
        CHECK(reasoning_coverage(join(s), off) == 0.0f);
    }

    // ---- 1c. effort splice on token ids ------------------------------------------------------------------------------
    {
        // token ids: 1 BOS, 2 <|im_start|>, 3 <|im_end|> (turn end), 4 "system", 5 "user", 6 "\n", 7 "\n\n"
        // high sentence = 100 101 102 103 ".", where "." is 104 and ".\n\n" glued is 105; low sentence = 200 201 104
        const std::vector<int32_t> high   = {100, 101, 102, 103, 104};
        const std::vector<int32_t> low    = {200, 201, 104};
        const std::vector<int32_t> high_g = {100, 101, 102, 103, 105};
        const std::vector<int32_t> low_g  = {200, 201, 105};
        const std::vector<splice_variant<int32_t>> vars = {{high, low}, {high_g, low_g}};
        auto is_end = [](int32_t t) { return t == 3; };

        // system turn holding only the sentence (the template's no-system-prompt case), then a user turn, then output
        std::vector<int32_t> p = {1, 2, 4, 6, 100, 101, 102, 103, 104, 3, 6, 2, 5, 6, 50, 51, 3, 6, 2, 300, 301, 302};
        const size_t fe = first_turn_end(p, is_end);
        CHECK(fe == 10);
        auto r = effort_splice(p, fe, vars);
        CHECK(r.ok && r.pos == 4 && r.n_old == 5 && r.n_new == 3);
        const std::vector<int32_t> want = {1, 2, 4, 6, 200, 201, 104, 3, 6, 2, 5, 6, 50, 51, 3, 6, 2, 300, 301, 302};
        CHECK(r.tokens == want);   // everything after the sentence (prompt rest + generated tokens) kept token for token

        // the system prompt follows the sentence, so the tokenizer glued "." to "\n\n": only the second variant matches
        std::vector<int32_t> g = {1, 2, 4, 6, 100, 101, 102, 103, 105, 60, 61, 3, 6, 2, 5, 6, 50, 3};
        auto rg = effort_splice(g, first_turn_end(g, is_end), vars);
        CHECK(rg.ok && rg.pos == 4 && rg.n_old == 5);
        CHECK(rg.tokens == (std::vector<int32_t>{1, 2, 4, 6, 200, 201, 105, 60, 61, 3, 6, 2, 5, 6, 50, 3}));

        // opposite control: the sentence appears only in a LATER (user) turn -> never touched
        std::vector<int32_t> u = {1, 2, 4, 6, 70, 71, 3, 6, 2, 5, 6, 100, 101, 102, 103, 104, 3};
        auto ru = effort_splice(u, first_turn_end(u, is_end), vars);
        CHECK(!ru.ok && ru.tokens.empty());

        // the sentence split differently (not whole tokens): no match, the caller degrades to stop
        std::vector<int32_t> s = {1, 2, 4, 6, 100, 999, 102, 103, 104, 3};
        CHECK(!effort_splice(s, first_turn_end(s, is_end), vars).ok);

        // a variant whose high equals its low is ignored (nothing to change)
        CHECK(!effort_splice(p, fe, std::vector<splice_variant<int32_t>>{{high, high}}).ok);

        // no turn end at all: the whole prompt is the first turn
        std::vector<int32_t> raw = {9, 9, 100, 101, 102, 103, 104, 9};
        CHECK(first_turn_end(raw, is_end) == raw.size());
        CHECK(effort_splice(raw, raw.size(), vars).ok);
    }

    // ---- 1d. mode parser --------------------------------------------------------------------------------------------
    {
        int32_t m = -1;
        CHECK(parse_mode("recover", m) && m == MODE_RECOVER);
        CHECK(parse_mode("true", m) && m == MODE_RECOVER);
        CHECK(parse_mode("stop", m) && m == MODE_STOP);
        CHECK(parse_mode("off", m) && m == MODE_OFF);
        CHECK(parse_mode("false", m) && m == MODE_OFF);
        CHECK(!parse_mode("resume", m));
        CHECK(std::string(mode_name(MODE_RECOVER)) == "recover");
    }

    // ---- 2. MEASUREMENT: coverage detector vs span detector ----------------------------------------------------------
    // span detector settings: span_min_len 16 with span_min_count 3 (and 2, its most sensitive setting)
    std::printf("MEASUREMENT (1 word = 1 token; span window %zu tokens; coverage %d-grams x%d, last %d of %d words, >= %.2f, every %d)\n",
                server_repetition::NONCONTIG_WINDOW, P.ngram, P.min_count, P.window, P.history, P.threshold, P.every);
    std::printf("%-44s %10s %10s %10s %8s\n", "scenario", "span(16,3)", "span(16,2)", "coverage", "cov");

    // A: the whole reasoning cycle (a 200-word verification pass + a code draft) is rewritten identically, period c
    long a_span3_miss_cov_hit = 0;
    // MEASURED 2026-10-17 (first run of this test): span(16, k) still fires while (k - 1) * period + 16 <= 4096, i.e. up to a
    // period of about 2040 for k = 3 and 4080 for k = 2. The coverage detector's own region is the periods beyond that.
    long a_span2_miss_cov_hit = 0;
    for (const int c : {600, 1000, 1400, 2000, 2500, 3000, 4000, 5000}) {
        rng r(7 + c);
        const auto cycle = random_words(r, (size_t) c);
        std::vector<int32_t> st = random_words(r, 500);
        while (st.size() < (size_t) (500 + 5 * c)) st.insert(st.end(), cycle.begin(), cycle.end());
        const first_fire f3 = run_stream(st, 16, 3, P);
        const first_fire f2 = run_stream(st, 16, 2, P);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "A identical cycle, period %d", c);
        std::printf("%-44s %10ld %10ld %10ld %8.3f\n", tag, f3.span, f2.span, f3.cov, f3.cov_at);
        CHECK(f3.cov >= 0); // coverage catches every period here
        if (c >= 2500) {
            // the decision: span (min_count 3) cannot hold 3 copies of a > ~2040-token period in its 4096-token window
            CHECK(f3.span < 0);
            a_span3_miss_cov_hit += (f3.span < 0 && f3.cov >= 0);
        } else {
            CHECK(f3.span >= 0);         // up to ~2040 the span detector already sees it ...
            CHECK(f3.span < f3.cov);     // ... and sooner than the coverage detector
        }
        if (c >= 5000) {
            CHECK(f2.span < 0);          // even its most sensitive setting (min_count 2) misses a period > ~4080
            a_span2_miss_cov_hit += (f2.span < 0 && f3.cov >= 0);
        } else {
            CHECK(f2.span >= 0);
        }
    }

    // A2: same as A, period 2500, but every copy has 2 % of its words changed (a model never redrafts byte-identically)
    {
        rng r(99);
        const auto cycle = random_words(r, 2500);
        std::vector<int32_t> st = random_words(r, 500);
        for (int k = 0; k < 5; ++k) {
            auto cp = cycle;
            for (auto & x : cp) {
                if (r.next() % 100 < 2) x = (int32_t) (60000 + r.next() % 50000);
            }
            st.insert(st.end(), cp.begin(), cp.end());
        }
        const first_fire f3 = run_stream(st, 16, 3, P);
        const first_fire f2 = run_stream(st, 16, 2, P);
        std::printf("%-44s %10ld %10ld %10ld %8.3f\n", "A2 period 2500, 2% words mutated per copy", f3.span, f2.span, f3.cov, f3.cov_at);
        CHECK(f3.span < 0);
        CHECK(f3.cov >= 0);
    }

    // B: the same 300-word verification pass with a NEW (never repeated) draft between copies, period c
    for (const int c : {800, 1200, 2000}) {
        rng r(31 + c);
        const auto pass = random_words(r, 300);
        std::vector<int32_t> st;
        for (int k = 0; k < 8; ++k) {
            const auto draft = random_words(r, (size_t) (c - 300));
            st.insert(st.end(), draft.begin(), draft.end());
            st.insert(st.end(), pass.begin(), pass.end());
        }
        const first_fire f3 = run_stream(st, 16, 3, P);
        const first_fire f2 = run_stream(st, 16, 2, P);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "B same pass, fresh drafts, period %d", c);
        std::printf("%-44s %10ld %10ld %10ld %8.3f\n", tag, f3.span, f2.span, f3.cov, f3.cov_at);
    }

    // C: control, 20,000 fresh words: nothing may fire
    {
        rng r(5);
        const auto st = random_words(r, 20000);
        const first_fire f3 = run_stream(st, 16, 3, P);
        std::printf("%-44s %10ld %10s %10ld %8.3f\n", "C control, 20000 fresh words", f3.span, "-", f3.cov, f3.cov_at);
        CHECK(f3.span < 0 && f3.cov < 0);
    }

    // D: short back-to-back loop (period 50): both see it; the span detector first
    {
        rng r(11);
        const auto cycle = random_words(r, 50);
        std::vector<int32_t> st = random_words(r, 300);
        while (st.size() < 4000) st.insert(st.end(), cycle.begin(), cycle.end());
        const first_fire f3 = run_stream(st, 16, 3, P);
        std::printf("%-44s %10ld %10s %10ld %8.3f\n", "D period 50 back to back", f3.span, "-", f3.cov, f3.cov_at);
        CHECK(f3.span >= 0 && f3.cov >= 0 && f3.span < f3.cov);
    }

    std::printf("scenario A periods >= 2500 caught by coverage and missed by span(16,3): %ld of 4; missed by span(16,2): %ld of 1\n",
                a_span3_miss_cov_hit, a_span2_miss_cov_hit);
    std::printf(fails ? "FAILED: %d\n" : "ALL PASSED%.0d\n", fails);
    return fails ? 1 : 0;
}
