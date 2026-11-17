// Named external n-gram corpora: registry, budget split, versioned incremental matcher, automaton cursor.
// Row S69-ENGREV-SGLANG-NGRAM-EXTERNAL-CORPUS-2026-10-27-01.   GATE-CLAIM-OK: documentation only.   SINGLE-SHOT-OK: unit test, not a data pipeline.
// PRIOR-ART-NONE: fork has common_suffix_automaton (self history only) and no named-corpus registry; scouted all branches by pickaxe on
// corpus_registry / external_corpus / ngram-corpus, only docs hit.
//
// FAIL-BEFORE: common/ngram-corpus-registry.h and the cursor API did not exist, so this file could not be compiled at the parent commit.
//   A  cursor walk: after every context token, cursor.len equals the brute-force longest suffix of the context that occurs in the corpus, and
//      the drafted tokens are what follows a REAL occurrence of that suffix.
//   B  registry: add/append/remove/list, version bumps, token budget refuses a whole add.
//   C  split_budget: sums within the budget, every live match keeps a slot, longest gets the remainder, zero matches get nothing.
//   D  matcher: a context that extends the last one feeds only the new tokens; a different prefix or a changed registry rebuilds.
//   E  two corpora: the draft budget goes only to corpora that match.
//   F  a cursor made before the corpus grew is reset, not read against stale state.
#include "ngram-corpus-registry.h"

#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

typedef std::vector<llama_token> toks;

// longest L such that the last L tokens of ctx occur as a substring of corpus
static int brute_len(const toks & corpus, const toks & ctx) {
    int best = 0;
    const int n = (int) ctx.size();
    for (int L = 1; L <= n && L <= (int) corpus.size(); ++L) {
        bool found = false;
        for (int s = 0; s + L <= (int) corpus.size() && !found; ++s) {
            found = std::equal(ctx.end() - L, ctx.end(), corpus.begin() + s);
        }
        if (found) {
            best = L;
        }
    }
    return best;
}

static bool is_continuation(const toks & corpus, const toks & ctx, int len, const toks & draft) {
    for (int s = 0; s + len <= (int) corpus.size(); ++s) {
        if (!std::equal(ctx.end() - len, ctx.end(), corpus.begin() + s)) {
            continue;
        }
        const int from = s + len;
        if (from + (int) draft.size() <= (int) corpus.size() && std::equal(draft.begin(), draft.end(), corpus.begin() + from)) {
            return true;
        }
    }
    return false;
}

int main() {
    std::mt19937 rng(7);

    // A
    for (int alpha : { 2, 3, 6 }) {
        for (int round = 0; round < 20; ++round) {
            toks corpus(40 + rng() % 80);
            for (auto & t : corpus) { t = (llama_token) (rng() % alpha); }
            common_suffix_automaton sa;
            sa.extend(corpus.data(), (int) corpus.size());
            common_suffix_automaton::cursor c;
            toks ctx;
            for (int i = 0; i < 60; ++i) {
                const llama_token t = (llama_token) (rng() % (alpha + 1));      // alpha itself is never in the corpus
                ctx.push_back(t);
                sa.advance(c, t);
                const int want = brute_len(corpus, ctx);
                CHECK(c.len == want);
                if (want > 0) {
                    const toks d = sa.draft_after(c, 5, 1);
                    CHECK(d.empty() || is_continuation(corpus, ctx, want, d));
                }
            }
        }
    }

    // B
    {
        common_ngram_corpus_registry reg(10);
        const toks a = { 1, 2, 3, 4 }, b = { 5, 6, 7, 8, 9, 10, 11 };
        CHECK(!reg.add("", a.data(), 4));
        CHECK(!reg.add("a", a.data(), 0));
        const uint64_t v0 = reg.version();
        CHECK(reg.add("a", a.data(), 4));
        CHECK(reg.version() != v0);
        CHECK(!reg.add("b", b.data(), 7));                 // 4 + 7 > 10: refused whole
        CHECK(reg.list().size() == 1 && reg.total_tokens() == 4);
        CHECK(reg.add("a", a.data(), 2));                  // append to existing
        CHECK(reg.list()[0].tokens == 6 && reg.list()[0].version == 2);
        CHECK(!reg.remove("zzz"));
        CHECK(reg.remove("a"));
        CHECK(reg.list().empty() && reg.total_tokens() == 0);
    }

    // C
    {
        auto s = common_ngram_corpus_registry::split_budget(10, { 4, 0, 6 });
        CHECK(s.size() == 3 && s[1] == 0 && s[0] >= 1 && s[2] >= s[0]);
        CHECK(s[0] + s[2] == 10);
        auto z = common_ngram_corpus_registry::split_budget(10, { 0, 0 });
        CHECK(z[0] == 0 && z[1] == 0);
        auto t = common_ngram_corpus_registry::split_budget(2, { 3, 9, 5 });
        CHECK(t[1] >= 1 && t[0] + t[1] + t[2] <= 2);       // fewer tokens than live matches: the longest keep a slot
        auto n = common_ngram_corpus_registry::split_budget(0, { 3 });
        CHECK(n[0] == 0);
    }

    // D / E
    {
        common_ngram_corpus_registry reg;
        const toks a = { 10, 11, 12, 13, 14, 15, 16, 17 };
        const toks b = { 20, 21, 22, 23, 24, 25, 26, 27 };
        CHECK(reg.add("a", a.data(), (int) a.size()));
        CHECK(reg.add("b", b.data(), (int) b.size()));
        common_ngram_corpus_registry::matcher m;
        toks ctx = { 10, 11, 12 };
        auto p = m.propose(reg, ctx.data(), (int) ctx.size(), 4, 2);
        CHECK(p.size() == 1 && p[0].corpus == "a" && p[0].match_len == 3 && p[0].tokens == toks({ 13, 14, 15, 16 }));
        CHECK(m.rebuilds() == 1 && m.fed() == 3);
        ctx.push_back(13);
        p = m.propose(reg, ctx.data(), (int) ctx.size(), 4, 2);
        CHECK(m.rebuilds() == 1 && m.fed() == 4);          // only the new token was fed
        CHECK(p.size() == 1 && p[0].tokens == toks({ 14, 15, 16, 17 }));
        toks other = { 10, 99, 12 };                       // different prefix: rebuild
        m.propose(reg, other.data(), (int) other.size(), 4, 2);
        CHECK(m.rebuilds() == 2);
        const toks c2 = { 30, 31 };
        reg.add("c", c2.data(), 2);                        // registry changed: rebuild
        m.propose(reg, other.data(), (int) other.size(), 4, 2);
        CHECK(m.rebuilds() == 3);
        reg.remove("c");
        // E: only corpora that match get budget
        const toks ctx2 = { 12, 13, 14, 15, 99, 24, 25 };
        p = m.propose(reg, ctx2.data(), (int) ctx2.size(), 6, 1);
        int got_a = 0, got_b = 0;
        for (auto & x : p) { (x.corpus == "a" ? got_a : got_b) = (int) x.tokens.size(); }
        CHECK(got_a == 0 && got_b > 0);                    // the match in "a" was broken by 99; only "b" matches "24 25"
        const toks ctx3 = { 13, 14, 15, 24, 25 };
        p = m.propose(reg, ctx3.data(), (int) ctx3.size(), 6, 1);
        got_a = got_b = 0;
        for (auto & x : p) { (x.corpus == "a" ? got_a : got_b) = (int) x.tokens.size(); }
        CHECK(got_b > 0 && got_a == 0);                    // "13 14 15 24 25" is not a substring of a; the suffix "24 25" is in b
    }

    // F
    {
        common_suffix_automaton sa;
        const toks a = { 1, 2, 3, 1, 2 };
        sa.extend(a.data(), (int) a.size());
        common_suffix_automaton::cursor c;
        sa.advance(c, 1);
        sa.advance(c, 2);
        CHECK(c.len == 2 && sa.cursor_end(c) >= 0);
        const toks more = { 9, 9, 9 };
        sa.extend(more.data(), 3);
        CHECK(sa.cursor_end(c) == -1);                     // stale generation is not read
        sa.advance(c, 3);
        CHECK(c.len == 1);                                 // reset, then walked one token
    }

    if (n_fail == 0) {
        printf("OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
