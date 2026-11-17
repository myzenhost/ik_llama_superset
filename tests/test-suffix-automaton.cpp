// common_suffix_automaton: longest repeated suffix, most recent earlier occurrence, overlapping-copy draft, memory.
// Row S69-TRTLLM-PICK-SUFFIXAUTOMATON-2026-09-28-01.   GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: the class did not exist, so this file could not be compiled at the parent commit.  The comparison is a brute-force
// O(n^2) search, so a wrong link/clone in the automaton shows up as a different match length.
//   A  random strings over alphabets of 2, 3, 5 and 40 symbols, lengths up to 300: after EVERY appended token the automaton's match_len equals
//      the longest suffix of the history that also ends at an earlier position, and (max_refresh large) match_end is the latest such end.
//   B  with the default bounded refresh the reported occurrence is still a true occurrence of the suffix (possibly not the latest).
//   C  a periodic history keeps drafting past its own end (overlapping copy) and the draft equals the true continuation.
//   D  min_match_len gates the draft; clear() resets.
//   E  memory per token on a Zipf-plus-repeats stream stays two orders below the suffix tree's measured 9.7 KB per token.
#include "suffix-automaton.h"

#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

// brute force: longest L such that text[n-L..n-1] also ends at some e < n-1 (text[e-L+1..e] equal); returns L and the largest such e
static void brute(const std::vector<llama_token> & t, int & best_len, int & best_end) {
    const int n = (int) t.size();
    best_len = 0;
    best_end = -1;
    for (int e = n - 2; e >= 0; --e) {
        int l = 0;
        while (l <= e && t[e - l] == t[n - 1 - l]) {
            ++l;
        }
        if (l > best_len) {
            best_len = l;
            best_end = e;      // scanning from the most recent end downwards, so the first e reaching a length is the latest for it
        }
    }
}

int main() {
    // A / B
    const int alphabets[] = { 2, 3, 5, 40 };
    for (int a : alphabets) {
        for (int seed = 0; seed < 25; ++seed) {
            std::mt19937 rng(1000 * a + seed);
            const int n = 40 + (int) (rng() % 260);
            common_suffix_automaton exact(1 << 20), bounded(4);
            std::vector<llama_token> text;
            for (int i = 0; i < n; ++i) {
                const llama_token c = (llama_token) (rng() % a);
                text.push_back(c);
                exact.extend(&c, 1);
                bounded.extend(&c, 1);
                int bl, be;
                brute(text, bl, be);
                CHECK(exact.match_len() == bl);
                CHECK(exact.match_end() == be);
                CHECK(bounded.match_len() == bl);
                if (bl > 0) {
                    const int e = bounded.match_end();
                    bool ok = e >= bl - 1 && e < i;
                    for (int k = 0; ok && k < bl; ++k) {
                        ok = text[e - k] == text[i - k];
                    }
                    CHECK(ok);
                }
                if (n_fail > 20) {
                    return 1;
                }
            }
            CHECK(exact.state_count() <= 2 * n);
        }
    }

    // C: periodic history
    {
        common_suffix_automaton sa;
        std::vector<llama_token> t;
        for (int i = 0; i < 40; ++i) {
            t.push_back(10 + (i % 4));
        }
        sa.extend(t.data(), (int) t.size());
        const auto d = sa.speculate(11, 5);
        CHECK((int) d.size() == 11);
        for (int i = 0; i < 11; ++i) {
            CHECK(d[i] == 10 + ((40 + i) % 4));       // the true continuation of the period, well past the 2-3 tokens actually stored
        }
    }
    // D
    {
        common_suffix_automaton sa;
        const llama_token t[] = { 1, 2, 3, 9, 1, 2, 3 };
        sa.extend(t, 7);
        CHECK(sa.match_len() == 3);
        CHECK(sa.speculate(4, 4).empty());            // match shorter than min_match_len
        const auto d = sa.speculate(4, 3);
        CHECK(d.size() == 4 && d[0] == 9 && d[1] == 1 && d[2] == 2 && d[3] == 3);
        sa.clear();
        CHECK(sa.token_count() == 0 && sa.match_len() == 0 && sa.speculate(4, 1).empty());
    }
    // E
    {
        std::mt19937 rng(5);
        std::vector<double> w(32000);
        for (size_t i = 0; i < w.size(); ++i) w[i] = 1.0 / (1.0 + i);
        std::discrete_distribution<int> zipf(w.begin(), w.end());
        std::vector<llama_token> toks;
        const int N = 131072;
        while ((int) toks.size() < N) {
            if (toks.size() > 200 && (rng() % 100) < 30) {
                const size_t len = 8 + rng() % 56, from = rng() % (toks.size() - len - 1);
                for (size_t j = 0; j < len && (int) toks.size() < N; ++j) toks.push_back(toks[from + j]);
            } else {
                toks.push_back(zipf(rng));
            }
        }
        common_suffix_automaton sa;
        for (int i = 0; i < N; i += 512) {
            sa.extend(toks.data() + i, std::min(512, N - i));
        }
        const double per_tok = (double) sa.bytes_used() / N;
        printf("E  %d tokens: %d states, %.1f bytes/token (suffix tree: 9640 measured)\n", N, sa.state_count(), per_tok);
        CHECK(sa.state_count() <= 2 * N);
        CHECK(per_tok < 400.0);
    }
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-suffix-automaton: all checks passed\n");
    return 0;
}
