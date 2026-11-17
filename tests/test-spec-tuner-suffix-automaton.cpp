// spec_tuner coordinates for the suffix_automaton speculative type.   GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: spec_tuner::init() built coordinates for DRAFT / SUFFIX / the ngram types / DFLASH only, so a suffix_automaton run with
// autotune got the single n_max coordinate and the match length (the one knob the type has) was never tuned: check A fails at the parent commit
// ("no suffix_min_match_len coordinate").
//   A  suffix_automaton gets exactly {n_max, suffix_min_match_len}: no p_min (the type has no per-token probability, so a p_min arm would be a
//      knob that does nothing and would dilute the shared reward)
//   B  controls: SUFFIX still gets {n_max, p_min, suffix_min_match_len}; NGRAM_MOD gets no suffix_min_match_len
//   C  propose() writes suffix_min_match_len into the params, always inside the grid 1..12
//   D  the bandit follows the knob: a separable throughput model whose best match length is 5 (and best n_max 7) is fed through
//      propose()/accept_feedback(); write_best() then reports 4..6, and a model whose best is 9 reports 8..10, at least 2 apart (the answer follows the reward)
#include "spec-tuner.h"

#include "ggml.h"

#include <cstdio>
#include <string>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static const spec_tuner_coord * find_coord(const spec_tuner & t, const char * name) {
    for (const auto & c : t.coords) {
        if (c.name == name) {
            return &c;
        }
    }
    return nullptr;
}

static std::string names(const spec_tuner & t) {
    std::string s;
    for (const auto & c : t.coords) {
        s += (s.empty() ? "" : ",") + c.name;
    }
    return s;
}

// throughput model: smooth peak at (n_max = nbest, len = lbest), strictly positive
static double tps(int n_max, int len, int nbest, int lbest) {
    const double dn = n_max - nbest;
    const double dl = len - lbest;
    return 100.0 - 2.0 * dn * dn - 3.0 * dl * dl;
}

static int tune_to(int lbest, int & n_seen_min, int & n_seen_max) {
    common_params_speculative user;
    user.n_max = 16;
    user.suffix_min_match_len = 3;
    spec_tuner t;
    t.init(COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON, user, nullptr);
    common_params_speculative p = user;
    n_seen_min = 1000;
    n_seen_max = -1000;
    for (int i = 0; i < 6000; ++i) {
        t.propose(p);
        n_seen_min = p.suffix_min_match_len < n_seen_min ? p.suffix_min_match_len : n_seen_min;
        n_seen_max = p.suffix_min_match_len > n_seen_max ? p.suffix_min_match_len : n_seen_max;
        t.accept_feedback(1, 1, tps(p.n_max, p.suffix_min_match_len, 7, lbest));
    }
    common_params_speculative best = user;
    t.write_best(best);
    return best.suffix_min_match_len;
}

int main() {
    // spec_tuner times itself with ggml_time_us(); on Windows that divides by a frequency ggml_time_init() sets (ggml_init() does it in a real process).
    // Without this call the first propose() dies with an integer divide-by-zero (exit 0xC0000094, measured 2026-10-06).
    ggml_time_init();

    common_params_speculative user;
    user.n_max = 16;
    user.suffix_min_match_len = 3;

    {   // A
        spec_tuner t;
        t.init(COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON, user, nullptr);
        printf("A  suffix_automaton coords: %s\n", names(t).c_str());
        CHECK(find_coord(t, "n_max") != nullptr);
        const auto * c = find_coord(t, "suffix_min_match_len");
        CHECK(c != nullptr);
        CHECK(find_coord(t, "p_min") == nullptr);
        CHECK(t.coords.size() == 2);
        if (c) {
            CHECK(c->arms.size() == 12);
            CHECK(c->arms.front().value == 1.0f && c->arms.back().value == 12.0f);
            CHECK((int) c->arms[c->user_idx].value == 3);   // the user's own value is the starting arm
        }
    }
    {   // B controls
        spec_tuner t;
        t.init(COMMON_SPECULATIVE_TYPE_SUFFIX, user, nullptr);
        printf("B  suffix coords: %s\n", names(t).c_str());
        CHECK(find_coord(t, "p_min") != nullptr && find_coord(t, "suffix_min_match_len") != nullptr);
        spec_tuner u;
        u.init(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, user, nullptr);
        printf("B  ngram_mod coords: %s\n", names(u).c_str());
        CHECK(find_coord(u, "suffix_min_match_len") == nullptr);
    }
    // C + D
    int lo = 0, hi = 0;
    const int got5 = tune_to(5, lo, hi);
    printf("C  proposed suffix_min_match_len stayed in [%d, %d]\n", lo, hi);
    CHECK(lo >= 1 && hi <= 12);
    CHECK(hi > lo);   // the tuner explores
    printf("D  best match length for a model peaking at 5: %d\n", got5);
    // epsilon-greedy with a random_device seed: the arms next to the optimum differ by 3 of ~100 throughput units, so the winner can land one arm off (measured: 6 for a
    // peak at 5 in one run, 9 for a peak at 9).  The asserted property is that the answer follows the reward: within one arm of the peak, and the two peaks separate.
    CHECK(got5 >= 4 && got5 <= 6);
    int lo2 = 0, hi2 = 0;
    const int got9 = tune_to(9, lo2, hi2);
    printf("D  best match length for a model peaking at 9: %d\n", got9);
    CHECK(got9 >= 8 && got9 <= 10);
    CHECK(got9 - got5 >= 2);

    if (n_fail) {
        fprintf(stderr, "test-spec-tuner-suffix-automaton: %d failure(s)\n", n_fail);
        return 1;
    }
    printf("test-spec-tuner-suffix-automaton: all checks passed\n");
    return 0;
}
