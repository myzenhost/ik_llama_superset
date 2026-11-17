// common_draft_policy: lookup-vs-model-drafter choice and chain depth from learned acceptance and measured round cost.
// Row S69-STRATA-S2-DRAFT-POLICY-COST-MODEL-MTP-LOOKUP-CHAIN-2026-10-07-01.   GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: the class did not exist; composite speculative chains ran with autotune off, so their draft depth never moved
// (common/speculative.cpp "Autotune disabled -- explicit speculative stage chains are not supported yet"). Each case below
// replays recorded rounds into the policy and checks the pick moves the way the expected-tokens-per-ms arithmetic says.
//   A  cold policy: a long match with a high prior wins a long window, a short match keeps the model drafter's window
//   B  per-draft unit: one window of t=5 with 2 accepted drafts is 2 successes + 1 failure, q = (2 + 4*0.75) / (3 + 4)
//   C  learning: windows that all fail drop q until the lookup is no longer taken; windows that all pass raise it again
//   D  measured cost moves the pick: the same acceptance, a steep measured cost for long windows flips lookup to the model drafter
//   E  probing: a size whose cost was only guessed is tried at most 3 times when the lookup looks confident
//   F  chain: appended lookup tokens are taken while the chain keeps passing, dropped when it keeps failing; a round that never
//      reached the chain says nothing about it
//   G  controls: no lookup proposal = model drafter, window clamped, a bad shape falls back, huge margin refuses the plain case
#include "draft-policy.h"

#include <cmath>
#include <cstdio>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)
#define CHECK_NEAR(a, b, tol) do { const double _a = (a), _b = (b); if (!(std::fabs(_a - _b) <= (tol))) { \
    fprintf(stderr, "FAIL %s:%d  %s = %.6f, want %.6f\n", __FILE__, __LINE__, #a, _a, _b); n_fail++; } } while (0)

int main() {
    // A
    {
        common_draft_policy p(8);
        const auto lng = p.choose(4, 7, 30);
        CHECK(lng.lookup && lng.t == 8);
        const auto shrt = p.choose(4, 7, 3);
        CHECK(!shrt.lookup && shrt.t == 4);
        printf("A  long match: lookup=%d t=%d   short match: lookup=%d t=%d\n", lng.lookup, lng.t, shrt.lookup, shrt.t);
    }
    // B
    {
        common_draft_policy p(8);
        CHECK_NEAR(p.lookup_rate(3), 0.75, 1e-12);
        p.observe(true, 5, 2, 3, 0.0);
        CHECK_NEAR(p.lookup_rate(3), 5.0 / 7.0, 1e-12);
        common_draft_policy full(8);
        full.observe(true, 5, 4, 3, 0.0); // all 4 drafts accepted: no failure counted
        CHECK_NEAR(full.lookup_rate(3), (4.0 + 3.0) / (4.0 + 4.0), 1e-12);
    }
    // C
    {
        common_draft_policy p(8);
        CHECK(p.choose(4, 7, 30).lookup);
        for (int i = 0; i < 60; ++i) {
            p.observe(true, 8, 0, 30, 30.0);
        }
        CHECK(p.lookup_rate(30) < 0.25);
        CHECK(!p.choose(4, 7, 30).lookup);
        // short-match bucket starts below the bar; windows that all pass lift it over
        CHECK(!p.choose(4, 7, 3).lookup);
        for (int i = 0; i < 80; ++i) {
            p.observe(true, 8, 7, 3, 30.0);
        }
        CHECK(p.lookup_rate(3) > 0.9);
        CHECK(p.choose(4, 7, 3).lookup);
        printf("C  q(long) after failures %.3f, q(short) after passes %.3f\n", p.lookup_rate(30), p.lookup_rate(3));
    }
    // D: model-drafter t=4 costs 20 ms for ~3.1 tokens (0.155 tok/ms); a lookup window of 8 at q~0.96 commits ~6.9 tokens
    {
        common_draft_policy cheap(8), steep(8);
        for (int i = 0; i < 30; ++i) {
            cheap.observe(false, 4, 2, 0, 20.0);
            steep.observe(false, 4, 2, 0, 20.0);
            cheap.observe(true, 8, 7, 30, 25.0);   // long windows barely cost more
            steep.observe(true, 8, 7, 30, 400.0);  // long windows cost 20x
        }
        const auto a = cheap.choose(4, 7, 30);
        const auto b = steep.choose(4, 7, 30);
        CHECK(a.lookup && a.t == 8);
        // steep long windows: the lookup shrinks to the size whose measured cost equals the model window's (4 tokens at 20 ms
        // beats the model drafter's ~3 at 20 ms), it does not stay at 8
        CHECK(b.lookup && b.t == 4);
        // and when the lookup is also unreliable there, the model drafter keeps the round
        for (int i = 0; i < 60; ++i) {
            steep.observe(true, 4, 0, 30, 20.0);
        }
        CHECK(!steep.choose(4, 7, 30).lookup);
        CHECK_NEAR(steep.cost_ms(8), 400.0, 1e-6);
        // a size never measured is the mean of the measured ones scaled by the shape (sizes 4 and 8 here, equal weight)
        CHECK_NEAR(steep.cost_ms(2), (20.0 * 1.35 / 2.05 + 400.0 * 1.35 / 3.6) / 2.0, 1e-6);
        printf("D  debug: q=%.4f cost2=%.1f cost4=%.1f cost8=%.1f pick lookup=%d t=%d\n", steep.lookup_rate(30),
               steep.cost_ms(2), steep.cost_ms(4), steep.cost_ms(8), (int) b.lookup, b.t);
        printf("D  cheap long windows: lookup=%d t=%d   steep: lookup=%d t=%d\n", a.lookup, a.t, b.lookup, b.t);
    }
    // E
    {
        common_draft_policy p(8, 10.0); // margin 10 = the plain comparison never passes, so only the probe can take the lookup
        int taken = 0;
        for (int i = 0; i < 12; ++i) {
            const auto pk = p.choose(4, 7, 30);
            if (!pk.lookup) {
                break;
            }
            ++taken;
            p.observe(true, pk.t, pk.t - 1, 30, 30.0);
        }
        CHECK(taken == 3);
        CHECK(p.n_cost_samples(8) == 3);
        CHECK(!p.choose(4, 7, 30).lookup);
        printf("E  probes taken: %d\n", taken);
    }
    // F
    {
        common_draft_policy p(8);
        const int k0 = p.chain(3, 0.95, 5, 30);
        CHECK(k0 > 0);
        for (int i = 0; i < 40; ++i) {
            p.observe_chain(3, 5, 2, 30, 20.0); // the model drafter's 2 drafts held, the 5 chained all failed
        }
        CHECK(p.chain_rate(30) < 0.2);
        CHECK(p.chain(3, 0.95, 5, 30) == 0);
        // a round whose model-drafter part failed never reached the chain: chain_rate must not move
        const double before = p.chain_rate(30);
        p.observe_chain(3, 5, 0, 30, 20.0);
        CHECK_NEAR(p.chain_rate(30), before, 1e-12);
        common_draft_policy good(8);
        for (int i = 0; i < 40; ++i) {
            good.observe_chain(3, 3, 5, 30, 18.0); // all 3 chained tokens accepted
        }
        CHECK(good.chain_rate(30) > 0.9);
        CHECK(good.chain(3, 0.95, 3, 30) > 0);
        printf("F  chain depth cold %d, after failures %d, rate %.3f\n", k0, p.chain(3, 0.95, 5, 30), p.chain_rate(30));
    }
    // G
    {
        common_draft_policy p(8);
        const auto none = p.choose(5, 0, 30);
        CHECK(!none.lookup && none.t == 5);
        CHECK(p.choose(20, 0, 0).t == 8);
        CHECK(p.choose(-3, 0, 0).t == 1);
        CHECK(p.chain(8, 1.0, 5, 30) == 0); // no room left in the window
        common_draft_policy::cost_shape bad{};
        common_draft_policy q(8, 0.03, bad);
        CHECK(q.cost_ms(4) > 0.0);
        common_draft_policy huge(8, 10.0);
        // t_full = 8 is unmeasured here, so the probe may fire; take 3 probes then the plain refusal holds
        for (int i = 0; i < 3; ++i) {
            const auto pk = huge.choose(4, 7, 3); // short match: q 0.75 < 0.85, no probe either
            CHECK(!pk.lookup);
        }
    }
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("all draft policy checks passed\n");
    return 0;
}
