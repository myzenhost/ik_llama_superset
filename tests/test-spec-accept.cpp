// Statistical test of the speculative-decoding acceptance rules in common/spec-accept.h.
// Row S69-SPEC-DECODE-REJECTION-SAMPLING-ACCEPTANCE-PICK-2026-09-28-01.
//
// Both rules must emit tokens distributed EXACTLY as the target p (chi-square vs p over 200k trials), and each must
// accept at its analytic rate (rejection: sum min(p,q); sample-and-match: sum p*q). Two NEGATIVE CONTROLS must FAIL
// the same chi-square, or the test cannot tell a wrong rule from a right one:
//   A. always accept the draft          -> emits q, not p
//   B. rejection with the WRONG residual (resample from p, not norm(max(0, p - q))) -- the subtle bug class
// Fixed seeds, so the run is deterministic.

#include "spec-accept.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } \
                              else { std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static const int    K       = 8;
static const int    N       = 200000;
static const double CHI2_CRIT = 29.88; // chi-square 0.9999 quantile, df = K - 1 = 7

static common_spec_dist make(const std::vector<double> & w) {
    common_spec_dist d;
    for (int i = 0; i < (int) w.size(); ++i) {
        if (w[i] > 0.0) {
            d.ids.push_back(i);
            d.probs.push_back((float) w[i]);
        }
    }
    return d;
}

static llama_token draw_idx(const std::vector<double> & w, std::mt19937 & rng) {
    std::discrete_distribution<int> dd(w.begin(), w.end());
    return dd(rng);
}

static double chi2(const std::vector<long> & counts, const std::vector<double> & p, long n) {
    double s = 0.0;
    for (int i = 0; i < K; ++i) {
        const double e = p[i] * n;
        if (e > 0.0) {
            s += (counts[i] - e) * (counts[i] - e) / e;
        } else if (counts[i] > 0) {
            s += 1e9; // a token with zero target mass was emitted
        }
    }
    return s;
}

enum rule_t { RULE_REJECTION, RULE_SAMPLE_MATCH, RULE_ALWAYS_ACCEPT, RULE_WRONG_RESIDUAL };

struct run_result { double chi2; double acc_rate; };

// x ~ draft_w (the drafter's real distribution); q is what the rule is TOLD the drafter's distribution is.
static run_result run(rule_t rule, const std::vector<double> & pw, const std::vector<double> & draft_w,
                      const common_spec_dist & q, uint32_t seed) {
    const common_spec_dist p = make(pw);
    std::mt19937 rng(seed);
    std::vector<long> counts(K, 0);
    long acc = 0;
    for (int t = 0; t < N; ++t) {
        const llama_token x = draw_idx(draft_w, rng);
        common_spec_accept_result r{};
        switch (rule) {
            case RULE_REJECTION:     r = common_spec_accept_rejection(p, q, x, rng);   break;
            case RULE_SAMPLE_MATCH:  r = common_spec_accept_sample_match(p, x, rng);    break;
            case RULE_ALWAYS_ACCEPT: r = { x, true };                                   break;
            case RULE_WRONG_RESIDUAL: {
                const double px = common_spec_dist_prob(p, x), qx = common_spec_dist_prob(q, x);
                std::uniform_real_distribution<double> u(0.0, 1.0);
                if (qx > 0.0 && u(rng) < std::min(1.0, px / qx)) { r = { x, true }; }
                else { r = { draw_idx(pw, rng), false }; } // BUG on purpose: resamples from p, not max(0, p - q)
            } break;
        }
        counts[r.token]++;
        acc += r.accepted;
    }
    return { chi2(counts, pw, N), (double) acc / N };
}

static double sigma(double a) { return std::sqrt(a * (1.0 - a) / N); }

int main() {
    const std::vector<double> pw = { .30, .20, .15, .10, .10, .08, .05, .02 };
    const std::vector<double> qw = { .05, .10, .30, .20, .05, .15, .10, .05 };
    const common_spec_dist p = make(pw), q = make(qw);

    const double a_rej = common_spec_accept_rate_rejection(p, q);
    const double a_sm  = common_spec_accept_rate_sample_match(p, q);
    CHECK(std::fabs(a_rej - 0.60) < 1e-6 && std::fabs(a_sm - 0.123) < 1e-6,
          "analytic rates: sum min(p,q) = %.6f (want 0.60), sum p*q = %.6f (want 0.123)", a_rej, a_sm);
    CHECK(a_rej > a_sm + 0.05, "rejection accepts more than sample-and-match for a stochastic draft (%.3f vs %.3f)", a_rej, a_sm);

    // 1. rejection rule: exact + analytic acceptance
    run_result r = run(RULE_REJECTION, pw, qw, q, 1234);
    CHECK(r.chi2 < CHI2_CRIT, "rejection emits p: chi2 = %.2f < %.2f", r.chi2, CHI2_CRIT);
    CHECK(std::fabs(r.acc_rate - a_rej) < 5 * sigma(a_rej), "rejection acceptance %.4f vs analytic %.4f (5 sigma = %.4f)",
          r.acc_rate, a_rej, 5 * sigma(a_rej));

    // 2. sample-and-match (today's rule): exact + analytic acceptance
    r = run(RULE_SAMPLE_MATCH, pw, qw, q, 5678);
    CHECK(r.chi2 < CHI2_CRIT, "sample-and-match emits p: chi2 = %.2f < %.2f", r.chi2, CHI2_CRIT);
    CHECK(std::fabs(r.acc_rate - a_sm) < 5 * sigma(a_sm), "sample-and-match acceptance %.4f vs analytic %.4f", r.acc_rate, a_sm);

    // 3. greedy draft (one-hot q): the two rules have the SAME acceptance -- the scope the row records
    {
        const std::vector<double> gw = { 0, 0, 1.0, 0, 0, 0, 0, 0 };
        const common_spec_dist g = make(gw);
        const double ar = common_spec_accept_rate_rejection(p, g), as = common_spec_accept_rate_sample_match(p, g);
        CHECK(std::fabs(ar - as) < 1e-9 && std::fabs(ar - pw[2]) < 1e-6, "greedy draft: rates equal (%.4f, %.4f) = p(x) %.2f", ar, as, pw[2]);
        r = run(RULE_REJECTION, pw, gw, g, 91);
        CHECK(r.chi2 < CHI2_CRIT && std::fabs(r.acc_rate - ar) < 5 * sigma(ar),
              "greedy draft rejection: chi2 = %.2f, acceptance %.4f vs %.4f", r.chi2, r.acc_rate, ar);
    }

    // 4. sparse q (top-4 support, renormalised internally) with the drafter really sampling from that support
    {
        const std::vector<double> sw = { 0, .10, .30, .20, 0, .15, 0, 0 };
        const common_spec_dist s = make(sw);
        r = run(RULE_REJECTION, pw, sw, s, 77);
        CHECK(r.chi2 < CHI2_CRIT, "sparse-support q: chi2 = %.2f < %.2f", r.chi2, CHI2_CRIT);
        const double as = common_spec_accept_rate_rejection(p, s);
        CHECK(std::fabs(r.acc_rate - as) < 5 * sigma(as), "sparse-support acceptance %.4f vs analytic %.4f", r.acc_rate, as);
    }

    // 5. support mismatch: the drafter draws tokens the reported q gives NO mass -> the fallback must keep it exact
    {
        const std::vector<double> sw = { 0, .10, .30, .20, 0, .15, 0, 0 };
        const common_spec_dist s = make(sw);
        const std::vector<double> off = { .4, 0, 0, 0, .3, 0, .2, .1 }; // every draft lies OUTSIDE s's support
        r = run(RULE_REJECTION, pw, off, s, 4242);
        CHECK(r.chi2 < CHI2_CRIT, "q(x)=0 fallback stays exact: chi2 = %.2f < %.2f", r.chi2, CHI2_CRIT);
    }

    // 6. RELAXED acceptance (S69-INFERFLOW-PICK-RELAXED-SPEC-ACCEPT-2026-10-05-01, opt-in, LOSSY). Nucleus of p at top_p = 0.65 is {0, 1, 2}
    //    (mass before token 2 is .50 < .65, before token 3 is .65 which is not < .65). Inside it a draft is always kept; outside it the exact rule
    //    decides with the same random stream; the emitted distribution is therefore NOT p (the price), and the analytic acceptance is
    //    sum_x q(x) * (x in nucleus ? 1 : min(1, p(x)/q(x))).
    {
        const float top_p = 0.65f;
        {   // a token the exact rule rejects is kept inside the nucleus, left alone outside it
            const llama_token inside = 2, outside = 3;      // p(2)=.15 < q(2)=.30, so the exact rule rejects it about half the time
            int exact_rej_in = 0, relaxed_kept = 0, differs_out = 0, not_kept = 0, miscounted = 0;
            for (int s = 0; s < 4000; ++s) {
                std::mt19937 r1(s), r2(s), r3(s), r4(s);
                const auto e_in  = common_spec_accept_rejection(p, q, inside, r1);
                const auto x_in  = common_spec_accept_relaxed(p, q, inside, r2, top_p);
                const auto e_out = common_spec_accept_rejection(p, q, outside, r3);
                const auto x_out = common_spec_accept_relaxed(p, q, outside, r4, top_p);
                if (!(x_in.accepted && x_in.token == inside)) { ++not_kept; }
                if (!e_in.accepted) { ++exact_rej_in; relaxed_kept += x_in.relaxed ? 1 : 0; }
                else if (x_in.relaxed) { ++miscounted; }
                if (e_out.token != x_out.token || e_out.accepted != x_out.accepted || x_out.relaxed) { ++differs_out; }
            }
            CHECK(not_kept == 0, "relaxed keeps the nucleus token in all 4000 seeds (%d not kept)", not_kept);
            CHECK(miscounted == 0, "an exact accept is never counted as relaxed (%d)", miscounted);
            CHECK(exact_rej_in > 1000 && relaxed_kept == exact_rej_in, "relaxed flag counts exactly the overturned rejections (%d of %d)", relaxed_kept, exact_rej_in);
            CHECK(differs_out == 0, "outside the nucleus the verdict is the exact rule's, bit for bit (%d differences)", differs_out);
        }
        {   // acceptance gain vs the exact rule with a stochastic drafter
            const std::vector<double> nuc = { 1, 1, 1, 0, 0, 0, 0, 0 };
            double analytic = 0.0;
            for (int i = 0; i < K; ++i) {
                const double pi = pw[i], qi = qw[i];
                analytic += qi * (nuc[i] > 0 ? 1.0 : std::min(1.0, pi / qi));
            }
            std::mt19937 rng(31337);
            long acc = 0, rel = 0;
            std::vector<long> counts(K, 0);
            for (int t = 0; t < N; ++t) {
                const llama_token x = draw_idx(qw, rng);
                const auto rr = common_spec_accept_relaxed(p, q, x, rng, top_p);
                acc += rr.accepted; rel += rr.relaxed; counts[rr.token]++;
            }
            const double rate = (double) acc / N;
            CHECK(std::fabs(rate - analytic) < 5 * sigma(analytic), "relaxed acceptance %.4f vs analytic %.4f (exact rule: %.4f)", rate, analytic, a_rej);
            CHECK(rate > a_rej + 0.02 && rel > 0, "relaxation gains acceptance over the exact rule and reports it (%ld relaxed of %ld)", rel, acc);
            CHECK(chi2(counts, pw, N) > 10 * CHI2_CRIT, "the relaxed rule is LOSSY: its emissions fail the chi-square against p (chi2 = %.1f), as documented", chi2(counts, pw, N));
        }
        {   // top_p outside (0,1) is the exact rule
            std::mt19937 r1(5), r2(5);
            const auto a = common_spec_accept_rejection(p, q, 1, r1);
            const auto b = common_spec_accept_relaxed(p, q, 1, r2, 0.0f);
            CHECK(a.token == b.token && a.accepted == b.accepted && !b.relaxed, "top_p = 0 is the exact rule");
        }
    }

    // NEGATIVE CONTROLS -- each must FAIL the chi-square by a wide margin, or the checks above prove nothing.
    r = run(RULE_ALWAYS_ACCEPT, pw, qw, q, 999);
    CHECK(r.chi2 > 10 * CHI2_CRIT, "NEGATIVE A (always accept -> emits q) is rejected: chi2 = %.1f", r.chi2);
    r = run(RULE_WRONG_RESIDUAL, pw, qw, q, 1001);
    CHECK(r.chi2 > 10 * CHI2_CRIT, "NEGATIVE B (wrong residual, resample from p) is rejected: chi2 = %.1f", r.chi2);

    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
