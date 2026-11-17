// Gumbel watermark: interop and detection tests for common/watermark.{h,cpp}.
//
// 1. PRF INTEROP: 288 known answers generated from vLLM's OWN reference (vllm/v1/watermarking/prfs/philox.py
//    @ 924707f1, CPU path) as float32 BIT PATTERNS -- 4 keys x 8 contexts (widths 1,3,4,5,8, with -1 padding)
//    x 9 tokens incl. 151935 and INT32_MAX. Exact equality, so our output is scored by vLLM's detector and
//    theirs by ours. Generator: temp scratch wm_kat.py (loads philox.py by file path, torch on CPU).
// 2. Gamma(n,1) survival against scipy.special.gammaincc.
// 3. DETECTION BOTH WAYS on a synthetic high-entropy model: keyed Gumbel sampling with key K is detected
//    with K (p <= 0.01); the same text under another key, and plain-RNG text under K, are not detected
//    in more than a small fraction of runs.
// 4. next_context / next_context_repeated against hand-worked cases.
// 5-9 (2026-09-28, KGW + SynthID): binomial tail vs hand-worked values; scheme names; KGW green fraction
//    = gamma and re-keyed by prev token and key; SynthID LCG hash = transformers' and one tournament layer
//    reproduces transformers' closed-form expected mean g; detection both ways per scheme + cross-scheme.

#include "watermark.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct wm_kat {
    uint64_t key;
    int      width;
    int32_t  ctx[8];
    int32_t  token;
    uint32_t bits;
};
struct gamma_kat {
    double  x;
    int64_t n;
    double  sf;
};
#include "test-watermark-kat.inc"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

using namespace common_watermark;

// Draw n tokens from softmax(logits) over a fixed random vocabulary, either keyed-Gumbel or plain RNG.
static std::vector<int32_t> generate(uint64_t key, bool keyed, int n, int vocab, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<int32_t> out;
    std::vector<int32_t> ids(vocab);
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; ++i) ids[i] = i;
    for (int t = 0; t < n; ++t) {
        for (auto & l : logits) l = nd(rng);
        int32_t tok;
        if (keyed && !next_context_repeated(out, DEFAULT_CONTEXT_WIDTH, DEFAULT_MAX_HISTORY)) {
            int32_t ctx[DEFAULT_CONTEXT_WIDTH];
            next_context(out, DEFAULT_CONTEXT_WIDTH, ctx);
            tok = ids[(size_t) gumbel_select(absorb(key, ctx, DEFAULT_CONTEXT_WIDTH), ids.data(), logits.data(), ids.size())];
        } else {
            std::vector<double> w(vocab);
            float mx = logits[0];
            for (float l : logits) mx = std::max(mx, l);
            for (int i = 0; i < vocab; ++i) w[i] = std::exp((double) logits[i] - mx);
            std::discrete_distribution<int> dd(w.begin(), w.end());
            tok = dd(rng);
        }
        out.push_back(tok);
    }
    return out;
}

int main() {
    // 1. PRF interop
    int n_kat = 0;
    for (const auto & k : KAT) {
        const float u = uniform(k.key, k.ctx, k.width, k.token);
        uint32_t bits;
        std::memcpy(&bits, &u, 4);
        CHECK(bits == k.bits, "PRF key=%016llx width=%d token=%d: got 0x%08x want 0x%08x",
              (unsigned long long) k.key, k.width, k.token, bits, k.bits);
        CHECK(u > 0.0f && u < 1.0f, "uniform out of (0,1): %g", u);
        n_kat++;
    }
    CHECK(n_kat == 288, "expected 288 PRF vectors, have %d", n_kat);

    // 2. Gamma survival
    for (const auto & g : GKAT) {
        const double sf = gamma_survival_int(g.x, g.n);
        const double tol = std::max(1e-12, 1e-9 * g.sf);
        CHECK(std::fabs(sf - g.sf) <= tol, "gamma_sf(x=%g, n=%lld) = %.17g, scipy %.17g",
              g.x, (long long) g.n, sf, g.sf);
    }
    CHECK(gamma_survival_int(0.0, 5) == 1.0 && gamma_survival_int(3.0, 0) == 1.0, "gamma edge cases");

    // 4. contexts
    {
        int32_t c[4];
        next_context({}, 4, c);
        CHECK(c[0] == -1 && c[1] == -1 && c[2] == -1 && c[3] == -1, "empty context not all -1");
        next_context({7, 8}, 4, c);
        CHECK(c[0] == -1 && c[1] == -1 && c[2] == 7 && c[3] == 8, "partial context padding wrong");
        CHECK(!next_context_repeated({}, 4, 8192), "empty history cannot repeat");
        CHECK(!next_context_repeated({1, 2, 3}, 2, 8192), "(2,3) never seen before");
        CHECK(next_context_repeated({1, 2, 3, 1, 2}, 2, 8192), "(1,2) seen before position 2");
        CHECK(!next_context_repeated({1, 2, 3, 1, 2}, 2, 2), "(1,2) outside a 2-position window");
    }

    // 3. detection both ways
    {
        const uint64_t K = 0x0123456789ABCDEFull, OTHER = 0xFEDCBA9876543210ull;
        const int runs = 50, n = 200, vocab = 1000;
        std::mt19937 rng(1234);
        int hit_marked = 0, hit_wrong_key = 0, hit_plain = 0;
        for (int r = 0; r < runs; ++r) {
            auto marked = generate(K, true, n, vocab, rng);
            auto plain  = generate(K, false, n, vocab, rng);
            hit_marked    += detect(K, marked).is_watermarked;
            hit_wrong_key += detect(OTHER, marked).is_watermarked;
            hit_plain     += detect(K, plain).is_watermarked;
        }
        std::printf("detection over %d runs of %d tokens: marked+key %d, marked+other-key %d, plain+key %d\n",
                    runs, n, hit_marked, hit_wrong_key, hit_plain);
        CHECK(hit_marked == runs, "watermarked text not detected in every run (%d/%d)", hit_marked, runs);
        // At p <= 0.01 the expected false-positive count is 0.5 of 50; allow up to 4 (P(X>=5) ~ 2e-4).
        CHECK(hit_wrong_key <= 4, "wrong key detects too often (%d/%d)", hit_wrong_key, runs);
        CHECK(hit_plain <= 4, "unmarked text detected too often (%d/%d)", hit_plain, runs);
    }

    // 5. binomial tail, hand-worked: Bin(10, .5) P(X>=8) = 56/1024; Bin(5, .3) P(X>=4) = 0.03078,
    //    P(X>=5) = 0.3^5; edges.
    CHECK(std::fabs(binomial_survival(8, 10, 0.5) - 56.0 / 1024.0) < 1e-12, "Bin(10,.5) P(X>=8) = %.17g",
          binomial_survival(8, 10, 0.5));
    CHECK(std::fabs(binomial_survival(4, 5, 0.3) - 0.03078) < 1e-12, "Bin(5,.3) P(X>=4) = %.17g",
          binomial_survival(4, 5, 0.3));
    CHECK(std::fabs(binomial_survival(5, 5, 0.3) - 0.00243) < 1e-12, "Bin(5,.3) P(X>=5)");
    CHECK(binomial_survival(0, 5, 0.3) == 1.0 && binomial_survival(6, 5, 0.3) == 0.0, "binomial edges");

    // 6. scheme names round-trip; an unknown name is refused, not mapped to a default
    CHECK(scheme_from_name("gumbel") == SCHEME_GUMBEL && scheme_from_name("kgw") == SCHEME_KGW &&
          scheme_from_name("greenlist") == SCHEME_KGW && scheme_from_name("synthid") == SCHEME_SYNTHID,
          "scheme names");
    CHECK(scheme_from_name("bogus") == -1 && scheme_from_name("") == -1 && scheme_from_name(nullptr) == -1,
          "unknown scheme not refused");
    for (int32_t s : { SCHEME_GUMBEL, SCHEME_KGW, SCHEME_SYNTHID }) {
        CHECK(scheme_from_name(scheme_name(s)) == s, "scheme %d does not round-trip", s);
    }

    // 7. KGW: the green list covers gamma of the vocabulary, and changes with the previous token and the key
    {
        const uint64_t K = 0x0123456789ABCDEFull;
        const int V = 100000;
        for (float gamma : { 0.25f, 0.5f }) {
            int g_a = 0, same = 0, same_key = 0;
            for (int v = 0; v < V; ++v) {
                const bool a = kgw_is_green(K, 42, v, gamma);
                g_a  += a;
                same += a == kgw_is_green(K, 43, v, gamma);
                same_key += a == kgw_is_green(K ^ 1, 42, v, gamma);
            }
            const double frac = (double) g_a / V;
            CHECK(std::fabs(frac - gamma) < 0.01, "kgw green fraction %.4f for gamma %.2f", frac, gamma);
            // independent lists agree on gamma^2 + (1-gamma)^2 of the vocabulary
            const double agree = (double) gamma * gamma + (1.0 - gamma) * (1.0 - gamma);
            CHECK(std::fabs((double) same / V - agree) < 0.01, "kgw list not re-keyed by prev (%.4f)", (double) same / V);
            CHECK(std::fabs((double) same_key / V - agree) < 0.01, "kgw list not re-keyed by key (%.4f)", (double) same_key / V);
        }
        CHECK(kgw_green(kgw_prev_state(K, 7), 99, 0.5f) == kgw_is_green(K, 7, 99, 0.5f), "kgw split != combined");
    }

    // 8. SynthID: the LCG hash is transformers' (h = (h + x) * 6364136223846793005 + 1 from h = 1), and one
    //    tournament layer over a uniform distribution lifts the mean g to 0.5 + 0.25 * (1 - 1/V) -- the
    //    closed form transformers gives as expected_mean_g_value (logits_process.py:2991-3006).
    {
        const uint64_t M = 6364136223846793005ull;
        const int32_t c1[1] = { 5 };
        CHECK(synthid_context_hash(c1, 0) == 1ull, "empty context hash must be 1");
        CHECK(synthid_context_hash(c1, 1) == (1ull + 5ull) * M + 1ull, "one-token LCG step");
        const int32_t cneg[2] = { -1, 7 };
        const uint64_t h1 = (1ull + (uint64_t) (int64_t) -1) * M + 1ull;
        CHECK(synthid_context_hash(cneg, 2) == (h1 + 7ull) * M + 1ull, "-1 padding must wrap as int64");

        const uint64_t K = 0x5EED5EED5EED5EEDull;
        const int V = 50, contexts = 4000;
        std::vector<int32_t> ids(V);
        for (int i = 0; i < V; ++i) ids[i] = i * 7 + 3;
        double mean_g = 0.0, max_sum_err = 0.0;
        for (int c = 0; c < contexts; ++c) {
            const int32_t ctx[4] = { c, c >> 3, 11, -1 };
            const uint64_t ch = synthid_context_hash(ctx, 4);
            std::vector<float> p(V, 1.0f / V);
            synthid_reweight(K, ch, ids.data(), p.data(), V, 1);
            double s = 0.0, eg = 0.0;
            for (int i = 0; i < V; ++i) {
                s  += p[i];
                eg += p[i] * synthid_g(K, ch, ids[i], 0);
            }
            max_sum_err = std::max(max_sum_err, std::fabs(s - 1.0));
            mean_g += eg / contexts;
        }
        const double want = 0.5 + 0.25 * (1.0 - 1.0 / V);
        std::printf("synthid depth-1 mean g over %d contexts: %.5f (closed form %.5f)\n", contexts, mean_g, want);
        CHECK(max_sum_err < 1e-5, "tournament does not preserve total mass (%.3g)", max_sum_err);
        // per-context value m(2-m) has sd ~ 0.07 at V=50; the mean of 4000 has sd ~ 0.0011
        CHECK(std::fabs(mean_g - want) < 0.006, "synthid mean g %.5f vs closed form %.5f", mean_g, want);
    }

    // 9. detection both ways for KGW and SynthID, same synthetic high-entropy model as section 3
    {
        const uint64_t K = 0x0123456789ABCDEFull, OTHER = 0xFEDCBA9876543210ull;
        const int runs = 50, n = 200, vocab = 1000;
        std::mt19937 rng(4321);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        std::vector<int32_t> ids(vocab);
        for (int i = 0; i < vocab; ++i) ids[i] = i;
        auto draw = [&](const std::vector<double> & w) {
            std::discrete_distribution<int> dd(w.begin(), w.end());
            return (int32_t) dd(rng);
        };
        auto softmax = [&](const std::vector<float> & l) {
            float mx = l[0];
            for (float x : l) mx = std::max(mx, x);
            std::vector<double> w(l.size());
            double s = 0.0;
            for (size_t i = 0; i < l.size(); ++i) { w[i] = std::exp((double) l[i] - mx); s += w[i]; }
            for (auto & x : w) x /= s;
            return w;
        };
        int kgw_hit = 0, kgw_other = 0, kgw_plain = 0, sid_hit = 0, sid_other = 0, sid_plain = 0;
        int cross_k_on_s = 0, cross_s_on_k = 0;
        for (int r = 0; r < runs; ++r) {
            std::vector<int32_t> kgw_txt, sid_txt, plain_txt;
            std::vector<float> logits(vocab);
            for (int t = 0; t < n; ++t) {
                for (auto & l : logits) l = nd(rng);
                plain_txt.push_back(draw(softmax(logits)));
                {   // KGW: bias, then sample
                    std::vector<float> b = logits;
                    const uint64_t st = kgw_prev_state(K, kgw_txt.empty() ? -1 : kgw_txt.back());
                    for (int i = 0; i < vocab; ++i) if (kgw_green(st, ids[i], DEFAULT_KGW_GAMMA)) b[i] += DEFAULT_KGW_DELTA;
                    kgw_txt.push_back(draw(softmax(b)));
                }
                {   // SynthID: tournament unless the context repeats
                    auto w = softmax(logits);
                    if (!next_context_repeated(sid_txt, DEFAULT_CONTEXT_WIDTH, DEFAULT_MAX_HISTORY)) {
                        int32_t ctx[DEFAULT_CONTEXT_WIDTH];
                        next_context(sid_txt, DEFAULT_CONTEXT_WIDTH, ctx);
                        std::vector<float> p(w.begin(), w.end());
                        synthid_reweight(K, synthid_context_hash(ctx, DEFAULT_CONTEXT_WIDTH), ids.data(), p.data(),
                                         p.size(), DEFAULT_SYNTHID_DEPTH);
                        w.assign(p.begin(), p.end());
                    }
                    sid_txt.push_back(draw(w));
                }
            }
            kgw_hit   += detect_kgw(K, kgw_txt).is_watermarked;
            kgw_other += detect_kgw(OTHER, kgw_txt).is_watermarked;
            kgw_plain += detect_kgw(K, plain_txt).is_watermarked;
            sid_hit   += detect_synthid(K, sid_txt).is_watermarked;
            sid_other += detect_synthid(OTHER, sid_txt).is_watermarked;
            sid_plain += detect_synthid(K, plain_txt).is_watermarked;
            cross_k_on_s += detect_kgw(K, sid_txt).is_watermarked;
            cross_s_on_k += detect_synthid(K, kgw_txt).is_watermarked;
        }
        std::printf("kgw     over %d runs of %d tokens: marked+key %d, marked+other-key %d, plain+key %d\n",
                    runs, n, kgw_hit, kgw_other, kgw_plain);
        std::printf("synthid over %d runs of %d tokens: marked+key %d, marked+other-key %d, plain+key %d\n",
                    runs, n, sid_hit, sid_other, sid_plain);
        CHECK(kgw_hit == runs, "kgw text not detected in every run (%d/%d)", kgw_hit, runs);
        CHECK(sid_hit == runs, "synthid text not detected in every run (%d/%d)", sid_hit, runs);
        CHECK(kgw_other <= 4 && kgw_plain <= 4, "kgw false positives %d/%d", kgw_other, kgw_plain);
        CHECK(sid_other <= 4 && sid_plain <= 4, "synthid false positives %d/%d", sid_other, sid_plain);
        // cross-scheme, same key: the domain separators must keep one scheme's detector blind to another's text
        std::printf("cross-scheme same key: kgw-detector on synthid text %d, synthid-detector on kgw text %d\n",
                    cross_k_on_s, cross_s_on_k);
        CHECK(cross_k_on_s <= 4 && cross_s_on_k <= 4, "cross-scheme detections %d/%d", cross_k_on_s, cross_s_on_k);
        CHECK(detect_kgw(K, {}).n_scored == 0 && detect_synthid(K, {}).n_scored == 0 &&
              detect_synthid(K, {}).p_value == 1.0, "empty input must score nothing and report p = 1");
    }

    std::printf("%s: %d failure(s), %d PRF vectors\n", g_fail ? "FAIL" : "PASS", g_fail, n_kat);
    return g_fail ? 1 : 0;
}
