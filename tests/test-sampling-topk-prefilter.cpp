// Exact top-k prefilter over raw logits (common/topk-scan.h) and its eligibility rule (common_sampler_topk_prefilter_k).
// Row S69-MNN-PICK-TOPK-RAW-LOGIT-THRESHOLD-PREFILTER-2026-10-05-01.   GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: neither common_topk_scan nor params.topk_prefilter existed, so this file could not be compiled at the parent commit; the old
// path's candidate array has n_vocab entries, the prefiltered one has k (check B).
//   A  scan == the order llama_sort produces (logit desc, equal logits by lower id) for random, tie-heavy, -inf-heavy, all-equal, sorted
//      and reverse-sorted datasets, vocab 32k/152k/262k, k 1/20/40/64/1024, AVX2 and scalar both, a NaN never inserted
//   B  eligibility: true with k = max(top_k, min_keep) for a plain top-k-first temperature chain; each fallback condition makes it false
#include "sampling.h"
#include "topk-scan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static std::vector<llama_token_data> reference(const std::vector<float> & l, int k) {
    std::vector<llama_token_data> v;
    for (size_t i = 0; i < l.size(); ++i) {
        if (l[i] == l[i]) v.push_back({ (llama_token) i, l[i], 0.0f });
    }
    auto comp = [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); };
    const size_t kk = std::min<size_t>(k, v.size());
    std::partial_sort(v.begin(), v.begin() + kk, v.end(), comp);
    v.resize(kk);
    return v;
}

static bool same(const std::vector<llama_token_data> & ref, const llama_token_data * got, size_t n) {
    if (ref.size() != n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (ref[i].id != got[i].id) return false;
        if (!(ref[i].logit == got[i].logit)) return false;
    }
    return true;
}

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    const int vocabs[] = { 32000, 151936, 262144 };
    const int ks[] = { 1, 20, 40, 64, 1024 };
    int n_cases = 0;
    double t_ref = 0, t_scalar = 0, t_scan = 0;
    const bool have_avx2 = common_topk_have_avx2();
    printf("A0 AVX2 scan path in this build and on this CPU: %s\n", have_avx2 ? "yes" : "NO (the scalar scan is what runs)");
#if defined(__AVX2__)
    printf("A0 this TU is compiled with -mavx2 (the scan needs no run-time check)\n");
#else
    printf("A0 this TU is compiled WITHOUT -mavx2 (the scan is a function-attribute build chosen at run time)\n");
#endif
    for (int V : vocabs) {
        for (int kind = 0; kind < 7; ++kind) {
            std::vector<float> l(V);
            for (int i = 0; i < V; ++i) {
                float x = nd(rng);
                switch (kind) {
                    case 0: break;                                              // random
                    case 1: x = std::round(x * 2.0f) / 2.0f; break;             // heavy ties
                    case 2: if (rng() % 10 < 3) x = -INFINITY; break;           // 30% -inf
                    case 3: x = 1.5f; break;                                    // all equal
                    case 4: x = (float) i; break;                               // ascending (every element enters the window)
                    case 5: x = (float) (V - i); break;                         // descending (window fills once, nothing enters after)
                    case 6: if (i == V / 3) x = NAN; break;                     // one NaN
                }
                l[i] = x;
            }
            for (int k : ks) {
                std::vector<llama_token_data> o1(k), o2(k);
                auto t0 = std::chrono::steady_clock::now();
                const auto ref = reference(l, k);
                auto t1 = std::chrono::steady_clock::now();
                auto ts0 = std::chrono::steady_clock::now();
                const size_t n1 = common_topk_scan_scalar(l.data(), V, k, o1.data());
                auto ts1 = std::chrono::steady_clock::now();
                CHECK(same(ref, o1.data(), n1));
                if (kind == 0 && k == 40) {
                    t_ref    += std::chrono::duration<double, std::micro>(t1 - t0).count();
                    t_scalar += std::chrono::duration<double, std::micro>(ts1 - ts0).count();
                }
#if defined(COMMON_TOPK_AVX2_PATH)
                if (have_avx2) {
                    auto t2 = std::chrono::steady_clock::now();
                    const size_t n2 = common_topk_scan_avx2(l.data(), V, k, o2.data());
                    auto t3 = std::chrono::steady_clock::now();
                    CHECK(same(ref, o2.data(), n2));
                    // the dispatcher must return the same answer as the explicit AVX2 call
                    std::vector<llama_token_data> o3(k);
                    const size_t n3 = common_topk_scan(l.data(), V, k, o3.data());
                    CHECK(same(ref, o3.data(), n3));
                    if (kind == 0 && k == 40) {
                        t_scan += std::chrono::duration<double, std::micro>(t3 - t2).count();
                    }
                }
#else
                (void) o2;
#endif
                ++n_cases;
            }
        }
    }
    printf("A  %d cases (vocab x dataset x k) agree with the llama_sort order; timing (info only) random k=40, summed over 3 vocabs: reference %.0f us, scalar scan %.0f us, avx2 scan %.0f us\n",
           n_cases, t_ref, t_scalar, t_scan);
    if (have_avx2) {
        CHECK(t_scan > 0.0);   // the AVX2 branch really ran (it did not in a TU built without -mavx2: the fail-before of this change)
    }

    // B
    {
        common_params_sampling p;
        p.topk_prefilter = true;
        p.temp  = 0.8f;
        p.top_k = 40;
        int32_t k = 0;
        CHECK(common_sampler_topk_prefilter_k(p, false, false, false, false, false, 151936, &k) && k == 40);
        p.min_keep = 100;
        CHECK(common_sampler_topk_prefilter_k(p, false, false, false, false, false, 151936, &k) && k == 100);
        p.min_keep = 0;
        common_params_sampling off = p; off.topk_prefilter = false;
        CHECK(!common_sampler_topk_prefilter_k(off, false, false, false, false, false, 151936, &k));
        CHECK(!common_sampler_topk_prefilter_k(p, true,  false, false, false, false, 151936, &k));   // penalties
        CHECK(!common_sampler_topk_prefilter_k(p, false, true,  false, false, false, 151936, &k));   // CFG
        CHECK(!common_sampler_topk_prefilter_k(p, false, false, true,  false, false, 151936, &k));   // grammar / constraint
        CHECK(!common_sampler_topk_prefilter_k(p, false, false, false, true,  false, 151936, &k));   // reasoning budget / FSD / adaptive-p
        CHECK(!common_sampler_topk_prefilter_k(p, false, false, false, false, true,  151936, &k));   // per-token server biases
        auto mod = [&](auto f) { common_params_sampling q = p; f(q); return common_sampler_topk_prefilter_k(q, false, false, false, false, false, 151936, &k); };
        CHECK(!mod([](common_params_sampling & q) { q.temp = 0.0f; }));                       // greedy
        CHECK(!mod([](common_params_sampling & q) { q.mirostat = 1; }));
        CHECK(!mod([](common_params_sampling & q) { q.adaptive_target = 0.5f; }));
        CHECK(!mod([](common_params_sampling & q) { q.fsd_alpha = 0.5f; }));
        CHECK(!mod([](common_params_sampling & q) { q.no_repeat_ngram_size = 3; }));
        CHECK(!mod([](common_params_sampling & q) { q.n_probs = 2; }));
        CHECK(!mod([](common_params_sampling & q) { q.top_k = 0; }));
        CHECK(!mod([](common_params_sampling & q) { q.top_k = 2000; }));
        CHECK(!mod([](common_params_sampling & q) { q.top_k = 151936; }));
        CHECK(!mod([](common_params_sampling & q) { q.samplers_sequence = { llama_sampler_type::TOP_P, llama_sampler_type::TOP_K, llama_sampler_type::TEMPERATURE }; }));
        CHECK(!mod([](common_params_sampling & q) { q.dry_multiplier = 0.8f; }));             // default chain starts with DRY: active DRY is first
        CHECK( mod([](common_params_sampling & q) { q.dry_multiplier = 0.0f; }));             // inactive DRY is skipped
    }
    // C greedy variant: temp == 0 only, same fallbacks, top-k and DRY play no part
    {
        common_params_sampling p;
        p.topk_prefilter = true;
        p.temp = 0.0f;
        p.top_k = 0;
        CHECK( common_sampler_greedy_prefilter_ok(p, false, false, false, false, false));
        CHECK(!common_sampler_greedy_prefilter_ok(p, true,  false, false, false, false));   // penalties
        CHECK(!common_sampler_greedy_prefilter_ok(p, false, true,  false, false, false));   // CFG
        CHECK(!common_sampler_greedy_prefilter_ok(p, false, false, true,  false, false));   // grammar
        CHECK(!common_sampler_greedy_prefilter_ok(p, false, false, false, true,  false));   // special sampler
        CHECK(!common_sampler_greedy_prefilter_ok(p, false, false, false, false, true));    // server bias
        common_params_sampling q = p; q.temp = 0.5f;
        CHECK(!common_sampler_greedy_prefilter_ok(q, false, false, false, false, false));
        q = p; q.n_probs = 3;
        CHECK(!common_sampler_greedy_prefilter_ok(q, false, false, false, false, false));
        q = p; q.topk_prefilter = false;
        CHECK(!common_sampler_greedy_prefilter_ok(q, false, false, false, false, false));
    }
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-sampling-topk-prefilter: all checks passed\n");
    return 0;
}
