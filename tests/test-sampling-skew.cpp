// Row S69-APHRODITE-PICK-SKEW-SAMPLING-2026-10-18-01: common/sampling-skew.h against the ExLlamaV2 definition
// (ext_sampling.cpp:239-281, MIT): sort by probability, draw u^(exp(-skew)) through the cumulative distribution.
// Checked analytically (token j gets F_j^a - F_{j-1}^a, a = exp(skew)) and by simulating the ExLlamaV2 draw itself.

#include "sampling-skew.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static std::vector<llama_token_data> from_probs(const std::vector<float> & p, const std::vector<int> & order) {
    std::vector<llama_token_data> v;
    for (int i : order) v.push_back({ (llama_token) i, std::log(p[i]), 0.0f });
    return v;
}

static float prob_of(const std::vector<llama_token_data> & v, llama_token id) {
    double s = 0.0, t = 0.0;
    for (const auto & d : v) {
        const double e = std::isfinite(d.logit) ? std::exp((double) d.logit) : 0.0;
        s += e;
        if (d.id == id) t = e;
    }
    return (float) (t / s);
}

static void test_closed_form() {
    const std::vector<float> p = { 0.2f, 0.5f, 0.3f }; // sorted: id1 0.5, id2 0.3, id0 0.2
    for (float skew : { -1.0f, -0.3f, 0.4f, 1.5f }) {
        auto v = from_probs(p, { 0, 1, 2 });
        llama_token_data_array a = { v.data(), v.size(), -1, true };
        assert(common_sampling_skew_apply(&a, skew));
        assert(!a.sorted);
        const double e = std::exp((double) skew);
        const double want1 = std::pow(0.5, e);
        const double want2 = std::pow(0.8, e) - std::pow(0.5, e);
        const double want0 = 1.0 - std::pow(0.8, e);
        assert(std::fabs(prob_of(v, 1) - want1) < 1e-5);
        assert(std::fabs(prob_of(v, 2) - want2) < 1e-5);
        assert(std::fabs(prob_of(v, 0) - want0) < 1e-5);
    }
    std::printf("closed form F_j^a - F_{j-1}^a: OK\n");
}

static void test_direction_and_identity() {
    const std::vector<float> p = { 0.6f, 0.25f, 0.1f, 0.05f };
    auto v = from_probs(p, { 3, 1, 0, 2 });
    llama_token_data_array a = { v.data(), v.size(), -1, false };
    assert(!common_sampling_skew_apply(&a, 0.0f)); // skew 0: untouched
    assert(v[0].id == 3 && std::fabs(v[0].logit - std::log(0.05f)) < 1e-7f);
    auto head = v;
    llama_token_data_array ah = { head.data(), head.size(), -1, false };
    common_sampling_skew_apply(&ah, -1.0f);
    assert(prob_of(head, 0) > 0.6f);  // negative skew: toward the head
    auto tail = v;
    llama_token_data_array at = { tail.data(), tail.size(), -1, false };
    common_sampling_skew_apply(&at, 1.0f);
    assert(prob_of(tail, 0) < 0.6f && prob_of(tail, 2) > 0.1f); // positive skew: toward the tail
    std::printf("direction (skew<0 head, skew>0 tail) and skew=0 identity: OK\n");
}

static void test_order_independent() {
    // the nuance against Aphrodite's skew.py (cumsum in token-id order): the result must not depend on input order
    const std::vector<float> p = { 0.1f, 0.4f, 0.2f, 0.3f };
    auto x = from_probs(p, { 0, 1, 2, 3 });
    auto y = from_probs(p, { 3, 2, 1, 0 });
    llama_token_data_array ax = { x.data(), x.size(), -1, false };
    llama_token_data_array ay = { y.data(), y.size(), -1, false };
    common_sampling_skew_apply(&ax, 0.7f);
    common_sampling_skew_apply(&ay, 0.7f);
    for (llama_token id = 0; id < 4; ++id) {
        assert(std::fabs(prob_of(x, id) - prob_of(y, id)) < 1e-6f);
    }
    std::printf("independent of vocabulary order: OK\n");
}

static void test_matches_exllamav2_draw() {
    // simulate ExLlamaV2's draw: sorted list, u' = u^(exp(-skew)), first j with F_j >= u'
    const std::vector<float> p = { 0.05f, 0.35f, 0.15f, 0.25f, 0.2f };
    const float skew = 0.6f;
    std::vector<int> sorted_ids = { 1, 3, 4, 2, 0 };
    std::vector<double> F;
    double c = 0.0;
    for (int id : sorted_ids) F.push_back(c += p[id]);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> hits(5, 0.0);
    const int N = 400000;
    for (int i = 0; i < N; ++i) {
        const double u = std::pow(U(rng), std::exp(-(double) skew));
        size_t j = 0;
        while (j + 1 < F.size() && F[j] < u) ++j;
        hits[sorted_ids[j]] += 1.0;
    }
    auto v = from_probs(p, { 0, 1, 2, 3, 4 });
    llama_token_data_array a = { v.data(), v.size(), -1, false };
    common_sampling_skew_apply(&a, skew);
    for (llama_token id = 0; id < 5; ++id) {
        const double emp = hits[id] / N;
        assert(std::fabs(emp - prob_of(v, id)) < 0.004); // ~5 sigma at N = 400k
    }
    std::printf("matches the simulated ExLlamaV2 draw (N=%d): OK\n", N);
}

int main() {
    test_closed_form();
    test_direction_and_identity();
    test_order_independent();
    test_matches_exllamav2_draw();
    std::printf("test-sampling-skew: all passed\n");
    return 0;
}
