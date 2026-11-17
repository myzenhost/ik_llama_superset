// LoRA delta merging strategies (common/lora-merge.h): linear, TIES, DARE-linear, DARE-TIES.   GATE-CLAIM-OK: documentation only.
// Row S69-LORAX-PICK-REQUEST-TIME-LORA-MERGE-TIES-DARE-2026-09-29-01.  CPU only, no model.
//
// FAIL-BEFORE: lora_merge_deltas did not exist, so this file could not be compiled or linked at the parent commit.  The expected values are derived by
// hand below (and by an independent sort-based reference for the pruning), not by running the implementation.
//   T1  TIES pruning keeps exactly floor(density * n) entries and they are the largest magnitudes (independent reference: sort by |x|, lowest index first on ties)
//   T2  sign election: values (+3, -1, -1) at one position: "total" elects + (sum +1) and returns 3/1 = 3; "frequency" elects - (sign sum -1) and returns (-1 + -1)/2 = -1
//   T3  weights are applied AFTER the election and the divisor is the COUNT of agreeing adapters: (+2, +4) with weights (0.5, 0.25) -> (1 + 1)/2 = 1.0, not
//       2/(0.75)
//   T4  DARE rescale: the mean of a pruned constant tensor is the constant within 3 sigma (a version that discards the rescale, as LoRAX's does, gives the
//       density instead; the test computes that value as its negative control and requires the two to be far apart)
//   T5  linear merge equals the weighted sum, to 1e-6; density 1 TIES with one adapter returns the input; invalid density is refused and leaves `out` alone
//   T6  determinism: same seed same DARE result, a different seed a different one
//   T7  equal magnitudes at the cut: lowest index kept
#include "lora-merge.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static bool merge1(lora_merge_strategy s, float density, const std::vector<std::vector<float>> & d, const std::vector<float> & w,
                   std::vector<float> & out, lora_sign_method sm = LORA_SIGN_TOTAL, uint64_t seed = 0, std::string * err = nullptr) {
    std::vector<const float *> ptr;
    for (const auto & v : d) {
        ptr.push_back(v.data());
    }
    lora_merge_params p;
    p.strategy = s; p.density = density; p.sign_method = sm; p.seed = seed;
    out.assign(d[0].size(), -12345.0f);
    return lora_merge_deltas(p, ptr.data(), w.empty() ? nullptr : w.data(), (int) d.size(), d[0].size(), out.data(), err);
}

int main() {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    {   // T1
        const size_t n = 1000;
        std::vector<float> x(n);
        for (auto & v : x) { v = nd(rng); }
        for (float dens : { 0.05f, 0.2f, 0.333f, 0.999f }) {
            std::vector<float> out;
            CHECK(merge1(LORA_MERGE_TIES, dens, { x }, { 1.0f }, out));
            const size_t k = (size_t) std::floor((double) dens * (double) n);
            size_t nz = 0;
            for (float v : out) { nz += v != 0.0f; }
            std::vector<size_t> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return std::fabs(x[a]) > std::fabs(x[b]); });
            std::vector<float> ref(n, 0.0f);
            for (size_t i = 0; i < k; ++i) { ref[idx[i]] = x[idx[i]]; }
            CHECK(nz == k);
            CHECK(out == ref);
            printf("T1 density %.3f: kept %zu of %zu (floor = %zu)\n", dens, nz, n, k);
        }
    }
    {   // T2
        std::vector<float> a = { 3.0f }, b = { -1.0f }, c = { -1.0f }, out;
        CHECK(merge1(LORA_MERGE_TIES, 1.0f, { a, b, c }, { 1.0f, 1.0f, 1.0f }, out, LORA_SIGN_TOTAL));
        CHECK(out[0] == 3.0f);
        CHECK(merge1(LORA_MERGE_TIES, 1.0f, { a, b, c }, { 1.0f, 1.0f, 1.0f }, out, LORA_SIGN_FREQUENCY));
        CHECK(out[0] == -1.0f);
        // a sum of exactly zero elects +
        std::vector<float> p = { 2.0f }, m = { -2.0f };
        CHECK(merge1(LORA_MERGE_TIES, 1.0f, { p, m }, { 1.0f, 1.0f }, out, LORA_SIGN_TOTAL));
        CHECK(out[0] == 2.0f);
        printf("T2 total -> 3, frequency -> -1, tie of signs -> +\n");
    }
    {   // T3
        std::vector<float> a = { 2.0f }, b = { 4.0f }, out;
        CHECK(merge1(LORA_MERGE_TIES, 1.0f, { a, b }, { 0.5f, 0.25f }, out));
        CHECK(std::fabs(out[0] - 1.0f) < 1e-6f);
        printf("T3 weighted disjoint merge = %.6f (expected 1)\n", out[0]);
    }
    {   // T4
        const size_t n = 200000;
        const float dens = 0.3f;
        std::vector<float> ones(n, 1.0f), out;
        CHECK(merge1(LORA_MERGE_DARE_LINEAR, dens, { ones }, { 1.0f }, out, LORA_SIGN_TOTAL, 11));
        double mean = 0.0;
        for (float v : out) { mean += v; }
        mean /= (double) n;
        const double sigma = std::sqrt((double) dens * (1.0 - dens) / (double) n) / (double) dens;   // std of the mean of the rescaled Bernoulli draws
        CHECK(std::fabs(mean - 1.0) < 3.0 * sigma);
        const double no_rescale_mean = dens;                                                         // what a discarded rescale would give
        CHECK(std::fabs(no_rescale_mean - 1.0) > 100.0 * sigma);                                     // the test can tell the two apart
        CHECK(merge1(LORA_MERGE_DARE_TIES, dens, { ones }, { 1.0f }, out, LORA_SIGN_TOTAL, 11));
        double mean2 = 0.0;
        for (float v : out) { mean2 += v; }
        mean2 /= (double) n;
        CHECK(std::fabs(mean2 - 1.0) < 3.0 * sigma);
        printf("T4 DARE mean %.5f / %.5f (sigma of the mean %.5f), a discarded rescale would be %.2f\n", mean, mean2, sigma, no_rescale_mean);
    }
    {   // T5
        const size_t n = 257;
        std::vector<float> a(n), b(n), out;
        for (auto & v : a) { v = nd(rng); }
        for (auto & v : b) { v = nd(rng); }
        CHECK(merge1(LORA_MERGE_LINEAR, 1.0f, { a, b }, { 0.7f, -0.2f }, out));
        double worst = 0.0;
        for (size_t i = 0; i < n; ++i) { worst = std::max(worst, (double) std::fabs(out[i] - (0.7f * a[i] - 0.2f * b[i]))); }
        CHECK(worst < 1e-6);
        CHECK(merge1(LORA_MERGE_TIES, 1.0f, { a }, {}, out));
        CHECK(out == a);
        std::string err;
        std::vector<float> keep = { 7.0f, 7.0f };
        std::vector<const float *> ptr = { a.data() };
        lora_merge_params bad;
        bad.strategy = LORA_MERGE_TIES; bad.density = 1.5f;
        CHECK(!lora_merge_deltas(bad, ptr.data(), nullptr, 1, 2, keep.data(), &err));
        CHECK(!err.empty() && keep[0] == 7.0f && keep[1] == 7.0f);
        bad.density = -0.1f;
        CHECK(!lora_merge_deltas(bad, ptr.data(), nullptr, 1, 2, keep.data(), &err));
        CHECK(!lora_merge_deltas(bad, ptr.data(), nullptr, 0, 2, keep.data(), &err));
        lora_merge_strategy s;
        CHECK(lora_merge_strategy_from_name("DARE_TIES", s) && s == LORA_MERGE_DARE_TIES);
        CHECK(!lora_merge_strategy_from_name("slerp", s));
        printf("T5 linear max |err| %.2e; invalid requests refused: %s\n", worst, err.c_str());
    }
    {   // T6
        std::vector<float> a(5000), out1, out2, out3;
        for (auto & v : a) { v = nd(rng); }
        CHECK(merge1(LORA_MERGE_DARE_LINEAR, 0.5f, { a }, { 1.0f }, out1, LORA_SIGN_TOTAL, 3));
        CHECK(merge1(LORA_MERGE_DARE_LINEAR, 0.5f, { a }, { 1.0f }, out2, LORA_SIGN_TOTAL, 3));
        CHECK(merge1(LORA_MERGE_DARE_LINEAR, 0.5f, { a }, { 1.0f }, out3, LORA_SIGN_TOTAL, 4));
        CHECK(out1 == out2);
        CHECK(out1 != out3);
        printf("T6 determinism ok\n");
    }
    {   // T7
        std::vector<float> a = { 1.0f, -1.0f, 1.0f, -1.0f }, out;
        CHECK(merge1(LORA_MERGE_TIES, 0.5f, { a }, { 1.0f }, out));
        CHECK(out == (std::vector<float>{ 1.0f, -1.0f, 0.0f, 0.0f }));
        printf("T7 ties at the cut keep the lowest index\n");
    }

    if (n_fail) {
        fprintf(stderr, "test-lora-merge: %d failure(s)\n", n_fail);
        return 1;
    }
    printf("test-lora-merge: all checks passed\n");
    return 0;
}
