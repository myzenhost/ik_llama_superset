// Default tensor-split normalisation (src/llama-split-points.h).
// The OLD arithmetic (cumulative sum, then divide by the sum) is reproduced here as the fail-before witness: with
// every device reporting 0 free bytes after the margin it yields NaN split points, which then index past the end of
// the device list. The helper must give finite, strictly increasing points ending at 1 for every input, and must be
// byte-identical to the old arithmetic whenever the old arithmetic was well defined.
// Upstream: ggml-org/llama.cpp#24795 / #27647.

#include "../src/llama-split-points.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static std::vector<float> old_normalise(std::vector<float> s) {
    float sum = 0.0f;
    for (auto & x : s) { sum += x; x = sum; }
    for (auto & x : s) { x /= sum; }
    return s;
}

static int n_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++n_fail; } } while (0)

static bool well_formed(const std::vector<float> & s) {
    for (size_t i = 0; i < s.size(); ++i) {
        if (!std::isfinite(s[i]) || s[i] <= 0.0f || s[i] > 1.0f) return false;
        if (i > 0 && !(s[i] > s[i - 1])) return false;
    }
    return s.empty() || s.back() == 1.0f;
}

int main() {
    // witness: the old code on the WDDM-full input produces NaN (this is the crash's first step)
    {
        auto o = old_normalise({0.0f, 0.0f});
        CHECK(std::isnan(o[0]) && std::isnan(o[1]), "old arithmetic no longer yields NaN on all-zero weights (%f %f)", o[0], o[1]);
    }
    // all-zero -> even split, fallback reported
    for (int n : {2, 3, 4, 8}) {
        std::vector<float> s(n, 0.0f);
        const bool used = llama_split_points_from_weights(s);
        CHECK(!used, "n=%d all-zero: fallback not reported", n);
        CHECK(well_formed(s), "n=%d all-zero: split points not well formed", n);
        for (int i = 0; i < n; ++i) {
            CHECK(std::fabs(s[i] - float(i + 1) / float(n)) < 1e-7f, "n=%d all-zero: point %d = %f", n, i, s[i]);
        }
    }
    // valid weights: identical to the old arithmetic, bit for bit
    const std::vector<std::vector<float>> valid = {{1.0f, 3.0f}, {24576.0f, 8192.0f, 0.0f}, {0.0f, 5.0f}, {7.0f}, {1e9f, 1e9f, 2e9f}};
    for (const auto & w : valid) {
        auto s = w;
        const bool used = llama_split_points_from_weights(s);
        const auto o = old_normalise(w);
        CHECK(used, "valid weights took the fallback");
        CHECK(s == o, "valid weights differ from the old arithmetic");
        CHECK(s.back() == 1.0f, "last point %f != 1", s.back());
    }
    // non-finite sum -> fallback
    {
        std::vector<float> s = {INFINITY, 1.0f};
        CHECK(!llama_split_points_from_weights(s) && well_formed(s), "infinite weight not handled");
    }
    // empty is a no-op
    {
        std::vector<float> s;
        CHECK(llama_split_points_from_weights(s) && s.empty(), "empty input changed");
    }
    std::printf("test-split-points: %s (%d failure(s))\n", n_fail ? "FAIL" : "PASS", n_fail);
    return n_fail ? 1 : 0;
}
