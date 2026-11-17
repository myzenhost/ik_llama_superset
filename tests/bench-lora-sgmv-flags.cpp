// Standalone timing of the SGMV per-token loop vs the grouped loop (common/lora-sgmv.h) under different compiler flags. Same fixture as G4 of
// tests/test-lora-sgmv-grouped.cpp when run with no arguments; arguments T NA R D change the shape. No ggml dependency, so any flag set can be
// compiled in a second: scripts/bench-lora-sgmv-flags.sh builds the flag matrix and runs it. Not a registered test (it asserts nothing about speed).
#include "lora-sgmv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    std::mt19937 rng(1234);
    const int T  = argc > 1 ? atoi(argv[1]) : 64;
    const int NA = argc > 2 ? atoi(argv[2]) : 4;
    const int R  = argc > 3 ? atoi(argv[3]) : 16;
    const int D  = argc > 4 ? atoi(argv[4]) : 2048;
    const int REPS = argc > 1 ? 15 : 30;
    std::normal_distribution<float> nd(0.0f, 0.3f);
    std::vector<std::vector<float>> A(NA), B(NA);
    std::vector<lora_sgmv_adapter> ad(NA);
    for (int a = 0; a < NA; ++a) {
        A[a].resize((size_t) R * D);
        B[a].resize((size_t) D * R);
        for (auto & v : A[a]) v = nd(rng);
        for (auto & v : B[a]) v = nd(rng);
        ad[a] = { A[a].data(), B[a].data(), R, 0.5f + 0.25f * a };
    }
    std::vector<int> ids(T);
    for (int t = 0; t < T; ++t) ids[t] = (t * 5 + 1) % NA;   // fully interleaved: worst case for contiguous runs
    std::vector<float> x((size_t) T * D), y((size_t) T * D, 0.0f);
    std::normal_distribution<float> n1(0, 1);
    for (auto & v : x) v = n1(rng);
    auto time_us = [&](auto fn) {
        std::vector<double> s;
        for (int r = 0; r < REPS + 3; ++r) {
            std::fill(y.begin(), y.end(), 0.0f);
            auto t0 = std::chrono::steady_clock::now();
            fn();
            auto t1 = std::chrono::steady_clock::now();
            if (r >= 3) s.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(s.begin(), s.end());
        return s[s.size() / 2];
    };
    const double tn = time_us([&] { lora_sgmv_apply_naive(x.data(), y.data(), ids.data(), T, D, D, ad.data(), NA); });
    std::vector<float> yn = y;
    lora_sgmv_plan plan = lora_sgmv_group(ids.data(), T, ad.data(), NA);
    const double tg = time_us([&] { lora_sgmv_apply_grouped(x.data(), y.data(), plan, D, D, ad.data()); });
    double num = 0, den = 0;
    for (size_t i = 0; i < y.size(); ++i) { const double d = (double) y[i] - yn[i]; num += d * d; den += (double) yn[i] * yn[i]; }
    printf("T=%d NA=%d R=%d D=%d  per-token %.0f us  grouped %.0f us  ratio %.3f  rel err %.1e\n", T, NA, R, D, tn, tg, tn / tg, std::sqrt(num / den));
    return 0;
}
