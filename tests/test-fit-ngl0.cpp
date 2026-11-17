// common_fit_model_per_layer (common/fit.cpp): the per-layer model share the multi-device context reduction subtracts.
// Row S69-ENGREV-ACCELERATE-FIT-NGL0-DIV-BY-ZERO-MULTIGPU-2026-10-27-01. With -ngl 0 on 2+ devices the divisor
// min(n_gpu_layers, hp_ngl) was 0: an integer division by zero BEFORE the "n_gpu_layers already set by user" abort.
// Needs no GPU: the arithmetic is a pure function.
#include "fit.h"

#include <cstdint>
#include <cstdio>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    // the defect: -ngl 0 must not divide by zero (before: SIGFPE / SEH divide error; after: 0, so nothing is subtracted)
    CHECK(common_fit_model_per_layer(1000, 0, 32) == 0, "ngl 0 must give 0");
    CHECK(common_fit_model_per_layer(1000, 8, 0)  == 0, "hp_ngl 0 must give 0");
    CHECK(common_fit_model_per_layer(0,    0, 0)  == 0, "all zero must give 0");

    // controls: every other input keeps the exact old value sum / min(ngl, hp_ngl)
    CHECK(common_fit_model_per_layer(1000, 8,  32) == 125, "ngl 8 of 32");
    CHECK(common_fit_model_per_layer(1000, 32, 32) == 31,  "ngl 32 of 32 (1000/32 = 31)");
    CHECK(common_fit_model_per_layer(1000, 99, 32) == 31,  "ngl above the layer count clamps to the layer count");
    CHECK(common_fit_model_per_layer(1000, -1, 32) == 31,  "ngl -1 (default: all layers) clamps to the layer count");
    CHECK(common_fit_model_per_layer(64ll << 30, 64, 64) == (1ll << 30), "64 GiB over 64 layers");

    if (g_fails) {
        std::printf("test-fit-ngl0: %d failure(s)\n", g_fails);
        return 1;
    }
    std::printf("test-fit-ngl0: all pass\n");
    return 0;
}
