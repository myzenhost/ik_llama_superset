// vLLM-engine merge: CPU sampler edge cases (no model). Each case fails on the pre-fix code.
//  1. legacy top_n_sigma with all-negative logits must keep the top candidate(s) (max was seeded with 0 instead of the first logit).
#undef NDEBUG
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static void test_top_n_sigma_negative_logits() {
    std::vector<llama_token_data> d;
    const float logits[] = {-10.f, -11.f, -12.f, -13.f};
    for (int i = 0; i < 4; ++i) d.push_back({i, logits[i], 0.f});
    llama_token_data_array arr = {d.data(), d.size(), -1, false};
    llama_sample_top_n_sigma(nullptr, &arr, 1.0f);
    CHECK(arr.size >= 1);
    CHECK(arr.size < 4);              // sigma ~1.12: -12 and -13 are more than 1 sigma below -10
    CHECK(arr.data[0].id == 0);       // the best candidate survives
    printf("top_n_sigma negative logits: kept %zu\n", arr.size);
}

int main() {
    // The sampler times itself with ggml_time_us(). On Windows that divides by a timer
    // frequency set only by ggml_time_init(), so without this the test dies with an
    // integer divide by zero (ctest: "Exception: Numerical", MinGW build 2026-10-11).
    llama_backend_init();
    test_top_n_sigma_negative_logits();
    llama_backend_free();
    printf(n_fail ? "FAILED %d\n" : "OK\n", n_fail);
    return n_fail ? 1 : 0;
}
