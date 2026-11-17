// common_count_nonfinite_logits: NaN and +inf are counted (the model produced garbage); -inf and finite values are
// not (-inf is how masks and bans are expressed). The server refuses a token when the raw row has any
// (tools/server/server-context.cpp, row S69-mistralrs-PICK-SAMPLER-FAIL-LOUD-NONFINITE-LOGITS-2026-09-28-01).

#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static int g_fail = 0;

static void expect(const char * what, int32_t got, int32_t want) {
    const bool ok = got == want;
    printf("  %-4s %-44s got %d want %d\n", ok ? "ok" : "FAIL", what, got, want);
    g_fail += !ok;
}

int main() {
    const float nan  = std::numeric_limits<float>::quiet_NaN();
    const float pinf = std::numeric_limits<float>::infinity();
    const float ninf = -std::numeric_limits<float>::infinity();

    std::vector<float> clean = { 0.0f, -3.5f, 12.25f, -1e30f, 1e30f };
    expect("finite row", common_count_nonfinite_logits(clean.data(), (int32_t) clean.size()), 0);

    std::vector<float> masked = { 1.0f, ninf, ninf, 2.0f };
    expect("-inf masks are not counted", common_count_nonfinite_logits(masked.data(), (int32_t) masked.size()), 0);

    std::vector<float> one_nan = { 1.0f, nan, 2.0f };
    expect("one NaN", common_count_nonfinite_logits(one_nan.data(), (int32_t) one_nan.size()), 1);

    std::vector<float> mixed = { nan, pinf, ninf, 0.5f, nan, pinf };
    expect("NaN + +inf counted, -inf not", common_count_nonfinite_logits(mixed.data(), (int32_t) mixed.size()), 4);

    std::vector<float> all_nan(1000, nan);
    expect("all NaN", common_count_nonfinite_logits(all_nan.data(), (int32_t) all_nan.size()), 1000);

    expect("empty row", common_count_nonfinite_logits(nullptr, 0), 0);
    expect("accessor on a null sampler", common_sampler_n_nonfinite_last(nullptr), 0);

    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
