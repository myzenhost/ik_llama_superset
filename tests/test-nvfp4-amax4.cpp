// NVFP4 encoder scale-code search with the amax/4 candidate (ggml/src/ggml-quants.c, quantize_row_nvfp4_search).
// Row S69-ENGREV-APHRODITE-NVFP4-AMAX4-CANDIDATE-2026-10-27-01. The +-2 search around the nominal UE4M3 code of amax/6 never reaches the
// code of amax/4 (a 1.5x larger scale). Measured on real weights (temp/qwork_qworker-fork10/amax4_sim.txt): -2.3% relative error.
// This test is the unit-level before/after: a block built so the 1.5x scale is EXACT (zero error), plus controls.
// Needs no model: ggml_quantize_chunk + the type's to_float.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

// relative squared error of quantise -> dequantise over `rows` rows of n values
static double rel_err(const std::vector<float> & x, int64_t rows, int64_t n, const float * imatrix = nullptr) {
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, n);
    std::vector<uint8_t> q(row_size * rows);
    ggml_quantize_chunk(GGML_TYPE_NVFP4, x.data(), q.data(), 0, rows, n, imatrix);
    std::vector<float> y(x.size());
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    for (int64_t r = 0; r < rows; ++r) {
        tr->to_float(q.data() + r * row_size, y.data() + r * n, n);
    }
    double num = 0, den = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        num += double(x[i] - y[i]) * double(x[i] - y[i]);
        den += double(x[i]) * double(x[i]);
    }
    return den > 0 ? num / den : 0.0;
}

int main() {
    // 64 values = one NVFP4 block of 4 sub-blocks of 16. Each sub-block: amax 6.0, then 0.75 and 2.25. With the scale 1.5 the E2M1 grid is
    // {0, .75, 1.5, 2.25, 3, 4.5, 6, 9}: 6.0 = 4*1.5, 0.75 = 0.5*1.5, 2.25 = 1.5*1.5, all exact. With the nominal scale 1.0 (grid {0,.5,1,1.5,2,3,4,6})
    // 0.75 and 2.25 are off-grid. Code of 1.5 is 0x3C, three above the nominal 0x38: outside the +-2 window.
    std::vector<float> a(64);
    for (int s = 0; s < 4; ++s) {
        for (int j = 0; j < 16; ++j) {
            a[s * 16 + j] = j == 0 ? 6.0f : (j % 2 ? 0.75f : 2.25f);
        }
    }
    const double e_a = rel_err(a, 1, 64);
    // BEFORE (+-2 search only): the best reachable code leaves a visible error (measured 0.0104 on this block). AFTER: the 1.5x scale is exact.
    CHECK(e_a < 1e-9, "amax/4 candidate must reach the exact 1.5x scale: rel err %.6g", e_a);

    // control 1: a block whose nominal scale is already exact (values on the E2M1 grid) stays exact -- nominal still wins ties
    std::vector<float> b(64);
    const float grid[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    for (int i = 0; i < 64; ++i) {
        b[i] = (i % 2 ? -1.0f : 1.0f) * grid[i % 8];
    }
    CHECK(rel_err(b, 1, 64) < 1e-12, "control: on-grid block must stay exact");

    // control 2: an all-zero block and a tiny-amax block do not crash and stay finite
    std::vector<float> z(64, 0.0f);
    CHECK(rel_err(z, 1, 64) == 0.0, "control: all-zero block");
    std::vector<float> t(64);
    for (int i = 0; i < 64; ++i) t[i] = 1e-9f * (i + 1);
    const double e_t = rel_err(t, 1, 64);
    CHECK(std::isfinite(e_t) && e_t <= 1.0, "control: tiny block finite (%.6g)", e_t);

    // control 3: random gaussian rows -- the candidate only ADDS a choice, so the error is never above the plain nominal reference encoder's
    // (reference = quantize_row_nvfp4_ref via the type traits' from_float_ref)
    std::vector<float> g(64 * 256);
    unsigned s = 12345u;
    for (auto & v : g) {
        float u = 0.0f;
        for (int k = 0; k < 12; ++k) { s = s * 1664525u + 1013904223u; u += (s >> 8) / float(1 << 24); }
        v = u - 6.0f;
    }
    const double e_g = rel_err(g, 256, 64);
    const size_t rs = ggml_row_size(GGML_TYPE_NVFP4, 64);
    std::vector<uint8_t> qr(rs * 256);
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    for (int r = 0; r < 256; ++r) tr->from_float_ref(g.data() + r * 64, qr.data() + r * rs, 64);
    std::vector<float> yr(g.size());
    for (int r = 0; r < 256; ++r) tr->to_float(qr.data() + r * rs, yr.data() + r * 64, 64);
    double num = 0, den = 0;
    for (size_t i = 0; i < g.size(); ++i) { num += double(g[i] - yr[i]) * (g[i] - yr[i]); den += double(g[i]) * g[i]; }
    CHECK(e_g <= num / den, "gaussian rows: search+amax4 error %.6g must not exceed the reference encoder's %.6g", e_g, num / den);

    if (g_fails) {
        std::printf("test-nvfp4-amax4: %d failure(s)\n", g_fails);
        return 1;
    }
    std::printf("test-nvfp4-amax4: all pass (crafted block rel err %.3g, gaussian %.5f)\n", e_a, e_g);
    return 0;
}
