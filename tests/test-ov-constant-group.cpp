// A Q4_K/Q5_K/Q4_1/Q5_1 group with decoded scale 0 and a nonzero min is the constant `min` (ggml y = q*d + m).
// The OpenVINO zero-point form (q - zp) * scale gives 0 for it. ggml_ov_make_constant_group re-encodes it as
// scale' = -bias, zp = 1, q = 0 so (0 - 1) * -bias == bias. Pairs:
//   naive affine (zp = round(-bias/scale), scale 0)  -> 0          (the defect, shown as a control)
//   re-encoded group                                  -> bias, for every q that was in the group
//   a normal group (scale != 0)                       -> not touched
//   scale 0 and bias 0                                -> not touched (already exact: 0)

#include "ggml-quants-constant-group.h"

#include <cmath>
#include <cstdio>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: %s -- ", #cond); printf(__VA_ARGS__); printf("\n"); g_fail++; } else { printf("  ok:   %s\n", #cond); } } while (0)

int main() {
    const float bias = -0.375f;

    printf("detector\n");
    CHECK(ggml_ov_constant_group(0.0f, bias), "scale 0, bias != 0 is a constant group");
    CHECK(!ggml_ov_constant_group(0.5f, bias), "scale != 0 is a normal group");
    CHECK(!ggml_ov_constant_group(0.0f, 0.0f), "scale 0 and bias 0 is already exact");

    printf("control: the naive affine form loses the constant\n");
    {
        volatile float scale_v = 0.0f;  // volatile: the compiler must not fold the guarded division by zero
        const float scale = scale_v;
        const uint8_t zp = (scale != 0.0f) ? (uint8_t) std::lround(-bias / scale) : 0;  // what the old code stored
        float worst = 0.0f;
        for (int q = 0; q < 32; ++q) worst = std::fmax(worst, std::fabs(ggml_ov_affine_dequant((uint8_t) q, zp, scale) - bias));
        CHECK(worst > 0.3f, "old encoding is off by %f from the constant", worst);
    }

    printf("re-encoded group\n");
    {
        float scale = 0.0f;
        uint8_t zp = 0;
        uint8_t q[32];
        for (int i = 0; i < 32; ++i) q[i] = (uint8_t) (i * 7 & 31);  // arbitrary quants, must be ignored
        ggml_ov_make_constant_group(bias, scale, zp, q, 32);
        bool all = true;
        for (int i = 0; i < 32; ++i) all = all && ggml_ov_affine_dequant(q[i], zp, scale) == bias;
        CHECK(all, "every weight dequantises to the constant %f", bias);
        CHECK(zp == 1 && scale == -bias, "zp %d scale %f", (int) zp, scale);
    }

    printf("positive bias\n");
    {
        float scale = 0.0f;
        uint8_t zp = 0;
        uint8_t q[16] = {};
        ggml_ov_make_constant_group(2.5f, scale, zp, q, 16);
        CHECK(ggml_ov_affine_dequant(q[3], zp, scale) == 2.5f, "constant +2.5");
    }

    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
