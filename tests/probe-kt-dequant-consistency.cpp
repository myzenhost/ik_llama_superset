// Standalone probe (links the static ggml archives, same build line as tests/probe-cpu-op-reference.cpp).
// Expected after fork 2026-09-26: every ratio 1.00000.
// GATE (2026-10-01, ctest probe-kt-dequant-consistency): the probe used to print the ratios and return 0 whatever they were,
// so it could not fail. It now returns non-zero when a ratio leaves 1 +- 1e-3 or a type is missing from this build, and it
// first proves its own comparison can say no (a ratio of exactly 2 must be rejected).
// Does to_float agree with vec_dot for the KT types? ratio = vec_dot / dot(dequant(w), dequant(act)).
// A consistent pair gives ~1.000; a missing scale multiplier in one path shows up as that multiplier.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static bool ratio_ok(double r) { return std::fabs(r - 1.0) <= 1e-3; }

int main() {
    ggml_cpu_init();
    int checked = 0, fails = 0, controls = 0;
    if (ratio_ok(2.0) || !ratio_ok(1.0)) { printf("FAIL: ratio_ok cannot tell 2.0 from 1.0\n"); return 1; }
    ++controls;
    const int n = 256 * 16;
    std::vector<float> w(n), a(n);
    std::mt19937 g(7); std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : w) v = nd(g);
    { std::uniform_int_distribution<int> ui(-127, 127); for (int i = 0; i < n; i++) a[i] = (i % 32 == 0) ? 127.f : (float)ui(g); }
    const char * names[] = {"iq1_kt", "iq2_kt", "iq3_kt", "iq4_kt", "iq4_ks"};  // iq4_ks = control, no KT scale
    for (auto nm : names) {
        for (int t = 0; t < GGML_TYPE_COUNT; t++) {
            const char * tn = ggml_type_name((ggml_type)t);
            if (!tn || strcmp(tn, nm)) continue;
            ggml_quantize_init((ggml_type)t);
            const auto * tr = ggml_get_type_traits((ggml_type)t);
            const auto * tc = ggml_get_type_traits_cpu((ggml_type)t);
            const ggml_type vt = tc->vec_dot_type;
            std::vector<uint8_t> qw(ggml_row_size((ggml_type)t, n) + 64), qa(ggml_row_size(vt, n) + 64);
            tc->from_float(w.data(), qw.data(), n);
            ggml_get_type_traits_cpu(vt)->from_float(a.data(), qa.data(), n);
            float vd = 0; tc->vec_dot(n, &vd, 0, qw.data(), 0, qa.data(), 0, 1);
            std::vector<float> dw(n), da(n);
            tr->to_float(qw.data(), dw.data(), n);
            const auto * tv = ggml_get_type_traits(vt);
            if (tv->to_float) tv->to_float(qa.data(), da.data(), n); else da = a;
            double ref = 0; for (int i = 0; i < n; i++) ref += (double)dw[i] * da[i];
            printf("%-7s vec_dot/dot(dequant) = %.5f   (act type %s%s)\n", nm, vd / ref, ggml_type_name(vt), tv->to_float ? "" : ", act not dequantisable -> used f32");
            ++checked;
            if (!ratio_ok(vd / ref)) { ++fails; printf("FAIL: %s ratio %.5f is not 1 +- 1e-3\n", nm, vd / ref); }
        }
    }
    // five types are named above; a type this build does not know would have been silently skipped before
    if (checked != 5) { ++fails; printf("FAIL: expected 5 types checked, got %d\n", checked); }
    printf("%d checked, %d failed (%d self-control)\n", checked, fails, controls);
    return fails ? 1 : 0;
}
