// SOFTPLUS numerics and the inplace API.
//
// 1. ggml_softplus_inplace must build GGML_UNARY_OP_SOFTPLUS. It built SQRT_SOFTPLUS, so an out-of-tree caller
//    silently got sqrt(softplus(x)) (row S69-FORK-GGML-SOFTPLUS-INPLACE-BUILDS-SQRT-SOFTPLUS-2026-10-19-01).
// 2. ggml_compute_softplus_f32 (ggml-impl.h, the helper the CPU SSM scan and ggml.c use) computed logf(1 + expf(x)):
//    exactly 0 for x below about -17 and short of relative precision from x ~ -5, while softplus(x) ~ e^x > 0
//    (row S69-FORK-SOFTPLUS-LOGF1P-UNDERFLOW-OTHER-COPIES-2026-10-19-01).
// CONTROL: the plain CPU softplus op (already log1pf) must pass the same precision bound; if it fails, the reference
//    is wrong and the verdicts on the other two mean nothing.
#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h"  // internal header: ggml_compute_softplus_f32 is a static inline helper there

#include <cmath>
#include <cstdio>
#include <vector>

static const int N = 61;  // x = -30 .. 30 step 1

static double ref_softplus(double x) {
    return x > 20.0 ? x : std::log1p(std::exp(x));
}

static double worst_rel(const std::vector<float> & got, const std::vector<float> & xs) {
    double worst = 0.0;
    for (size_t i = 0; i < xs.size(); i++) {
        const double r = ref_softplus(xs[i]);
        const double e = std::fabs((double) got[i] - r) / r;
        if (!(e <= worst)) {
            worst = e;  // also captures NaN
        }
    }
    return worst;
}

static std::vector<float> make_xs() {
    std::vector<float> xs(N);
    for (int i = 0; i < N; i++) {
        xs[i] = (float) (i - 30);
    }
    return xs;
}

// kind 0: ggml_softplus, 1: ggml_softplus_inplace
static std::vector<float> run_op(int kind, const std::vector<float> & xs) {
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) xs.size());
    ggml_set_name(x, "x");
    for (size_t i = 0; i < xs.size(); i++) {
        ((float *) x->data)[i] = xs[i];
    }
    ggml_tensor * y = kind == 0 ? ggml_softplus(ctx, x) : ggml_softplus_inplace(ctx, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((float *) y->data, (float *) y->data + xs.size());
    ggml_free(ctx);
    return out;
}

static int g_fail = 0;

static void check(const char * what, bool ok) {
    printf("  %-70s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) {
        g_fail++;
    }
}

int main() {
    const std::vector<float> xs = make_xs();

    // CONTROL
    const double ctl = worst_rel(run_op(0, xs), xs);
    printf("control: ggml_softplus worst rel err %.3g\n", ctl);
    check("control: CPU softplus op within 1e-5 of exact log1p(exp(x))", ctl < 1e-5);

    // 1. inplace op equals the plain op
    const std::vector<float> a = run_op(0, xs);
    const std::vector<float> b = run_op(1, xs);
    double maxd = 0.0;
    for (size_t i = 0; i < xs.size(); i++) {
        maxd = std::fmax(maxd, std::fabs((double) a[i] - (double) b[i]));
    }
    printf("inplace vs plain: max abs diff %.3g\n", maxd);
    check("ggml_softplus_inplace == ggml_softplus", maxd == 0.0);

    // 2. the shared scalar helper
    std::vector<float> h(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        h[i] = ggml_compute_softplus_f32(xs[i]);
    }
    const double wh = worst_rel(h, xs);
    printf("ggml_compute_softplus_f32 worst rel err %.3g\n", wh);
    check("ggml_compute_softplus_f32 within 1e-5 of exact, x in [-30, 30]", wh < 1e-5);
    check("ggml_compute_softplus_f32(-25) > 0", ggml_compute_softplus_f32(-25.0f) > 0.0f);

    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
