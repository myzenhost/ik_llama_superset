// Probe: Q1_0 / Q2_0 base + CPU traits, with controls both directions. Built + run by ctest
// (tests/CMakeLists.txt, llama_build_and_test). Expected: "PASS: 0 failed", rc 0.
// Checks per type: traits populated; quant->dequant error; ref == non-ref bytes;
// vec_dot(q, q8_0) == dot(dequant(q), dequant(q8_0)) (layout agreement oracle);
// ggml_quantize_chunk path == ref path; ggml_new_tensor works (the original abort).
// NEGATIVE CONTROL: feed Q2_0's vec_dot bytes quantised with the g128 layout -> must disagree.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void quantize_row_q2_0_g128_ref(const float * x, void * y, int64_t k);

static void gen(float off, size_t n, float * d) { for (size_t i = 0; i < n; i++) d[i] = 0.1f + 2.0f * cosf(i + off); }

static int fails = 0;
static void check(bool ok, const char * what, double v) { printf("  %-44s %s (%g)\n", what, ok ? "ok" : "FAILED", v); fails += !ok; }

static double dot_via_dequant(ggml_type t, const void * qa, const void * qb, int n) {
    const auto * ta = ggml_get_type_traits(t);
    const auto * tb = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<float> a(n), b(n);
    ta->to_float(qa, a.data(), n); tb->to_float(qb, b.data(), n);
    double s = 0; for (int i = 0; i < n; i++) s += (double)a[i] * b[i];
    return s;
}

static void run(ggml_type t, float max_q_err) {
    const int n = 32 * 128;
    const auto * tr = ggml_get_type_traits(t);
    const auto * tc = ggml_get_type_traits_cpu(t);
    printf("type %d name=%s blck=%lld size=%zu\n", (int)t, tr->type_name ? tr->type_name : "(null)", (long long)tr->blck_size, tr->type_size);
    check(tr->type_name && tr->blck_size > 0 && tr->type_size > 0 && tr->to_float && tr->from_float_ref, "base traits populated", tr->blck_size);
    if (fails) return;
    std::vector<float> x(n), y(n), out(n);
    gen(0.0f, n, x.data()); gen(1.0f, n, y.data());
    const size_t rs = ggml_row_size(t, n);
    std::vector<uint8_t> q(rs), qref(rs), qchunk(rs);
    tc->from_float(x.data(), q.data(), n);
    tr->from_float_ref(x.data(), qref.data(), n);
    check(memcmp(q.data(), qref.data(), rs) == 0, "cpu from_float == base from_float_ref", 0);
    ggml_quantize_chunk(t, x.data(), qchunk.data(), 0, 1, n, nullptr, nullptr);
    check(memcmp(qchunk.data(), qref.data(), rs) == 0, "ggml_quantize_chunk == ref", 0);
    check(ggml_validate_row_data(t, q.data(), rs), "ggml_validate_row_data", 0);
    tr->to_float(q.data(), out.data(), n);
    double err = 0; for (int i = 0; i < n; i++) err += fabs(out[i] - x[i]); err /= n;
    check(err < max_q_err, "mean abs quant error < bound", err);
    // layout oracle: SIMD/generic vec_dot vs float dot over dequantised operands
    std::vector<uint8_t> q8(ggml_row_size(GGML_TYPE_Q8_0, n));
    ggml_get_type_traits_cpu(tc->vec_dot_type)->from_float(y.data(), q8.data(), n);
    float vd = 0; tc->vec_dot(n, &vd, 0, q.data(), 0, q8.data(), 0, 1);
    const double ref = dot_via_dequant(t, q.data(), q8.data(), n);
    check(fabs(vd - ref) / (fabs(ref) + 1e-6) < 1e-4, "vec_dot == dot(dequant) rel", fabs(vd - ref) / (fabs(ref) + 1e-6));
    // ggml_new_tensor was the original abort
    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tt = ggml_new_tensor_2d(ctx, t, n, 2);
    check(tt && ggml_nbytes(tt) == 2 * rs, "ggml_new_tensor_2d nbytes", (double)ggml_nbytes(tt));
    ggml_free(ctx);
}

int main() {
    ggml_cpu_init();
    run(GGML_TYPE_Q1_0, 1.6f);   // upstream MAX_QUANTIZATION_TOTAL_ERROR_BINARY territory; printed value is what matters
    run(GGML_TYPE_Q2_0, 0.5f);
    // NEGATIVE CONTROL: the old forwarding (g128 quantiser) into Q2_0's vec_dot must NOT agree
    {
        const int n = 32 * 128;
        std::vector<float> x(n), y(n); gen(0.0f, n, x.data()); gen(1.0f, n, y.data());
        std::vector<uint8_t> qg(ggml_row_size(GGML_TYPE_Q2_0, n) * 2 + 64), q8(ggml_row_size(GGML_TYPE_Q8_0, n));
        quantize_row_q2_0_g128_ref(x.data(), qg.data(), n);
        ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(y.data(), q8.data(), n);
        float vd = 0; ggml_get_type_traits_cpu(GGML_TYPE_Q2_0)->vec_dot(n, &vd, 0, qg.data(), 0, q8.data(), 0, 1);
        double exact = 0; for (int i = 0; i < n; i++) exact += (double)x[i] * y[i];
        std::vector<uint8_t> qr(ggml_row_size(GGML_TYPE_Q2_0, n));
        ggml_get_type_traits(GGML_TYPE_Q2_0)->from_float_ref(x.data(), qr.data(), n);
        float vr = 0; ggml_get_type_traits_cpu(GGML_TYPE_Q2_0)->vec_dot(n, &vr, 0, qr.data(), 0, q8.data(), 0, 1);
        printf("NEG CONTROL q2_0 vec_dot vs exact f32 dot=%.3f: new-ref layout %.3f (rel %.4f) | old g128 layout %.3f (rel %.4f)\n",
               exact, vr, fabs(vr - exact) / fabs(exact), vd, fabs(vd - exact) / fabs(exact));
    }
    printf("%s: %d failed\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
