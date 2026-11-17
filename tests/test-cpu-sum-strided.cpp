// CPU GGML_OP_SUM over views whose rows are strided (nb[0] != element size).
//
// The CPU kernels handed each row to ggml_vec_sum_*_ggf, which reads ne00 CONSECUTIVE elements, so a
// transposed or permuted input was summed over overlapping windows of the wrong memory. Only an
// assert (compiled out in release) guarded it. test-backend-ops caught it as a CUDA "failure" on
// SUM(type=f32,ne=[33,256,1,1],permute=[1,0,2,3]): the CPU backend is the reference there, and the
// CUDA kernel, which sums the whole contiguous allocation, was the one with the right answer.
//
// Each case sums a view and compares it to a double-precision sum of the same elements read back
// through the view's own strides. The values are a ramp over the allocation, so summing the wrong
// elements cannot land near the right total by accident (a near-constant fill hides the bug: the
// overlapping windows then average to the same value).
//
// CONTROL: the plain contiguous case and a row-contiguous permute must pass with or without the fix;
// if either fails the comparison itself is wrong and the strided verdicts mean nothing.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <string>

static float read_elem(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * p = (const char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    switch (t->type) {
        case GGML_TYPE_F32:  return *(const float *) p;
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
        default:             return NAN;
    }
}

static void fill_ramp(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        const float v = (float) i / 1024.0f;
        switch (t->type) {
            case GGML_TYPE_F32:  ((float *) t->data)[i] = v; break;
            case GGML_TYPE_F16:  ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(v); break;
            case GGML_TYPE_BF16: ((ggml_bf16_t *) t->data)[i] = ggml_fp32_to_bf16(v); break;
            default: break;
        }
    }
}

static float out_value(const ggml_tensor * t) {
    return read_elem(t, 0, 0, 0, 0);
}

// view: 0 = as allocated, 1 = permute(1,0,2,3), 2 = transpose, 3 = permute(0,2,1,3) (rows still contiguous)
static bool run_case(ggml_type type, const int64_t ne[4], int view, bool is_control) {
    ggml_init_params ip = { 64u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne);
    fill_ramp(a);
    ggml_tensor * v = a;
    const char * vname = "contiguous";
    if (view == 1) { v = ggml_permute(ctx, a, 1, 0, 2, 3); vname = "permute(1,0,2,3)"; }
    if (view == 2) { v = ggml_transpose(ctx, a);           vname = "transpose"; }
    if (view == 3) { v = ggml_permute(ctx, a, 0, 2, 1, 3); vname = "permute(0,2,1,3)"; }

    ggml_tensor * out = ggml_sum(ctx, v);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    double ref = 0.0;
    for (int64_t i3 = 0; i3 < v->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < v->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < v->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < v->ne[0]; ++i0)
                    ref += read_elem(v, i0, i1, i2, i3);

    const double got = out_value(out);
    // The result is stored in the input type: allow its rounding at the magnitude of the total.
    const double tol = type == GGML_TYPE_F32 ? 1e-5 : type == GGML_TYPE_F16 ? 2e-3 : 1.6e-2;
    const double rel = std::fabs(got - ref) / std::fabs(ref);
    const bool ok = std::isfinite(got) && rel <= tol;
    printf("  %-5s ne=[%lld,%lld,%lld,%lld] %-17s nb0=%zu ref=%.4f got=%.4f rel=%.3e tol=%.1e %s%s\n",
           ggml_type_name(type), (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
           vname, v->nb[0], ref, got, rel, tol, ok ? "OK" : "FAIL", is_control ? " (control)" : "");
    ggml_free(ctx);
    return ok;
}

int main() {
    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 };
    const int64_t ne_2d[4] = { 33, 256, 1, 1 };  // the test-backend-ops shape
    const int64_t ne_4d[4] = { 11, 5, 6, 3 };
    int fails = 0, control_fails = 0, n = 0;
    for (ggml_type t : types) {
        control_fails += !run_case(t, ne_2d, 0, true);  ++n;
        control_fails += !run_case(t, ne_4d, 3, true);  ++n;
        fails         += !run_case(t, ne_2d, 1, false); ++n;
        fails         += !run_case(t, ne_2d, 2, false); ++n;
        fails         += !run_case(t, ne_4d, 1, false); ++n;
    }
    if (control_fails) {
        printf("CONTROL FAILED: %d control case(s) wrong -- the comparison is broken, verdicts are void\n", control_fails);
        return 2;
    }
    printf("%d cases, %d strided failure(s)\n", n, fails);
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails ? 1 : 0;
}
