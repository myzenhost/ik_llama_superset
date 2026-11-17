// CPU GGML_OP_ADD1 (a + scalar b) for every scalar type the op is built with, on contiguous and on
// row-strided a.
//
// ggml_add1 (ggml.c) asserts only that b is a scalar and that a is padded-1d; it does not constrain b's
// type. The CPU dispatcher (ggml-cpu/ops.cpp ggml_compute_forward_add1) sent every F32 a to
// ggml_compute_forward_add1_f32, which read the scalar as `*(float *) src1->data` -- for an F16 or BF16
// b that is the 2-byte value plus 2 bytes of whatever follows it, i.e. a wrong addend and an over-read.
// The CPU backend is the reference test-backend-ops compares the GPU backend against, so a wrong CPU
// ADD1 makes a correct GPU kernel report FAIL.
//
// Each case compares the op's output, element by element through the views' own strides, against
// a + (float) b computed here (a Q8_0 a goes through ggml_compute_forward_add1_q_f32; see run_case_q8 for
// its two defects). The addend 0.75 is exact in f32, f16 and bf16; a is a ramp, so a dropped
// or wrong addend cannot hide behind a constant fill.
//
// CONTROL: the F32-scalar cases and the F16-a cases were right before the fix; if one of them fails the
// comparison itself is wrong and the other verdicts mean nothing.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <initializer_list>
#include <vector>
#include <cstdio>

static float read_elem(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const char * p = (const char *) t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
    switch (t->type) {
        case GGML_TYPE_F32:  return *(const float *) p;
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) p);
        default:             return NAN;
    }
}

static void write_elem(ggml_tensor * t, int64_t i, float v) {
    switch (t->type) {
        case GGML_TYPE_F32:  ((float *) t->data)[i] = v; break;
        case GGML_TYPE_F16:  ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(v); break;
        case GGML_TYPE_BF16: ((ggml_bf16_t *) t->data)[i] = ggml_fp32_to_bf16(v); break;
        default: break;
    }
}

// strided = true: a is a padded-1d view (rows of 17 inside rows of 24), the layout ggml_add1 accepts
static bool run_case(ggml_type ta, ggml_type tb, bool strided, bool is_control) {
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t ne[4]   = { 17, 5, 3, 2 };
    const int64_t ne0_alloc = strided ? 24 : ne[0];

    ggml_tensor * base = ggml_new_tensor_4d(ctx, ta, ne0_alloc, ne[1], ne[2], ne[3]);
    for (int64_t i = 0; i < ggml_nelements(base); ++i) {
        write_elem(base, i, (float) (i % 509) / 64.0f - 3.0f);
    }
    ggml_tensor * a = base;
    if (strided) {
        a = ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base->nb[1], base->nb[2], base->nb[3], 0);
    }

    ggml_tensor * b = ggml_new_tensor_1d(ctx, tb, 1);
    write_elem(b, 0, 0.75f);

    ggml_tensor * out = ggml_add1(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 2);

    const float addend = read_elem(b, 0, 0, 0, 0);
    const double tol = ta == GGML_TYPE_F32 ? 1e-6 : ta == GGML_TYPE_F16 ? 2e-3 : 1.6e-2;
    double max_err = 0.0;
    int64_t n_bad = 0;
    for (int64_t i3 = 0; i3 < ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
                    const double ref = (double) read_elem(a, i0, i1, i2, i3) + addend;
                    const double got = read_elem(out, i0, i1, i2, i3);
                    const double err = std::fabs(got - ref) / std::fmax(1.0, std::fabs(ref));
                    if (!std::isfinite(got) || err > tol) {
                        ++n_bad;
                    }
                    if (err > max_err || !std::isfinite(got)) {
                        max_err = std::isfinite(got) ? err : INFINITY;
                    }
                }

    const bool ok = n_bad == 0;
    printf("  a=%-4s b=%-4s %-8s out[0]=%.4f expect=%.4f max_err=%.3e tol=%.1e bad=%lld %s%s\n",
           ggml_type_name(ta), ggml_type_name(tb), strided ? "strided" : "contig",
           read_elem(out, 0, 0, 0, 0), read_elem(a, 0, 0, 0, 0) + addend, max_err, tol, (long long) n_bad,
           ok ? "OK" : "FAIL", is_control ? " (control)" : "");
    ggml_free(ctx);
    return ok;
}

// Quantized a: ggml_compute_forward_add1_q_f32 asserted an F32 scalar (so F16/BF16 aborted), and it wrote
// each row to dst + i1*nb1 + i2*nb2 + i3*nb0 -- nb0, not nb3 -- so with ne3 > 1 every i3 > 0 row landed
// a few bytes past the start of the tensor and the real rows i3 > 0 were never written. Q8_0 a, compared
// row by row after dequantizing both a and the result; the tolerance covers one re-quantisation step (the
// ramp's block maxima are below 6, so a Q8_0 step is below 0.05) and is far below the 0.75 addend.
static bool run_case_q8(ggml_type tb, int64_t ne3, bool is_control) {
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t n0 = 64, ne1 = 5, nrows = ne1*ne3;
    std::vector<float> src(n0*nrows);
    for (int64_t i = 0; i < n0*nrows; ++i) {
        src[i] = (float) (i % 509) / 64.0f - 3.0f;
    }
    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, n0, ne1, 1, ne3);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src.data(), a->data, 0, nrows, n0, nullptr, nullptr);

    ggml_tensor * b = ggml_new_tensor_1d(ctx, tb, 1);
    write_elem(b, 0, 0.75f);

    ggml_tensor * out = ggml_add1(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 2);

    const float addend = read_elem(b, 0, 0, 0, 0);
    const auto * tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<float> ra(n0), ro(n0);
    const double tol = 0.05;
    double max_err = 0.0;
    int64_t n_bad = 0;
    for (int64_t r = 0; r < nrows; ++r) {
        tt->to_float((const char *) a->data   + r*a->nb[1],   ra.data(), n0);
        tt->to_float((const char *) out->data + r*out->nb[1], ro.data(), n0);
        for (int64_t i = 0; i < n0; ++i) {
            const double err = std::fabs((double) ro[i] - ((double) ra[i] + addend));
            if (!std::isfinite(ro[i]) || err > tol) {
                ++n_bad;
            }
            max_err = std::isfinite(ro[i]) ? std::fmax(max_err, err) : INFINITY;
        }
    }
    tt->to_float(out->data, ro.data(), n0);
    tt->to_float(a->data,   ra.data(), n0);

    const bool ok = n_bad == 0;
    printf("  a=%-4s b=%-4s %-8s out[0]=%.4f expect=%.4f max_err=%.3e tol=%.1e bad=%lld %s%s\n",
           ggml_type_name(GGML_TYPE_Q8_0), ggml_type_name(tb), ne3 > 1 ? "ne3=3" : "contig", ro[0], ra[0] + addend, max_err, tol,
           (long long) n_bad, ok ? "OK" : "FAIL", is_control ? " (control)" : "");
    ggml_free(ctx);
    return ok;
}

int main() {
    int fails = 0, control_fails = 0, n = 0;
    for (bool strided : { false, true }) {
        control_fails += !run_case(GGML_TYPE_F32, GGML_TYPE_F32,  strided, true);  ++n;
        control_fails += !run_case(GGML_TYPE_F16, GGML_TYPE_F32,  strided, true);  ++n;
        control_fails += !run_case(GGML_TYPE_F16, GGML_TYPE_F16,  strided, true);  ++n;
        fails         += !run_case(GGML_TYPE_F32, GGML_TYPE_F16,  strided, false); ++n;
        fails         += !run_case(GGML_TYPE_F32, GGML_TYPE_BF16, strided, false); ++n;
    }
    control_fails += !run_case_q8(GGML_TYPE_F32,  1, true);  ++n;
    fails         += !run_case_q8(GGML_TYPE_F32,  3, false); ++n;  // i3 row stride
    fails         += !run_case_q8(GGML_TYPE_F16,  1, false); ++n;  // scalar type (aborted before the fix)
    fails         += !run_case_q8(GGML_TYPE_BF16, 1, false); ++n;
    if (control_fails) {
        printf("CONTROL FAILED: %d control case(s) wrong -- the comparison is broken, verdicts are void\n", control_fails);
        return 2;
    }
    printf("%d cases, %d failure(s)\n", n, fails);
    printf(fails ? "FAIL\n" : "PASS\n");
    return fails ? 1 : 0;
}
