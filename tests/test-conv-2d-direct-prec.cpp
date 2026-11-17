// ggml_conv_2d_direct must accumulate in F32 on every backend.
//
// WHY IT EXISTS. Upstream GGML_OP_CONV_2D is one fused kernel whose accumulator is a float
// (llama.cpp ggml/src/ggml-cuda/conv2d.cu: the scalar path keeps `float acc`, the MMA path a float
// accumulator tile). This fork expresses ggml_conv_2d_direct as im2col + mul_mat. With an F16 kernel
// that mul_mat is F16 x F16, and the CUDA backend then picks the cuBLAS F16 compute type
// (ggml-cuda.cu ggml_cuda_mul_mat_cublas) unless the node asks for GGML_PREC_F32. F16 tops out at
// 65504, so a convolution whose outputs exceed that came back as inf on the GPU while the CPU was
// right. MEASURED 2026-10-20 on gemma3n (MobileNetV5 vision tower, mmproj F16): the first
// catastrophic node was MUL_MAT(v.blk.2.24.pw_exp.conv.weight, im2col) with sum -inf on CUDA0 and a
// finite -2.45e9 on CPU; the GPU encoder then described nothing ("0x0x0x..."), the CPU one
// described the image. Row S69-OLLAMA-PICK-GEMMA3N-MMPROJ-CPU-CORRUPTION-GUARD-2026-09-29-01.
//
// WHAT IT CHECKS.
//   1. graph shape: the MUL_MAT that ggml_conv_2d_direct builds carries GGML_PREC_F32 (F16 and F32
//      kernels). This is backend-free and is the fail-before check on a CPU-only build.
//   2. numbers: a 1x1 and a 3x3 convolution whose true outputs reach ~2x the F16 maximum are run on
//      every backend device present (CPU always; CUDA etc. when built in) and compared with a double
//      reference. All outputs must be finite and within 1e-3 relative RMS.
//
// SCOPE: the im2col + mul_mat emulation in ggml.c only; ggml_conv_2d (the classic im2col path, same
// precision contract as upstream) is not changed and not tested here.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static ggml_tensor * find_mul_mat(ggml_cgraph * gf) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        if (t->op == GGML_OP_MUL_MAT) {
            return t;
        }
    }
    return nullptr;
}

static int check_graph_prec(ggml_type ktype) {
    ggml_init_params ip = { 16u*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, ktype, 1, 1, 64, 32);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 64, 1);
    ggml_tensor * y = ggml_conv_2d_direct(ctx, k, x, 1, 1, 0, 0, 1, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_tensor * mm = find_mul_mat(gf);
    int fail = 0;
    if (!mm) {
        printf("FAIL prec  kernel %-4s: no MUL_MAT node in the conv_2d_direct graph\n", ggml_type_name(ktype));
        fail = 1;
    } else if (mm->op_params[0] != GGML_PREC_F32) {
        printf("FAIL prec  kernel %-4s: MUL_MAT op_params[0] = %d, want GGML_PREC_F32 (%d)\n",
               ggml_type_name(ktype), mm->op_params[0], (int) GGML_PREC_F32);
        fail = 1;
    } else {
        printf("ok   prec  kernel %-4s: MUL_MAT carries GGML_PREC_F32\n", ggml_type_name(ktype));
    }
    ggml_free(ctx);
    return fail;
}

struct conv_case {
    int kw, kh, ic, oc, w, h, s, p;
};

// Positive inputs in [40, 60] and kernel taps in [8, 12] make every output a sum of IC*KW*KH
// products near 500, so the 1x1 case with IC 256 lands near 128000 (about 2x the F16 maximum).
static int run_numeric(ggml_backend_t be, const char * be_name, const conv_case & c) {
    const int ow = (c.w + 2*c.p - c.kw) / c.s + 1;
    const int oh = (c.h + 2*c.p - c.kh) / c.s + 1;

    std::mt19937 rng(77);
    std::uniform_real_distribution<float> UX(40.0f, 60.0f), UK(8.0f, 12.0f);
    std::vector<float> xf((size_t) c.w*c.h*c.ic), kf((size_t) c.kw*c.kh*c.ic*c.oc);
    for (auto & v : xf) v = UX(rng);
    for (auto & v : kf) v = UK(rng);
    // Round both through F16 so the reference sees exactly the values the F16 path multiplies.
    std::vector<ggml_fp16_t> kh16(kf.size()), xh16(xf.size());
    ggml_fp32_to_fp16_row(kf.data(), kh16.data(), (int64_t) kf.size());
    ggml_fp16_to_fp32_row(kh16.data(), kf.data(), (int64_t) kf.size());
    ggml_fp32_to_fp16_row(xf.data(), xh16.data(), (int64_t) xf.size());
    ggml_fp16_to_fp32_row(xh16.data(), xf.data(), (int64_t) xf.size());

    // reference, layout: x [W, H, IC], k [KW, KH, IC, OC], y [OW, OH, OC]
    std::vector<double> ref((size_t) ow*oh*c.oc);
    double ref_max = 0;
    for (int o = 0; o < c.oc; ++o) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                double s = 0;
                for (int ci = 0; ci < c.ic; ++ci) {
                    for (int ky = 0; ky < c.kh; ++ky) {
                        for (int kx = 0; kx < c.kw; ++kx) {
                            const int ix = ox*c.s - c.p + kx, iy = oy*c.s - c.p + ky;
                            if (ix < 0 || iy < 0 || ix >= c.w || iy >= c.h) continue;
                            s += (double) xf[((size_t) ci*c.h + iy)*c.w + ix] *
                                 kf[(((size_t) o*c.ic + ci)*c.kh + ky)*c.kw + kx];
                        }
                    }
                }
                ref[((size_t) o*oh + oy)*ow + ox] = s;
                ref_max = std::fmax(ref_max, std::fabs(s));
            }
        }
    }

    ggml_init_params ip = { 16u*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.kw, c.kh, c.ic, c.oc);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.w, c.h, c.ic, 1);
    ggml_tensor * y = ggml_conv_2d_direct(ctx, k, x, c.s, c.s, c.p, c.p, 1, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    char tag[160];
    snprintf(tag, sizeof(tag), "%-6s %dx%d IC %3d OC %3d %2dx%2d s%d p%d (|ref| max %.0f)", be_name, c.kw, c.kh, c.ic,
             c.oc, c.w, c.h, c.s, c.p, ref_max);
    if (!buf) {
        printf("FAIL %s  buffer allocation failed\n", tag);
        ggml_free(ctx);
        return 1;
    }
    ggml_backend_tensor_set(k, kh16.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
    int fail = 0;
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("FAIL %s  graph compute did not succeed\n", tag);
        fail = 1;
    } else {
        std::vector<float> out(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
        size_t n_bad = 0;
        double num = 0, den = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            if (!std::isfinite(out[i])) { ++n_bad; continue; }
            const double d = out[i] - ref[i];
            num += d*d;
            den += ref[i]*ref[i];
        }
        const double rr = den > 0 ? std::sqrt(num/den) : 0.0;
        if (n_bad || rr > 1e-3) {
            printf("FAIL %s  non-finite %zu of %zu, rel RMS %.3g\n", tag, n_bad, out.size(), rr);
            fail = 1;
        } else {
            printf("ok   %s  all finite, rel RMS %.3g\n", tag, rr);
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return fail;
}

int main() {
    int n_fail = 0;
    n_fail += check_graph_prec(GGML_TYPE_F16);
    n_fail += check_graph_prec(GGML_TYPE_F32);

    const conv_case cases[] = {
        { 1, 1, 256, 32, 8, 8, 1, 0 },   // pointwise, the gemma3n pw_exp shape class
        { 3, 3,  32, 16, 9, 9, 2, 1 },   // strided, padded, ~144000 at the centre
    };

    int n_dev_run = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) {
            printf("skip device %s: init failed\n", ggml_backend_dev_name(dev));
            continue;
        }
        ++n_dev_run;
        for (const auto & c : cases) {
            n_fail += run_numeric(be, ggml_backend_dev_name(dev), c);
        }
        ggml_backend_free(be);
    }
    if (n_dev_run == 0) {
        ggml_backend_t be = ggml_backend_cpu_init();
        for (const auto & c : cases) {
            n_fail += run_numeric(be, "CPU", c);
        }
        ggml_backend_free(be);
        n_dev_run = 1;
    }
    printf("%s: %d failure(s) over %d device(s)\n", n_fail ? "FAIL" : "PASS", n_fail, n_dev_run);
    return n_fail ? 1 : 0;
}
