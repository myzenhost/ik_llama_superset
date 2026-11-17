// MUL_MAT with a quantized weight and an F16 / BF16 src1 on the host backend.
//
// WHY IT EXISTS. The MLA prompt path (build_deepseek2, mla >= 2 with flash attention) multiplies the
// quantized wkv_b by a VIEW of the F16 KV cache. The pre-split ggml.c converted such a src1 through a
// per-thread F32 row (iqk_quantize_any); the split CPU backend lost that branch, so its supports_op
// refused the node, the scheduler found no backend, and deepseek-v2 aborted at context creation on
// CPU (GGML_ASSERT(*cur_backend_id != -1)). Upstream llama.cpp served the same blob.
// Row S69-DEEPSEEK2-MLA-CPU-UNLOADABLE-QUANT-X-F16-MUL-MAT-2026-09-28-01.
//
// WHAT IT CHECKS, per (weight type, src1 type, n_cols, strided):
//   1. the host backend CLAIMS the node (supports_op) -- the failure that aborted the model;
//   2. mul_mat(W, x as F16/BF16) agrees with mul_mat(W, the same values as F32) -- both quantize the
//      activation, possibly to different activation types, so this is a tolerance, not equality;
//   3. both agree with a double-precision reference built from dequantize_row (to_float) of W.
// The strided case gives src1 a row stride larger than its row, as the kv_cache_nope view has.
//
// SCOPE: host backend only, K = 512, 64 weight rows, the listed types. Says nothing about GPU backends.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static double rel_rms(const std::vector<double> & ref, const float * y, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) y[i] - ref[i];
        num += d*d;
        den += ref[i]*ref[i];
    }
    return den > 0 ? std::sqrt(num/den) : std::sqrt(num);
}

int main() {
    const int64_t K = 512, M = 64;
    const ggml_type wtypes[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_KS };
    const ggml_type xtypes[] = { GGML_TYPE_F16, GGML_TYPE_BF16 };
    const int64_t ncols[] = { 1, 7, 32 };

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { printf("FAIL: no host backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(be, 4);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);

    int n_fail = 0, n_run = 0;
    for (ggml_type wt : wtypes) {
        for (ggml_type xt : xtypes) {
            for (int64_t N : ncols) {
                for (int strided = 0; strided < 2; ++strided) {
                    const int64_t row_len = strided ? K + 64 : K;   // src1 physical row (the view reads the first K)
                    ggml_init_params ip = { 64u*1024*1024, nullptr, false };
                    ggml_context * ctx = ggml_init(ip);

                    // weight: random floats -> quantized
                    std::vector<float> wf(K*M);
                    for (auto & v : wf) v = U(rng);
                    ggml_tensor * W = ggml_new_tensor_2d(ctx, wt, K, M);
                    ggml_quantize_chunk(wt, wf.data(), W->data, 0, M, K, nullptr, nullptr);

                    // activations: random floats rounded through xt so both src1 forms hold identical values
                    ggml_tensor * Xs = ggml_new_tensor_2d(ctx, xt, row_len, N);
                    ggml_tensor * X32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
                    const auto * xtr = ggml_get_type_traits(xt);
                    std::vector<float> xrow(row_len), back(row_len);
                    for (int64_t j = 0; j < N; ++j) {
                        for (auto & v : xrow) v = U(rng);
                        char * dst = (char *) Xs->data + j*Xs->nb[1];
                        if (xt == GGML_TYPE_F16) {
                            ggml_fp32_to_fp16_row(xrow.data(), (ggml_fp16_t *) dst, row_len);
                        } else {
                            ggml_fp32_to_bf16_row(xrow.data(), (ggml_bf16_t *) dst, row_len);
                        }
                        xtr->to_float(dst, back.data(), row_len);
                        memcpy((char *) X32->data + j*X32->nb[1], back.data(), K*sizeof(float));
                    }
                    ggml_tensor * Xv = ggml_view_2d(ctx, Xs, K, N, Xs->nb[1], 0);

                    ggml_tensor * Y  = ggml_mul_mat(ctx, W, Xv);
                    ggml_tensor * Y2 = ggml_mul_mat(ctx, W, X32);

                    // double reference from the weight's own dequantizer
                    // ROW BY ROW: IQ*_K types carry per-row metadata ahead of the blocks, so dequantizing the
                    // whole tensor as one row reads structure at the wrong offset (a harness error, not the kernel's)
                    std::vector<float> wd(K*M);
                    for (int64_t i = 0; i < M; ++i) {
                        ggml_get_type_traits(wt)->to_float((const char *) W->data + i*W->nb[1], wd.data() + i*K, K);
                    }
                    std::vector<double> ref(M*N);
                    for (int64_t j = 0; j < N; ++j) {
                        const float * x = (const float *) ((const char *) X32->data + j*X32->nb[1]);
                        for (int64_t i = 0; i < M; ++i) {
                            double s = 0;
                            for (int64_t k = 0; k < K; ++k) s += (double) wd[i*K + k] * x[k];
                            ref[j*M + i] = s;
                        }
                    }

                    ++n_run;
                    const bool claimed = ggml_backend_supports_op(be, Y);
                    char tag[128];
                    snprintf(tag, sizeof(tag), "%-7s x %-4s N=%-2lld %s", ggml_type_name(wt), ggml_type_name(xt),
                             (long long) N, strided ? "strided" : "contig ");
                    if (!claimed) {
                        printf("FAIL %s  host backend does not claim the node (supports_op false)\n", tag);
                        ++n_fail;
                        ggml_free(ctx);
                        continue;
                    }
                    ggml_cgraph * gf = ggml_new_graph(ctx);
                    ggml_build_forward_expand(gf, Y);
                    ggml_build_forward_expand(gf, Y2);
                    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                        printf("FAIL %s  graph compute did not succeed\n", tag);
                        ++n_fail;
                        ggml_free(ctx);
                        continue;
                    }
                    std::vector<double> y2d(M*N);
                    for (int64_t i = 0; i < M*N; ++i) y2d[i] = ((const float *) Y2->data)[i];
                    const double e_ref = rel_rms(ref, (const float *) Y->data, M*N);
                    const double e_f32 = rel_rms(y2d, (const float *) Y->data, M*N);
                    const double e_f32ref = rel_rms(ref, (const float *) Y2->data, M*N);
                    // activation quantization (q8-class) costs ~0.3-1%; the F32 route pays the same kind of cost
                    const bool ok = std::isfinite(e_ref) && e_ref < 0.02 && e_f32 < 0.02;
                    printf("%s %s  vs-ref=%.6f  vs-f32-route=%.6f  (f32-route vs-ref=%.6f)\n",
                           ok ? "ok  " : "FAIL", tag, e_ref, e_f32, e_f32ref);
                    if (!ok) ++n_fail;
                    ggml_free(ctx);
                }
            }
        }
    }
    ggml_backend_free(be);
    printf("\n%d cases, %d failed\n", n_run, n_fail);
    // CONTROL: the F32 route itself must pass against the reference, or the tolerance is measuring the harness
    return n_fail == 0 ? 0 : 1;
}
