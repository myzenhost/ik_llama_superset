// MUL_MAT with a quantized weight and an F16 activation (src1), for activation row counts that are NOT a multiple of the
// interleave factor num_rows() reports for the weight's vec_dot_type.
//
// Why: the CPU backend converts a src1 that is neither F32 nor the weight's vec_dot_type through iqk_quantize_any()
// (ggml-cpu.c, the branch restored for the MLA path: an F16 KV view against a quantized weight). iqk_quantize_any asserts
// `ne1 % num_rows(vec_dot_type) == 0`, and num_rows(Q8_1) is 8 (16 with FANCY_SIMD), so a q4_1 / q5_1 weight against an F16
// src1 with ne11 = 1 (a decode step) aborted the process:
//     iqk_quantize.cpp:354: GGML_ASSERT(ne1 % n_interleaved == 0) failed
// Found 2026-10-01 by test-backend-ops -b CUDA0 -o MUL_MAT (the CPU backend is its reference), where it killed the run with
// rc 0xC0000409. For a plain (non-repacked) vec_dot_type the interleave only groups rows into one call: the bytes written
// are identical row by row, so the conversion must fall back to one row at a time instead of asserting.
// BENCH-PENDING: no performance claim is made anywhere in this change; it is a correctness fix for an abort.
//
// This test drives the public API only: build the graph, run it on the CPU backend, compare with a double-precision
// reference computed from the dequantized weights and the F16-rounded activations.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct case_result { bool ran = false; double rel_rms = 0; };

static case_result run_case(ggml_type type_a, int64_t m, int64_t n, int64_t k, std::mt19937 & rng) {
    case_result res;
    std::normal_distribution<float> nd(0.0f, 1.0f);

    ggml_init_params ip = { /*mem_size*/ 16u * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, type_a, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ggml_backend_supports_op(cpu, c)) {
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return res; // this build's CPU backend declines the op: nothing to check, reported as not run
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);

    // weights: random floats -> quantized rows
    std::vector<float> wf((size_t) m * k);
    for (auto & v : wf) v = nd(rng);
    std::vector<uint8_t> wq(ggml_row_size(type_a, k) * m);
    ggml_quantize_init(type_a);
    ggml_quantize_chunk(type_a, wf.data(), wq.data(), 0, m, k, nullptr, nullptr);
    ggml_backend_tensor_set(a, wq.data(), 0, wq.size());

    // dequantized weights = what the kernel is supposed to multiply by
    const auto * tt = ggml_get_type_traits(type_a);
    std::vector<float> wd((size_t) m * k);
    for (int64_t i = 0; i < m; ++i) {
        tt->to_float(wq.data() + i * ggml_row_size(type_a, k), wd.data() + (size_t) i * k, k);
    }

    // activations: random floats -> F16
    std::vector<float> xf((size_t) n * k);
    for (auto & v : xf) v = nd(rng);
    std::vector<ggml_fp16_t> xh((size_t) n * k);
    ggml_fp32_to_fp16_row(xf.data(), xh.data(), (int64_t) xf.size());
    ggml_backend_tensor_set(b, xh.data(), 0, xh.size() * sizeof(ggml_fp16_t));

    ggml_backend_graph_compute(cpu, gf);

    std::vector<float> out((size_t) m * n);
    ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(float));

    double se = 0, ss = 0;
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            double ref = 0;
            for (int64_t t = 0; t < k; ++t) {
                ref += (double) wd[(size_t) i * k + t] * (double) ggml_fp16_to_fp32(xh[(size_t) j * k + t]);
            }
            const double d = (double) out[(size_t) j * m + i] - ref;
            se += d * d;
            ss += ref * ref;
        }
    }
    res.ran = true;
    res.rel_rms = ss > 0 ? std::sqrt(se / ss) : 0.0;

    ggml_backend_buffer_free(buf);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return res;
}

int main() {
    // Weight types whose CPU vec_dot_type is a plain 8-bit activation block (Q8_0 / Q8_1 / Q8_K): the F16 src1 must be converted.
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K };
    // Activation row counts: 1 (decode), primes and non-multiples of 8/16, and exact multiples as controls.
    const int64_t ns[] = { 1, 2, 3, 5, 8, 16, 17 };
    const double limit = 0.05; // 8-bit activation quantization of N(0,1) data: ~1e-2 relative RMS

    std::mt19937 rng(1234);
    int failed = 0, ran = 0, skipped = 0;
    for (ggml_type t : types) {
        for (int64_t n : ns) {
            std::printf("  running %-6s n=%-3lld ...\n", ggml_type_name(t), (long long) n);
            std::fflush(stdout);
            const case_result r = run_case(t, 16, n, 256, rng);
            if (!r.ran) {
                ++skipped;
                std::printf("  skip %-6s n=%-3lld (CPU backend declines this op in this build)\n", ggml_type_name(t), (long long) n);
                continue;
            }
            ++ran;
            const bool ok = r.rel_rms < limit;
            failed += !ok;
            std::printf("  %s %-6s n=%-3lld rel_rms %.5f (limit %.2f)\n", ok ? "ok  " : "FAIL", ggml_type_name(t), (long long) n, r.rel_rms, limit);
            std::fflush(stdout);
        }
    }
    std::printf("%d case(s) ran, %d skipped, %d failed\n", ran, skipped, failed);
    if (ran == 0) {
        std::printf("FAIL: no case ran -- the test measured nothing\n");
        return 2;
    }
    return failed ? 1 : 0;
}
