// Native FP4 prefill GEMM for NVFP4 weights (row S69-FASTLLM-PICK-SM120-NATIVE-NVFP4-FP8-PREFILL-2026-10-05-01).
//
// MUL_MAT with an NVFP4 src0 and a prefill-sized F32 src1 runs on the Blackwell FP4 tensor cores when
// GGML_CUDA_NVFP4_PREFILL=1. This builds one weight, runs the same graph with the flag OFF and ON for several batch
// sizes, and checks against a double-precision host reference made from the dequantised weights:
//   M=64   below the 128-row threshold: must NOT be handled (decode-sized batches keep the old path)
//   M=128  exactly one 128-row tile (no staging buffer)
//   M=200  partial last tile (output staged through a scratch buffer, only 200 rows written)
//   M=5000 two chunks (4096 + 904): chunk loop and a short last chunk
// Activation rows are scaled 10^-2..10^2 so the per-row global scale is exercised. The ON arm quantises activations
// to E2M1, so it is checked against a normalised-RMS bound, not for equality; a layout bug (swapped nibbles, wrong
// scale tile) gives ~1.0, the correct path ~0.1. The OFF arm is the DEFAULT NVFP4 path: since plain NVFP4 MMQ (c2d7f5cf23,
// upstream 84f82e846c semantics) that is MMQ with q8_1-quantised activations, whose nRMS is the q8_1 rounding floor
// (~1/127 = 0.0079; MEASURED 0.00748-0.00754 at M=64..5000 on CUDA0, gate b1115b 2026-11-15), so the OFF arm is held to
// 0.02 (2.7x that floor). Before c2d7f5cf23 the OFF arm was dequantise + cuBLAS (exact) and was held to 2e-3; a layout
// or scale bug in the MMQ tile loader still gives ~1.0 and fails by 50x. It also reads the fire counters:
// every ON batch >= 128 must increment "handled", and none may be "declined".
//
// Fail-before arm: GGML_CUDA_NVFP4_PREFILL_MIN_M=100000000 makes the path refuse every batch, which is exactly the
// behaviour of a build without it; handled stays 0 and this exits 1.
//
// Repacked-weight cache (row S69-FASTLLM-PICK-NVFP4-PREFILL-REPACK-CACHE-AUTOTUNE-2026-10-27-01): after the batches above the
// weight buffer is declared WEIGHTS and the test checks that the first call repacks once, later calls hit the cache and give the
// same bits, a tensor_set (whole or one row) drops the entry so the result follows the NEW bytes, and a budget of 0 repacks on
// every call with identical output. Its fail-before arm is GGML_CUDA_NVFP4_PREFILL_CACHE_MB=0 in the environment (a build
// without the cache behaves the same way): the "second call must reuse" check fails and this exits 1.
//
// `test-cuda-nvfp4-prefill --bench` times the op per shape instead (see bench()).
//
// Exit: 0 pass, 1 fail, 77 skip (no CUDA device, no Blackwell, or the backend could not be created).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void set_env(const char * k, const char * v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double rnd01() {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return ((g_rng >> 11) + 0.5) / 9007199254740992.0;
}
static float gauss() {
    return (float) (std::sqrt(-2.0 * std::log(rnd01())) * std::cos(2.0 * M_PI * rnd01()));
}

static bool run_graph(ggml_backend_t backend, ggml_tensor * w, const std::vector<float> & xh, int K, int M, int N, std::vector<float> & out_h) {
    const size_t meta_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params ip = { meta_size, meta.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(galloc, gf);
    if (ok) {
        ggml_backend_tensor_set(x, xh.data(), 0, sizeof(float) * xh.size());
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out_h.resize((size_t) N * M);
        ggml_backend_tensor_get(y, out_h.data(), 0, sizeof(float) * out_h.size());
    }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return ok;
}

// --graphs: one persistent graph computed repeatedly with CUDA graphs ON (2nd call warms, 3rd captures, later calls replay),
// then the weight is overwritten and the same graph computed again. A replayed graph must follow the NEW weight: if a captured
// call had baked a cache entry's pointers, the entry freed by the write would leave the replay on stale or freed memory.
static int graphs_arm(ggml_backend_t backend) {
    const int K = 256, N = 256, M = 200;
    ggml_quantize_init(GGML_TYPE_NVFP4);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_NVFP4, K);
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    auto make_w = [&](std::vector<uint8_t> & wq, std::vector<float> & wd) {
        std::vector<float> wf((size_t) K * N);
        for (auto & v : wf) v = gauss() * 0.5f;
        wq.assign(row_bytes * N, 0);
        ggml_quantize_chunk(GGML_TYPE_NVFP4, wf.data(), wq.data(), 0, N, K, nullptr, nullptr);
        wd.assign((size_t) K * N, 0.0f);
        for (int r = 0; r < N; ++r) tr->to_float(wq.data() + row_bytes * r, wd.data() + (size_t) K * r, K);
    };
    std::vector<uint8_t> wq1, wq2;
    std::vector<float> wd1, wd2;
    make_w(wq1, wd1);
    make_w(wq2, wd2);
    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_NVFP4, K, N);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(w, wq1.data(), 0, wq1.size());

    const size_t meta_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params ip = { meta_size, meta.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { printf("graphs: alloc failed\n"); return 1; }
    std::vector<float> xh((size_t) K * M);
    for (auto & v : xh) v = gauss();
    ggml_backend_tensor_set(x, xh.data(), 0, sizeof(float) * xh.size());
    set_env("GGML_CUDA_NVFP4_PREFILL", "1");

    auto ref_for = [&](const std::vector<float> & wd) {
        std::vector<double> ref((size_t) N * M);
        for (int r = 0; r < M; ++r)
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) acc += (double) xh[(size_t) r * K + k] * (double) wd[(size_t) n * K + k];
                ref[(size_t) r * N + n] = acc;
            }
        return ref;
    };
    auto nrms = [&](const std::vector<float> & got, const std::vector<double> & ref) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) { const double d = got[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; }
        return std::sqrt(num / den);
    };
    std::vector<float> out((size_t) N * M);
    for (int i = 0; i < 8; ++i) ggml_backend_graph_compute(backend, gf); // warm, capture, replay x6
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(y, out.data(), 0, sizeof(float) * out.size());
    const double e_old_before = nrms(out, ref_for(wd1));
    ggml_backend_tensor_set(w, wq2.data(), 0, wq2.size());
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(y, out.data(), 0, sizeof(float) * out.size());
    const double e_new = nrms(out, ref_for(wd2)), e_old = nrms(out, ref_for(wd1));
    printf("graphs: before write nRMS vs OLD %.5f | after write nRMS vs NEW %.5f, vs OLD %.5f\n", e_old_before, e_new, e_old);
    int fail = 0;
    if (e_old_before > 0.2) { printf("  FAIL: replayed result does not match the weight\n"); ++fail; }
    if (e_new > 0.2)        { printf("  FAIL: replay after a weight write does not follow the NEW weight\n"); ++fail; }
    if (e_old < 0.5)        { printf("  FAIL: replay after a weight write still follows the OLD weight (stale baked cache)\n"); ++fail; }
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}

// --bench: time the MUL_MAT op itself (graph built once, weights in a WEIGHTS buffer) so the repacked-weight cache and the
// timed algorithm pick can be measured one at a time with the env switches:
//   GGML_CUDA_NVFP4_PREFILL_CACHE_MB=0 (cache off)  GGML_CUDA_NVFP4_PREFILL_AUTOTUNE=0 (heuristic first pick)
// Prints one line per shape: ms/call for the dequantise+cuBLAS path (flag off) and for the FP4 path, median of 5 blocks.
static int bench(ggml_backend_t backend) {
    struct shape { int K, N, M; };
    const shape shapes[] = { {4096, 4096, 512}, {4096, 4096, 2048}, {4096, 11008, 512}, {4096, 11008, 2048}, {2048, 6144, 512} };
    ggml_quantize_init(GGML_TYPE_NVFP4);
    for (const shape & s : shapes) {
        const size_t row_bytes = ggml_row_size(GGML_TYPE_NVFP4, s.K);
        // 256 distinct quantised rows tiled down the weight: timing does not depend on the values
        std::vector<float> wf((size_t) s.K * 256);
        for (auto & v : wf) v = gauss() * 0.5f;
        std::vector<uint8_t> tile(row_bytes * 256);
        ggml_quantize_chunk(GGML_TYPE_NVFP4, wf.data(), tile.data(), 0, 256, s.K, nullptr, nullptr);
        std::vector<uint8_t> wq(row_bytes * s.N);
        for (int r = 0; r < s.N; ++r) std::memcpy(wq.data() + row_bytes * r, tile.data() + row_bytes * (r % 256), row_bytes);

        ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context * wctx = ggml_init(wp);
        ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_NVFP4, s.K, s.N);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());

        const size_t meta_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
        std::vector<uint8_t> meta(meta_size);
        ggml_init_params ip = { meta_size, meta.data(), true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.K, s.M);
        ggml_set_input(x);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        ggml_set_output(y);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { printf("bench: alloc failed\n"); return 1; }
        std::vector<float> xh((size_t) s.K * s.M);
        for (auto & v : xh) v = gauss();
        ggml_backend_tensor_set(x, xh.data(), 0, sizeof(float) * xh.size());

        double ms_arm[2] = {0, 0};
        for (int arm = 0; arm < 2; ++arm) {
            set_env("GGML_CUDA_NVFP4_PREFILL", arm ? "1" : "0");
            for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf); // warm-up: fills the cache, tunes the plan
            ggml_backend_synchronize(backend);
            std::vector<double> blocks;
            for (int blk = 0; blk < 5; ++blk) {
                const int iters = 20;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
                ggml_backend_synchronize(backend);
                const auto t1 = std::chrono::steady_clock::now();
                blocks.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
            }
            std::sort(blocks.begin(), blocks.end());
            ms_arm[arm] = blocks[2];
        }
        unsigned long long h, rp, t;
        ggml_backend_cuda_nvfp4_prefill_cache_stats(&h, &rp, &t);
        printf("BENCH K=%d N=%d M=%d | dequant+cuBLAS %.4f ms | FP4 path %.4f ms | cache hits %llu repacks %llu tuned plans %llu\n",
               s.K, s.N, s.M, ms_arm[0], ms_arm[1], h, rp, t);
        fflush(stdout);
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
    }
    return 0;
}

// --native: the in-place mma.sync W4A4 kernel (nvfp4.cu, GGML_CUDA_NVFP4_PREFILL=native; row S69-DRIFT-1110-W4A4-BLACKWELL-NVFP4-MXFP4).
// Dense MUL_MAT at M = 8 (below GGML_CUDA_NVFP4_NATIVE_MIN_M=16: must NOT fire), 16, 200 (partial tile), 5000; MUL_MAT_ID (MoE) with
// 8 experts / top-2 and 300 tokens; the fused up/gate + SiLU graph (ggml_mul_mat_id x2 + swiglu is not built here: the dense fused
// kernel is checked through two MUL_MATs + SILU-mul graph the CUDA backend fuses only when it is adjacent, so only the counters of
// dense and id are asserted). Each ON result is compared with a double-precision reference from the dequantised weight, nRMS < 0.2
// (E2M1 activations; a layout/scale bug gives ~1.0). Fail-before: without the native path (flag unset, or a build without it) the
// "fired" counters stay 0 and this exits 1. Exit: 0 pass, 1 fail, 77 skip (no sm_120a build: the "NVFP4 W4A4 MMQ" path never fires
// and the counters stay 0 with the flag ON on a device that is not 1200/1210 -> reported as SKIP only when the device is not Blackwell).
static int native_arm(ggml_backend_t backend) {
    char desc[256] = {0};
    ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    const int K = 512, N = 256, NE = 8;
    ggml_quantize_init(GGML_TYPE_NVFP4);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_NVFP4, K);
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    std::vector<float> wf((size_t) K * N * NE);
    for (auto & v : wf) v = gauss() * 0.5f;
    std::vector<uint8_t> wq(row_bytes * N * NE);
    ggml_quantize_chunk(GGML_TYPE_NVFP4, wf.data(), wq.data(), 0, N * NE, K, nullptr, nullptr);
    std::vector<float> wd((size_t) K * N * NE);
    for (int r = 0; r < N * NE; ++r) tr->to_float(wq.data() + row_bytes * r, wd.data() + (size_t) K * r, K);

    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w2 = ggml_new_tensor_2d(wctx, GGML_TYPE_NVFP4, K, N);
    ggml_tensor * w3 = ggml_new_tensor_3d(wctx, GGML_TYPE_NVFP4, K, N, NE);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_tensor_set(w2, wq.data(), 0, row_bytes * N);
    ggml_backend_tensor_set(w3, wq.data(), 0, wq.size());

    int fail = 0;
    unsigned long long fired_total = 0;
    auto nrms = [](const std::vector<float> & got, const std::vector<double> & ref) {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) { const double d = got[i] - ref[i]; num += d * d; den += ref[i] * ref[i]; }
        return std::sqrt(num / den);
    };
    // ---- dense ----
    const int Ms[4] = { 8, 16, 200, 5000 };
    for (int M : Ms) {
        std::vector<float> xh((size_t) K * M);
        for (int r = 0; r < M; ++r) {
            const float scale = std::pow(10.0f, (float) (r % 5) - 2.0f);
            for (int k = 0; k < K; ++k) xh[(size_t) r * K + k] = gauss() * scale;
        }
        std::vector<double> ref((size_t) N * M);
        for (int r = 0; r < M; ++r)
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) acc += (double) xh[(size_t) r * K + k] * (double) wd[(size_t) n * K + k];
                ref[(size_t) r * N + n] = acc;
            }
        std::vector<float> on;
        unsigned long long d0, i0, g0, d1, i1, g1;
        set_env("GGML_CUDA_NVFP4_PREFILL", "native");
        ggml_backend_cuda_nvfp4_native_stats(&d0, &i0, &g0);
        if (!run_graph(backend, w2, xh, K, M, N, on)) { printf("FAIL: native compute failed M=%d\n", M); return 1; }
        ggml_backend_cuda_nvfp4_native_stats(&d1, &i1, &g1);
        set_env("GGML_CUDA_NVFP4_PREFILL", "0");
        const double e = nrms(on, ref);
        const bool expect = M >= 16;
        printf("dense M=%-5d nrms %.5f | native dense +%llu\n", M, e, d1 - d0);
        fired_total += d1 - d0;
        if (expect && d1 - d0 != 1) { printf("  FAIL: native kernel did not fire (want +1)\n"); ++fail; }
        if (!expect && d1 != d0)    { printf("  FAIL: batch below GGML_CUDA_NVFP4_NATIVE_MIN_M was handled\n"); ++fail; }
        if (e > 0.2) { printf("  FAIL: nRMS %.5f > 0.2\n", e); ++fail; }
    }
    // ---- MUL_MAT_ID: 8 experts, top-2, T tokens ----
    {
        const int T = 300, NU = 2;
        std::vector<float> xh((size_t) K * T);
        for (auto & v : xh) v = gauss();
        std::vector<int32_t> ids((size_t) NU * T);
        for (int t = 0; t < T; ++t) { ids[(size_t) t * NU] = t % NE; ids[(size_t) t * NU + 1] = (t * 3 + 1) % NE; if (ids[(size_t) t * NU + 1] == ids[(size_t) t * NU]) ids[(size_t) t * NU + 1] = (ids[(size_t) t * NU] + 1) % NE; }
        const size_t meta_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
        std::vector<uint8_t> meta(meta_size);
        ggml_init_params ip = { meta_size, meta.data(), true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
        ggml_tensor * ie = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NU, T);
        ggml_set_input(x); ggml_set_input(ie);
        ggml_tensor * y = ggml_mul_mat_id(ctx, w3, x, ie);
        ggml_set_output(y);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { printf("FAIL: id alloc failed\n"); return 1; }
        ggml_backend_tensor_set(x, xh.data(), 0, sizeof(float) * xh.size());
        ggml_backend_tensor_set(ie, ids.data(), 0, sizeof(int32_t) * ids.size());
        unsigned long long d0, i0, g0, d1, i1, g1;
        set_env("GGML_CUDA_NVFP4_PREFILL", "native");
        ggml_backend_cuda_nvfp4_native_stats(&d0, &i0, &g0);
        const bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_cuda_nvfp4_native_stats(&d1, &i1, &g1);
        set_env("GGML_CUDA_NVFP4_PREFILL", "0");
        if (!ok) { printf("FAIL: id compute failed\n"); return 1; }
        std::vector<float> out((size_t) N * NU * T);
        ggml_backend_tensor_get(y, out.data(), 0, sizeof(float) * out.size());
        std::vector<double> ref(out.size());
        for (int t = 0; t < T; ++t)
            for (int u = 0; u < NU; ++u)
                for (int n = 0; n < N; ++n) {
                    double acc = 0.0;
                    const size_t wr = (size_t) ids[(size_t) t * NU + u] * N + n;
                    for (int k = 0; k < K; ++k) acc += (double) xh[(size_t) t * K + k] * (double) wd[wr * K + k];
                    ref[((size_t) t * NU + u) * N + n] = acc;
                }
        const double e = nrms(out, ref);
        printf("mul_mat_id E=%d top%d T=%d nrms %.5f | native id +%llu\n", NE, NU, T, e, i1 - i0);
        fired_total += i1 - i0;
        if (i1 - i0 != 1) { printf("  FAIL: native MUL_MAT_ID kernel did not fire (want +1)\n"); ++fail; }
        if (e > 0.2) { printf("  FAIL: nRMS %.5f > 0.2\n", e); ++fail; }
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    if (fired_total == 0 && !(std::strstr(desc, "RTX 50") || std::strstr(desc, "Blackwell"))) {
        printf("SKIP: device %s is not Blackwell, native path correctly never fired\n", desc);
        return 77;
    }
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (ggml_backend_cuda_get_device_count() < 1) {
        printf("SKIP: no CUDA device\n");
        return 77;
    }
    // The counter arms below read host-side counters after each graph_compute. With CUDA graphs on, the second identical
    // graph is captured and the third replayed, and a replay runs no host code, so the counters would not move (measured
    // 2026-10-27: repacks +0 hits +0 in the cache arms with graphs on, PASS with GGML_CUDA_DISABLE_GRAPHS=1). The graph
    // switch is read once at the first compute, so it is set here, before the backend is created. --bench leaves it to the caller.
    const bool is_bench = argc > 1 && (std::strcmp(argv[1], "--bench") == 0 || std::strcmp(argv[1], "--graphs") == 0);
    if (!is_bench && getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr) {
        set_env("GGML_CUDA_DISABLE_GRAPHS", "1");
    }
    char desc[256] = {0};
    ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    ggml_backend_t backend = ggml_backend_cuda_init(0, NULL, NULL);
    if (backend == nullptr) {
        printf("SKIP: CUDA backend could not be created\n");
        return 77;
    }
    printf("device 0: %s\n", desc);
    if (argc > 1 && std::strcmp(argv[1], "--graphs") == 0) {
        const int rc = graphs_arm(backend);
        ggml_backend_free(backend);
        return rc;
    }
    if (argc > 1 && std::strcmp(argv[1], "--native") == 0) {
        const int rc = native_arm(backend);
        ggml_backend_free(backend);
        return rc;
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        const int rc = bench(backend);
        ggml_backend_free(backend);
        return rc;
    }

    const int K = 256, N = 256;
    ggml_quantize_init(GGML_TYPE_NVFP4);

    // weight: N rows of K, quantised row by row, then dequantised for the host reference
    std::vector<float> wf((size_t) K * N);
    for (auto & v : wf) v = gauss() * 0.5f;
    const size_t row_bytes = ggml_row_size(GGML_TYPE_NVFP4, K);
    std::vector<uint8_t> wq(row_bytes * N);
    ggml_quantize_chunk(GGML_TYPE_NVFP4, wf.data(), wq.data(), 0, N, K, nullptr, nullptr);
    std::vector<float> wd((size_t) K * N);
    const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_NVFP4);
    for (int r = 0; r < N; ++r) {
        tr->to_float(wq.data() + row_bytes * r, wd.data() + (size_t) K * r, K);
    }

    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_NVFP4, K, N);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());

    const int Ms[4] = { 64, 128, 200, 5000 };
    int fail = 0;
    for (int mi = 0; mi < 4; ++mi) {
        const int M = Ms[mi];
        std::vector<float> xh((size_t) K * M);
        for (int r = 0; r < M; ++r) {
            const float scale = std::pow(10.0f, (float) (r % 5) - 2.0f);
            for (int k = 0; k < K; ++k) xh[(size_t) r * K + k] = gauss() * scale;
        }
        std::vector<double> ref((size_t) N * M);
        for (int r = 0; r < M; ++r) {
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) acc += (double) xh[(size_t) r * K + k] * (double) wd[(size_t) n * K + k];
                ref[(size_t) r * N + n] = acc;
            }
        }

        std::vector<float> off, on;
        unsigned long long h0, d0, h1, d1, h2, d2;
        set_env("GGML_CUDA_NVFP4_PREFILL", "0");
        ggml_backend_cuda_nvfp4_prefill_stats(&h0, &d0);
        if (!run_graph(backend, w, xh, K, M, N, off)) { printf("FAIL: OFF compute failed M=%d\n", M); return 1; }
        ggml_backend_cuda_nvfp4_prefill_stats(&h1, &d1);
        set_env("GGML_CUDA_NVFP4_PREFILL", "1");
        if (!run_graph(backend, w, xh, K, M, N, on)) { printf("FAIL: ON compute failed M=%d\n", M); return 1; }
        ggml_backend_cuda_nvfp4_prefill_stats(&h2, &d2);

        auto nrms = [&](const std::vector<float> & got) {
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double d = got[i] - ref[i];
                num += d * d; den += ref[i] * ref[i];
            }
            return std::sqrt(num / den);
        };
        const double e_off = nrms(off), e_on = nrms(on);
        // per-row error: a wrong row scale shows on the rows of one magnitude class even if the total is small
        double worst_row = 0.0;
        for (int r = 0; r < M; ++r) {
            double num = 0.0, den = 0.0;
            for (int n = 0; n < N; ++n) {
                const double d = on[(size_t) r * N + n] - ref[(size_t) r * N + n];
                num += d * d; den += ref[(size_t) r * N + n] * ref[(size_t) r * N + n];
            }
            worst_row = std::fmax(worst_row, std::sqrt(num / den));
        }
        printf("  row0 ref/off/on:");
        for (int n = 0; n < 4; ++n) printf(" [%.4f %.4f %.4f]", ref[n], off[n], on[n]);
        printf("\n  last-row ref/off/on:");
        for (int n = 0; n < 4; ++n) printf(" [%.4f %.4f %.4f]", ref[(size_t) (M - 1) * N + n], off[(size_t) (M - 1) * N + n], on[(size_t) (M - 1) * N + n]);
        printf("\n");
        const bool expect_handled = M >= 128;
        printf("M=%-5d nrms OFF %.5f  ON %.5f  worst-row ON %.4f | counters handled +%llu (off arm +%llu) declined +%llu\n",
               M, e_off, e_on, worst_row, h2 - h1, h1 - h0, d2 - d1);

        if (e_off > 2e-2) { printf("  FAIL: OFF arm error %.5f > 2e-2 (q8_1 MMQ floor ~0.0079)\n", e_off); ++fail; }
        if (h1 != h0 || d1 != d0) { printf("  FAIL: counters moved with the flag OFF\n"); ++fail; }
        if (d2 != d1) { printf("  FAIL: %llu batches declined\n", d2 - d1); ++fail; }
        if (expect_handled) {
            if (h2 - h1 != 1) { printf("  FAIL: native FP4 path did not fire (handled +%llu, want +1)\n", h2 - h1); ++fail; }
            if (e_on > 0.2)   { printf("  FAIL: ON arm nRMS %.5f > 0.2 (layout or scale bug)\n", e_on); ++fail; }
            if (worst_row > 0.35) { printf("  FAIL: worst row error %.4f > 0.35\n", worst_row); ++fail; }
            double diff = 0.0;
            for (size_t i = 0; i < on.size(); ++i) diff = std::fmax(diff, std::fabs((double) on[i] - off[i]));
            if (diff == 0.0) { printf("  FAIL: ON result is bit-identical to OFF, so the FP4 path did not produce it\n"); ++fail; }
        } else if (h2 != h1) {
            printf("  FAIL: batch below the threshold was handled by the FP4 path\n"); ++fail;
        }
    }

    // ---- repacked-weight cache + invalidation (row S69-FASTLLM-PICK-NVFP4-PREFILL-REPACK-CACHE-AUTOTUNE-2026-10-27-01) ----
    // The loop above ran with the weight buffer in the default (ANY) usage, which must NOT populate the cache. From here the
    // buffer is declared WEIGHTS, as the model loader does, so the repack is kept after the first call.
    {
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const int Mc = 200;
        std::vector<float> xh((size_t) K * Mc);
        for (int r = 0; r < Mc; ++r) {
            const float scale = std::pow(10.0f, (float) (r % 5) - 2.0f);
            for (int k = 0; k < K; ++k) xh[(size_t) r * K + k] = gauss() * scale;
        }
        auto make_ref = [&](const std::vector<float> & wdq) {
            std::vector<double> ref((size_t) N * Mc);
            for (int r = 0; r < Mc; ++r)
                for (int n = 0; n < N; ++n) {
                    double acc = 0.0;
                    for (int k = 0; k < K; ++k) acc += (double) xh[(size_t) r * K + k] * (double) wdq[(size_t) n * K + k];
                    ref[(size_t) r * N + n] = acc;
                }
            return ref;
        };
        // nRMS error of column n (all Mc rows) against a reference; n < 0 means every column
        auto col_nrms = [&](const std::vector<float> & got, const std::vector<double> & ref, int n) {
            double num = 0.0, den = 0.0;
            for (int r = 0; r < Mc; ++r)
                for (int c = 0; c < N; ++c) {
                    if (n >= 0 && c != n) continue;
                    const double dd = got[(size_t) r * N + c] - ref[(size_t) r * N + c];
                    num += dd * dd; den += ref[(size_t) r * N + c] * ref[(size_t) r * N + c];
                }
            return std::sqrt(num / den);
        };
        auto max_abs_diff = [](const std::vector<float> & x, const std::vector<float> & y) {
            double m = 0.0;
            for (size_t i = 0; i < x.size(); ++i) m = std::fmax(m, std::fabs((double) x[i] - y[i]));
            return m;
        };
        struct cs { unsigned long long hits, repacks, tuned; };
        auto cstats = []() { cs s; ggml_backend_cuda_nvfp4_prefill_cache_stats(&s.hits, &s.repacks, &s.tuned); return s; };

        const std::vector<double> ref1 = make_ref(wd);
        set_env("GGML_CUDA_NVFP4_PREFILL", "1");
        std::vector<float> a, b;
        const cs c0 = cstats();
        if (!run_graph(backend, w, xh, K, Mc, N, a)) { printf("FAIL: cache call 1 failed\n"); return 1; }
        const cs c1 = cstats();
        if (!run_graph(backend, w, xh, K, Mc, N, b)) { printf("FAIL: cache call 2 failed\n"); return 1; }
        const cs c2 = cstats();
        printf("cache: call1 repacks +%llu hits +%llu | call2 repacks +%llu hits +%llu | call1 vs call2 max|diff| %.3g (tuned plans so far %llu)\n",
               c1.repacks - c0.repacks, c1.hits - c0.hits, c2.repacks - c1.repacks, c2.hits - c1.hits, max_abs_diff(a, b), c2.tuned);
        if (c1.repacks - c0.repacks != 1) { printf("  FAIL: first call on a WEIGHTS buffer must repack exactly once\n"); ++fail; }
        if (c2.repacks != c1.repacks || c2.hits - c1.hits != 1) {
            printf("  FAIL: second call must reuse the cached repack (want repacks +0 hits +1)\n"); ++fail;
        }
        if (max_abs_diff(a, b) != 0.0) { printf("  FAIL: cached call differs from the call that filled the cache\n"); ++fail; }
        if (col_nrms(b, ref1, -1) > 0.2) { printf("  FAIL: cached result nRMS %.5f > 0.2\n", col_nrms(b, ref1, -1)); ++fail; }

        // overwrite the whole weight with new contents: the cached repack is stale and must be dropped
        std::vector<float> wf2((size_t) K * N);
        for (auto & v : wf2) v = gauss() * 0.5f;
        std::vector<uint8_t> wq2(row_bytes * N);
        ggml_quantize_chunk(GGML_TYPE_NVFP4, wf2.data(), wq2.data(), 0, N, K, nullptr, nullptr);
        std::vector<float> wd2((size_t) K * N);
        for (int r = 0; r < N; ++r) tr->to_float(wq2.data() + row_bytes * r, wd2.data() + (size_t) K * r, K);
        const std::vector<double> ref2 = make_ref(wd2);
        ggml_backend_tensor_set(w, wq2.data(), 0, wq2.size());
        std::vector<float> d;
        const cs c3 = cstats();
        if (!run_graph(backend, w, xh, K, Mc, N, d)) { printf("FAIL: call after tensor_set failed\n"); return 1; }
        const cs c4 = cstats();
        printf("cache after full tensor_set: repacks +%llu hits +%llu | nRMS vs NEW weights %.5f, vs OLD weights %.5f\n",
               c4.repacks - c3.repacks, c4.hits - c3.hits, col_nrms(d, ref2, -1), col_nrms(d, ref1, -1));
        if (c4.repacks - c3.repacks != 1 || c4.hits != c3.hits) { printf("  FAIL: write to the weight must invalidate the cached repack\n"); ++fail; }
        if (col_nrms(d, ref2, -1) > 0.2) { printf("  FAIL: result after tensor_set does not match the NEW weights\n"); ++fail; }
        if (col_nrms(d, ref1, -1) < 0.5) { printf("  FAIL: result after tensor_set still follows the OLD weights (stale cache)\n"); ++fail; }

        // overwrite only row 0 (output column 0) with the original row 0: a partial write inside the cached range
        ggml_backend_tensor_set(w, wq.data(), 0, row_bytes);
        std::vector<float> wd3 = wd2;
        std::memcpy(wd3.data(), wd.data(), sizeof(float) * K);
        const std::vector<double> ref3 = make_ref(wd3);
        std::vector<float> e;
        const cs c5 = cstats();
        if (!run_graph(backend, w, xh, K, Mc, N, e)) { printf("FAIL: call after partial tensor_set failed\n"); return 1; }
        const cs c6 = cstats();
        printf("cache after row-0 tensor_set: repacks +%llu | column 0 nRMS vs row-0-new %.5f, vs row-0-stale %.5f\n",
               c6.repacks - c5.repacks, col_nrms(e, ref3, 0), col_nrms(e, ref2, 0));
        if (c6.repacks - c5.repacks != 1) { printf("  FAIL: a partial write inside the weight must invalidate the cached repack\n"); ++fail; }
        if (col_nrms(e, ref3, 0) > 0.35) { printf("  FAIL: column 0 does not match the rewritten row\n"); ++fail; }
        if (col_nrms(e, ref2, 0) < 0.6)  { printf("  FAIL: column 0 still follows the stale repack\n"); ++fail; }

        // control arm: the cache budget set to 0 (DISABLE-REVERSIBLE: GGML_CUDA_NVFP4_PREFILL_CACHE_MB, restored right after)
        // -> every call repacks, none hits, and the output bits are the same as the cached run
        const char * keep = getenv("GGML_CUDA_NVFP4_PREFILL_CACHE_MB");
        const std::string keep_s = keep ? keep : "";
        set_env("GGML_CUDA_NVFP4_PREFILL_CACHE_MB", "0");
        std::vector<float> f1, f2;
        const cs c7 = cstats();
        if (!run_graph(backend, w, xh, K, Mc, N, f1) || !run_graph(backend, w, xh, K, Mc, N, f2)) { printf("FAIL: budget-0 calls failed\n"); return 1; }
        const cs c8 = cstats();
        set_env("GGML_CUDA_NVFP4_PREFILL_CACHE_MB", keep_s.c_str());
        printf("budget 0: repacks +%llu hits +%llu | cached-vs-uncached max|diff| %.3g\n",
               c8.repacks - c7.repacks, c8.hits - c7.hits, max_abs_diff(e, f1));
        if (c8.repacks - c7.repacks != 2 || c8.hits != c7.hits) { printf("  FAIL: with budget 0 every call must repack and none may hit\n"); ++fail; }
        if (max_abs_diff(e, f1) != 0.0 || max_abs_diff(f1, f2) != 0.0) { printf("  FAIL: cache on/off changed the output bits\n"); ++fail; }
    }

    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    printf("%s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
