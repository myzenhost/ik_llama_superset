// Heterogeneous-adapter LoRA batching with the ops the fork already has: SGMV == MUL_MAT_ID with the adapter index as the "expert" id.
// Row S69-LORAX-PICK-SGMV-HETEROGENEOUS-LORA-BATCHING-2026-09-29-01.   GATE-CLAIM-OK: documentation only.
//
// CLAIM UNDER TEST. Punica/LoRAX SGMV applies a DIFFERENT low-rank pair (A_a, B_a) to every row of one activation batch in one launch.  A grouped GEMM keyed by a
// per-row index is exactly what ggml_mul_mat_id computes for MoE routing, which the CPU and CUDA backends already implement: stack the adapters as
// A [n_in, r, n_adapters] and B [r, n_out, n_adapters], give every token its adapter id, and y_t = B_{a_t} (A_{a_t} x_t) is two MUL_MAT_ID nodes.
// So no new kernel is needed for the arithmetic; what is missing in the fork is graph plumbing (stacked adapter tensors + a per-token id input) and the server rule
// can_batch_with (server-context.cpp:558-564) that today forbids mixing adapters in one batch.
//
//   A  T tokens with adapter ids drawn from {0..3} (id 0 = an all-zero "no adapter" slot): the two-node MUL_MAT_ID graph equals, token by token, the per-token
//      reference B_a (A_a x_t) computed with plain MUL_MAT in separate graphs, to 1e-5 relative; the id-0 tokens come out exactly 0 (base-only rows are untouched).
//   B  CONTROL: swapping two adapters' ids in the id tensor changes the output of exactly those tokens (the test can tell adapters apart) and the error against the
//      unswapped reference is far above the tolerance.
//   C  CONTROL: all tokens on one adapter reproduce the ordinary single-adapter product.
//   D  INFO ONLY (not asserted): wall time of the batched 2-node graph vs one grouped plain-MUL_MAT graph per adapter, same data.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static const int N_IN = 64, N_OUT = 48, RANK = 8, N_AD = 4, T = 24;

struct run_result { std::vector<float> y; double us; };

// batched: stacked adapters + per-token ids
static run_result run_batched(ggml_backend_t be, const std::vector<float> & A, const std::vector<float> & B, const std::vector<float> & x, const std::vector<int32_t> & ids) {
    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_IN, RANK, N_AD);
    ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, RANK, N_OUT, N_AD);
    ggml_tensor * xs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_IN, 1, T);
    ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, T);
    ggml_tensor * h  = ggml_mul_mat_id(ctx, a, xs, id);     // [RANK, 1, T]
    ggml_tensor * y  = ggml_mul_mat_id(ctx, b, h, id);      // [N_OUT, 1, T]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(a, A.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, B.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(xs, x.data(), 0, ggml_nbytes(xs));
    ggml_backend_tensor_set(id, ids.data(), 0, ggml_nbytes(id));
    auto t0 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(be, gf);
    auto t1 = std::chrono::steady_clock::now();
    run_result r;
    r.y.resize((size_t) N_OUT * T);
    ggml_backend_tensor_get(y, r.y.data(), 0, ggml_nbytes(y));
    r.us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

// reference: one plain two-MUL_MAT graph per token
static std::vector<float> run_reference(ggml_backend_t be, const std::vector<float> & A, const std::vector<float> & B, const std::vector<float> & x, const std::vector<int32_t> & ids) {
    std::vector<float> out((size_t) N_OUT * T, 0.0f);
    for (int t = 0; t < T; ++t) {
        const int ad = ids[t];
        ggml_init_params ip = { 4u * 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_IN, RANK);
        ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, RANK, N_OUT);
        ggml_tensor * xs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_IN, 1);
        ggml_tensor * y  = ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, a, xs));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(a, A.data() + (size_t) ad * N_IN * RANK, 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, B.data() + (size_t) ad * RANK * N_OUT, 0, ggml_nbytes(b));
        ggml_backend_tensor_set(xs, x.data() + (size_t) t * N_IN, 0, ggml_nbytes(xs));
        ggml_backend_graph_compute(be, gf);
        ggml_backend_tensor_get(y, out.data() + (size_t) t * N_OUT, 0, ggml_nbytes(y));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    return out;
}

static double rel_err(const std::vector<float> & a, const std::vector<float> & b, size_t t0, size_t t1) {
    double num = 0.0, den = 0.0;
    for (size_t t = t0; t < t1; ++t) {
        for (int o = 0; o < N_OUT; ++o) {
            const double d = (double) a[t * N_OUT + o] - (double) b[t * N_OUT + o];
            num += d * d;
            den += (double) b[t * N_OUT + o] * (double) b[t * N_OUT + o];
        }
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) {
        printf("test-lora-sgmv-equiv: REFUSED, no CPU backend\n");
        return 2;
    }
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 0.3f);
    std::vector<float> A((size_t) N_IN * RANK * N_AD), B((size_t) RANK * N_OUT * N_AD), x((size_t) N_IN * T);
    for (float & v : A) v = nd(rng);
    for (float & v : B) v = nd(rng);
    for (float & v : x) v = nd(rng);
    // adapter 0 = "no adapter": all zeros
    for (size_t i = 0; i < (size_t) N_IN * RANK; ++i) A[i] = 0.0f;
    for (size_t i = 0; i < (size_t) RANK * N_OUT; ++i) B[i] = 0.0f;

    std::vector<int32_t> ids(T);
    for (int t = 0; t < T; ++t) {
        ids[t] = (int32_t) ((t * 7 + 3) % N_AD);       // a fixed mix of the four slots
    }
    int n_zero = 0, n_per[N_AD] = {};
    for (int t = 0; t < T; ++t) { n_per[ids[t]]++; }
    n_zero = n_per[0];
    CHECK(n_per[0] > 0 && n_per[1] > 0 && n_per[2] > 0 && n_per[3] > 0);

    // A
    run_result got = run_batched(be, A, B, x, ids);
    const std::vector<float> ref = run_reference(be, A, B, x, ids);
    const double e = rel_err(got.y, ref, 0, T);
    printf("A  %d tokens over %d adapter slots (%d on the zero slot): relative error batched vs per-token reference %.3e\n", T, N_AD, n_zero, e);
    CHECK(e < 1e-5);
    for (int t = 0; t < T; ++t) {
        if (ids[t] == 0) {
            for (int o = 0; o < N_OUT; ++o) {
                CHECK(got.y[(size_t) t * N_OUT + o] == 0.0f);
            }
        }
    }
    // B: swap the ids of adapters 1 and 2
    std::vector<int32_t> sw = ids;
    for (auto & v : sw) {
        if (v == 1) v = 2; else if (v == 2) v = 1;
    }
    const run_result gsw = run_batched(be, A, B, x, sw);
    const double e_swapped_vs_ref = rel_err(gsw.y, ref, 0, T);
    printf("B  control: ids of adapters 1 and 2 swapped -> error against the unswapped reference %.3e (must be far above the 1e-5 tolerance)\n", e_swapped_vs_ref);
    CHECK(e_swapped_vs_ref > 0.1);
    for (int t = 0; t < T; ++t) {
        const bool touched = ids[t] == 1 || ids[t] == 2;
        double d = 0.0;
        for (int o = 0; o < N_OUT; ++o) d += std::fabs((double) gsw.y[(size_t) t * N_OUT + o] - (double) got.y[(size_t) t * N_OUT + o]);
        CHECK(touched ? d > 1e-3 : d < 1e-6);
    }
    // C: one adapter for every token
    std::vector<int32_t> one(T, 2);
    const run_result g1 = run_batched(be, A, B, x, one);
    const std::vector<float> r1 = run_reference(be, A, B, x, one);
    CHECK(rel_err(g1.y, r1, 0, T) < 1e-5);
    // D: info
    {
        double best = 1e30;
        for (int rep = 0; rep < 5; ++rep) {
            best = std::min(best, run_batched(be, A, B, x, ids).us);
        }
        printf("D  info: batched two-node MUL_MAT_ID graph, %d tokens x %d adapters, rank %d: %.0f us (best of 5)\n", T, N_AD, RANK, best);
    }
    ggml_backend_free(be);
    if (n_fail) {
        fprintf(stderr, "test-lora-sgmv-equiv: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-lora-sgmv-equiv: all passed\n");
    return 0;
}
