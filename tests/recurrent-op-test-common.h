// Shared scaffolding for test-gated-linear-attn / test-gated-delta-net / test-rwkv-wkv6 / test-rwkv-wkv7:
// build a one-op ggml graph on the CPU backend, run it, and slice token-indexed inputs for the state-carry check.
// No model file and no device: inputs are random.
#pragma once
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

struct tin {
    std::vector<float> d;
    int64_t ne[4] = {1, 1, 1, 1};
    int64_t row   = 1;     // floats per token (token-indexed inputs)
    int     tokdim = 0;    // 0: not token-indexed; 2: ne[2] = n_seqs*T; 3: ne[2] = T, ne[3] = n_seqs
};

static tin make_in(std::mt19937 & rng, float lo, float hi, int64_t n0, int64_t n1, int64_t n2, int64_t n3, int tokdim) {
    tin t;
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = n3;
    t.tokdim = tokdim;
    t.row = tokdim ? n0 * n1 : 0;
    t.d.resize(n0 * n1 * n2 * n3);
    std::uniform_real_distribution<float> u(lo, hi);
    for (auto & x : t.d) x = u(rng);
    return t;
}

// tokens [t0, t1) of every sequence; layout is sequence-major, t within a sequence
static tin slice_tokens(const tin & in, int64_t n_seqs, int64_t T, int64_t t0, int64_t t1) {
    if (!in.tokdim) return in;
    tin o = in;
    const int64_t n = t1 - t0;
    o.d.clear();
    for (int64_t s = 0; s < n_seqs; ++s)
        for (int64_t t = t0; t < t1; ++t)
            o.d.insert(o.d.end(), in.d.begin() + (s * T + t) * in.row, in.d.begin() + (s * T + t + 1) * in.row);
    if (in.tokdim == 2) o.ne[2] = n_seqs * n; else o.ne[2] = n;
    return o;
}

using build_fn = std::function<ggml_tensor *(ggml_context *, std::vector<ggml_tensor *> &)>;

static std::vector<float> run_graph(const std::vector<tin> & in, int n_threads, const build_fn & build) {
    ggml_init_params ip = { 4ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<ggml_tensor *> t;
    for (const tin & x : in) t.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, x.ne[0], x.ne[1], x.ne[2], x.ne[3]));
    ggml_tensor * out = build(ctx, t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, n_threads);
    CHECK(ggml_backend_supports_op(be, out));  // the CPU backend must claim the op it is about to run
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (size_t i = 0; i < in.size(); ++i) ggml_backend_tensor_set(t[i], in[i].d.data(), 0, in[i].d.size() * 4);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return got;
}

static double max_abs_diff(const double * a, const float * b, size_t n, double & max_ref) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
        m = std::fmax(m, std::fabs(a[i] - (double) b[i]));
        max_ref = std::fmax(max_ref, std::fabs(a[i]));
    }
    return m;
}
