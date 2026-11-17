// GGML_OP_GATED_DELTA_NET on the CPU backend against a plain double-precision reference of the recurrence.
// COMPATIBILITY-INVESTIGATED: pure numerics on the CPU backend; no device.   GATE-CLAIM-OK: documentation only.
//
// Row: S69-EXECUTORCH-PICK-CPU-GDN-TWO-PASS-SWEEP-2026-10-05-01. The CPU kernel used to sweep the S_v x S_v state four
// times per token (scale, dot, rank-1 update, dot); it now sweeps it twice (read-only dot with the decay folded in, then one
// fused update + read-out). test-backend-ops can not check that on a CPU-only build (its reference IS the CPU backend), so this
// test carries its own reference:
//     M[j][i] = state[(b*H+h)*S_v*S_v + j*S_v + i]      (state stored transposed: row j = column j of S)
//     per token:  M *= exp(g)   (scalar g, or per-i for the KDA gate)
//                 delta[j] = (v[j] - sum_i M[j][i]*k[i]) * beta
//                 M[j][i] += delta[j]*k[i]
//                 out[j] = sum_i M[j][i]*q[i] / sqrt(S_v)
// Cases cover the scalar and the KDA gate, K = 1 and K = 3 snapshot slots, several seqs/heads/tokens and S_v values on both
// sides of the SIMD step (16, 48 leave a scalar tail; 128 is the real size). A MUST-FAIL partner perturbs the reference (drops the
// decay) and asserts the comparison then reports a difference, so the tolerance can not be vacuous.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

struct gdn_case { int64_t S, H, T, B, K; bool kda; };

// reference, double precision. drop_decay is the mutation used by the must-fail partner.
static void reference(const gdn_case & c, const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
        const std::vector<float> & g, const std::vector<float> & beta, const std::vector<float> & state0,
        std::vector<double> & out_attn, std::vector<std::vector<double>> & out_state, bool drop_decay) {
    const int64_t S = c.S, H = c.H, T = c.T, B = c.B;
    out_attn.assign(S*H*T*B, 0.0);
    out_state.assign(c.K, std::vector<double>(S*S*H*B, 0.0));
    const double scale = 1.0 / std::sqrt((double) S);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            std::vector<double> M(S*S);
            for (int64_t n = 0; n < S*S; ++n) M[n] = state0[(b*H + h)*S*S + n];
            for (int64_t t = 0; t < T; ++t) {
                const float * qd = &q[((b*T + t)*H + h)*S];
                const float * kd = &k[((b*T + t)*H + h)*S];
                const float * vd = &v[((b*T + t)*H + h)*S];
                const float * gd = &g[((b*T + t)*H + h)*(c.kda ? S : 1)];
                const double bt  = beta[(b*T + t)*H + h];
                for (int64_t j = 0; j < S; ++j) {
                    for (int64_t i = 0; i < S; ++i) {
                        const double dec = drop_decay ? 1.0 : std::exp((double) gd[c.kda ? i : 0]);
                        M[j*S + i] *= dec;
                    }
                }
                std::vector<double> delta(S);
                for (int64_t j = 0; j < S; ++j) {
                    double sum = 0.0;
                    for (int64_t i = 0; i < S; ++i) sum += M[j*S + i] * kd[i];
                    delta[j] = (vd[j] - sum) * bt;
                }
                for (int64_t j = 0; j < S; ++j) {
                    for (int64_t i = 0; i < S; ++i) M[j*S + i] += delta[j] * kd[i];
                }
                for (int64_t j = 0; j < S; ++j) {
                    double sum = 0.0;
                    for (int64_t i = 0; i < S; ++i) sum += M[j*S + i] * qd[i];
                    out_attn[((b*T + t)*H + h)*S + j] = sum * scale;
                }
                const int64_t slot = T - 1 - t;
                if (slot < c.K) {
                    for (int64_t n = 0; n < S*S; ++n) out_state[slot][(b*H + h)*S*S + n] = M[n];
                }
            }
        }
    }
}

static void fill(std::vector<float> & x, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto & e : x) e = d(rng);
}

static void l2_rows(std::vector<float> & x, int64_t S) {
    for (size_t r = 0; r < x.size() / S; ++r) {
        double s = 0.0;
        for (int64_t i = 0; i < S; ++i) s += (double) x[r*S + i] * x[r*S + i];
        const float inv = (float) (1.0 / std::sqrt(s + 1e-12));
        for (int64_t i = 0; i < S; ++i) x[r*S + i] *= inv;
    }
}

struct result { double max_attn_err, max_state_err, max_ref; double us_per_call; };

static result run_case(const gdn_case & c, int n_threads, int timing_reps, bool mutate_reference) {
    std::mt19937 rng(1234 + (unsigned) (c.S * 7 + c.H * 13 + c.T * 17 + c.B * 19 + c.K * 23 + (c.kda ? 99 : 0)));
    const int64_t S = c.S, H = c.H, T = c.T, B = c.B;
    std::vector<float> q(S*H*T*B), k(S*H*T*B), v(S*H*T*B), g(c.kda ? S*H*T*B : H*T*B), beta(H*T*B), state0(S*S*H*B);
    fill(q, rng, -1, 1); fill(k, rng, -1, 1); fill(v, rng, -1, 1);
    l2_rows(q, S); l2_rows(k, S);
    fill(g, rng, -2.0f, -0.01f); fill(beta, rng, 0.0f, 1.0f); fill(state0, rng, -0.1f, 0.1f);

    ggml_init_params ip = { 512ull * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.kda ? S : 1, H, T, B);
    ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, B);
    ggml_tensor * out = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tb, ts, c.K);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, n_threads);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(tq, q.data(), 0, q.size()*4);
    ggml_backend_tensor_set(tk, k.data(), 0, k.size()*4);
    ggml_backend_tensor_set(tv, v.data(), 0, v.size()*4);
    ggml_backend_tensor_set(tg, g.data(), 0, g.size()*4);
    ggml_backend_tensor_set(tb, beta.data(), 0, beta.size()*4);
    ggml_backend_tensor_set(ts, state0.data(), 0, state0.size()*4);
    ggml_backend_graph_compute(be, gf);

    std::vector<float> got(ggml_nelements(out));
    ggml_backend_tensor_get(out, got.data(), 0, got.size()*4);

    double us = 0.0;
    if (timing_reps > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < timing_reps; ++r) ggml_backend_graph_compute(be, gf);
        us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / timing_reps;
    }

    std::vector<double> ra; std::vector<std::vector<double>> rs;
    reference(c, q, k, v, g, beta, state0, ra, rs, mutate_reference);
    result r{0, 0, 0, us};
    for (size_t n = 0; n < ra.size(); ++n) {
        r.max_attn_err = std::fmax(r.max_attn_err, std::fabs(ra[n] - got[n]));
        r.max_ref      = std::fmax(r.max_ref, std::fabs(ra[n]));
    }
    const size_t attn_n = ra.size();
    for (int64_t s = 0; s < c.K; ++s) {
        for (size_t n = 0; n < rs[s].size(); ++n) {
            r.max_state_err = std::fmax(r.max_state_err, std::fabs(rs[s][n] - got[attn_n + (size_t) s*rs[s].size() + n]));
            r.max_ref       = std::fmax(r.max_ref, std::fabs(rs[s][n]));
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    const bool perf = argc > 1 && strcmp(argv[1], "--perf") == 0;
    const gdn_case cases[] = {
        {16, 2, 1, 1, 1, false}, {16, 2, 5, 2, 3, false}, {48, 3, 4, 2, 1, true}, {48, 3, 4, 2, 3, true},
        {64, 4, 1, 2, 1, false}, {96, 2, 6, 1, 2, true}, {128, 4, 1, 1, 1, false}, {128, 4, 7, 2, 3, true}, {128, 2, 9, 1, 1, false},
    };
    for (const gdn_case & c : cases) {
        const result r = run_case(c, 3, 0, false);
        const double tol = 2e-4 * (1.0 + r.max_ref);
        printf("S=%3lld H=%lld T=%lld B=%lld K=%lld kda=%d  max|attn err| %.2e  max|state err| %.2e  (max|ref| %.2f, tol %.1e)\n",
               (long long) c.S, (long long) c.H, (long long) c.T, (long long) c.B, (long long) c.K, (int) c.kda, r.max_attn_err, r.max_state_err, r.max_ref, tol);
        CHECK(r.max_attn_err <= tol);
        CHECK(r.max_state_err <= tol);
        // must-fail partner: a reference that drops the decay must differ clearly from the kernel
        const result m = run_case(c, 3, 0, true);
        CHECK(m.max_attn_err > 20 * tol || m.max_state_err > 20 * tol);
    }
    if (perf) {
        // informational only (shared host): S_v=128, H=32, one thread, T=1 (decode) and T=128 (prefill chunk)
        for (int T : {1, 128}) {
            const gdn_case c = {128, 32, T, 1, 1, false};
            const result r = run_case(c, 1, T == 1 ? 400 : 20, false);
            printf("PERF S=128 H=32 T=%d 1 thread: %.1f us per graph\n", T, r.us_per_call);
        }
    }
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-gdn-cpu-ref: all checks passed\n");
    return 0;
}
