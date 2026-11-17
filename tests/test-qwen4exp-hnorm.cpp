// qwen4exp MTP draft head: the trunk hand-over `h` is hc streams of n_embd, and hnorm must apply ONE RMS PER
// STREAM (upstream ggml-org/llama.cpp qwen4exp: hnorm is [n_embd, hc], "RMS per hc stream").
// Before 2026-10-07 the fork ran ggml_rms_norm over the whole flat [hc*n_embd] vector, so every stream was
// scaled by the RMS of all of them and the draft logits differed for any input.
//
// This calls the fork's own qwen4exp_grouped_rms (src/graphs/build_qwen4exp.cpp), which the MTP head now
// routes `h` through, on the CPU backend, and compares it with a hand-written per-stream reference.
// Control: the OLD flat composition must DIFFER from the reference on streams of unequal scale, so the test
// can tell the two apart.
// NOT covered: the call site itself (the reshape_3d feeding the helper) and a real checkpoint; no
// qwen4exp MTP fixture exists in the tree.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../src/llama-hparams.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

ggml_tensor * qwen4exp_grouped_rms(ggml_context * ctx0, const llama_hparams & hparams, ggml_tensor * x,
                                   ggml_tensor * w, int32_t hc_dim, int32_t nt);

static int g_fail = 0;
static void check(bool ok, const char * what, double got, double limit) {
    printf("  %-4s %-62s %.3g (limit %.3g)\n", ok ? "ok" : "FAIL", what, got, limit);
    if (!ok) ++g_fail;
}

// Not `mode_t`: that is a POSIX typedef (sys/types.h), pulled in on MinGW through <mutex> ->
// pthread.h, and g++ rejects `enum mode_t` there ("using typedef-name 'mode_t' after 'enum'").
// MSVC has no such typedef, which is why only the MinGW build failed (glm5n-1109, 2026-11-09).
enum hnorm_mode { NEW_PER_STREAM, OLD_FLAT };

static std::vector<float> run(ggml_backend_t be, hnorm_mode mode, int n_embd, int hc, int nt, float eps,
                              const std::vector<float> & xs, const std::vector<float> & gamma) {
    const int hc_dim = n_embd * hc;
    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, nt);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);
    ggml_tensor * y;
    if (mode == NEW_PER_STREAM) {
        llama_hparams hp;
        hp.f_norm_rms_eps = eps;
        y = qwen4exp_grouped_rms(ctx, hp, ggml_reshape_3d(ctx, x, n_embd, hc, nt), w, hc_dim, nt);
    } else {
        y = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);   // the pre-fix expression
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(x, xs.data(), 0, xs.size()*sizeof(float));
    ggml_backend_tensor_set(w, gamma.data(), 0, gamma.size()*sizeof(float));
    std::vector<float> out(xs.size(), 0.0f);
    if (ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// hand reference: out[t][s][i] = x[t][s][i] / sqrt(mean_i(x[t][s][:]^2) + eps) * gamma[s*n_embd + i]
static std::vector<float> reference(int n_embd, int hc, int nt, float eps, const std::vector<float> & xs,
                                    const std::vector<float> & gamma) {
    std::vector<float> out(xs.size());
    for (int t = 0; t < nt; ++t) {
        for (int s = 0; s < hc; ++s) {
            const float * p = xs.data() + ((size_t) t*hc + s)*n_embd;
            double ss = 0;
            for (int i = 0; i < n_embd; ++i) ss += (double) p[i]*p[i];
            const double inv = 1.0/std::sqrt(ss/n_embd + eps);
            for (int i = 0; i < n_embd; ++i) {
                out[((size_t) t*hc + s)*n_embd + i] = (float)(p[i]*inv*gamma[(size_t) s*n_embd + i]);
            }
        }
    }
    return out;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs((double) a[i] - b[i]));
    return m;
}

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    const int n_embd = 64, hc = 4, nt = 3;
    const float eps = 1e-6f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> gamma((size_t) n_embd*hc);
    for (auto & g : gamma) g = 0.5f + std::fabs(nd(rng));

    // streams of clearly unequal scale: 0.1, 1, 5, 20
    const float scale[4] = { 0.1f, 1.0f, 5.0f, 20.0f };
    std::vector<float> xu((size_t) n_embd*hc*nt), xe(xu.size());
    for (int t = 0; t < nt; ++t)
        for (int s = 0; s < hc; ++s)
            for (int i = 0; i < n_embd; ++i) {
                const float v = nd(rng);
                xu[((size_t) t*hc + s)*n_embd + i] = v*scale[s];
                xe[((size_t) t*hc + s)*n_embd + i] = v;
            }

    printf("qwen4exp MTP hnorm: per-stream RMS (CPU)\n");
    const auto ref_u = reference(n_embd, hc, nt, eps, xu, gamma);
    const auto new_u = run(be, NEW_PER_STREAM, n_embd, hc, nt, eps, xu, gamma);
    const auto old_u = run(be, OLD_FLAT,       n_embd, hc, nt, eps, xu, gamma);
    check(max_abs_diff(new_u, ref_u) < 1e-4, "fork helper == per-stream reference (unequal stream scales)", max_abs_diff(new_u, ref_u), 1e-4);
    check(max_abs_diff(old_u, ref_u) > 0.1,  "CONTROL old flat norm differs from reference (unequal scales)", max_abs_diff(old_u, ref_u), -0.1);

    const auto ref_e = reference(n_embd, hc, nt, eps, xe, gamma);
    const auto new_e = run(be, NEW_PER_STREAM, n_embd, hc, nt, eps, xe, gamma);
    check(max_abs_diff(new_e, ref_e) < 1e-4, "fork helper == reference (equal-scale streams)", max_abs_diff(new_e, ref_e), 1e-4);

    ggml_backend_free(be);
    printf(g_fail ? "FAILED (%d)\n" : "All tests passed!\n", g_fail);
    return g_fail ? 1 : 0;
}
