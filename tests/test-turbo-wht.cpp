// GGML_OP_TURBO_WHT on a GPU backend (default CUDA0), checked by PROPERTIES rather than against a CPU run:
// the CPU backend declines this op as GPU-only (ggml-cpu.cpp supports_op), so test-backend-ops -- which
// compares a backend against CPU -- skips it and has never covered it.
//
// The kernel (ggml/src/ggml-cuda/turbo-wht.cu) computes, per group of g = 32/64/128 floats,
//   forward  y = s2 * H(s1 * x) / sqrt(g)        inverse  x = s1 * H(s2 * y) / sqrt(g)
// with s1, s2 fixed +-1 sign vectors and H the Sylvester Hadamard matrix. H is symmetric and H*H = g*I, so
// the transform is orthonormal whatever the sign vectors are. That gives three checks that need no copy of
// the sign tables:
//   1. inverse(forward(x)) == x                            (round trip)
//   2. ||forward(x)|| == ||x|| per group                   (orthonormal)
//   3. forward(e_i) is +-1/sqrt(g) on every entry of i's group and exactly 0 elsewhere (a real Hadamard mix)
// plus two controls: forward(x) must DIFFER from x (an identity kernel passes 1 and 2), and InnerQ scale_inv
// applied on the forward side only must come back as x * scale_inv after an unscaled inverse.
// Written 2026-09-27 when the missing CUDA dispatch case for this op was restored (ggml-cuda.cu); before that
// restore the compute aborts on the unsupported op, which is this test's fail-before.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * what, double got, double limit) {
    printf("  %-4s %-58s %.3g (limit %.3g)\n", ok ? "ok" : "FAIL", what, got, limit);
    if (!ok) ++g_fail;
}

// y = wht(x, scale_fwd, 0, g); z = wht(y, scale_inv_side, 1, g). Either scale may be null.
static bool run(ggml_backend_t be, int g, int64_t head_dim, int64_t n_heads, const std::vector<float> & xs,
                const std::vector<float> * scale_fwd, const std::vector<float> * scale_inv,
                std::vector<float> & ys, std::vector<float> & zs, bool & supported) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, n_heads);
    ggml_tensor * sf = scale_fwd ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, g) : nullptr;
    ggml_tensor * si = scale_inv ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, g) : nullptr;
    ggml_tensor * y  = ggml_turbo_wht(ctx, x, sf, 0, g);
    ggml_tensor * z  = ggml_turbo_wht(ctx, y, si, 1, g);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, z);

    supported = ggml_backend_supports_op(be, y) && ggml_backend_supports_op(be, z);
    if (!supported) { ggml_free(ctx); return false; }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(x, xs.data(), 0, xs.size()*sizeof(float));
    if (sf) ggml_backend_tensor_set(sf, scale_fwd->data(), 0, g*sizeof(float));
    if (si) ggml_backend_tensor_set(si, scale_inv->data(), 0, g*sizeof(float));
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    ys.assign(xs.size(), 0.0f);
    zs.assign(xs.size(), 0.0f);
    if (ok) {
        ggml_backend_tensor_get(y, ys.data(), 0, ys.size()*sizeof(float));
        ggml_backend_tensor_get(z, zs.data(), 0, zs.size()*sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// One WHT (group 128) over `xs` placed `float_off` floats into a buffer tensor. float_off 0 gives a 16-byte aligned src
// (the warp-shuffle kernel takes it); float_off 1 puts src 4 bytes off 16-byte alignment, which the dispatcher refuses for
// the fast kernel, so the SAME maths runs through the shared-memory kernel. out receives the result.
static bool run_placed(ggml_backend_t be, int direction, const std::vector<float> * scale, int float_off, int64_t head_dim, int64_t n_heads,
                       const std::vector<float> & xs, std::vector<float> & out) {
    const int g = 128;
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) xs.size() + 4);
    ggml_tensor * x    = ggml_view_2d(ctx, base, head_dim, n_heads, head_dim*sizeof(float), (size_t) float_off*sizeof(float));
    ggml_tensor * sc   = scale ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, g) : nullptr;
    ggml_tensor * y    = ggml_turbo_wht(ctx, x, sc, direction, g);
    ggml_cgraph * gf   = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(base, xs.data(), (size_t) float_off*sizeof(float), xs.size()*sizeof(float));
    if (sc) ggml_backend_tensor_set(sc, scale->data(), 0, g*sizeof(float));
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    out.assign(xs.size(), 0.0f);
    if (ok) ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// The warp-shuffle kernel (group 128) must be BIT-IDENTICAL to the shared-memory kernel, both directions, with and
// without the InnerQ scale. Control: the comparison must notice a change in one input element.
static void check_fast_vs_generic(ggml_backend_t be, std::mt19937 & rng) {
    const int64_t head_dim = 256, n_heads = 6;
    std::uniform_real_distribution<float> U(-3.0f, 3.0f);
    std::vector<float> xs(head_dim*n_heads), sc(128);
    for (float & v : xs) v = U(rng);
    xs[17] = 0.0f;   // a zero: the sign flip must give -0.0 exactly where the multiply by -1.0f does
    for (float & v : sc) v = 0.5f + (U(rng) + 3.0f) / 3.0f;
    printf("fast (warp-shuffle) vs generic (shared-memory) kernel, group_size 128, bit for bit\n");
    for (int direction = 0; direction < 2; ++direction) {
        for (int with_scale = 0; with_scale < 2; ++with_scale) {
            std::vector<float> fast, gen, gen_perturbed;
            const std::vector<float> * s = with_scale ? &sc : nullptr;
            const bool ok = run_placed(be, direction, s, 0, head_dim, n_heads, xs, fast) && run_placed(be, direction, s, 1, head_dim, n_heads, xs, gen);
            char what[96];
            snprintf(what, sizeof(what), "dir %d %-8s fast == generic bitwise (differing floats)", direction, with_scale ? "InnerQ" : "no scale");
            size_t diff = ok ? 0 : xs.size();
            for (size_t i = 0; ok && i < xs.size(); ++i) diff += std::memcmp(&fast[i], &gen[i], sizeof(float)) != 0;
            check(ok && diff == 0, what, (double) diff, 0.0);

            std::vector<float> xp = xs;
            xp[200] += 1.0f;   // a real change in one element (a 1-ULP change can vanish below the output rounding after the 1/sqrt(128) scale)
            const bool okp = run_placed(be, direction, s, 1, head_dim, n_heads, xp, gen_perturbed);
            size_t dp = 0;
            for (size_t i = 0; okp && i < xs.size(); ++i) dp += std::memcmp(&fast[i], &gen_perturbed[i], sizeof(float)) != 0;
            snprintf(what, sizeof(what), "CONTROL dir %d %-8s a changed input element IS seen (floats that differ, must be > 0)", direction, with_scale ? "InnerQ" : "no scale");
            check(okp && dp > 0, what, (double) dp, 1.0);
        }
    }
}

// Opt-in timing (env TURBO_WHT_TIME=1; not part of the default pass/fail run): the SAME op through the warp-shuffle kernel
// (float_off 0, 16-byte aligned) and through the shared-memory kernel (float_off 1, forced by misalignment), interleaved
// runs, median of 7 timed blocks of `iters` launches each, backend synchronised around every block.
static double time_placed(ggml_backend_t be, int float_off, int64_t head_dim, int64_t n_heads, int iters) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim*n_heads + 4);
    ggml_tensor * x    = ggml_view_2d(ctx, base, head_dim, n_heads, head_dim*sizeof(float), (size_t) float_off*sizeof(float));
    ggml_tensor * y    = ggml_turbo_wht(ctx, x, nullptr, 0, 128);
    ggml_cgraph * gf   = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<float> xs((size_t) (head_dim*n_heads), 1.0f);
    ggml_backend_tensor_set(base, xs.data(), (size_t) float_off*sizeof(float), xs.size()*sizeof(float));
    for (int i = 0; i < 5; ++i) ggml_backend_graph_compute(be, gf);   // warm-up
    ggml_backend_synchronize(be);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ms;
}

static void time_fast_vs_generic(ggml_backend_t be) {
    const int64_t head_dim = 128;
    const int64_t shapes[3] = { 8*512, 32*4096, 32*32768 };   // heads x tokens: decode-ish, prefill-ish, long prefill
    printf("timing (ms per launch, median of 7 blocks, fast = warp-shuffle, generic = shared-memory):\n");
    for (int64_t n_heads : shapes) {
        const int iters = n_heads >= 32*32768 ? 50 : 200;
        std::vector<double> f, g;
        for (int r = 0; r < 7; ++r) { f.push_back(time_placed(be, 0, head_dim, n_heads, iters)); g.push_back(time_placed(be, 1, head_dim, n_heads, iters)); }
        std::sort(f.begin(), f.end()); std::sort(g.begin(), g.end());
        printf("  n_groups %9lld  fast %.4f ms  generic %.4f ms  ratio generic/fast %.2f  (fast min..max %.4f..%.4f, generic %.4f..%.4f)\n",
               (long long) n_heads, f[3], g[3], g[3]/f[3], f[0], f[6], g[0], g[6]);
    }
}

int main(int argc, char ** argv) {
    const char * be_name = argc > 1 ? argv[1] : "CUDA0";
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name(be_name, nullptr);
    if (!be) {
        printf("SKIPPED: backend %s could not be initialised in this build. Not a verdict.\n", be_name);
        return 0;
    }
    printf("backend %s\n", be_name);
    std::mt19937 rng(20260927);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);

    const int64_t head_dim = 256, n_heads = 6;   // 256 is a multiple of every group size: no tail
    for (int g : { 32, 64, 128 }) {
        printf("group_size %d  (tensor %lld x %lld)\n", g, (long long) head_dim, (long long) n_heads);
        std::vector<float> xs(head_dim*n_heads), ys, zs;
        for (float & v : xs) v = U(rng);

        bool supported = false;
        if (!run(be, g, head_dim, n_heads, xs, nullptr, nullptr, ys, zs, supported)) {
            printf("  FAIL %s\n", supported ? "compute failed" : "backend does not support GGML_OP_TURBO_WHT");
            ++g_fail;
            continue;
        }

        double max_rt = 0, max_norm = 0, max_move = 0;
        for (size_t i = 0; i < xs.size(); ++i) {
            max_rt   = std::max(max_rt,   (double) std::fabs(zs[i] - xs[i]));
            max_move = std::max(max_move, (double) std::fabs(ys[i] - xs[i]));
        }
        for (size_t base = 0; base < xs.size(); base += g) {
            double nx = 0, ny = 0;
            for (int k = 0; k < g; ++k) { nx += xs[base+k]*(double) xs[base+k]; ny += ys[base+k]*(double) ys[base+k]; }
            max_norm = std::max(max_norm, std::fabs(std::sqrt(ny) - std::sqrt(nx)) / std::sqrt(nx));
        }
        check(max_rt   <= 1e-5, "round trip max |inverse(forward(x)) - x|", max_rt, 1e-5);
        check(max_norm <= 1e-5, "orthonormal: max relative | ||y|| - ||x|| | per group", max_norm, 1e-5);
        check(max_move >= 0.1,  "CONTROL forward is not the identity: max |y - x| (must be LARGE)", max_move, 0.1);

        // basis vectors: one per head, at a different offset each time, so several groups are probed
        std::vector<float> es(head_dim*n_heads, 0.0f);
        std::vector<int64_t> hot(n_heads);
        for (int64_t h = 0; h < n_heads; ++h) { hot[h] = (h*37 + 5) % head_dim; es[h*head_dim + hot[h]] = 1.0f; }
        run(be, g, head_dim, n_heads, es, nullptr, nullptr, ys, zs, supported);
        const double mag = 1.0 / std::sqrt((double) g);
        double worst_in = 0, worst_out = 0;
        for (int64_t h = 0; h < n_heads; ++h) {
            const int64_t g0 = (hot[h] / g) * g;
            for (int64_t k = 0; k < head_dim; ++k) {
                const double v = ys[h*head_dim + k];
                if (k >= g0 && k < g0 + g) worst_in  = std::max(worst_in,  std::fabs(std::fabs(v) - mag));
                else                       worst_out = std::max(worst_out, std::fabs(v));
            }
        }
        check(worst_in  <= 1e-6, "basis: inside its group every |y| == 1/sqrt(g), max deviation", worst_in, 1e-6);
        check(worst_out == 0.0,  "basis: outside its group y == 0 exactly, max |y|", worst_out, 0.0);

        // InnerQ: scale on the forward side only -> an unscaled inverse returns x * scale
        std::vector<float> sc(g);
        for (float & v : sc) v = 0.5f + (U(rng) + 1.0f);   // 0.5 .. 2.5, never ~1, so a dropped scale shows
        run(be, g, head_dim, n_heads, xs, &sc, nullptr, ys, zs, supported);
        double max_sc = 0;
        for (size_t i = 0; i < xs.size(); ++i) max_sc = std::max(max_sc, (double) std::fabs(zs[i] - xs[i]*sc[i % g]));
        check(max_sc <= 1e-5, "InnerQ forward scale_inv: inverse(forward_s(x)) == x*s", max_sc, 1e-5);
    }

    check_fast_vs_generic(be, rng);
    if (const char * tm = std::getenv("TURBO_WHT_TIME")) { if (tm[0] == '1') time_fast_vs_generic(be); }

    ggml_backend_free(be);
    printf("%s: %d check(s) failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
