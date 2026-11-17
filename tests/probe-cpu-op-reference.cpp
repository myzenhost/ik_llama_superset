// Independent numeric check for the 2026-09-25 ik_llama_fork CPU changes (commit c9c758f31).
// Not in CMake on purpose -- it links the static ggml archives directly. Build (mingw, build-static):
//   g++ -O1 -std=c++17 -fopenmp -I ggml/include tests/probe-cpu-op-reference.cpp -o probe.exe
//       -Wl,--start-group build-static/ggml/src/ggml.a build-static/ggml/src/ggml-cpu.a
//       build-static/ggml/src/ggml-base.a -Wl,--end-group -fopenmp -lstdc++
// Expected: "25 checks, 0 failed", rc 0 (16 before the 2026-09-26 softcap / mask_to_idx ports).
// test-backend-ops -b CPU compares CPU against the CPU reference, so it proves execution, not
// numbers. Here every op is checked against a scalar reference written from the FORMULA, and each
// check has a control that MUST fail -- a probe that can only say "match" measures nothing.
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_checks = 0, g_fails = 0;

static void check(const char * label, double err, double tol, bool expect_match) {
    ++g_checks;
    const bool match = err <= tol;
    const bool ok = match == expect_match;
    if (!ok) ++g_fails;
    printf("  %-4s %-66s max|err| %.3e  [%s]\n", ok ? "ok" : "FAIL", label, err,
           expect_match ? "must MATCH" : "control: must DIFFER");
}

static ggml_context * mk() {
    ggml_init_params p = { 256u * 1024 * 1024, nullptr, false };
    return ggml_init(p);
}

static void run(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
}

static std::mt19937 rng(1234);
static void fill(ggml_tensor * t, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = d(rng);
}

// ---------------------------------------------------------------- rope window offset
// corrupt_slot15 < 0: leave what ggml_rope_set_offset wrote; >= 0: overwrite slot 15 with it.
static double rope_err(int mode, int n_dims, int off, int corrupt_slot15) {
    const int ne0 = 128, nh = 4, nt = 3;
    const float base = 10000.0f;
    ggml_context * ctx = mk();
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, nh, nt);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nt);
    fill(x, -1.0f, 1.0f);
    const int32_t P[3] = { 0, 7, 1000 };
    memcpy(pos->data, P, sizeof(P));

    ggml_tensor * r = ggml_rope_ext(ctx, x, pos, nullptr, n_dims, mode, 4096, base, 1.0f,
                                    0.0f, 1.0f, 32.0f, 1.0f);
    if (off) r = ggml_rope_set_offset(r, off);
    if (corrupt_slot15 >= 0) ((int32_t *) r->op_params)[15] = corrupt_slot15;
    run(ctx, r);

    double err = 0.0;
    const float * X = (const float *) x->data;
    const float * Y = (const float *) r->data;
    for (int t = 0; t < nt; ++t) {
        for (int h = 0; h < nh; ++h) {
            const float * xr = X + (t * nh + h) * ne0;
            const float * yr = Y + (t * nh + h) * ne0;
            std::vector<double> ref(xr, xr + ne0);
            const int half = n_dims / 2;
            for (int i = 0; i < half; ++i) {
                const double theta = P[t] * pow((double) base, -2.0 * i / n_dims);
                const int a = mode == GGML_ROPE_TYPE_NEOX ? off + i : off + 2 * i;
                const int b = mode == GGML_ROPE_TYPE_NEOX ? off + i + half : off + 2 * i + 1;
                ref[a] = xr[a] * cos(theta) - xr[b] * sin(theta);
                ref[b] = xr[a] * sin(theta) + xr[b] * cos(theta);
            }
            for (int i = 0; i < ne0; ++i) err = fmax(err, fabs(ref[i] - yr[i]));
        }
    }
    ggml_free(ctx);
    return err;
}

// ---------------------------------------------------------------- sqrt_softplus
// ref: 0 = the SPEC formula in float, exactly as ggml-cuda/unary.cu op_sqrt_softplus and the
//          pre-split ggml.c define it: sqrtf(x > 20 ? x : logf(1.0f + expf(x)))
//      1 = control: plain softplus, must differ
//      2 = EXACT math, sqrt(log1p(exp(x))) in double -- measures the spec's own precision
static double sqrt_softplus_err(int ref_kind) {
    ggml_context * ctx = mk();
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 9);
    fill(x, -30.0f, 30.0f);
    ggml_tensor * y = ggml_sqrt_softplus(ctx, x);
    run(ctx, y);
    double err = 0.0;
    for (int64_t i = 0; i < ggml_nelements(x); ++i) {
        const float v = ((float *) x->data)[i];
        double ref;
        if (ref_kind == 0)      ref = sqrtf(v > 20.0f ? v : logf(1.0f + expf(v)));
        else if (ref_kind == 1) ref = v > 20.0f ? v : logf(1.0f + expf(v));
        else                    ref = sqrt(v > 20.0 ? (double) v : log1p(exp((double) v)));
        err = fmax(err, fabs(ref - ((float *) y->data)[i]) / fmax(1.0, fabs(ref)));
    }
    ggml_free(ctx);
    return err;
}

// ---------------------------------------------------------------- unary swiglu [x | g]
static double swiglu_err(bool control) {
    const int nc = 37, rows = 5;
    ggml_context * ctx = mk();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * nc, rows);
    fill(a, -6.0f, 6.0f);
    ggml_tensor * y = ggml_swiglu(ctx, a);
    run(ctx, y);
    double err = 0.0;
    for (int r = 0; r < rows; ++r) {
        const float * A = (const float *) a->data + r * 2 * nc;
        for (int i = 0; i < nc; ++i) {
            const double x = control ? A[i + nc] : A[i];   // control: halves swapped
            const double g = control ? A[i] : A[i + nc];
            const double ref = x / (1.0 + exp(-x)) * g;
            err = fmax(err, fabs(ref - ((float *) y->data)[r * nc + i]) / fmax(1.0, fabs(ref)));
        }
    }
    ggml_free(ctx);
    return err;
}

// ---------------------------------------------------------------- col2im_1d
static double col2im_err(int p0, int ref_p0) {
    const int K = 4, OC = 3, T_in = 11, s0 = 2;
    ggml_context * ctx = mk();
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K * OC, T_in);
    fill(a, -1.0f, 1.0f);
    ggml_tensor * y = ggml_col2im_1d(ctx, a, s0, OC, p0);
    run(ctx, y);
    const int T_out = (int) y->ne[0];
    const float * A = (const float *) a->data;
    double err = 0.0;
    for (int oc = 0; oc < OC; ++oc) {
        for (int to = 0; to < T_out; ++to) {
            double s = 0.0;
            for (int ti = 0; ti < T_in; ++ti)
                for (int k = 0; k < K; ++k)
                    if (ti * s0 + k == to + ref_p0) s += A[(oc * K + k) + ti * K * OC];
            err = fmax(err, fabs(s - ((float *) y->data)[to + oc * T_out]));
        }
    }
    ggml_free(ctx);
    return err;
}

// ---------------------------------------------------------------- softcap (ported 2026-09-26)
// ref_kind 0 = s_after * tanh(s_before * x); 1 = control: s_after * s_before * x (no tanh)
static double softcap_err(int ref_kind, bool inplace) {
    const float sb = 0.37f, sa = 30.0f;
    ggml_context * ctx = mk();
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 67, 7);   // 67: exercises the SIMD tail
    fill(x, -300.0f, 300.0f);   // |sb*x| up to ~111: past the e^2x overflow the old AVX-512/SSE path hit
    std::vector<float> X((float *) x->data, (float *) x->data + ggml_nelements(x));
    ggml_tensor * y = inplace ? ggml_softcap_inplace(ctx, x, sb, sa) : ggml_softcap(ctx, x, sb, sa);
    run(ctx, y);
    double err = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        const double ref = ref_kind == 0 ? sa * tanh((double) sb * X[i]) : (double) sa * sb * X[i];
        const float got = ((float *) y->data)[i];
        if (!std::isfinite(got)) return INFINITY;
        err = fmax(err, fabs(ref - got) / fmax(1.0, fabs(ref)));
    }
    ggml_free(ctx);
    return err;
}

// softcap_max: softmax_row( scale*s_after*tanh(s_before*x) + mask ), max_bias = 0
// control: the mask is ignored
static double softcap_max_err(bool control) {
    const int nc = 45, nr = 6;
    const float scale = 0.5f, sb = 0.1f, sa = 20.0f;
    ggml_context * ctx = mk();
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
    fill(x, -50.0f, 50.0f);
    fill(m, -4.0f, 0.0f);
    for (int r = 0; r < nr; ++r) ((float *) m->data)[r * nc + (r % nc)] = -INFINITY;   // a masked slot per row
    ggml_tensor * y = ggml_softcap_max(ctx, x, m, scale, 0.0f, sb, sa);
    run(ctx, y);
    double err = 0.0;
    for (int r = 0; r < nr; ++r) {
        std::vector<double> w(nc);
        double mx = -INFINITY;
        for (int i = 0; i < nc; ++i) {
            w[i] = scale * sa * tanh((double) sb * ((float *) x->data)[r * nc + i]);
            if (!control) w[i] += ((float *) m->data)[r * nc + i];
            mx = fmax(mx, w[i]);
        }
        double s = 0.0;
        for (int i = 0; i < nc; ++i) s += exp(w[i] - mx);
        for (int i = 0; i < nc; ++i) err = fmax(err, fabs(exp(w[i] - mx) / s - ((float *) y->data)[r * nc + i]));
    }
    ggml_free(ctx);
    return err;
}

// mask_to_idx: row -> the first max_row ascending j with mask[j] == 0 (F16 -0.0 counts), pad -1.
// ref_kind 0 = that; 1 = control: pad with 0 instead of -1; 2 = control: F16 compared by BITS
static double mask_to_idx_err(ggml_type type, int ref_kind) {
    const int ne0 = 40, nr = 5, max_row = 9;
    ggml_context * ctx = mk();
    ggml_tensor * m = ggml_new_tensor_2d(ctx, type, ne0, nr);
    std::vector<float> M(ne0 * nr);
    std::uniform_int_distribution<int> d(0, 3);
    for (int r = 0; r < nr; ++r)
        for (int j = 0; j < ne0; ++j) {
            // row 0 has 20 zeros (> max_row: the overflow the old kernel wrote past); row 4 has none
            const bool zero = r == 0 ? (j % 2 == 0) : r == 4 ? false : d(rng) == 0;
            M[r * ne0 + j] = zero ? ((j % 3 == 0) ? -0.0f : 0.0f) : -INFINITY;
        }
    for (int i = 0; i < ne0 * nr; ++i) {
        if (type == GGML_TYPE_F16) ((ggml_fp16_t *) m->data)[i] = ggml_fp32_to_fp16(M[i]);
        else                       ((float *) m->data)[i] = M[i];
    }
    ggml_tensor * y = ggml_mask_to_index(ctx, m, max_row);
    run(ctx, y);
    double err = 0.0;
    for (int r = 0; r < nr; ++r) {
        std::vector<int> ref;
        for (int j = 0; j < ne0 && (int) ref.size() < max_row; ++j) {
            const float v = M[r * ne0 + j];
            const bool is_zero = ref_kind == 2 && type == GGML_TYPE_F16 ? (v == 0.0f && !std::signbit(v)) : v == 0.0f;
            if (is_zero) ref.push_back(j);
        }
        while ((int) ref.size() < max_row) ref.push_back(ref_kind == 1 ? 0 : -1);
        for (int i = 0; i < max_row; ++i) err = fmax(err, fabs((double) ref[i] - ((int32_t *) y->data)[r * max_row + i]));
    }
    ggml_free(ctx);
    return err;
}

// ================================================================ 2026-09-26 ports (7834d2fbc + DS4 set)
// test-backend-ops carries NO cases for any of these ops, so this is their only numeric check.

static void run_n(ggml_context * ctx, ggml_tensor * out, int nth) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, nth);
}

// multi_add: out[:, t] = sum_j X[:, j, t] over n experts; a is the view the MoE graph builds
// control: sum of the first n-1 experts only
static double multi_add_err(bool control) {
    const int ne0 = 70, n = 5, T = 6;
    ggml_context * ctx = mk();
    ggml_tensor * X = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, n, T);
    fill(X, -2.0f, 2.0f);
    ggml_tensor * a = ggml_view_2d(ctx, X, ne0, T, X->nb[2], 0);
    ggml_tensor * y = ggml_multi_add(ctx, a, n);
    run(ctx, y);
    double err = 0.0;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < ne0; ++i) {
            double s = 0.0;
            for (int j = 0; j < (control ? n - 1 : n); ++j) s += ((float *) X->data)[(t * n + j) * ne0 + i];
            err = fmax(err, fabs(s - ((float *) y->data)[t * ne0 + i]));
        }
    ggml_free(ctx);
    return err;
}

// hc_post: y[t][i][c] = x[t][c]*post[t][i] + sum_j comb[t][j][i]*res[t][j][c]   (comb flat [j*S + i])
// T == 1 has its own kernel branch; with ne0 = 4096 and nth = 128 the OLD chunk count wrote only
// channels < 2048. control_old_chunks: the reference zeroes what the old code left unwritten.
static double hc_post_err(int T, int ne0, int nth, bool control_old_chunks) {
    const int S = 4;
    ggml_context * ctx = mk();
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, T);
    ggml_tensor * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, T);
    ggml_tensor * res  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, S, T);
    ggml_tensor * comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, S, T);
    fill(x, -1.0f, 1.0f); fill(post, -1.0f, 1.0f); fill(res, -1.0f, 1.0f); fill(comb, 0.0f, 1.0f);
    ggml_tensor * y = ggml_hc_post(ctx, x, post, res, comb);
    memset(y->data, 0, ggml_nbytes(y));
    run_n(ctx, y, nth);
    const int old_cover = 64 * ((ne0 + nth - 1) / nth);
    double err = 0.0;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < S; ++i)
            for (int c = 0; c < ne0; ++c) {
                double v = (double) ((float *) x->data)[t * ne0 + c] * ((float *) post->data)[t * S + i];
                for (int j = 0; j < S; ++j)
                    v += (double) ((float *) comb->data)[t * S * S + j * S + i] * ((float *) res->data)[(t * S + j) * ne0 + c];
                if (control_old_chunks && T == 1 && c >= old_cover) v = 0.0;
                err = fmax(err, fabs(v - ((float *) y->data)[(t * S + i) * ne0 + c]));
            }
    ggml_free(ctx);
    return err;
}

// Sinkhorn-Knopp as the op documents it: row softmax + eps, one column normalisation, then
// (iters-1) rounds of row + column normalisation, every normaliser carrying +eps.
static void sinkhorn_ref(const float * x, int S, int iters, double eps, std::vector<double> & m) {
    m.assign(S * S, 0.0);
    for (int r = 0; r < S; ++r) {
        double mx = x[r * S];
        for (int c = 1; c < S; ++c) mx = fmax(mx, x[r * S + c]);
        double s = 0.0;
        for (int c = 0; c < S; ++c) { m[r * S + c] = exp(x[r * S + c] - mx); s += m[r * S + c]; }
        for (int c = 0; c < S; ++c) m[r * S + c] = m[r * S + c] / s + eps;
    }
    auto cols = [&]() { for (int c = 0; c < S; ++c) { double s = eps; for (int r = 0; r < S; ++r) s += m[r*S+c]; for (int r = 0; r < S; ++r) m[r*S+c] /= s; } };
    auto rows = [&]() { for (int r = 0; r < S; ++r) { double s = eps; for (int c = 0; c < S; ++c) s += m[r*S+c]; for (int c = 0; c < S; ++c) m[r*S+c] /= s; } };
    cols();
    for (int i = 0; i < iters - 1; ++i) { rows(); cols(); }
}

// control_layout: compare the transposed output against the untransposed reference
static double sinkhorn_err(bool transposed, bool control_layout) {
    const int S = 4, T = 9, iters = 5;
    const float eps = 1e-6f;
    ggml_context * ctx = mk();
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * S, T);
    fill(x, -3.0f, 3.0f);
    ggml_tensor * y = ggml_sinkhorn(ctx, x, S, iters, eps, transposed);
    run(ctx, y);
    double err = 0.0;
    std::vector<double> m;
    for (int t = 0; t < T; ++t) {
        sinkhorn_ref((float *) x->data + t * S * S, S, iters, eps, m);
        for (int r = 0; r < S; ++r)
            for (int c = 0; c < S; ++c) {
                const bool tr = transposed != control_layout;
                const double ref = m[r * S + c];
                const float got = ((float *) y->data)[t * S * S + (tr ? c * S + r : r * S + c)];
                err = fmax(err, fabs(ref - got));
            }
    }
    ggml_free(ctx);
    return err;
}

// hc_pre: per token, pre = sigmoid(x_pre*s0 + b) + eps, post = 2*sigmoid(x_post*s1 + b),
// comb = sinkhorn(x_comb*s2 + b). Output flat: [pre S*T | post S*T | comb S*S*T].
// control: post without the factor 2
static double hc_pre_err(bool control) {
    const int S = 4, T = 5, iters = 4, ntot = S * S + 2 * S;
    const float eps = 1e-6f;
    ggml_context * ctx = mk();
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ntot, T);
    ggml_tensor * sc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ntot);
    fill(x, -2.0f, 2.0f); fill(sc, 0.5f, 1.5f); fill(b, -0.5f, 0.5f);
    ggml_tensor * y = ggml_hc_pre(ctx, x, sc, b, S, iters, eps);
    run(ctx, y);
    const float * X = (float *) x->data, * SC = (float *) sc->data, * B = (float *) b->data, * Y = (float *) y->data;
    double err = 0.0;
    std::vector<double> m;
    for (int t = 0; t < T; ++t) {
        const float * xt = X + t * ntot;
        for (int i = 0; i < S; ++i) {
            const double pre  = 1.0 / (1.0 + exp(-(xt[i] * SC[0] + B[i]))) + eps;
            const double post = (control ? 1.0 : 2.0) / (1.0 + exp(-(xt[S + i] * SC[1] + B[S + i])));
            err = fmax(err, fabs(pre - Y[S * t + i]));
            err = fmax(err, fabs(post - Y[S * (T + t) + i]));
        }
        std::vector<float> z(S * S);
        for (int i = 0; i < S * S; ++i) z[i] = xt[2 * S + i] * SC[2] + B[2 * S + i];
        sinkhorn_ref(z.data(), S, iters, eps, m);
        for (int i = 0; i < S * S; ++i) err = fmax(err, fabs(m[i] - Y[2 * S * T + S * S * t + i]));
    }
    ggml_free(ctx);
    return err;
}

// ds4_comp: out[b][j] = sum_r softmax_r(score[row_r][c_r(j)]) * state[row_r][c_r(j)].
// type 1: rows idx[ratio*b + r], column j. type 0: ratio rows from the first half of idx at column j
// PLUS ratio rows from the second half at column ne0 + j (state/score are 2*ne0 wide).
// control: type 0 with the second-half rows dropped
static double ds4_comp_err(int type, bool control) {
    const int ratio = 3, nblock = 4, nrow = 20;
    const int ne0 = type == 0 ? 64 : 64;          // output width
    const int W = type == 0 ? 2 * ne0 : ne0;      // state/score width
    const int nidx = type == 0 ? 2 * ratio * nblock : ratio * nblock;
    ggml_context * ctx = mk();
    ggml_tensor * st = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, nrow);
    ggml_tensor * sc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W, nrow);
    ggml_tensor * id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nidx);
    fill(st, -1.0f, 1.0f); fill(sc, -3.0f, 3.0f);
    std::uniform_int_distribution<int> d(0, nrow - 1);
    for (int i = 0; i < nidx; ++i) ((int32_t *) id->data)[i] = d(rng);
    ggml_tensor * y = ggml_ds4_comp(ctx, st, sc, id, ratio, type);
    run(ctx, y);
    const int32_t * I = (int32_t *) id->data;
    double err = 0.0;
    for (int b = 0; b < nblock; ++b)
        for (int j = 0; j < ne0; ++j) {
            std::vector<std::pair<int, int>> rc;   // (row, column)
            for (int r = 0; r < ratio; ++r) rc.push_back({ I[ratio * b + r], j });
            if (type == 0 && !control)
                for (int r = 0; r < ratio; ++r) rc.push_back({ I[ratio * b + r + nidx / 2], ne0 + j });
            double mx = -INFINITY;
            for (auto & p : rc) mx = fmax(mx, ((float *) sc->data)[p.first * W + p.second]);
            double s = 0.0, acc = 0.0;
            for (auto & p : rc) {
                const double w = exp(((float *) sc->data)[p.first * W + p.second] - mx);
                s += w; acc += w * ((float *) st->data)[p.first * W + p.second];
            }
            err = fmax(err, fabs(acc / s - ((float *) y->data)[b * ne0 + j]));
        }
    ggml_free(ctx);
    return err;
}

// latent_attn: out[h][t][d] = sum over [prefix | cache] of softmax(scale*q.k + mask) * value,
// value = prefix_v[d][p] for prefix rows, cache[n][dv_off + d] for cache rows.
// indexed: cache rows restricted to indices[t][k] (mask gathered). control: prefix ignored.
static double latent_attn_err(bool indexed, bool control) {
    const int Dk = 48, T = 3, H = 2, N = 17, P = 3, dv = 16, dv_off = 32, topk = 6;
    const float scale = 0.3f;
    ggml_context * ctx = mk();
    ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Dk, T, H);
    ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Dk, N);
    ggml_tensor * pk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Dk, P);
    ggml_tensor * pv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, dv);
    ggml_tensor * m  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, T);
    fill(q, -1.0f, 1.0f); fill(c, -1.0f, 1.0f); fill(pk, -1.0f, 1.0f); fill(pv, -1.0f, 1.0f); fill(m, -2.0f, 0.0f);
    for (int t = 0; t < T; ++t) ((float *) m->data)[t * N + (5 + t)] = -INFINITY;
    ggml_tensor * ix = nullptr;
    if (indexed) {
        ix = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, T);
        std::uniform_int_distribution<int> d(0, N - 1);
        for (int i = 0; i < topk * T; ++i) ((int32_t *) ix->data)[i] = d(rng);
    }
    ggml_tensor * y = indexed
        ? ggml_latent_attn_indexed_ext(ctx, q, c, pk, pv, m, ix, dv, dv_off, scale, 0.0f)
        : ggml_latent_attn_prefix_ext (ctx, q, c, pk, pv, m,     dv, dv_off, scale, 0.0f);
    run(ctx, y);
    const float * Q = (float *) q->data, * C = (float *) c->data, * PK = (float *) pk->data,
                * PV = (float *) pv->data, * M = (float *) m->data, * Y = (float *) y->data;
    double err = 0.0;
    for (int h = 0; h < H; ++h)
        for (int t = 0; t < T; ++t) {
            const float * qr = Q + (h * T + t) * Dk;
            std::vector<double> s; std::vector<int> src;   // src < 0: prefix row -src-1, else cache row
            if (!control)
                for (int p = 0; p < P; ++p) {
                    double v = 0; for (int i = 0; i < Dk; ++i) v += qr[i] * PK[p * Dk + i];
                    s.push_back(scale * v); src.push_back(-p - 1);
                }
            const int ncand = indexed ? topk : N;
            for (int k = 0; k < ncand; ++k) {
                const int n = indexed ? ((int32_t *) ix->data)[t * topk + k] : k;
                double v = 0; for (int i = 0; i < Dk; ++i) v += qr[i] * C[n * Dk + i];
                s.push_back(scale * v + M[t * N + n]); src.push_back(n);
            }
            double mx = -INFINITY; for (double v : s) mx = fmax(mx, v);
            double z = 0; for (double & v : s) { v = exp(v - mx); z += v; }
            for (int d = 0; d < dv; ++d) {
                double acc = 0;
                for (size_t k = 0; k < s.size(); ++k)
                    acc += s[k] / z * (src[k] < 0 ? PV[d * P + (-src[k] - 1)] : C[src[k] * Dk + dv_off + d]);
                err = fmax(err, fabs(acc - Y[(h * T + t) * dv + d]));
            }
        }
    ggml_free(ctx);
    return err;
}

// argsort_thresh: descending argsort; entries from min_entries on whose value < max*thresh -> -1.
// control: no thresholding (plain argsort)
static double argsort_thresh_err(bool control) {
    const int ne0 = 16, nr = 7, min_entries = 3;
    const float thresh = 0.5f;
    ggml_context * ctx = mk();
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, nr);
    fill(x, 0.0f, 1.0f);
    ggml_tensor * y = ggml_argsort_thresh(ctx, x, min_entries, thresh);
    run(ctx, y);
    double err = 0.0;
    for (int r = 0; r < nr; ++r) {
        const float * xr = (float *) x->data + r * ne0;
        std::vector<int> idx(ne0);
        for (int i = 0; i < ne0; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return xr[a] > xr[b]; });
        const float mx = xr[idx[0]];
        for (int j = 0; j < ne0; ++j) {
            const int ref = (!control && j >= min_entries && xr[idx[j]] < mx * thresh) ? -1 : idx[j];
            const int got = ((int32_t *) y->data)[r * ne0 + j];
            // values are distinct floats, so any tie-order difference cannot arise
            err = fmax(err, fabs((double) ref - got));
        }
    }
    ggml_free(ctx);
    return err;
}

// fused_norm: y = (x - mean) / sqrt(var + eps) * w      control: no weight
static double fused_norm_err(bool control) {
    const int ne0 = 96, nr = 5;
    const float eps = 1e-5f;
    ggml_context * ctx = mk();
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, nr);
    ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    fill(x, -3.0f, 5.0f); fill(w, 0.5f, 1.5f);
    ggml_tensor * y = ggml_fused_norm(ctx, x, w, eps);
    run(ctx, y);
    double err = 0.0;
    for (int r = 0; r < nr; ++r) {
        const float * xr = (float *) x->data + r * ne0;
        double mean = 0; for (int i = 0; i < ne0; ++i) mean += xr[i]; mean /= ne0;
        double var = 0;  for (int i = 0; i < ne0; ++i) var += (xr[i] - mean) * (xr[i] - mean); var /= ne0;
        for (int i = 0; i < ne0; ++i) {
            const double ref = (xr[i] - mean) / sqrt(var + eps) * (control ? 1.0 : ((float *) w->data)[i]);
            err = fmax(err, fabs(ref - ((float *) y->data)[r * ne0 + i]));
        }
    }
    ggml_free(ctx);
    return err;
}

int main() {
    printf("EXIT-CODE-SEMANTICS: findings=1\n");
    const double TOL = 2e-5;

    printf("rope window offset (ne0=128, n_dims=32, positions 0/7/1000)\n");
    for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX }) {
        const char * m = mode == GGML_ROPE_TYPE_NEOX ? "neox" : "normal";
        char lab[128];
        for (int off : { 0, 32, 96 }) {   // 96 == ne0 - n_dims: the "flipped" window
            snprintf(lab, sizeof lab, "%s offset %d (as written by ggml_rope_set_offset)", m, off);
            check(lab, rope_err(mode, 32, off, -1), TOL, true);
        }
        // CONTROL = the pre-fix encoding for an explicit offset: slot 15 = offset + 2.
        snprintf(lab, sizeof lab, "%s offset 32 with OLD encoding (slot15 = 34)", m);
        check(lab, rope_err(mode, 32, 32, 34), TOL, false);
    }

    printf("sqrt_softplus\n");
    check("vs the SPEC formula (CUDA/ggml.c), x in [-30, 30]", sqrt_softplus_err(0), 1e-6, true);
    check("vs plain softplus", sqrt_softplus_err(1), TOL, false);
    // Not a check of the port: the spec's logf(1+expf(x)) rounds 1+exp(x) to 1.0f for
    // x < ~-16.6, and sqrt magnifies that near 0. Bounded here so a regression is visible.
    check("SPEC PRECISION vs exact log1p (bound 3e-4; shared by CUDA)", sqrt_softplus_err(2), 3e-4, true);

    printf("unary swiglu (fused [x | g], nc = 37)\n");
    check("silu(x) * g", swiglu_err(false), TOL, true);
    check("vs halves swapped: silu(g) * x", swiglu_err(true), TOL, false);

    printf("col2im_1d (K=4, OC=3, T_in=11, s0=2)\n");
    check("p0 = 0 (the TTS decoders' call; divided by zero before)", col2im_err(0, 0), TOL, true);
    check("p0 = 1", col2im_err(1, 1), TOL, true);
    check("p0 = 1 vs a reference that ignores p0", col2im_err(1, 0), TOL, false);

    printf("softcap (s_before 0.37, s_after 30, |x| to 300) -- ported 2026-09-26\n");
    check("s_after*tanh(s_before*x)", softcap_err(0, false), 1e-5, true);
    check("in place", softcap_err(0, true), 1e-5, true);
    check("vs no tanh (linear)", softcap_err(1, false), 1e-5, false);

    printf("softcap_max (scale 0.5, s_before 0.1, s_after 20, F32 mask)\n");
    check("softmax(scale*s_after*tanh(s_before*x) + mask)", softcap_max_err(false), 1e-5, true);
    check("vs mask ignored", softcap_max_err(true), 1e-5, false);

    printf("mask_to_idx (max_row 9; row 0 has 20 zeros, row 4 none; -0.0 present)\n");
    check("F32", mask_to_idx_err(GGML_TYPE_F32, 0), 0.0, true);
    check("F16 (-0.0 counts as zero)", mask_to_idx_err(GGML_TYPE_F16, 0), 0.0, true);
    check("F32 vs padding with 0", mask_to_idx_err(GGML_TYPE_F32, 1), 0.0, false);
    check("F16 vs the OLD bitwise compare (misses -0.0)", mask_to_idx_err(GGML_TYPE_F16, 2), 0.0, false);

    printf("multi_add (5 experts, the strided view the MoE graph builds) -- ported 2026-09-26\n");
    check("sum over all experts", multi_add_err(false), 1e-5, true);
    check("vs one expert short", multi_add_err(true), 1e-5, false);

    printf("hc_post (S=4)\n");
    check("T=3, ne0=200, 4 threads (per-token branch)", hc_post_err(3, 200, 4, false), 1e-5, true);
    check("T=1, ne0=4096, 128 threads (the chunk-count REPAIR)", hc_post_err(1, 4096, 128, false), 1e-5, true);
    check("T=1, ne0=4096, 128 threads vs what the OLD chunk count wrote", hc_post_err(1, 4096, 128, true), 1e-5, false);

    printf("sinkhorn (S=4, 5 iterations)\n");
    check("row-major output", sinkhorn_err(false, false), 1e-5, true);
    check("transposed output", sinkhorn_err(true, false), 1e-5, true);
    check("transposed output vs row-major reference", sinkhorn_err(true, true), 1e-5, false);

    printf("hc_pre (S=4, 4 iterations)\n");
    check("sigmoid pre/post + sinkhorn comb", hc_pre_err(false), 1e-5, true);
    check("vs post without the factor 2", hc_pre_err(true), 1e-5, false);

    printf("ds4_comp (ratio 3, 4 blocks)\n");
    check("type 1 (one half)", ds4_comp_err(1, false), 1e-5, true);
    check("type 0 (both halves)", ds4_comp_err(0, false), 1e-5, true);
    check("type 0 vs second-half rows dropped", ds4_comp_err(0, true), 1e-5, false);

    printf("latent_attn (Dk 48, P 3, N 17, dv 16 at offset 32)\n");
    check("dense: joint softmax over [prefix | cache]", latent_attn_err(false, false), 1e-5, true);
    check("indexed (topk 6, mask gathered)", latent_attn_err(true, false), 1e-5, true);
    check("dense vs prefix ignored", latent_attn_err(false, true), 1e-5, false);

    printf("argsort_thresh (min_entries 3, thresh 0.5)\n");
    check("descending argsort, tail below max*thresh -> -1", argsort_thresh_err(false), 0.0, true);
    check("vs no threshold", argsort_thresh_err(true), 0.0, false);

    printf("fused_norm (layer norm with the weight folded in)\n");
    check("(x-mean)/sqrt(var+eps)*w", fused_norm_err(false), 1e-5, true);
    check("vs weight ignored", fused_norm_err(true), 1e-5, false);

    printf("%d checks, %d failed\nVERDICT: %s\n", g_checks, g_fails, g_fails ? "FAIL" : "PASS");
    return g_fails ? 1 : 0;
}
