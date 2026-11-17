// ggml_gated_delta_net (upstream-shaped op, alongside the fork's ggml_delta_net) on the CPU backend against a scalar
// double-precision reference. Row: S69-GGML-OP-1-GATED-DELTA-NET. No model file, no device.
// Complements test-gdn-cpu-ref (single call, K snapshots, SIMD tails, must-fail partner): this file adds n_seqs 1/>1 x T 1/>1 x S 16/64
// with a scalar gate and adds the check that the state carried across two calls equals one call over the concatenated tokens.
//
// Recurrence (per sequence b, head h; S = head size; state M[j][i] stored row-major M[j*S+i], row j = column j of the math state):
//     per token:   M[j][i] *= exp(g)                                  (scalar gate per head/token)
//                  d[j]     = beta * ( v[j] - sum_i M[j][i]*k[i] )     (delta rule: write the prediction error)
//                  M[j][i] += d[j]*k[i]
//                  out[j]   = sum_i M[j][i]*q[i] / sqrt(S)
// q,k,v are [S,H,T,B], g is [1,H,T,B], beta is [1,H,T,B], state is [S,S,H,B]; result is [S*H, T*B + K*S*B]: outputs, then K state slots
// (slot s = state after token T-1-s; K = 1 keeps only the final state).
#include "recurrent-op-test-common.h"

struct cs { int64_t S, H, B, T; };

static void reference(const cs & c, const std::vector<tin> & x, std::vector<double> & out, std::vector<double> & state) {
    const tin & q = x[0], & k = x[1], & v = x[2], & g = x[3], & beta = x[4], & st = x[5];
    const int64_t S = c.S, H = c.H, B = c.B, T = c.T;
    out.assign(S * H * B * T, 0.0);
    state.assign(st.d.begin(), st.d.end());
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
        double * M = &state[(b * H + h) * S * S];
        for (int64_t t = 0; t < T; ++t) {
            const int64_t o = ((b * T + t) * H + h) * S, s1 = (b * T + t) * H + h;
            const double dec = std::exp((double) g.d[s1]);
            std::vector<double> d(S);
            for (int64_t j = 0; j < S; ++j) {
                double sum = 0;
                for (int64_t i = 0; i < S; ++i) { M[j * S + i] *= dec; sum += M[j * S + i] * k.d[o + i]; }
                d[j] = ((double) v.d[o + j] - sum) * beta.d[s1];
            }
            for (int64_t j = 0; j < S; ++j) {
                double sum = 0;
                for (int64_t i = 0; i < S; ++i) { M[j * S + i] += d[j] * k.d[o + i]; sum += M[j * S + i] * q.d[o + i]; }
                out[o + j] = sum / std::sqrt((double) S);
            }
        }
    }
}

static ggml_tensor * build(ggml_context * ctx, std::vector<ggml_tensor *> & t) {
    return ggml_gated_delta_net(ctx, t[0], t[1], t[2], t[3], t[4], t[5], 1);
}

static void l2_rows(tin & x, int64_t S) {
    for (size_t r = 0; r < x.d.size() / S; ++r) {
        double s = 0;
        for (int64_t i = 0; i < S; ++i) s += (double) x.d[r * S + i] * x.d[r * S + i];
        for (int64_t i = 0; i < S; ++i) x.d[r * S + i] = (float) (x.d[r * S + i] / std::sqrt(s + 1e-12));
    }
}

int main() {
    const cs cases[] = { {16,2,1,1}, {16,2,1,5}, {16,3,2,1}, {16,3,2,6}, {64,2,1,1}, {64,2,1,4}, {64,4,2,1}, {64,4,2,5} };
    for (const cs & c : cases) for (int nth : {1, 3}) {
        std::mt19937 rng(400 + c.S * 3 + c.H * 5 + c.B * 7 + c.T * 11);
        auto mk = [&](float lo, float hi, int64_t w) { return make_in(rng, lo, hi, w, c.H, c.T, c.B, 3); };
        // in: q k v g beta state
        std::vector<tin> in = { mk(-1,1,c.S), mk(-1,1,c.S), mk(-1,1,c.S), mk(-2.0f,-0.01f,1), mk(0,1,1),
                                make_in(rng, -0.1f, 0.1f, c.S, c.S, c.H, c.B, 0) };
        l2_rows(in[0], c.S); l2_rows(in[1], c.S);
        const std::vector<float> got = run_graph(in, nth, build);
        std::vector<double> ro, rs;
        reference(c, in, ro, rs);
        double mref = 0;
        const double e_out = max_abs_diff(ro.data(), got.data(), ro.size(), mref);
        const double e_st  = max_abs_diff(rs.data(), got.data() + ro.size(), rs.size(), mref);
        const double tol = 2e-4 * (1.0 + mref);
        printf("GDN S=%2lld H=%lld B=%lld T=%lld thr=%d  max|out err| %.2e  max|state err| %.2e  (max|ref| %.2f tol %.1e)\n",
               (long long) c.S, (long long) c.H, (long long) c.B, (long long) c.T, nth, e_out, e_st, mref, tol);
        CHECK(e_out <= tol); CHECK(e_st <= tol);

        if (c.T >= 2) {
            const int64_t T1 = c.T / 2, T2 = c.T - T1;
            auto part = [&](int64_t t0, int64_t t1) {
                std::vector<tin> p;
                for (int i = 0; i < 5; ++i) p.push_back(slice_tokens(in[i], c.B, c.T, t0, t1));
                return p;
            };
            std::vector<tin> p1 = part(0, T1), p2 = part(T1, c.T);
            p1.push_back(in[5]);
            const std::vector<float> g1 = run_graph(p1, nth, build);
            tin s1 = in[5]; s1.d.assign(g1.begin() + c.S * c.H * c.B * T1, g1.end());
            p2.push_back(s1);
            const std::vector<float> g2 = run_graph(p2, nth, build);
            double md = 0;
            for (int64_t b = 0; b < c.B; ++b) for (int64_t t = 0; t < c.T; ++t) {
                const float * a = t < T1 ? &g1[(b * T1 + t) * c.S * c.H] : &g2[(b * T2 + t - T1) * c.S * c.H];
                const float * w = &got[(b * c.T + t) * c.S * c.H];
                for (int64_t n = 0; n < c.S * c.H; ++n) md = std::fmax(md, std::fabs(a[n] - w[n]));
            }
            for (size_t n = 0; n < rs.size(); ++n) md = std::fmax(md, std::fabs(g2[c.S * c.H * c.B * T2 + n] - got[ro.size() + n]));
            printf("GDN   carry split %lld+%lld: max|two calls - one call| %.2e\n", (long long) T1, (long long) T2, md);
            CHECK(md <= 1e-5);
        }
    }
    if (n_fail) { fprintf(stderr, "%d check(s) failed\n", n_fail); return 1; }
    printf("test-gated-delta-net: all checks passed\n");
    return 0;
}
