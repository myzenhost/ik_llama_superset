// ggml_rwkv_wkv6 on the CPU backend against a scalar double-precision reference of the recurrence.
// Row: S69-GGML-OP-3-RWKV-WKV6-WKV7. No model file, no device.
//
// Recurrence (per sequence, per head; S = head size; state M[i][j], i = key index, j = value index, row-major M[i*S+j]):
//     per token:   out[j] = sum_i r[i] * ( M[i][j] + tf[i]*k[i]*v[j] )        (bonus "time_first" applies to the current token only)
//                  M[i][j] = M[i][j]*td[i] + k[i]*v[j]                          (td = per-token, per-channel decay)
// Inputs k,v,r,td are [S, H, n_seqs*T] (sequence-major), tf is [S, H] (shared by all tokens), state is [S*S*H, n_seqs],
// result is [S*H, n_seqs*T + S*n_seqs]: outputs, then the new state. Cases: n_seqs 1/2, T 1/>1, S 16/64, carry across two calls.
#include "recurrent-op-test-common.h"

struct cs { int64_t S, H, B, T; };

static void reference(const cs & c, const tin & k, const tin & v, const tin & r, const tin & tf, const tin & td, const tin & st,
        std::vector<double> & out, std::vector<double> & state) {
    const int64_t S = c.S, H = c.H, B = c.B, T = c.T;
    out.assign(S * H * B * T, 0.0);
    state.assign(st.d.begin(), st.d.end());
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < H; ++h) {
        double * M = &state[(b * H + h) * S * S];
        for (int64_t t = 0; t < T; ++t) {
            const int64_t o = ((b * T + t) * H + h) * S;
            for (int64_t j = 0; j < S; ++j) {
                double s = 0;
                for (int64_t i = 0; i < S; ++i)
                    s += (double) r.d[o + i] * (M[i * S + j] + (double) tf.d[h * S + i] * k.d[o + i] * v.d[o + j]);
                out[o + j] = s;
            }
            for (int64_t i = 0; i < S; ++i) for (int64_t j = 0; j < S; ++j)
                M[i * S + j] = M[i * S + j] * td.d[o + i] + (double) k.d[o + i] * v.d[o + j];
        }
    }
}

static ggml_tensor * build(ggml_context * ctx, std::vector<ggml_tensor *> & t) {
    return ggml_rwkv_wkv6(ctx, t[0], t[1], t[2], t[3], t[4], t[5]);
}

int main() {
    const cs cases[] = { {16,2,1,1}, {16,2,1,5}, {16,3,2,1}, {16,3,2,6}, {64,2,1,1}, {64,2,1,4}, {64,4,2,1}, {64,4,2,5} };
    for (const cs & c : cases) for (int nth : {1, 3}) {
        std::mt19937 rng(200 + c.S * 3 + c.H * 5 + c.B * 7 + c.T * 11);
        auto mk = [&](float lo, float hi) { return make_in(rng, lo, hi, c.S, c.H, c.B * c.T, 1, 2); };
        // in: k v r tf td state
        std::vector<tin> in = { mk(-1,1), mk(-1,1), mk(-1,1), make_in(rng, -1, 1, c.S, c.H, 1, 1, 0), mk(0.5f,0.99f),
                                make_in(rng, -0.1f, 0.1f, c.S * c.S * c.H, c.B, 1, 1, 0) };
        const std::vector<float> got = run_graph(in, nth, build);
        std::vector<double> ro, rs;
        reference(c, in[0], in[1], in[2], in[3], in[4], in[5], ro, rs);
        double mref = 0;
        const double e_out = max_abs_diff(ro.data(), got.data(), ro.size(), mref);
        const double e_st  = max_abs_diff(rs.data(), got.data() + ro.size(), rs.size(), mref);
        const double tol = 1e-4 * (1.0 + mref);
        printf("WKV6 S=%2lld H=%lld B=%lld T=%lld thr=%d  max|out err| %.2e  max|state err| %.2e  (max|ref| %.2f tol %.1e)\n",
               (long long) c.S, (long long) c.H, (long long) c.B, (long long) c.T, nth, e_out, e_st, mref, tol);
        CHECK(e_out <= tol); CHECK(e_st <= tol);

        if (c.T >= 2) {
            const int64_t T1 = c.T / 2, T2 = c.T - T1;
            auto part = [&](int64_t t0, int64_t t1) {
                std::vector<tin> p;
                for (int i : {0, 1, 2, 3, 4}) p.push_back(slice_tokens(in[i], c.B, c.T, t0, t1));
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
            printf("WKV6   carry split %lld+%lld: max|two calls - one call| %.2e\n", (long long) T1, (long long) T2, md);
            CHECK(md <= 1e-6);
        }
    }
    if (n_fail) { fprintf(stderr, "%d check(s) failed\n", n_fail); return 1; }
    printf("test-rwkv-wkv6: all checks passed\n");
    return 0;
}
