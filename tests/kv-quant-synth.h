// Synthetic post-RoPE key / value generators and error metrics shared by test-kvquant-perchannel-k and test-kivi-2bit.
// No model needed. DATA RECIPE (printed by the tests too):
//   pre-RoPE key  x_c = mu_c + s_c * g,  g ~ N(0,1),  s_c = exp(0.25*N(0,1)) (mild per-channel spread),  mu_c ~ N(0, 0.5*s_c)
//   outlier channels: n_out_pairs RoPE pairs (i, i+D/2) get s *= U(10,50) and mu *= same factor; the pairs are taken from
//     the LOW-frequency end (i >= D/2 - n_out_pairs: they rotate slowly, so after RoPE the channel stays large and keeps a
//     stable sign/bias -- the structure reported for real post-RoPE keys) or the HIGH-frequency end (i < n_out_pairs: the
//     pair rotates fast, so the channel is large but oscillates). Both variants are measured.
//   RoPE (NeoX layout, theta=10000): pair (i, i+D/2) rotated by pos*theta^(-2i/D).
//   queries: same generator, independent draw, no outlier amplification beyond what the key recipe gives to the same channels
//     (real q and k outlier channels coincide), so q.k logits are dominated by the outlier channels exactly as in practice.
// The numbers are a property of THIS recipe; they are not a claim about any particular model.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace kvsynth {

struct rng {
    uint64_t s;
    explicit rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) { for (int i = 0; i < 4; i++) u(); }
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double u() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    double n() { double a = u(), b = u(); if (a < 1e-300) a = 1e-300; return std::sqrt(-2.0 * std::log(a)) * std::cos(6.283185307179586 * b); }
};

struct key_model {
    int D;
    std::vector<double> s, mu;
    key_model(int D_, int n_out_pairs, bool low_freq, uint64_t seed) : D(D_), s(D_), mu(D_) {
        rng r(seed);
        for (int c = 0; c < D; c++) { s[c] = std::exp(0.25 * r.n()); mu[c] = 0.5 * s[c] * r.n(); }
        for (int j = 0; j < n_out_pairs; j++) {
            const int i = low_freq ? D / 2 - 1 - j : j;
            const double f = 10.0 + 40.0 * r.u();
            s[i] *= f; mu[i] *= f; s[i + D / 2] *= f; mu[i + D / 2] *= f;
        }
    }
    // one post-RoPE key row at position pos
    void sample(rng & r, int pos, float * out) const {
        std::vector<double> x(D);
        for (int c = 0; c < D; c++) x[c] = mu[c] + s[c] * r.n();
        const int h = D / 2;
        for (int i = 0; i < h; i++) {
            const double ang = pos * std::pow(10000.0, -2.0 * i / D), cs = std::cos(ang), sn = std::sin(ang);
            out[i]     = (float)(x[i] * cs - x[i + h] * sn);
            out[i + h] = (float)(x[i + h] * cs + x[i] * sn);
        }
    }
};

// in-place orthonormal Walsh-Hadamard transform, n a power of two (stand-in for the fork's attn_rot_k Hadamard,
// src/llama-kv-cache.cpp:1505, which for D=128 uses a full 128x128 rotation)
inline void wht(float * x, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) { float a = x[j], b = x[j + len]; x[j] = a + b; x[j + len] = a - b; }
    const float sc = 1.0f / std::sqrt((float)n);
    for (int i = 0; i < n; i++) x[i] *= sc;
}

inline double rel_rmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) { double d = (double)ref[i] - got[i]; num += d * d; den += (double)ref[i] * ref[i]; }
    return std::sqrt(num / den);
}

} // namespace kvsynth
