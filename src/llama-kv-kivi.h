// KIVI-style asymmetric low-bit KV container (CPU reference, standalone).
// Row S69-FORK-KV-KIVI-2BIT-PORT-01.
//
// Upstream: https://github.com/jy-yuan/KIVI @ 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6, MIT licence (LICENSE, (c) 2024
// jiayi yuan). This file is a from-scratch C++ re-implementation of the algorithm read in models/llama_kivi.py and
// quant/new_pack.py; no upstream code is copied. Behaviour matched:
//   * K is quantised PER CHANNEL: a group is G consecutive tokens; for each channel the group's min/max over those G
//     tokens give an asymmetric scale/zero (scale=(mx-mn)/(2^bits-1), code=round((x-mn)/scale) clamped).
//   * V is quantised PER TOKEN: a group is G consecutive channels of ONE token.
//   * Residual window: the newest tokens stay full precision. K: tokens are collected in a full-precision buffer; when it
//     holds exactly R tokens the whole buffer is quantised (R/G groups) and emptied (the "group flush"), so the K window
//     holds 0..R-1 tokens after any append. V: the window is a sliding R tokens; once it would exceed R the OLDEST token
//     is quantised alone and leaves the window, so the V window holds min(T,R) tokens.
//   * Because both rules are per-append, feeding a prefill of T tokens one by one gives the same state KIVI's batch
//     prefill gives (K: floor(T/R)*R quantised; V: T-R quantised when T>R). Requires R % G == 0, D % G == 0.
// Differences from upstream: scale/zero stored as half bits (as upstream's fp16) but codes are computed against the
// half-rounded scale/zero (consistent round trip); codes use plain byte packing (2/4/8 bit); no CUDA/Triton kernels.
#pragma once

#include "llama-kv-lowbit-common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llama_kivi {

using namespace llama_kvlb;

class cache {
public:
    // D = head_dim, bits in {2,4,8}, G = group size, R = residual length (full-precision window)
    cache(uint32_t D_, uint32_t bits_, uint32_t G_, uint32_t R_) : D(D_), bits(bits_), G(G_), R(R_) {
        if (!bits_supported(bits) || !D || !G || !R || D % G || R % G || G % (8 / bits) || D % (8 / bits))
            throw std::invalid_argument("kivi::cache: need bits in {2,4,8}, D%G==0, R%G==0, G and D multiples of 8/bits");
    }

    size_t size() const { return kq_tokens + kfull.size() / D; } // K and V always hold the same number of tokens
    size_t quantized_k_tokens() const { return kq_tokens; }
    size_t residual_k_tokens() const { return kfull.size() / D; }
    size_t quantized_v_tokens() const { return vq.size(); }
    size_t residual_v_tokens() const { return vfull.size() / D; }

    void append(const float * k, const float * v) {
        kfull.insert(kfull.end(), k, k + D);
        if (kfull.size() / D == R) flush_k();
        vfull.insert(vfull.end(), v, v + D);
        if (vfull.size() / D > R) {
            vq.push_back(quant_v_token(vfull.data()));
            vfull.erase(vfull.begin(), vfull.begin() + D);
        }
        if (kq_tokens + kfull.size() / D != vq.size() + vfull.size() / D) throw std::logic_error("kivi: K/V token count diverged");
    }

    void read_k(size_t t, float * out) const {
        if (t >= size()) throw std::out_of_range("kivi::read_k");
        if (t >= kq_tokens) { std::copy_n(&kfull[(t - kq_tokens) * D], D, out); return; }
        const size_t gi = t / G, ti = t % G;
        const kgroup & g = kgroups[gi];
        std::vector<uint8_t> c(G);
        for (uint32_t ch = 0; ch < D; ch++) {
            unpack_codes(&g.codes[(size_t)ch * G * bits / 8], G, bits, c.data());
            out[ch] = (float)c[ti] * f16_to_f32(g.scale[ch]) + f16_to_f32(g.mn[ch]);
        }
    }

    void read_v(size_t t, float * out) const {
        if (t >= size()) throw std::out_of_range("kivi::read_v");
        if (t >= vq.size()) { std::copy_n(&vfull[(t - vq.size()) * D], D, out); return; }
        const vtoken & v = vq[t];
        std::vector<uint8_t> c(D);
        unpack_codes(v.codes.data(), D, bits, c.data());
        for (uint32_t ch = 0; ch < D; ch++) out[ch] = (float)c[ch] * f16_to_f32(v.scale[ch / G]) + f16_to_f32(v.mn[ch / G]);
    }

    // scores[t] = q . K[t] straight from the packed K groups (+ full-precision window), like KIVI's fused qK kernel.
    void k_scores(const float * q, float * scores) const {
        std::vector<uint8_t> c(G);
        for (size_t gi = 0; gi < kgroups.size(); gi++) {
            const kgroup & g = kgroups[gi];
            for (uint32_t i = 0; i < G; i++) scores[gi * G + i] = 0.f;
            for (uint32_t ch = 0; ch < D; ch++) {
                unpack_codes(&g.codes[(size_t)ch * G * bits / 8], G, bits, c.data());
                const float s = f16_to_f32(g.scale[ch]), m = f16_to_f32(g.mn[ch]);
                for (uint32_t i = 0; i < G; i++) scores[gi * G + i] += q[ch] * ((float)c[i] * s + m);
            }
        }
        for (size_t t = 0; t < kfull.size() / D; t++) {
            float a = 0.f;
            for (uint32_t ch = 0; ch < D; ch++) a += q[ch] * kfull[t * D + ch];
            scores[kq_tokens + t] = a;
        }
    }

    // out = sum_t w[t] * V[t], from packed V tokens + window (KIVI's fused attn*V kernel analogue).
    void v_weighted_sum(const float * w, float * out) const {
        std::fill(out, out + D, 0.f);
        std::vector<uint8_t> c(D);
        for (size_t t = 0; t < vq.size(); t++) {
            const vtoken & v = vq[t];
            unpack_codes(v.codes.data(), D, bits, c.data());
            for (uint32_t ch = 0; ch < D; ch++) out[ch] += w[t] * ((float)c[ch] * f16_to_f32(v.scale[ch / G]) + f16_to_f32(v.mn[ch / G]));
        }
        for (size_t t = 0; t < vfull.size() / D; t++)
            for (uint32_t ch = 0; ch < D; ch++) out[ch] += w[vq.size() + t] * vfull[t * D + ch];
    }

    // Storage actually held: packed codes + half scale/zero for quantised parts, 16 bits/element for the window
    // (the window is f32 here for arithmetic simplicity, but a deployment keeps it f16, so it is counted as 16).
    size_t bits_used_quantized() const {
        size_t b = 0;
        for (auto & g : kgroups) b += g.codes.size() * 8 + (g.scale.size() + g.mn.size()) * 16;
        for (auto & v : vq)      b += v.codes.size() * 8 + (v.scale.size() + v.mn.size()) * 16;
        return b;
    }
    size_t bits_used_window() const { return (kfull.size() + vfull.size()) * 16; }

private:
    struct kgroup { std::vector<uint8_t> codes; std::vector<uint16_t> scale, mn; }; // codes: D channels x G tokens packed
    struct vtoken { std::vector<uint8_t> codes; std::vector<uint16_t> scale, mn; }; // codes: D channels packed

    static void fit(const float * lo_hi_min, const float * lo_hi_max, uint32_t bits, uint16_t & s, uint16_t & m) {
        const float mn = round_f16(*lo_hi_min);
        float sc = (*lo_hi_max - *lo_hi_min) / (float)((1u << bits) - 1);
        s = f32_to_f16(sc); m = f32_to_f16(mn);
    }
    static uint8_t code_of(float x, float scale, float mn, uint32_t bits) {
        if (!(scale > 0.f)) return 0;
        float q = std::floor((x - mn) / scale + 0.5f);
        return (uint8_t)std::min(std::max(q, 0.f), (float)((1u << bits) - 1));
    }

    void flush_k() {
        const size_t ntok = kfull.size() / D; // == R
        for (size_t g0 = 0; g0 < ntok; g0 += G) {
            kgroup g; g.scale.resize(D); g.mn.resize(D); g.codes.resize((size_t)D * G * bits / 8);
            std::vector<uint8_t> c(G);
            for (uint32_t ch = 0; ch < D; ch++) {
                float mn = kfull[g0 * D + ch], mx = mn;
                for (uint32_t i = 1; i < G; i++) { float x = kfull[(g0 + i) * D + ch]; mn = std::min(mn, x); mx = std::max(mx, x); }
                fit(&mn, &mx, bits, g.scale[ch], g.mn[ch]);
                const float sc = f16_to_f32(g.scale[ch]), m = f16_to_f32(g.mn[ch]);
                for (uint32_t i = 0; i < G; i++) c[i] = code_of(kfull[(g0 + i) * D + ch], sc, m, bits);
                pack_codes(c.data(), G, bits, &g.codes[(size_t)ch * G * bits / 8]);
            }
            kgroups.push_back(std::move(g));
        }
        kq_tokens += ntok;
        kfull.clear();
    }

    vtoken quant_v_token(const float * v) const {
        vtoken t; t.scale.resize(D / G); t.mn.resize(D / G); t.codes.resize((size_t)D * bits / 8);
        std::vector<uint8_t> c(D);
        for (uint32_t gi = 0; gi < D / G; gi++) {
            float mn = v[gi * G], mx = mn;
            for (uint32_t i = 1; i < G; i++) { mn = std::min(mn, v[gi * G + i]); mx = std::max(mx, v[gi * G + i]); }
            fit(&mn, &mx, bits, t.scale[gi], t.mn[gi]);
            const float sc = f16_to_f32(t.scale[gi]), m = f16_to_f32(t.mn[gi]);
            for (uint32_t i = 0; i < G; i++) c[gi * G + i] = code_of(v[gi * G + i], sc, m, bits);
        }
        pack_codes(c.data(), D, bits, t.codes.data());
        return t;
    }

    uint32_t D, bits, G, R;
    std::vector<float> kfull, vfull;   // residual windows, row-major [token][channel]
    std::vector<kgroup> kgroups;       // each covers G consecutive quantised K tokens
    std::vector<vtoken> vq;            // one per quantised V token (oldest first)
    size_t kq_tokens = 0;
};

} // namespace llama_kivi
