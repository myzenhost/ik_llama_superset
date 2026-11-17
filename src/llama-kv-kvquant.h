// Static, calibrated, PER-CHANNEL K quantisation (KVQuant-style), CPU reference container.
// Row S69-KVQUANT-PICK-STATIC-PERCHANNEL-K-POSTROPE-2026-09-29-01.
//
// SPEC (written first; implemented from scratch, no upstream code copied -- SqueezeAILab/KVQuant carries no
// licence file at its root, so nothing is taken from it, only the method as described in the paper/README):
//   * K is quantised AFTER RoPE (the vectors that are actually stored in the cache), one row = one (token, kv-head).
//   * Each (layer, kv-head, channel) has a STATIC dense range [lo, hi], fixed at calibration time from sample K
//     rows. At run time there is no per-token scale: row bits are only the codes (+ sparse outliers).
//   * Codes are uniform asymmetric: code = round((x-lo)/s), s = (hi-lo)/(2^bits-1), clamped to [0, 2^bits-1].
//   * "Outlier-aware": calibration takes lo/hi as the (1-f)/2 and 1-(1-f)/2 quantiles (f = dense_frac, e.g. 0.99).
//     A value outside [lo,hi] at run time is an OUTLIER: it is kept exactly (as half) in a sparse (channel, value)
//     list and its dense code is the clamped one. dense_frac == 1 degenerates to min/max with no outliers.
//   * Dequantise: x = lo + code*s, replaced by the sparse value when the channel is listed.
//   * Not done here: non-uniform (k-means / "nuq") code books, per-token first-token (attention sink) fp16
//     special-casing, Fisher-weighted calibration.
//
// TABLE FILE FORMAT (little endian, version 1) -- see table::serialize / table::load:
//   u32 magic = bytes 'K','V','Q','T' (0x5451564B as a LE u32)
//   u32 version = 1
//   u32 n_layer, u32 n_kv_head, u32 head_dim, u32 bits (2|4|8)
//   f32 dense_frac
//   f32 lo[n_layer*n_kv_head*head_dim]   index ((l*n_kv_head)+h)*head_dim + c
//   f32 hi[same]
#pragma once

#include "llama-kv-lowbit-common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace llama_kvq {

using namespace llama_kvlb;

static constexpr uint32_t MAGIC   = 0x5451564Bu; // bytes "KVQT"
static constexpr uint32_t VERSION = 1;

struct table {
    uint32_t n_layer = 0, n_kv_head = 0, head_dim = 0, bits = 4;
    float    dense_frac = 1.0f;
    std::vector<float> lo, hi;

    size_t idx(uint32_t l, uint32_t h, uint32_t c) const { return ((size_t)l * n_kv_head + h) * head_dim + c; }

    bool valid() const {
        const size_t n = (size_t)n_layer * n_kv_head * head_dim;
        return n_layer && n_kv_head && head_dim && bits_supported(bits) && (head_dim * bits) % 8 == 0 &&
               lo.size() == n && hi.size() == n;
    }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> out;
        auto put32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) out.push_back((uint8_t)(v >> (8*i))); };
        auto putf  = [&](float f)    { uint32_t v; std::memcpy(&v, &f, 4); put32(v); };
        put32(MAGIC); put32(VERSION); put32(n_layer); put32(n_kv_head); put32(head_dim); put32(bits);
        putf(dense_frac);
        for (float v : lo) putf(v);
        for (float v : hi) putf(v);
        return out;
    }

    // Returns false (and leaves *this unspecified) on bad magic/version, unsupported bits, zero dims, or a size mismatch.
    bool load(const uint8_t * p, size_t n) {
        size_t o = 0;
        auto get32 = [&](uint32_t & v) { if (o + 4 > n) return false; v = 0; for (int i = 0; i < 4; i++) v |= (uint32_t)p[o+i] << (8*i); o += 4; return true; };
        uint32_t magic, ver, nl, nh, hd, b, fr;
        if (!get32(magic) || !get32(ver) || !get32(nl) || !get32(nh) || !get32(hd) || !get32(b) || !get32(fr)) return false;
        if (magic != MAGIC || ver != VERSION || !bits_supported(b) || !nl || !nh || !hd || (hd * b) % 8) return false;
        const uint64_t cnt = (uint64_t)nl * nh * hd;
        if (cnt > (1ull << 30) || (n - o) != cnt * 8) return false; // exact size, no trailing bytes
        n_layer = nl; n_kv_head = nh; head_dim = hd; bits = b;
        std::memcpy(&dense_frac, &fr, 4);
        lo.resize(cnt); hi.resize(cnt);
        for (auto * v : { &lo, &hi }) for (auto & f : *v) { uint32_t u; get32(u); std::memcpy(&f, &u, 4); }
        return true;
    }
};

// Accumulates post-RoPE K sample rows per (layer, kv-head) and fits the static per-channel ranges.
class calibrator {
public:
    calibrator(uint32_t n_layer, uint32_t n_kv_head, uint32_t head_dim, uint32_t bits, float dense_frac)
        : nl(n_layer), nh(n_kv_head), hd(head_dim), bits_(bits), frac(dense_frac), samples((size_t)n_layer * n_kv_head) {}

    void add_row(uint32_t l, uint32_t h, const float * k) {
        auto & s = samples[(size_t)l * nh + h];
        s.insert(s.end(), k, k + hd);
    }

    table finish() const {
        table t; t.n_layer = nl; t.n_kv_head = nh; t.head_dim = hd; t.bits = bits_; t.dense_frac = frac;
        t.lo.assign((size_t)nl * nh * hd, 0.f); t.hi = t.lo;
        std::vector<float> col;
        for (uint32_t l = 0; l < nl; l++) for (uint32_t h = 0; h < nh; h++) {
            const auto & s = samples[(size_t)l * nh + h];
            const size_t n = s.size() / hd;
            if (!n) continue;
            for (uint32_t c = 0; c < hd; c++) {
                col.resize(n);
                for (size_t i = 0; i < n; i++) col[i] = s[i*hd + c];
                std::sort(col.begin(), col.end());
                const double tail = (1.0 - (double)frac) / 2.0;
                const size_t a = (size_t)std::floor(tail * (double)(n - 1) + 0.5);
                const size_t b = (n - 1) - a;
                t.lo[t.idx(l, h, c)] = col[a];
                t.hi[t.idx(l, h, c)] = col[b];
            }
        }
        return t;
    }
private:
    uint32_t nl, nh, hd, bits_; float frac;
    std::vector<std::vector<float>> samples;
};

// One quantised K row of one (layer, kv-head).
struct row {
    std::vector<uint8_t>                   codes;     // packed, head_dim*bits/8 bytes
    std::vector<std::pair<uint16_t, uint16_t>> outl;  // (channel, half bits) sparse exact values, ascending channel
    // storage cost in bits for this row (dense codes + 32 bits per outlier = 16 idx + 16 value)
    size_t bits_used() const { return codes.size() * 8 + outl.size() * 32; }
};

inline row quantize_row(const table & t, uint32_t l, uint32_t h, const float * k) {
    row r;
    const uint32_t D = t.head_dim, maxq = (1u << t.bits) - 1;
    std::vector<uint8_t> c(D);
    for (uint32_t i = 0; i < D; i++) {
        const float lo = t.lo[t.idx(l, h, i)], hi = t.hi[t.idx(l, h, i)];
        const float s = (hi > lo) ? (hi - lo) / (float)maxq : 0.f;
        float q = s > 0.f ? std::floor((k[i] - lo) / s + 0.5f) : 0.f;
        q = std::min(std::max(q, 0.f), (float)maxq);
        c[i] = (uint8_t)q;
        if (k[i] < lo || k[i] > hi) r.outl.emplace_back((uint16_t)i, f32_to_f16(k[i]));
    }
    r.codes.resize(D * t.bits / 8);
    pack_codes(c.data(), D, t.bits, r.codes.data());
    return r;
}

inline void dequantize_row(const table & t, uint32_t l, uint32_t h, const row & r, float * out) {
    const uint32_t D = t.head_dim, maxq = (1u << t.bits) - 1;
    std::vector<uint8_t> c(D);
    unpack_codes(r.codes.data(), D, t.bits, c.data());
    for (uint32_t i = 0; i < D; i++) {
        const float lo = t.lo[t.idx(l, h, i)], hi = t.hi[t.idx(l, h, i)];
        const float s = (hi > lo) ? (hi - lo) / (float)maxq : 0.f;
        out[i] = lo + (float)c[i] * s;
    }
    for (const auto & o : r.outl) out[o.first] = f16_to_f32(o.second);
}

// q . dequant(row) computed straight from the codes (what a fused CPU kernel would do), no float row materialised.
inline float dot_row(const table & t, uint32_t l, uint32_t h, const row & r, const float * q) {
    const uint32_t D = t.head_dim, maxq = (1u << t.bits) - 1;
    std::vector<uint8_t> c(D);
    unpack_codes(r.codes.data(), D, t.bits, c.data());
    float acc = 0.f;
    for (uint32_t i = 0; i < D; i++) {
        const float lo = t.lo[t.idx(l, h, i)], hi = t.hi[t.idx(l, h, i)];
        const float s = (hi > lo) ? (hi - lo) / (float)maxq : 0.f;
        acc += q[i] * (lo + (float)c[i] * s);
    }
    for (const auto & o : r.outl) {
        const uint32_t i = o.first;
        const float lo = t.lo[t.idx(l, h, i)], hi = t.hi[t.idx(l, h, i)];
        const float s = (hi > lo) ? (hi - lo) / (float)maxq : 0.f;
        acc += q[i] * (f16_to_f32(o.second) - (lo + (float)c[i] * s));
    }
    return acc;
}

// Append-only container for one (layer, kv-head) stream of K rows.
class k_store {
public:
    k_store(const table & tbl, uint32_t layer, uint32_t head) : t(&tbl), l(layer), h(head) {}
    void   append(const float * k) { rows.push_back(quantize_row(*t, l, h, k)); }
    size_t size() const { return rows.size(); }
    void   read(size_t i, float * out) const { dequantize_row(*t, l, h, rows.at(i), out); }
    void   scores(const float * q, float * out) const { for (size_t i = 0; i < rows.size(); i++) out[i] = dot_row(*t, l, h, rows[i], q); }
    size_t bits_used() const { size_t b = 0; for (auto & r : rows) b += r.bits_used(); return b; }
private:
    const table * t; uint32_t l, h; std::vector<row> rows;
};

} // namespace llama_kvq
