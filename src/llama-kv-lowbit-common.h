// Shared helpers for the CPU-side low-bit KV containers (llama-kv-kvquant.h, llama-kv-kivi.h).
// Header-only, no ggml dependency, so the containers are testable without linking the library.
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

namespace llama_kvlb {

// IEEE half <-> float, round-to-nearest-even. Scales, zero points and sparse outliers are stored as half so the
// bits-per-element figures the tests print are the real storage cost, not a float32 stand-in.
inline uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) {                       // inf / nan
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);       // overflow -> inf
    if (exp <= 0) {                                         // subnormal or zero
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++; // carry may roll into the exponent, which is correct
    return (uint16_t)(sign | h);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, x;
    if (exp == 0) {
        if (mant == 0) { x = sign; }
        else {                                              // subnormal -> normalise
            int e = -1;
            do { e++; mant <<= 1; } while (!(mant & 0x400u));
            x = sign | ((uint32_t)(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &x, 4);
    return f;
}

inline float round_f16(float f) { return f16_to_f32(f32_to_f16(f)); }

inline bool bits_supported(uint32_t bits) { return bits == 2 || bits == 4 || bits == 8; }

// Pack/unpack n codes of `bits` bits (2/4/8) little-endian within each byte. n*bits must be a multiple of 8.
inline void pack_codes(const uint8_t * codes, size_t n, uint32_t bits, uint8_t * out) {
    const size_t per = 8 / bits;
    for (size_t i = 0; i < n / per; i++) {
        uint8_t b = 0;
        for (size_t j = 0; j < per; j++) b |= (uint8_t)(codes[i*per + j] << (bits * j));
        out[i] = b;
    }
}
inline void unpack_codes(const uint8_t * in, size_t n, uint32_t bits, uint8_t * codes) {
    const size_t per = 8 / bits;
    const uint8_t mask = (uint8_t)((1u << bits) - 1u);
    for (size_t i = 0; i < n / per; i++)
        for (size_t j = 0; j < per; j++) codes[i*per + j] = (in[i] >> (bits * j)) & mask;
}

} // namespace llama_kvlb
