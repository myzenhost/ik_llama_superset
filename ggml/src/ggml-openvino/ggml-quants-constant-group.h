#pragma once
// A quantisation group whose decoded scale is exactly 0 but whose min/bias is not (Q4_K, Q5_K, Q4_1, Q5_1 with
// d*sc == 0 and dmin*m != 0) dequantises to the constant `bias` for every weight (ggml: y = q*d + m).
//
// The OpenVINO integer-zero-point path computes (q - zp) * scale. With scale 0 that is 0 whatever zp is, so the
// group's constant is lost. The bias path (w*s + b) keeps it and is correct. For the zero-point path the constant is
// re-encoded as scale' = -bias, zp = 1, q = 0 : (0 - 1) * (-bias) == bias.
// Row S69-ENGREV-EXECUTORCH-OV-QK-ZERO-SCALE-CONSTANT-SUBBLOCK-2026-10-27-01.
// Idea from the executorch OpenVINO quantiser review; reimplemented, no code copied.

#include <cstddef>
#include <cstdint>
#include <cstring>

// true when the group is a nonzero constant that an affine (q - zp) * scale form would turn into 0
static inline bool ggml_ov_constant_group(float scale, float bias) {
    return scale == 0.0f && bias != 0.0f;
}

// Re-encodes a constant group in place for the (q - zp) * scale form. `zp` is the 4/8-bit zero point of the group.
// quants: the group's quantised values as one byte each (n entries); they are zeroed.
static inline void ggml_ov_make_constant_group(float bias, float & scale, uint8_t & zp, uint8_t * quants, size_t n) {
    scale = -bias;
    zp    = 1;
    if (quants && n) {
        memset(quants, 0, n);
    }
}

// the value the zero-point form dequantises a quant to
static inline float ggml_ov_affine_dequant(uint8_t q, uint8_t zp, float scale) {
    return ((float) q - (float) zp) * scale;
}
