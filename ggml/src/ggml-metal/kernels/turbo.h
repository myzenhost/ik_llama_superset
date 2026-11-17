#pragma once

// Included at the end of dequantize.h, after the ggml-common.h block definitions (block_turbo{2,3,4}_0, QK_TURBO*), so it declares no includes of its own.

// tests/test-turbo-metal-host.cpp compiles this header as plain C++ with a shim for the address-space keywords (device/thread/constant) and runs the
// writer and the reader against the CPU reference: keep it free of Metal-only constructs other than those keywords.
//
// TurboQuant KV (turbo2_0 / turbo3_0 / turbo4_0) for the Metal backend: constants, whole-group writer and 16-element reader.
//
// The tables below are the ONE Metal copy. They must stay identical to the CPU copy in ggml-turbo-quant.c, the CUDA copy in
// ggml-cuda/turbo-quant.cuh, the Vulkan copy in ggml-vulkan/vulkan-shaders/turbo_tables.glsl and the SYCL copy in ggml-sycl/turbo-quant.hpp: a
// KV block is written by whichever device the scheduler picks and read by another, and a diverging table decodes silently to the wrong
// vector. scripts/check-turbo-tables-consistent.py parses this file (ctest test-turbo-tables-consistent) and its --selftest proves it can fail.
//
// Block layout (ggml-common.h): 128 elements per block, stored in the ROTATED basis, value = centroid[idx] * norm:
//   turbo2 half norm; qs[32]  (2-bit)            turbo3 half norm; qs[32] (low 2 bits) + signs[16] (bit 2)      turbo4 half norm; qs[64] (4-bit, low nibble first)
// The readers below do NOT undo the rotation. The flash-attention wrapper in ggml-metal-ops.cpp rotates Q once before the kernel and the
// finished output once after it (kernel_turbo_wht), so (R q).(R k) = q.k and sum p_i (R v_i) = R sum p_i v_i hold exactly.
//
// DESIGN SOURCE: TheTom/llama-cpp-turboquant bcb85fc3ae85 ggml-metal.metal (MIT, read 2026-10-05). The tip's Metal quantiser works on 32-element
// sub-blocks with a per-sub-block norm and leaves the rotation to a separate kernel; this tree stores 128-element blocks with ONE corrected
// norm per block (ggml-turbo-quant.c quantize_row_turbo{2,3,4}_0_ref), so the writer here is that CPU algorithm step for step instead.

constant float TURBO_WHT_SIGNS1[128] = {
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f
};

constant float TURBO_WHT_SIGNS2[128] = {
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f
};

constant float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

constant float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

constant float TURBO_CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

constant float TURBO_MID_3BIT[7] = {
    -0.154496f, -0.092804f, -0.044243f, 0.0f,
     0.044243f,  0.092804f,  0.154496f
};

constant float TURBO_CENTROIDS_4BIT[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};

constant float TURBO_MID_4BIT[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f,
    -0.070671f, -0.046174f, -0.022824f,  0.000000f,
     0.022824f,  0.046174f,  0.070671f,  0.097164f,
     0.127026f,  0.162947f,  0.212203f
};

#ifndef TURBO_HOSTTEST
#define TURBO_NORM_LOAD(x)      float(x)
#define TURBO_NORM_STORE(d, v)  ((d) = half(v))
#endif

// First midpoint the value is below (ascending), else the top level: the thresholds of nearest_centroid_*bit in ggml-turbo-quant.c.
static inline int turbo_nearest_2bit(float v) {
    int idx = 3;
    for (int m = 0; m < 3; m++) {
        if (v < TURBO_MID_2BIT[m]) { idx = m; break; }
    }
    return idx;
}

static inline int turbo_nearest_3bit(float v) {
    int idx = 7;
    for (int m = 0; m < 7; m++) {
        if (v < TURBO_MID_3BIT[m]) { idx = m; break; }
    }
    return idx;
}

static inline int turbo_nearest_4bit(float v) {
    int idx = 15;
    for (int m = 0; m < 15; m++) {
        if (v < TURBO_MID_4BIT[m]) { idx = m; break; }
    }
    return idx;
}

// Whole 128-group writer: L2-normalise -> signs1 -> Walsh-Hadamard butterfly -> (1/sqrt(128)) * signs2 -> nearest centroid -> corrected norm.
// Same operation order as ggml-cuda/turbo-write.h turbo_write_group (sequential norm, sequential butterfly, sequential reconstruction norm), which
// tests/test-turbo-write-host.cpp pins to quantize_row_turbo*_ref. lq/ls are packed locally and stored once by the caller. Returns the fp32 block norm.
template <short NBITS>
static float turbo_write_group(device const float * src, thread uint8_t * lq, thread uint8_t * ls) {
    float buf[128];

    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        buf[j] = src[j];
        norm_sq += buf[j] * buf[j];
    }
    const float grp_norm = sqrt(norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    for (int j = 0; j < 128; j++) {
        buf[j] *= inv_norm;
    }

    for (int j = 0; j < 128; j++) {
        buf[j] *= TURBO_WHT_SIGNS1[j];
    }
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                const float a = buf[j];
                const float b = buf[j + h];
                buf[j]     = a + b;
                buf[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt = 0.08838834764831845f;
    for (int j = 0; j < 128; j++) {
        buf[j] *= inv_sqrt * TURBO_WHT_SIGNS2[j];
    }

    for (int j = 0; j < 64; j++) lq[j] = 0;
    for (int j = 0; j < 16; j++) ls[j] = 0;

    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        int   idx;
        float c;
        if (NBITS == 4) {
            idx = turbo_nearest_4bit(buf[j]);
            c   = TURBO_CENTROIDS_4BIT[idx];
            lq[j / 2] |= (uint8_t) ((idx & 0xF) << ((j % 2) * 4));
        } else if (NBITS == 3) {
            idx = turbo_nearest_3bit(buf[j]);
            c   = TURBO_CENTROIDS_3BIT[idx];
            lq[j / 4] |= (uint8_t) ((idx & 0x3) << ((j % 4) * 2));
            if (idx & 0x4) {
                ls[j / 8] |= (uint8_t) (1 << (j % 8));
            }
        } else {
            idx = turbo_nearest_2bit(buf[j]);
            c   = TURBO_CENTROIDS_2BIT[idx];
            lq[j / 4] |= (uint8_t) ((idx & 0x3) << ((j % 4) * 2));
        }
        recon_sq += c * c;
    }

    const float recon_norm = sqrt(recon_sq);
    return (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
}

static void quantize_turbo2_0(device const float * src, device block_turbo2_0 & dst) {
#pragma METAL fp math_mode(safe)
    uint8_t lq[64];
    uint8_t ls[16];
    const float norm = turbo_write_group<2>(src, lq, ls);
    TURBO_NORM_STORE(dst.norm, norm);
    for (int j = 0; j < QK_TURBO2 / 4; j++) {
        dst.qs[j] = lq[j];
    }
}

static void quantize_turbo3_0(device const float * src, device block_turbo3_0 & dst) {
#pragma METAL fp math_mode(safe)
    uint8_t lq[64];
    uint8_t ls[16];
    const float norm = turbo_write_group<3>(src, lq, ls);
    TURBO_NORM_STORE(dst.norm, norm);
    for (int j = 0; j < QK_TURBO3 / 4; j++) {
        dst.qs[j] = lq[j];
    }
    for (int j = 0; j < QK_TURBO3 / 8; j++) {
        dst.signs[j] = ls[j];
    }
}

static void quantize_turbo4_0(device const float * src, device block_turbo4_0 & dst) {
#pragma METAL fp math_mode(safe)
    uint8_t lq[64];
    uint8_t ls[16];
    const float norm = turbo_write_group<4>(src, lq, ls);
    TURBO_NORM_STORE(dst.norm, norm);
    for (int j = 0; j < QK_TURBO4 / 2; j++) {
        dst.qs[j] = lq[j];
    }
}

// 16 elements per call (il = 0..7): element j = il*16 + 4*i + k lands in reg[i][k]; value = centroid[idx(j)] * norm (rotated basis).
template <typename type4x4>
void dequantize_turbo2_0(device const block_turbo2_0 * xb, short il, thread type4x4 & reg) {
    const float norm = TURBO_NORM_LOAD(xb->norm);

    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            const int j   = il*16 + 4*i + k;
            const int idx = (xb->qs[j >> 2] >> ((j & 3) * 2)) & 0x3;
            reg_f[i][k] = TURBO_CENTROIDS_2BIT[idx] * norm;
        }
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_turbo3_0(device const block_turbo3_0 * xb, short il, thread type4x4 & reg) {
    const float norm = TURBO_NORM_LOAD(xb->norm);

    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            const int j   = il*16 + 4*i + k;
            const int idx = ((xb->qs[j >> 2] >> ((j & 3) * 2)) & 0x3) | (((xb->signs[j >> 3] >> (j & 7)) & 0x1) << 2);
            reg_f[i][k] = TURBO_CENTROIDS_3BIT[idx] * norm;
        }
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_turbo4_0(device const block_turbo4_0 * xb, short il, thread type4x4 & reg) {
    const float norm = TURBO_NORM_LOAD(xb->norm);

    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 4; k++) {
            const int j   = il*16 + 4*i + k;
            const int idx = (xb->qs[j >> 1] >> ((j & 1) * 4)) & 0xF;
            reg_f[i][k] = TURBO_CENTROIDS_4BIT[idx] * norm;
        }
    }

    reg = (type4x4) reg_f;
}
