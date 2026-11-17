// TurboQuant KV WRITER, whole 128-element group: L2-normalise -> rotate -> centroid quantise -> corrected norm.
//
// WHY THIS FILE EXISTS. The device functions in turbo-quant.cuh (quantize_f32_turbo{2,3}_0_block) only
// centroid-quantise input that is ALREADY normalised and rotated; they never write the block norm. Nothing
// called them (task row S69-TURBO-KV-CUDA-WRITER-NEVER-WIRED-QUANTIZE-HAS-ZERO-CALLERS-2026-09-29-01), so a
// served model had no writer for the KV cache at all. This header is the complete per-group writer that
// cpy.cu (GGML_OP_CPY F32 -> turbo) and set-rows.cu (GGML_OP_SET_ROWS -> turbo) call.
//
// SOURCE-READ: ggml/src/ggml-turbo-quant.c quantize_row_turbo3_0_ref (:288-345) and quantize_row_turbo2_0_ref
// (:383-430), turbo_cpu_fwht (:187-207). This is a re-expression of that algorithm with the same operation
// order (sequential norm accumulation, sequential butterfly, sequential recon-norm accumulation); the tables
// arrive as pointers so ONE definition of the algorithm serves two builds:
//   * nvcc device code, tables = the __constant__ arrays in turbo-quant.cuh,
//   * plain host C++ (tests/test-turbo-write-host.cpp), tables = host copies, so the exact logic the GPU runs
//     is executed on the CPU and compared with quantize_row_turbo3_0_ref BEFORE any GPU time is spent.
//
// Block layout at QK_TURBO{2,3,4} == 128 (ggml-common.h:719-749):
//   turbo3: half norm; uint8 qs[32] (low 2 bits of the 3-bit index, 4 per byte); uint8 signs[16] (bit 2, 8 per byte)
//   turbo2: half norm; uint8 qs[32] (2-bit index, 4 per byte)
//   turbo4: half norm; uint8 qs[64] (4-bit index, 2 per byte, low nibble first)   -- added 2026-10-05 with the 4-bit layout
// One 128-group == one block == one attention head's K (or V) row slice, so a group is never split.
//
// InnerQ per-channel scaling is NOT applied here: the d_innerq_* globals stay at their identity defaults
// (calibration off), which is what the fused read kernel assumes. Adding it means adding it to BOTH sides.

#pragma once

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define TURBO_WRITE_HD __host__ __device__
#else
#define TURBO_WRITE_HD
#endif

#define TURBO_WRITE_GROUP 128

struct turbo_write_tables {
    const float * signs1;    // [128]  applied before the butterfly
    const float * signs2;    // [128]  applied after the butterfly (together with 1/sqrt(128))
    const float * mid;       // NBITS=4: 15 ascending midpoints; NBITS=3: 7; NBITS=2: 3
    const float * centroid;  // NBITS=4: 16 centroids;           NBITS=3: 8; NBITS=2: 4
};

// Writes the quantised indices of one group into qs (and signs for NBITS == 3; pass nullptr for NBITS == 2)
// and returns the corrected block norm (fp32; the caller converts to fp16).
template <int NBITS>
static TURBO_WRITE_HD inline float turbo_write_group(const float * src, uint8_t * qs, uint8_t * signs,
                                                     const turbo_write_tables & t) {
    static_assert(NBITS == 2 || NBITS == 3 || NBITS == 4, "turbo_write_group handles the 2-, 3- and 4-bit layouts");
    constexpr int N  = TURBO_WRITE_GROUP;
    constexpr int NC = 1 << NBITS;

    float buf[N];

    // 1. L2 norm over the group, accumulated in element order
    float norm_sq = 0.0f;
    for (int j = 0; j < N; j++) {
        buf[j] = src[j];
        norm_sq += buf[j] * buf[j];
    }
    const float grp_norm = sqrtf(norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // 2. normalise
    for (int j = 0; j < N; j++) buf[j] *= inv_norm;

    // 3. forward rotation: signs1 -> Walsh-Hadamard butterfly -> (1/sqrt(128)) * signs2
    for (int j = 0; j < N; j++) buf[j] *= t.signs1[j];
    for (int h = 1; h < N; h *= 2) {
        for (int i = 0; i < N; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                const float a = buf[j];
                const float b = buf[j + h];
                buf[j]     = a + b;
                buf[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt = 0.08838834764831845f;
    for (int j = 0; j < N; j++) buf[j] *= inv_sqrt * t.signs2[j];

    // 4. nearest centroid via the midpoints (first midpoint the value is below), packed into LOCAL byte arrays and
    //    stored once at the end: the destination is global memory, and a per-element |= there is a load and a store
    //    per index.
    // lq is sized for the widest layout: 4-bit packs two indices per byte (64 bytes), 2/3-bit pack four (32 bytes).
    uint8_t lq[N / 2];
    uint8_t ls[N / 8];
    for (int j = 0; j < N / 2; j++) lq[j] = 0;
    for (int j = 0; j < N / 8; j++) ls[j] = 0;
    float recon_sq = 0.0f;
    for (int j = 0; j < N; j++) {
        int idx = NC - 1;
        for (int m = 0; m < NC - 1; m++) {
            if (buf[j] < t.mid[m]) { idx = m; break; }
        }
        if constexpr (NBITS == 4) {
            // 4-bit PolarQuant (TURBO4_USE_4BIT=1, 66-byte block): nibble pack, low nibble first, as quantize_row_turbo4_0_ref.
            lq[j / 2] |= (uint8_t)((idx & 0xF) << ((j % 2) * 4));
        } else {
            lq[j / 4] |= (uint8_t)((idx & 0x3) << ((j % 4) * 2));
        }
        if constexpr (NBITS == 3) {
            if (idx & 0x4) {
                ls[j / 8] |= (uint8_t)(1 << (j % 8));
            }
        }
        recon_sq += t.centroid[idx] * t.centroid[idx];
    }
    constexpr int NQ = (NBITS == 4) ? N / 2 : N / 4;
    for (int j = 0; j < NQ; j++) qs[j] = lq[j];
    if constexpr (NBITS == 3) {
        for (int j = 0; j < N / 8; j++) signs[j] = ls[j];
    }

    // 5. corrected norm: original group norm over the norm of what the centroids reconstruct
    const float recon_norm = sqrtf(recon_sq);
    return (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
}
