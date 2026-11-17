#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#if defined(__ARM_NEON)
#define B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#define B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#define B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#define B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#define B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#define B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#define B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#define B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
static const uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#endif

// ---------------------------------------------------------------------------------------------
// DUPLICATE DEFINITIONS REMOVED 2026-10-20 (the arm64 twin of the x86 pass, ../x86/quants.c:788-811).
// ggml-base's ggml-quants.c defines the same 24 symbols (q4_0..q8_0, q2_K..q6_K, iq1_s..iq4_xs,
// tq1_0/tq2_0 vec_dots and quantize_row_q8_0/q8_1/q8_K), so every aarch64 executable failed to link:
//     ld: libggml-base.a(ggml-quants.c.o): multiple definition of `quantize_row_q8_0';
//         libggml-cpu.a(quants.c.o): first defined here                  ... 24 in total
// The fork's ggml-quants.c copies survive (several call iqk_* or carry fork changes), as on x86.
// KEPT HERE: the _generic fallbacks and the symbols only this file provides (q1_0, q2_0, mxfp4,
// nvfp4 vec_dots). Found by the qemu-user aarch64 lane (engq-c-1019).
// ---------------------------------------------------------------------------------------------
// placeholder implementation for Apple targets
//===================================== Dot products =================================

void ggml_vec_dot_q1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK1_0;  // 128
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q1_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        // Process 4 Q8_0 blocks (each has 32 elements)
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);

            // Get the 4 bytes of bits for this Q8_0 block (32 bits = 4 bytes)
            // Bits are at offset k*4 bytes in x[i].qs
            const uint8_t * bits = &x[i].qs[k * 4];

            // Load 32 int8 values from y
            const int8x16_t y0 = vld1q_s8(yb->qs);
            const int8x16_t y1 = vld1q_s8(yb->qs + 16);

            // Byte 0-1: bits for y0[0..15]
            const uint64_t expand0 = table_b2b_0[bits[0]];
            const uint64_t expand1 = table_b2b_0[bits[1]];
            // Byte 2-3: bits for y1[0..15]
            const uint64_t expand2 = table_b2b_0[bits[2]];
            const uint64_t expand3 = table_b2b_0[bits[3]];

            // Build the sign vectors by reinterpreting the table values
            uint8x8_t e0 = vcreate_u8(expand0);
            uint8x8_t e1 = vcreate_u8(expand1);
            uint8x8_t e2 = vcreate_u8(expand2);
            uint8x8_t e3 = vcreate_u8(expand3);

            // Shift right by 4 to get 0 or 1
            int8x8_t s0 = vreinterpret_s8_u8(vshr_n_u8(e0, 4));
            int8x8_t s1 = vreinterpret_s8_u8(vshr_n_u8(e1, 4));
            int8x8_t s2 = vreinterpret_s8_u8(vshr_n_u8(e2, 4));
            int8x8_t s3 = vreinterpret_s8_u8(vshr_n_u8(e3, 4));

            // Convert 0/1 to -1/+1: sign = 2*val - 1
            int8x8_t one = vdup_n_s8(1);
            s0 = vsub_s8(vadd_s8(s0, s0), one);  // 2*s0 - 1
            s1 = vsub_s8(vadd_s8(s1, s1), one);
            s2 = vsub_s8(vadd_s8(s2, s2), one);
            s3 = vsub_s8(vadd_s8(s3, s3), one);

            // Combine into 16-element vectors
            int8x16_t signs0 = vcombine_s8(s0, s1);
            int8x16_t signs1 = vcombine_s8(s2, s3);

            // Multiply signs with y values and accumulate
            // dot(signs, y) where signs are +1/-1
            int32x4_t p0 = ggml_vdotq_s32(vdupq_n_s32(0), signs0, y0);
            int32x4_t p1 = ggml_vdotq_s32(p0, signs1, y1);

            // Scale by d1 and accumulate
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p1), d0 * d1);
        }
    }

    *s = vaddvq_f32(sumv);
#else
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK2_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0f;

#if defined(__ARM_NEON)
    // Replicate pattern: each byte repeated 4 times
    static const uint8_t tbl_idx_lo[16] = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3};
    static const uint8_t tbl_idx_hi[16] = {4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7};
    // Right-shift amounts: 0,2,4,6 repeated for each group of 4
    static const int8_t shift_vals[16] = {0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6};

    const uint8x16_t idx_lo  = vld1q_u8(tbl_idx_lo);
    const uint8x16_t idx_hi  = vld1q_u8(tbl_idx_hi);
    const int8x16_t  shifts  = vld1q_s8(shift_vals);
    const uint8x16_t mask2   = vdupq_n_u8(0x03);
    const int8x16_t  one     = vdupq_n_s8(1);

    float32x4_t sumv = vdupq_n_f32(0.0f);

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        // group 64: one Q2_0 block (64 weights) maps to two Q8_0 blocks (2 * 32 = 64)
        for (int k = 0; k < 2; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 2 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);

            // Load 8 bytes of packed 2-bit values
            const uint8x8_t raw = vld1_u8(&x[i].qs[k * 8]);
            const uint8x16_t raw16 = vcombine_u8(raw, raw);

            // First 16 elements: replicate bytes 0-3, shift, mask, subtract 1
            uint8x16_t bytes0 = ggml_vqtbl1q_u8(raw16, idx_lo);
            int8x16_t qv0 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bytes0, shifts), mask2)),
                one);

            // Second 16 elements: replicate bytes 4-7, shift, mask, subtract 1
            uint8x16_t bytes1 = ggml_vqtbl1q_u8(raw16, idx_hi);
            int8x16_t qv1 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshlq_u8(bytes1, shifts), mask2)),
                one);

            // Load Q8_0 values and dot product
            const int8x16_t y0 = vld1q_s8(yb->qs);
            const int8x16_t y1 = vld1q_s8(yb->qs + 16);

            int32x4_t p0 = ggml_vdotq_s32(vdupq_n_s32(0), qv0, y0);
            int32x4_t p1 = ggml_vdotq_s32(p0, qv1, y1);

            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(p1), d0 * d1);
        }
    }

    sumf = vaddvq_f32(sumv);
#else
    ggml_vec_dot_q2_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
    return;
#endif

    *s = sumf;
}

void ggml_vec_dot_mxfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    const block_mxfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_MXFP4;

    int ib = 0;
    float sumf = 0;

#if defined __ARM_NEON
    const int8x16_t values = vld1q_s8(kvalues_mxfp4);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    uint8x16x2_t q4bits;
    int8x16x4_t q4b;
    int8x16x4_t q8b;
    int32x4_t prod_1;
    int32x4_t prod_2;

    for (; ib + 1 < nb; ib += 2) {
        q4bits.val[0] = vld1q_u8(x[ib + 0].qs);
        q4bits.val[1] = vld1q_u8(x[ib + 1].qs);
        q8b.val[0]    = vld1q_s8(y[ib + 0].qs);
        q8b.val[1]    = vld1q_s8(y[ib + 0].qs + 16);
        q8b.val[2]    = vld1q_s8(y[ib + 1].qs);
        q8b.val[3]    = vld1q_s8(y[ib + 1].qs + 16);

        q4b.val[0] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[0], m4b));
        q4b.val[1] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
        q4b.val[2] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[1], m4b));
        q4b.val[3] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));

        prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[0], q8b.val[0]), q4b.val[1], q8b.val[1]);
        prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[2], q8b.val[2]), q4b.val[3], q8b.val[3]);

        sumf +=
            GGML_E8M0_TO_FP32_HALF(x[ib + 0].e) * GGML_CPU_FP16_TO_FP32(y[ib + 0].d) * vaddvq_s32(prod_1) +
            GGML_E8M0_TO_FP32_HALF(x[ib + 1].e) * GGML_CPU_FP16_TO_FP32(y[ib + 1].d) * vaddvq_s32(prod_2);
    }

#endif
    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_E8M0_TO_FP32_HALF(x[ib].e);
        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    // Each NVFP4 super-block (64 elements) spans 2 q8_0 blocks
    const int nb = n / QK_NVFP4;

    float sumf = 0;

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
    const int8x16_t values = vld1q_s8(kvalues_mxfp4);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int ib = 0; ib < nb; ++ib) {
        const uint8x16_t q4bits_0 = vld1q_u8(x[ib].qs);
        const uint8x16_t q4bits_1 = vld1q_u8(x[ib].qs + 16);

        const int8x16_t q4_lo_0 = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits_0, m4b));
        const int8x16_t q4_hi_0 = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits_0, 4));
        const int8x16_t q4_lo_1 = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits_1, m4b));
        const int8x16_t q4_hi_1 = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits_1, 4));

#if defined(__ARM_FEATURE_DOTPROD)
        const int8x16_t q8_0a = vld1q_s8(y[2*ib].qs);
        const int8x16_t q8_0b = vld1q_s8(y[2*ib].qs + 16);
        const int8x16_t q8_lo_0 = vcombine_s8(vget_low_s8(q8_0a), vget_low_s8(q8_0b));
        const int8x16_t q8_hi_0 = vcombine_s8(vget_high_s8(q8_0a), vget_high_s8(q8_0b));

        const int8x16_t q8_1a = vld1q_s8(y[2*ib+1].qs);
        const int8x16_t q8_1b = vld1q_s8(y[2*ib+1].qs + 16);
        const int8x16_t q8_lo_1 = vcombine_s8(vget_low_s8(q8_1a), vget_low_s8(q8_1b));
        const int8x16_t q8_hi_1 = vcombine_s8(vget_high_s8(q8_1a), vget_high_s8(q8_1b));

        const int32x4_t p0 = vaddq_s32(
            vdotq_s32(vdupq_n_s32(0), q4_lo_0, q8_lo_0),
            vdotq_s32(vdupq_n_s32(0), q4_hi_0, q8_hi_0));
        const int32x4_t p1 = vaddq_s32(
            vdotq_s32(vdupq_n_s32(0), q4_lo_1, q8_lo_1),
            vdotq_s32(vdupq_n_s32(0), q4_hi_1, q8_hi_1));

        const int32x4_t sumi = vpaddq_s32(p0, p1);
#else
        const int8x8_t q4_0_lo = vget_low_s8(q4_lo_0);
        const int8x8_t q4_0_hi = vget_low_s8(q4_hi_0);
        const int8x8_t q4_1_lo = vget_high_s8(q4_lo_0);
        const int8x8_t q4_1_hi = vget_high_s8(q4_hi_0);
        const int8x8_t q4_2_lo = vget_low_s8(q4_lo_1);
        const int8x8_t q4_2_hi = vget_low_s8(q4_hi_1);
        const int8x8_t q4_3_lo = vget_high_s8(q4_lo_1);
        const int8x8_t q4_3_hi = vget_high_s8(q4_hi_1);

        const int8x8_t q8_0_lo = vld1_s8(y[2*ib].qs);
        const int8x8_t q8_0_hi = vld1_s8(y[2*ib].qs + 8);
        const int8x8_t q8_1_lo = vld1_s8(y[2*ib].qs + 16);
        const int8x8_t q8_1_hi = vld1_s8(y[2*ib].qs + 24);
        const int8x8_t q8_2_lo = vld1_s8(y[2*ib+1].qs);
        const int8x8_t q8_2_hi = vld1_s8(y[2*ib+1].qs + 8);
        const int8x8_t q8_3_lo = vld1_s8(y[2*ib+1].qs + 16);
        const int8x8_t q8_3_hi = vld1_s8(y[2*ib+1].qs + 24);

        const int32x4_t sumi = vld1q_s32(((const int32_t[4]) {
            vaddvq_s32(ggml_nvfp4_dot8(q4_0_lo, q8_0_lo, q4_0_hi, q8_0_hi)),
            vaddvq_s32(ggml_nvfp4_dot8(q4_1_lo, q8_1_lo, q4_1_hi, q8_1_hi)),
            vaddvq_s32(ggml_nvfp4_dot8(q4_2_lo, q8_2_lo, q4_2_hi, q8_2_hi)),
            vaddvq_s32(ggml_nvfp4_dot8(q4_3_lo, q8_3_lo, q4_3_hi, q8_3_hi)),
        }));
#endif

        const float dy0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d);
        const float dy1 = GGML_CPU_FP16_TO_FP32(y[2*ib+1].d);
        const float32x4_t nvsc = vld1q_f32(((const float[4]) {
            GGML_CPU_UE4M3_TO_FP32(x[ib].d[0]),
            GGML_CPU_UE4M3_TO_FP32(x[ib].d[1]),
            GGML_CPU_UE4M3_TO_FP32(x[ib].d[2]),
            GGML_CPU_UE4M3_TO_FP32(x[ib].d[3])
        }));
        const float32x4_t scales = vmulq_f32(nvsc, vld1q_f32(((const float[4]) {dy0, dy0, dy1, dy1})));

        acc = vfmaq_f32(acc, vcvtq_f32_s32(sumi), scales);
    }
    sumf = vaddvq_f32(acc);
#else
    for (int ib = 0; ib < nb; ++ib) {
        for (int si = 0; si < 4; ++si) {
            const float d = ggml_ue4m3_to_fp32(x[ib].d[si]);
            const int q8b = si / 2;
            const int q8o = (si % 2) * QK_NVFP4_SUB;
            const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8b].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[si*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8b].qs[q8o + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8b].qs[q8o + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }
            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
#endif
    *s = sumf;
}

#ifdef __ARM_FEATURE_SVE
static inline svuint32_t ggml_decode_q4scales_and_mins_for_mmla(const uint32_t * vx_scales) {
    const svbool_t pg_all   = svptrue_pat_b32(SV_VL4);
    const svbool_t pg_false = svpfalse_b();            // 0x0000
    const svbool_t pg_lo_8  = svwhilelt_b8_s32(0,  8); // 0x00ff
    const svbool_t pg_odd   = svzip1_b32(pg_false, pg_lo_8);

    svuint32_t vutmp_hi, vutmp_lo;
    svuint32_t vx01 = svld1_u32(pg_lo_8, vx_scales);
    vutmp_hi = svzip1_u32(vx01, vx01);
    vutmp_hi = svlsr_n_u32_m(pg_odd, vutmp_hi, 2);
    vutmp_hi = svreinterpret_u32_u64(svand_n_u64_x(pg_all, svreinterpret_u64_u32(vutmp_hi), UINT64_C(0x303030303f3f3f3f)));
    const svuint32_t vx2 = svdup_u32(vx_scales[2]);
    vutmp_lo = svlsr_u32_x(pg_all, vx2, svreinterpret_u32_s32(svindex_s32(-2, 2)));
    vutmp_lo = svand_n_u32_z(pg_odd, vutmp_lo, UINT32_C(0x0f0f0f0f));
    svuint32_t vutmp = svorr_u32_z(pg_all, vutmp_hi, vutmp_lo);
    return vutmp;
}
#endif

#if defined (__ARM_NEON)
static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#endif

