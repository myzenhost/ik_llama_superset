#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "vec.h"
#include "ggml-quants.h"
#include "quants.h"

#include "arch-fallback.h"
// quantize_row_mxfp4_ref and the two *_g128_ref quantisers live here. The header is
// safe from C -- it carries an extern "C" guard and ggml.c and ggml-quants.c already
// include it.
#include "iqk/iqk_quantize.h"

#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

// Upstream Q1_0 / Q2_0 (types 400/401) -- NOT the fork's Q1_0_G128 / Q2_0_G128. These used to
// forward to the *_g128_ref quantisers, which is wrong for Q2_0: QK2_0 is 64 and the g128
// quantiser writes 128-element groups, while ggml_vec_dot_q2_0_q8_0 below reads 64. The refs
// are now ported from upstream into ggml-quants.c (row S69-UPSTREAM-Q1_0-Q2_0-DECLARED-WITHOUT-
// BASE-TRAITS-2026-09-25-01).
void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q1_0_ref(x, (block_q1_0 *) y, k);
}

void quantize_row_q2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q2_0_ref(x, (block_q2_0 *) y, k);
}









void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

void quantize_row_q8_1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_1_ref(x, y, k);
}



void quantize_row_nvfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_nvfp4_ref(x, y, k);
}

void quantize_row_f8_e4m3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_f8_e4m3_ref(x, y, k);
}

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization



//========================= 3-bit (de)-quantization



// ====================== 4-bit (de)-quantization



// ====================== 5-bit (de)-quantization



// ====================== 6-bit (de)-quantization



// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)





//===================================== Q8_K ==============================================

void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

void ggml_vec_dot_q1_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK1_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q1_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;

            const uint8_t * GGML_RESTRICT bits = &x[i].qs[k * 4];
            const int8_t  * GGML_RESTRICT qy   = yb->qs;

            for (int b = 0; b < 4; ++b, qy += 8) {
                const unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                           +  ((mask & 0x02) ? qy[1] : -qy[1])
                           +  ((mask & 0x04) ? qy[2] : -qy[2])
                           +  ((mask & 0x08) ? qy[3] : -qy[3])
                           +  ((mask & 0x10) ? qy[4] : -qy[4])
                           +  ((mask & 0x20) ? qy[5] : -qy[5])
                           +  ((mask & 0x40) ? qy[6] : -qy[6])
                           +  ((mask & 0x80) ? qy[7] : -qy[7]);
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q2_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);

        float sumi = 0.0f;

        // group 64: one Q2_0 block (64 weights) maps to two Q8_0 blocks (2 * 32 = 64)
        for (int k = 0; k < 2; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 2 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;

            const uint8_t * GGML_RESTRICT qs = &x[i].qs[k * 8];
            const int8_t  * GGML_RESTRICT qy = yb->qs;

            for (int b = 0; b < 8; ++b) {
                const uint8_t byte = qs[b];
                // Extract 4 two-bit values, map {0,1,2,3} -> {-1,0,1,2}
                sumi_block += ((int)((byte >> 0) & 3) - 1) * qy[b*4 + 0];
                sumi_block += ((int)((byte >> 2) & 3) - 1) * qy[b*4 + 1];
                sumi_block += ((int)((byte >> 4) & 3) - 1) * qy[b*4 + 2];
                sumi_block += ((int)((byte >> 6) & 3) - 1) * qy[b*4 + 3];
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

// TODO: add WASM SIMD
void ggml_vec_dot_q4_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

// NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
void ggml_vec_dot_nvfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_NVFP4;

    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = ggml_ue4m3_to_fp32(x[ib].d[s_idx]);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

void ggml_vec_dot_f8_e4m3_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const ggml_fp8_e4m3_t * GGML_RESTRICT x = vx;
    const float * GGML_RESTRICT y = vy;

    float sumf = 0.0f;
    int pos = 0;
#if defined(GGML_SIMD) && !defined(__ARM_FEATURE_SVE) && !defined(__riscv_v_intrinsic)
    GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
    const int np = n - n % GGML_F32_STEP;
    for (; pos < np; pos += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; ++j) {
            const int offset = pos + j*GGML_F32_EPR;
            GGML_F32_VEC ax;
#if defined(__AVX512F__) && defined(__AVX512BW__)
            const __m128i codes = _mm_loadu_si128((const __m128i *) (x + offset));
            ax = _mm512_i32gather_ps(_mm512_cvtepu8_epi32(codes), ggml_table_f32_f8_e4m3, sizeof(float));
#elif defined(__AVX2__) && !defined(__AVX512F__)
            const __m128i codes = _mm_loadl_epi64((const __m128i *) (x + offset));
            ax = _mm256_i32gather_ps(ggml_table_f32_f8_e4m3, _mm256_cvtepu8_epi32(codes), sizeof(float));
#else
            float values[GGML_F32_EPR];
            for (int k = 0; k < GGML_F32_EPR; ++k) {
                values[k] = ggml_table_f32_f8_e4m3[x[offset + k].bits];
            }
            ax = GGML_F32_VEC_LOAD(values);
#endif
            sum[j] = GGML_F32_VEC_FMA(sum[j], ax, GGML_F32_VEC_LOAD(y + offset));
        }
    }
    GGML_F32_VEC_REDUCE(sumf, sum);
#endif
    for (; pos < n; ++pos) {
        sumf += ggml_table_f32_f8_e4m3[x[pos].bits] * y[pos];
    }
    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot_tq1_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int sum = 0;

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        for (size_t l = 0; l < 4; ++l) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[sizeof(x->qs)*5 + l*sizeof(x->qh) + j];
            }
        }

        sumf += (float) sum * (GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
}

void ggml_vec_dot_tq2_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        sumf += (float) sumi * d;
    }

    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
}

void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const uint8_t  * GGML_RESTRICT sc = x[i].scales;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}

void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint8_t * GGML_RESTRICT signs = x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    *s = sumf;
}

void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8*(ls1 - 32);
            const float d2 = d4d8*(ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}

// ============================ 4-bit non-linear quants

// ---------------------------------------------------------------------------------------------
// DUPLICATE DEFINITIONS REMOVED 2026-09-13. The definitions that stood here duplicated ggml-quants.c,
// which is compiled into ggml-base, so BOTH archives defined them and every executable in the tree
// failed to link:
//
//     ld: ggml-base.a(ggml-quants.c.obj): multiple definition of `ggml_vec_dot_q4_0_q8_0';
//         ggml-cpu.a(quants.c.obj): first defined here                      ... 37 in total
//
// THE FORK'S COPIES ARE THE ONES THAT SURVIVE, and that direction is measured, not assumed. Ten of
// the thirty-seven call iqk_mul_mat / iqk_quantize_q4_0 / iqk_quantize_row_q8_K, or carry a
// commented-out upstream call beside the replacement -- ggml-quants.c:709 is
// `quantize_row_q4_0(...) { iqk_quantize_q4_0(x, y, k); //quantize_row_q4_0_ref(x, y, k); }`.
// Twenty-two more differ in body size by 2-4x. Removing THOSE instead would have swapped this
// fork's IQK kernels for upstream's generic ones with nothing in the build saying so.
//
// Before this change both definitions existed and LINK ORDER decided which one ran. This does not
// introduce a choice; it removes an accident.
//
// KEPT HERE: the _generic fallbacks, and the eight symbols this file uniquely provides --
// ggml_vec_dot_{mxfp4,nvfp4,q1_0,q2_0}_q8_0 and quantize_row_{mxfp4,nvfp4,q1_0,q2_0} -- which are
// this fork's own quant types and are referenced by ggml-cpu.c and ggml.c. Those references are why
// this file could not simply be dropped from the target; measured with nm -u, see
// temp/_cpuside_out.txt.
// ---------------------------------------------------------------------------------------------

// quantize_row_mxfp4 REMOVED 2026-09-13. It duplicated iqk/iqk_quantize.cpp:4205, and because both
// objects land in ggml-cpu.a every executable failed to link on it:
//     ld: ggml-cpu.a(iqk_quantize.cpp.obj): multiple definition of `quantize_row_mxfp4';
//         ggml-cpu.a(quants.c.obj): first defined here
// The copy that stood here was upstream's reference wrapper, `quantize_row_mxfp4_ref(x, y, k)`.
// The fork's routes through its own quantiser, `quantize_mxfp4(x, y, 1, k, ...)`. Same direction as
// the 37 removed from this file and arch/x86/quants.c: the fork's kernel is the one this fork
// exists for. Before this change both were defined and link order decided which ran.

// ---------------------------------------------------------------------------------------------
// PQ2_0 x q8_0 dot, compiled in THIS translation unit (ggml-cpu, built with AVX2) rather than in ggml-quants.c.
//
// ggml-quants.c belongs to ggml-base, which is built arch-neutral on purpose (see ggml/src/CMakeLists.txt), so the
// __AVX2__ branch of ggml_vec_dot_pq2_0_q8_0 there is never compiled and the scalar path ran: 2.12 G weights/s, 0.81x of the
// vectorised Q8_0 dot (tests/test-prism-dot-throughput.cpp, 2026-10-04). Same arithmetic as that branch, one q8_0 block per step.
// The CPU traits table points here; ggml.c's base table and the other callers keep the base-library function. Without __AVX2__
// this forwards to it, so nothing changes on a build without AVX2.
// ---------------------------------------------------------------------------------------------
#if defined(__AVX2__)
static inline float pq2_hsum_f8(const __m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    const __m128 hi = _mm256_extractf128_ps(x, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

static inline __m256i pq2_dot_u8s8(const __m256i u, const __m256i s) {
    // unsigned x signed byte pairs summed into int16 (codes 0..2 and ones: a pair is at most 2*2*127), then pairs of int16 into int32
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1));
}
#endif

void ggml_cpu_vec_dot_pq2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK_PQ2_0 == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_pq2_0 * GGML_RESTRICT x = vx;
    const block_q8_0  * GGML_RESTRICT y = vy;
    const int nb = n / QK_PQ2_0;

    const __m256i shifts  = _mm256_setr_epi64x(0, 2, 4, 6);
    const __m256i mask3   = _mm256_set1_epi8(3);
    const __m256i ones_8  = _mm256_set1_epi8(1);
    const __m256i qy_shuf = _mm256_setr_epi8(0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15,
                                             0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15);
    const __m256i qy_perm = _mm256_setr_epi32(0,4,1,5,2,6,3,7);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; i++) {
        const float dw = GGML_FP16_TO_FP32(x[i].d);
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[4 * i + k];
            const __m256i qy  = _mm256_loadu_si256((const __m256i *) yb->qs);
            const __m256i qyp = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(qy, qy_shuf), qy_perm);
            int64_t xq; memcpy(&xq, &x[i].qs[8 * k], sizeof(xq));
            const __m256i codes = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(xq), shifts), mask3);
            const __m256i s32 = _mm256_sub_epi32(pq2_dot_u8s8(codes, qyp), pq2_dot_u8s8(ones_8, qyp));
            const __m256 scale = _mm256_set1_ps(dw * GGML_FP16_TO_FP32(yb->d));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(scale, _mm256_cvtepi32_ps(s32)));
        }
    }
    *s = pq2_hsum_f8(acc);
#else
    ggml_vec_dot_pq2_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// ===== MXFP6 / MXFP8 x Q8_0 dot products: AVX2 (row S69-MXFP6-MXFP8-INTEGRATION-FTYPE-GGUFPY-SIMD-MSVC-BUILD-2026-10-08-01) =====
// ggml-quants.c holds the scalar reference (ggml_vec_dot_mxfp*_q8_0, one decode per element). These *_cpu entry points are what the CPU
// type-traits table points at. With AVX2+FMA each 8 element codes are decoded without a table: value = M * 2^(Ee - bias - mbits) with
// M = mantissa | hidden bit (hidden bit only when the exponent field E != 0) and Ee = max(E, 1), the scale built directly as a normal float.
// (A first version shifted the code bits into a float's exponent/mantissa and multiplied by 2^(127-bias); that is exact but makes every
// subnormal code a DENORMAL float, and denormal inputs cost a microcode assist: MEASURED 170 ns/block for E2M3 vs 90 for the scalar code.)
// FP6 has no Inf/NaN; E4M3's only NaN is S.1111.111 and E5M2's Inf/NaN are exponent 31, both patched in with blends.
// The sum order differs from the scalar reference (8 lanes, FMA), so equality is to fp32 rounding, not bitwise; tests/test-mx-quant.cpp
// compares every code at every position and random rows against the scalar reference. Without AVX2/FMA the *_cpu functions forward to it.
static inline float mx_cpu_scale(uint8_t e) {
    uint32_t u = e == 255 ? 0x7FC00000u : (e ? (uint32_t) e << 23 : 0x00400000u);
    float f; memcpy(&f, &u, sizeof f); return f;
}

#if defined(__AVX2__) && defined(__FMA__)
static inline float mx_hsum_f8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

// MSVC has neither __attribute__((always_inline)) nor __builtin_nanf/__builtin_inff.
#if defined(_MSC_VER) && !defined(__clang__)
#define MX_ALWAYS_INLINE __forceinline
#else
#define MX_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

// 8 codes (int32 lanes) -> floats. kind: 0 none, 1 E4M3, 2 E5M2. signbit = index of the sign bit in the code. exp_bias_shift = 127 - bias - mbits.
static MX_ALWAYS_INLINE __m256 mx_decode8(__m256i c, int mbits, int signbit, int magmask, int exp_bias_shift, int kind) {
    const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(c, _mm256_set1_epi32(1 << signbit)), 31 - signbit);
    const __m256i mag  = _mm256_and_si256(c, _mm256_set1_epi32(magmask));
    const __m256i E    = _mm256_srli_epi32(mag, mbits);
    const __m256i m    = _mm256_and_si256(mag, _mm256_set1_epi32((1 << mbits) - 1));
    const __m256i hid  = _mm256_andnot_si256(_mm256_cmpeq_epi32(E, _mm256_setzero_si256()), _mm256_set1_epi32(1 << mbits));
    const __m256i Ee   = _mm256_max_epi32(E, _mm256_set1_epi32(1));
    const __m256  sc   = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(Ee, _mm256_set1_epi32(exp_bias_shift)), 23));
    __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_or_si256(m, hid)), sc);
    if (kind == 1) {            // E4M3: S.1111.111 is NaN
        v = _mm256_blendv_ps(v, _mm256_set1_ps(NAN), _mm256_castsi256_ps(_mm256_cmpeq_epi32(mag, _mm256_set1_epi32(0x7F))));
    } else if (kind == 2) {     // E5M2: exponent 31 -> Inf (m == 0) / NaN
        v = _mm256_blendv_ps(v, _mm256_set1_ps(INFINITY), _mm256_castsi256_ps(_mm256_cmpeq_epi32(mag, _mm256_set1_epi32(0x7C))));
        v = _mm256_blendv_ps(v, _mm256_set1_ps(NAN), _mm256_castsi256_ps(_mm256_cmpgt_epi32(mag, _mm256_set1_epi32(0x7C))));
    }
    return _mm256_or_ps(v, _mm256_castsi256_ps(sign));
}

static inline float mx_dot_codes32(const uint8_t * codes, const block_q8_0 * y, int mbits, int signbit, int magmask, int exp_bias_shift, int kind) {
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < 4; ++k) {
        const __m256i c = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(codes + 8*k)));
        const __m256  q = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(y->qs + 8*k))));
        acc = _mm256_fmadd_ps(mx_decode8(c, mbits, signbit, magmask, exp_bias_shift, kind), q, acc);
    }
    return mx_hsum_f8(acc);
}

static inline void mx_unpack6_32(const uint8_t * qs, uint8_t * c) {
    for (int i = 0; i < 8; ++i) {
        const unsigned b0 = qs[3*i], b1 = qs[3*i + 1], b2 = qs[3*i + 2];
        c[4*i + 0] = (uint8_t)( b0 & 63);
        c[4*i + 1] = (uint8_t)(((b0 >> 6) | (b1 << 2)) & 63);
        c[4*i + 2] = (uint8_t)(((b1 >> 4) | (b2 << 4)) & 63);
        c[4*i + 3] = (uint8_t)( b2 >> 2);
    }
}

#define MX_CPU_DOT(NAME, BLOCK, IS6, MBITS, SIGNBIT, MAGMASK, EXPSHIFT, KIND)                                                       \
void ggml_vec_dot_##NAME##_q8_0_cpu(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,       \
                                    const void * GGML_RESTRICT vy, size_t by, int nrc) {                                      \
    assert(nrc == 1); assert(n % 32 == 0);                                                                                     \
    UNUSED(nrc); UNUSED(bs); UNUSED(bx); UNUSED(by);                                                                           \
    const BLOCK * GGML_RESTRICT x = (const BLOCK *) vx;                                                                        \
    const block_q8_0 * GGML_RESTRICT y = (const block_q8_0 *) vy;                                                              \
    float sumf = 0.0f;                                                                                                         \
    for (int i = 0; i < n/32; ++i) {                                                                                           \
        uint8_t c6[32];                                                                                                        \
        const uint8_t * codes = x[i].qs;                                                                                       \
        if (IS6) { mx_unpack6_32(x[i].qs, c6); codes = c6; }                                                                   \
        const float dot = mx_dot_codes32(codes, &y[i], MBITS, SIGNBIT, MAGMASK, EXPSHIFT, KIND);                                    \
        sumf += GGML_CPU_FP16_TO_FP32(y[i].d) * mx_cpu_scale(x[i].e) * dot;                                                    \
    }                                                                                                                          \
    *s = sumf;                                                                                                                 \
}

MX_CPU_DOT(mxfp6_e2m3, block_mxfp6_e2m3, 1, 3, 5, 0x1F, 127 - 1 - 3, 0)
MX_CPU_DOT(mxfp6_e3m2, block_mxfp6_e3m2, 1, 2, 5, 0x1F, 127 - 3 - 2, 0)
MX_CPU_DOT(mxfp8_e4m3, block_mxfp8_e4m3, 0, 3, 7, 0x7F, 127 - 7 - 3, 1)
MX_CPU_DOT(mxfp8_e5m2, block_mxfp8_e5m2, 0, 2, 7, 0x7F, 127 - 15 - 2, 2)
#else
#define MX_CPU_DOT_FWD(NAME)                                                                                                    \
void ggml_vec_dot_##NAME##_q8_0_cpu(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,       \
                                    const void * GGML_RESTRICT vy, size_t by, int nrc) {                                      \
    ggml_vec_dot_##NAME##_q8_0(n, s, bs, vx, bx, vy, by, nrc);                                                                 \
}
MX_CPU_DOT_FWD(mxfp6_e2m3)
MX_CPU_DOT_FWD(mxfp6_e3m2)
MX_CPU_DOT_FWD(mxfp8_e4m3)
MX_CPU_DOT_FWD(mxfp8_e5m2)
#endif

// ---------------------------------------------------------------------------------------------
// TQ3_1S / TQ4_1S x Q8_0 for the CPU type-traits table (types 74 / 75; TheTom/llama-cpp-turboquant ca28b4bc0, MIT).
// The reference dot in ggml-turbo-quant.c (ggml-base, compiled without -mavx2) dequantizes every weight block, inverse
// 32-point WHT included, once per (row, column) pair, scalar. MEASURED 2026-10-17, Qwen3-0.6B-Base, 8 threads, c=512:
// 92.5 s per perplexity chunk for TQ3_1S vs 1.04 s for Q4_0 (temp/w13-1017/w13_1017_ppl_probe_*.log).
// This is the same arithmetic in AVX2: 3-/4-bit index unpack by variable shifts, centroid lookup by permutevar8x32,
// the five butterfly stages in the scalar code's order (step 1, 2, 4 inside a register, 8 and 16 across registers; each
// stage computes a + b and a - b of the same operands as tq3_0_rht_inverse, so the dequantized block is bit-identical),
// then the (1/sqrt 32) * sign scale and an FMA dot against the Q8_0 block. Only the dot's summation order differs.
// The tables are copies of ggml-turbo-quant.c TQ3_0_CENTROIDS / TQ4_0_CENTROIDS / TQ3_0_SIGNS (:671-690 @ ca28b4bc0);
// tests/test-tq1s-quant.cpp compares this path against a double dequant-then-dot on every run, which catches a drifted table.
// ---------------------------------------------------------------------------------------------
#if defined(__AVX2__) && defined(__FMA__)
static const float tq1s_cpu_cent3[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};
static const float tq1s_cpu_cent4[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};
static const float tq1s_cpu_sign[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

// butterfly steps 1, 2, 4 inside one register of 8: lane j (low of its pair) gets a + b, lane j + step gets a - b
static MX_ALWAYS_INLINE __m256 tq1s_wht_in8(__m256 v) {
    __m256 s = _mm256_permute_ps(v, 0xB1);                            // [1,0,3,2]
    v = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xAA);
    s = _mm256_permute_ps(v, 0x4E);                                   // [2,3,0,1]
    v = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xCC);
    s = _mm256_permute2f128_ps(v, v, 0x01);                           // swap 128-bit halves
    v = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xF0);
    return v;
}

// w[0..3] = centroid * half-scale for elements 0..31 in order; returns sum_j invRHT(w)_j * q_j (q = Q8_0 ints, unscaled)
static MX_ALWAYS_INLINE __m256 tq1s_rht_dot(__m256 w0, __m256 w1, __m256 w2, __m256 w3, const int8_t * GGML_RESTRICT qs) {
    w0 = tq1s_wht_in8(w0); w1 = tq1s_wht_in8(w1); w2 = tq1s_wht_in8(w2); w3 = tq1s_wht_in8(w3);
    __m256 a, b;
    a = _mm256_add_ps(w0, w1); b = _mm256_sub_ps(w0, w1); w0 = a; w1 = b;   // step 8
    a = _mm256_add_ps(w2, w3); b = _mm256_sub_ps(w2, w3); w2 = a; w3 = b;
    a = _mm256_add_ps(w0, w2); b = _mm256_sub_ps(w0, w2); w0 = a; w2 = b;   // step 16
    a = _mm256_add_ps(w1, w3); b = _mm256_sub_ps(w1, w3); w1 = a; w3 = b;
    const __m256 inv = _mm256_set1_ps(0.17677669529663688f);
    w0 = _mm256_mul_ps(w0, _mm256_mul_ps(inv, _mm256_loadu_ps(tq1s_cpu_sign +  0)));
    w1 = _mm256_mul_ps(w1, _mm256_mul_ps(inv, _mm256_loadu_ps(tq1s_cpu_sign +  8)));
    w2 = _mm256_mul_ps(w2, _mm256_mul_ps(inv, _mm256_loadu_ps(tq1s_cpu_sign + 16)));
    w3 = _mm256_mul_ps(w3, _mm256_mul_ps(inv, _mm256_loadu_ps(tq1s_cpu_sign + 24)));
#define TQ1S_Q8(k) _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs + 8*(k)))))
    __m256 p = _mm256_mul_ps(w0, TQ1S_Q8(0));
    p = _mm256_fmadd_ps(w1, TQ1S_Q8(1), p);
    p = _mm256_fmadd_ps(w2, TQ1S_Q8(2), p);
    p = _mm256_fmadd_ps(w3, TQ1S_Q8(3), p);
#undef TQ1S_Q8
    return p;
}

void ggml_vec_dot_tq3_1s_q8_0_cpu(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
                                  const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1); assert(n % QK_TQ3_0 == 0);
    UNUSED(nrc); UNUSED(bs); UNUSED(bx); UNUSED(by);
    const block_tq3_1s * GGML_RESTRICT x = (const block_tq3_1s *) vx;
    const block_q8_0   * GGML_RESTRICT y = (const block_q8_0 *) vy;
    const __m256  cent  = _mm256_loadu_ps(tq1s_cpu_cent3);
    const __m256i shift = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256i m7    = _mm256_set1_epi32(7);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / QK_TQ3_0; ++i) {
        const __m256 d0 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d0));
        const __m256 d1 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d1));
        __m256 w[4];
        for (int g = 0; g < 4; ++g) {
            const uint8_t * qp = x[i].qs + 3*g;
            const int32_t bits = (int32_t) qp[0] | ((int32_t) qp[1] << 8) | ((int32_t) qp[2] << 16);
            const __m256i idx = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(bits), shift), m7);
            w[g] = _mm256_mul_ps(_mm256_permutevar8x32_ps(cent, idx), g < 2 ? d0 : d1);
        }
        const __m256 p = tq1s_rht_dot(w[0], w[1], w[2], w[3], y[i].qs);
        acc = _mm256_fmadd_ps(p, _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[i].d)), acc);
    }
    *s = mx_hsum_f8(acc);
}

void ggml_vec_dot_tq4_1s_q8_0_cpu(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
                                  const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1); assert(n % QK_TQ4_1S == 0);
    UNUSED(nrc); UNUSED(bs); UNUSED(bx); UNUSED(by);
    const block_tq4_1s * GGML_RESTRICT x = (const block_tq4_1s *) vx;
    const block_q8_0   * GGML_RESTRICT y = (const block_q8_0 *) vy;
    const __m256  clo   = _mm256_loadu_ps(tq1s_cpu_cent4);
    const __m256  chi   = _mm256_loadu_ps(tq1s_cpu_cent4 + 8);
    const __m256i shift = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i m15   = _mm256_set1_epi32(15);
    const __m256i seven = _mm256_set1_epi32(7);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n / QK_TQ4_1S; ++i) {
        const __m256 d0 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d0));
        const __m256 d1 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d1));
        __m256 w[4];
        for (int g = 0; g < 4; ++g) {
            int32_t bits; memcpy(&bits, x[i].qs + 4*g, 4);   // element 8g + k sits in nibble k of this little-endian word
            const __m256i idx = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(bits), shift), m15);
            const __m256  lo  = _mm256_permutevar8x32_ps(clo, idx);
            const __m256  hi  = _mm256_permutevar8x32_ps(chi, idx);
            const __m256  c   = _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx, seven)));
            w[g] = _mm256_mul_ps(c, g < 2 ? d0 : d1);
        }
        const __m256 p = tq1s_rht_dot(w[0], w[1], w[2], w[3], y[i].qs);
        acc = _mm256_fmadd_ps(p, _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[i].d)), acc);
    }
    *s = mx_hsum_f8(acc);
}
#else
void ggml_vec_dot_tq3_1s_q8_0_cpu(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
                                  const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_tq3_1s_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}
void ggml_vec_dot_tq4_1s_q8_0_cpu(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,
                                  const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_tq4_1s_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}
#endif
