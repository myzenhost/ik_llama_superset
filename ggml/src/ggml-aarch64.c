// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"   // CPU-side declarations (ggml_graph_plan/compute, ggml_cpu_has_*, ggml_get/set_*) moved here, per upstream

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#include "ggml-aarch64.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#elif defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define UNUSED GGML_UNUSED

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//
static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave, unsigned int xor_mask) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < QK4_0 * 2; i++) {
        int src_offset = (i / (4 * blck_size_interleave)) * blck_size_interleave;
        int src_id = (i % (4 * blck_size_interleave)) / blck_size_interleave;
        src_offset += (i % blck_size_interleave);

        out.qs[i] = in[src_id].qs[src_offset] ^ xor_mask;
    }

    return out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
static block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned int blck_size_interleave, unsigned int xor_mask) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    for (int i = 0; i < QK4_0 * 4; i++) {
        int src_offset = (i / (8 * blck_size_interleave)) * blck_size_interleave;
        int src_id = (i % (8 * blck_size_interleave)) / blck_size_interleave;
        src_offset += (i % blck_size_interleave);

        out.qs[i] = in[src_id].qs[src_offset] ^ xor_mask;
    }

    return out;
}

void quantize_q8_0_4x4(const float * restrict x, void * restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * restrict y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 3] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 11] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 15] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

void quantize_q8_0_4x8(const float * restrict x, void * restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * restrict y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 4; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][2 * j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][2 * j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][2 * j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

void quantize_mat_q8_0(const float * restrict x, void * restrict vy, int64_t nrow, int64_t n_per_row, int64_t blck_size_interleave) {
    assert(nrow == 4);
    UNUSED(nrow);
    if (blck_size_interleave == 4) {
        quantize_q8_0_4x4(x, vy, n_per_row);
    } else if (blck_size_interleave == 8) {
        quantize_q8_0_4x8(x, vy, n_per_row);
    } else {
        assert(false);
    }
}

static size_t quantize_q4_0_nr_bl(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, int nrows_interleaved, int blck_size_interleave) {
    assert(n_per_row % QK4_0 == 0);
    const int nb = n_per_row / QK4_0;

    void * out_ptr = NULL;
    if (nrows_interleaved == 8) {
        out_ptr = (block_q4_0x8 *) dst;
    }
    else if (nrows_interleaved == 4) {
        out_ptr = (block_q4_0x4 *) dst;
    }
    assert(nrows_interleaved <= 8);
    block_q4_0 dst_tmp[8];

    for (int b = 0; b < (nrow * n_per_row); b += nrows_interleaved * n_per_row) {

        for (int64_t x = 0; x < nb; x++) {

            for (int i  = 0; i < nrows_interleaved; i++ ) {
                quantize_row_q4_0_ref(src + b + i * n_per_row + x * QK4_0, (block_q4_0 *) dst_tmp + i, QK4_0);
            }

            if (nrows_interleaved == 8) {
                *(block_q4_0x8 *) out_ptr = make_block_q4_0x8(dst_tmp, blck_size_interleave, 0x88);
                out_ptr = (block_q4_0x8 *) out_ptr + 1;
            }
            else if (nrows_interleaved == 4) {
                *(block_q4_0x4 *) out_ptr = make_block_q4_0x4(dst_tmp, blck_size_interleave, 0x88);
                out_ptr = (block_q4_0x4 *) out_ptr + 1;
            }
        }
    }

    return ((nrow * n_per_row) / QK4_0 * sizeof(block_q4_0));
}

size_t quantize_q4_0_4x4(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 4, 4);
    }
    else {
        assert(false);
        return 0;
    }
}

size_t quantize_q4_0_4x8(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 4, 8);
    }
    else {
        assert(false);
        return 0;
    }
}

size_t quantize_q4_0_8x8(const float * restrict src, void * restrict dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    if (!quant_weights) {
        return quantize_q4_0_nr_bl(src, dst, nrow, n_per_row, 8, 8);
    }
    else {
        assert(false);
        return 0;
    }
}


// ---------------------------------------------------------------------------------------------
// THE SIX ggml_gemv_/ggml_gemm_q4_0_* DEFINITIONS THAT USED TO FOLLOW WERE REMOVED 2026-09-10.
//
// They were duplicates. ggml-cpu/repack.cpp defines the same six C-spelled symbols, and because
// this file is compiled into ggml-base while repack.cpp is compiled into ggml-cpu, EVERY executable
// in the tree failed to link:
//
//     ld: ggml-base.a(ggml-aarch64.c.obj): multiple definition of `ggml_gemv_q4_0_4x4_q8_0';
//         ggml-cpu.a(repack.cpp.obj): first defined here
//
// That is why there was no runnable test binary for the autodiff work, and why a hand link with
// -Wl,--allow-multiple-definition produced a binary that computed ggml_sum(ggml_add(4,3)) = 0.
//
// This file is upstream's OLD ggml-aarch64.c. Upstream renamed it to ggml-cpu/repack.cpp, which
// this tree already compiles (ggml/src/ggml-cpu/CMakeLists.txt:32) and which carries both the
// _generic bodies and the dispatch that calls the non-generic spelling (repack.cpp:3964, :4061).
// The merge kept this file under the comment "None of these exist in upstream at any path" --
// true by PATH, false by CONTENT, because a path check cannot see a rename.
//
// The file is NOT deleted, because ggml.c still calls four C-spelled symbols that only it defines:
// quantize_mat_q8_0, quantize_q4_0_4x4, quantize_q4_0_4x8 and quantize_q4_0_8x8 (measured with
// nm -u over the three built archives). upstream's repack.cpp does that work behind C++ template
// instantiations, a spelling a C caller cannot bind to. Those four, and the two unreferenced
// quantize_q8_0_4x* beside them, stay above.
//
// NOT CLAIMED: that repack.cpp's six implementations are bit-identical to the ones removed here.
// What IS true is that ggml-cpu already dispatched to repack.cpp's versions, so before this change
// the same six operations had two different implementations in one process and which one ran
// depended on link order. A numeric equivalence check between them is filed separately.
// ---------------------------------------------------------------------------------------------
