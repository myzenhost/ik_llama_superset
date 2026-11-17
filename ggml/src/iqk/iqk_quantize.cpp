//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#if GGML_USE_IQK_MULMAT
#include "iqk_mul_mat.h"
#endif
#include "ggml-quants.h"
#include "ggml-impl.h"
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "iqk_quantize.h"
#include "iqk_config.h"

#include "iqk_gemm_ktquants.h"

#include <climits>
#include <vector>
#include <utility>
#include <cstdint>
#include <climits>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <random>
#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <string>
#include <functional>

// HALF 2 (S69-GGML-BASE-DLL-CANNOT-LINK-CIRCULAR-ACROSS-A-DLL-BOUNDARY-2026-09-15-01): the type_traits[] .to_float / .from_float_ref
// implementations with an arch-neutral call closure now live in iqk_quantize_base.cpp (ggml-base). The internal-linkage helpers both
// halves need are in iqk_quantize_shared.h (a private copy in each translation unit).
#include "iqk_quantize_shared.h"
#include "iqk_scale_search.h"

namespace {












static inline int num_rows([[maybe_unused]] ggml_type type) {
#ifdef HAVE_FANCY_SIMD
    switch (type) {
        case GGML_TYPE_Q2_K_R4:
        case GGML_TYPE_Q3_K_R4:
        case GGML_TYPE_Q6_K_R4:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ2_XXS_R4:
        case GGML_TYPE_IQ2_XS_R4:
        case GGML_TYPE_IQ2_S_R4:
        case GGML_TYPE_IQ3_XXS_R4:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
        case GGML_TYPE_IQ3_S_R4: return 4;
        case GGML_TYPE_IQ4_NL_R4:
        case GGML_TYPE_Q5_0_R4:
        case GGML_TYPE_Q6_0_R4:
        case GGML_TYPE_IQ2_BN_R4:
        case GGML_TYPE_IQ4_XS_R8:
        case GGML_TYPE_Q4_K_R4:
        case GGML_TYPE_Q5_K_R4:
        case GGML_TYPE_Q8_KV:
        case GGML_TYPE_Q8_KV_R8:
        case GGML_TYPE_Q8_K_R8:
        case GGML_TYPE_Q2_0_R8: return 8;
        case GGML_TYPE_Q4_0_R8:
        case GGML_TYPE_Q8_0_R8:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q8_K_R16:
        case GGML_TYPE_BF16_R16: return 16;
        default: return 1;
    }
#else
    switch (type) {
        case GGML_TYPE_Q2_K_R4:
        case GGML_TYPE_Q3_K_R4:
        case GGML_TYPE_Q4_K_R4:
        case GGML_TYPE_Q5_K_R4:
        case GGML_TYPE_Q6_K_R4:
        case GGML_TYPE_Q5_0_R4:
        case GGML_TYPE_Q6_0_R4:
        case GGML_TYPE_IQ4_NL_R4:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ2_XXS_R4:
        case GGML_TYPE_IQ2_XS_R4:
        case GGML_TYPE_IQ2_S_R4:
        case GGML_TYPE_IQ3_XXS_R4:
        case GGML_TYPE_IQ3_S_R4:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
        case GGML_TYPE_IQ2_BN_R4: return 4;
        case GGML_TYPE_IQ4_XS_R8:
        case GGML_TYPE_Q4_0_R8:
        case GGML_TYPE_Q8_0_R8:
        case GGML_TYPE_Q8_KV:
        case GGML_TYPE_Q8_KV_R8:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q8_K_R8:
        case GGML_TYPE_Q2_0_R8: return 8;
        case GGML_TYPE_Q8_K_R16:
        case GGML_TYPE_BF16_R16: return 16;
        default: return 1;
    }
#endif
}


}

void iqk_quantize_any(int from_type, int to_type,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                      uint64_t nb0, uint64_t nb1, uint64_t nb2, uint64_t nb3,
                      const void * x, void * y, void * work_buffer,
                      to_float_t to_float, from_float_t from_float, int ith, int nth) {
    auto type_x = ggml_type(from_type);
    GGML_ASSERT(ggml_type_size(type_x) == nb0);
    auto type_y = ggml_type(to_type);
    auto row_size_y = ggml_row_size(type_y, ne0);
    auto n_interleaved = num_rows(type_y);
    // Row-interleaved Q8_1 (8 or 16 rows per group) only exists for a src1 whose row count fills a group. A narrower F16 src1 (decode, n < 8)
    // is quantized one row at a time, the layout the plain-Q8_1 kernels read. tests/test-mul-mat-f16-src1.cpp aborted here for q4_1 n=1.
    if (type_y == GGML_TYPE_Q8_1 && ne1 % n_interleaved != 0) n_interleaved = 1;
    GGML_ASSERT(ne1 % n_interleaved == 0);
    // callers (ggml-cpu.c, GGML_CPU_SRC1_SCRATCH_ROWS) reserve work_buffer for at most 16 rows of ne0 floats
    GGML_ASSERT(n_interleaved <= 16);
    int64_t ne1i  = ne1/n_interleaved;
    int64_t nrows = ne1i*ne2*ne3;
    int64_t nrows_per_thread = (nrows + nth - 1)/nth;
    int64_t first_row = nrows_per_thread*ith;
    if (first_row >= nrows) return;
    int64_t last_row = std::min(first_row + nrows_per_thread, nrows);
    for (int64_t row = first_row; row < last_row; ++row) {
        int64_t i3 = row/(ne1i*ne2);
        int64_t i2 = (row - i3*ne1i*ne2)/ne1i;
        int64_t i1 = row - i3*ne1i*ne2 - i2*ne1i;
        auto cx = (const char *)x + i1*n_interleaved*nb1 + i2*nb2 + i3*nb3;
        auto cy = (char *)y + (i3*ne1*ne2 + i2*ne1 + i1*n_interleaved)*row_size_y;
        // TODO: special case common types such as f16, q8_0
        //       (although the performance gains may be too small to justify the added complexity)
        if (type_x != GGML_TYPE_F32) {
            to_float((const void *)cx, (float *)work_buffer, ne0*n_interleaved);
            from_float((const float *)work_buffer, (void *)cy, ne0*n_interleaved);
        } else {
            from_float((const float *)cx, (void *)cy, ne0*n_interleaved);
        }
    }
}




void quantize_row_iq1_bn(const float * x, void * y, int64_t k) {
    quantize_iq1_bn(x, y, 1, k, nullptr, nullptr);
}




void quantize_row_iq2_bn(const float * x, void * y, int64_t k) {
    quantize_iq2_bn(x, y, 1, k, nullptr, nullptr);
}


namespace {
inline int8_t iq1bn_dequant(uint8_t q, int i) {
    uint8_t v = IQ1BNQuantizer::k_mult[i]*q;
    //int8_t vs = (v + (v << 1)) >> 8;
    int8_t vs = 3*v >> 8;
    return vs - 1;
}
}

static const int8_t iq1bn_values[1280] = {
    -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  0, -1, -1, -1,  0,  0, -1, -1, -1,  1,  0,
    -1, -1, -1, -1,  1, -1, -1, -1,  0,  1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1,  0, -1, -1,  0, -1,  0, -1, -1,  1, -1,  0, -1,
    -1, -1,  0,  0, -1, -1,  0,  0,  0, -1, -1,  1,  0,  0, -1, -1, -1,  1,  0, -1, -1,  0,  1,  0, -1, -1,  1,  1,  0, -1, -1, -1,
    -1,  1, -1, -1,  0,  0,  0,  0,  0,  0, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,  0,  1, -1, -1,  0,  0,  1, -1, -1,  1,  0,  1,
    -1, -1, -1,  1,  1, -1, -1,  0,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  0, -1,  0, -1, -1,  0, -1,  1, -1, -1,  0, -1,
    -1,  0, -1,  0, -1,  0,  0, -1,  0, -1,  1,  0, -1,  0, -1, -1,  1, -1,  0, -1,  0,  1, -1,  0, -1,  1,  1, -1,  0, -1, -1, -1,
     0,  0, -1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  1, -1,  0,  0, -1, -1,  0,  0,  0, -1,  0,  0,  0,  0, -1,  1,  0,  0,  0,
    -1, -1,  1,  0,  0, -1,  0,  1,  0,  0, -1,  1,  1,  0,  0, -1, -1, -1,  1,  0, -1,  0, -1,  1,  0, -1,  1, -1,  1,  0, -1, -1,
     0,  1,  0, -1,  0,  0,  1,  0, -1,  1,  0,  1,  0, -1, -1,  1,  1,  0, -1,  0,  1,  1,  0, -1,  1,  1,  1,  0, -1, -1, -1, -1,
     1, -1,  0, -1, -1,  1, -1,  1, -1, -1,  1, -1,  0,  0,  0,  0,  0, -1,  0, -1,  1, -1,  0,  0, -1,  1, -1,  1,  0, -1,  1, -1,
    -1,  1, -1,  1, -1,  0,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1,  0,  1, -1,  0, -1,  0,  1, -1,  1, -1,  0,  1, -1, -1,  0,
     0,  1, -1,  0,  0,  0,  1, -1,  1,  0,  0,  1, -1, -1,  1,  0,  1, -1,  0,  1,  0,  1, -1,  1,  1,  0,  1, -1, -1, -1,  1,  1,
    -1,  0, -1,  1,  1, -1,  1, -1,  1,  1, -1,  0,  0,  0,  0,  0, -1,  0,  1,  1, -1,  0,  0,  1,  1, -1,  1,  0,  1,  1, -1, -1,
     1,  1,  1, -1,  0,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1,  0,  1, -1, -1, -1,  0, -1,  0, -1,
    -1,  0,  0,  0, -1, -1,  0,  1,  0, -1, -1,  0, -1,  1, -1, -1,  0,  0,  1, -1, -1,  0,  1,  1, -1, -1,  0, -1, -1,  0, -1,  0,
     0, -1,  0, -1,  0,  1, -1,  0, -1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0, -1,  0, -1,  1,
     0, -1,  0,  0,  1,  0, -1,  0,  1,  1,  0, -1,  0, -1, -1,  1, -1,  0,  0, -1,  1, -1,  0,  1, -1,  1, -1,  0, -1,  0,  1, -1,
     0,  0,  0,  1, -1,  0,  1,  0,  1, -1,  0, -1,  1,  1, -1,  0,  0,  1,  1, -1,  0,  1,  1,  1, -1,  0, -1, -1, -1,  0,  0,  0,
    -1, -1,  0,  0,  1, -1, -1,  0,  0, -1,  0, -1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0, -1,  0,  0, -1,  1, -1,
     0,  0,  0,  1, -1,  0,  0,  1,  1, -1,  0,  0, -1, -1,  0,  0,  0,  0, -1,  0,  0,  0,  1, -1,  0,  0,  0, -1,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0,  0, -1, -1,  1,  0,  0,  0, -1,
     1,  0,  0,  1, -1,  1,  0,  0, -1,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0, -1,  1,  1,  0,
     0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0, -1, -1, -1,  1,  0,  0, -1, -1,  1,  0,  1, -1, -1,  1,  0, -1,  0, -1,  1,  0,  0,
     0, -1,  1,  0,  1,  0, -1,  1,  0, -1,  1, -1,  1,  0,  0,  1, -1,  1,  0,  1,  1, -1,  1,  0, -1, -1,  0,  1,  0,  0, -1,  0,
     1,  0,  1, -1,  0,  1,  0, -1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0, -1,  1,  0,  1,  0,
     0,  1,  0,  1,  0,  1,  1,  0,  1,  0, -1, -1,  1,  1,  0,  0, -1,  1,  1,  0,  1, -1,  1,  1,  0, -1,  0,  1,  1,  0,  0,  0,
     1,  1,  0,  1,  0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0, -1, -1, -1, -1,  1,  0, -1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  0, -1, -1,  1,  0,  0, -1, -1,  1,  1,  0, -1, -1,  1, -1,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,
     1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  0, -1,  1,  0, -1,  0, -1,  1,  1, -1,  0, -1,  1, -1,  0,  0, -1,  1,  0,  0,  0,
    -1,  1,  1,  0,  0, -1,  1, -1,  1,  0, -1,  1,  0,  1,  0, -1,  1,  1,  1,  0, -1,  1, -1, -1,  1, -1,  1,  0, -1,  1, -1,  1,
     1, -1,  1, -1,  1, -1,  0,  1, -1,  1,  0,  0,  1, -1,  1,  1,  0,  1, -1,  1, -1,  1,  1, -1,  1,  0,  0,  0,  0,  0,  0,  1,
     1, -1,  1,  1,  1,  1, -1,  1, -1, -1, -1,  0,  1,  0, -1, -1,  0,  1,  1, -1, -1,  0,  1, -1,  0, -1,  0,  1,  0,  0, -1,  0,
     1,  1,  0, -1,  0,  1, -1,  1, -1,  0,  1,  0,  1, -1,  0,  1,  1,  1, -1,  0,  1, -1, -1,  0,  0,  1,  0, -1,  0,  0,  1,  1,
    -1,  0,  0,  1, -1,  0,  0,  0,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  1, -1,  1,  0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  0,
     0,  0,  1,  1,  0,  0,  1, -1, -1,  1,  0,  1,  0, -1,  1,  0,  1,  1, -1,  1,  0,  1, -1,  0,  1,  0,  1,  0,  0,  1,  0,  1,
     1,  0,  1,  0,  1, -1,  1,  1,  0,  1,  0,  1,  1,  0,  1,  1,  1,  1,  0,  1, -1, -1, -1,  1,  1,  0, -1, -1,  1,  1,  1, -1,
    -1,  1,  1, -1,  0, -1,  1,  1,  0,  0, -1,  1,  1,  1,  0, -1,  1,  1, -1,  1, -1,  1,  1,  0,  1, -1,  1,  1,  1,  1, -1,  1,
     1,  0,  0,  0,  0,  0, -1, -1,  0,  1,  1,  0, -1,  0,  1,  1,  1, -1,  0,  1,  1, -1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  1,
     0,  0,  1,  1, -1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,  1, -1, -1,  1,  1,  1,  0, -1,  1,  1,  1,  1, -1,  1,
     1,  1, -1,  0,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  1,  1,  1, -1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
};

void ggml_vec_dot_iq1_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq1_bn requires a block size of 64");

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ1_BN, vx, 0, GGML_TYPE_Q8_K64, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const block_iq1_bn * x = (const block_iq1_bn *)vx;

    const float * d8 = (const float *)vy;
    const int8_t * q8 = (const int8_t *)(d8 + 4);
    int nblock = n / QK_IQ1BN;

    int sumi[8] = {};
    int8_t q1[16];

    for (int ii = 0; ii < nblock; ii += 32) {
        int16_t sum16[8] = {};
        int nb = std::min(ii + 32, nblock);
        for (int i = ii; i < nb; ++i) {
            auto ql = x[i].ql;
            const int8_t * extra = iq1bn_values + 5*x[i].extra;
            for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
                for (int k = 0; k < 3; ++k) {
                    uint8_t q = *ql++;
                    const int8_t * vs = iq1bn_values + 5*q;
                    for (int j = 0; j < 5; ++j) q1[5*k+j] = vs[j];
                }
                q1[15] = extra[i16];
                // We collect 8 q8 values per block into each element of sum16
                // => 32 x 8 = 256 values in each loop over i, so this cannot overflow the int16_t range
                //    (q8 is in -127...127, and hence the sum is in -32512...32512
                for (int j = 0; j < 8; ++j) sum16[j] += q8[2*j+0]*q1[2*j+0] + q8[2*j+1]*q1[2*j+1];
                q8 += 16;
            }
        }
        for (int j = 0; j < 8; ++j) sumi[j] += sum16[j];
    }

    *s = d8[0] * (sumi[0] + sumi[1]) + d8[1] * (sumi[2] + sumi[3]) + d8[2] * (sumi[4] + sumi[5]) + d8[3] * (sumi[6] + sumi[7]);
}

void vec_dot_iq2_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq2_bn requires a block size of 64");

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_BN, vx, 0, GGML_TYPE_Q8_K64, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    constexpr int Nj = QK_IQ1BN/4;

    const block_iq2_bn * x = (const block_iq2_bn *)vx;
    int nblock = n / QK_IQ1BN;

    const float * d = (const float *)vy;
    const int8_t * q8 = (const int8_t *)(d + 4);

    int sum[16] = { };
    int sum0[4] = { };

    for (int i = 0; i < nblock; ++i) {
        for (int j = 0; j < Nj/4; ++j) {
            for (int l = 0; l < 4; ++l) {
                sum[4*j + 0] += q8[4*j + l +    0] * (x[i].qs[4*j+l] & 0x03);
                sum[4*j + 1] += q8[4*j + l + 1*Nj] * (x[i].qs[4*j+l] & 0x0c);
                sum[4*j + 2] += q8[4*j + l + 2*Nj] * (x[i].qs[4*j+l] & 0x30);
                sum[4*j + 3] += q8[4*j + l + 3*Nj] * (x[i].qs[4*j+l] & 0xc0);
                sum0[j] += q8[4*j + l] + q8[4*j + l + 1*Nj] + q8[4*j + l + 2*Nj] + q8[4*j + l + 3*Nj];
            }
        }
        q8 += QK_IQ1BN;
    }

    float sumf = 0;
    for (int j = 0; j < 4; ++j) {
        sumf += d[j] * (sum[4*j + 0] + 0.25f*sum[4*j + 1] + 0.0625*sum[4*j + 2] + 0.015625*sum[4*j + 3] - sum0[j]);
    }
    *s = sumf;

}

void quantize_row_q8_K64_ref(const float * x, block_q8_K64 * y, int64_t k) {

    GGML_ASSERT(k >= 8*QK_IQ1BN);

    float * dptr = (float *)y;
    auto qs = (int8_t *)(dptr + 8);
#ifdef __ARM_NEON
    static const uint8_t k_shuffle[16] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
    auto shuffle = vld1q_u8(k_shuffle);
    float32x4_t max[4] = { };
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = vld1q_f32(x + j + 4*i);
            val = vabsq_f32(val);
            max[i] = vmaxq_f32(max[i], val);
        }
    }
    float32x4_t vid[4];
    for (int i = 0; i < 4; ++i) {
        dptr[i] = vmaxvq_f32(max[i])/127;
        float id = dptr[i] > 0 ? 1/dptr[i] : 0.f;
        vid[i] = vdupq_n_f32(id);
    }
    int8x16x4_t q;
    int32x4_t qsum = {};
    const int8x16_t m1 = vdupq_n_s8(1);
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = vld1q_f32(x + j + 4*i);
            val = vmulq_f32(vid[i], val);
            auto ival = vcvtnq_s32_f32(val);
            q.val[i] = vreinterpretq_s8_s32(ival);
        }
        auto qi = vqtbl4q_s8(q, shuffle);
        qsum = ggml_vdotq_s32(qsum, qi, m1);
        vst1q_s8(qs, qi);
        qs += 16;
    }
    auto sumf = vmulq_f32(vld1q_f32(dptr), vcvtq_f32_s32(qsum));
    vst1q_f32(dptr + 4, sumf);
#elif defined __AVX__
    __m128 max[4] = {};
    __m128 sign_bit = _mm_set1_ps(-0.f);
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = _mm_loadu_ps(x + j + 4*i);
            val = _mm_andnot_ps(sign_bit, val);
            max[i] = _mm_max_ps(max[i], val);
        }
    }
    __m128 vid[4];
    for (int i = 0; i < 4; ++i) {
        max[i] = _mm_max_ps(max[i], _mm_movehl_ps(max[i], max[i]));
        max[i] = _mm_max_ss(max[i], _mm_movehdup_ps(max[i]));
        float maxi = _mm_cvtss_f32(max[i]);
        dptr[i] = maxi/127;
        float id = dptr[i] > 0 ? 1/dptr[i] : 0.f;
        vid[i] = _mm_set1_ps(id);
    }
    __m128i q[4];
    __m128i sums = _mm_setzero_si128();
    __m128i m1_8 = _mm_set1_epi8(1);
    __m128i m1_16 = _mm_set1_epi16(1);
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = _mm_loadu_ps(x + j + 4*i);
            val = _mm_round_ps(_mm_mul_ps(vid[i], val), _MM_ROUND_NEAREST);
            q[i] = _mm_cvtps_epi32(val);
        }
        auto q1 = _mm_packs_epi32(q[0], q[1]);
        auto q2 = _mm_packs_epi32(q[2], q[3]);
        auto qi = _mm_packs_epi16(q1, q2);
        auto aux = _mm_maddubs_epi16(m1_8, qi);
        sums = _mm_add_epi32(sums, _mm_madd_epi16(m1_16, aux));
        _mm_storeu_si128((__m128i *)qs, qi);
        qs += 16;
    }
    auto minus = _mm_mul_ps(_mm_loadu_ps(dptr), _mm_cvtepi32_ps(sums));
    _mm_storeu_ps(dptr + 4, minus);
#else
    float aux[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            for (int l = 0; l < 4; ++l) {
                float ax = fabsf(x[j+4*i+l]);
                aux[i] = std::max(aux[i], ax);
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        dptr[i] = aux[i]/127;
        aux[i] = dptr[i] > 0 ? 1/dptr[i] : 0.f;
    }
    int32_t sum[4] = {};
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            for (int l = 0; l < 4; ++l) {
                qs[j+4*i+l] = nearest_int(aux[i]*x[j+4*i+l]);
                sum[i] += qs[j+4*i+l];
            }
        }
    }
    for (int i = 0; i < 4; ++i) dptr[4+i] = dptr[i]*sum[i];
#endif
}

void quantize_row_q8_K64(const float * x, void * y, int64_t k) {
    quantize_row_q8_K64_ref(x, (block_q8_K64 *)y, k);
}

void quantize_row_q8_K16(const float * x, void * vy, int64_t nk) {
    float * dptr = (float *)vy;
    int8_t * qy = (int8_t *)(dptr + 5);
    int n64 = nk / 64;
#ifdef z__AVX2__
    __m256 sign_bit = _mm256_set1_ps(-0.f);
    __m256 vmax[4] = {};
    __m256 vsum[4] = {};
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            auto v1 = _mm256_loadu_ps(x + 64*i64 + 16*k + 0);
            auto v2 = _mm256_loadu_ps(x + 64*i64 + 16*k + 8);
            vsum[k] = _mm256_add_ps(vsum[k], _mm256_add_ps(v1, v2));
            v1 = _mm256_andnot_ps(sign_bit, v1);
            v2 = _mm256_andnot_ps(sign_bit, v2);
            vmax[k] = _mm256_max_ps(vmax[k], _mm256_max_ps(v1, v2));
        }
    }
    __m256 sum = _mm256_add_ps(_mm256_add_ps(vsum[0], vsum[1]), _mm256_add_ps(vsum[2], vsum[3]));
    dptr[4] = hsum_float_8(sum);
    for (int k = 0; k < 4; ++k) {
        float max = hmax_f32_8(vmax[k]);
        dptr[k] = max/127;
        vmax[k] = _mm256_set1_ps(dptr[k] > 0 ? 1/dptr[k] : 0.f);
    }
    __m256i ival[8];
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            __m256 v0 = _mm256_mul_ps(vmax[k], _mm256_loadu_ps(x + 64*i64 + 16*k + 0));
            __m256 v1 = _mm256_mul_ps(vmax[k], _mm256_loadu_ps(x + 64*i64 + 16*k + 8));
            v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
            v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
            ival[2*k+0] = _mm256_cvtps_epi32(v0);
            ival[2*k+1] = _mm256_cvtps_epi32(v1);
        }
        for (int k = 0; k < 2; ++k) {
            auto i0 = _mm256_packs_epi32(ival[4*k+0], ival[4*k+1]);
            auto i1 = _mm256_packs_epi32(ival[4*k+2], ival[4*k+3]);
            i0 = _mm256_packs_epi16(i0, i1);
            i0 = _mm256_permutevar8x32_epi32(i0, perm);
            _mm256_storeu_si256((__m256i *)qy, i0);
            qy += 32;
        }
    }
#elif defined z__ARM_NEON
    static const uint8_t k_shuffle[16] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
    auto shuffle = vld1q_u8(k_shuffle);
    float32x4_t vmax[4] = {};
    float32x4_t vsum[4] = {};
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            auto v = vld1q_f32_x4(x + 64*i64 + 16*k);
            vsum[k] = vaddq_f32(vsum[k], vaddq_f32(v.val[0], v.val[1]));
            vsum[k] = vaddq_f32(vsum[k], vaddq_f32(v.val[2], v.val[3]));
            vmax[k] = vmaxq_f32(vmax[k], vmaxq_f32(vabsq_f32(v.val[0]), vabsq_f32(v.val[1])));
            vmax[k] = vmaxq_f32(vmax[k], vmaxq_f32(vabsq_f32(v.val[2]), vabsq_f32(v.val[3])));
        }
    }
    dptr[4] = vaddvq_f32(vaddq_f32(vaddq_f32(vsum[0], vsum[1]), vaddq_f32(vsum[2], vsum[3])));
    for (int k = 0; k < 4; ++k) {
        float max = vmaxvq_f32(vmax[k]);
        dptr[k] = max/127;
        vmax[k] = vdupq_n_f32(dptr[k] > 0 ? 1/dptr[k] : 0.f);
    }
    int8x16x4_t q;
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            auto v = vld1q_f32_x4(x + 64*i64 + 16*k);
            for (int j = 0; j < 4; ++j) {
                q.val[j] = vreinterpretq_s8_s32(vcvtnq_s32_f32(vmulq_f32(vmax[k], v.val[j])));
            }
            auto qi = vqtbl4q_s8(q, shuffle);
            vst1q_s8(qy, qi);
            qy += 16;
        }
    }
#else
    float amax[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            for (int j = 0; j < 16; ++j) {
                float ax = std::abs(x[64*i64 + 16*k + j]);
                amax[k] = std::max(amax[k], ax);
            }
        }
    }
    for (int k = 0; k < 4; ++k) {
        dptr[k] = amax[k]/127;
        amax[k] = dptr[k] > 0 ? 1/dptr[k] : 0.f;
    }
    int sumi[4] = {};
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            for (int j = 0; j < 16; ++j) {
                int ix = nearest_int(amax[k]*x[64*i64 + 16*k + j]);
                sumi[k] += ix;
                qy[64*i64 + 16*k + j] = ix;
            }
        }
    }
    dptr[4] = dptr[0]*sumi[0] + dptr[1]*sumi[1] + dptr[2]*sumi[2] + dptr[3]*sumi[3];
#endif
}

void iqk_quantize_q4_0(const float * x, void * vy, int64_t k) {
    const int nb = k / QK4_0;
    auto y = (block_q4_0 *)vy;
#ifdef __AVX2__
    static_assert(QK4_0 == 32);
    __m256 vx[4], rx[4];
    __m256i ix[4];
    auto v7 = _mm256_set1_ps(7.0f);
    auto perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            vx[k] = _mm256_loadu_ps(x + 8*k);
        }
        auto max1 = _mm256_max_ps(vx[0], vx[1]);
        auto max2 = _mm256_max_ps(vx[2], vx[3]);
        auto vmax = _mm256_max_ps(max1, max2);
        auto min1 = _mm256_min_ps(vx[0], vx[1]);
        auto min2 = _mm256_min_ps(vx[2], vx[3]);
        auto vmin = _mm256_min_ps(min1, min2);
        float max = hmax_float_8(vmax);
        float min = hmin_float_8(vmin);
        float amax = std::abs(max);
        float amin = std::abs(min);
        float d, id;
        if (amax > amin) {
            d = max / -8;
            id = amax > 1e-13f ? 1/d : 0.0f;
        } else {
            d = min / -8;
            id = amin > 1e-13f ? 1/d : 0.0f;
        }
        auto vid = _mm256_set1_ps(id);
        auto vsumqx = _mm256_setzero_ps();
        auto vsumq2 = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) {
            rx[k] = _mm256_mul_ps(vid, vx[k]);
            rx[k] = _mm256_round_ps(rx[k], _MM_ROUND_NEAREST);
            rx[k] = _mm256_min_ps(rx[k], v7);
            ix[k] = _mm256_cvtps_epi32(rx[k]);
            ix[k] = _mm256_add_epi32(ix[k], _mm256_set1_epi32(8));
            auto w  = _mm256_mul_ps(vx[k], vx[k]);
            auto wr = _mm256_mul_ps(w, rx[k]);
            vsumqx  = _mm256_fmadd_ps(wr, vx[k], vsumqx);
            vsumq2  = _mm256_fmadd_ps(wr, rx[k], vsumq2);
        }
        auto sumq2 = hsum_float_8(vsumq2);
        if (sumq2 > 0) {
            auto sumqx = hsum_float_8(vsumqx);
            d = sumqx/sumq2;
        }
        y[ib].d = GGML_FP32_TO_FP16(d);
        auto i0 = _mm256_packs_epi32(ix[0], ix[1]);
        auto i2 = _mm256_packs_epi32(ix[2], ix[3]);
        i0 = _mm256_packs_epi16(i0, i2);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        auto q = _mm_or_si128(_mm256_castsi256_si128(i0), _mm_slli_epi16(_mm256_extracti128_si256(i0, 1), 4));
        _mm_storeu_si128((__m128i *)y[ib].qs, q);
        x += QK4_0;
    }
#else
    for (int ib = 0; ib < nb; ++ib) {
        float max = 0, amax = 0;
        for (int j = 0; j < QK4_0; ++j) {
            float ax = std::abs(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        float d = max / -8;
        float id = amax > 1e-13f ? 1/d : 0.0f;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < QK4_0/2; ++j) {
            float v0 = x[j], v1 = x[j+QK4_0/2];
            int i0 = nearest_int(id*v0), i1 = nearest_int(id*v1);
            i0 = std::min(i0, 7);
            i1 = std::min(i1, 7);
            float w0 = v0*v0, w1 = v1*v1;
            sumqx += w0*i0*v0 + w1*i1*v1;
            sumq2 += w0*i0*i0 + w1*i1*i1;
            y[ib].qs[j] = (i0 + 8) | ((i1 + 8) << 4);
        }
        if (sumq2 > 0) d = sumqx/sumq2;
        y[ib].d = GGML_FP32_TO_FP16(d);
        x += QK4_0;
    }
#endif
}

void quantize_row_q8_0_x4(const float * x, void * vy, int64_t k) {
    const int nb = k / QK8_0;
    const int nb4 = 4*(nb/4);

    block_q8_0    * y  = (block_q8_0    *)vy;
    block_q8_0_x4 * y4 = (block_q8_0_x4 *)vy;
#if defined(__aarch64__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[32*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[32*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[32*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[32*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        const float d = maxScalar / 127.f;
        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#endif
}

namespace {
template <typename Block, typename Block_x4>
void quantize_row_q8_1_x4_T(const float * x, Block * y, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    const int nb4 = 4*(nb/4);
    Block_x4 * y4 = (Block_x4 *)y;
#if defined(__aarch64__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[QK8_1*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[QK8_1*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[QK8_1*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[QK8_1*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }

            accv = vaddq_s32(accv, vi);
        }

        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
            } else {
                y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
            }
        } else {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_BF16(d * vaddvq_s32(accv)).bits;
            } else {
                y[i].s = GGML_FP32_TO_BF16(d * vaddvq_s32(accv)).bits;
            }
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        float d = max_scalar / 127.f;
        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
            } else {
                y[i].d = GGML_FP32_TO_FP16(d);
            }
        } else {
            auto t = GGML_FP32_TO_BF16(d);
            d = ggml_bf16_to_fp32(t);
            if (i < nb4) {
                y4[i4].d[ir] = t.bits;
            } else {
                y[i].d = t.bits;
            }
        }
        const float id = d > 0 ? 1/d : 0.f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Compute the sum of the quants and set y[i].s
        int isum = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * isum);
            } else {
                y[i].s = GGML_FP32_TO_FP16(d * isum);
            }
        } else {
            if (i < nb4) {
                auto i16 = (int16_t *)y4[i4].d;
                i16[ir+4] = isum;
            } else {
                auto i16 = (int16_t *)&y[i].s;
                i16[0] = isum;
            }
        }

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#endif
}
}

void quantize_row_q8_1_x4(const float * x, void * vy, int64_t k) {
    quantize_row_q8_1_x4_T<block_q8_1, block_q8_1_x4>(x, (block_q8_1 *)vy, k);
}

void quantize_row_q8_2_x4(const float * x, void * vy, int64_t k) {
    quantize_row_q8_1_x4_T<block_q8_2, block_q8_2_x4>(x, (block_q8_2 *)vy, k);
}

//
// ============================================== iq2_K
//





void quantize_row_iq2_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_k * y = (block_iq2_k *)vy;
    quantize_row_iq2_k_ref(x, y, k);
}



void vec_dot_iq2_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    GGML_ABORT("not implemented");


    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

namespace {
#if defined(__AVX2__)
inline void to_values_i32(__m256i idx, __m256i ivalues, __m256i * iv) {
    auto ival = _mm256_shuffle_epi8(ivalues, idx);
    auto ival_1 = _mm256_srli_si256(ival, 8);
    iv[0] = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(ival));
    iv[1] = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(ival_1));
    iv[2] = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(ival, 1));
    iv[3] = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(ival_1, 1));
}
inline __m256i to_int8(const __m256i * ibest) {
    auto i0 = _mm256_packs_epi32(ibest[0], ibest[1]); // 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15
    auto i1 = _mm256_packs_epi32(ibest[2], ibest[3]); // 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31
    auto idx = _mm256_packs_epi16(i0, i1); // 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
    auto idx_l = _mm256_castsi256_si128(idx);
    auto idx_h = _mm256_extracti128_si256(idx, 1);
    auto idx1  = _mm_unpacklo_epi32(idx_l, idx_h); // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    auto idx2  = _mm_unpackhi_epi32(idx_l, idx_h);
    return MM256_SET_M128I(idx2, idx1);
}
bool compute_1block_iq2ks(float d, const __m256 * vx, const __m256 * vw, const int8_t * values, __m256i & this_idx, float & best_d, float & score) {
    constexpr int kBlockSize = 32;
    uint32_t aux32;
    std::memcpy(&aux32, values, sizeof(aux32));
    auto ivalues = _mm256_set1_epi32(aux32);
    __m256 vbest[8];
    __m256i ibest[8];
    auto val = _mm256_set1_ps(d*values[0]);
    auto ival = _mm256_set1_epi32(0);
    for (int k = 0; k < kBlockSize/8; ++k) {
        auto diff = _mm256_sub_ps(vx[k], val);
        vbest[k] = _mm256_mul_ps(diff, diff);
        ibest[k] = ival;
        diff = _mm256_add_ps(vx[k], val);
        vbest[k+4] = _mm256_mul_ps(diff, diff);
        ibest[k+4] = ival;
    }
    for (int j = 1; j < 4; ++j) {
        val = _mm256_set1_ps(d*values[j]);
        ival = _mm256_set1_epi32(j);
        for (int k = 0; k < kBlockSize/8; ++k) {
            auto diff = _mm256_sub_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            auto mask = _mm256_cmp_ps(diff, vbest[k], _CMP_LT_OQ);
            vbest[k] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k]));
            auto imask = _mm256_castps_si256(mask);
            ibest[k] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k]));
            diff = _mm256_add_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            mask = _mm256_cmp_ps(diff, vbest[k+4], _CMP_LT_OQ);
            vbest[k+4] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k+4]));
            imask = _mm256_castps_si256(mask);
            ibest[k+4] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k+4]));
        }
    }
    bool result = false;
    auto idx1 = to_int8(ibest+0);
    auto idx2 = to_int8(ibest+4);
    to_values_i32(idx1, ivalues, ibest+0);
    to_values_i32(idx2, ivalues, ibest+4);
    auto vsqx_1 = _mm256_setzero_ps();
    auto vsq2_1 = _mm256_setzero_ps();
    auto vsqx_2 = _mm256_setzero_ps();
    auto vsq2_2 = _mm256_setzero_ps();
    for (int k = 0; k < 4; ++k) {
        auto vq1  = _mm256_cvtepi32_ps(ibest[k+0]);
        auto vwq1 = _mm256_mul_ps(vw[k], vq1);
        auto vq2  = _mm256_cvtepi32_ps(ibest[k+4]);
        auto vwq2 = _mm256_mul_ps(vw[k], vq2);
        vsqx_1 = _mm256_fmadd_ps(vwq1, vx[k], vsqx_1);
        vsq2_1 = _mm256_fmadd_ps(vwq1, vq1,   vsq2_1);
        vsqx_2 = _mm256_fmadd_ps(vwq2, vx[k], vsqx_2);
        vsq2_2 = _mm256_fmadd_ps(vwq2, vq2,   vsq2_2);
    }
    auto sumqx_1 = hsum_float_8(vsqx_1);
    auto sumq2_1 = hsum_float_8(vsq2_1);
    auto sumqx_2 = hsum_float_8(vsqx_2);
    auto sumq2_2 = hsum_float_8(vsq2_2);
    if (sumq2_1 > 0) {
        best_d = sumqx_1/sumq2_1;
        score = sumqx_1 * best_d;
        this_idx = idx1;
        result = true;
    }
    if (sumq2_2 > 0 && (!result || sumqx_2*sumqx_2 > score*sumq2_2)) {
        best_d = sumqx_2/sumq2_2;
        score = sumqx_2 * best_d;
        this_idx = idx2;
        result = true;
    }
    return result;
}
float compute_1block_iq2ks_rmse(float d, const __m256 * vx, const __m256 * vw, const int8_t * values, __m256i & this_idx) {
    constexpr int kBlockSize = 32;
    uint32_t aux32;
    std::memcpy(&aux32, values, sizeof(aux32));
    auto ivalues = _mm256_set1_epi32(aux32);
    __m256 vbest[4];
    __m256i ibest[4];
    auto val = _mm256_set1_ps(d*values[0]);
    auto ival = _mm256_set1_epi32(0);
    for (int k = 0; k < kBlockSize/8; ++k) {
        auto diff = _mm256_sub_ps(vx[k], val);
        vbest[k] = _mm256_mul_ps(diff, diff);
        ibest[k] = ival;
    }
    for (int j = 1; j < 4; ++j) {
        val = _mm256_set1_ps(d*values[j]);
        ival = _mm256_set1_epi32(j);
        for (int k = 0; k < kBlockSize/8; ++k) {
            auto diff = _mm256_sub_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            auto mask = _mm256_cmp_ps(diff, vbest[k], _CMP_LT_OQ);
            vbest[k] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k]));
            auto imask = _mm256_castps_si256(mask);
            ibest[k] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k]));
        }
    }
    auto idx = to_int8(ibest);
    to_values_i32(idx, ivalues, ibest);
    auto vd = _mm256_set1_ps(-d);
    auto vrmse = _mm256_setzero_ps();
    for (int k = 0; k < 4; ++k) {
        auto vq   = _mm256_cvtepi32_ps(ibest[k]);
        auto diff = _mm256_fmadd_ps(vd, vq, vx[k]);
        auto wdiff = _mm256_mul_ps(vw[k], diff);
        vrmse = _mm256_fmadd_ps(wdiff, diff, vrmse);
    }
    this_idx = idx;
    return hsum_float_8(vrmse);
}
void quantize_row_iq2_ks_fast_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;

    ggml_half * dptr = (ggml_half *)vy;
    *dptr = GGML_FP32_TO_FP16(0.f);

    block_iq2_ks * y = (block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row/QK_K;

    __m256 vx[4], vw[4];

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_ks));

        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto sw = all_sw + ibl*(QK_K/kBlockSize);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0, sumw = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
                sumw += weight[j];
            }
            sw[ib] = sumw;
            if (amax < 1e-14f) {
                scales[ib] = 0;
                continue;
            }
            for (int k = 0; k < 4; ++k) {
                vx[k] = _mm256_loadu_ps(xb + 8*k);
                vw[k] = _mm256_loadu_ps(weight + 8*k);
            }
            float d = max/iq2nl_values[7];
            float best = 0;
            __m256i this_idx;
            float this_d, this_score;
            if (compute_1block_iq2ks(d, vx, vw, iq2nl_values, this_idx, this_d, this_score)) {
                best = this_score; d = this_d;
            }
            for (int itry = -13; itry <= 13; ++itry) {
                if (compute_1block_iq2ks(max/(iq2nl_values[0] + 0.5f*itry), vx, vw, iq2nl_values, this_idx, this_d, this_score)) {
                    if (this_score > best) {
                        best = this_score; d = this_d;
                    }
                }
            }
            bool is_shifted = false;
            for (int itry = -13; itry <= 13; ++itry) {
                if (compute_1block_iq2ks(max/(iq2nl_values[4] + 0.5f*itry), vx, vw, iq2nl_values + 4, this_idx, this_d, this_score)) {
                    if (this_score > best) {
                        best = this_score; d = this_d; is_shifted = true;
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
        }
        y[ibl].extra = extra;
    }

    float d = make_qx_quants(nblock*(QK_K/kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    auto vsumqx = _mm256_setzero_ps();
    auto vsumq2 = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;
        auto Ls = all_Ls + ibl*(QK_K/kBlockSize);
        __m256i idx[4];
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
            uint32_t aux32;
            std::memcpy(&aux32, block_values, sizeof(aux32));
            auto ivalues = _mm256_set1_epi32(aux32);
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            for (int k = 0; k < 4; ++k) {
                vx[k] = _mm256_loadu_ps(xb + 8*k);
                vw[k] = _mm256_loadu_ps(weight + 8*k);
            }
            int ls = Ls[ib] - 16;
            float dl = d*ls;
            __m256i idx1, idx2;
            auto rmse1 = compute_1block_iq2ks_rmse(dl, vx, vw, block_values, idx1);
            if (Ls[ib] > 0 && dl > scales[ib]) {
                auto rmse2 = compute_1block_iq2ks_rmse(d*(Ls[ib] - 17), vx, vw, block_values, idx2);
                if (rmse2 < rmse1) {
                    --Ls[ib]; idx1 = idx2;
                }
            }
            else if (Ls[ib] < 15 && dl < scales[ib]) {
                auto rmse2 = compute_1block_iq2ks_rmse(d*(Ls[ib] - 15), vx, vw, block_values, idx2);
                if (rmse2 < rmse1) {
                    ++Ls[ib]; idx1 = idx2;
                }
            }
            __m256i iv[4];
            to_values_i32(idx1, ivalues, iv);
            auto vd = _mm256_set1_ps(Ls[ib] - 16);
            for (int k = 0; k < 4; ++k) {
                auto vq = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iv[k]));
                auto wvq = _mm256_mul_ps(vw[k], vq);
                vsumqx = _mm256_fmadd_ps(wvq, vx[k], vsumqx);
                vsumq2 = _mm256_fmadd_ps(wvq, vq,    vsumq2);
            }
            ls = Ls[ib];
            y[ibl].scales[ib/2] |= ((ls & 0xf) << 4*(ib%2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            idx[ib % 4] = idx1;
            if ((ib % 4) == 3) {
                auto vqs1 = _mm256_or_si256(idx[0], _mm256_slli_epi16(idx[1], 2));
                auto vqs2 = _mm256_or_si256(_mm256_slli_epi16(idx[2], 4), _mm256_slli_epi16(idx[3], 6));
                auto vqs = _mm256_or_si256(vqs1, vqs2);
                _mm256_storeu_si256((__m256i *)y[ibl].qs + ib/4, vqs);
            }
        }
    }
    float sumqx = hsum_float_8(vsumqx);
    float sumq2 = hsum_float_8(vsumq2);
    *dptr = GGML_FP32_TO_FP16(1.000f*(sumq2 > 0 ? sumqx/sumq2 : d));
}
#endif
void quantize_row_iq2_ks_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;
    constexpr int kMax_i1 = 3*kBlockSize/4;
    constexpr int kMin_i3 = kBlockSize/4;

    ggml_half * dptr = (ggml_half *)vy;
    *dptr = GGML_FP32_TO_FP16(0.f);

    block_iq2_ks * y = (block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    float val [4] = {float(iq2nl_values[0]), float(iq2nl_values[1]), float(iq2nl_values[2]), float(iq2nl_values[3])};
    float sval[4] = {float(iq2nl_values[4]), float(iq2nl_values[5]), float(iq2nl_values[6]), float(iq2nl_values[7])};

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row/QK_K;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_ks));

        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto sw = all_sw + ibl*(QK_K/kBlockSize);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            sw[ib] = 0;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j+1] = sumw[j] + weight[jj];
                sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kMax_i1; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = std::max(i2, kMin_i3); i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*val[0] + (sumx[i2] - sumx[i1])*val[1]
                              + (sumx[i3] - sumx[i2])*val[2] + (sumx[kBlockSize] - sumx[i3])*val[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[0]*val[0] + (sumw[i2] - sumw[i1])*val[1]*val[1]
                              + (sumw[i3] - sumw[i2])*val[2]*val[2] + (sumw[kBlockSize] - sumw[i3])*val[3]*val[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[0] + (sumx[i2] - sumx[i1])*sval[1]
                              + (sumx[i3] - sumx[i2])*sval[2] + (sumx[kBlockSize] - sumx[i3])*sval[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[0]*sval[0] + (sumw[i2] - sumw[i1])*sval[1]*sval[1]
                              + (sumw[i3] - sumw[i2])*sval[2]*sval[2] + (sumw[kBlockSize] - sumw[i3])*sval[3]*sval[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*val[3] + (sumx[i2        ] - sumx[i1])*val[2]
                              + (sumx[i3] - sumx[i2])*val[1] + (sumx[kBlockSize] - sumx[i3])*val[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[3]*val[3] + (sumw[i2        ] - sumw[i1])*val[2]*val[2]
                              + (sumw[i3] - sumw[i2])*val[1]*val[1] + (sumw[kBlockSize] - sumw[i3])*val[0]*val[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[3] + (sumx[i2        ] - sumx[i1])*sval[2]
                              + (sumx[i3] - sumx[i2])*sval[1] + (sumx[kBlockSize] - sumx[i3])*sval[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[3]*sval[3] + (sumw[i2        ] - sumw[i1])*sval[2]*sval[2]
                              + (sumw[i3] - sumw[i2])*sval[1]*sval[1] + (sumw[kBlockSize] - sumw[i3])*sval[0]*sval[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
        }
        y[ibl].extra = extra;
    }

    float d = make_qx_quants(nblock*(QK_K/kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;
        auto Ls = all_Ls + ibl*(QK_K/kBlockSize);
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = Ls[ib];
            y[ibl].scales[ib/2] |= ((ls & 0xf) << 4*(ib%2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            ls -= 16;
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                uint8_t * qs = y[ibl].qs + 32*(ib/4);
                for (int j = 0; j < 32; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
    }
    *dptr = GGML_FP32_TO_FP16(1.030f*(sumq2 > 0 ? sumqx/sumq2 : d));
}
}

void quantize_row_iq2_ks_ref(const float * x, block_iq2_ks * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_ks(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_iq2_ks(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_ks * y = (block_iq2_ks *)vy;
    quantize_row_iq2_ks_ref(x, y, k);
}

size_t quantize_iq2_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    std::vector<float> all_scales(nblock*(QK_K/kBlockSize)), all_sw(nblock*(QK_K/kBlockSize));
    std::vector<int8_t> all_Ls(nblock*(QK_K/kBlockSize));
    auto q_func = [&all_scales, &all_sw, &all_Ls] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
#ifdef __AVX2__
        if (user_data && !user_data->slow_iq2_ks) {
            quantize_row_iq2_ks_fast_impl(x, vy, n_per_row, imatrix, all_scales.data(), all_sw.data(), all_Ls.data());
            return;
        }
#endif
        quantize_row_iq2_ks_impl(x, vy, n_per_row, imatrix, all_scales.data(), all_sw.data(), all_Ls.data());
    };
    QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}


void vec_dot_iq2_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const ggml_half * dptr = (const ggml_half *)vx;
    const float d = GGML_FP16_TO_FP32(*dptr);
    const block_iq2_ks * x = (const block_iq2_ks *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K *)vy;

    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        int sumi = 0;
        for (int ib128 = 0; ib128 < QK_K/128; ++ib128) {
            int d1 = (((x[i].scales[2*ib128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            int d2 = (((x[i].scales[2*ib128+0] >>  4) | ((extra >> 5) & 0x10)) - 16);
            int d3 = (((x[i].scales[2*ib128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16);
            int d4 = (((x[i].scales[2*ib128+1] >>  4) | ((extra >> 7) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values3 = extra & 4 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values4 = extra & 8 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 4;
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
            for (int j = 0; j < 32; ++j) {
                sumi1 += q8[j+ 0] * values1[(qs[j] >> 0) & 3];
                sumi2 += q8[j+32] * values2[(qs[j] >> 2) & 3];
                sumi3 += q8[j+64] * values3[(qs[j] >> 4) & 3];
                sumi4 += q8[j+96] * values4[(qs[j] >> 6) & 3];
            }
            sumi += d1*sumi1 + d2*sumi2 + d3*sumi3 + d4*sumi4;
            q8 += 128;
            qs +=  32;
        }
        sumf += y[i].d * sumi;
    }

    *s = d * sumf;

}

//
// ======================================== iq2_kl
//




void quantize_row_iq2_kl(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_kl * y = (block_iq2_kl *)vy;
    quantize_row_iq2_kl_ref(x, y, k);
}



void vec_dot_iq2_kl_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_KL, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

#ifdef __AVX2__
namespace {
template <int block_size, int nmap>
inline void best_index_avx2(float id, const int8_t * values, const float * f_values, const int8_t * k_simple_index,
        const __m256 * vx, __m256 * vq) {
    static_assert(block_size == 32 || block_size == 16);
    auto vid = _mm256_set1_ps(id);
    auto flip = _mm256_set1_ps(-0.0f);
    auto val0 = _mm256_set1_epi32(values[0]);
    for (int i = 0; i < block_size/8; ++i) {
        auto vxi_p = _mm256_mul_ps(vid, vx[i]);
        auto vxi_m = _mm256_xor_ps(vxi_p, flip);
        auto vidx_p = _mm256_sub_epi32(_mm256_cvttps_epi32(vxi_p), val0);
        auto vidx_m = _mm256_sub_epi32(_mm256_cvttps_epi32(vxi_m), val0);
        vidx_p = _mm256_min_epi32(_mm256_max_epi32(vidx_p, _mm256_setzero_si256()), _mm256_set1_epi32(nmap));
        vidx_m = _mm256_min_epi32(_mm256_max_epi32(vidx_m, _mm256_setzero_si256()), _mm256_set1_epi32(nmap));
        auto tab_p0 = _mm256_i32gather_epi32((const int32_t *)k_simple_index, vidx_p, 1);
        auto tab_m0 = _mm256_i32gather_epi32((const int32_t *)k_simple_index, vidx_m, 1);
        tab_p0 = _mm256_and_si256(tab_p0, _mm256_set1_epi32(0xff));
        tab_m0 = _mm256_and_si256(tab_m0, _mm256_set1_epi32(0xff));
        auto tab_p1 = _mm256_add_epi32(tab_p0, _mm256_set1_epi32(1));
        auto tab_m1 = _mm256_add_epi32(tab_m0, _mm256_set1_epi32(1));
        auto cand0_p = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_p0,  4));
        auto cand1_p = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_p1,  4));
        auto cand0_m = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_m0,  4));
        auto cand1_m = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_m1,  4));
        auto mask_p  = _mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(vxi_p, cand0_p), _mm256_sub_ps(cand1_p, vxi_p), _CMP_LT_OQ));
        auto mask_m  = _mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(vxi_m, cand0_m), _mm256_sub_ps(cand1_m, vxi_m), _CMP_LT_OQ));
        //vidx[2*i+0]  = _mm256_blendv_epi8(tab_p0, tab_p1, mask_p);
        //vidx[2*i+1]  = _mm256_blendv_epi8(tab_m0, tab_m1, mask_m);
        vq[2*i+0]    = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(cand1_p), _mm256_castps_si256(cand0_p), mask_p));
        vq[2*i+1]    = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(cand1_m), _mm256_castps_si256(cand0_m), mask_m));
    }
}
template <int block_size, int nmap>
inline bool evaluate_block_avx2(float id, const int8_t * values, const float * f_values, const int8_t * k_simple_index,
        const __m256 * vx, const __m256 * vw, float & d, float & best) {
    __m256  vq[block_size/4];
    best_index_avx2<block_size, nmap>(id, values, f_values, k_simple_index, vx, vq);
    __m256 sumqx_p = _mm256_setzero_ps(), sumq2_p = _mm256_setzero_ps();
    __m256 sumqx_m = _mm256_setzero_ps(), sumq2_m = _mm256_setzero_ps();
    for (int i = 0; i < block_size/8; ++i) {
        auto wp = _mm256_mul_ps(vw[i], vq[2*i+0]);
        auto wm = _mm256_mul_ps(vw[i], vq[2*i+1]);
        sumqx_p = _mm256_fmadd_ps(wp, vx[i],     sumqx_p);
        sumq2_p = _mm256_fmadd_ps(wp, vq[2*i+0], sumq2_p);
        sumqx_m = _mm256_fmadd_ps(wm, vx[i],     sumqx_m);
        sumq2_m = _mm256_fmadd_ps(wm, vq[2*i+1], sumq2_m);
    }
    auto sqx_p = hsum_float_8(sumqx_p);
    auto sq2_p = hsum_float_8(sumq2_p);
    auto sqx_m = hsum_float_8(sumqx_m);
    auto sq2_m = hsum_float_8(sumq2_m);
    bool updated = false;
    if (sq2_p > 0 && sqx_p*sqx_p > sq2_p*best) {
        d = sqx_p/sq2_p; best = d * sqx_p; updated = true;
    }
    if (sq2_m > 0 && sqx_m*sqx_m > sq2_m*best) {
        d = sqx_m/sq2_m; best = d * sqx_m; updated = true;
    }
    return updated;
}
template <int block_size, int nmap>
std::tuple<float, float, bool> find_best_scale_avx2(int nval, const int8_t * values, const float * f_values, const int8_t * k_simple_index,
        const float * xb, const float * imatrix, float sigma2, int ntry, bool use_sigma2 = false, float step = 1.0f, bool final_step = false) {
    __m256 vx[block_size/8], vw[block_size/8];
    for (int j = 0; j < block_size/8; ++j) vx[j] = _mm256_loadu_ps(xb + 8*j);
    if (imatrix) {
        auto v_sigma2 = _mm256_set1_ps(sigma2);
        for (int j = 0; j < block_size/8; ++j) {
            vw[j] = _mm256_mul_ps(_mm256_loadu_ps(imatrix+8*j), _mm256_sqrt_ps(_mm256_fmadd_ps(vx[j], vx[j], v_sigma2)));
        }
    } else {
        if (use_sigma2) {
            auto v_sigma2 = _mm256_set1_ps(0.25f*sigma2);
            for (int j = 0; j < block_size/8; ++j) {
                vw[j] = _mm256_fmadd_ps(vx[j], vx[j], v_sigma2);
            }
        } else {
            for (int j = 0; j < block_size/8; ++j) {
                vw[j] = _mm256_mul_ps(vx[j], vx[j]);
            }
        }
    }
    float amax = 0, max = 0;
    for (int j = 0; j < block_size; ++j) {
        float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax; max = xb[j];
        }
    }
    if (amax < 1e-16f) {
        return {0.0f, 0.0f, false};
    }
    float d = ntry > 0 ? -max/values[0] : max/values[0];
    float id = 1/d;
    bool is_shifted = false;
    float best = 0;
    (void)evaluate_block_avx2<block_size, nmap>(id, values, f_values, k_simple_index, vx, vw, d, best);
    for (int itry = -ntry; itry <= ntry; ++itry) {
        id = (step*itry + values[0])/max;
        if (evaluate_block_avx2<block_size, nmap>(id, values, f_values, k_simple_index, vx, vw, d, best)) {
            is_shifted = false;
        }
        id = (step*itry + values[nval])/max;
        if (evaluate_block_avx2<block_size, nmap>(id, values+nval, f_values+nval, k_simple_index, vx, vw, d, best)) {
            is_shifted = true;
        }
    }
    if (final_step) {
        int shifted = is_shifted ? nval : 0;
        evaluate_block_avx2<block_size, nmap>(id, values + shifted, f_values + shifted, k_simple_index, vx, vw, d, best);
    }
    return {d, best, is_shifted};
}
// Hand the AVX2 scale search to the IQ3_K / IQ3_KS / IQ5_K / IQ5_KS / IQ6_K quantizers in
// iqk_quantize_base.cpp (ggml-base, built without arch flags, so its `#ifdef __AVX2__` calls were dead).
// One wrapper per <block_size, nmap> those quantizers use; ik compiles them in the same TU (ik 422c259a2
// ggml/src/iqk/iqk_quantize.cpp, quantize_row_iq3_k_impl / _iq3_ks / _iq5_k / _iq5_ks / _iq6_k), MIT.
template <int block_size, int nmap>
iqk_scale_search_result iqk_scale_search_avx2(int nval, const int8_t * values, const float * f_values,
        const int8_t * k_simple_index, const float * xb, const float * imatrix, float sigma2, int ntry,
        bool use_sigma2, float step, bool final_step) {
    auto [d, best, is_shifted] = find_best_scale_avx2<block_size, nmap>(nval, values, f_values, k_simple_index, xb,
            imatrix, sigma2, ntry, use_sigma2, step, final_step);
    return {d, best, is_shifted};
}
}
#endif

extern "C" int iqk_register_scale_search_cpu(void) {
#ifdef __AVX2__
    iqk_set_scale_search(IQK_SCALE_SEARCH_16_127, iqk_scale_search_avx2<16, 127>);
    iqk_set_scale_search(IQK_SCALE_SEARCH_32_127, iqk_scale_search_avx2<32, 127>);
    iqk_set_scale_search(IQK_SCALE_SEARCH_16_240, iqk_scale_search_avx2<16, 240>);
    iqk_set_scale_search(IQK_SCALE_SEARCH_32_240, iqk_scale_search_avx2<32, 240>);
    iqk_set_scale_search(IQK_SCALE_SEARCH_16_248, iqk_scale_search_avx2<16, 248>);
    return IQK_SCALE_SEARCH_COUNT;
#else
    return 0;
#endif
}


//
// ============================================== iq3_k
//




void quantize_row_iq3_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq3_k * y = (block_iq3_k *)vy;
    quantize_row_iq3_k_ref(x, y, k);
}



void vec_dot_iq3_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    GGML_ABORT("not implemented");

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ============================================== iq3_ks
//


void quantize_row_iq3_ks(const float * x, void * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr, nullptr);
}



void  vec_dot_iq3_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_ABORT("Not implemented");

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


void vec_dot_iq4_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const int nb = n / QK_K;

    const block_iq4_k * x = (const block_iq4_k *)vx;
    const block_q8_K  * y = (const block_q8_K *)vy;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t extra = x[ibl].extra;
        uint32_t h = *((const uint32_t *)x[ibl].scales_h);
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        int32_t sum = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            const int8_t * values1 = iq4k_values + 16*(extra & 1);
            const int8_t * values2 = iq4k_values +  8*(extra & 2);
            extra >>= 2;
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * values1[qs[j] & 0xf];
                sumi2 += q8[j+16] * values2[qs[j] >>  4];
            }
            sum += ls1*sumi1 + ls2*sumi2;
            qs += 16;
            q8 += 32;
        }
        sumf += d4d8 * sum;
    }
    *s = sumf;

}


namespace {
// iq4nl_index + best_index_iq4nl: ik main re-adds them here; the fork has them in iqk_quantize_shared.h:228-243 (identical).
// k_simple_index_iq4nl moved to iqk_quantize_shared.h too: iqk_ks_simple_index() (iqk_quantize_base.cpp) needs it.
#ifdef __AVX2__
template <int block_size>
inline void best_index_iq4k(float id, const int8_t * values, const float * f_values,
        const __m256 * vx, /*__m256i * vidx,*/ __m256 * vq) {
    static_assert(block_size == 32 || block_size == 16);

    auto vid = _mm256_set1_ps(id);
    auto flip = _mm256_set1_ps(-0.0f);
    auto val0 = _mm256_set1_epi32(values[0]);
    for (int i = 0; i < block_size/8; ++i) {
        auto vxi_p = _mm256_mul_ps(vid, vx[i]);
        auto vxi_m = _mm256_xor_ps(vxi_p, flip);
        auto vidx_p = _mm256_sub_epi32(_mm256_cvttps_epi32(vxi_p), val0);
        auto vidx_m = _mm256_sub_epi32(_mm256_cvttps_epi32(vxi_m), val0);
        vidx_p = _mm256_min_epi32(_mm256_max_epi32(vidx_p, _mm256_setzero_si256()), _mm256_set1_epi32(240));
        vidx_m = _mm256_min_epi32(_mm256_max_epi32(vidx_m, _mm256_setzero_si256()), _mm256_set1_epi32(240));
        auto tab_p0 = _mm256_i32gather_epi32((const int32_t *)k_simple_index_iq4nl, vidx_p, 1);
        auto tab_m0 = _mm256_i32gather_epi32((const int32_t *)k_simple_index_iq4nl, vidx_m, 1);
        tab_p0 = _mm256_and_si256(tab_p0, _mm256_set1_epi32(0xff));
        tab_m0 = _mm256_and_si256(tab_m0, _mm256_set1_epi32(0xff));
        auto tab_p1 = _mm256_add_epi32(tab_p0, _mm256_set1_epi32(1));
        auto tab_m1 = _mm256_add_epi32(tab_m0, _mm256_set1_epi32(1));
        auto cand0_p = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_p0,  4));
        auto cand1_p = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_p1,  4));
        auto cand0_m = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_m0,  4));
        auto cand1_m = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_m1,  4));
        auto mask_p  = _mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(vxi_p, cand0_p), _mm256_sub_ps(cand1_p, vxi_p), _CMP_LT_OQ));
        auto mask_m  = _mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(vxi_m, cand0_m), _mm256_sub_ps(cand1_m, vxi_m), _CMP_LT_OQ));
        //vidx[2*i+0]  = _mm256_blendv_epi8(tab_p0, tab_p1, mask_p);
        //vidx[2*i+1]  = _mm256_blendv_epi8(tab_m0, tab_m1, mask_m);
        vq[2*i+0]    = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(cand1_p), _mm256_castps_si256(cand0_p), mask_p));
        vq[2*i+1]    = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(cand1_m), _mm256_castps_si256(cand0_m), mask_m));
    }
}
template <int block_size>
inline bool evaluate_block_iq4k(float id, const int8_t * values, const float * f_values,
        const __m256 * vx, const __m256 * vw, float & d, float & best) {
    __m256  vq[block_size/4];
    //__m256i vidx[block_size/4];
    //best_index_iq4k<block_size>(id, values, f_values, vx, vidx, vq);
    best_index_iq4k<block_size>(id, values, f_values, vx, vq);
    __m256 sumqx_p = _mm256_setzero_ps(), sumq2_p = _mm256_setzero_ps();
    __m256 sumqx_m = _mm256_setzero_ps(), sumq2_m = _mm256_setzero_ps();
    for (int i = 0; i < block_size/8; ++i) {
        auto wp = _mm256_mul_ps(vw[i], vq[2*i+0]);
        auto wm = _mm256_mul_ps(vw[i], vq[2*i+1]);
        sumqx_p = _mm256_fmadd_ps(wp, vx[i],     sumqx_p);
        sumq2_p = _mm256_fmadd_ps(wp, vq[2*i+0], sumq2_p);
        sumqx_m = _mm256_fmadd_ps(wm, vx[i],     sumqx_m);
        sumq2_m = _mm256_fmadd_ps(wm, vq[2*i+1], sumq2_m);
    }
    auto sqx_p = hsum_float_8(sumqx_p);
    auto sq2_p = hsum_float_8(sumq2_p);
    auto sqx_m = hsum_float_8(sumqx_m);
    auto sq2_m = hsum_float_8(sumq2_m);
    bool updated = false;
    if (sq2_p > 0 && sqx_p*sqx_p > sq2_p*best) {
        d = sqx_p/sq2_p; best = d * sqx_p; updated = true;
    }
    if (sq2_m > 0 && sqx_m*sqx_m > sq2_m*best) {
        d = sqx_m/sq2_m; best = d * sqx_m; updated = true;
    }
    return updated;
}
#endif

static void quantize_row_iq4_k_impl_bs16(const float * x,
        block_iq4_k * y,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 16;

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(y, 0, sizeof(block_iq4_k));
    y->d = GGML_FP32_TO_FP16(0.f);

    uint16_t * scales_h = (uint16_t *)y->scales_h;

    const int8_t * shifted_values = values + 16;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_K);

#ifdef __AVX2__
    float f_values[32];
    for (int j = 0; j < 32; ++j) f_values[j] = values[j];
#endif

    float max_scale = 0, amax_scale = 0;
    uint16_t extra = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < 1e-16f) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        bool is_shifted = false;
#ifdef __AVX2__
        float best = 0;
        __m256 vx[block_size/8], vw[block_size/8];
        for (int j = 0; j < block_size/8; ++j) vx[j] = _mm256_loadu_ps(xb + 8*j);
        for (int j = 0; j < block_size/8; ++j) vw[j] = _mm256_loadu_ps(weight + 8*j);
        (void)evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best);
#else
        float sumqx_p = 0, sumq2_p = 0;
        float sumqx_m = 0, sumq2_m = 0;
        for (int j = 0; j < block_size; ++j) {
            float w = weight[j];
            float al = id*xb[j];
            int l = best_index_iq4nl(values, al);
            float q = values[l];
            sumqx_p += w*q*xb[j];
            sumq2_p += w*q*q;
            l = best_index_iq4nl(values, -al);
            q = values[l];
            sumqx_m += w*q*xb[j];
            sumq2_m += w*q*q;
        }
        d = sumqx_p/sumq2_p;
        float best = d*sumqx_p;
        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            d = sumqx_m/sumq2_m; best = d*sumqx_m;
        }
#endif
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
#ifdef __AVX2__
            if (evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
                is_shifted = false;
            }
#else
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
            }
#endif
            id = (itry + shifted_values[0])/max;
#ifdef __AVX2__
            if (evaluate_block_iq4k<block_size>(id, values+16, f_values+16, vx, vw, d, best)) {
                is_shifted = true;
            }
#else
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(shifted_values, al);
                float q = shifted_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(shifted_values, -al);
                q = shifted_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
            }
#endif
        }
        if (is_shifted) extra |= (1 << ib);
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }
    float d = -max_scale/32;
    y->d = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_K)*d);
    y->extra = extra;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const int8_t * block_values = extra & (1 << ib) ? shifted_values : values;
        int l = nearest_int(id*scales[ib]);
        l = MAX(-32, MIN(31, l));
        float dl = d * l;
        float idl = dl ? 1/dl : 0.f;
        uint8_t * Lb = L + ib*block_size;
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        for (int j = 0; j < block_size; ++j) {
            Lb[j] = best_index_iq4nl(block_values, idl*xb[j]);
            float w = weight[j];
            float q = block_values[Lb[j]]*l;
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        l += 32;
        uint8_t l_l = l & 0xf;
        uint8_t l_h = l >>  4;
        if (ib%2 == 0) y->scales_l[ib/2] = l_l;
        else y->scales_l[ib/2] |= (l_l << 4);
        scales_h[ib/8] |= (l_h << 2*(ib%8));
    }
    if (sumq2 > 0) y->d = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_K)*sumqx/sumq2);

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y->qs[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

}

void quantize_row_iq4_k_ref(const float * x, block_iq4_k * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_k(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_iq4_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq4_k * y = (block_iq4_k *)vy;
    quantize_row_iq4_k_ref(x, y, k);
}

size_t quantize_iq4_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    uint8_t L[QK_K];
    float weight[16];
    float scales[QK_K/16];
    auto q_func = [&L, &weight, &scales] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        block_iq4_k * iq4 = (block_iq4_k *)vy;
        int nblock = n_per_row/QK_K;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = imatrix ? imatrix + QK_K*ibl : nullptr;
            quantize_row_iq4_k_impl_bs16(x + QK_K*ibl, iq4 + ibl,
                    scales, weight, L, iq4k_values, qw, 7);
        }
    };
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_K, n_per_row);
    QHelper helper(imatrix, user_data, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}


void vec_dot_iq5_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const int nb = n / QK_K;

    const block_iq5_k * x = (const block_iq5_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;

    float sumf = 0;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;
        const int8_t  * q8 = y[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        int sumb  = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            int dl1 = (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
            int dl2 = (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
            int dl3 = (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
            int dl4 = (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
            const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
            const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
            const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
                sumi2 += q8[j+16] * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
                sumi3 += q8[j+32] * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
                sumi4 += q8[j+48] * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
            }
            sumb += dl1 * sumi1 + dl2 * sumi2 + dl3 * sumi3 + dl4 * sumi4;
            q8 += 64;
            qs += 32;
            extra >>= 4;
            shift += 2;
        }
        sumf += d * sumb;

    }

    *s = sumf;

}





void quantize_row_iq5_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq5_k * y = (block_iq5_k *)vy;
    quantize_row_iq5_k_ref(x, y, k);
}

//
// ============================================== iq5_K / iq5_KS : AVX2 bulk quantizer (fast path)
//
// Port of ik_llama 739839a5c "AVX2 quantization for IQ5_K and IQ5_KS (#2608)", MIT, Iwan Kawrakow.
// SOURCE-READ: ik_llama@739839a5c ggml/src/iqk/iqk_quantize.cpp:2932-3045 (best_index_avx2, evaluate_block_avx2,
//   find_best_scale_avx2), :3488-3497 (k_simple_index_iq5nl), :3531-3545 (IQ5_K call site), :4964-4975 (IQ5_KS call site).
//
// COEXISTENCE, two named contracts (row S69-DECL-OVERTURN-IQ5K-AVX2-QUANT-2026-10-17-01):
//   REFERENCE  quantize_iq5_k / quantize_iq5_ks in iqk_quantize_base.cpp (ggml-base, compiled without arch flags). They stay
//              the scalar search, they back type_traits[].from_float_ref (also of the _R4 types, via quantize_iq5_k_r4 /
//              quantize_iq5_ks_r4), and test-iqk-golden pins them.
//   FAST       quantize_iq5_k_bulk / quantize_iq5_ks_bulk below, which ggml_quantize_chunk (so llama-quantize) calls, and the
//              _R4 bulk entries quantize_iq5_k_r4_bulk / quantize_iq5_ks_r4_bulk (same row search, then repack). Upstream
//              REPLACED its scalar search with this one; here it is only the bulk path. The search is different by design (a
//              nearest-lower-value table + two-candidate compare + FMA sums instead of iq5nl_index), so the bytes differ from
//              the reference; tests/test-iqk-quant-fast.cpp bounds the weighted RMSE against the reference instead.
//   GGML_IQK_QUANT_REFERENCE=1 (any value except "0") makes the bulk entries call the reference, for byte-reproducible files.
//   Builds without __AVX2__ always take the reference.
// Deviations from upstream:
//   * upstream writes the 256-entry lookup as a literal; here it is generated from iq5nl_values by its rule (largest i <= 30
//     with values[i] - values[0] <= t). Checked equal to upstream's literal for all 256 entries (temp check_k5index.py, 0 diffs).
//   * find_best_scale_avx2 GGML_ABORTs when the first candidate scale does not beat best = 0. Aborting llama-quantize on one
//     degenerate block is worse than the reference's behaviour there, so the search continues from d = 0 instead.
//   * IQ5_K only: without an imatrix the search weighs by 0.25*sigma2 + x^2 (the reference's weight; upstream passes x^2), and
//     the reference's final least-squares refit of each block scale is kept (upstream dropped it). Measured with upstream's
//     choices: RMSE 1.0187x the reference on the no-imatrix case of test-iqk-quant-fast, over that test's 1.01 gate.
namespace {
std::atomic<int64_t> g_iqk_fast_quant_rows{0};

bool iqk_quant_reference_forced() {
    const char * e = getenv("GGML_IQK_QUANT_REFERENCE");
    return e && e[0] && !(e[0] == '0' && e[1] == 0);
}

#ifdef __AVX2__
// t = trunc(id*x) - values[0], clamped to [0, 240] by the caller -> index of the nearest table value at or below, capped at 30
// so that i+1 is always a valid second candidate. The shifted half of iq5nl_values is the unshifted half + 2 everywhere, so
// one table serves both halves. 256 entries: the 32-bit gather at byte offset 240 reads up to 243.
const int8_t * iq5nl_lower_index() {
    struct Table {
        int8_t v[256];
        Table() {
            for (int t = 0; t < 256; ++t) {
                int i = 0;
                while (i < 30 && iq5nl_values[i+1] - iq5nl_values[0] <= t) ++i;
                v[t] = (int8_t)i;
            }
        }
    };
    static const Table table;
    return table.v;
}

template <int block_size>
inline void best_index_avx2(float id, const int8_t * values, const float * f_values, const int8_t * lower_index,
        const __m256 * vx, __m256 * vq) {
    static_assert(block_size == 32 || block_size == 16);
    auto vid = _mm256_set1_ps(id);
    auto flip = _mm256_set1_ps(-0.0f);
    auto val0 = _mm256_set1_epi32(values[0]);
    for (int i = 0; i < block_size/8; ++i) {
        auto vxi_p = _mm256_mul_ps(vid, vx[i]);
        auto vxi_m = _mm256_xor_ps(vxi_p, flip);
        auto vidx_p = _mm256_sub_epi32(_mm256_cvttps_epi32(vxi_p), val0);
        auto vidx_m = _mm256_sub_epi32(_mm256_cvttps_epi32(vxi_m), val0);
        vidx_p = _mm256_min_epi32(_mm256_max_epi32(vidx_p, _mm256_setzero_si256()), _mm256_set1_epi32(240));
        vidx_m = _mm256_min_epi32(_mm256_max_epi32(vidx_m, _mm256_setzero_si256()), _mm256_set1_epi32(240));
        auto tab_p0 = _mm256_i32gather_epi32((const int32_t *)lower_index, vidx_p, 1);
        auto tab_m0 = _mm256_i32gather_epi32((const int32_t *)lower_index, vidx_m, 1);
        tab_p0 = _mm256_and_si256(tab_p0, _mm256_set1_epi32(0xff));
        tab_m0 = _mm256_and_si256(tab_m0, _mm256_set1_epi32(0xff));
        auto tab_p1 = _mm256_add_epi32(tab_p0, _mm256_set1_epi32(1));
        auto tab_m1 = _mm256_add_epi32(tab_m0, _mm256_set1_epi32(1));
        auto cand0_p = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_p0,  4));
        auto cand1_p = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_p1,  4));
        auto cand0_m = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_m0,  4));
        auto cand1_m = _mm256_castsi256_ps(_mm256_i32gather_epi32((const int32_t *)f_values, tab_m1,  4));
        auto mask_p  = _mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(vxi_p, cand0_p), _mm256_sub_ps(cand1_p, vxi_p), _CMP_LT_OQ));
        auto mask_m  = _mm256_castps_si256(_mm256_cmp_ps(_mm256_sub_ps(vxi_m, cand0_m), _mm256_sub_ps(cand1_m, vxi_m), _CMP_LT_OQ));
        vq[2*i+0]    = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(cand1_p), _mm256_castps_si256(cand0_p), mask_p));
        vq[2*i+1]    = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(cand1_m), _mm256_castps_si256(cand0_m), mask_m));
    }
}
template <int block_size>
inline bool evaluate_block_avx2(float id, const int8_t * values, const float * f_values, const int8_t * lower_index,
        const __m256 * vx, const __m256 * vw, float & d, float & best) {
    __m256  vq[block_size/4];
    best_index_avx2<block_size>(id, values, f_values, lower_index, vx, vq);
    __m256 sumqx_p = _mm256_setzero_ps(), sumq2_p = _mm256_setzero_ps();
    __m256 sumqx_m = _mm256_setzero_ps(), sumq2_m = _mm256_setzero_ps();
    for (int i = 0; i < block_size/8; ++i) {
        auto wp = _mm256_mul_ps(vw[i], vq[2*i+0]);
        auto wm = _mm256_mul_ps(vw[i], vq[2*i+1]);
        sumqx_p = _mm256_fmadd_ps(wp, vx[i],     sumqx_p);
        sumq2_p = _mm256_fmadd_ps(wp, vq[2*i+0], sumq2_p);
        sumqx_m = _mm256_fmadd_ps(wm, vx[i],     sumqx_m);
        sumq2_m = _mm256_fmadd_ps(wm, vq[2*i+1], sumq2_m);
    }
    auto sqx_p = hsum_float_8(sumqx_p);
    auto sq2_p = hsum_float_8(sumq2_p);
    auto sqx_m = hsum_float_8(sumqx_m);
    auto sq2_m = hsum_float_8(sumq2_m);
    bool updated = false;
    if (sq2_p > 0 && sqx_p*sqx_p > sq2_p*best) {
        d = sqx_p/sq2_p; best = d * sqx_p; updated = true;
    }
    if (sq2_m > 0 && sqx_m*sqx_m > sq2_m*best) {
        d = sqx_m/sq2_m; best = d * sqx_m; updated = true;
    }
    return updated;
}
// nval = number of values in one (unshifted) table; values+nval is the shifted table.
template <int block_size>
std::pair<float, bool> find_best_scale_avx2(int nval, const int8_t * values, const float * f_values, const int8_t * lower_index,
        const float * xb, const float * imatrix, float sigma2, int ntry, bool use_sigma2 = false) {
    __m256 vx[block_size/8], vw[block_size/8];
    for (int j = 0; j < block_size/8; ++j) vx[j] = _mm256_loadu_ps(xb + 8*j);
    if (imatrix) {
        auto v_sigma2 = _mm256_set1_ps(sigma2);
        for (int j = 0; j < block_size/8; ++j) {
            vw[j] = _mm256_mul_ps(_mm256_loadu_ps(imatrix+8*j), _mm256_sqrt_ps(_mm256_fmadd_ps(vx[j], vx[j], v_sigma2)));
        }
    } else {
        if (use_sigma2) {
            auto v_sigma2 = _mm256_set1_ps(0.25f*sigma2);
            for (int j = 0; j < block_size/8; ++j) {
                vw[j] = _mm256_fmadd_ps(vx[j], vx[j], v_sigma2);
            }
        } else {
            for (int j = 0; j < block_size/8; ++j) {
                vw[j] = _mm256_mul_ps(vx[j], vx[j]);
            }
        }
    }
    float amax = 0, max = 0;
    for (int j = 0; j < block_size; ++j) {
        float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax; max = xb[j];
        }
    }
    if (amax < 1e-16f) {
        return {0.0f, false};
    }
    float d = ntry > 0 ? -max/values[0] : max/values[0];
    float id = 1/d;
    bool is_shifted = false;
    float best = 0;
    if (!evaluate_block_avx2<block_size>(id, values, f_values, lower_index, vx, vw, d, best)) {
        d = 0; // upstream GGML_ABORTs here; see the deviation note above
    }
    for (int itry = -ntry; itry <= ntry; ++itry) {
        id = (itry + values[0])/max;
        if (evaluate_block_avx2<block_size>(id, values, f_values, lower_index, vx, vw, d, best)) {
            is_shifted = false;
        }
        id = (itry + values[nval])/max;
        if (evaluate_block_avx2<block_size>(id, values+nval, f_values+nval, lower_index, vx, vw, d, best)) {
            is_shifted = true;
        }
    }
    return {d, is_shifted};
}

// IQ5_K row, AVX2 scale search. Everything after the search loop is the reference's encode pass, unchanged.
void quantize_row_iq5_k_fast_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {
    const int ntry = 5;

    block_iq5_k * y = (block_iq5_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    float f_values[64];
    for (int j = 0; j < 64; ++j) f_values[j] = iq5nl_values[j];
    const int8_t * lower_index = iq5nl_lower_index();

    const int8_t * shifted_values = iq5nl_values + 32;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ5_K);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq5_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 2*sumx2/QK_K;

        float max_scale = 0, max_abs_scale = 0;
        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            // use_sigma2 = true: without an imatrix the reference weighs by 0.25*sigma2 + x^2 (upstream's call site has this
            // argument commented out and uses x^2). With x^2, test-iqk-quant-fast measured RMSE 1.0187x the reference.
            auto [d, is_shifted] = find_best_scale_avx2<16>(32, iq5nl_values, f_values, lower_index, xbl + ib*16,
                    quant_weights ? quant_weights + QK_K*ibl + 16*ib : nullptr, sigma2, ntry, true);
            if (d) {
                // the reference's least-squares refit of d with the indices chosen at 1/d (base quantize_row_iq5_k_impl);
                // upstream dropped it. Scalar: 16 values, one pass, after 22 vector evaluations.
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                const int8_t * block_values = is_shifted ? shifted_values : iq5nl_values;
                const float idd = 1/d;
                float sqx = 0, sq2 = 0;
                for (int j = 0; j < 16; ++j) {
                    const float q = block_values[best_index_iq5nl(block_values, idd*xb[j])];
                    sqx += weight[j]*q*xb[j];
                    sq2 += weight[j]*q*q;
                }
                if (sq2 > 0) d = sqx/sq2;
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }
        }

        if (!max_abs_scale) continue;
        float d = -max_scale/32;
        y[ibl].d = GGML_FP32_TO_FP16(fudge*d);
        y[ibl].extra = extra;

        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = MAX(-32, MIN(31, ls));
            int uls = ls + 32;
            y[ibl].scales_l[ib/2] |= ((uls & 0xf) << 4*(ib%2));
            y[ibl].scales_h[ib/4] |= ((uls >>  4) << 2*(ib%4));
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq5nl_values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/2) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq5nl(block_values, al);
                    qs[j] |= ((ibest & 0xf) << 4*(ib32%2));
                    qh[j] |= ((ibest >>  4) << (ib32%8));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(fudge*sumqx/sumq2);

    }
}

// IQ5_KS row, AVX2 scale search. The per-row-scale encode pass after the search is the reference's, unchanged.
void quantize_row_iq5_ks_fast_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float f_values[64];
    for (int j = 0; j < 64; ++j) f_values[j] = values[j];
    const int8_t * lower_index = iq5nl_lower_index();

    float * dptr = (float *)cy;
    dptr[0] = 0;
    block_iq5_ks * y = (block_iq5_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 32;

    float amax_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ5_KS);

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq5_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            auto [d, is_shifted] = find_best_scale_avx2<block_size>(32, values, f_values, lower_index, xbl + ib*block_size,
                    quant_weights ? quant_weights + super_block_size*ibl + block_size*ib : nullptr, sigma2, ntry);
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = fudge*d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            for (int j = 0; j < block_size; ++j) {
                uint8_t idx = best_index_iq5nl(block_values, idl*xb[j]);
                y[ibl].qs[block_size*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float w = weight[j];
                float q = block_values[idx]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = fudge*sumqx/sumq2;
}
#endif // __AVX2__
} // namespace

int64_t iqk_quantize_fast_path_rows(void) {
    return g_iqk_fast_quant_rows.load(std::memory_order_relaxed);
}

size_t quantize_iq5_k_bulk(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
#ifdef __AVX2__
    if (!iqk_quant_reference_forced()) {
        GGML_ASSERT(n_per_row%QK_K == 0);
        QHelper helper(imatrix, user_data, n_per_row, 16);
        auto row_size = ggml_row_size(GGML_TYPE_IQ5_K, n_per_row);
        helper.quantize(nrows, src, dst, row_size, quantize_row_iq5_k_fast_impl);
        g_iqk_fast_quant_rows.fetch_add(nrows, std::memory_order_relaxed);
        return nrows * row_size;
    }
#endif
    return quantize_iq5_k(src, dst, nrows, n_per_row, imatrix, user_data);
}

size_t quantize_iq5_ks_bulk(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
#ifdef __AVX2__
    if (!iqk_quant_reference_forced()) {
        constexpr int kBlockSize = 32;
        GGML_ASSERT(n_per_row%QK_K == 0);
        auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
        float weight[kBlockSize];
        std::vector<float> all_scales(n_per_row/kBlockSize);
        QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
        auto q_func = [&all_scales, &weight] (const float * x, void * vy, int n_per_row, const float * imatrix,
                 [[maybe_unused]] const quantize_user_data * user_data) {
            quantize_row_iq5_ks_fast_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq5nl_values, imatrix, 5);
        };
        helper.quantize(nrows, src, dst, row_size, q_func);
        g_iqk_fast_quant_rows.fetch_add(nrows, std::memory_order_relaxed);
        return nrows * row_size;
    }
#endif
    return quantize_iq5_ks(src, dst, nrows, n_per_row, imatrix, user_data);
}

// IQ5_K_R4 / IQ5_KS_R4 bulk quantizers (row S69-IQ5K-R4-AVX2-QUANT-FASTPATH-2026-10-17-01).
// SOURCE-READ: the reference quantize_iq5_k_r4 / quantize_iq5_ks_r4 (iqk_quantize_base.cpp:2835 / :2881) quantize each group
// of 4 rows with the scalar quantize_iq5_k / quantize_iq5_ks, then interleave them with repack_iq5_k / repack_iq5_ks
// (iqk_quantize_shared.h:786 / :832). These do the same with the fast row search above, so an R4 group is exactly
// repack(4 fast IQ5_K / IQ5_KS rows) -- what upstream gets for free, since there quantize_iq5_k itself is the AVX2 search.
// One QHelper (imatrix weights) and one scratch buffer serve every group. GGML_IQK_QUANT_REFERENCE=1, or a build without
// AVX2, calls the reference R4 quantizer, so those bytes equal from_float_ref and the pre-change output.
size_t quantize_iq5_k_r4_bulk(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
#ifdef __AVX2__
    if (!iqk_quant_reference_forced()) {
        GGML_ASSERT(nrows%4 == 0);
        GGML_ASSERT(n_per_row%QK_K == 0);
        auto row_size = ggml_row_size(GGML_TYPE_IQ5_K, n_per_row);
        std::vector<char> qtmp(4*row_size);
        QHelper helper(imatrix, user_data, n_per_row, 16);
        char * qcur = (char *)dst;
        for (int64_t row = 0; row < nrows; row += 4) {
            helper.quantize(4, src, qtmp.data(), row_size, quantize_row_iq5_k_fast_impl);
            repack_iq5_k(4, n_per_row, (const block_iq5_k *)qtmp.data(), (block_iq5_k_r4 *)qcur, false);
            qcur += 4*row_size;
            src  += 4*n_per_row;
        }
        g_iqk_fast_quant_rows.fetch_add(nrows, std::memory_order_relaxed);
        return nrows * row_size;
    }
#endif
    return quantize_iq5_k_r4(src, dst, nrows, n_per_row, imatrix, user_data);
}

size_t quantize_iq5_ks_r4_bulk(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
#ifdef __AVX2__
    if (!iqk_quant_reference_forced()) {
        constexpr int kBlockSize = 32;
        GGML_ASSERT(nrows%4 == 0);
        GGML_ASSERT(n_per_row%QK_K == 0);
        auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
        std::vector<char> qtmp(4*row_size);
        float weight[kBlockSize];
        std::vector<float> all_scales(n_per_row/kBlockSize);
        QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
        auto q_func = [&all_scales, &weight] (const float * x, void * vy, int n_per_row, const float * imatrix,
                 [[maybe_unused]] const quantize_user_data * user_data) {
            quantize_row_iq5_ks_fast_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq5nl_values, imatrix, 5);
        };
        char * qcur = (char *)dst;
        for (int64_t row = 0; row < nrows; row += 4) {
            helper.quantize(4, src, qtmp.data(), row_size, q_func);
            repack_iq5_ks(4, n_per_row, (const block_iq5_ks *)qtmp.data(), (block_iq5_ks_r4 *)qcur, false);
            qcur += 4*row_size;
            src  += 4*n_per_row;
        }
        g_iqk_fast_quant_rows.fetch_add(nrows, std::memory_order_relaxed);
        return nrows * row_size;
    }
#endif
    return quantize_iq5_ks_r4(src, dst, nrows, n_per_row, imatrix, user_data);
}


//
// ============================================== iq6_K
//


void vec_dot_iq6_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ6_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    GGML_ABORT("not implemented");

    // TODO
    //const int nb = n / QK_K;

    //const block_iq5_k * x = (const block_iq5_k *)vx;
    //const block_q8_K  * y = (const block_q8_K  *)vy;

    //float sumf = 0;

    //for (int i = 0; i < nb; i++) {

    //    const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
    //    const uint8_t * qs = x[i].qs;
    //    const uint8_t * qh = x[i].qh;
    //    const uint8_t * sl = x[i].scales_l;
    //    const uint8_t * sh = x[i].scales_h;
    //    const int8_t  * q8 = y[i].qs;

    //    uint16_t extra = x[i].extra;

    //    int shift = 0;
    //    int sumb  = 0;
    //    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

    //        int dl1 = (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
    //        int dl2 = (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
    //        int dl3 = (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
    //        int dl4 = (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
    //        const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
    //        const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
    //        const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
    //        const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
    //        int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
    //        for (int j = 0; j < 16; ++j) {
    //            sumi1 += q8[j+ 0] * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
    //            sumi2 += q8[j+16] * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
    //            sumi3 += q8[j+32] * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
    //            sumi4 += q8[j+48] * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
    //        }
    //        sumb += dl1 * sumi1 + dl2 * sumi2 + dl3 * sumi3 + dl4 * sumi4;
    //        q8 += 64;
    //        qs += 32;
    //        extra >>= 4;
    //        shift += 2;
    //    }
    //    sumf += d * sumb;

    //}

    //*s = sumf;

}

namespace {

inline int best_index(int n, const float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}



}


void quantize_row_iq6_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq6_k * y = (block_iq6_k *)vy;
    quantize_row_iq6_k_ref(x, y, k);
}


namespace {
template <int q8_type>
void iqk_quantize_row_q8_K_T(const float * x, void * vy, int64_t k) {
    const int nb = k / QK_K;
    const int nb32 = k / 32;
    const int nbtail = nb32 - 8*nb;
    //printf("Quantizing row with k = %ld, nb = %d, nbtail = %d\n", k, nb, nbtail);
    block_q8_K * y = (block_q8_K *)vy;
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    auto quantize_block_32 = [&perm] (__m256 mul, const float * x, int8_t * q8, int16_t & sum1, int16_t & sum2) {
        __m256 v0 = _mm256_mul_ps(mul, _mm256_loadu_ps(x +  0));
        __m256 v1 = _mm256_mul_ps(mul, _mm256_loadu_ps(x +  8));
        __m256 v2 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 16));
        __m256 v3 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 24));
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        sum1 = hsum_i32_8(_mm256_add_epi32(i0, i1));
        sum2 = hsum_i32_8(_mm256_add_epi32(i2, i3));
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256((__m256i *)q8, i0);
    };
    for (int i = 0; i < nb; i++) {
        const float * xb = x + i*QK_K;
        __m256 maxAbs = _mm256_setzero_ps();
        const float * xx = xb;
        for (int ib = 0; ib < QK_K/8; ++ib) {
            const __m256 v = _mm256_loadu_ps(xx); xx += 8;
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
        }
        const float maxScalar = hmax_f32_8(maxAbs);
        const float d = maxScalar / 127.f;
        y[i].d = d;
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);
        int8_t * q8 = y[i].qs;
        int block_sum_i32 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int16_t s1, s2;
            quantize_block_32(mul, xb + 32*ib, q8, s1, s2);
            int s = s1 + s2;
            if constexpr (q8_type == 1) {
                auto bs = (float *)y[i].bsums;
                bs[ib] = d*s;
            } else {
                y[i].bsums[2*ib+0] = s1;
                y[i].bsums[2*ib+1] = s2;
            }
            block_sum_i32 += s;
            q8 += 32;
        }
        y[i].sum = d*block_sum_i32;
        //if constexpr (q8_type == 2) {
        //    auto bs = (float *)y[i].bsums;
        //    float sum = 0;
        //    for (int ib = 0; ib < QK_K/32; ++ib) sum += bs[ib];
        //    bs[0] = sum;
        //}
    }
    if (nbtail > 0) {
        const float * xb = x + nb*QK_K;
        __m256 maxAbs = _mm256_setzero_ps();
        const float * xx = xb;
        for (int ib = 0; ib < 4*nbtail; ++ib) {
            const __m256 v = _mm256_loadu_ps(xx); xx += 8;
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
        }
        const float maxScalar = hmax_f32_8(maxAbs);
        const float d = maxScalar / 127.f;
        y[nb].d = d;
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);
        int8_t * q8 = y[nb].qs;
        auto bsums = (int16_t *)(y[nb].qs + 32*nbtail);
        int block_sum_i32 = 0;
        for (int ib = 0; ib < nbtail; ++ib) {
            int16_t s1, s2;
            quantize_block_32(mul, xb + 32*ib, q8, s1, s2);
            bsums[2*ib+0] = s1;
            bsums[2*ib+1] = s2;
            block_sum_i32 += s1 + s2;
            q8 += 32;
        }
        y[nb].sum = d*block_sum_i32;
    }
#else
    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        float d = 1/iscale;
        if constexpr (q8_type == 1) {
            auto bs = (float *)y[i].bsums;
            int tot = 0;
            for (int j = 0; j < QK_K/32; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 32; ++ii) {
                    sum += y[i].qs[j*32 + ii];
                }
                bs[j] = d*sum;
                tot += sum;
            }
            y[i].sum = d*tot;
        } else {
            int tot = 0;
            for (int j = 0; j < QK_K/16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) {
                    sum += y[i].qs[j*16 + ii];
                }
                y[i].bsums[j] = sum;
                tot += sum;
            }
            y[i].sum = d*tot;
        }
        y[i].d = d;
        x += QK_K;
    }
#endif
}
}

void iqk_quantize_row_q8_K(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_K_T<0>(x, vy, k);
}

void quantize_row_q8_K32(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_K_T<1>(x, vy, k);
}

void quantize_row_q8_KR8(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_K_T<2>(x, vy, k);
}

namespace {
// TODO: merge this with the above template
void iqk_quantize_row_q8_K128(const float * x, void * vy, int64_t k) {
    constexpr int kBlockSize = 128;
    assert(k % kBlockSize == 0);
    const int nb = k / kBlockSize;
    auto y = (block_q8_K128 *)vy;
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int i = 0; i < nb; i++) {
        const float * xb = x + i*kBlockSize;
        __m256 maxAbs = _mm256_setzero_ps();
        const float * xx = xb;
        for (int ib = 0; ib < kBlockSize/8; ++ib) {
            const __m256 v = _mm256_loadu_ps(xx); xx += 8;
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
        }
        const float maxScalar = hmax_f32_8(maxAbs);
        const float d = maxScalar / 127.f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );
        xx = xb;
        int8_t * q8 = y[i].qs;
        int32_t isum = 0;
        for (int ib = 0; ib < kBlockSize/32; ++ib) {
            __m256 v0 = _mm256_mul_ps(mul, _mm256_loadu_ps(xx)); xx += 8;
            __m256 v1 = _mm256_mul_ps(mul, _mm256_loadu_ps(xx)); xx += 8;
            __m256 v2 = _mm256_mul_ps(mul, _mm256_loadu_ps(xx)); xx += 8;
            __m256 v3 = _mm256_mul_ps(mul, _mm256_loadu_ps(xx)); xx += 8;
            v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
            v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
            v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
            v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
            __m256i i0 = _mm256_cvtps_epi32(v0);
            __m256i i1 = _mm256_cvtps_epi32(v1);
            __m256i i2 = _mm256_cvtps_epi32(v2);
            __m256i i3 = _mm256_cvtps_epi32(v3);
            int s = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
            isum += s;
            y[i].bsums[ib] = s;
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            i0 = _mm256_packs_epi16( i0, i2 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );
            _mm256_storeu_si256((__m256i *)q8, i0);
            q8 += 32;
        }
        y[i].s = isum;
    }
#elif defined __ARM_NEON
    int32x4_t ival[8];
    for (int i = 0; i < nb; i++) {
        const float * xb = x + i*kBlockSize;
        auto vmax = vdupq_n_f32(0.f);
        for (int j = 0; j < kBlockSize; j += 4) {
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xb + j)));
        }
        auto smax = vmaxvq_f32(vmax);
        if (!smax) {
            std::memset(&y[i], 0, sizeof(y[i]));
            continue;
        }
        y[i].d = smax/127;
        auto vid = vdupq_n_f32(127/smax);
        int32_t isum_tot = 0;
        for (int ib = 0; ib < kBlockSize/32; ++ib) {
            auto isum = vdupq_n_s32(0);
            for (int k = 0; k < 8; ++k) {
                auto val = vld1q_f32(xb + 32*ib + 4*k);
                ival[k] = vcvtnq_s32_f32(vmulq_f32(val, vid));
                isum = vaddq_s32(isum, ival[k]);
            }
            int s = vaddvq_s32(isum);
            isum_tot += s;
            y[i].bsums[ib] = s;
            for (int k = 0; k < 4; ++k) {
                auto i16 = vcombine_s16(vmovn_s32(ival[2*k+0]), vmovn_s32(ival[2*k+1]));
                vst1_s8(y[i].qs + 32*ib + 8*k, vmovn_s16(i16));
            }
        }
        y[i].s = isum_tot;
    }
#else
    for (int i = 0; i < nb; i++) {

        float amax = 0;
        for (int j = 0; j < kBlockSize; ++j) {
            float ax = std::abs(x[j]);
            amax = std::max(amax, ax);
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, kBlockSize);
            memset(y[i].bsums, 0, kBlockSize/32*(sizeof(int16_t)));
            x += kBlockSize;
            continue;
        }
        const float iscale = 127.f/amax;
        for (int j = 0; j < kBlockSize; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = v;
        }
        int isum_tot = 0;
        for (int j = 0; j < kBlockSize/32; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 32; ++ii) {
                sum += y[i].qs[j*32 + ii];
            }
            isum_tot += sum;
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        y[i].s = isum_tot;
        x += kBlockSize;
    }
#endif
}
// TODO: merge this with the above template
void iqk_quantize_row_q8_KV(const float * x, void * vy, int64_t k) {
    assert(k % 32 == 0);
    auto dptr = (float *)vy;
    auto q8 = (int8_t *)(dptr + 2);
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m256 maxAbs = _mm256_setzero_ps();
    for (int ib = 0; ib < k/8; ++ib) {
        const __m256 v = _mm256_loadu_ps(x + 8*ib);
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
    }
    const float maxScalar = hmax_f32_8(maxAbs);
    if (!maxScalar) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = maxScalar / 127.f;
    auto mul = _mm256_set1_ps(1/dptr[0]);
    auto isum = _mm256_setzero_si256();
    for (int i = 0; i < k/32; i++) {
        __m256 v0 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i +  0));
        __m256 v1 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i +  8));
        __m256 v2 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i + 16));
        __m256 v3 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i + 24));
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        isum = _mm256_add_epi32(isum, _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        i0 = _mm256_packs_epi32( i0, i1 );
        i2 = _mm256_packs_epi32( i2, i3 );
        i0 = _mm256_packs_epi16( i0, i2 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );
        _mm256_storeu_si256((__m256i *)q8, i0);
        q8 += 32;
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = hsum_i32_8(isum);
#elif defined __ARM_NEON
    int32x4_t ival[8];
    auto vmax = vdupq_n_f32(0.f);
    for (int j = 0; j < k; j += 4) {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + j)));
    }
    auto smax = vmaxvq_f32(vmax);
    if (!smax) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = smax/127;
    auto vid = vdupq_n_f32(1/dptr[0]);
    auto isum = vdupq_n_s32(0);
    for (int ib = 0; ib < k/32; ++ib) {
        auto xb = x + 32*ib;
        for (int k = 0; k < 8; ++k) {
            auto val = vld1q_f32(xb + 4*k);
            ival[k] = vcvtnq_s32_f32(vmulq_f32(val, vid));
            isum = vaddq_s32(isum, ival[k]);
        }
        for (int k = 0; k < 4; ++k) {
            auto i16 = vcombine_s16(vmovn_s32(ival[2*k+0]), vmovn_s32(ival[2*k+1]));
            vst1_s8(q8, vmovn_s16(i16));
            q8 += 8;
        }
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = vaddvq_s32(isum);
#else
    float amax = 0;
    for (int j = 0; j < k; ++j) {
        float ax = std::abs(x[j]);
        amax = std::max(amax, ax);
    }
    if (!amax) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = amax/127;
    float id = 1/dptr[0];
    int isum = 0;
    for (int i = 0; i < k; i++) {
        q8[i] = nearest_int(id*x[i]);
        isum += q8[i];
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = isum;
#endif
}
}

void quantize_row_q8_K128(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_K128(x, vy, k);
}

// ============================== MXFP4



void quantize_row_mxfp4(const float * x, void * y, int64_t k) {
    float weight[QK_MXFP4];
    quantize_row_mxfp4_impl(k, x, (char *)y, weight, kvalues_mxfp4, nullptr, 7, false);
}

size_t quantize_mxfp4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
         [[maybe_unused]] const quantize_user_data * user_data) {
    constexpr int kBlockSize = QK_MXFP4;
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    auto row_size = ggml_row_size(GGML_TYPE_MXFP4, n_per_row);
    char * qrow = (char *)dst;
    float weight[kBlockSize];
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_mxfp4_impl(n_per_row, src, qrow, weight, kvalues_mxfp4, imatrix, 7, true);   // offline quantiser: block-scale code search
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}


void  vec_dot_mxfp4_q8_0_x4(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_MXFP4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_MXFP4 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    //const block_mxfp4 * x = (const block_mxfp4 *)vx;
    //const block_q8_K  * y = (const block_q8_K    *)vy;
    //int nblock = n/QK_MXFP4;
    //float sumf = 0;
    //for (int ibl = 0; ibl < nblock; ++ibl) {
    //    //int sumi = 0;
    //    auto qy = y[ibl].qs;
    //    auto qx = x[ibl].qs;
    //    float db = d * y[ibl].d;
    //    for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
    //        float dl = db * ((x[ibl].scales[ib] & 254) - 127);
    //        //int ls = (x[ibl].scales[ib] & 254) - 127;
    //        const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
    //        int suml = 0;
    //        for (int j = 0; j < kBlockSize/2; ++j) {
    //            suml += qy[j               ] * values[qx[j] & 0xf]
    //                  + qy[j + kBlockSize/2] * values[qx[j] >>  4];
    //        }
    //        sumf += dl * suml;
    //        //sumi += ls * suml;
    //        qy += kBlockSize;
    //        qx += kBlockSize/2;
    //    }
    //    //sumf += d * y[ibl].d * sumi;
    //}
    //*s = sumf;
}


void quantize_row_mxfp4_r8(const float * x, void * y, int64_t k) {
    quantize_mxfp4_r8(x, (void *)y, 8, k/8, nullptr, nullptr);
}



void vec_dot_mxfp4_r8_q8_2_x4(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_MXFP4_R8, vx, 0, GGML_TYPE_Q8_2_X4, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_MXFP4 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

namespace {
// ggml_ue4m3_to_fp32 here is the fork's (ggml-impl.h): it already returns raw*0.5 (kvalues_mxfp4 convention), which is
// the 0.5f ik 30903876b multiplies in by hand with its own raw-valued helper. Same numbers except scale byte 0x7F.
inline float nvfp4_scale(float row_scale, uint8_t byte) {
    return row_scale*ggml_ue4m3_to_fp32(byte);
}
void dequantize_nvfp4_blocks(const block_nvfp4 * x, float * y, int nblock, float row_scale) {
    for (int ib = 0; ib < nblock; ++ib) {
        for (int is = 0; is < QK_NVFP4/16; ++is) {
            const float d = nvfp4_scale(row_scale, x[ib].d[is]);
            const uint8_t * qs = x[ib].qs + 8*is;
            for (int j = 0; j < 8; ++j) {
                y[j    ] = d*kvalues_mxfp4[qs[j] & 0xf];
                y[j + 8] = d*kvalues_mxfp4[qs[j] >>  4];
            }
            y += 16;
        }
    }
}
void quantize_row_nvfp4_ks_impl(int n_per_row, const float * x, char * cy, const float * quant_weights) {
    float * dptr = (float *)cy;
    block_nvfp4 * y = (block_nvfp4 *)(dptr + 1);
    const int nblock = n_per_row/QK_NVFP4;
    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (!amax_row) {
        *dptr = 0;
        memset(y, 0, nblock*sizeof(block_nvfp4));
        return;
    }
    const float row_scale = amax_row/(6*448.f);
    *dptr = row_scale;
    float weight[16];
    for (int ib = 0; ib < nblock; ++ib) {
        const float * xb = x + ib*QK_NVFP4;
        float sigma2 = 0;
        if (quant_weights) {
            for (int j = 0; j < QK_NVFP4; ++j) sigma2 += xb[j]*xb[j];
            sigma2 *= 2.f/QK_NVFP4;
        }
        for (int is = 0; is < QK_NVFP4/16; ++is) {
            const float * xs = xb + 16*is;
            float amax = 0;
            for (int j = 0; j < 16; ++j) amax = std::max(amax, std::abs(xs[j]));
            uint8_t byte = ggml_fp32_to_ue4m3(amax/6/row_scale);
            if (quant_weights) {
                const float * qw = quant_weights + ib*QK_NVFP4 + 16*is;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j]*sqrtf(sigma2 + xs[j]*xs[j]);
                auto block_error = [&] (int b) {
                    const float d = nvfp4_scale(row_scale, b);
                    float err = 0;
                    for (int j = 0; j < 16; ++j) {
                        const float diff = xs[j] - d*kvalues_mxfp4[best_index_mxfp4(d, kvalues_mxfp4, xs[j])];
                        err += weight[j]*diff*diff;
                    }
                    return err;
                };
                const int rtn = byte;
                float best = block_error(rtn);
                for (int b = std::max(0, rtn - 2); b <= std::min(0x7e, rtn + 2); ++b) {
                    if (b == rtn) continue;
                    const float err = block_error(b);
                    if (err < best) {
                        best = err;
                        byte = b;
                    }
                }
            }
            y[ib].d[is] = byte;
            const float d = nvfp4_scale(row_scale, byte);
            for (int j = 0; j < 8; ++j) {
                y[ib].qs[8*is + j] = best_index_mxfp4(d, kvalues_mxfp4, xs[j]) | (best_index_mxfp4(d, kvalues_mxfp4, xs[j + 8]) << 4);
            }
        }
    }
}
}

void quantize_row_nvfp4_ks_ref(const float * x, block_nvfp4 * y, int64_t k) {
    quantize_nvfp4_ks(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_nvfp4_ks(const float * x, void * y, int64_t k) {
    quantize_nvfp4_ks(x, y, 1, k, nullptr, nullptr);
}

size_t quantize_nvfp4_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row%QK_NVFP4 == 0);
    auto row_size = ggml_row_size(GGML_TYPE_NVFP4_KS, n_per_row);
    QHelper helper(imatrix, user_data, n_per_row, 16);
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_nvfp4_ks_impl(n_per_row, x, (char *)vy, imatrix);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows*row_size;
}

void dequantize_row_nvfp4_ks(const block_nvfp4 * x, float * y, int64_t k) {
    const float * dptr = (const float *)x;
    dequantize_nvfp4_blocks((const block_nvfp4 *)(dptr + 1), y, k/QK_NVFP4, dptr[0]);
}

// SOURCE-READ: ik_llama.cpp c7ea52b33 ggml/src/iqk/iqk_quantize.cpp ggml_mxfp4_to_nvfp4_ks (MIT, adapted; scale helpers are the fork's).
// One MXFP4 row -> one NVFP4_KS row (float row scale + 64-wide blocks of four 16-wide UE4M3-scaled groups). Each MXFP4 block's
// 2^k scale becomes a UE4M3 byte relative to the row scale 2^(emax-134) (ik uses 135 and k up to 8; the fork's ggml_fp32_to_ue4m3 clamps exponent field 15 to 0x7E, so k is kept <= 7); blocks more than 16 octaves under the largest are
// requantized onto the smallest representable scale and the call returns false if any of those held a nonzero value.
bool ggml_mxfp4_to_nvfp4_ks(const void * vx, void * vy, int n_per_row) {
    const block_mxfp4 * x = (const block_mxfp4 *)vx;
    const int nblock = n_per_row/QK_MXFP4;
    auto nonzero = [] (const block_mxfp4 & b) {
        for (int j = 0; j < QK_MXFP4/2; ++j) if (b.qs[j] & 0x77) return true;
        return false;
    };
    int emax = -1;
    for (int ib = 0; ib < nblock; ++ib) {
        if (nonzero(x[ib])) emax = std::max(emax, int(x[ib].e));
    }
    float * dptr = (float *)vy;
    block_nvfp4 * y = (block_nvfp4 *)(dptr + 1);
    if (emax < 0) {
        *dptr = 0;
        memset(y, 0, (nblock/2)*sizeof(block_nvfp4));
        return true;
    }
    const float row_scale = ldexpf(1.0f, emax - 134);
    *dptr = row_scale;
    bool exact = true;
    for (int ib = 0; ib < nblock; ++ib) {
        auto & yb = y[ib/2];
        const int is = 2*(ib%2);
        const int k = x[ib].e - emax + 7;
        if (k >= -9) {
            const uint8_t byte = ggml_fp32_to_ue4m3(ldexpf(1.0f, k));
            for (int h = 0; h < 2; ++h) {
                yb.d[is + h] = byte;
                for (int j = 0; j < 8; ++j) {
                    yb.qs[8*(is + h) + j] = ((x[ib].qs[j] >> 4*h) & 0xf) | (((x[ib].qs[j + 8] >> 4*h) & 0xf) << 4);
                }
            }
        } else {
            exact &= !nonzero(x[ib]);
            const float d0 = GGML_E8M0_TO_FP32_HALF(x[ib].e);
            const float d  = nvfp4_scale(row_scale, 1);
            for (int h = 0; h < 2; ++h) {
                yb.d[is + h] = 1;
                for (int j = 0; j < 8; ++j) {
                    const int i0 = best_index_mxfp4(d, kvalues_mxfp4, d0*kvalues_mxfp4[(x[ib].qs[j    ] >> 4*h) & 0xf]);
                    const int i1 = best_index_mxfp4(d, kvalues_mxfp4, d0*kvalues_mxfp4[(x[ib].qs[j + 8] >> 4*h) & 0xf]);
                    yb.qs[8*(is + h) + j] = i0 | (i1 << 4);
                }
            }
        }
    }
    return exact;
}

void vec_dot_nvfp4_ks_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_NVFP4_KS, vx, 0, ggml_internal_get_type_traits(GGML_TYPE_NVFP4_KS).vec_dot_type, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);
    const float * dptr = (const float *)vx;
    const float row_scale = dptr[0];
    const block_nvfp4 * x = (const block_nvfp4 *)(dptr + 1);
    const block_q8_0  * y = (const block_q8_0  *)vy;
    float sumf = 0;
    for (int ib = 0; ib < n/QK_NVFP4; ++ib) {
        for (int is = 0; is < QK_NVFP4/16; ++is) {
            const block_q8_0 & yb = y[2*ib + is/2];
            const int8_t  * q8 = yb.qs + 16*(is%2);
            const uint8_t * qs = x[ib].qs + 8*is;
            int sumi = 0;
            for (int j = 0; j < 8; ++j) {
                sumi += q8[j]*kvalues_mxfp4[qs[j] & 0xf] + q8[j + 8]*kvalues_mxfp4[qs[j] >> 4];
            }
            sumf += GGML_FP16_TO_FP32(yb.d)*nvfp4_scale(row_scale, x[ib].d[is])*sumi;
        }
    }
    *s = sumf;
}

namespace {
static void quantize_row_iq4_k_impl_bs32(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    block_iq4_ks * y = (block_iq4_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    float amax_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KS);

#ifdef __AVX2__
    float f_values[32];
    for (int j = 0; j < 32; ++j) f_values[j] = values[j];
#endif

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            bool is_shifted = false;
            float best = 0;
#ifdef __AVX2__
            __m256 vx[block_size/8], vw[block_size/8];
            for (int j = 0; j < block_size/8; ++j) vx[j] = _mm256_loadu_ps(xb + 8*j);
            for (int j = 0; j < block_size/8; ++j) vw[j] = _mm256_loadu_ps(weight + 8*j);
            (void)evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best);
#else
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
#endif
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
#ifdef __AVX2__
                if (evaluate_block_iq4k<block_size>(id, values, f_values, vx, vw, d, best)) {
                    is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                if (evaluate_block_iq4k<block_size>(id, values+16, f_values+16, vx, vw, d, best)) {
                    is_shifted = true;
                }
#else
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
#endif
            }
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KS)*d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + ib*(block_size/2);
            for (int j = 0; j < block_size/2; ++j) {
                uint8_t i1 = best_index_iq4nl(block_values, idl*xb[j]);
                uint8_t i2 = best_index_iq4nl(block_values, idl*xb[j+block_size/2]);
                qs[j] = i1 | (i2 << 4);
                float w1 = weight[j];
                float w2 = weight[j+block_size/2];
                float q1 = block_values[i1]*l;
                float q2 = block_values[i2]*l;
                sumqx += w1*q1*xb[j] + w2*q2*xb[j+block_size/2];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) *dptr = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KS)*sumqx/sumq2;
}
}

void quantize_row_iq4_ks_ref(const float * x, block_iq4_ks * y, int64_t k) {
    quantize_iq4_ks(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_iq4_ks(const float * x, void * y, int64_t k) {
    quantize_iq4_ks(x, (void *)y, 1, k, nullptr, nullptr);
}

size_t quantize_iq4_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq4_k_impl_bs32(n_per_row, x, (char *)vy, all_scales.data(), weight, iq4k_values, imatrix, 7);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}


void  vec_dot_iq4_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    constexpr int kBlockSize = 32;
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    const float * dptr = (const float *)vx;
    const float d = *dptr;
    //printf("%s: n = %d, d = %g\n", __func__, n, d);
    const block_iq4_ks * x = (const block_iq4_ks *)(dptr + 1);
    const block_q8_K    * y = (const block_q8_K    *)vy;
    int nblock = n/QK_K;
    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        //int sumi = 0;
        auto qy = y[ibl].qs;
        auto qx = x[ibl].qs;
        float db = d * y[ibl].d;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            float dl = db * ((x[ibl].scales[ib] & 254) - 127);
            //int ls = (x[ibl].scales[ib] & 254) - 127;
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            int suml = 0;
            for (int j = 0; j < kBlockSize/2; ++j) {
                suml += qy[j               ] * values[qx[j] & 0xf]
                      + qy[j + kBlockSize/2] * values[qx[j] >>  4];
            }
            sumf += dl * suml;
            //sumi += ls * suml;
            qy += kBlockSize;
            qx += kBlockSize/2;
        }
        //sumf += d * y[ibl].d * sumi;
    }
    *s = sumf;
}



void quantize_row_iq5_ks(const float * x, void * y, int64_t k) {
    quantize_iq5_ks(x, (void *)y, 1, k, nullptr, nullptr);
}



void  vec_dot_iq5_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    constexpr int kBlockSize = 32;
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    const float * dptr = (const float *)vx;
    const float d = *dptr;
    const block_iq5_ks * x = (const block_iq5_ks *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K    *)vy;
    int nblock = n/QK_K;
    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qy = y[ibl].qs;
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        float db = d * y[ibl].d;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            float dl1 = db * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            float dl2 = db * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * values1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            int suml1 = 0;
            int suml2 = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                suml1 += qy[j           ] * values1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                suml2 += qy[j+kBlockSize] * values2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
            }
            sumf += dl1*suml1 + dl2*suml2;
            y  += 2*kBlockSize;
            qs += kBlockSize;
        }
    }
    *s = sumf;
}




void quantize_row_iq4_kss(const float * x, void * y, int64_t k) {
    quantize_iq4_kss(x, (block_iq4_kss *)y, 1, k, nullptr, nullptr);
}


void vec_dot_iq4_kss_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KSS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


void quantize_row_iq4_nl_r4(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_iq4_nl_r4(x, y, 4, k/4, nullptr, nullptr);
}




void vec_dot_iq4_nl_r4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_NL_R4, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= q4_0_r8
//
void quantize_row_q4_0_r8_ref(const float * x, block_iq4_nl_r8  * y, int64_t k) {
    // we assume we are called with 8 rows
    quantize_q4_0_r8(x, (void *)y, 8, k/8, nullptr, nullptr);
}

void quantize_row_q4_0_r8(const float * x, void * y, int64_t k) {
    // we assume we are called with 8 rows
    quantize_q4_0_r8(x, y, 8, k/8, nullptr, nullptr);
}

static void repack_q4_0(int nrows, int n_per_row, const block_q4_0 * x, block_iq4_nl_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK4_0 == 0);
    int nblock = n_per_row/QK4_0;
    const block_q4_0 * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 8; ++k) {
                y[ib].d[k] = x8[k][ib].d;
                for (int l = 0; l < 4; ++l) {
                    for (int i = 0; i < 4; ++i) {
                        y[ib].qs[32*l+4*k+i] = x8[k][ib].qs[4*l + i];
                    }
                }
            }
#ifdef __ARM_NEON
            if (online) {
                for (int l = 0; l < 8; ++l) {
                    auto v = vld1q_u8(y[ib].qs + 16*l);
                    vst1q_u8(y[ib].qs + 16*l, veorq_u8(v, vdupq_n_u8(0x88)));
                }
            }
#endif
        }
        x += 8*nblock;
        y += nblock;
    }
}
#ifdef __ARM_NEON
static void modify_q4_0_r8(int64_t k, char * cy) {
    auto y = (block_iq4_nl_r8 *)cy;
    int nb = k/(32*8);
    for (int ib = 0; ib < nb; ++ib) {
        auto v1 = vld1q_u8_x4(y[ib].qs);
        auto v2 = vld1q_u8_x4(y[ib].qs+64);
        for (int j = 0; j < 4; ++j) {
            v1.val[j] = veorq_u8(v1.val[j], vdupq_n_u8(0x88));
            v2.val[j] = veorq_u8(v2.val[j], vdupq_n_u8(0x88));
        }
        vst1q_u8_x4(y[ib].qs+ 0, v1);
        vst1q_u8_x4(y[ib].qs+64, v2);
    }
}
#endif

static void repack_mxfp4(int nrows, int n_per_row, const block_mxfp4 * x, block_mxfp4_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK_MXFP4 == 0);
    int nblock = n_per_row/QK_MXFP4;
    const block_mxfp4 * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 8; ++k) {
                y[ib].e[k] = x8[k][ib].e;
                for (int l = 0; l < 4; ++l) {
                    for (int i = 0; i < 4; ++i) {
                        y[ib].qs[32*l+4*k+i] = x8[k][ib].qs[4*l + i];
                    }
                }
            }
        }
        x += 8*nblock;
        y += nblock;
    }
}

size_t quantize_q4_0_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%8 == 0);
    auto row_size_nl = ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
    std::vector<char> qtmp(8*row_size_nl);
    QHelper helper(imatrix, user_data, n_per_row, 32);
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_q4_0(x, (char *)vy, 1, n_per_row, imatrix, nullptr);
    };
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 8) {
        helper.quantize(8, src, qtmp.data(), row_size_nl, q_func);
        repack_q4_0(8, n_per_row, (const block_q4_0 *)qtmp.data(), (block_iq4_nl_r8 *)qrow, false);
        src += 8*n_per_row;
        qrow += 8*row_size_nl;
    }
    return nrows*row_size_nl;
}


void vec_dot_q4_0_r8_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q4_0_R8, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


//
// ========================================= q2_0_r8
//
void quantize_row_q2_0_r8_ref(const float * x, block_q2_0_r8  * y, int64_t k) {
    // we assume we are called with 8 rows
    quantize_q2_0_r8(x, (void *)y, 8, k/8, nullptr, nullptr);
}

void quantize_row_q2_0_r8(const float * x, void * y, int64_t k) {
    // we assume we are called with 8 rows
    quantize_q2_0_r8(x, y, 8, k/8, nullptr, nullptr);
}

static void repack_q2_0(int nrows, int n_per_row, const block_q2_0 * x, block_q2_0_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK2_0 == 0);
    int nblock = n_per_row/QK2_0;
    const block_q2_0 * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 8; ++k) y[ib].d[k] = x8[k][ib].d;
            for (int p = 0; p < 2; ++p) {
                for (int k = 0; k < 8; ++k) {
                    const uint8_t * src = x8[k][ib].qs + 8*p;
                    for (int i = 0; i < 4; ++i) {
                        uint8_t b1 = 0, b2 = 0;
                        for (int t = 0; t < 4; ++t) {
                            b1 |= ((src[t    ] >> 2*i) & 3) << (2*t);
                            b2 |= ((src[4 + t] >> 2*i) & 3) << (2*t);
                        }
                        y[ib].qs[64*p +     4*k + i] = b1;
                        y[ib].qs[64*p + 32 + 4*k + i] = b2;
                    }
                }
            }
        }
        x += 8*nblock;
        y += nblock;
    }
}

size_t quantize_q2_0_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%8 == 0);
    auto row_size_nl = ggml_row_size(GGML_TYPE_Q2_0, n_per_row);
    std::vector<char> qtmp(8*row_size_nl);
    QHelper helper(imatrix, user_data, n_per_row, QK2_0);
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        // fork quantize_q2_0 (ggml-quants.h:84) is the 5-argument upstream form: no user_data (ik passes nullptr)
        quantize_q2_0(x, (char *)vy, 1, n_per_row, imatrix);
    };
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 8) {
        helper.quantize(8, src, qtmp.data(), row_size_nl, q_func);
        repack_q2_0(8, n_per_row, (const block_q2_0 *)qtmp.data(), (block_q2_0_r8 *)qrow, false);
        src += 8*n_per_row;
        qrow += 8*row_size_nl;
    }
    return nrows*row_size_nl;
}

void dequantize_row_q2_0_r8(const block_q2_0_r8 * x, float * y, int64_t k) {
    // we assume we are called with 8 rows
    int n_per_row = k/8;
    int nb = n_per_row/QK2_0;
    float * yk[8];
    for (int kk = 0; kk < 8; ++kk) yk[kk] = y + kk*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int kk = 0; kk < 8; ++kk) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[kk]);
            for (int p = 0; p < 2; ++p) {
                for (int i = 0; i < 4; ++i) {
                    uint8_t b1 = x[ib].qs[64*p +     4*kk + i];
                    uint8_t b2 = x[ib].qs[64*p + 32 + 4*kk + i];
                    for (int t = 0; t < 4; ++t) {
                        yk[kk][QK2_0*ib + 32*p +  4*t + i] = scale * (((b1 >> 2*t) & 3) - 1);
                        yk[kk][QK2_0*ib + 32*p + 16 + 4*t + i] = scale * (((b2 >> 2*t) & 3) - 1);
                    }
                }
            }
        }
    }
}

void vec_dot_q2_0_r8_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q2_0_R8, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK2_0 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q8_0_r8
//
void quantize_row_q8_0_r8_ref(const float * x, block_q8_0_r8  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q8_0_r8(x, (void *)y, 8, k/8, nullptr, nullptr);
}

void quantize_row_q8_0_r8(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q8_0_r8(x, y, 8, k/8, nullptr, nullptr);
}

static void repack_q8_0(int nrows, int n_per_row, const block_q8_0 * x, block_q8_0_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK8_0 == 0);
    int nblock = n_per_row/QK8_0;
    const block_q8_0 * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 8; ++k) y[ib].d[k] = x8[k][ib].d;
            for (int l = 0; l < 4; ++l) {
                for (int k = 0; k < 8; ++k) for (int i = 0; i < 4; ++i) {
                    y[ib].qs[32*l+4*k+i+  0] = x8[k][ib].qs[i+4*l+ 0];
                    y[ib].qs[32*l+4*k+i+128] = x8[k][ib].qs[i+4*l+16];
                }
            }
#ifdef HAVE_FANCY_SIMD
            if (online) {
                for (int l = 0; l < 4; ++l) {
                    auto v = _mm512_add_epi8(_mm512_loadu_si512((const __m512i *)y[ib].qs + l), _mm512_set1_epi8(127));
                    _mm512_storeu_si512((__m512i *)y[ib].qs + l, v);
                }
            }
#endif
        }
        x += 8*nblock;
        y += nblock;
    }
}

#ifdef HAVE_FANCY_SIMD
static void modify_q8_0_r8(int64_t k, char * cy) {
    auto y = (block_q8_0_r8 *)cy;
    int nb = k/(32*8);
    for (int ib = 0; ib < nb; ++ib) {
        for (int l = 0; l < 4; ++l) {
            auto v = _mm512_add_epi8(_mm512_loadu_si512((const __m512i *)y[ib].qs + l), _mm512_set1_epi8(127));
            _mm512_storeu_si512((__m512i *)y[ib].qs + l, v);
        }
    }
}
#endif

size_t quantize_q8_0_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%8 == 0);
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<char> qtmp(8*row_size_0);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 8) {
        quantize_q8_0(src, qtmp.data(), 8, n_per_row, imatrix, nullptr);
        repack_q8_0(8, n_per_row, (const block_q8_0 *)qtmp.data(), (block_q8_0_r8 *)qrow, false);
        src += 8*n_per_row;
        qrow += 8*row_size_0;
    }
    return nrows*row_size_0;
}


void vec_dot_q8_0_r8_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_0_R8, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


void quantize_row_q5_0_r4(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q5_0_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_q5_0_r4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q5_0_R4, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


void quantize_row_q6_0_r4(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q6_0_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_q6_0_r4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q6_0_R4, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq4_xs_r8
//


void quantize_row_iq4_xs_r8(const float * x, void * y, int64_t k) {
    quantize_iq4_xs_r8(x, y, 8, k/8, nullptr, nullptr);
}




void vec_dot_iq4_xs_r8_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_XS_R8, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq4_ks_r4
//

void quantize_row_iq4_ks_r4_ref(const float * x, block_iq4_ks_r4 * y, int64_t k) {
    quantize_iq4_ks_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}

void quantize_row_iq4_ks_r4(const float * x, void * y, int64_t k) {
    quantize_iq4_ks_r4(x, y, 4, k/4, nullptr, nullptr);
}

void repack_iq4_ks(int nrows, int n_per_row, const block_iq4_ks * x, block_iq4_ks_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    char * cy = (char *)y;
    const char * cx = (const char *)x;
    const block_iq4_ks * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        float * dptr = (float *)cy;
        block_iq4_ks_r4 * y = (block_iq4_ks_r4 *)(dptr + 4);
        for (int k = 0; k < 4; ++k) {
            auto dk = (const float *)(cx + k*row_size);
            dptr[k] = dk[0];
            x4[k] = (const block_iq4_ks *)(dk + 1);
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = (x4[k][ibl].qs[16*ib+i+0] & 0xf) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0x0f) << 4);  //  0....3 +  8...11 from each row
                        y[ibl].qs[64*ib+4*k+i+16] = (x4[k][ibl].qs[16*ib+i+0] >>  4) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0xf0));       // 16...19 + 24...27 from each row
                        y[ibl].qs[64*ib+4*k+i+32] = (x4[k][ibl].qs[16*ib+i+4] & 0xf) | ((x4[k][ibl].qs[16*ib+i+12] & 0x0f) << 4);  //  4....7 + 12...15 from each row
                        y[ibl].qs[64*ib+4*k+i+48] = (x4[k][ibl].qs[16*ib+i+4] >>  4) | ((x4[k][ibl].qs[16*ib+i+12] & 0xf0));       // 20...23 + 28...31 from each row
                    }
                }
            }
        }
        cx += 4*row_size;
        cy += 4*row_size;
    }
}

size_t quantize_iq4_ks_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq4_ks(src, (void *)qtmp.data(), 4, n_per_row, imatrix, user_data);
        repack_iq4_ks(4, n_per_row, (const block_iq4_ks *)qtmp.data(), (block_iq4_ks_r4 *)qcur, false);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}


void vec_dot_iq4_ks_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq4_ks_r16
//

void quantize_row_iq4_ks_r16_ref(const float * x, block_iq4_ks_r16 * y, int64_t k) {
    // FORK FIX of ik_llama #2480 (still so at ik main cdf232cc1): ik passes (4, k/4) here, copied from the _r4 sibling, but
    // quantize_iq4_ks_r16 asserts nrows%16 == 0, so any caller of from_float_ref for this type aborts (found by test-iqk-golden,
    // which enumerates every IQ*_K type). 16 rows of k/16 is what quantize_row_iq4_ks_r16 right below does.
    quantize_iq4_ks_r16(x, (void *)y, 16, k/16, nullptr, nullptr);
}

void quantize_row_iq4_ks_r16(const float * x, void * y, int64_t k) {
    quantize_iq4_ks_r16(x, y, 16, k/16, nullptr, nullptr);
}

void repack_iq4_ks_r16(int nrows, int n_per_row, const block_iq4_ks * x, block_iq4_ks_r16 * y, [[maybe_unused]] bool online, int n_per_row_stored) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    GGML_ASSERT(n_per_row_stored <= n_per_row);
    GGML_ASSERT(n_per_row_stored % QK8_0 == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    auto row_size_stored = ggml_row_size(GGML_TYPE_IQ4_KS_R16, n_per_row_stored);
    int nblock = n_per_row/QK_K;
    char * cy = (char *)y;
    const char * cx = (const char *)x;
    const block_iq4_ks * x16[16];
    for (int row = 0; row < nrows; row += 16) {
        float * dptr = (float *)cy;
        block_iq4_ks_r16 * y = (block_iq4_ks_r16 *)(dptr + 16);
        for (int k = 0; k < 16; ++k) {
            auto dk = (const float *)(cx + k*row_size);
            dptr[k] = dk[0];
            x16[k] = (const block_iq4_ks *)(dk + 1);
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            int first = ibl*QK_K;
            int last  = std::min(first + QK_K, n_per_row_stored);
            int nval  = last - first;
            for (int ib = 0; ib < nval/32; ++ib) {
                int ir16 = 8*ibl + ib;
                auto qy = (uint32_t *)y[ir16].qs;
                for (int k = 0; k < 16; ++k) {
                    y[ir16].scales[k] = x16[k][ibl].scales[ib];
                    auto qx = (const uint32_t *)x16[k][ibl].qs + 4*ib;
                    qy[k +  0] = qx[0]; //  0....3 and 16...19
                    qy[k + 16] = qx[1]; //  4....7 and 20...23
                    qy[k + 32] = qx[2]; //  8...11 and 24...27
                    qy[k + 48] = qx[3]; // 12...15 and 28...31
                }
            }
        }
        cx += 16*row_size;
        cy += 16*row_size_stored;
    }
}

size_t quantize_iq4_ks_r16(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%QK8_0 == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS_R16, n_per_row);
    if (n_per_row % QK_K == 0) {
        char * qcur = (char *)dst;
        std::vector<char> qtmp(16*row_size);
        for (int row = 0; row < nrows; row += 16) {
            quantize_iq4_ks(src, (void *)qtmp.data(), 16, n_per_row, imatrix, user_data);
            repack_iq4_ks_r16(16, n_per_row, (const block_iq4_ks *)qtmp.data(), (block_iq4_ks_r16 *)qcur, false, n_per_row);
            qcur += 16*row_size;
            src += 16*n_per_row;
        }
        return nrows*row_size;
    }
    int nblock256 = (n_per_row + QK_K - 1)/QK_K;
    int n_per_row_padded = nblock256 * QK_K;
    std::vector<float> imatrix_padded;
    if (imatrix) {
        imatrix_padded.resize(n_per_row_padded);
        double sum2 = 0;
        for (int j = 0; j < n_per_row; ++j) {
            imatrix_padded[j] = imatrix[j];
            sum2 += imatrix[j]*imatrix[j];
        }
        float padded_value = sqrt(sum2/n_per_row);
        for (int j = n_per_row; j < n_per_row_padded; ++j) imatrix_padded[j] = padded_value;
        imatrix = imatrix_padded.data();
    }
    auto row_size_padded = ggml_row_size(GGML_TYPE_IQ4_KS_R16, n_per_row_padded);
    std::vector<char> qtmp(16*row_size_padded);
    std::vector<float> ftmp(16*n_per_row_padded);
    char * qcur = (char *)dst;
    int npad = n_per_row_padded - n_per_row;
    for (int row = 0; row < nrows; row += 16) {
        for (int k = 0; k < 16; ++k) {
            std::memcpy(ftmp.data() + k*n_per_row_padded, src + k*n_per_row, n_per_row*sizeof(float));
            std::memcpy(ftmp.data() + k*n_per_row_padded + n_per_row, src + (k+1)*n_per_row - npad, npad*sizeof(float));
        }
        quantize_iq4_ks(ftmp.data(), (void *)qtmp.data(), 16, n_per_row_padded, imatrix, user_data);
        repack_iq4_ks_r16(16, n_per_row_padded, (const block_iq4_ks *)qtmp.data(), (block_iq4_ks_r16 *)qcur, false, n_per_row);
        qcur += 16*row_size;
        src += 16*n_per_row;
    }
    return nrows*row_size;
}


void vec_dot_iq4_ks_r16_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KS_R16, vx, 0, GGML_TYPE_Q8_2_X4, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq5_ks_r16
//

void quantize_row_iq5_ks_r16_ref(const float * x, block_iq5_ks_r16 * y, int64_t k) {
    // FORK FIX, same as quantize_row_iq4_ks_r16_ref above: ik 1b18524bd passes (4, k/4), copied from the _r4 sibling, but
    // quantize_iq5_ks_r16 asserts nrows%16 == 0, so every from_float_ref caller (test-iqk-golden) would abort.
    quantize_iq5_ks_r16(x, (void *)y, 16, k/16, nullptr, nullptr);
}

void quantize_row_iq5_ks_r16(const float * x, void * y, int64_t k) {
    quantize_iq5_ks_r16(x, y, 16, k/16, nullptr, nullptr);
}

void repack_iq5_ks_r16(int nrows, int n_per_row, const block_iq5_ks * x, block_iq5_ks_r16 * y, [[maybe_unused]] bool online, int n_per_row_stored) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    GGML_ASSERT(n_per_row_stored <= n_per_row);
    GGML_ASSERT(n_per_row_stored % QK8_0 == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    auto row_size_stored = ggml_row_size(GGML_TYPE_IQ5_KS_R16, n_per_row_stored);
    int nblock = n_per_row/QK_K;
    char * cy = (char *)y;
    const char * cx = (const char *)x;
    const block_iq5_ks * x16[16];
    for (int row = 0; row < nrows; row += 16) {
        float * dptr = (float *)cy;
        block_iq5_ks_r16 * y = (block_iq5_ks_r16 *)(dptr + 16);
        for (int k = 0; k < 16; ++k) {
            auto dk = (const float *)(cx + k*row_size);
            dptr[k] = dk[0];
            x16[k] = (const block_iq5_ks *)(dk + 1);
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            int first = ibl*QK_K;
            int last  = std::min(first + QK_K, n_per_row_stored);
            int nval  = last - first;
            for (int ib = 0; ib < nval/32; ++ib) {
                int ir16 = 8*ibl + ib;
                auto qyl = (uint32_t *)y[ir16].qs;
                auto qyh = (uint32_t *)y[ir16].qh;
                for (int k = 0; k < 16; ++k) {
                    y[ir16].scales[k] = x16[k][ibl].scales[ib];
                    auto qxl = (const uint32_t *)x16[k][ibl].qs + 8*(ib/2);
                    auto qxh = (const uint32_t *)x16[k][ibl].qh;
                    if (ib%2 == 0) {
                        qyl[k +  0] = ((qxl[0] & 0x0f0f0f0f) | ((qxl[4] & 0x0f0f0f0f) << 4)); //  0....3 and 32...35
                        qyl[k + 16] = ((qxl[1] & 0x0f0f0f0f) | ((qxl[5] & 0x0f0f0f0f) << 4)); //  4....7 and 20...23
                        qyl[k + 32] = ((qxl[2] & 0x0f0f0f0f) | ((qxl[6] & 0x0f0f0f0f) << 4)); //  8...11 and 24...27
                        qyl[k + 48] = ((qxl[3] & 0x0f0f0f0f) | ((qxl[7] & 0x0f0f0f0f) << 4)); // 12...15 and 28...31
                    } else {
                        qyl[k +  0] = (((qxl[0] >> 4) & 0x0f0f0f0f) | (qxl[4] & 0xf0f0f0f0)); //  0....3 and 32...35
                        qyl[k + 16] = (((qxl[1] >> 4) & 0x0f0f0f0f) | (qxl[5] & 0xf0f0f0f0)); //  4....7 and 20...23
                        qyl[k + 32] = (((qxl[2] >> 4) & 0x0f0f0f0f) | (qxl[6] & 0xf0f0f0f0)); //  8...11 and 24...27
                        qyl[k + 48] = (((qxl[3] >> 4) & 0x0f0f0f0f) | (qxl[7] & 0xf0f0f0f0)); // 12...15 and 28...31
                    }
                    qyh[k] = 0;
                    for (int j = 0; j < 8; ++j) qyh[k] |= (((qxh[j] >> ib) & 0x01010101) << j);
                }
            }
        }
        cx += 16*row_size;
        cy += 16*row_size_stored;
    }
}

// The IQ5_KS rows that get interleaved come from `base`: the scalar reference quantize_iq5_ks for quantize_iq5_ks_r16
// (from_float / from_float_ref, so test-quantize-fns compares like with like), or the AVX2 fast row search
// quantize_iq5_ks_bulk for quantize_iq5_ks_r16_bulk (ggml_quantize_chunk, i.e. llama-quantize). Upstream ik 1b18524bd
// calls quantize_iq5_ks, which in ik IS the fast search; in this fork the fast search lives in the _bulk entry
// (iqk_quantize.cpp quantize_iq5_ks_bulk), so the bulk R16 entry keeps upstream's quantization speed and quality.
typedef size_t (*iq5_ks_base_quantizer_t)(const float *, void *, int64_t, int64_t, const float *, const quantize_user_data *);
static size_t quantize_iq5_ks_r16_impl(iq5_ks_base_quantizer_t base, const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix, [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%QK8_0 == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS_R16, n_per_row);
    if (n_per_row % QK_K == 0) {
        char * qcur = (char *)dst;
        std::vector<char> qtmp(16*row_size);
        for (int row = 0; row < nrows; row += 16) {
            base(src, (void *)qtmp.data(), 16, n_per_row, imatrix, user_data);
            repack_iq5_ks_r16(16, n_per_row, (const block_iq5_ks *)qtmp.data(), (block_iq5_ks_r16 *)qcur, false, n_per_row);
            qcur += 16*row_size;
            src += 16*n_per_row;
        }
        return nrows*row_size;
    }
    int nblock256 = (n_per_row + QK_K - 1)/QK_K;
    int n_per_row_padded = nblock256 * QK_K;
    std::vector<float> imatrix_padded;
    if (imatrix) {
        imatrix_padded.resize(n_per_row_padded);
        double sum2 = 0;
        for (int j = 0; j < n_per_row; ++j) {
            imatrix_padded[j] = imatrix[j];
            sum2 += imatrix[j]*imatrix[j];
        }
        float padded_value = sqrt(sum2/n_per_row);
        for (int j = n_per_row; j < n_per_row_padded; ++j) imatrix_padded[j] = padded_value;
        imatrix = imatrix_padded.data();
    }
    auto row_size_padded = ggml_row_size(GGML_TYPE_IQ5_KS_R16, n_per_row_padded);
    std::vector<char> qtmp(16*row_size_padded);
    std::vector<float> ftmp(16*n_per_row_padded);
    char * qcur = (char *)dst;
    int npad = n_per_row_padded - n_per_row;
    for (int row = 0; row < nrows; row += 16) {
        for (int k = 0; k < 16; ++k) {
            std::memcpy(ftmp.data() + k*n_per_row_padded, src + k*n_per_row, n_per_row*sizeof(float));
            std::memcpy(ftmp.data() + k*n_per_row_padded + n_per_row, src + (k+1)*n_per_row - npad, npad*sizeof(float));
        }
        base(ftmp.data(), (void *)qtmp.data(), 16, n_per_row_padded, imatrix, user_data);
        repack_iq5_ks_r16(16, n_per_row_padded, (const block_iq5_ks *)qtmp.data(), (block_iq5_ks_r16 *)qcur, false, n_per_row);
        qcur += 16*row_size;
        src += 16*n_per_row;
    }
    return nrows*row_size;
}

size_t quantize_iq5_ks_r16(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    return quantize_iq5_ks_r16_impl(quantize_iq5_ks, src, dst, nrows, n_per_row, imatrix, user_data);
}

size_t quantize_iq5_ks_r16_bulk(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    return quantize_iq5_ks_r16_impl(quantize_iq5_ks_bulk, src, dst, nrows, n_per_row, imatrix, user_data);
}

void dequantize_row_iq5_ks_r16(const block_iq5_ks_r16 * x, float * y, int64_t k) {
    auto n_per_row = k/16;
    float * y16[16];
    for (int k = 0; k < 16; ++k) y16[k] = y + n_per_row*k;
    int nblock = n_per_row/QK8_0;
    const float * dptr = (const float *)x;
    x = (const block_iq5_ks_r16 *)(dptr + 16);
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 16; ++k) {
            const float d = dptr[k];
            float dl = d * ((x[ib].scales[k] & 254) - 127);
            auto values = iq5nl_values + ((x[ib].scales[k] & 1) << 5);
            for (int i = 0; i < 4; ++i) {
                y16[k][QK8_0*ib+i+ 0] = dl * values[(x[ib].qs[4*k+i+  0] & 0xf) | ((x[ib].qh[4*k+i] << 4) & 0x10)];
                y16[k][QK8_0*ib+i+16] = dl * values[(x[ib].qs[4*k+i+  0] >>  4) | ((x[ib].qh[4*k+i] >> 0) & 0x10)];
                y16[k][QK8_0*ib+i+ 4] = dl * values[(x[ib].qs[4*k+i+ 64] & 0xf) | ((x[ib].qh[4*k+i] << 3) & 0x10)];
                y16[k][QK8_0*ib+i+20] = dl * values[(x[ib].qs[4*k+i+ 64] >>  4) | ((x[ib].qh[4*k+i] >> 1) & 0x10)];
                y16[k][QK8_0*ib+i+ 8] = dl * values[(x[ib].qs[4*k+i+128] & 0xf) | ((x[ib].qh[4*k+i] << 2) & 0x10)];
                y16[k][QK8_0*ib+i+24] = dl * values[(x[ib].qs[4*k+i+128] >>  4) | ((x[ib].qh[4*k+i] >> 2) & 0x10)];
                y16[k][QK8_0*ib+i+12] = dl * values[(x[ib].qs[4*k+i+192] & 0xf) | ((x[ib].qh[4*k+i] << 1) & 0x10)];
                y16[k][QK8_0*ib+i+28] = dl * values[(x[ib].qs[4*k+i+192] >>  4) | ((x[ib].qh[4*k+i] >> 3) & 0x10)];
            }
        }
    }
}

void vec_dot_iq5_ks_r16_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_KS_R16, vx, 0, GGML_TYPE_Q8_2_X4, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}


void quantize_row_iq2_bn_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_bn_r4(x, y, 4, k/4, nullptr, nullptr);
}




void vec_dot_iq2_bn_r4_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_BN_R4, vx, 0, GGML_TYPE_Q8_K64, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= q4_k_r4
//


void quantize_row_q4_k_r4(const float * x, void * y, int64_t k) {
    quantize_q4_k_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_q4_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q4_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= q6_k_r4
//


void quantize_row_q6_k_r4(const float * x, void * y, int64_t k) {
    quantize_q6_k_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_q6_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q6_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


//
// ========================================= q5_k_r4
//


void quantize_row_q5_k_r4(const float * x, void * y, int64_t k) {
    quantize_q5_k_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_q5_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q5_K_R4, vx, 0, GGML_TYPE_Q8_K32, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= q3_k_r4
//


void quantize_row_q3_k_r4(const float * x, void * y, int64_t k) {
    quantize_q3_k_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_q3_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q3_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= q2_k_r4
//


void quantize_row_q2_k_r4(const float * x, void * y, int64_t k) {
    quantize_q2_k_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_q2_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q2_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq4_k_r4
//

void quantize_row_iq4_k_r4_ref(const float * x, block_iq4_k_r4 * y, int64_t k) {
    quantize_iq4_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}

void quantize_row_iq4_k_r4(const float * x, void * y, int64_t k) {
    quantize_iq4_k_r4(x, y, 4, k/4, nullptr, nullptr);
}

static void repack_iq4_k(int nrows, int n_per_row, const block_iq4_k * x, block_iq4_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq4_k * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].extra, 0, 8);
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto extra = x4[k][ibl].extra;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    if (extra & 1) y[ibl].extra[k+0] |= (1 << ib);
                    if (extra & 2) y[ibl].extra[k+4] |= (1 << ib);
                    extra >>= 2;
                    uint8_t sl1 = x4[k][ibl].scales_l[ib] & 0xf;
                    uint8_t sl2 = x4[k][ibl].scales_l[ib] >>  4;
                    uint8_t sh  = x4[k][ibl].scales_h[ib/2] >> 4*(ib%2);
                    uint8_t sh1 = (sh >> 0) & 3;
                    uint8_t sh2 = (sh >> 2) & 3;
                    int i = 8*ib + k;
                    y[ibl].scales_l[i%32] |= (sl1 << 4*(i/32));
                    y[ibl].scales_h[i%16] |= (sh1 << 2*(i/16));
                    i += 4;
                    y[ibl].scales_l[i%32] |= (sl2 << 4*(i/32));
                    y[ibl].scales_h[i%16] |= (sh2 << 2*(i/16));
                }
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 4; ++k) for (int i = 0; i < 4; ++i) {
                    y[ibl].qs[64*ib+4*k+i+ 0] = (x4[k][ibl].qs[16*ib+i+0] & 0xf) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0x0f) << 4);  //  0....3 +  8...11 from each row
                    y[ibl].qs[64*ib+4*k+i+16] = (x4[k][ibl].qs[16*ib+i+0] >>  4) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0xf0));       // 16...19 + 24...27 from each row
                    y[ibl].qs[64*ib+4*k+i+32] = (x4[k][ibl].qs[16*ib+i+4] & 0xf) | ((x4[k][ibl].qs[16*ib+i+12] & 0x0f) << 4);  //  4....7 + 12...15 from each row
                    y[ibl].qs[64*ib+4*k+i+48] = (x4[k][ibl].qs[16*ib+i+4] >>  4) | ((x4[k][ibl].qs[16*ib+i+12] & 0xf0));       // 20...23 + 28...31 from each row
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq4_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq4_k(src, (void *)qtmp.data(), 4, n_per_row, imatrix, user_data);
        repack_iq4_k(4, n_per_row, (const block_iq4_k *)qtmp.data(), (block_iq4_k_r4 *)qcur, false);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}


void vec_dot_iq4_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq5_k_r4
//


void quantize_row_iq5_k_r4(const float * x, void * y, int64_t k) {
    quantize_iq5_k_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_iq5_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq5_ks_r4
//


void quantize_row_iq5_ks_r4(const float * x, void * y, int64_t k) {
    quantize_iq5_ks_r4(x, y, 4, k/4, nullptr, nullptr);
}




void vec_dot_iq5_ks_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_KS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= q8_k_r8
//

void quantize_row_q8_k_r8_ref(const float * x, block_q8_k_r8 * y, int64_t k) {
    quantize_q8_k_r8(x, (void *)y, 8, k/8, nullptr, nullptr);
}

void quantize_row_q8_k_r8(const float * x, void * y, int64_t k) {
    quantize_q8_k_r8(x, y, 8, k/8, nullptr, nullptr);
}

static void repack_q8_k(int nrows, int n_per_row, const block_q8_K * x, block_q8_k_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q8_K * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 8; ++k) {
                y[ibl].d[k] = GGML_FP32_TO_FP16(x8[k][ibl].d);
                for (int ib = 0; ib < QK_K/4; ++ib) {
                    for (int i = 0; i < 4; ++i) y[ibl].qs[32*ib + 4*k + i] = x8[k][ibl].qs[4*ib+i];
                }
            }
#ifdef HAVE_FANCY_SIMD
            if (online) {
                for (int l = 0; l < 32; ++l) {
                    auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[ibl].qs + l), _mm512_set1_epi8(-128));
                    _mm512_storeu_si512((__m512i *)y[ibl].qs + l, v);
                }
            }
#endif
        }
        x += 8*nblock;
        y += nblock;
    }
}
#ifdef HAVE_FANCY_SIMD
static void modify_q8_k_r8(int64_t k, char * cy) {
    auto y = (block_q8_k_r8 *)cy;
    int nb = k/(256*8);
    for (int ib = 0; ib < nb; ++ib) {
        for (int l = 0; l < 32; ++l) {
            auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[ib].qs + l), _mm512_set1_epi8(-128));
            _mm512_storeu_si512((__m512i *)y[ib].qs + l, v);
        }
    }
}
#endif

size_t quantize_q8_k_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_K, n_per_row);
    auto row_size_1 = ggml_row_size(GGML_TYPE_Q8_K_R8, n_per_row);
    std::vector<char> qtmp(8*row_size_0);
    for (int row = 0; row < nrows; row += 8) {
        quantize_row_q8_K32(src, (void *)qtmp.data(), 8*n_per_row);
        repack_q8_k(8, n_per_row, (const block_q8_K *)qtmp.data(), (block_q8_k_r8 *)qcur, false);
        qcur += 8*row_size_1;
        src += 8*n_per_row;
    }
    return nrows*row_size_1;
}


void vec_dot_q8_k_r8_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_K_R8, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= q8_k_r16
//

void quantize_row_q8_k_r16_ref(const float * x, block_q8_k_r16 * y, int64_t k) {
    quantize_q8_k_r16(x, (void *)y, 16, k/16, nullptr, nullptr);
}

void quantize_row_q8_k_r16(const float * x, void * y, int64_t k) {
    quantize_q8_k_r16(x, y, 16, k/16, nullptr, nullptr);
}

static void repack_q16_k(int nrows, int n_per_row, const block_q8_K * x, block_q8_k_r16 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q8_K * x16[16];
    for (int row = 0; row < nrows; row += 16) {
        for (int k = 0; k < 16; ++k) x16[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 16; ++k) {
                y[ibl].d[k] = x16[k][ibl].d;
                for (int ib = 0; ib < QK_K/4; ++ib) {
                    for (int i = 0; i < 4; ++i) y[ibl].qs[64*ib + 4*k + i] = x16[k][ibl].qs[4*ib+i];
                }
            }
            // block_q8_k_r16.qs is DECLARED unsigned (ggml-common.h: "stored as unsigned ints"): the only GEMM kernel
            // (mul_mat_q8_k_r16_q8_k, AVX-512 dpbusd) and dequantize_row_q8_k_r16 both read an unsigned byte holding
            // value+128 and subtract the 128 back out. The flip used
            // to exist only under HAVE_FANCY_SIMD, so an AVX2/NEON quantize_q8_k_r16 stored SIGNED bytes into the
            // unsigned format: to_float decoded every value wrong on those hosts (rel RMS 1.89 vs the source,
            // test-rowint-dequant-bounds, 2026-09-28) and the file's bytes depended on the quantizing host.
            // Same bytes on every host now; the AVX-512 result is unchanged.
#ifdef HAVE_FANCY_SIMD
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[ibl].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[ibl].qs + l, v);
            }
#else
            for (int l = 0; l < 16*QK_K; ++l) {
                y[ibl].qs[l] = (int8_t)((uint8_t)y[ibl].qs[l] ^ 0x80);
            }
#endif
        }
        x += 16*nblock;
        y += nblock;
    }
}

size_t quantize_q8_k_r16(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_K, n_per_row);
    auto row_size_1 = ggml_row_size(GGML_TYPE_Q8_K_R16, n_per_row);
    std::vector<char> qtmp(16*row_size_0);
    for (int row = 0; row < nrows; row += 16) {
        quantize_row_q8_K32(src, (void *)qtmp.data(), 16*n_per_row);
        repack_q16_k(16, n_per_row, (const block_q8_K *)qtmp.data(), (block_q8_k_r16 *)qcur, false);
        qcur += 16*row_size_1;
        src += 16*n_per_row;
    }
    return nrows*row_size_1;
}


void vec_dot_q8_k_r16_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_K_R16, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= q8_KV_r8
//

void quantize_row_q8_KV_r8_ref(const float * x, void * y, int64_t k) {
    quantize_q8_KV_r8(x, y, 8, k/8, nullptr, nullptr);
}

void quantize_row_q8_KV_r8(const float * x, void * y, int64_t k) {
    quantize_q8_KV_r8(x, y, 8, k/8, nullptr, nullptr);
}

static void repack_q8_KV(int nrows, int n_per_row, const char * cx, char * cy, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%16 == 0);
    auto row_size_x = ggml_row_size(GGML_TYPE_Q8_KV,    n_per_row);
    auto row_size_y = ggml_row_size(GGML_TYPE_Q8_KV_R8, n_per_row);
    const int8_t * x8[8];
#ifdef __ARM_NEON
    int8x16x2_t m0, m1, m2, m3;
#endif
    for (int row = 0; row < nrows; row += 8) {
        auto dy = (float *)cy;
        auto qy = (int8_t *)(dy + 8);
        for (int k = 0; k < 8; ++k) {
            auto dx = (const float *)(cx + k*row_size_x);
            dy[k] = dx[0];
            x8[k] = (const int8_t *)(dx + 2);
        }
        for (int ib = 0; ib < n_per_row/16; ++ib) {
#ifdef __AVX2__
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
            auto m0 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[4]+ib), _mm_loadu_si128((const __m128i *)x8[0]+ib));
            auto m1 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[5]+ib), _mm_loadu_si128((const __m128i *)x8[1]+ib));
            auto m2 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[6]+ib), _mm_loadu_si128((const __m128i *)x8[2]+ib));
            auto m3 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[7]+ib), _mm_loadu_si128((const __m128i *)x8[3]+ib));
            auto t0 = _mm256_unpacklo_epi32(m0, m1);
            auto t1 = _mm256_unpacklo_epi32(m2, m3);
            auto t2 = _mm256_unpackhi_epi32(m0, m1);
            auto t3 = _mm256_unpackhi_epi32(m2, m3);
            m0 = _mm256_unpacklo_epi64(t0, t1);
            m1 = _mm256_unpackhi_epi64(t0, t1);
            m2 = _mm256_unpacklo_epi64(t2, t3);
            m3 = _mm256_unpackhi_epi64(t2, t3);
#ifdef HAVE_FANCY_SIMD
            if (online) {
                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
            }
#endif
            _mm256_storeu_si256((__m256i *)qy + 4*ib+0, m0);
            _mm256_storeu_si256((__m256i *)qy + 4*ib+1, m1);
            _mm256_storeu_si256((__m256i *)qy + 4*ib+2, m2);
            _mm256_storeu_si256((__m256i *)qy + 4*ib+3, m3);
#elif defined __ARM_NEON
            m0.val[0] = vld1q_s8(x8[0]+16*ib); m0.val[1] = vld1q_s8(x8[4]+16*ib);
            m1.val[0] = vld1q_s8(x8[1]+16*ib); m1.val[1] = vld1q_s8(x8[5]+16*ib);
            m2.val[0] = vld1q_s8(x8[2]+16*ib); m2.val[1] = vld1q_s8(x8[6]+16*ib);
            m3.val[0] = vld1q_s8(x8[3]+16*ib); m3.val[1] = vld1q_s8(x8[7]+16*ib);
            auto row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[0]), vreinterpretq_s32_s8(m1.val[0]));
            auto row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[0]), vreinterpretq_s32_s8(m3.val[0]));
            m0.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
            m1.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
            m2.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
            m3.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
            row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[1]), vreinterpretq_s32_s8(m1.val[1]));
            row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[1]), vreinterpretq_s32_s8(m3.val[1]));
            m0.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
            m1.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
            m2.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
            m3.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
            vst1q_s8_x2(qy +  0 + 128*ib, m0);
            vst1q_s8_x2(qy + 32 + 128*ib, m1);
            vst1q_s8_x2(qy + 64 + 128*ib, m2);
            vst1q_s8_x2(qy + 96 + 128*ib, m3);
#else
            // Generic path. THIS ARM HAD NEVER BEEN COMPILED BY ANYTHING: it was marked "// TODO"
            // and read `y[ib].qs[...] = x8[k][ib].qs[...]`, naming a `y` this function does not
            // declare and a `.qs` member that `const int8_t *` does not have. It was written
            // against an earlier block-struct data model; the SIMD arms were later rewritten to
            // raw pointers and this one was left behind, invisible because every build of this
            // tree defines __AVX2__ and so never selects it. Found 2026-09-15 by compiling this
            // file without /arch:AVX2 (error C2065: 'y': undeclared identifier).
            //
            // The layout is an 8-row interleave of 4-byte words, and it is NOT re-derived by
            // reading: both SIMD arms above independently produce 128 bytes per ib at
            // qy + 128*ib + 32*j, holding word j of all 8 rows in row order. The AVX2 chain
            // (unpacklo/hi_epi32 then unpacklo/hi_epi64, each acting per 128-bit lane) and the
            // NEON chain (vtrnq_s32 then vtrn1q/vtrn2q_s64) agree.
            //
            // VERIFIED, not asserted: temp/_test_repack_q8kv_scalar.cpp runs this exact loop and
            // the verbatim AVX2 arm over the same pseudo-random input and requires byte-identical
            // output -- 6 sizes (16, 32, 64, 128, 256, 4096), all identical, with a deliberately
            // transposed variant caught at every size as the harness's own control. A quant repack
            // that is 99% right corrupts data silently rather than failing, so agreement with the
            // SIMD path is the only acceptable evidence here.
            //
            // The `online` +127 adjustment above is guarded by HAVE_FANCY_SIMD, which implies
            // AVX-512 and therefore __AVX2__, so this arm is unreachable when it applies and
            // deliberately does not repeat it.
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 8; ++k) for (int i = 0; i < 4; ++i) {
                    qy[128*ib + 32*j + 4*k + i] = x8[k][16*ib + 4*j + i];
                }
            }
#endif

        }
        cx += 8*row_size_x;
        cy += online ? 8*row_size_x : 8*row_size_y;
        //So, if we are run-time-repacking (online = true) we don't want to change the stride, so we just leave some unused space at the end of each row
    }
}
#ifdef HAVE_FANCY_SIMD
static void modify_q8_KV_r8(int64_t k, char * cy) {
    int8_t * q8 = (int8_t *)(cy + 8*sizeof(float));
    for (int j = 0; j < k; ++j) q8[j] += 127;
}
#endif

size_t quantize_q8_KV_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%16 == 0);
    char * qcur = (char *)dst;
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_KV, n_per_row);
    auto row_size_1 = ggml_row_size(GGML_TYPE_Q8_KV_R8, n_per_row);
    std::vector<char> qtmp(8*row_size_0);
    for (int row = 0; row < nrows; row += 8) {
        quantize_q8_KV(src, (void *)qtmp.data(), 8, n_per_row, imatrix, user_data);
        repack_q8_KV(8, n_per_row, qtmp.data(), qcur, false);
        qcur += 8*row_size_1;
        src += 8*n_per_row;
    }
    return nrows*row_size_1;
}


void vec_dot_q8_KV_r8_q8_KV(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_KV_R8, vx, 0, GGML_TYPE_Q8_KV, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= bf16_r4
//
namespace {
// Round-to-nearest-even, NaN kept quiet -- the same conversion as ggml_fp32_to_bf16_row_ref, which plain
// bf16 weights use. This TRUNCATED (u >> 16), so f32 -> bf16_r16 carried ~2x the rounding error of f32 -> bf16:
// measured through ggml_mul_mat vs exact f32, bf16 1.51e-03 vs bf16_r16 3.08e-03 (fork 2026-09-26).
inline ggml_bf16_t to_bf16(const float& x) {
    union { float f; uint32_t u; } helper;
    helper.f = x;
    if ((helper.u & 0x7fffffff) > 0x7f800000) {                  // NaN: keep it a NaN
        return ggml_bf16_t{(uint16_t)((helper.u >> 16) | 64)};
    }
    return ggml_bf16_t{(uint16_t)((helper.u + (0x7fff + ((helper.u >> 16) & 1))) >> 16)};
}
inline ggml_bf16_t to_bf16(const ggml_half& x) { return to_bf16(GGML_FP16_TO_FP32(x)); }
inline ggml_bf16_t to_bf16(const ggml_bf16_t& x) { return x; }
template <typename T>
void repack_bf16(int nrows, int n_per_row, const T * x, ggml_bf16_t * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%2 == 0);
    for (int row = 0; row < nrows; row += 16) {
        for (int k = 0; k < 16; ++k) {
            auto x8 = x + k*n_per_row;
            for (int ib = 0; ib < n_per_row/2; ++ib) {
                y[32*ib + 2*k + 0] = to_bf16(x8[2*ib+0]);
                y[32*ib + 2*k + 1] = to_bf16(x8[2*ib+1]);
            }
        }
        x += 16*n_per_row;
        y += 16*n_per_row;
    }
}
}

void repack_f32_bf16_r16(const void * src, void * dst, int64_t nrows, int64_t n_per_row) {
    repack_bf16(nrows, n_per_row, (const float *)src, (ggml_bf16_t *)dst, false);
}

void repack_bf16_bf16_r16(const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row) {
    repack_bf16(nrows, n_per_row, (const ggml_bf16_t *)src, (ggml_bf16_t *)dst, false);
}

//
// ========================================= iq3_k_r4
//


void quantize_row_iq3_k_r4(const float * x, void * y, int64_t k) {
    quantize_iq3_k_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_iq3_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq2_k_r4
//


void quantize_row_iq2_k_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_k_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_iq2_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


//
// ========================================= iq2_xxs_r4
//


void quantize_row_iq2_xxs_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_xxs_r4(x, y, 4, k/4, nullptr, nullptr);
}




void vec_dot_iq2_xxs_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_XXS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq2_xs_r4
//


void quantize_row_iq2_xs_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_xs_r4(x, y, 4, k/4, nullptr, nullptr);
}




void vec_dot_iq2_xs_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_XS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq2_s_r4
//


void quantize_row_iq2_s_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_s_r4(x, y, 4, k/4, nullptr, nullptr);
}




void vec_dot_iq2_s_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_S_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq3_xxs_r4
//


void quantize_row_iq3_xxs_r4(const float * x, void * y, int64_t k) {
    quantize_iq3_xxs_r4(x, y, 4, k/4, nullptr, nullptr);
}





void vec_dot_iq3_xxs_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_XXS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

//
// ========================================= iq3_s_r4
//


void quantize_row_iq3_s_r4(const float * x, void * y, int64_t k) {
    quantize_iq3_s_r4(x, y, 4, k/4, nullptr, nullptr);
}




void vec_dot_iq3_s_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_S_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


void quantize_row_iq1_s_r4(const float * x, void * y, int64_t k) {
    quantize_iq1_s_r4(x, y, 4, k/4, nullptr, nullptr);
}



void vec_dot_iq1_s_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ1_S_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}


void quantize_row_iq1_m_r4(const float * x, void * y, int64_t k) {
    quantize_iq1_m_r4(x, y, 4, k/4, nullptr, nullptr);
}



void vec_dot_iq1_m_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ1_M_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

void quantize_row_q8_KV(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_KV(x, vy, k);
}

void quantize_row_q8_KV_ref(const float * x, void * y, int64_t k) {
    quantize_row_q8_KV(x, y, k);
}

size_t quantize_q8_KV(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    (void)imatrix;
    auto row_size = ggml_row_size(GGML_TYPE_Q8_KV, n_per_row);
    auto q = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        quantize_row_q8_KV(src, q, n_per_row);
        src += n_per_row;
        q += row_size;
    }
    return row_size*nrows;
}


void vec_dot_q8_KV_q8_KV(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    // SCALAR, not iqk_mul_mat(1, 1, ...). A vec_dot is ONE row by contract, and the x86
    // mul_mat_q8_KV_q8_KV kernel GGML_ASSERTs nrc_x%4 == 0 (iqk_gemm_kquants.cpp), so the old
    // iqk-first call aborted the process on EVERY invocation -- test-quantize-fns died here and
    // never reached any type numbered after q8_KV. Row layout (dequantize_row_q8_KV above):
    // float d, one more float of row meta, then n int8. The multi-row matmul still goes
    // through iqk_mul_mat directly; this only covers the single-row entry point.
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    const float  * dx = (const float *)vx;
    const float  * dy = (const float *)vy;
    const int8_t * qx = (const int8_t *)(dx + 2);
    const int8_t * qy = (const int8_t *)(dy + 2);
    int64_t isum = 0;
    for (int j = 0; j < n; ++j) isum += (int32_t)qx[j] * (int32_t)qy[j];
    *s = dx[0] * dy[0] * (float)isum;
}


//================================================

namespace {
struct Repack {
    using repack_func = void (*) (int nrows, int n_per_row, const char * src, char * dst, bool online);
    ggml_type   new_type;
    int         num_rows;
    repack_func repack;
};
struct Modify {
    using modify_func_t = void (*)(int64_t k, char * src_dst);
    modify_func_t  mod_func;
    int            nrows;
};
const Modify * get_modify_info(ggml_type type) {
    static const std::unordered_map<ggml_type, Modify> k_mod_map = {
#ifdef __ARM_NEON
        { GGML_TYPE_Q4_0_R8, {modify_q4_0_r8, 8} },
#endif
#ifdef HAVE_FANCY_SIMD
        { GGML_TYPE_Q8_0_R8,  {modify_q8_0_r8,  8} },
        { GGML_TYPE_Q8_K_R8,  {modify_q8_k_r8,  8} },
        { GGML_TYPE_Q8_KV_R8, {modify_q8_KV_r8, 8} },
#endif
    };
    auto it = k_mod_map.find(type);
    return it != k_mod_map.end() ? &it->second : nullptr;
}
// Tensors a row-interleaved layout cannot serve: the token embeddings (read by get_rows) by exact name, and the DeepSeek-V4
// engram tensors blk.N.engram_{embd,k,q}.weight (get_rows / as_f32 in src/graphs/build_deepseek4.cpp) by suffix.
// Same rule as ik_llama 11b54e80f (#2612); tests/test-iqk-repack-forbidden.cpp.
bool is_forbidden_tensor(const std::string& name) {
    if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") return true;
    static const char * k_suffixes[] = { ".engram_embd.weight", ".engram_k.weight", ".engram_q.weight" };
    for (const char * s : k_suffixes) {
        const size_t n = std::strlen(s);
        if (name.size() >= n && name.compare(name.size() - n, n, s) == 0) return true;
    }
    return false;
}
}

bool iqk_should_modify_tensor([[maybe_unused]] const struct ggml_tensor * tensor) {
    return false;
    //if (is_forbidden_tensor(tensor->name)) return false;
    //auto mptr = get_modify_info(tensor->type);
    //return mptr ? true : false;
}

bool iqk_modify_tensor(struct ggml_tensor * tensor) {
    return false;
    auto mptr = get_modify_info(tensor->type);
    if (!mptr) return false;
    if (is_forbidden_tensor(std::string{tensor->name})) return false;

    auto& m = *mptr;
    int nrows = ggml_nrows(tensor);
    int nchunks = nrows/m.nrows;
    int max_thread = std::max(1, int(std::thread::hardware_concurrency()/2));
    int nthread = std::min(nchunks, max_thread);
    auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    std::atomic<int> counter(0);
    auto compute = [&counter, &m, tensor, row_size, nchunks] () {
        int64_t n_per_call = m.nrows*tensor->ne[0];
        while (true) {
            int row = counter.fetch_add(1);
            if (row >= nchunks) break;
            m.mod_func(n_per_call, (char *)tensor->data + row_size*row*m.nrows);
        }
    };
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();

    return true;
}

namespace {
const Repack * get_repack_info(ggml_type type) {
    static const std::unordered_map<ggml_type, Repack> k_map = {
        { GGML_TYPE_IQ2_K,  { GGML_TYPE_IQ2_K_R4,  4,  (Repack::repack_func)repack_iq2_k}   },
        { GGML_TYPE_IQ3_K,  { GGML_TYPE_IQ3_K_R4,  4,  (Repack::repack_func)repack_iq3_k}   },
        { GGML_TYPE_IQ4_K,  { GGML_TYPE_IQ4_K_R4,  4,  (Repack::repack_func)repack_iq4_k}   },
        { GGML_TYPE_IQ5_K,  { GGML_TYPE_IQ5_K_R4,  4,  (Repack::repack_func)repack_iq5_k}   },
        { GGML_TYPE_IQ4_XS, { GGML_TYPE_IQ4_XS_R8, 8,  (Repack::repack_func)repack_iq4_xs}  },
        { GGML_TYPE_IQ4_KS, { GGML_TYPE_IQ4_KS_R4, 4,  (Repack::repack_func)repack_iq4_ks}  },
        { GGML_TYPE_IQ5_KS, { GGML_TYPE_IQ5_KS_R4, 4,  (Repack::repack_func)repack_iq5_ks}  },
        { GGML_TYPE_IQ4_NL, { GGML_TYPE_IQ4_NL_R4, 4,  (Repack::repack_func)repack_iq4_nl}  },
        { GGML_TYPE_IQ2_BN, { GGML_TYPE_IQ2_BN_R4, 4,  (Repack::repack_func)repack_iq2_bn}  },
        { GGML_TYPE_IQ2_XXS,{ GGML_TYPE_IQ2_XXS_R4,4,  (Repack::repack_func)repack_iq2_xxs} },
        { GGML_TYPE_IQ2_XS, { GGML_TYPE_IQ2_XS_R4, 4,  (Repack::repack_func)repack_iq2_xs}  },
        { GGML_TYPE_IQ2_S,  { GGML_TYPE_IQ2_S_R4,  4,  (Repack::repack_func)repack_iq2_s}   },
        { GGML_TYPE_IQ3_XXS,{ GGML_TYPE_IQ3_XXS_R4,4,  (Repack::repack_func)repack_iq3_xxs} },
        { GGML_TYPE_IQ3_S,  { GGML_TYPE_IQ3_S_R4,  4,  (Repack::repack_func)repack_iq3_s}   },
        { GGML_TYPE_Q2_K,   { GGML_TYPE_Q2_K_R4,   4,  (Repack::repack_func)repack_q2_k}    },
        { GGML_TYPE_Q3_K,   { GGML_TYPE_Q3_K_R4,   4,  (Repack::repack_func)repack_q3_k}    },
        { GGML_TYPE_Q4_K,   { GGML_TYPE_Q4_K_R4,   4,  (Repack::repack_func)repack_q4_k}    },
        { GGML_TYPE_Q5_K,   { GGML_TYPE_Q5_K_R4,   4,  (Repack::repack_func)repack_q5_k}    },
        { GGML_TYPE_Q6_K,   { GGML_TYPE_Q6_K_R4,   4,  (Repack::repack_func)repack_q6_k}    },
        { GGML_TYPE_Q4_0,   { GGML_TYPE_Q4_0_R8,   8,  (Repack::repack_func)repack_q4_0}    },
        { GGML_TYPE_Q2_0,   { GGML_TYPE_Q2_0_R8,   8,  (Repack::repack_func)repack_q2_0}    },
        { GGML_TYPE_Q5_0,   { GGML_TYPE_Q5_0_R4,   4,  (Repack::repack_func)repack_q5_0}    },
        { GGML_TYPE_Q6_0,   { GGML_TYPE_Q6_0_R4,   4,  (Repack::repack_func)repack_q6_0}    },
        { GGML_TYPE_Q8_0,   { GGML_TYPE_Q8_0_R8,   8,  (Repack::repack_func)repack_q8_0}    },
        { GGML_TYPE_Q8_K,   { GGML_TYPE_Q8_K_R8,   8,  (Repack::repack_func)repack_q8_k}    },
        { GGML_TYPE_Q8_KV,  { GGML_TYPE_Q8_KV_R8,  8,  (Repack::repack_func)repack_q8_KV}   },
        { GGML_TYPE_MXFP4,  { GGML_TYPE_MXFP4_R8,  8,  (Repack::repack_func)repack_mxfp4}   },
        { GGML_TYPE_Q1_0_G128, { GGML_TYPE_Q1_0_G128_R8, QK1_0_G128_R8_ROWS, (Repack::repack_func)repack_q1_0_g128_r8} },
        { GGML_TYPE_PQ2_0,     { GGML_TYPE_PQ2_0_R8,     QK_PQ2_0_R8_ROWS,     (Repack::repack_func)repack_pq2_0_r8} },
        { GGML_TYPE_PTQ1_0,    { GGML_TYPE_PTQ1_0_R8,    QK_PTQ1_0_R8_ROWS,    (Repack::repack_func)repack_ptq1_0_r8} },
#ifdef __AVX512BF16__
        { GGML_TYPE_BF16,   { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_bf16_t>}},
        { GGML_TYPE_F16,    { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_half>}  },
#endif
    };
    auto it = k_map.find(type);
    return it != k_map.end() ? &it->second : nullptr;
}
}

int iqk_repacked_type(const struct ggml_tensor * tensor) {
    if (!ggml_is_contiguous(tensor)) return (int)tensor->type;
    if (is_forbidden_tensor(tensor->name)) return (int)tensor->type;
    auto rptr = get_repack_info(tensor->type);
    return rptr && tensor->ne[1] % rptr->num_rows == 0 ? (int)rptr->new_type : (int)tensor->type;
}

void iqk_repack_tensor(struct ggml_tensor * tensor) {
    constexpr int kChunk = 8;
    if (!tensor) return;
    if (!ggml_is_contiguous(tensor)) return;
    if (is_forbidden_tensor(tensor->name)) return;
    if (tensor->ne[1] % 4) return;

    auto rptr = get_repack_info(tensor->type);
    if (!rptr) return;
    if (tensor->ne[1] % rptr->num_rows) return;

    auto& r = *rptr;

    auto nrows = ggml_nrows(tensor);

    int max_thread = std::max(1, int(std::thread::hardware_concurrency()/2));
    int num_chunks = (nrows + kChunk*r.num_rows - 1)/(kChunk*r.num_rows);
    int nthread = std::min(num_chunks, max_thread);

    //printf("%s(%s): %s -> %s. %d rows, %d chunks, %d threads\n", __func__, tensor->name, ggml_type_name(tensor->type), ggml_type_name(r.new_type),
    //        int(tensor->ne[1]), num_chunks, nthread);

    std::atomic<int> counter(0);;
    auto compute = [&counter, &r, tensor, num_chunks, chunkSize = kChunk] () {
        int nrows = ggml_nrows(tensor);
        int n_per_row = tensor->ne[0];
        auto row_size = ggml_row_size(tensor->type, n_per_row);
        std::vector<char> qtmp(r.num_rows*row_size);
        auto data = (char *)tensor->data;
        while (true) {
            int chunk = counter.fetch_add(1);
            if (chunk >= num_chunks) break;
            int first_row = chunk*chunkSize*r.num_rows;
            int last_row = std::min(first_row + chunkSize*r.num_rows, nrows);
            for (int row = first_row; row < last_row; row += r.num_rows) {
                std::memcpy(qtmp.data(), data + row*row_size, r.num_rows*row_size);
                //r.repack(r.num_rows, n_per_row, qtmp.data(), data + row*row_size, true);
                r.repack(r.num_rows, n_per_row, qtmp.data(), data + row*row_size, false);
            }
        }
    };
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();

    tensor->type = r.new_type;
}


namespace {


template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::pair<float, float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_scale(
        const float * xb, const float * weight, const int * best_idx) const {
    float sumqx = 0, sumq2 = 0;
#ifdef __AVX2__
    auto vqx = _mm256_setzero_ps();
    auto vq2 = _mm256_setzero_ps();
    for (int l = 0; l < kBlockSize; l += 8) {
        auto vx = _mm256_loadu_ps(xb+l);
        auto vw = _mm256_loadu_ps(weight+l);
        auto vq = kGroupSize == 8 ? _mm256_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize]) :
            _mm256_set_m128(_mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+1]),
                            _mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+0]));
        auto vqw = _mm256_mul_ps(vq, vw);
        vqx = _mm256_fmadd_ps(vqw, vx, vqx);
        vq2 = _mm256_fmadd_ps(vqw, vq, vq2);
    }
    sumqx = hsum_float_8(vqx);
    sumq2 = hsum_float_8(vq2);
#else
    for (int l = 0; l < kNg; ++l) {
        auto xl = xb + kGroupSize*l;
        auto wl = weight + kGroupSize*l;
        auto ql = m_values.data() + kGroupSize*best_idx[l];
        for (int k = 0; k < kGroupSize; ++k) {
            sumqx += wl[k]*ql[k]*xl[k];
            sumq2 += wl[k]*ql[k]*ql[k];
        }
    }
#endif
    return sumq2 > 0 ? std::make_pair(sumqx/sumq2, sumqx*sumqx/sumq2) : std::make_pair(0.f, 0.f);
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
float QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_inverse_scale(
        const float * xb, const float * weight, const int * best_idx) const {
    float sumqx = 0, sumx2 = 0;
#ifdef __AVX2__
    auto vqx = _mm256_setzero_ps();
    auto vx2 = _mm256_setzero_ps();
    for (int l = 0; l < kBlockSize; l += 8) {
        auto vx = _mm256_loadu_ps(xb+l);
        auto vw = _mm256_loadu_ps(weight+l);
        auto vq = kGroupSize == 8 ? _mm256_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize]) :
            _mm256_set_m128(_mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+1]),
                            _mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+0]));
        auto vxw = _mm256_mul_ps(vx, vw);
        vx2 = _mm256_fmadd_ps(vxw, vx, vx2);
        vqx = _mm256_fmadd_ps(vxw, vq, vqx);
    }
    sumqx = hsum_float_8(vqx);
    sumx2 = hsum_float_8(vx2);
#else
    for (int l = 0; l < kNg; ++l) {
        auto xl = xb + kGroupSize*l;
        auto wl = weight + kGroupSize*l;
        auto ql = m_values.data() + kGroupSize*best_idx[l];
        for (int k = 0; k < kGroupSize; ++k) {
            sumqx += wl[k]*ql[k]*xl[k];
            sumx2 += wl[k]*xl[k]*xl[k];
        }
    }
#endif
    return sumx2 > 0 ? sumqx/sumx2 : 0.f;
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d,
        [[maybe_unused]] const float * xb, [[maybe_unused]] const float * weight, int * best_idx) const {
    if (!d) {
        std::memset(best_idx, 0, kNg*sizeof(int));
        return;
    }
    [[maybe_unused]] int ncluster = m_clusters.size()/kGroupSize;
    [[maybe_unused]] float id = 1/d;
#ifdef __AVX2__
    if constexpr (kGroupSize == 8) {
        __m256 sqx[8];
        const __m256i add_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        float sx[8];
        int   index[8];
        auto vid = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + 8*l;
            auto wl = weight + 8*l;
            auto vx = _mm256_mul_ps(vid, _mm256_loadu_ps(xl));
            auto vw = _mm256_loadu_ps(wl);
            int jbest = -1;
            if (kGroupSize == 8 && (ncluster == 256 || ncluster == 6561)) {
                _mm256_store_ps(sx, vx);
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int j = 0; j < 8; ++j) if (sx[j] > m_mid[j]) u |= (1 << j);
                } else {
                    int s = 1;
                    for (int j = 0; j < 8; ++j) { u += s*bin3(j, sx[j]); s *= 3; }
                }
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 8; ++i) {
                        auto vq = _mm256_loadu_ps(m_clusters.data() + kGroupSize*(j+i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                    }
                    auto score = hsum_float_8x8(sqx);
                    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                            _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i *)index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                }
            }
            auto& points = m_in_cluster[jbest];
            auto& values = points.empty() ? m_values : m_c_values[jbest];
            int npoint = values.size()/kGroupSize;
            GGML_ASSERT(npoint > 0 && npoint%8 == 0);
            int jbest_cluster = jbest;
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            auto best = INFINITY; jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < npoint; j += 8) {
                for (int i = 0; i < 8; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize*(j+i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                }
                auto score = hsum_float_8x8(sqx);
                auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                        _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i *)index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
            }
            if (jbest < 0) {
                fprintf(stderr, "Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                GGML_ASSERT(false);
            }
            best_idx[l] = points.empty() ? jbest : points[jbest];
        }
    } else {
        __m256 sqx[4];
        const __m256i add_idx = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
        const __m256 sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        float sx[8];
        int   index[8];
        auto vid_p = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + 4*l;
            auto wl = weight + 4*l;
            auto vx4 = _mm_loadu_ps(xl);
            auto vx = _mm256_mul_ps(vid_p, _mm256_set_m128(vx4, vx4));
            auto vw4 = _mm_loadu_ps(wl);
            auto vw = _mm256_set_m128(vw4, vw4);
            int jbest = -1;
            if (ncluster == 256 || ncluster == 625) {
                _mm256_storeu_ps(sx, vx);
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < 4; ++k) u |= (bin4(sx[k]) << 2*k);
                } else {
                    int l = 1;
                    for (int k = 0; k < 4; ++k) { u += bin5(sx[k])*l; l *= 5; }
                }
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 4; ++i) {
                        auto vq = _mm256_loadu_ps(m_clusters.data() + kGroupSize*(j+2*i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        vdiff = _mm256_and_ps(sign_bit, vdiff);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, _mm256_mul_ps(vdiff, vdiff)));
                    }
                    auto score = hsum_float_4x8(sqx);
                    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                            _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i *)index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                }
            }
            auto& points = m_in_cluster[jbest];
            auto& values = m_c_values[jbest];
            GGML_ASSERT(!points.empty() && points.size()%8 == 0);
            int jbest_cluster = jbest;
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            float best = INFINITY; jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < int(points.size()); j += 8) {
                for (int i = 0; i < 4; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize*(j+2*i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                }
                auto score = hsum_float_4x8(sqx);
                auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                                       _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i *)index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
            }
            if (jbest < 0) {
                fprintf(stderr, "Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                GGML_ASSERT(false);
            }
            best_idx[l] = points[jbest];
        }
    }
#else
    // TODO
    std::memset(best_idx, 0, kNg*sizeof(int));
#endif
}



// ========================================== iq1_kt ====================================================

namespace {
void iqk_kt_can_quantize([[maybe_unused]] ggml_type type) {
#ifndef __AVX2__
    if (type == GGML_TYPE_IQ1_KT || type == GGML_TYPE_IQ2_KT || type == GGML_TYPE_IQ3_KT || type == GGML_TYPE_IQ4_KT) {
        fprintf(stderr, "\n==========================================================================================\n");
        fprintf(stderr, "Trellis quantization is not supported on CPUs without AVX2 or AVX512 -> aborting\n");
        fprintf(stderr, "\nFor IQ1_KT, IQ2_KT, IQ3_KT and IQ4_KT quantization can be done with the CUDA backend by adding\n");
        fprintf(stderr, "    --cuda-quantize\n");
        fprintf(stderr, "to the command line\n");
        fprintf(stderr, "==========================================================================================\n\n\n");
        GGML_ABORT("Fatal error");
    }
#endif
}
}
// ik main (cea2d61ff) defines QuantizerIQ1KT + iq1kt_quantizer() here; the fork already has both in
// iqk_quantize_shared.h:1482-1489 (shared with iqk_quantize_base.cpp), so the duplicate is dropped.



void quantize_row_iq1_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ1KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq1_kt * y = (block_iq1_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq1kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq1_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = 0;
                continue;
            }
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg; score_p = score_m;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

            scale_0 -= 8;
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp1, score_p1] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm1, score_m1] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            if (score_p1 > score_p || score_m1 > score_p) {
                idx = best_idx;
                if (score_p1 > score_m1) scales[ib] = dp1;
                else {
                    scales[ib] = dm1; idx += Q::kNg;
                }
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale/iq4k_values[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry + iq4k_values[0])/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx += wb[jj]*xb[jj]*q;
                        sumq2 += wb[jj]*q*q;
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ1_KT);

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
            y[ibl].sh[ib] = ls;
        }
    }

    *dptr = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ1_KT)*d;
    if (!d) return;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[y[ibl].sh[ib] & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

                for (int j = 0; j < Q::kNg; ++j) {
                    y[ibl].ql[ib*Q::kNg+j] = best_idx[j] & 0xff;
                    y[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg+j] |= (((best_idx[j] >> 8) & 0xf) << 4*(ib/(Q::kNblock/2)));
                    y[ibl].sh[ib] |= ((best_idx[j] >> 12) << (4+j));
                    auto xl = xb + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * ggml_get_quantize_fudge_factor(GGML_TYPE_IQ1_KT);
            if (!d) return;
        } else {
            break;
        }

    }

}
}

void quantize_row_iq1_kt_ref(const float * GGML_RESTRICT x, block_iq1_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % 32 == 0);
    quantize_iq1_kt(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_iq1_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % 32 == 0);
    block_iq1_kt * y = (block_iq1_kt *)vy;
    quantize_row_iq1_kt_ref(x, y, k);
}

size_t quantize_iq1_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    iqk_kt_can_quantize(GGML_TYPE_IQ1_KT);
    GGML_ASSERT(n_per_row%32 == 0);
    const int64_t n_per_row_padded = GGML_PAD(n_per_row, QK_K);
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_KT, n_per_row);
    std::vector<float> scales(n_per_row_padded/QuantizerIQ1KT::kBlockSize);
    std::vector<float> weights(n_per_row_padded);
    std::vector<int> idx(n_per_row_padded/QuantizerIQ1KT::kGroupSize);
    char * qrow = (char *)dst;
    if (n_per_row % QK_K == 0) {
        for (int64_t row = 0; row < nrows; ++row) {
            quantize_row_iq1_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), idx.data());
            src += n_per_row;
            qrow += row_size;
        }
        return nrows * row_size;
    }
    std::vector<float> xpad(n_per_row_padded), padded_imatrix;
    std::vector<char>  ypad(ggml_row_size(GGML_TYPE_IQ1_KT, n_per_row_padded));
    if (imatrix) {
        padded_imatrix.resize(n_per_row_padded);
        iqk_kt_pad_row(imatrix, padded_imatrix.data(), n_per_row);
    }
    for (int64_t row = 0; row < nrows; ++row) {
        iqk_kt_pad_row(src, xpad.data(), n_per_row);
        quantize_row_iq1_kt_impl(xpad.data(), ypad.data(), n_per_row_padded, imatrix ? padded_imatrix.data() : nullptr, scales.data(), weights.data(), idx.data());
        iqk_kt_finish_row(GGML_TYPE_IQ1_KT, ypad.data(), qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}


void vec_dot_iq1_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ1_KT, vx, 0, IQK_KT_ACTIVATION_TYPE, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif


    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

// ========================================== iq2_kt ====================================================

namespace {



void quantize_row_iq2_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ2KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq2_kt * y = (block_iq2_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq2kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

    float amax_scale = 0, max_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_KT);

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = 0;
                continue;
            }
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale/iq4k_values[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry + iq4k_values[0])/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx += wb[jj]*xb[jj]*q;
                        sumq2 += wb[jj]*q*q;
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            int ls1 = best_index_iq4nl(iq4k_values, id*scales[ib]);
            int ls2 = best_index_iq4nl(iq4k_values, id*scales[ib + Q::kNblock/2]);
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d * ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_KT);
    if (!d) return;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            auto qs = (uint16_t *)y[ibl].ql;
            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[(y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

                for (int j = 0; j < Q::kNg; ++j) {
                    qs[j] = best_idx[j];
                    auto xl = xb + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
                qs += Q::kNg;
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_KT);
            if (!d) return;
        } else {
            break;
        }

        if (false) {
            for (int ibl = 0; ibl < nblock; ++ibl) {
                const float * xbl = x + ibl*Q::kSuperBlockSize;
                auto scales = all_scales + ibl*Q::kNblock;
                auto qs = (uint16_t *)y[ibl].ql;
                for (int ib = 0; ib < Q::kNblock; ++ib) {
                    const float * xb = xbl + Q::kBlockSize*ib;
                    const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                    for (int j = 0; j < Q::kNg; ++j) best_idx[j] = qs[ib*Q::kNg+j];
                    auto pair = quantizer.find_best_scale(xb, weight, best_idx);
                    scales[ib] = pair.first;
                }
            }
            float id = d ? 1/d : 0.f;
            for (int ibl = 0; ibl < nblock; ++ibl) {
                auto scales = all_scales + ibl*Q::kNblock;
                for (int ib = 0; ib < Q::kNblock/2; ++ib) {
                    int ls1 = best_index_iq4nl(iq4k_values, id*scales[ib]);
                    int ls2 = best_index_iq4nl(iq4k_values, id*scales[ib + Q::kNblock/2]);
                    y[ibl].scales[ib] = ls1 | (ls2 << 4);
                }
            }
        }

    }

}
}

namespace {

template <typename Impl>
size_t quantize_kt_rows(ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix, Impl impl) {
    GGML_ASSERT(n_per_row%32 == 0);
    const size_t row_size = ggml_row_size(type, n_per_row);
    const int nt = (n_per_row & (QK_K - 1))/32;
    char * qrow = (char *)dst;
    if (nt == 0) {
        for (int64_t row = 0; row < nrows; ++row) {
            impl(src, qrow, imatrix);
            src += n_per_row;
            qrow += row_size;
        }
        return nrows*row_size;
    }
    std::vector<float> xpad(GGML_PAD(n_per_row, QK_K), 0.f), wpad;
    if (imatrix) {
        wpad.assign(imatrix, imatrix + n_per_row);
        wpad.resize(xpad.size(), 0.f);
    }
    std::vector<char> ypad(ggml_row_size(type, GGML_PAD(n_per_row, QK_K)));
    for (int64_t row = 0; row < nrows; ++row) {
        std::memcpy(xpad.data(), src, n_per_row*sizeof(float));
        impl(xpad.data(), ypad.data(), imatrix ? wpad.data() : nullptr);
        iqk_kt_finish_row(type, ypad.data(), qrow, n_per_row);
        src += n_per_row;
        qrow += row_size;
    }
    return nrows*row_size;
}

void repack_tail_iq1_kt(const block_iq1_kt * x, int nt, uint8_t * tail) {
    std::memcpy(tail, x->sh, sizeof(x->sh));
    std::memcpy(tail + sizeof(x->sh), x->ql, 4*nt);
    std::memcpy(tail + sizeof(x->sh) + 4*nt, x->qh, std::min(4*nt, int(sizeof(x->qh))));
}

void repack_tail_iq2_kt(const block_iq2_kt * x, int nt, uint8_t * tail) {
    if (nt < 1) return;
    std::memcpy(tail+0, x->scales, 4);
    std::memcpy(tail+4, x->ql, 8*nt);
}

inline void repack_kt_scales_iq3kt(const uint8_t * scales, int nt, uint8_t * tail) {
    std::memset(tail, 0, (nt + 1)/2);
    for (int ib = 0; ib < nt; ++ib) {
        tail[ib/2] |= ((scales[ib%4] >> 4*(ib/4)) & 0xf) << 4*(ib&1);
    }
}

void repack_tail_iq3_kt(const block_iq3_kt * x, int nt, uint8_t * tail) {
    std::memcpy(tail, x->ql, 8*nt);
    uint8_t * qh = tail + 8*nt;
    std::memset(qh, 0, 4*nt);
    for (int ib = 0; ib < nt; ++ib) {
        for (int j = 0; j < 32; ++j) {
            if (x->qh[j] & (1 << ib)) qh[4*ib + j/8] |= 1 << (j&7);
        }
    }
    repack_kt_scales_iq3kt(x->scales, nt, tail + 12*nt);
}

void repack_tail_iq4_kt(const block_iq4_kt * x, int nt, uint8_t * tail) {
    const uint8_t * ql = (const uint8_t *)(x->qs + 8);
    const uint8_t * qh = ql + 64;
    for (int ib = 0; ib < nt; ++ib) {
        uint8_t * record = tail + 16*ib;
        std::memcpy(record, x->qs + ib, 4);
        std::memcpy(record + 4, ql + 8*ib, 8);
        uint8_t * qht = record + 12;
        std::memset(qht, 0, 4);
        for (int jj = 0; jj < 8; ++jj) {
            const int j = 8*ib + jj;
            qht[jj/2] |= ((qh[j%32] >> 4*(j/32)) & 0xf) << 4*(jj&1);
        }
    }
}

}

void iqk_kt_finish_row(int type, const void * ypad, void * qrow, int64_t n_per_row) {
    const int nb = n_per_row/QK_K;
    const int nt = (n_per_row & (QK_K - 1))/32;
    const size_t head = ggml_row_size(ggml_type(type), nb*QK_K);
    std::memset(qrow, 0, ggml_row_size(ggml_type(type), n_per_row));
    std::memcpy(qrow, ypad, head);
    if (nt == 0) return;
    const char * blocks = (const char *)ypad + sizeof(float);
    if (type == GGML_TYPE_IQ1_KT) {
        repack_tail_iq1_kt((const block_iq1_kt *)blocks + nb, nt, (uint8_t *)qrow + head);
    } else if (type == GGML_TYPE_IQ2_KT) {
        repack_tail_iq2_kt((const block_iq2_kt *)blocks + nb, nt, (uint8_t *)qrow + head);
    } else if (type == GGML_TYPE_IQ3_KT) {
        repack_tail_iq3_kt((const block_iq3_kt *)blocks + nb, nt, (uint8_t *)qrow + head);
    } else {
        repack_tail_iq4_kt((const block_iq4_kt *)blocks + nb, nt, (uint8_t *)qrow + head);
    }
}

void iqk_kt_pad_row(const float * src, float * dst, int64_t n_per_row) {
    const int n_extra = GGML_PAD(n_per_row, QK_K) - n_per_row;
    std::memcpy(dst, src, n_per_row*sizeof(float));
    if (n_per_row >= n_extra) {
        std::memcpy(dst + n_per_row, src + n_per_row - n_extra, n_extra*sizeof(float));
    } else {
        for (int ib = 0; ib < n_extra/32; ++ib) std::memcpy(dst + n_per_row + 32*ib, src + n_per_row - 32, 32*sizeof(float));
    }
}

void quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, block_iq2_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kt(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_iq2_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_kt * y = (block_iq2_kt *)vy;
    quantize_row_iq2_kt_ref(x, y, k);
}

size_t quantize_iq2_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    iqk_kt_can_quantize(GGML_TYPE_IQ2_KT);
    GGML_ASSERT(n_per_row%32== 0);
    int n_per_row_padded = GGML_PAD(n_per_row, QK_K);
    std::vector<float> scales(n_per_row_padded/QuantizerIQ2KT::kBlockSize);
    std::vector<float> weights(n_per_row_padded);
    std::vector<int>   idx(n_per_row_padded/QuantizerIQ2KT::kGroupSize);
    if (n_per_row % QK_K == 0) {
    return quantize_kt_rows(GGML_TYPE_IQ2_KT, src, dst, nrows, n_per_row, imatrix,
            [&] (const float * x, void * y, const float * w) { quantize_row_iq2_kt_impl(x, y, n_per_row, w, scales.data(), weights.data(), idx.data()); });
    }
    size_t row_size = ggml_row_size(GGML_TYPE_IQ2_KT, n_per_row);
    size_t row_size_padded = ggml_row_size(GGML_TYPE_IQ2_KT, n_per_row_padded);
    size_t row_size_nb = ggml_row_size(GGML_TYPE_IQ2_KT, QK_K*(n_per_row/QK_K));
    std::vector<float> xpad(n_per_row_padded);
    std::vector<char>  ypad(row_size_padded);
    std::vector<float> padded_imatrix;
    if (imatrix) {
        padded_imatrix.resize(n_per_row_padded);
        iqk_kt_pad_row(imatrix, padded_imatrix.data(), n_per_row);
    }
    int nblock256 = n_per_row/QK_K;
    int nt = (n_per_row - nblock256*QK_K)/32;
    size_t size = 0;
    auto cy = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        iqk_kt_pad_row(src, xpad.data(), n_per_row);
        quantize_row_iq2_kt_impl(xpad.data(), ypad.data(), n_per_row_padded, imatrix ? padded_imatrix.data() : nullptr, scales.data(), weights.data(), idx.data());
        block_iq2_kt *qy, *qy_padded;
        if (nblock256 >= 1) {
            std::memcpy(cy, ypad.data(), row_size_nb);
            qy = (block_iq2_kt *)(cy + row_size_nb);
            qy_padded = (block_iq2_kt *)(ypad.data() + row_size_nb);
        } else {
            std::memcpy(cy, ypad.data(), 4);
            qy = (block_iq2_kt *)(cy + sizeof(float));
            qy_padded = (block_iq2_kt *)(ypad.data() + sizeof(float));
        }
        std::memcpy(qy->scales, qy_padded->scales, 4);
        std::memcpy(qy->ql, qy_padded->ql, 8*nt);
        src += n_per_row;
        cy  += row_size;
        size += row_size;
    }
    return size;
}


void vec_dot_iq2_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_KT, vx, 0, IQK_KT_ACTIVATION_TYPE, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif


    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

namespace {


void quantize_row_iq3_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales,
        float * all_weights, float * qtmp) {

    constexpr float kSigmaScale = 2.0f;
    constexpr float kStep = 8.0f;

    using Q = QuantizerIQ3KT;

    static_assert(Q::kNumVal%8 == 0);

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;

    float * dptr = (float *)vy;

    block_iq3_kt * y = (block_iq3_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq3kt_quantizer();

    int nblock = GGML_PAD(n_per_row, Q::kSuperBlockSize) / Q::kSuperBlockSize;

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (!amax_row) {
        *dptr = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq3_kt));
        return;
    }

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_scale = 0, max_scale = 0;

    float xaux[Q::kBlockSize];

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_KT);

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_kt));

        auto scales = all_scales + ibl*Q::kNblock;
        auto xbl = x + ibl*Q::kSuperBlockSize;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                xaux[j] = ax;
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                continue;
            }

            //quantizer.find_best_match(amax/96.f, xaux, weight, best_idx+Q::kNg);
            //scales[ib] = quantizer.find_best_scale(xaux, weight, best_idx+Q::kNg).first;

            float scale_0 = std::max(84.f, 123.f*amax/amax_row);
            //float scale_0 = std::max(64.f, 123.f*amax/amax_row);
            float best = 0;
            bool found_solution = false;
            for (int itry = -3; itry <= 3; ++itry) {
                quantizer.find_best_match(amax/(scale_0 + kStep*itry), xaux, weight, best_idx);
                auto [d, score] = quantizer.find_best_scale(xaux, weight, best_idx);
                if (score > best) {
                    best = score;
                    found_solution = true;
                    scales[ib] = d;
                    std::memcpy(best_idx+Q::kNg, best_idx, Q::kNg*sizeof(int));
                }
            }
            if (!found_solution) {
                fprintf(stderr, "======================= %s: failed to find solution for a block\n", __func__);
                fprintf(stderr, "Model weights and importances:\n");
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    fprintf(stderr, "%2d  %g  %g\n", j, xaux[j], weight[j]);
                }
                GGML_ASSERT(false);
            }

            auto xt = qtmp + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                auto q = quantizer.values() + Q::kGroupSize*best_idx[Q::kNg+ig];
                for (int j = 0; j < Q::kGroupSize; ++j) *xt++ = q[j];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    GGML_ASSERT(max_scale >= 0);
    float d = max_scale/15;
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry*0.2f + 15)/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * qb = qtmp + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = nearest_int(id*scales[ib]);
                ls = std::max(0, std::min(15, ls));
                float dl = ls;
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    float q = dl*qb[j];
                    sumqx += wb[j]*std::abs(xb[j])*q;
                    sumq2 += wb[j]*q*q;
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
                qb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            int ls1 = nearest_int(id*scales[ib]);
            int ls2 = nearest_int(id*scales[ib + Q::kNblock/2]);
            ls1 = std::max(0, std::min(15, ls1));
            ls2 = std::max(0, std::min(15, ls2));
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d * ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_KT);

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            uint16_t * ql = (uint16_t *)y[ibl].ql;

            std::memset(y[ibl].qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    xaux[j] = std::abs(xb[j]);
                    if (xb[j] < 0) y[ibl].qh[j] |= (1 << ib);
                }
                int ls = (y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf;
                float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    ql[ib*Q::kNg+j] = best_idx[j];
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_KT);
            if (!d) break;
        } else {
            break;
        }
    }
}
}

void quantize_row_iq3_kt_ref(const float * x, block_iq3_kt * y, int64_t k) {
    assert(k % 32 == 0);
    quantize_iq3_kt(x, (void *)y, 1, k, nullptr, nullptr);
}

// (this fork's relocated pre-#2616 KT-tail helpers were dropped here: ik's generalised set,
// with iqk_kt_finish_row, now sits before quantize_iq2_kt). The Thireus d9e286846 port's IQ3/IQ4-only
// iqk_kt_finish_row that git placed here on the ik-main merge was a second definition of the same function;
// ik's one above covers IQ1/IQ2/IQ3/IQ4_KT and returns early when the row has no tail.

void quantize_row_iq3_kt(const float * x, void * vy, int64_t k) {
    assert(k % 32 == 0);
    block_iq3_kt * y = (block_iq3_kt *)vy;
    quantize_row_iq3_kt_ref(x, y, k);
}

size_t quantize_iq3_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    iqk_kt_can_quantize(GGML_TYPE_IQ3_KT);
    std::vector<float> scales(GGML_PAD(n_per_row, QK_K)/QuantizerIQ3KT::kBlockSize);
    std::vector<float> weights(GGML_PAD(n_per_row, QK_K));
    std::vector<float> xtmp(GGML_PAD(n_per_row, QK_K));
    return quantize_kt_rows(GGML_TYPE_IQ3_KT, src, dst, nrows, n_per_row, imatrix,
            [&] (const float * x, void * y, const float * w) { quantize_row_iq3_kt_impl(x, y, n_per_row, w, scales.data(), weights.data(), xtmp.data()); });
}


void vec_dot_iq3_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % 32 == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_KT, vx, 0, IQK_KT_ACTIVATION_TYPE, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif


    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}

// ======================================== iq4_kt

namespace{


const QuantizerIQ4KT& iq4kt_quantizer(bool with_offset = false) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ4KT> quantizer1;
    static std::unique_ptr<QuantizerIQ4KT> quantizer2;
    if (with_offset) {
        if (!quantizer2) quantizer2 = std::make_unique<QuantizerIQ4KT>(625, 6, 4096+32768);
        return *quantizer2;
    }
    if (!quantizer1) quantizer1 = std::make_unique<QuantizerIQ4KT>(625, 6, 4096);
    return *quantizer1;
}


void quantize_row_iq4_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights) {

    constexpr float kSigmaScale = 2.0f;
    constexpr int kNtry = 2;
    using Q = QuantizerIQ4KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq4_kt * y = (block_iq4_kt *)(dptr + 1);

    auto& quantizer1 = iq4kt_quantizer();
    auto& quantizer2 = iq4kt_quantizer(true);

    int nblock = GGML_PAD(n_per_row, Q::kSuperBlockSize) / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights, n_per_row);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }
    if (!amax_row) {
        dptr[0] = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq4_kt));
        return;
    }

    int   best_idx[2*Q::kNg];
    float xaux[Q::kBlockSize];

    float amax_scale = 0, max_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KT);

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq4_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                xaux[j] = xbl[ib*Q::kBlockSize+j];
                float ax = std::abs(xaux[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer1.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) {
                    best = score_p; scales[ib] = dp;
                }
                quantizer1.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) {
                    best = score_m; scales[ib] = dm;
                }
            }

            quantizer2.find_best_match(scales[ib], xaux, weight, best_idx);
            auto [d, score] = quantizer2.find_best_scale(xaux, weight, best_idx);
            if (score > best) {
                best = score;
                scales[ib] = d;
                y[ibl].qs[ib] = 1;
            }
            bool with_offset = false;
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer2.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) {
                    best = score_p; scales[ib] = dp; with_offset = true;
                }
                quantizer2.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) {
                    best = score_m; scales[ib] = dm; with_offset = true;
                }
            }
            if (with_offset) y[ibl].qs[ib] = 1;

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    float d = -max_scale/64;

    dptr[0] = d * ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KT);
    if (!d) return;

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;

    for (int iloop = 0; iloop < 1; ++iloop) {

        const float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            // high 3 bits + scales
            // each block of 32 needs 8 x 3 (high bits) + 1 x 8 (scale) = 32 bits = 1 x uint32_t
            // we have 8 blocks
            auto shb = y[ibl].qs;  // high 3 bits + scales
            auto ql = (uint8_t *)(shb + Q::kNblock);
            auto qh = ql + kNumGroups;
            std::memset(qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                auto& quantizer = y[ibl].qs[ib] & 1 ? quantizer2 : quantizer1;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) xaux[j] = xbl[ib*Q::kBlockSize+j];
                int ls = nearest_int(id*scales[ib]);
                ls = std::min(ls, 63);
                *(uint8_t *)(shb + ib) = ((ls + 64) << 1) | (shb[ib] & 1);
                float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    shb[ib] |= ((best_idx[j] >> 12) << (8 + 3*j));
                    ql[Q::kNg*ib + j] = best_idx[j] & 255;
                    qh[(Q::kNg*ib + j)%(kNumGroups/2)] |= ((best_idx[j] >> 8) & 0xf) << 4*((Q::kNg*ib + j)/(kNumGroups/2));
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + Q::kGroupSize*best_idx[j];
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            dptr[0] = d * ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KT);
            if (!d) break;
        } else {
            break;
        }
    }
}
}

int iqk_kt_codebook(int type, int bank, const int ** offsets, const int ** points, const float ** values, const float ** mid) {
    struct Flat { std::vector<int> offsets, points; };
    static std::mutex mutex;
    static Flat flat[5];
    std::lock_guard<std::mutex> lock(mutex);
    auto flatten = [offsets, points, values, mid] (const auto& q, Flat& f) {
        if (f.offsets.empty()) {
            f.offsets.push_back(0);
            for (auto& p : q.in_cluster()) {
                f.points.insert(f.points.end(), p.begin(), p.end());
                f.offsets.push_back(f.points.size());
            }
        }
        *offsets = f.offsets.data();
        *points = f.points.data();
        *values = q.values();
        *mid = q.mid();
        return int(f.offsets.size()) - 1;
    };
    if (type == GGML_TYPE_IQ4_KT) return flatten(iq4kt_quantizer(bank == 1), flat[bank]);
    if (type == GGML_TYPE_IQ3_KT) return flatten(iq3kt_quantizer(), flat[2]);
    if (type == GGML_TYPE_IQ2_KT) return flatten(iq2kt_quantizer(), flat[3]);
    if (type == GGML_TYPE_IQ1_KT) return flatten(iq1kt_quantizer(), flat[4]);
    return 0;
}

// iqk_ks_index / iqk_ks_simple_index (ik main) live in iqk_quantize_base.cpp: the IQ3/IQ5 tables they return are
// file-local there, and the caller (ggml-cuda/ks-encoder.cu) links ggml-base.

void quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, block_iq4_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % 32 == 0);
    quantize_iq4_kt(x, (void *)y, 1, k, nullptr, nullptr);
}

void quantize_row_iq4_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % 32 == 0);
    block_iq4_kt * y = (block_iq4_kt *)vy;
    quantize_row_iq4_kt_ref(x, y, k);
}

size_t quantize_iq4_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    iqk_kt_can_quantize(GGML_TYPE_IQ4_KT);
    std::vector<float> scales(GGML_PAD(n_per_row, QK_K)/QuantizerIQ4KT::kBlockSize);
    std::vector<float> weights(GGML_PAD(n_per_row, QK_K));
    return quantize_kt_rows(GGML_TYPE_IQ4_KT, src, dst, nrows, n_per_row, imatrix,
            [&] (const float * x, void * y, const float * w) { quantize_row_iq4_kt_impl(x, y, n_per_row, w, scales.data(), weights.data()); });
}


void vec_dot_iq4_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % 32 == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KT, vx, 0, IQK_KT_ACTIVATION_TYPE, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif


    // NO FALLBACK EXISTS, so a declined pair must not return quietly. Without this, the
    // function falls off the end with *s NEVER ASSIGNED and the caller reads whatever was
    // already in that memory as the dot product. Measured before this was added: iq4_kt
    // reported one-vs-ref 2.261e+37, identical across 3 runs of one binary and DIFFERENT
    // across builds -- a computation does not change when unrelated code is recompiled, a
    // stale buffer does. A successful iqk_mul_mat returns above, so every working pairing is
    // unaffected; this only fires where the answer was already wrong and silent.
    GGML_ABORT("%s: iqk_mul_mat declined this (weight, activation) pair, and there is no\n"
               "scalar fallback here, so returning would leave *s unwritten and the caller\n"
               "would read stale memory as a result. The .vec_dot_type in type_traits_cpu and\n"
               "the activation type this function hands iqk_mul_mat must BOTH match what\n"
               "iqk_set_kernels_*() accepts on this architecture.", __func__);
}



size_t quantize_q1_0_g128(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row % QK1_0_G128 == 0);
    int64_t ntot = nrows * n_per_row;
    quantize_row_q1_0_g128(src, dst, ntot);
    int64_t nblock = ntot / QK1_0_G128;
    return nblock * sizeof(block_q1_0_g128);
}


void vec_dot_q1_0_g128_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK1_0_G128 == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int nb = n / QK1_0_G128;

    constexpr uint8_t k_mask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};

    constexpr int n4 = QK1_0_G128 / QK8_0;

    auto x = (const block_q1_0_g128 *)vx;
    auto y = (const block_q8_0_x4 *)vy;
    int16_t sumi[QK1_0_G128/8];
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        auto dx = GGML_FP16_TO_FP32(x[ib].d);
        auto qx = x[ib].qs;
        auto qy = y[ib].qs;
        for (int k = 0; k < QK1_0_G128/8; ++k) {
            uint8_t bits = qx[k];
            int16_t s = 0;
            for (int j = 0; j < 8; ++j) {
                s += (bits & k_mask[j] ? qy[j] : -qy[j]);
            }
            qy += 8;
            sumi[k] = s;
        }
        auto s = sumi;
        for (int k = 0; k < n4; ++k) {
            float dy = GGML_FP16_TO_FP32(y[ib].d[k]);
            sumf += dx*dy*(s[0] + s[1] + s[2] + s[3]);
            s += 4;
        }
    }
    *s = sumf;
}

//
// Q2_0_G128 -- Bonsai 2-bit ternary, group 128.
//
// Four 2-bit codes per byte, LSB-first (shifts 0,2,4,6), elements sequential; the
// dequantized value is d * (code - 1), so codes 0/1/2 mean -1/0/+1 and code 3 (+2) is
// unused by a ternary quantizer. That mapping is taken VERBATIM from the PrismML fork's
// ggml/src/ggml-cpu/quants.c ("map {0,1,2,3} -> {-1,0,1,2}") rather than inferred: a wrong
// guess decodes to silently WRONG weights instead of an error, and this family has already
// produced fluent word salad once from exactly that mistake.
//
// Layout mirrors the working Q1_0_G128 above (fp16 scale first, then packed codes), same
// fork generation. 32 code bytes + 2 scale bytes = 34 bytes per 128 weights = 2.125 bpw,
// which is what Ternary-Bonsai-27B-Q2_0.gguf actually measures.
//



size_t quantize_q2_0_g128(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row % QK2_0_G128 == 0);
    int64_t ntot = nrows * n_per_row;
    quantize_row_q2_0_g128(src, dst, ntot);
    int64_t nblock = ntot / QK2_0_G128;
    return nblock * sizeof(block_q2_0_g128);
}


void vec_dot_q2_0_g128_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    // THIS is the kernel that runs, and only because the type is excluded at every iqk
    // matmul dispatch site. In the authoring tree it was measured NEVER REACHED while
    // dequantize_row_q2_0_g128 above DID fire, in the same run of the same binary --
    // iqk had claimed the matmul and answered with garbage. Ours is excluded at FIVE
    // matmul sites (the authoring tree had three) and the three fused_up_gate sites are
    // unreachable because the graph declines to build a fused node for this type.
    // lint_iqk_dispatch_guards.py enforces that and reports 0 violations.
    //
    // Scalar, and UNMEASURED on this host -- correctness first. The authoring tree
    // reported ~1.8 t/s for this scalar path against ~6 t/s for the 1-bit sibling's AVX2
    // iqk kernel; those are its numbers, not ours. A vectorised kernel for this type is
    // separate work and should be judged on a measurement taken here.
    assert(n % QK2_0_G128 == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int nb = n / QK2_0_G128;

    constexpr int n4 = QK2_0_G128 / QK8_0;   // 4 sub-blocks of 32 per 128-weight group

    auto x = (const block_q2_0_g128 *)vx;
    auto y = (const block_q8_0_x4 *)vy;
    int16_t sumi[QK2_0_G128/4];              // one partial per code byte (4 weights each)
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        auto dx = GGML_FP16_TO_FP32(x[ib].d);
        auto qx = x[ib].qs;
        auto qy = y[ib].qs;
        for (int k = 0; k < QK2_0_G128/4; ++k) {
            const uint8_t bits = qx[k];
            // max |partial| here is 4*127 = 508, and a 32-weight sub-block sums 8 of
            // them (<= 4064), so int16 cannot overflow.
            int16_t acc = 0;
            acc += (int16_t)((((int)((bits >> 0) & 3)) - 1) * qy[0]);
            acc += (int16_t)((((int)((bits >> 2) & 3)) - 1) * qy[1]);
            acc += (int16_t)((((int)((bits >> 4) & 3)) - 1) * qy[2]);
            acc += (int16_t)((((int)((bits >> 6) & 3)) - 1) * qy[3]);
            qy += 4;
            sumi[k] = acc;
        }
        auto sp = sumi;
        for (int k = 0; k < n4; ++k) {
            float dy = GGML_FP16_TO_FP32(y[ib].d[k]);
            int32_t t = 0;
            for (int j = 0; j < 8; ++j) t += sp[j];   // 8 code bytes = 32 weights
            sumf += dx*dy*t;
            sp += 8;
        }
    }
    *s = sumf;
}


//
// xr[0][ib].qs[0]: bits 0...3 go into bytes 0...3, bit 0
//                  bits 4...7 go into bytes 0...3, bit 1
// xr[1][ib].qs[0]: bits 0...3 go into bytes 4...7, bit 0
//                  bits 4...7 go into bytes 4...7, bit 1
// ...
// xr[7][ib].qs[0]: bits 0...3 go into bytes 28...31, bit 0
//                  bits 4...7 go into bytes 28...31, bit 1
//
// xr[0][ib].qs[1]: bits 0...3 go into bytes 0...3, bit 2
//                  bits 4...7 go into bytes 0...3, bit 3
// xr[1][ib].qs[1]: bits 0...3 go into bytes 4...7, bit 2
//                  bits 4...7 go into bytes 4...7, bit 3
// ...
// xr[7][ib].qs[1]: bits 0...3 go into bytes 28...31, bit 2
//                  bits 4...7 go into bytes 28...31, bit 3
//
// xr[0][ib].qs[2]: bits 0...3 go into bytes 0...3, bit 4
//                  bits 4...7 go into bytes 0...3, bit 5
// xr[1][ib].qs[2]: bits 0...3 go into bytes 4...7, bit 4
//                  bits 4...7 go into bytes 4...7, bit 5
// ...
// xr[7][ib].qs[2]: bits 0...3 go into bytes 28...31, bit 4
//                  bits 4...7 go into bytes 28...31, bit 5
//
// xr[0][ib].qs[3]: bits 0...3 go into bytes 0...3, bit 6
//                  bits 4...7 go into bytes 0...3, bit 7
// xr[1][ib].qs[3]: bits 0...3 go into bytes 4...7, bit 6
//                  bits 4...7 go into bytes 4...7, bit 7
// ...
// xr[7][ib].qs[3]: bits 0...3 go into bytes 28...31, bit 6
//                  bits 4...7 go into bytes 28...31, bit 7
//
// xr[0][ib].qs[4]: bits 0...3 go into bytes 32+0...3, bit 0
//                  bits 4...7 go into bytes 32+0...3, bit 1
// xr[1][ib].qs[4]: bits 0...3 go into bytes 32+4...7, bit 0
//                  bits 4...7 go into bytes 32+4...7, bit 1
// ...
// xr[7][ib].qs[4]: bits 0...3 go into bytes 32+28...31, bit 0
//                  bits 4...7 go into bytes 32+28...31, bit 1
//
// xr[0][ib].qs[5]: bits 0...3 go into bytes 32+0...3, bit 2
//                  bits 4...7 go into bytes 32+0...3, bit 3
// xr[1][ib].qs[5]: bits 0...3 go into bytes 32+4...7, bit 2
//                  bits 4...7 go into bytes 32+4...7, bit 3
// ...
// xr[7][ib].qs[5]: bits 0...3 go into bytes 32+28...31, bit 2
//                  bits 4...7 go into bytes 32+28...31, bit 3
//
// xr[0][ib].qs[6]: bits 0...3 go into bytes 32+0...3, bit 4
//                  bits 4...7 go into bytes 32+0...3, bit 5
// xr[1][ib].qs[6]: bits 0...3 go into bytes 32+4...7, bit 4
//                  bits 4...7 go into bytes 32+4...7, bit 5
// ...
// xr[7][ib].qs[6]: bits 0...3 go into bytes 32+28...31, bit 4
//                  bits 4...7 go into bytes 32+28...31, bit 5
//
// xr[0][ib].qs[7]: bits 0...3 go into bytes 32+0...3, bit 6
//                  bits 4...7 go into bytes 32+0...3, bit 7
// xr[1][ib].qs[7]: bits 0...3 go into bytes 32+4...7, bit 6
//                  bits 4...7 go into bytes 32+4...7, bit 7
// ...
// xr[7][ib].qs[7]: bits 0...3 go into bytes 32+28...31, bit 6
//                  bits 4...7 go into bytes 32+28...31, bit 7


static void dequantize_block_q1_0_g128_r8(const block_q1_0_g128_r8 & x, int8_t * y) {
    for (int k = 0; k < 8; ++k) {
        auto yk = y + 128*k;
        for (int l = 0; l < 4; ++l) {
            auto qx = x.qs + 32*l;
            for (int i = 0; i < 4; ++i) {
                uint8_t mask1 = 1 << (2*i+0);
                uint8_t mask2 = 1 << (2*i+1);
                for (int j = 0; j < 4; ++j) {
                    yk[32*l + 8*i + j + 0] = qx[4*k + j] & mask1 ? 1 : 0;
                }
                for (int j = 0; j < 4; ++j) {
                    yk[32*l + 8*i + j + 4] = qx[4*k + j] & mask2 ? 1 : 0;
                }
            }
        }
    }
}




// ggml_quantize_chunk entry for Q1_0_G128_R8, ADDED IN THE FORK: ik #2541 gave the type no chunk case, so a caller that
// quantizes it through ggml_quantize_chunk (test-mulmat-gemv does, for every row-interleaved type) hit
// GGML_ASSERT(result == nrows * row_size) at ggml.c:30787. Rows come in groups of 8, one repacked group per call.
size_t quantize_q1_0_g128_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        [[maybe_unused]] const float * imatrix, [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows % QK1_0_G128_R8_ROWS == 0);
    GGML_ASSERT(n_per_row % QK1_0_G128 == 0);
    const size_t group_bytes = (size_t)(n_per_row/QK1_0_G128) * sizeof(block_q1_0_g128_r8);
    for (int64_t row = 0; row < nrows; row += QK1_0_G128_R8_ROWS) {
        quantize_row_q1_0_g128_r8(src + row*n_per_row, (char *)dst + (size_t)(row/QK1_0_G128_R8_ROWS)*group_bytes,
                QK1_0_G128_R8_ROWS*n_per_row);
    }
    return (size_t)(nrows/QK1_0_G128_R8_ROWS) * group_bytes;
}




// ggml_quantize_chunk entry for PQ2_0_R8, ADDED IN THE FORK (ik #2550 has none; same omission as Q1_0_G128_R8 above): rows
// come in groups of 8, one repacked group per call, so llama-quantize can write the R8 type directly and test-mulmat-gemv
// style callers can quantize it through ggml_quantize_chunk.
size_t quantize_pq2_0_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        [[maybe_unused]] const float * imatrix, [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows % QK_PQ2_0_R8_ROWS == 0);
    GGML_ASSERT(n_per_row % QK_PQ2_0 == 0);
    const size_t group_bytes = (size_t)(n_per_row/QK_PQ2_0) * sizeof(block_pq2_0_r8);
    for (int64_t row = 0; row < nrows; row += QK_PQ2_0_R8_ROWS) {
        quantize_row_pq2_0_r8(src + row*n_per_row, (char *)dst + (size_t)(row/QK_PQ2_0_R8_ROWS)*group_bytes,
                QK_PQ2_0_R8_ROWS*n_per_row);
    }
    return (size_t)(nrows/QK_PQ2_0_R8_ROWS) * group_bytes;
}

static void ptq1_0_to_iq1_bn([[maybe_unused]] const block_ptq1_0 & b, [[maybe_unused]] block_iq1_bn * out) {
#ifdef __AVX2__
    static const int k_nb[6] = {1, 3, 9, 27, 81, 243};
    alignas(16) uint8_t trit[QK_PTQ1_0];
    auto digits16 = [](__m128i v, int kn) -> __m128i {
        auto k  = _mm_set1_epi16((short)kn);
        auto lo = _mm_unpacklo_epi8(v, _mm_setzero_si128());
        auto hi = _mm_unpackhi_epi8(v, _mm_setzero_si128());
        lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_and_si128(_mm_mullo_epi16(lo, k), _mm_set1_epi16(0xFF)), _mm_set1_epi16(3)), 8);
        hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_and_si128(_mm_mullo_epi16(hi, k), _mm_set1_epi16(0xFF)), _mm_set1_epi16(3)), 8);
        return _mm_packus_epi16(lo, hi);
    };
    auto v16 = _mm_loadu_si128((const __m128i *)b.qs);
    for (int n = 0; n < 5; ++n) _mm_store_si128((__m128i *)(trit + 16*n), digits16(v16, k_nb[n]));
    auto v8 = _mm_loadl_epi64((const __m128i *)(b.qs + 16));
    for (int n = 0; n < 5; ++n) _mm_storel_epi64((__m128i *)(trit + 80 + 8*n), digits16(v8, k_nb[n]));
    for (int n = 0; n < 4; ++n) for (int h = 0; h < 2; ++h) {
        uint8_t q = (uint8_t)(b.qh[h]*k_nb[n]);
        trit[120 + 2*n + h] = (uint8_t)(((uint16_t)q*3) >> 8);
    }
    for (int sub = 0; sub < QK_PTQ1_0/QK_IQ1BN; ++sub) {
        auto & o = out[sub];
        int v13 = 0;
        for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
            for (int kk = 0; kk < 3; ++kk) {
                int idx = 0;
                for (int j = 0; j < 5; ++j) idx += k_nb[j]*trit[sub*QK_IQ1BN + 16*i16 + 5*kk + j];
                o.ql[3*i16 + kk] = (uint8_t)((256*idx + k_nb[5] - 1)/k_nb[5]);
            }
            v13 += k_nb[i16]*trit[sub*QK_IQ1BN + 16*i16 + 15];
        }
        o.extra = (uint8_t)((256*v13 + k_nb[5] - 1)/k_nb[5]);
    }
#else
    GGML_ABORT("Not implemented");
#endif
}







// ggml_quantize_chunk entry for PTQ1_0_R8, ADDED IN THE FORK (see quantize_pq2_0_r8): groups of 8 rows per call.
size_t quantize_ptq1_0_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        [[maybe_unused]] const float * imatrix, [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows % QK_PTQ1_0_R8_ROWS == 0);
    GGML_ASSERT(n_per_row % QK_PTQ1_0 == 0);
    const size_t group_bytes = (size_t)(n_per_row/QK_PTQ1_0) * sizeof(block_ptq1_0_r8);
    for (int64_t row = 0; row < nrows; row += QK_PTQ1_0_R8_ROWS) {
        quantize_row_ptq1_0_r8(src + row*n_per_row, (char *)dst + (size_t)(row/QK_PTQ1_0_R8_ROWS)*group_bytes,
                QK_PTQ1_0_R8_ROWS*n_per_row);
    }
    return (size_t)(nrows/QK_PTQ1_0_R8_ROWS) * group_bytes;
}

void vec_dot_ptq1_0_r8_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_PTQ1_0_R8, vx, 0, GGML_TYPE_Q8_K128, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n % QK_PTQ1_0 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_ABORT("vec_dot_ptq1_0_r8_q8_K: iqk_mul_mat is required for PTQ1_0_R8");
}

void vec_dot_pq2_0_r8_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_PQ2_0_R8, vx, 0, GGML_TYPE_Q8_K128, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n % QK_PQ2_0 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_ABORT("vec_dot_pq2_0_r8_q8_K: iqk_mul_mat is required for PQ2_0_R8");
}

void vec_dot_q1_0_g128_r8_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q1_0_G128_R8, vx, 0, GGML_TYPE_Q8_K128, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n % QK1_0_G128 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

namespace {
template <typename Block>
inline int check_row_for_blocks_256_fp16(int nblock, const Block * x) {
    int nbad = 0;
    for (int ib = 0; ib < nblock; ++ib) {
        float d = GGML_FP16_TO_FP32(x[ib].d);
        if (isnan(d)) ++nbad;
    }
    return nbad;
}
template <typename Block>
bool check_tensor_for_blocks_256_fp16(const ggml_tensor * tensor) {
    int nblock = tensor->ne[0]/QK_K;
    int nbad = 0;
    for (int row = 0; row < ggml_nrows(tensor); ++row) {
        auto x = (const Block *)((const char *)tensor->data + tensor->nb[1]*row);
        nbad += check_row_for_blocks_256_fp16(nblock, x);
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN block scales out of %g blocks in tensor %s\n", __func__,
                nbad, 1.*ggml_nrows(tensor)*nblock, tensor->name);
        if (tensor->ne[2] > 1) {
            int nb = tensor->ne[0]/QK_K;
            for (int64_t i02 = 0; i02 < tensor->ne[2]; ++i02) {
                int nbad_expert = 0;
                auto xex = (const char *)((const char *)tensor->data + i02*tensor->nb[2]);
                for (int64_t i01 = 0; i01 < tensor->ne[1]; ++i01) {
                    auto xr = (const Block *)(xex + i01*tensor->nb[1]);
                    nbad_expert += check_row_for_blocks_256_fp16(nb, xr);
                }
                if (nbad_expert > 0) fprintf(stderr,"    there are %d NaN block scales for expert %g\n", nbad_expert, 1.*i02);
            }
        }
        return false;
    }
    return true;
}
template <typename Block>
inline int check_row_for_blocks_256_fp16(int nblock, const Block * x, int nr) {
    int nbad = 0;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int j = 0; j < nr; ++j) {
            if (!isfinite(GGML_FP16_TO_FP32(x[ib].d[j]))) ++nbad;
        }
    }
    return nbad;
}
template <typename Block, int nr>
bool check_tensor_for_blocks_256_fp16_repacked(const ggml_tensor * tensor) {
    int nblock = tensor->ne[0]/QK_K;
    int nbad = 0;
    for (int row = 0; row < ggml_nrows(tensor); row += nr) {
        auto x = (const Block *)((const char *)tensor->data + tensor->nb[1]*row);
        nbad += check_row_for_blocks_256_fp16(nblock, x, nr);
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN block scales out of %g blocks in tensor %s\n", __func__,
                nbad, 1.*ggml_nrows(tensor)*nblock, tensor->name);
        if (tensor->ne[2] > 1) {
            int nb = tensor->ne[0]/QK_K;
            for (int64_t i02 = 0; i02 < tensor->ne[2]; ++i02) {
                int nbad_expert = 0;
                auto xex = (const char *)((const char *)tensor->data + i02*tensor->nb[2]);
                for (int64_t i01 = 0; i01 < tensor->ne[1]; i01 += nr) {
                    auto xr = (const Block *)(xex + i01*tensor->nb[1]);
                    nbad_expert += check_row_for_blocks_256_fp16(nb, xr, nr);
                }
                if (nbad_expert > 0) fprintf(stderr,"    there are %d NaN block scales for expert %g\n", nbad_expert, 1.*i02);
            }
        }
        return false;
    }
    return true;
}
struct F32Scale {
    static inline int check_row(const char * data) {
        float d = *(const float *)data;
        return isfinite(d) ? 0 : 1;
    }
};
struct F16Scale {
    static inline int check_row(const char * data) {
        float d = GGML_FP16_TO_FP32(*(const ggml_half *)data);
        return isfinite(d) ? 0 : 1;
    }
};
template <int nr>
struct F32ScaleRX {
    static inline int check_row(const char * data) {
        auto d = (const float *)data;
        int nbad = 0;
        for (int i = 0; i < nr; ++i) {
            if (!isfinite(d[i])) ++nbad;
        }
        return nbad;
    }
};
template <int nr>
struct F16ScaleRX {
    static inline int check_row(const char * data) {
        auto d = (const ggml_half *)data;
        int nbad = 0;
        for (int i = 0; i < nr; ++i) {
            if (!isfinite(GGML_FP16_TO_FP32(d[i]))) ++nbad;
        }
        return nbad;
    }
};
template <typename RS>
bool check_tensor_row_scales(const ggml_tensor * tensor) {
    auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    int num_rows = ggml_nrows(tensor);
    auto data = (const char *)tensor->data;
    int nbad = 0;
    for (int row = 0; row < num_rows; ++row) {
        nbad += RS::check_row(data);
        data += row_size;
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN row scales out of %d rows in tensor %s\n", __func__,
                nbad, num_rows, tensor->name);
        return false;
    }
    return true;
}
}

bool iqk_validate_tensor(const ggml_tensor * tensor) {
    if (!tensor) return true;
    if (!ggml_is_contiguous(tensor)) return true;

    switch (tensor->type) {
        case GGML_TYPE_IQ2_K:      return check_tensor_for_blocks_256_fp16<block_iq2_k>(tensor);
        case GGML_TYPE_IQ3_K:      return check_tensor_for_blocks_256_fp16<block_iq3_k>(tensor);
        case GGML_TYPE_IQ4_K:      return check_tensor_for_blocks_256_fp16<block_iq4_k>(tensor);
        case GGML_TYPE_IQ5_K:      return check_tensor_for_blocks_256_fp16<block_iq5_k>(tensor);
        case GGML_TYPE_IQ6_K:      return check_tensor_for_blocks_256_fp16<block_iq6_k>(tensor);
        case GGML_TYPE_IQ2_XXS:    return check_tensor_for_blocks_256_fp16<block_iq2_xxs>(tensor);
        case GGML_TYPE_IQ2_XS:     return check_tensor_for_blocks_256_fp16<block_iq2_xs>(tensor);
        case GGML_TYPE_IQ2_S:      return check_tensor_for_blocks_256_fp16<block_iq2_s>(tensor);
        case GGML_TYPE_IQ3_XXS:    return check_tensor_for_blocks_256_fp16<block_iq3_xxs>(tensor);
        case GGML_TYPE_IQ3_S:      return check_tensor_for_blocks_256_fp16<block_iq3_s>(tensor);
        case GGML_TYPE_IQ4_XS:     return check_tensor_for_blocks_256_fp16<block_iq4_xs>(tensor);
        case GGML_TYPE_IQ2_K_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq2_k_r4, 4>(tensor);
        case GGML_TYPE_IQ3_K_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq3_k_r4, 4>(tensor);
        case GGML_TYPE_IQ4_K_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq4_k_r4, 4>(tensor);
        case GGML_TYPE_IQ5_K_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq5_k_r4, 4>(tensor);
        case GGML_TYPE_IQ2_XXS_R4: return check_tensor_for_blocks_256_fp16_repacked<block_iq2_xxs_r4, 4>(tensor);
        case GGML_TYPE_IQ2_XS_R4:  return check_tensor_for_blocks_256_fp16_repacked<block_iq2_xs_r4, 4>(tensor);
        case GGML_TYPE_IQ2_S_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq2_s_r4, 4>(tensor);
        case GGML_TYPE_IQ3_XXS_R4: return check_tensor_for_blocks_256_fp16_repacked<block_iq3_xxs_r4, 4>(tensor);
        case GGML_TYPE_IQ3_S_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq3_s_r4, 4>(tensor);
        case GGML_TYPE_IQ4_XS_R8:  return check_tensor_for_blocks_256_fp16_repacked<block_iq4_xs_r8, 8>(tensor);
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS:     return check_tensor_row_scales<F32Scale>(tensor);
        case GGML_TYPE_IQ2_BN_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:  return check_tensor_row_scales<F32ScaleRX<4>>(tensor);
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:     return check_tensor_row_scales<F16Scale>(tensor);
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:   return check_tensor_row_scales<F16ScaleRX<4>>(tensor);

        default: break;
    }
    return true;
}
