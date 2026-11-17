//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// ggml-base half of the fork quantizer file. HALF 2 of S69-GGML-BASE-DLL-CANNOT-LINK-CIRCULAR-ACROSS-A-DLL-BOUNDARY-2026-09-15-01.
//
// The dequantize_row_* / quantize_row_*_ref implementations that ggml.c type_traits[] .to_float / .from_float_ref point at,
// plus every helper they (transitively) call, moved here VERBATIM from iqk_quantize.cpp. Only functions whose whole call
// closure is architecture-neutral were moved (no AVX2/AVX512/NEON arm in the closure); this file is compiled WITHOUT arch flags,
// exactly like ggml-quants.c. The roots whose closure reaches an arch arm stay in iqk_quantize.cpp (ggml-cpu).
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

#include "iqk_quantize_shared.h"
#include "iqk_scale_search.h"

// AVX2 scale search registered by ggml-cpu (see iqk_scale_search.h). Written once from ggml_cpu_init, read by the
// quantizer threads: atomics so a quantize running while a backend initialises sees either the old or the new pointer.
static std::atomic<iqk_scale_search_fn> g_iqk_scale_search[IQK_SCALE_SEARCH_COUNT];
extern "C" void iqk_set_scale_search(enum iqk_scale_search_kind kind, iqk_scale_search_fn fn) {
    if (kind >= 0 && kind < IQK_SCALE_SEARCH_COUNT) g_iqk_scale_search[kind].store(fn, std::memory_order_release);
}
extern "C" iqk_scale_search_fn iqk_get_scale_search(enum iqk_scale_search_kind kind) {
    return kind >= 0 && kind < IQK_SCALE_SEARCH_COUNT ? g_iqk_scale_search[kind].load(std::memory_order_acquire) : nullptr;
}
#define A_IQ6K -127.f
#define B_IQ6K 6.2568f
#define C_IQ6K 0.11218f
#define D_IQ6K 0.0011972f
#define S_IQ6K 1.f

namespace {
template <int block_size, typename Block, typename Block_repacked, int n_repack, typename Func, typename RepackFunc>
size_t quantize_repack(ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data, const Func& q_func, const RepackFunc& repack) {
    GGML_ASSERT(nrows%n_repack == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(type, n_per_row);
    std::vector<char> qtmp(n_repack*row_size);
    QHelper helper(imatrix, user_data, n_per_row, block_size);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += n_repack) {
        helper.quantize(n_repack, src, qtmp.data(), row_size, q_func);
        repack(n_repack, n_per_row, (const Block *)qtmp.data(), (Block_repacked *)qrow, false);
        src += n_repack*n_per_row;
        qrow += n_repack*row_size;
    }
    return nrows*row_size;
}
}
size_t quantize_iq1_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data *) {
    IQ1BNQuantizer iq1bn;
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_BN, n_per_row);
    auto qrow = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        iq1bn.quantize_one_row_1bn(src + row*n_per_row, (block_iq1_bn *)qrow, n_per_row, imatrix);
        qrow += row_size;
    }
    return nrows*row_size;
}
void quantize_row_iq1_bn_ref(const float * x, block_iq1_bn * y, int64_t k) {
    quantize_iq1_bn(x, y, 1, k, nullptr, nullptr);
}
// ROW-META FIX (fork, 2026-09-26; upstream ik_llama is identical and has the same defect). iq1_bn
// rows start with a 2-byte fp16 scale (row_meta_size = 2) written by quantize_one_row_1bn, and
// to_float receives the ROW START -- the same contract dequantize_row_q8_KV already honours. This
// read blocks from byte 0 and never applied the scale, so a from_float -> to_float round trip on
// EXACT ternary input returned 2496 of 4096 values wrong. Now: read the scale, skip it, scale.
// Contract: k is ONE row (row meta is per row, so a multi-row k cannot be decoded here).
void dequantize_row_iq1_bn(const block_iq1_bn * x, float * y, int64_t k) {
    assert(k%QK_IQ1BN == 0);
    int nblock = k / QK_IQ1BN;
    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq1_bn *)(dptr + 1);

    for (int i = 0; i < nblock; ++i) {
        uint8_t extra = x[i].extra;
        auto ql = x[i].ql;
        for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < 5; ++j) {
                    uint8_t v = ql[k]*IQ1BNQuantizer::k_mult[j];
                    int8_t vs = ((v + (v >> 1)) >> 7);
                    *y++ = d*(vs - 1);
                }
            }
            ql += 3;
            uint8_t v = extra*IQ1BNQuantizer::k_mult[i16];
            int8_t vs = ((v + (v >> 1)) >> 7);
            *y++ = d*(vs - 1);
        }
    }
}
size_t quantize_iq2_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    IQ1BNQuantizer iq1bn;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_BN, n_per_row);
    auto qrow = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        iq1bn.quantize_one_row_2bn(src + row*n_per_row, (block_iq2_bn *)qrow, n_per_row, imatrix);
        qrow += row_size;
    }
    return nrows*row_size;
}
void quantize_row_iq2_bn_ref(const float * x, block_iq2_bn * y, int64_t k) {
    quantize_iq2_bn(x, y, 1, k, nullptr, nullptr);
}
// ROW-META FIX, same defect as dequantize_row_iq1_bn above: iq2_bn rows start with a 4-byte float
// scale (row_meta_size = 4, written by quantize_one_row_2bn). Skip it and apply it. One row per call.
void dequantize_row_iq2_bn(const block_iq2_bn * x, float * y, int64_t k) {
    assert(k%QK_IQ1BN == 0);
    int nblock = k / QK_IQ1BN;
    const float * dptr = (const float *)x;
    const float d = *dptr;
    x = (const block_iq2_bn *)(dptr + 1);

    auto d1 = d, d2 = 0.25f*d, d3 = d2*0.25f, d4 = d3*0.25f;
    auto m = -d;
    constexpr int Nj = QK_IQ1BN/4;
    for (int i = 0; i < nblock; ++i) {
        for (int j = 0; j < Nj; ++j) {
            y[j+   0] = d1*(x[i].qs[j] & 0x03) + m;
            y[j+1*Nj] = d2*(x[i].qs[j] & 0x0c) + m;
            y[j+2*Nj] = d3*(x[i].qs[j] & 0x30) + m;
            y[j+3*Nj] = d4*(x[i].qs[j] & 0xc0) + m;
        }
        y += QK_IQ1BN;
    }
}
namespace {
void quantize_row_iq2_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {

    constexpr int kBlockSize = 16;

    block_iq2_k * y = (block_iq2_k *)vy;

    float scales[QK_K/kBlockSize];
    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];
    float sw[QK_K/kBlockSize];
    int8_t Ls[QK_K/kBlockSize];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    const int8_t * shifted_values = iq2nl_values + 4;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_K);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

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
            for (int i1 = 0; i1 < kBlockSize; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = i2; i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[0] + (sumx[i2] - sumx[i1])*iq2nl_values[1]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[2] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[0]*iq2nl_values[0] + (sumw[i2] - sumw[i1])*iq2nl_values[1]*iq2nl_values[1]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[2]*iq2nl_values[2] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[3]*iq2nl_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[0] + (sumx[i2] - sumx[i1])*shifted_values[1]
                              + (sumx[i3] - sumx[i2])*shifted_values[2] + (sumx[kBlockSize] - sumx[i3])*shifted_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[0]*shifted_values[0] + (sumw[i2] - sumw[i1])*shifted_values[1]*shifted_values[1]
                              + (sumw[i3] - sumw[i2])*shifted_values[2]*shifted_values[2] + (sumw[kBlockSize] - sumw[i3])*shifted_values[3]*shifted_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[3] + (sumx[i2] - sumx[i1])*iq2nl_values[2]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[1] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[3]*iq2nl_values[3] + (sumw[i2] - sumw[i1])*iq2nl_values[2]*iq2nl_values[2]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[1]*iq2nl_values[1] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[0]*iq2nl_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[3] + (sumx[i2] - sumx[i1])*shifted_values[2]
                              + (sumx[i3] - sumx[i2])*shifted_values[1] + (sumx[kBlockSize] - sumx[i3])*shifted_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[3]*shifted_values[3] + (sumw[i2] - sumw[i1])*shifted_values[2]*shifted_values[2]
                              + (sumw[i3] - sumw[i2])*shifted_values[1]*shifted_values[1] + (sumw[kBlockSize] - sumw[i3])*shifted_values[0]*shifted_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }

        if (!max_abs_scale) continue;
        float d = make_qx_quants(QK_K/kBlockSize, 8, scales, Ls, sw);
        if (!d) continue;

        //float d = -max_scale/8;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-8, std::min(7, ls));
            y[ibl].scales[ib/2] |= ((ls + 8) << 4*(ib%2));
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
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib32%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_K)*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}
}
void quantize_row_iq2_k_ref(const float * x, block_iq2_k  * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_k(x, (void *)y, 1, k, nullptr, nullptr);
}
size_t quantize_iq2_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    QHelper helper(imatrix, user_data, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq2_k_impl);
    return nrows * row_size;
}
void dequantize_row_iq2_k(const block_iq2_k  * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((x[i].scales[ib32] & 0xf) - 8);
            float dl2 = d * ((x[i].scales[ib32] >>  4) - 8);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] >> shift) & 3];
                y[j+16] = dl2 * values2[(qs[j+16] >> shift) & 3];
            }
            y += 32;
            shift += 2;
            if (shift == 8) { qs += 32; shift = 0; }
        }

    }

}
void dequantize_row_iq2_ks(const block_iq2_ks  * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq2_ks *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (((x[i].scales[ib64] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            float dl2 = d * (((x[i].scales[ib64] >>  4) | ((extra >> 5) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 32; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j] >> (shift+0)) & 3];
                y[j+32] = dl2 * values2[(qs[j] >> (shift+2)) & 3];
            }
            y += 64;
            shift += 4;
            if (shift == 8) { qs += 32; shift = 0; }
        }

    }

}
namespace {
const int8_t iq3nl_index[111] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
  9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
  4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
  6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};
inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}
// +3 zero bytes: the AVX2 search gathers 32-bit words at byte index <= nmap = 127, i.e. up to byte 130
static const int8_t k_simple_index_iq3nl[128 + 3] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,
    6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
};
void quantize_row_iq2_kl_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales) {
    constexpr int kBlockSize = 32;
    constexpr float kSigmaFactor = 2.25f;
    constexpr int ntry = 5;
    static const int k_index[64] = {-1, -2, 0, -3, -4, 1, -5, -6, 2, -7, -8, 3, -9, 4, -10, 5, -11, 6, 7, -12, 8, 9, 10, -13, 11, -14, -15, -16, 12, 13, -17,
        14, -18, -19, 15, 16, 17, 18, 19, -20, -21, 20, 21, 22, 23, 24, -22, -23, 25, -24, 26, -25, 27, -26, 28, 29, -27, -28, 30, -29, -30, 31, -31, -32};
    static const std::vector<std::vector<int>> k_neighbours = {
        { 2, 0, 6, 11, 7, 3, 8, 15,  },
        { 0, 2, 3, 6, 7, 1, 8, 4,  },
        { 0, 1, 3, 4, 8, 7, 9, 6,  },
        { 1, 0, 3, 4, 8, 9, 7, 10,  },
        { 1, 4, 5, 10, 9, 3, 8, 0,  },
        { 5, 1, 4, 10, 9, 14, 8, 3,  },
        { 6, 2, 7, 0, 3, 11, 8, 15,  },
        { 3, 7, 0, 6, 8, 4, 12, 9,  },
        { 3, 4, 8, 9, 1, 7, 12, 10,  },
        { 4, 10, 5, 9, 1, 8, 13, 14,  },
        { 11, 2, 6, 7, 20, 15, 25, 21,  },
        { 8, 7, 3, 12, 9, 16, 17, 13,  },
        { 14, 5, 10, 19, 9, 13, 4, 18,  },
        { 6, 15, 7, 11, 20, 21, 16, 2,  },
        { 15, 7, 16, 6, 21, 12, 17, 22,  },
        { 12, 16, 17, 8, 15, 7, 13, 22,  },
        { 19, 10, 13, 18, 14, 9, 12, 24,  },
        { 11, 20, 25, 6, 15, 2, 21, 7,  },
        { 20, 15, 21, 6, 11, 7, 16, 26,  },
        { 14, 19, 29, 10, 28, 18, 13, 24,  },
        { 25, 11, 20, 21, 15, 6, 26, 30,  },
        { 19, 24, 28, 18, 29, 23, 13, 17,  },
        { 29, 19, 14, 28, 24, 18, 10, 13,  },
        { 20, 26, 21, 25, 30, 15, 22, 16,  },
        { 27, 26, 22, 23, 21, 30, 16, 24,  },
        { 27, 24, 28, 31, 23, 18, 22, 17,  },
        { 25, 30, 20, 26, 21, 11, 15, 22,  },
        { 30, 26, 25, 20, 21, 27, 22, 15,  },
        { 30, 27, 31, 26, 22, 23, 21, 24,  },
        { 31, 27, 30, 26, 28, 23, 22, 24,  },
        { 31, 28, 29, 27, 24, 23, 19, 18,  },
        { 29, 28, 31, 24, 19, 27, 14, 18,  },
    };
    auto values = iq3nl_values;
    std::pair<int8_t, int8_t> grid[32];
    for (int j = 0; j < 64; ++j) {
        if (int i = k_index[j]; i >= 0) {
            int i1 = j/8, i2 = j%8;
            grid[i] = {values[i1], values[i2]};
        }
    }

    ggml_half * dptr = (ggml_half *)vy;
    auto y = (block_iq2_kl *)(dptr + 1);

    float weight[kBlockSize];

    auto index = [&grid, values] (float id, float x1, float x2, float w1, float w2) {
        float sx1 = id*x1;
        float sx2 = id*x2;
        int l1 = best_index_iq3nl(values, sx1);
        int l2 = best_index_iq3nl(values, sx2);
        int i = k_index[8*l1 + l2];
        if (i >= 0) return i;
        auto& neigh = k_neighbours[-i-1];
        float best = std::numeric_limits<float>::max();
        int ibest = -1;
        for (auto& n : neigh) {
            float diff1 = grid[n].first  - sx1;
            float diff2 = grid[n].second - sx2;
            float score = w1*diff1*diff1 + w2*diff2*diff2;
            if (score < best) {
                best = score; ibest = n;
            }
        }
        GGML_ASSERT(ibest >= 0);
        return ibest;
    };

    float max_scale = 0, max_abs_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_KL);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq2_kl));
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
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
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < kBlockSize; j += 2) {
                float w1 = weight[j+0];
                float w2 = weight[j+1];
                int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                float q1 = grid[idx].first ;
                float q2 = grid[idx].second;
                sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_p += w1*q1*q1 + w2*q2*q2;
                idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                q1 = grid[idx].first ;
                q2 = grid[idx].second;
                sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_m += w1*q1*q1 + w2*q2*q2;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; j += 2) {
                    float w1 = weight[j+0];
                    float w2 = weight[j+1];
                    int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                    float q1 = grid[idx].first ;
                    float q2 = grid[idx].second;
                    sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_p += w1*q1*q1 + w2*q2*q2;
                    idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                    q1 = grid[idx].first ;
                    q2 = grid[idx].second;
                    sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_m += w1*q1*q1 + w2*q2*q2;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m;
                }
            }
            scales[ib] = d;
            float ad = std::abs(d);
            if (ad > max_abs_scale) {
                max_abs_scale = ad; max_scale = d;
            }
        }
    }

    if (!max_abs_scale) {
        dptr[0] = GGML_FP32_TO_FP16(0.f);
        return;
    }

    float d = -max_scale/32;
    float id = 1/d;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-32, std::min(31, ls));
            int lsmin = std::max(-32, ls-1);
            int lsmax = std::min( 31, ls+1);
            float best_score = std::numeric_limits<float>::max();
            int best_ls = ls;
            for (int ils = lsmin; ils <= lsmax; ++ils) {
                float dl = d*ils;
                float idl = dl ? 1/dl : 0.f;
                float score = 0;
                for (int j = 0; j < kBlockSize/2; ++j) {
                    float w1 = weight[2*j+0];
                    float w2 = weight[2*j+1];
                    int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                    float diff1 = dl*grid[idx].first  - xb[2*j+0];
                    float diff2 = dl*grid[idx].second - xb[2*j+1];
                    score += w1*diff1*diff1 + w2*diff2*diff2;
                }
                if (score < best_score) {
                    best_score = score;
                    best_ls = ils;
                }
            }
            ls = best_ls;
            int uls = ls + 32;
            y[ibl].scales_l[ib%4] |= ((uls & 0xf) << 4*(ib/4));
            y[ibl].scales_h |= ((uls >> 4) << 2*ib);
            if (ls == 0) continue;
            float dl = d*ls;
            float idl = 1/dl;
            for (int j = 0; j < kBlockSize/2; ++j) {
                float w1 = weight[2*j+0];
                float w2 = weight[2*j+1];
                int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                y[ibl].qs[16*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float q1 = ls*grid[idx].first ;
                float q2 = ls*grid[idx].second;
                sumqx += w1*q1*xb[2*j] + w2*q2*xb[2*j+1];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) d = sumqx/sumq2;

    dptr[0] = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ2_KL) * d);

}
}
void quantize_row_iq2_kl_ref(const float * x, block_iq2_kl * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kl(x, (void *)y, 1, k, nullptr, nullptr);
}
size_t quantize_iq2_kl(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KL, n_per_row);
    int nblock = n_per_row/QK_K;
    std::vector<float> all_scales(nblock*(QK_K/kBlockSize));
    auto q_func = [&all_scales] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq2_kl_impl(x, vy, n_per_row, imatrix, all_scales.data());
    };
    QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void dequantize_row_iq2_kl(const block_iq2_kl  * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq2_kl *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        auto qs = x[i].qs;
        auto qh = x[i].qh;
        auto scales_h = x[i].scales_h;

        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (int(((x[i].scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+0)) & 3) << 4)) - 32);
            float dl2 = d * (int(((x[i].scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+2)) & 3) << 4)) - 32);
            for (int j = 0; j < 16; ++j) {
                const int8_t * val1 = (const int8_t *)(iq2kl_values + ((qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)));
                const int8_t * val2 = (const int8_t *)(iq2kl_values + ((qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)));
                y[2*j+ 0] = dl1 * val1[0];
                y[2*j+ 1] = dl1 * val1[1];
                y[2*j+32] = dl2 * val2[0];
                y[2*j+33] = dl2 * val2[1];
            }
            y  += 64;
            qs += 16;
        }

    }
}
namespace {
static void quantize_row_iq3_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {

    constexpr int ntry = 3;

    // was `#ifdef __AVX2__`, dead in this arch-neutral TU: the AVX2 search now comes from ggml-cpu at run time
    const iqk_scale_search_fn search = iqk_get_scale_search(IQK_SCALE_SEARCH_16_127);
    float f_values[16];
    for (int j = 0; j < 16; ++j) f_values[j] = iq3nl_values[j];

    block_iq3_k * y = (block_iq3_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    const int8_t * shifted_values = iq3nl_values + 8;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_K);

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

        if (search) {
        for (int ib = 0; ib < QK_K/16; ++ib) {
            auto xb = xbl + ib*16;
            auto [d, _, is_shifted] = search(8, iq3nl_values, f_values, k_simple_index_iq3nl, xb,
                    quant_weights ? quant_weights + QK_K*ibl + 16*ib : nullptr, sigma2, ntry, true, 2.0f, false);

            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }
        } else {
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq3nl_values[0] : max/iq3nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(iq3nl_values, al);
                float q = iq3nl_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(iq3nl_values, -al);
                q = iq3nl_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (2*itry + iq3nl_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(iq3nl_values, al);
                    float q = iq3nl_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(iq3nl_values, -al);
                    q = iq3nl_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (2*itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
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
            }
            if (!d) {
                scales[ib] = 0; continue;
            }

            scales[ib] = d;

            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = MAX(max_abs_scale, abs_scale);
        }
        }

        if (!max_abs_scale) continue;

        float d = max_abs_scale/31;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(0.5f*(id*fabsf(scales[ib])-1));
            ls = MAX(0, MIN(15, ls));
            y[ibl].scales_l[ib/2] |= (ls << 4*(ib%2));
            if (scales[ib] < 0) y[ibl].scales_h |= (1 << ib);
            ls = (2*ls + 1) * (scales[ib] < 0 ? -1 : 1);
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq3nl_values;
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
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq3nl(block_values, al);
                    qs[j] |= ((ibest &  3) << 2*(ib32%4));
                    qh[j] |= ((ibest >> 2) << (ib32%8));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_K)*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}
}
void quantize_row_iq3_k_ref(const float * x, block_iq3_k * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_k(x, (void *)y, 1, k, nullptr, nullptr);
}
size_t quantize_iq3_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    QHelper helper(imatrix, user_data, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq3_k_impl);
    return nrows * row_size;
}
void dequantize_row_iq3_k(const block_iq3_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        uint16_t sh = x[i].scales_h;
        uint16_t extra = x[i].extra;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((2*(x[i].scales_l[ib32] & 0xf) + 1) * ((sh & 1) ? -1 : 1));
            float dl2 = d * ((2*(x[i].scales_l[ib32] >>  4) + 1) * ((sh & 2) ? -1 : 1));
            sh >>= 2;
            const int8_t * values1 = extra & 1 ? iq3nl_values + 8 : iq3nl_values;
            const int8_t * values2 = extra & 2 ? iq3nl_values + 8 : iq3nl_values;
            extra >>= 2;
            int shift_l = 2*(ib32%4);
            int shift_h = ib32%8;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2)];
                y[j+16] = dl2 * values2[((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2)];
            }
            y += 32;
            if (shift_l == 6) qs += 32;
        }

    }
}
namespace {
static void quantize_row_iq3_ks_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    const iqk_scale_search_fn search = iqk_get_scale_search(IQK_SCALE_SEARCH_32_127);
    float f_values[16];
    for (int j = 0; j < 16; ++j) f_values[j] = values[j];

    ggml_half * dptr = (ggml_half *)cy;
    block_iq3_ks * y = (block_iq3_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 8;

    float amax_scale = 0;
    float max_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_KS);

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq3_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        if (search) {
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            auto [d, _, is_shifted] = search(8, values, f_values, k_simple_index_iq3nl, xbl + ib*block_size,
                    quant_weights ? quant_weights + super_block_size*ibl + block_size*ib : nullptr, sigma2, ntry, false, 1.0f, false);
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
        } else
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
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(values, -al);
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
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
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
            }
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
    }
    float d = -max_scale/16;
    *dptr = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_KS)*d);
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
            const int8_t * block_values = (y[ibl].extra >> (8 + ib)) & 0x01 ? shifted_values : values;
            int l = nearest_int(id*scales[ib]);
            l = std::max(-16, std::min(15, l));
            uint8_t ul = l + 16;
            y[ibl].scales[ib%4] |= (ul & 0xf) << 4*(ib/4);
            y[ibl].extra |= (ul >> 4) << ib;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + (ib/4)*block_size;
            auto qh = y[ibl].qh + (ib/8)*block_size;
            for (int j = 0; j < block_size; ++j) {
                uint8_t i = best_index_iq3nl(block_values, idl*xb[j]);
                qs[j] |= ((i &  3) << 2*(ib%4));
                qh[j] |= ((i >> 2) << (ib%8));
                float w = weight[j];
                float q = block_values[i]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ3_KS)*sumqx/sumq2);
}
}
void quantize_row_iq3_ks_ref(const float * x, block_iq3_ks * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr, nullptr);
}
size_t quantize_iq3_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_KS, n_per_row);
    QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq3_ks_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq3nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void dequantize_row_iq3_ks(const block_iq3_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32;
    static_assert(QK_K/kBlockSize == 8);
    GGML_ASSERT(k%QK_K == 0);
    const ggml_half * dptr = (const ggml_half *)x;
    float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq3_ks *)(dptr + 1);
    float dl[8];
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int j = 0; j < 4; ++j) {
            int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j+0)) & 1) << 4);
            int ls2 = (x[ibl].scales[j] >>  4) | (((x[ibl].extra >> (j+4)) & 1) << 4);
            dl[j+0] = d*(ls1 - 16);
            dl[j+4] = d*(ls2 - 16);
        }
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            for (int ib = 0; ib < 4; ++ib) {
                const int8_t * values = iq3nl_values + ((x[ibl].extra >> (8 + (4*i128+ib)) & 1) << 3);
                for (int j = 0; j < kBlockSize; ++j) {
                    y[j] = dl[4*i128 + ib] * values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)];
                }
                y += kBlockSize;
            }
            qs += kBlockSize;
        }
    }
}
//

//
// ============================================== iq3_ks_r16
//
static void quantize_row_iq3ks_r16_impl(int n_per_row, const float * x, char * band, int krow,
        float * all_scales, float * weight, const float * quant_weights) {

    constexpr int kBlockSize = 32;
    const int nbl = n_per_row/kBlockSize;
    float * dptr = (float *)band;
    auto * blocks = (block_iq3_ks_r16 *)(dptr + 16);

    const int8_t * values = iq3nl_values;
    const int8_t * shifted_values = values + 8;
    const int ntry = 5;

    float amax_scale = 0;
    float max_scale = 0;

    for (int ib0 = 0; ib0 < nbl; ib0 += 4) {
        const int nwin = std::min(4, nbl - ib0);
        float sigma2 = 0;
        for (int j = 0; j < nwin*kBlockSize; ++j) {
            float v = x[ib0*kBlockSize + j];
            sigma2 += v*v;
        }
        sigma2 *= 2.f/(nwin*kBlockSize);
        for (int ib = ib0; ib < ib0 + nwin; ++ib) {
            const float * xb = x + ib*kBlockSize;
            if (quant_weights) {
                const float * qw = quant_weights + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) { amax = ax; max = xb[j]; }
            }
            if (amax < 1e-16f) { all_scales[ib] = 0; continue; }
            float d = -max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) { d = sumqx_p/sumq2_p; best = d*sumqx_p; }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = true;
                }
            }
            if (is_shifted) blocks[ib].extra |= (1u << (16 + krow));
            all_scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) { amax_scale = ascale; max_scale = d; }
        }
    }
    float d = -max_scale/16;
    if (!d) { dptr[krow] = 0.f; return; }
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;
    for (int ib0 = 0; ib0 < nbl; ib0 += 4) {
        const int nwin = std::min(4, nbl - ib0);
        float sigma2 = 0;
        for (int j = 0; j < nwin*kBlockSize; ++j) {
            float v = x[ib0*kBlockSize + j];
            sigma2 += v*v;
        }
        sigma2 *= 2.f/(nwin*kBlockSize);
        for (int ib = ib0; ib < ib0 + nwin; ++ib) {
            const int8_t * block_values = (blocks[ib].extra >> (16 + krow)) & 1 ? shifted_values : values;
            int l = nearest_int(id*all_scales[ib]);
            l = std::max(-16, std::min(15, l));
            uint8_t ul = l + 16;
            blocks[ib].scales[krow & 7] |= (uint8_t)((ul & 0xf) << (4*(krow >> 3)));
            blocks[ib].extra |= (uint32_t(ul >> 4) & 1) << krow;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = x + ib*kBlockSize;
            if (quant_weights) {
                const float * qw = quant_weights + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = xb[j]*xb[j];
            }
            for (int j = 0; j < kBlockSize; ++j) {
                uint8_t i = best_index_iq3nl(block_values, idl*xb[j]);
                const int byte = krow*4 + (j & 3) + (j >= 16 ? 64 : 0);
                const int pair = (j >> 2) & 3;
                blocks[ib].qs[byte] |= (uint8_t)((i & 3) << (2*pair));
                const int qhbyte = krow*4 + (j & 3);
                blocks[ib].qh[qhbyte] |= (uint8_t)((i >> 2) << (j >> 2));
                float w = weight[j];
                float q = block_values[i]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    dptr[krow] = sumq2 > 0 ? sumqx/sumq2 : d;
}

size_t quantize_iq3ks_r16(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%32 == 0);
    constexpr int kBlockSize = 32;
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_KS_R16, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
    char * qcur = (char *)dst;
    for (int64_t row = 0; row < nrows; row += 16) {
        std::memset(qcur, 0, 16*row_size);
        for (int k = 0; k < 16; ++k) {
            const float * x = src + k*n_per_row;
            quantize_row_iq3ks_r16_impl(n_per_row, x, qcur, k, all_scales.data(), weight, helper.row_weights(x));
        }
        qcur += 16*row_size;
        src += 16*n_per_row;
    }
    return nrows*row_size;
}

void quantize_row_iq3ks_r16_ref(const float * x, block_iq3_ks_r16 * y, int64_t k) {
    quantize_iq3ks_r16(x, (void *)y, 16, k/16, nullptr, nullptr);
}

void quantize_row_iq3ks_r16(const float * x, void * y, int64_t k) {
    quantize_iq3ks_r16(x, y, 16, k/16, nullptr, nullptr);
}

void dequantize_row_iq3ks_r16(const block_iq3_ks_r16 * x, float * y, int64_t k) {
    auto n_per_row = k/16;
    GGML_ASSERT(n_per_row%32 == 0);
    float * y16[16];
    for (int k2 = 0; k2 < 16; ++k2) y16[k2] = y + n_per_row*k2;
    const float * dptr = (const float *)x;
    x = (const block_iq3_ks_r16 *)(dptr + 16);
    const int nbl = n_per_row/32;
    for (int ib = 0; ib < nbl; ++ib) {
        for (int k2 = 0; k2 < 16; ++k2) {
            const float d = dptr[k2];
            const int ul = ((x[ib].scales[k2 & 7] >> (4*(k2 >> 3))) & 0xf) | (((x[ib].extra >> k2) & 1) << 4);
            const float dl = d * (ul - 16);
            const int8_t * values = iq3nl_values + (((x[ib].extra >> (16 + k2)) & 1) << 3);
            for (int c = 0; c < 32; ++c) {
                const int byte = k2*4 + (c & 3) + (c >= 16 ? 64 : 0);
                const int pair = (c >> 2) & 3;
                const int qhbyte = k2*4 + (c & 3);
                const int idx = ((x[ib].qs[byte] >> (2*pair)) & 3) | (((x[ib].qh[qhbyte] >> (c >> 2)) & 1) << 2);
                y16[k2][32*ib + c] = dl * values[idx];
            }
        }
    }
}

void vec_dot_iq3ks_r16_q8_2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_KS_R16, vx, 0, GGML_TYPE_Q8_0_X4, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%32 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);

    const float * dptr = (const float *)vx;
    const block_iq3_ks_r16 * blocks = (const block_iq3_ks_r16 *)(dptr + 16);
    const int nbl = n/32;

    float sumf = 0;
    for (int ib = 0; ib < nbl; ++ib) {
        const block_q8_0_x4 * y4 = (const block_q8_0_x4 *)vy + ib/4;
        const int g = ib & 3;
        const int8_t * qs8 = y4->qs + 32*g;
        const float yd = GGML_FP16_TO_FP32(y4->d[g]);
        for (int k = 0; k < 16; ++k) {
            const float d = dptr[k];
            const int ul = ((blocks[ib].scales[k & 7] >> (4*(k >> 3))) & 0xf) | (((blocks[ib].extra >> k) & 1) << 4);
            const float dl = d * (ul - 16);
            const int8_t * values = iq3nl_values + (((blocks[ib].extra >> (16 + k)) & 1) << 3);
            int sumi = 0;
            for (int c = 0; c < 32; ++c) {
                const int byte = k*4 + (c & 3) + (c >= 16 ? 64 : 0);
                const int pair = (c >> 2) & 3;
                const int qhbyte = k*4 + (c & 3);
                const int idx = ((blocks[ib].qs[byte] >> (2*pair)) & 3) | (((blocks[ib].qh[qhbyte] >> (c >> 2)) & 1) << 2);
                int yb = (int)qs8[c]; if (yb < -127) yb = -127;
                sumi += (int)values[idx] * yb;
            }
            sumf += dl * yd * (float)sumi;
        }
    }
    *s = sumf;
}

// ============================================== iq4_K
//
void dequantize_row_iq4_k(const block_iq4_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint16_t extra = x[i].extra;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
            const float dl1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const float dl2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
            const int8_t * values1 = extra & 1 ? iq4k_values + 16 : iq4k_values;
            const int8_t * values2 = extra & 2 ? iq4k_values + 16 : iq4k_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[qs[j] & 0xf];
                y[j+16] = dl2 * values2[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}
//
// ============================================== iq5_K
//
void dequantize_row_iq5_k(const block_iq5_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            float dl1 = d * (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
            float dl2 = d * (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
            float dl3 = d * (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
            float dl4 = d * (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
            const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
            const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
            const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
                y[j+16] = dl2 * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
                y[j+32] = dl3 * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
                y[j+48] = dl4 * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 2;
            if (shift == 8) { qh += 32; shift = 0; }
        }

    }
}
namespace {
// iq5nl_index / best_index_iq5nl moved VERBATIM to iqk_quantize_shared.h: the AVX2 bulk IQ5_K/IQ5_KS quantizer in
// iqk_quantize.cpp (ggml-cpu) needs the same nearest-value search for its final encode pass.
// k_simple_index_iq5nl (ik #2671/#2682 scale search + iqk_ks_simple_index) stays here: only this file reads it.
static const int8_t k_simple_index_iq5nl[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
    6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9,10,10,10,10,10,10,
    11,11,11,11,11,11,12,12,12,12,12,12,13,13,13,13,13,13,14,14,14,14,14,14,15,15,15,15,15,16,16,16,
    16,16,16,17,17,17,17,17,17,18,18,18,18,18,18,19,19,19,19,19,19,20,20,20,20,20,20,21,21,21,21,21,
    21,21,22,22,22,22,22,22,22,23,23,23,23,23,23,23,23,24,24,24,24,24,24,24,24,25,25,25,25,25,25,25,
    25,25,26,26,26,26,26,26,26,26,26,27,27,27,27,27,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,29,
    29,29,29,29,29,29,29,29,29,29,29,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
};
void quantize_row_iq5_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights,
        [[maybe_unused]] const quantize_user_data * user_data) {
    const int ntry = 5;

    block_iq5_k * y = (block_iq5_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    const iqk_scale_search_fn search = iqk_get_scale_search(IQK_SCALE_SEARCH_16_240);
    float f_values[64];
    for (int j = 0; j < 64; ++j) f_values[j] = iq5nl_values[j];
    const float step = 1.f;

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

        if (search) {
        for (int ib = 0; ib < QK_K/16; ++ib) {
            auto [d, _, is_shifted] = search(32, iq5nl_values, f_values, k_simple_index_iq5nl, xbl + ib*16,
                    quant_weights ? quant_weights + QK_K*ibl + 16*ib : nullptr, sigma2, ntry, false, 1.0f, false); //, true);
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }
        }
        } else
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq5nl_values[0] : max/iq5nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq5nl(iq5nl_values, al);
                float q = iq5nl_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq5nl(iq5nl_values, -al);
                q = iq5nl_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry*step + iq5nl_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(iq5nl_values, al);
                    float q = iq5nl_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(iq5nl_values, -al);
                    q = iq5nl_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry*step + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(shifted_values, -al);
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
            }
            if (d) {
                const int8_t * block_values = is_shifted ? shifted_values : iq5nl_values;
                float sumqx = 0, sumq2 = 0;
                id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(block_values, al);
                    float q = block_values[l];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
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
        y[ibl].d = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ5_K)*d);
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
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ5_K)*sumqx/sumq2);

    }

}
}
void quantize_row_iq5_k_ref(const float * x, block_iq5_k * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq5_k(x, (void *)y, 1, k, nullptr, nullptr);
}
size_t quantize_iq5_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    QHelper helper(imatrix, user_data, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq5_k_impl);
    return nrows * row_size;
}
void dequantize_row_iq6_k(const block_iq6_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sl = x[i].scales;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            float dl1 = d * sl[4*ib64 + 0];
            float dl2 = d * sl[4*ib64 + 1];
            float dl3 = d * sl[4*ib64 + 2];
            float dl4 = d * sl[4*ib64 + 3];
            float m1 = extra & 1 ? S_IQ6K : 0;
            float m2 = extra & 2 ? S_IQ6K : 0;
            float m3 = extra & 4 ? S_IQ6K : 0;
            float m4 = extra & 8 ? S_IQ6K : 0;
            for (int j = 0; j < 16; ++j) {
                float q1 = ((qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 0x03) << 4));
                float q2 = ((qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 0x03) << 4));
                float q3 = ((qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 0x0c) << 2));
                float q4 = ((qs[j+16] >>  4) | (((qh[j+16] >> shift) & 0x0c) << 2));
                y[j+ 0] = dl1 * (A_IQ6K + q1*(B_IQ6K + q1*(-C_IQ6K + q1*D_IQ6K)) + m1);
                y[j+16] = dl2 * (A_IQ6K + q2*(B_IQ6K + q2*(-C_IQ6K + q2*D_IQ6K)) + m2);
                y[j+32] = dl3 * (A_IQ6K + q3*(B_IQ6K + q3*(-C_IQ6K + q3*D_IQ6K)) + m3);
                y[j+48] = dl4 * (A_IQ6K + q4*(B_IQ6K + q4*(-C_IQ6K + q4*D_IQ6K)) + m4);
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 4;
            if (shift == 8) { qh += 32; shift = 0; }
        }

    }
}
namespace {
static const uint8_t iq6nl_index[249] = {
   0,   0,   0,  64,   1,   1,   1,   1,   1,  65,   2,   2,   2,   2,   2,  66,   3,   3,   3,   3,  67,  67,   4,   4,   4,   4,  68,   5,   5,   5,   5,  69,
  69,   6,   6,   6,  70,  70,   7,   7,   7,  71,   8,   8,   8,  72,  72,   9,   9,   9,  73,  73,  10,  10,  10,  74,  11,  11,  11,  75,  12,  12,  12,  76,
  13,  13,  13,  77,  14,  14,  14,  78,  15,  15,  79,  79,  16,  16,  80,  17,  17,  81,  81,  18,  18,  82,  19,  19,  83,  83,  20,  84,  84,  21,  85,  85,
  22,  86,  86,  23,  87,  87,  24,  88,  88,  25,  89,  89,  26,  90,  90,  27,  91,  91,  28,  92,  29,  93,  93,  30,  94,  94,  31,  95,  95,  32,  96,  33,
  97,  97,  34,  98,  98,  35,  99,  99,  36, 100, 100,  37, 101,  38, 102, 102,  39, 103, 103,  40, 104, 104,  41,  41, 105,  42,  42, 106, 106,  43, 107, 107,
  44, 108, 108,  45,  45, 109,  46,  46,  46, 110,  47,  47, 111, 111,  48,  48, 112,  49,  49,  49, 113,  50,  50,  50, 114,  51,  51,  51, 115,  52,  52,  52,
 116, 116,  53,  53,  53, 117,  54,  54,  54, 118, 118,  55,  55,  55, 119, 119,  56,  56,  56, 120, 120,  57,  57,  57, 121, 121,  58,  58,  58,  58, 122,  59,
  59,  59,  59, 123, 123,  60,  60,  60,  60, 124,  61,  61,  61,  61,  61, 125,  62,  62,  62,  62,  62, 126,  63,  63, 63,
};
inline int best_index_iq6nl(const float * values, float x) {
    int ix = (int)(x - values[0]);
    if (ix < 0 || ix >= 249) return ix < 0 ? 0 : 63;
    ix = iq6nl_index[ix];
    return ix < 64 ? ix : x - values[ix-64] < values[ix-63] - x ? ix-64 : ix-63;
}
// ik 379ede410 (#2619): index table for the AVX2 IQ6_K scale search (find_best_scale_avx2, registered from
// ggml-cpu through iqk_scale_search.h); unconditional now that the search is chosen at run time
static int8_t k_simple_index_iq6nl[256] = {
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9,10,10,10,10,11,11,11,11,12,12,12,
    12,13,13,13,13,14,14,14,14,15,15,15,16,16,16,16,17,17,17,18,18,18,18,19,19,19,20,20,20,21,21,21,
    22,22,22,23,23,23,24,24,24,25,25,25,26,26,26,27,27,27,28,28,29,29,29,30,30,30,31,31,31,32,32,33,
    33,33,34,34,34,35,35,35,36,36,36,37,37,38,38,38,39,39,39,40,40,40,41,41,41,41,42,42,42,43,43,43,
    44,44,44,45,45,45,45,46,46,46,46,47,47,47,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,52,52,
    52,52,52,53,53,53,53,54,54,54,54,54,55,55,55,55,55,56,56,56,56,56,57,57,57,57,57,58,58,58,58,58,
    58,59,59,59,59,59,60,60,60,60,60,60,61,61,61,61,61,61,62,62,62,62,62,62,63,63,63,63,63,63,63,63,
};
void quantize_row_iq6_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, const float * values, const float * shifted_values) {
    const int ntry = 5;

    block_iq6_k * y = (block_iq6_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ6_K);

    const iqk_scale_search_fn search = iqk_get_scale_search(IQK_SCALE_SEARCH_16_248);
    float f_values[128];
    for (int j = 0; j < 128; ++j) f_values[j] = iq6nl_values[j];
    const float step = 1.f;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq6_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 2*sumx2/QK_K;

        float max_scale = 0, max_abs_scale = 0;
        uint16_t extra = 0;

        if (search) {
        for (int ib = 0; ib < QK_K/16; ++ib) {
            auto [d, _, is_shifted] = search(64, iq6nl_values, f_values, k_simple_index_iq6nl, xbl + ib*16,
                    quant_weights ? quant_weights + QK_K*ibl + 16*ib : nullptr, sigma2, ntry, false, 1.0f, false); // , true, 1.0f, true) -> corresponds to not AVX2, but not really better.
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }
        }
        } else
        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
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
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq6nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq6nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry*step + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq6nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq6nl(values, -al);
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
                id = (itry*step + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq6nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq6nl(shifted_values, -al);
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
            }
            if (d) {
                const float * block_values = is_shifted ? shifted_values : values;
                float sumqx = 0, sumq2 = 0;
                id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq6nl(block_values, al);
                    float q = block_values[l];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }

        }

        if (!max_abs_scale) continue;
        float d = -max_scale/127;
        y[ibl].d = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ6_K)*d);
        y[ibl].extra = extra;

        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = MAX(-127, MIN(127, ls));
            y[ibl].scales[ib] |= ls;
            float dl = d * ls;
            if (dl) {
                const float * block_values = y[ibl].extra & (1 << ib) ? shifted_values : values;
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
                uint8_t * qh = y[ibl].qh + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    //int ibest = best_index(64, block_values, al);
                    int ibest = best_index_iq6nl(block_values, al);
                    qs[j] |= ((ibest & 0xf) << 4*(ib32%2));
                    qh[j] |= ((ibest >>  4) << 2*(ib32%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(ggml_get_quantize_fudge_factor(GGML_TYPE_IQ6_K)*sumqx/sumq2);

    }
}
}
void quantize_row_iq6_k_ref(const float * x, block_iq6_k * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq6_k(x, (void *)y, 1, k, nullptr, nullptr);
}
size_t quantize_iq6_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    float values[128];
    for (int i = 0; i < 64; ++i) {
        values[i] = iq6nl_values[i];
        values[i+64] = values[i] + S_IQ6K;
    }
    auto q_func = [values] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq6_k_impl(x, vy, n_per_row, imatrix, values, values + 64);
    };
    auto row_size = ggml_row_size(GGML_TYPE_IQ6_K, n_per_row);
    QHelper helper(imatrix, user_data, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
// The reference / from_float entries keep the OCP floor rule (no search): they are the fixed point other code compares against.
void quantize_row_mxfp4_ref(const float * x, block_mxfp4 * y, int64_t k) {
    float weight[QK_MXFP4];
    quantize_row_mxfp4_impl(k, x, (char *)y, weight, kvalues_mxfp4, nullptr, 7, false);
}
void dequantize_row_mxfp4(const block_mxfp4 * x, float * y, int64_t k) {
    constexpr int kBlockSize = QK_MXFP4;
    GGML_ASSERT(k%kBlockSize == 0);
    int nblock = k/kBlockSize;
    for (int ib = 0; ib < nblock; ++ib) {
        float d = GGML_E8M0_TO_FP32_HALF(x[ib].e);
        for (int j = 0; j < kBlockSize/2; ++j) {
            y[j             ] = d * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            y[j+kBlockSize/2] = d * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        y  += kBlockSize;
    }
}
void quantize_row_mxfp4_r8_ref(const float * x, block_mxfp4_r8 * y, int64_t k) {
    quantize_mxfp4_r8(x, (void *)y, 8, k/8, nullptr, nullptr);
}
size_t quantize_mxfp4_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
         [[maybe_unused]] const float * imatrix,
         [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows % 8 == 0);
    constexpr int kBlockSize = QK_MXFP4;
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    auto row_size = ggml_row_size(GGML_TYPE_MXFP4, n_per_row);

    block_mxfp4_r8 * y = (block_mxfp4_r8 *)dst;

    int nblock = n_per_row/QK_MXFP4;

    // Each row is quantised by the SAME routine as plain MXFP4 (quantize_row_mxfp4_impl, block-scale code search) and
    // only then interleaved 8 rows wide, so the _r8 type can never quantise differently from its base type
    // (probe-interleaved-vs-base-mulmat flagged mxfp4_r8 at ratio 1.057 once the base got the search and this loop kept
    // its own floor(log2(amax)) rule). The packing is identical: low nibble = element j, high nibble = element j+16.
    std::vector<block_mxfp4> tmp(nblock);
    float weight[QK_MXFP4];
    for (int row = 0; row < nrows; row += 8) {
        for (int ib = 0; ib < nblock; ++ib) {
            memset(&y[ib], 0, sizeof(block_mxfp4_r8));
        }
        for (int k = 0; k < 8; ++k) {
            quantize_row_mxfp4_impl(n_per_row, src + (row + k)*n_per_row, (char *)tmp.data(), weight, kvalues_mxfp4, imatrix, 7, true);
            for (int ib = 0; ib < nblock; ++ib) {
                y[ib].e[k] = tmp[ib].e;
                for (int j1 = 0; j1 < QK_MXFP4/8; ++j1) {
                    for (int j2 = 0; j2 < 4; ++j2) {
                        // for each j1 we have 4 values per row with 8 interleaved rows
                        y[ib].qs[32*j1 + 4*k + j2] = tmp[ib].qs[4*j1 + j2];
                    }
                }
            }
        }
        y += nblock;
    }

    return nrows * row_size;
}
void dequantize_row_mxfp4_r8(const block_mxfp4_r8 * x, float * y, int64_t k) {
    constexpr int kBlockSize = QK_MXFP4;
    int n_per_row = k/8;
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    int nblock = n_per_row/kBlockSize;   // one block_mxfp4_r8 covers 8 rows: blocks per ROW, not per k
    float d[8];
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    float * y8[8];
    for (int k = 0; k < 8; ++k) y8[k] = y + k*n_per_row;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 8; ++k) d[k] = GGML_E8M0_TO_FP32_HALF(x[ib].e[k]);
        auto qs = (const uint32_t *)x[ib].qs;
        for (int j = 0; j < kBlockSize/8; ++j) {
            for (int k = 0; k < 8; ++k) {
                aux32[0] = qs[8*j+k] & 0x0f0f0f0f;
                aux32[1] = (qs[8*j+k] >> 4) & 0x0f0f0f0f;
                for (int i = 0; i < 4; ++i) {
                    y8[k][kBlockSize*ib + 4*j + i               ] = d[k] * kvalues_mxfp4[aux8[i+0]];
                    y8[k][kBlockSize*ib + 4*j + i + kBlockSize/2] = d[k] * kvalues_mxfp4[aux8[i+4]];
                }
            }
        }
    }
}
void dequantize_row_iq4_ks(const block_iq4_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32; //128;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq4_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            float dl = d * ((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < kBlockSize/2; ++j) {
                y[j             ] = dl * values[qs[j] & 0xf];
                y[j+kBlockSize/2] = dl * values[qs[j] >>  4];
            }
            y  += kBlockSize;
            qs += kBlockSize/2;
        }
    }
}
namespace {
static void quantize_row_iq5_ks_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    const iqk_scale_search_fn search = iqk_get_scale_search(IQK_SCALE_SEARCH_32_240);
    float f_values[64];
    for (int j = 0; j < 64; ++j) f_values[j] = values[j];
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
        if (search) {
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            auto [d, _, is_shifted] = search(32, values, f_values, k_simple_index_iq5nl, xbl + ib*block_size,
                    quant_weights ? quant_weights + super_block_size*ibl + block_size*ib : nullptr, sigma2, ntry, false, 1.0f, false);
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
        } else
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
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq5nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq5nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            bool is_shifted = false;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(values, -al);
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
                    int l = best_index_iq5nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(shifted_values, -al);
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
            }
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ5_KS)*d;
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
    if (sumq2 > 0) *dptr = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ5_KS)*sumqx/sumq2;
}
}
void quantize_row_iq5_ks_ref(const float * x, block_iq5_ks * y, int64_t k) {
    quantize_iq5_ks(x, (void *)y, 1, k, nullptr, nullptr);
}
size_t quantize_iq5_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq5_ks_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq5nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void dequantize_row_iq5_ks(const block_iq5_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq5_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            float dl1 = d * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            float dl2 = d * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * values1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            for (int j = 0; j < kBlockSize; ++j) {
                y[j           ] = dl1 * values1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                y[j+kBlockSize] = dl2 * values2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
            }
            y  += 2*kBlockSize;
            qs += kBlockSize;
        }
    }
}
namespace {
const uint16_t * scramble_table() {
    static std::mutex mutex;
    static std::vector<uint16_t> table;
    std::lock_guard<std::mutex> lock(mutex);
    if (table.empty()) {
        table.resize(1 << 15);
        for (int i = 0; i < int(table.size()); ++i) {
            uint16_t val = i;
            int non = popcount(val);
            if (non%2) val |= (1 << 15);
            bool found = false;
            for (int j = 0; j < int(table.size()); ++j) {
                if ((j ^ (j << 1)) == val) {
                    table[i] = j; found = true; break;
                }
            }
            if (!found) {
                printf("Oops: did not find for %d %u\n", i, val);
                exit(1);
            }
        }
    }
    return table.data();
}
uint16_t prune_iq4ks(uint16_t v, const int8_t * values, const float * x, const float * w, float dl) {
    if (popcount(v)%2 == 0) return v;
    float best_score = std::numeric_limits<float>::max();
    uint8_t q4[4];
    int jbest = -1;
    uint8_t bestq = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t q = (v >> 4*j) & 0xf;
        q4[j] = q;
        auto pc = popcount(q);
        float diff0 = dl*iq4k_values[q] - x[j];
        int qmin = std::max(int(q)-2,  0);
        int qmax = std::min(int(q)+2, 15);
        for (int iq = qmin; iq <= qmax; ++iq) {
            uint8_t qq = iq;
            if (qq == q) continue;
            int pci = popcount(qq);
            if (std::abs(pci - pc)%2) {
                float diff1 = dl*values[qq] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) {
                    best_score = score; jbest = j; bestq = qq;
                }
            }
        }
    }
    GGML_ASSERT(jbest >= 0);
    q4[jbest] = bestq;
    return (q4[0] | (q4[1] << 4) | (q4[2] << 8) | (q4[3] << 12));
}
static void quantize_row_iq4_kss_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const uint16_t * table,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    *dptr = 0;
    block_iq4_kss * y = (block_iq4_kss *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    uint16_t vps[block_size/2], vms[block_size/2], vs[block_size/2];
    float xv[4], wv[4];

    float amax_scale = 0;

    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KSS);

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_kss));
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
            float best = 0;
            float d = -max/iq4k_values[0];
            std::memset(vs, 0, block_size);
            for (int itry = -ntry; itry <= ntry; ++itry) {
                float id = (itry + values[0])/max;
                float sumqx_p = 0, sumq2_p = 0;
                float sumqx_m = 0, sumq2_m = 0;
                float this_d = 1/id;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(values,  al) << 4*j);
                        vm |= (best_index_iq4nl(values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                bool copy_p = false, copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }

                id = (itry + shifted_values[0])/max;
                this_d = 1/id;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(shifted_values,  al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, shifted_values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = shifted_values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                copy_p = copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }
            }
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KSS)*d;
    if (!d) return;
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        const float * xbl = x + ibl*super_block_size;
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
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = (std::max(0, std::min(127, l)) << 1) - 127;
            if (l) {
                float dl = d*l;
                float idl = 1/dl;
                float mse_p = 0, mse_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = idl*xv[j];
                        vp |= (best_index_iq4nl(        values, al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, al) << 4*j);
                    }
                    vp = prune_iq4ks(vp,         values, xv, wv,  dl);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  dl);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        mse_p += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        mse_m += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                const uint16_t * v = vps;
                const int8_t * block_values = values;
                if (mse_m < mse_p) {
                    v = vms;
                    block_values = values + 16;
                }
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    for (int j = 0; j < 4; ++j) {
                        float q = block_values[(v[k] >> 4*j) & 0xf] * l;
                        sumqx += wv[j]*q*xv[j];
                        sumq2 += wv[j]*q*q;
                    }
                }
                l += 127;
                if (mse_m < mse_p) l |= 1;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    auto val = table[v[k] & 0x7fff];
                    q16[k] = (val << 1) | ((l >> k) & 1);
                }
            } else {
                l += 127;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    q16[k] = ((l >> k) & 1);
                }
            }
        }
    }
    if (sumq2 > 0) *dptr = sumqx/sumq2 * ggml_get_quantize_fudge_factor(GGML_TYPE_IQ4_KSS);
}
}
size_t quantize_iq4_kss(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size    = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);
    std::vector<float> all_scales(n_per_row/kBlockSize);
    float weight[kBlockSize];
    auto table = scramble_table();
    QHelper helper(imatrix, user_data, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, table] (const float * x, void * vy, int n_per_row, const float * imatrix,
             [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_row_iq4_kss_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq4k_values, imatrix, table, 7);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void quantize_row_iq4_kss_ref(const float * x, block_iq4_kss * y, int64_t k) {
    quantize_iq4_kss(x, y, 1, k, nullptr, nullptr);
}
void dequantize_row_iq4_kss(const block_iq4_kss * x, float * y, int64_t k) {
    const float * dptr = (const float *)x;
    const float d = *dptr;
    x = (const block_iq4_kss *)(dptr + 1);
    uint16_t aux16[8];
    const uint8_t * aux8 = (const uint8_t *)aux16;
    for (int ibl = 0; ibl < k/QK_K; ++ibl) {
        auto qs = (const uint16_t *)x[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int16_t ls = 0;
            for (int k = 0; k < 8; ++k) {
                aux16[k] = qs[k] & 0xfffe;
                aux16[k] ^= (aux16[k] >> 1);
                ls |= (qs[k] & 1) << k;
            }
            const int8_t * values = iq4k_values + ((ls & 1) << 4);
            float dl = d * ((ls & 254) - 127);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * values[aux8[j] & 0xf];
                y[j+16] = dl * values[aux8[j] >>  4];
            }
            y  += 32;
            qs += 8;
        }
    }
}
//
// ========================================= iq4_nl_r4
//
void quantize_row_iq4_nl_r4_ref(const float * x, block_iq4_nl_r4  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_iq4_nl_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq4_nl_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    auto row_size_nl = ggml_row_size(GGML_TYPE_IQ4_NL, n_per_row);
    std::vector<char> qtmp(4*row_size_nl);
    QHelper helper(imatrix, user_data, n_per_row, 32);
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_iq4_nl(x, (char *)vy, 1, n_per_row, imatrix, nullptr);
    };
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        helper.quantize(4, src, qtmp.data(), row_size_nl, q_func);
        repack_iq4_nl(4, n_per_row, (const block_iq4_nl *)qtmp.data(), (block_iq4_nl_r4 *)qrow, false);
        src += 4*n_per_row;
        qrow += 4*row_size_nl;
    }
    return nrows*row_size_nl;
}
void dequantize_row_iq4_nl_r4(const block_iq4_nl_r4 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/4;
    int nb = n_per_row/QK4_NL;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[k]);
            for (int i = 0; i < 4; ++i) {
                yk[k][QK4_NL*ib+i+ 0] = scale * iq4k_values[x[ib].qs[4*k+i+ 0] & 0xf];
                yk[k][QK4_NL*ib+i+ 8] = scale * iq4k_values[x[ib].qs[4*k+i+ 0] >>  4];
                yk[k][QK4_NL*ib+i+16] = scale * iq4k_values[x[ib].qs[4*k+i+16] & 0xf];
                yk[k][QK4_NL*ib+i+24] = scale * iq4k_values[x[ib].qs[4*k+i+16] >>  4];
                yk[k][QK4_NL*ib+i+ 4] = scale * iq4k_values[x[ib].qs[4*k+i+32] & 0xf];
                yk[k][QK4_NL*ib+i+12] = scale * iq4k_values[x[ib].qs[4*k+i+32] >>  4];
                yk[k][QK4_NL*ib+i+20] = scale * iq4k_values[x[ib].qs[4*k+i+48] & 0xf];
                yk[k][QK4_NL*ib+i+28] = scale * iq4k_values[x[ib].qs[4*k+i+48] >>  4];
            }
        }
    }
}
void dequantize_row_q4_0_r8(const block_iq4_nl_r8 * x, float * y, int64_t k) {
    // we assume we are called with 8 rows
    int n_per_row = k/8;
    int nb = n_per_row/QK4_0;
    float * yk[8];
    for (int k = 0; k < 8; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 8; ++k) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[k]);
            for (int l = 0; l < 4; ++l) {
                for (int i = 0; i < 4; ++i) {
                    yk[k][QK4_0*ib+4*l+i+ 0] = scale * ((x[ib].qs[32*l+4*k+i] & 0xf) - 8);
                    yk[k][QK4_0*ib+4*l+i+16] = scale * ((x[ib].qs[32*l+4*k+i] >>  4) - 8);
                }
            }
        }
    }
}
void dequantize_row_q8_0_r8(const block_q8_0_r8 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/8;
    int nb = n_per_row/QK8_0;
    float * yk[8];
    for (int k = 0; k < 8; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 8; ++k) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[k]);
            for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                yk[k][QK8_0*ib+4*l+i+ 0] = scale * x[ib].qs[32*l+4*k+i+  0];
                yk[k][QK8_0*ib+4*l+i+16] = scale * x[ib].qs[32*l+4*k+i+128];
            }
        }
    }
}
//
// ========================================= q5_0_r4
//
void quantize_row_q5_0_r4_ref(const float * x, block_q5_0_r4  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q5_0_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_q5_0_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q5_0, n_per_row);
    std::vector<char> qtmp(4*row_size_0);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        quantize_q5_0(src, qtmp.data(), 4, n_per_row, imatrix, user_data);
        repack_q5_0(4, n_per_row, (const block_q5_0 *)qtmp.data(), (block_q5_0_r4 *)qrow, false);
        src += 4*n_per_row;
        qrow += 4*row_size_0;
    }
    return nrows*row_size_0;
}
void dequantize_row_q5_0_r4(const block_q5_0_r4 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/4;
    int nb = n_per_row/QK8_0;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float d = GGML_FP16_TO_FP32(x[ib].d[k]);
            float m = -16*d;
            for (int l = 0; l < 4; ++l) {
                int ll = 16*(l%2) + 4*(l/2);
                for (int i = 0; i < 4; ++i) {
                    yk[k][QK4_0*ib+i+ll+0] = d * ((x[ib].qs[4*k+i+16*l] & 0xf) | (((x[ib].qh[4*k+i] >> (l+0)) & 1) << 4)) + m;
                    yk[k][QK4_0*ib+i+ll+8] = d * ((x[ib].qs[4*k+i+16*l] >>  4) | (((x[ib].qh[4*k+i] >> (l+4)) & 1) << 4)) + m;
                }
            }
        }
    }
}
//
// ========================================= q6_0_r4
//
void quantize_row_q6_0_r4_ref(const float * x, block_q6_0_r4  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q6_0_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_q6_0_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q6_0, n_per_row);
    std::vector<char> qtmp(4*row_size_0);
    char * qrow = (char *)dst;
    QHelper helper(imatrix, user_data, n_per_row, 32);
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_q6_0(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    for (int row = 0; row < nrows; row += 4) {
        helper.quantize(4, src, qtmp.data(), row_size_0, q_func);
        repack_q6_0(4, n_per_row, (const block_q6_0 *)qtmp.data(), (block_q6_0_r4 *)qrow, false);
        src += 4*n_per_row;
        qrow += 4*row_size_0;
    }
    return nrows*row_size_0;
}
void dequantize_row_q6_0_r4(const block_q6_0_r4 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/4;
    int nb = n_per_row/QK6_0;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float d = GGML_FP16_TO_FP32(x[ib].d[k]);
            float m = -32*d;
            for (int l = 0; l < 4; ++l) {
                int ll = 16*(l%2) + 4*(l/2);
                for (int i = 0; i < 4; ++i) {
                    yk[k][QK4_0*ib+i+ll+0] = d * ((x[ib].qs[4*k+i+16*l] & 0xf) | (((x[ib].qh[4*k+i+16*(l%2)] >> (2*(l/2)+0)) & 3) << 4)) + m;
                    yk[k][QK4_0*ib+i+ll+8] = d * ((x[ib].qs[4*k+i+16*l] >>  4) | (((x[ib].qh[4*k+i+16*(l%2)] >> (2*(l/2)+4)) & 3) << 4)) + m;
                }
            }
        }
    }
}
void quantize_row_iq4_xs_r8_ref(const float * x, block_iq4_xs_r8 * y, int64_t k) {
    quantize_iq4_xs_r8(x, (void *)y, 8, k/8, nullptr, nullptr);
}
size_t quantize_iq4_xs_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_iq4_xs(x, (char *)vy, 1, n_per_row, imatrix, nullptr);
    };
    return quantize_repack<32, block_iq4_xs, block_iq4_xs_r8, 8>(GGML_TYPE_IQ4_XS, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_iq4_xs);
}
void dequantize_row_iq4_xs_r8(const block_iq4_xs_r8 * x, float * y, int64_t k) {
    auto n_per_row = k/8;
    float * y8[8];
    for (int k = 0; k < 8; ++k) y8[k] = y + n_per_row*k;
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 8; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                    y8[k][QK_K*ibl+32*ib+8*l+i+0] = dl * iq4k_values[x[ibl].qs[128*ib+4*k+i+32*l] & 0xf];
                    y8[k][QK_K*ibl+32*ib+8*l+i+4] = dl * iq4k_values[x[ibl].qs[128*ib+4*k+i+32*l] >>  4];
                }
            }
        }
    }
}
void dequantize_row_iq4_ks_r4(const block_iq4_ks_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    const float * dptr = (const float *)x;
    x = (const block_iq4_ks_r4 *)(dptr + 4);
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = dptr[k];
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl = d * ((x[ibl].scales[4*ib + k] & 254) - 127);
                auto values = iq4k_values + ((x[ibl].scales[4*ib + k] & 1) << 4);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * values[x[ibl].qs[64*ib+4*k+i+ 0] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * values[x[ibl].qs[64*ib+4*k+i+ 0] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * values[x[ibl].qs[64*ib+4*k+i+16] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * values[x[ibl].qs[64*ib+4*k+i+16] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * values[x[ibl].qs[64*ib+4*k+i+32] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * values[x[ibl].qs[64*ib+4*k+i+32] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * values[x[ibl].qs[64*ib+4*k+i+48] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * values[x[ibl].qs[64*ib+4*k+i+48] >>  4];
                }
            }
        }
    }
}
void dequantize_row_iq4_ks_r16(const block_iq4_ks_r16 * x, float * y, int64_t k) {
    auto n_per_row = k/16;
    float * y16[16];
    for (int k = 0; k < 16; ++k) y16[k] = y + n_per_row*k;
    int nblock = n_per_row/QK8_0;
    const float * dptr = (const float *)x;
    x = (const block_iq4_ks_r16 *)(dptr + 16);
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 16; ++k) {
            const float d = dptr[k];
            float dl = d * ((x[ib].scales[k] & 254) - 127);
            auto values = iq4k_values + ((x[ib].scales[k] & 1) << 4);
            for (int i = 0; i < 4; ++i) {
                y16[k][QK8_0*ib+i+ 0] = dl * values[x[ib].qs[4*k+i+  0] & 0xf];
                y16[k][QK8_0*ib+i+16] = dl * values[x[ib].qs[4*k+i+  0] >>  4];
                y16[k][QK8_0*ib+i+ 4] = dl * values[x[ib].qs[4*k+i+ 64] & 0xf];
                y16[k][QK8_0*ib+i+20] = dl * values[x[ib].qs[4*k+i+ 64] >>  4];
                y16[k][QK8_0*ib+i+ 8] = dl * values[x[ib].qs[4*k+i+128] & 0xf];
                y16[k][QK8_0*ib+i+24] = dl * values[x[ib].qs[4*k+i+128] >>  4];
                y16[k][QK8_0*ib+i+12] = dl * values[x[ib].qs[4*k+i+192] & 0xf];
                y16[k][QK8_0*ib+i+28] = dl * values[x[ib].qs[4*k+i+192] >>  4];
            }
        }
    }
}
//
// ========================================= iq2_bn_r4
//
void quantize_row_iq2_bn_r4_ref(const float * x, block_iq2_bn  * y, int64_t k) {
    quantize_iq2_bn_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq2_bn_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_IQ1BN == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_BN, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq2_bn(src, (void *)qtmp.data(), 4, n_per_row, imatrix, user_data);
        repack_iq2_bn(4, n_per_row, qtmp.data(), qcur, false);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}
void dequantize_row_iq2_bn_r4(const block_iq2_bn * x, float * y, int64_t k) {
    static_assert(QK_IQ1BN == 64);
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    const float * d4 = (const float *)x;
    const uint8_t * qx = (const uint8_t *)(d4 + 4);
    int nblock = n_per_row/QK_IQ1BN;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 4; ++k) {
            for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                uint8_t q = qx[4*k + i + 16*l];
                y4[k][64*ib + 16*l + i +  0] = d4[k] * (((q >> 0) & 3) - 1);
                y4[k][64*ib + 16*l + i +  4] = d4[k] * (((q >> 2) & 3) - 1);
                y4[k][64*ib + 16*l + i +  8] = d4[k] * (((q >> 4) & 3) - 1);
                y4[k][64*ib + 16*l + i + 12] = d4[k] * (((q >> 6) & 3) - 1);
            }
        }
        qx += 64;
    }
}
void quantize_row_q4_k_r4_ref(const float * x, block_q4_k_r4 * y, int64_t k) {
    quantize_q4_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_q4_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_q4_K(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<32, block_q4_K, block_q4_k_r4, 4>(GGML_TYPE_Q4_K, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_q4_k);
}
void dequantize_row_q4_k_r4(const block_q4_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k+0]);
            const float m = GGML_FP16_TO_FP32(x[ibl].d[k+4]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 4*ib + k;
                float dl = d * ((x[ibl].scales_l[is] & 0xf) | (((x[ibl].scales_h[is%16] >> 4*(is/16)) & 0x03) << 4));
                float ml = m * ((x[ibl].scales_l[is] >>  4) | (((x[ibl].scales_h[is%16] >> 4*(is/16)) & 0x0c) << 2));
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * (x[ibl].qs[64*ib+4*k+i+ 0] & 0xf) - ml;
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * (x[ibl].qs[64*ib+4*k+i+ 0] >>  4) - ml;
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * (x[ibl].qs[64*ib+4*k+i+16] & 0xf) - ml;
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * (x[ibl].qs[64*ib+4*k+i+16] >>  4) - ml;
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * (x[ibl].qs[64*ib+4*k+i+32] & 0xf) - ml;
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * (x[ibl].qs[64*ib+4*k+i+32] >>  4) - ml;
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * (x[ibl].qs[64*ib+4*k+i+48] & 0xf) - ml;
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * (x[ibl].qs[64*ib+4*k+i+48] >>  4) - ml;
                }
            }
        }
    }
}
void quantize_row_q6_k_r4_ref(const float * x, block_q6_k_r4 * y, int64_t k) {
    quantize_q6_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_q6_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_q6_K(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<16, block_q6_K, block_q6_k_r4, 4>(GGML_TYPE_Q6_K, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_q6_k);
}
void dequantize_row_q6_k_r4(const block_q6_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            auto ql = x[ibl].ql;
            auto qh = x[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl1 = d * x[ibl].scales[8*ib+k+0];
                float dl2 = d * x[ibl].scales[8*ib+k+4];
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * (((ql[4*k+i+ 0] & 0xf) | ((qh[4*k+i+ 0] << 4) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * (((ql[4*k+i+ 0] >>  4) | ((qh[4*k+i+ 0] << 2) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * (((ql[4*k+i+16] & 0xf) | ((qh[4*k+i+16] << 4) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * (((ql[4*k+i+16] >>  4) | ((qh[4*k+i+16] << 2) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * (((ql[4*k+i+32] & 0xf) | ((qh[4*k+i+ 0] >> 0) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * (((ql[4*k+i+32] >>  4) | ((qh[4*k+i+ 0] >> 2) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * (((ql[4*k+i+48] & 0xf) | ((qh[4*k+i+16] >> 0) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * (((ql[4*k+i+48] >>  4) | ((qh[4*k+i+16] >> 2) & 0x30)) - 32);
                }
                ql += 64;
                qh += 32;
            }
        }
    }
}
void quantize_row_q5_k_r4_ref(const float * x, block_q5_k_r4 * y, int64_t k) {
    quantize_q5_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_q5_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_q5_K(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<32, block_q5_K, block_q5_k_r4, 4>(GGML_TYPE_Q5_K, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_q5_k);
}
void dequantize_row_q5_k_r4(const block_q5_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k+0]);
            const float m = GGML_FP16_TO_FP32(x[ibl].d[k+4]);
            auto ql = x[ibl].qs;
            auto qh = x[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 4*ib + k;
                float dl = d * ((x[ibl].scales_l[is] & 0xf) | (((x[ibl].scales_h[is%16] >> 4*(is/16)) & 0x03) << 4));
                float ml = m * ((x[ibl].scales_l[is] >>  4) | (((x[ibl].scales_h[is%16] >> 4*(is/16)) & 0x0c) << 2));
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * ((ql[4*k+i+ 0] & 0xf) | ((qh[4*k+i] << 4) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * ((ql[4*k+i+ 0] >>  4) | ((qh[4*k+i] << 3) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * ((ql[4*k+i+16] & 0xf) | ((qh[4*k+i] >> 0) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * ((ql[4*k+i+16] >>  4) | ((qh[4*k+i] >> 1) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * ((ql[4*k+i+32] & 0xf) | ((qh[4*k+i] << 2) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * ((ql[4*k+i+32] >>  4) | ((qh[4*k+i] << 1) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * ((ql[4*k+i+48] & 0xf) | ((qh[4*k+i] >> 2) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * ((ql[4*k+i+48] >>  4) | ((qh[4*k+i] >> 3) & 0x10)) - ml;
                }
                ql += 64;
                qh += 16;
            }
        }
    }
}
void quantize_row_q3_k_r4_ref(const float * x, block_q3_k_r4 * y, int64_t k) {
    quantize_q3_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_q3_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_q3_K(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<16, block_q3_K, block_q3_k_r4, 4>(GGML_TYPE_Q3_K, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_q3_k);
}
void dequantize_row_q3_k_r4(const block_q3_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            auto ql = x[ibl].qs;
            auto qh = x[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 0x03) << 4)) - 32);
                is += 4;
                float dl2 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 0x03) << 4)) - 32);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * ((((ql[4*k+i+ 0] >> 0) & 3) | ((qh[4*k+i] << 2) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * ((((ql[4*k+i+ 0] >> 2) & 3) | ((qh[4*k+i] << 1) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * ((((ql[4*k+i+ 0] >> 4) & 3) | ((qh[4*k+i] << 0) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * ((((ql[4*k+i+ 0] >> 6) & 3) | ((qh[4*k+i] >> 1) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * ((((ql[4*k+i+16] >> 0) & 3) | ((qh[4*k+i] >> 2) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * ((((ql[4*k+i+16] >> 2) & 3) | ((qh[4*k+i] >> 3) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * ((((ql[4*k+i+16] >> 4) & 3) | ((qh[4*k+i] >> 4) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * ((((ql[4*k+i+16] >> 6) & 3) | ((qh[4*k+i] >> 5) & 4)) - 4);
                }
                ql += 32;
                qh += 16;
            }
        }
    }
}
void quantize_row_q2_k_r4_ref(const float * x, block_q2_k_r4 * y, int64_t k) {
    quantize_q2_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_q2_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_q2_K(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<16, block_q2_K, block_q2_k_r4, 4>(GGML_TYPE_Q2_K, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_q2_k);
}
void dequantize_row_q2_k_r4(const block_q2_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k+0]);
            const float m = GGML_FP16_TO_FP32(x[ibl].d[k+4]);
            auto ql = x[ibl].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl1 = d * (x[ibl].scales[8*ib + k + 0] & 0xf);
                float ml1 = m * (x[ibl].scales[8*ib + k + 0] >>  4);
                float dl2 = d * (x[ibl].scales[8*ib + k + 4] & 0xf);
                float ml2 = m * (x[ibl].scales[8*ib + k + 4] >>  4);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * ((ql[4*k+i+ 0] >> 0) & 3) - ml1;
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * ((ql[4*k+i+ 0] >> 2) & 3) - ml1;
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * ((ql[4*k+i+ 0] >> 4) & 3) - ml1;
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * ((ql[4*k+i+ 0] >> 6) & 3) - ml1;
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * ((ql[4*k+i+16] >> 0) & 3) - ml2;
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * ((ql[4*k+i+16] >> 2) & 3) - ml2;
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * ((ql[4*k+i+16] >> 4) & 3) - ml2;
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * ((ql[4*k+i+16] >> 6) & 3) - ml2;
                }
                ql += 32;
            }
        }
    }
}
void dequantize_row_iq4_k_r4(const block_iq4_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                is += 4;
                float dl2 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                auto values1 = iq4k_values + (x[ibl].extra[k+0] & (1 << ib) ? 16 : 0);
                auto values2 = iq4k_values + (x[ibl].extra[k+4] & (1 << ib) ? 16 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+ 0] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+ 0] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+16] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+16] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+32] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+32] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+48] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+48] >>  4];
                }
            }
        }
    }
}
void quantize_row_iq5_k_r4_ref(const float * x, block_iq5_k_r4 * y, int64_t k) {
    quantize_iq5_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq5_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq5_k(src, (void *)qtmp.data(), 4, n_per_row, imatrix, user_data);
        repack_iq5_k(4, n_per_row, (const block_iq5_k *)qtmp.data(), (block_iq5_k_r4 *)qcur, false);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}
void dequantize_row_iq5_k_r4(const block_iq5_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                is += 4;
                float dl2 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                auto values1 = iq5nl_values + (x[ibl].extra[k+0] & (1 << ib) ? 32 : 0);
                auto values2 = iq5nl_values + (x[ibl].extra[k+4] & (1 << ib) ? 32 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[(x[ibl].qs[64*ib+4*k+i+ 0] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 0) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[(x[ibl].qs[64*ib+4*k+i+ 0] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 1) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[(x[ibl].qs[64*ib+4*k+i+16] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 2) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[(x[ibl].qs[64*ib+4*k+i+16] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 3) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[(x[ibl].qs[64*ib+4*k+i+32] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 4) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[(x[ibl].qs[64*ib+4*k+i+32] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 5) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[(x[ibl].qs[64*ib+4*k+i+48] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 6) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[(x[ibl].qs[64*ib+4*k+i+48] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 7) & 1) << 4)];
                }
            }
        }
    }
}
void quantize_row_iq5_ks_r4_ref(const float * x, block_iq5_ks_r4 * y, int64_t k) {
    quantize_iq5_ks_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq5_ks_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq5_ks(src, (void *)qtmp.data(), 4, n_per_row, imatrix, user_data);
        repack_iq5_ks(4, n_per_row, (const block_iq5_ks *)qtmp.data(), (block_iq5_ks_r4 *)qcur, false);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}
void dequantize_row_iq5_ks_r4(const block_iq5_ks_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    //auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    const float * dptr = (const float *)x;
    x = (const block_iq5_ks_r4 *)(dptr + 4);
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = dptr[k];
            //if (!isfinite(d)) {
            //    printf("Oops: d = %g for ibl = %d, k = %d\n", d, ibl, k); exit(1);
            //}
            for (int ib = 0; ib < QK_K/32; ++ib) {
                uint8_t sc = x[ibl].scales[4*ib+k];
                float dl = d * ((sc & 254) - 127);
                //if (!isfinite(dl)) {
                //    printf("Oops: dl = %g for ibl = %d, k = %d, ib = %d, d = %g, sc = %u\n", dl, ibl, k, ib, d, sc); exit(1);
                //}
                auto values = iq5nl_values + ((sc & 1) << 5);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * values[(x[ibl].qs[64*ib+4*k+i+ 0] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 0) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * values[(x[ibl].qs[64*ib+4*k+i+ 0] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 1) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * values[(x[ibl].qs[64*ib+4*k+i+16] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 2) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * values[(x[ibl].qs[64*ib+4*k+i+16] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 3) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * values[(x[ibl].qs[64*ib+4*k+i+32] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 4) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * values[(x[ibl].qs[64*ib+4*k+i+32] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 5) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * values[(x[ibl].qs[64*ib+4*k+i+48] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 6) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * values[(x[ibl].qs[64*ib+4*k+i+48] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 7) & 1) << 4)];
                }
                //for (int i = 0; i < 32; ++i) {
                //    if (!isfinite(y4[k][QK_K*ibl+32*ib+i])) {
                //        printf("Oops: y4[%d][%d, %d, %d] = %g\n", k, ibl, ib, i, y4[k][QK_K*ibl+32*ib+i]);
                //        printf("d = %g, dl = %g\n", d, dl);
                //        exit(1);
                //    }
                //}
            }
        }
    }
}
void dequantize_row_q8_k_r8(const block_q8_k_r8 * x, float * y, int64_t k) {
    auto n_per_row = k/8;
    float * y8[8];
    for (int k = 0; k < 8; ++k) y8[k] = y + n_per_row*k;
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 8; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/4; ++ib) {
                for (int i = 0; i < 4; ++i) {
                    y8[k][QK_K*ibl+4*ib+i] = d * x[ibl].qs[32*ib+4*k+i];
                }
            }
        }
    }
}
void dequantize_row_q8_k_r16(const block_q8_k_r16 * x, float * y, int64_t k) {
    auto n_per_row = k/16;
    float * y16[16];
    for (int k = 0; k < 16; ++k) y16[k] = y + n_per_row*k;
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = (const uint8_t *)x[ibl].qs;
        for (int k = 0; k < 16; ++k) {
            const float d = x[ibl].d[k];
            const float m = -128.f*d;
            for (int ib = 0; ib < QK_K/4; ++ib) {
                for (int i = 0; i < 4; ++i) {
                    y16[k][QK_K*ibl+4*ib+i] = d * qs[64*ib+4*k+i] + m;
                }
            }
        }
    }
}
void dequantize_row_q8_KV_r8(const void * vx, float * y, int64_t k) {
    auto n_per_row = k/8;
    float * y8[8];
    for (int k = 0; k < 8; ++k) y8[k] = y + n_per_row*k;
    auto dptr = (const float *)vx;
    auto q8 = (const int8_t *)(dptr + 8);
    for (int ib = 0; ib < n_per_row/16; ++ib) {
        for (int k = 0; k < 8; ++k) {
            for (int l = 0; l < 4; ++l) {
                for (int i = 0; i < 4; ++i) y8[k][16*ib + 4*l + i] = dptr[k] * q8[128*ib + 32*l + 4*k + i];
            }
        }
    }
}
void quantize_row_iq3_k_r4_ref(const float * x, block_iq3_k_r4 * y, int64_t k) {
    quantize_iq3_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq3_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq3_k(src, (void *)qtmp.data(), 4, n_per_row, imatrix, user_data);
        repack_iq3_k(4, n_per_row, (const block_iq3_k *)qtmp.data(), (block_iq3_k_r4 *)qcur, false);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}
void dequantize_row_iq3_k_r4(const block_iq3_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            auto ql = x[ibl].qs;
            auto qh = x[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * (2*((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) + 1) * ((x[ibl].scales_h[is%8] >> (is/8)) & 1 ? -1 : 1);
                is += 4;
                float dl2 = d * (2*((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) + 1) * ((x[ibl].scales_h[is%8] >> (is/8)) & 1 ? -1 : 1);
                auto values1 = iq3nl_values + (x[ibl].extra[k+0] & (1 << ib) ? 8 : 0);
                auto values2 = iq3nl_values + (x[ibl].extra[k+4] & (1 << ib) ? 8 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[((ql[4*k+i+ 0] >> 0) & 3) | ((qh[4*k+i] << 2) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[((ql[4*k+i+ 0] >> 2) & 3) | ((qh[4*k+i] << 1) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[((ql[4*k+i+ 0] >> 4) & 3) | ((qh[4*k+i] << 0) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[((ql[4*k+i+ 0] >> 6) & 3) | ((qh[4*k+i] >> 1) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[((ql[4*k+i+16] >> 0) & 3) | ((qh[4*k+i] >> 2) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[((ql[4*k+i+16] >> 2) & 3) | ((qh[4*k+i] >> 3) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[((ql[4*k+i+16] >> 4) & 3) | ((qh[4*k+i] >> 4) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[((ql[4*k+i+16] >> 6) & 3) | ((qh[4*k+i] >> 5) & 4)];
                }
                ql += 32;
                qh += 16;
            }
        }
    }
}
void quantize_row_iq2_k_r4_ref(const float * x, block_iq2_k_r4 * y, int64_t k) {
    quantize_iq2_k_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq2_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq2_k(src, (void *)qtmp.data(), 4, n_per_row, imatrix, user_data);
        repack_iq2_k(4, n_per_row, (const block_iq2_k *)qtmp.data(), (block_iq2_k_r4 *)qcur, false);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}
void dequantize_row_iq2_k_r4(const block_iq2_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            auto ql = x[ibl].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * (((x[ibl].scales[is%32] >> 4*(is/32)) & 0xf) - 8);
                is += 4;
                float dl2 = d * (((x[ibl].scales[is%32] >> 4*(is/32)) & 0xf) - 8);
                auto values1 = iq2nl_values + (x[ibl].extra[k+0] & (1 << ib) ? 4 : 0);
                auto values2 = iq2nl_values + (x[ibl].extra[k+4] & (1 << ib) ? 4 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[(ql[4*k+i+ 0] >> 0) & 3];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[(ql[4*k+i+ 0] >> 2) & 3];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[(ql[4*k+i+ 0] >> 4) & 3];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[(ql[4*k+i+ 0] >> 6) & 3];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[(ql[4*k+i+16] >> 0) & 3];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[(ql[4*k+i+16] >> 2) & 3];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[(ql[4*k+i+16] >> 4) & 3];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[(ql[4*k+i+16] >> 6) & 3];
                }
                ql += 32;
            }
        }
    }
}
void quantize_row_iq2_xxs_r4_ref(const float * x, block_iq2_xxs_r4 * y, int64_t k) {
    quantize_iq2_xxs_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq2_xxs_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_iq2_xxs(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<32, block_iq2_xxs, block_iq2_xxs_r4, 4>(GGML_TYPE_IQ2_XXS, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_iq2_xxs);
}
void dequantize_row_iq2_xxs_r4(const block_iq2_xxs_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    uint32_t s32;
    const uint8_t * s8 = (const uint8_t *)&s32;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint32_t * sas = (const uint32_t *)x[ibl].sas;
        for (int k = 0; k < 4; ++k) {
            const float d = 0.125f*GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                uint32_t aux32 = sas[4*ib+k];
                s32 = aux32 & 0x01010101;
                uint8_t scale = s8[0] | (s8[1] << 1) | (s8[2] << 2) | (s8[3] << 3);
                float dl = d*(2*scale+1);
                aux32 &= 0xfefefefe;
                aux32 ^= (aux32 >> 1);
                for (int i = 0; i < 4; ++i) {
                    auto val = (const int8_t *)(iq2xxs_grid + x[ibl].qs[16*ib+4*k+i]);
                    for (int j = 0; j < 8; ++j) y4[k][QK_K*ibl+32*ib+8*i+j] = dl * val[j] * (aux32 & (1 << j) ? -1 : 1);
                    aux32 >>= 8;
                }
            }
        }
    }
}
void quantize_row_iq2_xs_r4_ref(const float * x, block_iq2_xs_r4 * y, int64_t k) {
    quantize_iq2_xs_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq2_xs_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_iq2_xs(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<16, block_iq2_xs, block_iq2_xs_r4, 4>(GGML_TYPE_IQ2_XS, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_iq2_xs);
}
void dequantize_row_iq2_xs_r4(const block_iq2_xs_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = 0.125f*GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl1 = d * (2*(x[ibl].scales[4*ib+k] & 0xf) + 1);
                float dl2 = d * (2*(x[ibl].scales[4*ib+k] >>  4) + 1);
                for (int i = 0; i < 4; ++i) {
                    auto val = (const int8_t *)(iq2xs_grid + (x[ibl].qs[16*ib+4*k+i] & 511));
                    auto signs = x[ibl].qs[16*ib+4*k+i] >> 9;
                    signs ^= (signs << 1);
                    float dl = i < 2 ? dl1 : dl2;
                    for (int j = 0; j < 8; ++j) y4[k][QK_K*ibl+32*ib+8*i+j] = dl * val[j] * (signs & (1 << j) ? -1 : 1);
                }
            }
        }
    }
}
void quantize_row_iq2_s_r4_ref(const float * x, block_iq2_s_r4 * y, int64_t k) {
    quantize_iq2_s_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq2_s_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_iq2_s(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<16, block_iq2_s, block_iq2_s_r4, 4>(GGML_TYPE_IQ2_S, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_iq2_s);
}
void dequantize_row_iq2_s_r4(const block_iq2_s_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = 0.125f*GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl1 = d * (2*(x[ibl].scales[4*ib+k] & 0xf) + 1);
                float dl2 = d * (2*(x[ibl].scales[4*ib+k] >>  4) + 1);
                for (int i = 0; i < 4; ++i) {
                    auto val = (const int8_t *)(iq2s_grid + (x[ibl].qs[16*ib+4*k+i] | ((x[ibl].qh[4*ib+k] << (8 - 2*i)) & 0x300)));
                    auto signs = x[ibl].signs[16*ib+4*k+i];
                    float dl = i < 2 ? dl1 : dl2;
                    for (int j = 0; j < 8; ++j) y4[k][QK_K*ibl+32*ib+8*i+j] = dl * val[j] * (signs & (1 << j) ? -1 : 1);
                }
            }
        }
    }
}
void quantize_row_iq3_xxs_r4_ref(const float * x, block_iq3_xxs_r4 * y, int64_t k) {
    quantize_iq3_xxs_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq3_xxs_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_iq3_xxs(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<32, block_iq3_xxs, block_iq3_xxs_r4, 4>(GGML_TYPE_IQ3_XXS, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_iq3_xxs);
}
void dequantize_row_iq3_xxs_r4(const block_iq3_xxs_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    uint32_t s32;
    const uint8_t * s8 = (const uint8_t *)&s32;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint32_t * sas = (const uint32_t *)x[ibl].sas;
        for (int k = 0; k < 4; ++k) {
            const float d = 0.25f*GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                uint32_t aux32 = sas[4*ib+k];
                s32 = aux32 & 0x01010101;
                uint8_t scale = s8[0] | (s8[1] << 1) | (s8[2] << 2) | (s8[3] << 3);
                float dl = d*(2*scale+1);
                aux32 &= 0xfefefefe;
                aux32 ^= (aux32 >> 1);
                for (int i = 0; i < 8; ++i) {
                    auto val = (const int8_t *)(iq3xxs_grid + x[ibl].qs[32*ib+8*k+i]);
                    for (int j = 0; j < 4; ++j) y4[k][QK_K*ibl+32*ib+4*i+j] = dl * val[j] * (aux32 & (1 << j) ? -1 : 1);
                    aux32 >>= 4;
                }
            }
        }
    }
}
void quantize_row_iq3_s_r4_ref(const float * x, block_iq3_s_r4 * y, int64_t k) {
    quantize_iq3_s_r4(x, (void *)y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq3_s_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix,
            [[maybe_unused]] const quantize_user_data * user_data) {
        quantize_iq3_s(x, (char *)vy, 1, n_per_row, imatrix, user_data);
    };
    return quantize_repack<16, block_iq3_s, block_iq3_s_r4, 4>(GGML_TYPE_IQ3_S, src, dst, nrows, n_per_row, imatrix, user_data,
            q_func, repack_iq3_s);
}
void dequantize_row_iq3_s_r4(const block_iq3_s_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int l = 4*ib + k;
                float dl = d * (1 + 2*((x[ibl].scales[l%16] >> 4*(l/16)) & 0xf));
                for (int i = 0; i < 4; ++i) {
                    auto grid1 = (const uint8_t *)(iq3s_grid + x[ibl].qs[32*ib+k+8*i+0] + ((x[ibl].qh[4*ib+k] << (8-i)) & 0x100));
                    auto grid2 = (const uint8_t *)(iq3s_grid + x[ibl].qs[32*ib+k+8*i+4] + ((x[ibl].qh[4*ib+k] << (4-i)) & 0x100));
                    for (int j = 0; j < 4; ++j) {
                        y4[k][QK_K*ibl+32*ib+4*i+ 0+j] = dl * grid1[j] * (x[ibl].signs[16*ib+4*k+j] & (1 << (i+0)) ? -1 : 1);
                        y4[k][QK_K*ibl+32*ib+4*i+16+j] = dl * grid2[j] * (x[ibl].signs[16*ib+4*k+j] & (1 << (i+4)) ? -1 : 1);
                    }
                }
            }
        }
    }
}
void quantize_row_iq1_s_r4_ref(const float * x, block_iq1_s_r4  * y, int64_t k) {
    quantize_iq1_s_r4(x, y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq1_s_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    int nblock = n_per_row/kBlockSize;
    float weight[kBlockSize];
    int8_t L[kBlockSize];
    float pairs[2*kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];
    float max[4];
    uint16_t index[4];
    int shift;
    float invd[4];
    std::vector<float> scales(4*nblock);
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_S_R4, n_per_row);
    char * cy = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        ggml_half * dptr = (ggml_half *)cy;
        auto y = (block_iq1_s_r4 *)(dptr + 4);
        for (int k = 0; k < 4; ++k) max[k] = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                auto xb = src + k*n_per_row + kBlockSize*ibl;
                float sumx2 = 0;
                for (int j = 0; j < kBlockSize; ++j) sumx2 += xb[j]*xb[j];
                if (sumx2 < 1e-14f) {
                    //printf("Found block with all zeros\n");
                    // all zero
                    int ind = 1029; // this is the grid entry with all zeros
                    scales[4*ibl+k] = 0;
                    uint16_t h = 0;
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[4*i + k] = ind & 255;
                        h |= (ind >> 8) << 3*i;
                    }
                    y[ibl].qh[k] = h;
                    continue;
                }
                float sigma2 = 1.5f*sumx2/kBlockSize;
                bool have_imatrix = false;
                if (imatrix) {
                    have_imatrix = true;
                    float sumwx = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        weight[j] = imatrix[kBlockSize*ibl + j]*sqrt(sigma2 + xb[j]*xb[j]);
                        sumwx += weight[j]*std::abs(xb[j]);
                    }
                    if (sumwx < 1e-14f) {
                        printf("Found block with mismatching importance/model weights\n");
                        // Either all weights are zero, or xb is zero where weight is not zero.
                        // In both of these cases it is better to simply ignore the imatrix
                        have_imatrix = false;
                    }
                }
                if (!have_imatrix) {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                }
                iq1s_process_1block(kBlockSize, xb, weight, L, scales.data() + 4*ibl + k, index, &shift, pairs, sumx, sumw);
                GGML_ASSERT(scales[4*ibl+k] >= 0);
                max[k] = std::max(max[k], scales[4*ibl+k]);
                uint16_t h = 0;
                for (int i = 0; i < 4; ++i) {
                    GGML_ASSERT(index[i] >= 0 && index[i] < 2048);
                    y[ibl].qs[4*i + k] = index[i] & 255;
                    h |= (index[i] >> 8) << 3*i;
                }
                if (shift < 0) h |= 0x8000;
                y[ibl].qh[k] = h;
            }
        }
        for (int k = 0; k < 4; ++k) {
            dptr[k] = GGML_FP32_TO_FP16(1.0625f*max[k]/15);;
            invd[k] = max[k] ? 15/max[k] : 0.f;
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                int ls = nearest_int(0.5f*(scales[4*ibl+k]*invd[k] - 1));
                ls = std::max(0, std::min(7, ls));
                y[ibl].qh[k] |= (ls << 12);
            }
        }
        cy  += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}
void dequantize_row_iq1_s_r4(const block_iq1_s_r4  * x, float * y, int64_t n) {
    auto dptr = (const ggml_half *)x;
    x = (const block_iq1_s_r4 *)(dptr + 4);
    float d[4];
    for (int k = 0; k < 4; ++k) d[k] = GGML_FP16_TO_FP32(dptr[k]);
    int n_per_row = n/4;
    GGML_ASSERT(n_per_row%32 == 0);
    int nblock = n_per_row/32;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float shift = x[ib].qh[k] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            float dl = d[k]*(2*((x[ib].qh[k] >> 12) & 7) + 1);
            for (int i = 0; i < 4; ++i) {
                auto idx = x[ib].qs[4*i+k] | (((x[ib].qh[k] >> 3*i) & 7) << 8);
                auto grid = (const int8_t *)(iq1s_grid + idx);
                for (int j = 0; j < 8; ++j) yk[k][32*ib + 8*i + j] = dl*(grid[j] + shift);
            }
        }
    }
}
void quantize_row_iq1_m_r4_ref(const float * x, block_iq1_m_r4  * y, int64_t k) {
    quantize_iq1_m_r4(x, y, 4, k/4, nullptr, nullptr);
}
size_t quantize_iq1_m_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        [[maybe_unused]] const quantize_user_data * user_data) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    int nblock = n_per_row/kBlockSize;
    float weight[kBlockSize];
    int8_t L[kBlockSize];
    float pairs[2*kBlockSize];
    float max[4];
    uint16_t index[4];
    int shift1, shift2;
    float invd[4];
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};
    std::vector<float> scales(8*nblock);
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_M_R4, n_per_row);
    char * cy = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        ggml_half * dptr = (ggml_half *)cy;
        auto y = (block_iq1_m_r4 *)(dptr + 4);
        for (int k = 0; k < 4; ++k) max[k] = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                auto xb = src + k*n_per_row + kBlockSize*ibl;
                float sumx2l = 0, sumx2h = 0;
                for (int j = 0; j < kBlockSize/2; ++j) sumx2l += xb[j]*xb[j];
                for (int j = kBlockSize/2; j < kBlockSize; ++j) sumx2h += xb[j]*xb[j];
                float sumx2 = sumx2l + sumx2h;
                if (sumx2 < 1e-14f) {
                    scales[8*ibl+2*k+0] = scales[8*ibl+2*k+1] = 0;
                    int ind = 1029;
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[4*i + k] = ind & 255;
                    }
                    for (int i = 0; i < 2; ++i) {
                        y[ibl].qh[4*i+k] = (ind >> 8) | ((ind >> 8) << 4);
                    }
                    continue;
                }
                float sigma2 = 1.5f*sumx2/kBlockSize;
                if (imatrix) {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = imatrix[kBlockSize*ibl + j]*sqrt(sigma2 + xb[j]*xb[j]);
                    float sumwx = 0;
                    for (int j = 0; j < kBlockSize/2; ++j) sumwx += weight[j]*std::abs(xb[j]);
                    if (sumwx < 1e-14f) {
                        for (int j = 0; j < kBlockSize/2; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                    }
                    sumwx = 0;
                    for (int j = kBlockSize/2; j < kBlockSize; ++j) sumwx += weight[j]*std::abs(xb[j]);
                    if (sumwx < 1e-14) {
                        for (int j = kBlockSize/2; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                    }
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                }
                if (sumx2l > 1e-14f) {
                    iq1m_process_1block(xb+ 0, weight+ 0, L, scales.data() + 8*ibl + 2*k+0, index+0, &shift1, pairs);
                } else {
                    scales[8*ibl+2*k+0] = 0;
                    index[0] = index[1] = 1029;
                }
                if (sumx2h > 1e-14f) {
                    iq1m_process_1block(xb+16, weight+16, L, scales.data() + 8*ibl + 2*k+1, index+2, &shift2, pairs);
                } else {
                    scales[8*ibl+2*k+1] = 0;
                    index[2] = index[3] = 1029;
                }
                max[k] = std::max(max[k], std::max(scales[8*ibl+2*k+0], scales[8*ibl+2*k+1]));
                for (int i = 0; i < 4; ++i) {
                    y[ibl].qs[4*i + k] = index[i] & 255;
                }
                for (int i = 0; i < 2; ++i) {
                    y[ibl].qh[4*i+k] = (index[2*i+0] >> 8) | ((index[2*i+1] >> 8) << 4);
                }
                y[ibl].qh[0+k] |= masks[shift1];
                y[ibl].qh[4+k] |= masks[shift2];
            }
        }
        for (int k = 0; k < 4; ++k) {
            dptr[k] = GGML_FP32_TO_FP16(1.0625f*max[k]/15);;
            invd[k] = max[k] ? 15/max[k] : 0.f;
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                int ls1 = nearest_int(scales[8*ibl+2*k+0]*invd[k]);
                int ls2 = nearest_int(scales[8*ibl+2*k+1]*invd[k]);
                ls1 = std::max(0, std::min(15, ls1));
                ls2 = std::max(0, std::min(15, ls2));
                y[ibl].scales[k] = ls1 | (ls2 << 4);
            }
        }
        cy  += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}
void dequantize_row_iq1_m_r4(const block_iq1_m_r4  * x, float * y, int64_t n) {
    auto dptr = (const ggml_half *)x;
    x = (const block_iq1_m_r4 *)(dptr + 4);
    float d[4];
    for (int k = 0; k < 4; ++k) d[k] = GGML_FP16_TO_FP32(dptr[k]);
    int n_per_row = n/4;
    GGML_ASSERT(n_per_row%32 == 0);
    int nblock = n_per_row/32;
    float dl[2];
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 4; ++k) {
            dl[0] = d[k]*(x[ib].scales[k] & 0xf);
            dl[1] = d[k]*(x[ib].scales[k] >>  4);
            for (int i = 0; i < 2; ++i) {
                auto idx1 = x[ib].qs[8*i+k+0] | ((x[ib].qh[4*i+k] & 0x07) << 8);
                auto idx2 = x[ib].qs[8*i+k+4] | ((x[ib].qh[4*i+k] & 0x70) << 4);
                auto grid1 = (const int8_t *)(iq1s_grid + idx1);
                auto grid2 = (const int8_t *)(iq1s_grid + idx2);
                auto delta1 = x[ib].qh[4*i+k] & 0x08 ? -IQ1M_DELTA : IQ1M_DELTA;
                auto delta2 = x[ib].qh[4*i+k] & 0x80 ? -IQ1M_DELTA : IQ1M_DELTA;
                for (int j = 0; j < 8; ++j) yk[k][32*ib + 16*i + j + 0] = dl[i]*(grid1[j] + delta1);
                for (int j = 0; j < 8; ++j) yk[k][32*ib + 16*i + j + 8] = dl[i]*(grid2[j] + delta2);
            }
        }
    }
}
void dequantize_row_q8_KV(const void * x, float * y, int64_t k) {
    auto dptr = (const float *)x;
    float d = dptr[0];
    auto q8 = (const int8_t *)(dptr + 2);
    for (int j = 0; j < k; ++j) y[j] = d * q8[j];
}
void dequantize_row_ms_i2s(const void * vx, float * y, int64_t k) {
    constexpr int kBlockSize = 128;
    constexpr int kGroupSize = kBlockSize/4;
    GGML_ASSERT(k % kBlockSize == 0);
    const uint8_t * x = (const uint8_t *)vx;
    const float * dptr = (const float *)(x + k/4);
    const float d = dptr[0];
    int nb = k/kBlockSize;
    for (int ib = 0; ib < nb; ++ib) {
        for (int ig = 0; ig < kBlockSize/kGroupSize; ++ig) {
            int shift = 6 - 2*ig;
            for (int j = 0; j < kGroupSize; ++j) {
                y[j] = d * (((x[j] >> shift) & 3) - 1);
            }
            y += kGroupSize;
        }
        x += kGroupSize;
    }
}
void dequantize_row_iq1_kt(const block_iq1_kt * x, float * y, int64_t k) {
    using Q = QuantizerIQ1KT;
    assert(k % Q::kBlockSize == 0);
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * Q::kScale;
    x = (const block_iq1_kt *)(dptr + 1);
    auto& deq = iq1kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            float sl = d * iq4k_values[x[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < Q::kNg; ++ig) {
                uint16_t idx = x[ibl].ql[ib*Q::kNg + ig] | ((x[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg + ig] << (8 - 4*(ib/(Q::kNblock/2)))) & 0xf00);
                idx |= (x[ibl].sh[ib] << (8 - ig) & 0x1000);
                deq.set_values(idx, y, sl);
                y += Q::kGroupSize;
            }
        }
    }
    const int nt = (k % QK_K)/32;
    if (nt == 0) return;
    const uint8_t * sh = (const uint8_t *)(x + nb);
    const uint8_t * ql = sh + QK_K/32;
    const uint8_t * qh = ql + 4*nt;
    for (int ib = 0; ib < nt; ++ib) {
        float sl = d * iq4k_values[sh[ib] & 0xf];
        for (int ig = 0; ig < Q::kNg; ++ig) {
            const int j = ib*Q::kNg + ig;
            uint16_t idx = ql[j] | ((qh[j%16] << (8 - 4*(j/16))) & 0xf00);
            idx |= (sh[ib] << (8 - ig) & 0x1000);
            deq.set_values(idx, y, sl);
            y += Q::kGroupSize;
        }
    }
}
void dequantize_row_iq2_kt(const block_iq2_kt * x, float * y, int64_t k) {
    using Q = QuantizerIQ2KT;
    assert(k % Q::kBlockSize == 0);
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    // x1.05: the run-time scale every other iq2_kt decoder applies -- CPU matmul (iqk_gemm_ktquants.cpp
    // "31.75f * 1.05f"), CUDA mmq/dmmv, and CUDA dequant (convert.cu dequantize_block_iq2_kt). Without it
    // this to_float decoded iq2_kt exactly 5% low vs the matmul (measured vec_dot/dot(dequant) = 1.05000
    // on exact activations, controls 1.00000). Upstream ik_llama's CPU dequant has the same omission.
    const float d = *dptr * QuantizerIQ2KT::kScale * 1.05f;
    x = (const block_iq2_kt *)(dptr + 1);
    auto& deq = iq2kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto yl = y + ibl*Q::kSuperBlockSize;
        auto yh = yl + Q::kSuperBlockSize/2;
        const uint16_t * ql = (const uint16_t *)x[ibl].ql;
        const uint16_t * qh = ql + Q::kNg*QuantizerIQ2KT::kNblock/2;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            float sl = d * iq4k_values[x[ibl].scales[ib] & 0xf];
            float sh = d * iq4k_values[x[ibl].scales[ib] >>  4];
            for (int ig = 0; ig < Q::kNg; ++ig) {
                deq.set_values(ql[ig], yl, sl);
                deq.set_values(qh[ig], yh, sh);
                yl += QuantizerIQ2KT::kGroupSize;
                yh += QuantizerIQ2KT::kGroupSize;
            }
            ql += QuantizerIQ2KT::kNg;
            qh += QuantizerIQ2KT::kNg;
        }
    }
    const int nt = (k % QK_K)/32;
    if (nt == 0) return;
    const uint8_t * scales = (const uint8_t *)(x + nb);
    const uint16_t * ql = (const uint16_t *)(scales + 4);
    y += nb*QK_K;
    for (int ib = 0; ib < nt; ++ib) {
        float s = d * iq4k_values[(scales[ib%4] >> 4*(ib/4)) & 0xf];
        for (int ig = 0; ig < Q::kNg; ++ig) {
            deq.set_values(ql[Q::kNg*ib + ig], y, s);
            y += Q::kGroupSize;
        }
    }
}
void dequantize_row_iq3_kt(const block_iq3_kt * x, float * y, int64_t k) {
#ifdef __AVX2__
    //if (iqk_dequantize_ktquants(GGML_TYPE_IQ3_KT, k, x, 0, y, 0, 1)) return;
#endif
    using Q = QuantizerIQ3KT;
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    assert(k % 32 == 0);
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    // x1.01: matches the CPU matmul, CUDA mmq/dmmv and CUDA dequant (convert.cu dequantize_block_iq3_kt).
    // Measured before: vec_dot/dot(dequant) = 1.01000. Upstream ik_llama added this same factor here.
    const float d = *dptr * Q::kScale * 1.01f;
    x = (const block_iq3_kt *)(dptr + 1);
    auto& deq = iq3kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto yl = y + ibl*Q::kSuperBlockSize;
        auto yh = yl + Q::kSuperBlockSize/2;
        auto qll = (const uint16_t *)x[ibl].ql;
        auto qlh = qll + kNumGroups/2;
        int jj = 0;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            float sl = d * (x[ibl].scales[ib] & 0xf);
            float sh = d * (x[ibl].scales[ib] >>  4);
            uint8_t l_mask = 1 << ib;
            uint8_t h_mask = l_mask << (Q::kNblock/2);
            for (int ig = 0; ig < Q::kNg; ++ig) {
                deq.set_values(qll[jj], yl, sl);
                deq.set_values(qlh[jj], yh, sh);
                for (int j = 0; j < Q::kGroupSize; ++j) {
                    if (x[ibl].qh[ig*Q::kGroupSize+j] & l_mask) yl[j] = -yl[j];
                    if (x[ibl].qh[ig*Q::kGroupSize+j] & h_mask) yh[j] = -yh[j];
                }
                yl += Q::kGroupSize;
                yh += Q::kGroupSize;
                ++jj;
            }
        }
    }
    // KT tail after the full blocks (layout: repack_tail_iq3_kt)
    const int nt = (k % QK_K)/32;
    if (nt > 0) {
        const uint8_t * tail = (const uint8_t *)(x + nb);
        const uint16_t * ql = (const uint16_t *)tail;
        const uint8_t * qh = tail + 8*nt;
        const uint8_t * scales = tail + 12*nt;
        y += nb*QK_K;
        for (int ib = 0; ib < nt; ++ib) {
            float s = d * ((scales[ib/2] >> 4*(ib&1)) & 0xf);
            for (int ig = 0; ig < Q::kNg; ++ig) {
                deq.set_values(ql[Q::kNg*ib + ig], y, s);
                for (int j = 0; j < Q::kGroupSize; ++j) {
                    if (qh[4*ib + ig] & (1 << j)) y[j] = -y[j];
                }
                y += Q::kGroupSize;
            }
        }
    }
}
namespace {
const QuantizerIQ4KT& iq4kt_dequantizer() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ4KT> dequantizer;
    if (!dequantizer) dequantizer = std::make_unique<QuantizerIQ4KT>(0, 0, 4096);
    return *dequantizer;
}
}
void dequantize_row_iq4_kt(const block_iq4_kt * x, float * y, int64_t k) {
#ifdef __AVX2__
    //if (iqk_dequantize_ktquants(GGML_TYPE_IQ4_KT, k, x, 0, y, 0, 1)) return;
#endif
    using Q = QuantizerIQ4KT;
    assert(k % 32 == 0);
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = dptr[0] * Q::kScale;
    x = (const block_iq4_kt *)(dptr + 1);
    auto& deq = iq4kt_dequantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto shb = x[ibl].qs;
        auto ql = (const uint8_t *)(shb + Q::kNblock);
        auto qh = ql + kNumGroups;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int offset = shb[ib] & 1 ? 32768 + 4096 : 4096;
            int ls = int((shb[ib] & 0xff) >> 1) - 64;
            float sl = d * ls;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                int jj = ib*Q::kNg+ig;
                uint16_t idx = ql[jj] | ((qh[jj%(kNumGroups/2)] << (8 - 4*(jj/(kNumGroups/2)))) & 0xf00) | (((shb[ib] >> (8 + 3*ig)) & 7) << 12);
                deq.set_values(idx, y, sl, offset);
                y += Q::kGroupSize;
            }
        }
    }
    // KT tail: nt 16-byte records after the full blocks (layout: repack_tail_iq4_kt)
    const int nt = (k % QK_K)/32;
    if (nt > 0) {
        const uint8_t * tail = (const uint8_t *)(x + nb);
        const uint32_t * shb = (const uint32_t *)tail;
        for (int ib = 0; ib < nt; ++ib) {
            const uint32_t sh = shb[4*ib];
            const uint8_t * ql = tail + 16*ib + 4;
            const uint8_t * qh = tail + 16*ib + 12;
            int offset = sh & 1 ? 32768 + 4096 : 4096;
            int ls = int((sh & 0xff) >> 1) - 64;
            float sl = d * ls;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                uint16_t idx = ql[ig] | (((qh[ig/2] >> 4*(ig&1)) & 0xf) << 8) | (((sh >> (8 + 3*ig)) & 7) << 12);
                deq.set_values(idx, y, sl, offset);
                y += Q::kGroupSize;
            }
        }
    }
}
void quantize_row_q1_0_g128_ref(const float * x, block_q1_0_g128  * y, int64_t k) {
    quantize_row_q1_0_g128(x, y, k);
}
void quantize_row_q1_0_g128(const float * x, void * vy, int64_t k) {
    assert(k % QK1_0_G128 == 0);
    int nb = k / QK1_0_G128;
    auto y = (block_q1_0_g128 *)vy;
    for (int ib = 0; ib < nb; ++ib) {
        float sum = 0;
        for (int j = 0; j < QK1_0_G128; ++j) sum += std::abs(x[j]);
        float d = sum / QK1_0_G128;
        y[ib].d = GGML_FP32_TO_FP16(d);
        std::memset(y[ib].qs, 0, QK1_0_G128/8);
        for (int j = 0; j < QK1_0_G128; ++j) {
            if (x[j] >= 0.0f) {
                y[ib].qs[j / 8] |= (1 << (j % 8));
            }
        }
        x += QK1_0_G128;
    }
}
void dequantize_row_q1_0_g128(const block_q1_0_g128  * x, float * y, int64_t k) {
    assert(k % QK1_0_G128 == 0);
    constexpr uint8_t k_mask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    int nb = k / QK1_0_G128;
    for (int ib = 0; ib < nb; ++ib) {
        float d = GGML_FP16_TO_FP32(x[ib].d);
        for (int i = 0; i < QK1_0_G128/8; ++i) {
            for (int j = 0; j < 8; ++j) {
                *y++ = x[ib].qs[i] & k_mask[j] ? d : -d;
            }
        }
    }
}
void quantize_row_q2_0_g128(const float * x, void * vy, int64_t k) {
    assert(k % QK2_0_G128 == 0);
    int nb = k / QK2_0_G128;
    auto y = (block_q2_0_g128 *)vy;
    for (int ib = 0; ib < nb; ++ib) {
        // absmean scale, as in the 1-bit sibling: the ternary reconstruction is
        // d * {-1,0,+1}, so d is the magnitude of a non-zero weight and rounding x/d
        // picks the code.
        // SCALE RULE, selectable for measurement via GGML_Q2_0_G128_SCALE.
        // MEASURED 2026-08-22 on TriLM_830M (ternary-TRAINED), 12 chunks, same corpus:
        //   f16 parent 4.0306 | mainline tq2_0 4.0325 | THIS quantiser (absmean) 4708.05
        // Root cause of that 1000x gap: absmean over ALL 128 weights counts the zeros. For a
        // ternary tensor {-s, 0, +s} with zero-fraction z, absmean = s*(1-z), so every weight
        // reconstructs as s*(1-z) instead of s -- a systematic ~2x under-scale when half the
        // weights are zero. The CODES are right; only the magnitude is wrong.
        // Mainline quantize_row_tq2_0_ref uses `const float d = amax`, which for a ternary tensor
        // is exactly s -- which is why it reproduces its parent to 0.05%.
        // amax is NOT universally better: on Gaussian weights it is set by an outlier and crushes
        // the bulk toward zero. "lsq" therefore refines d to the least-squares optimum for the
        // chosen codes, d* = sum(x_j*q_j)/sum(q_j*q_j), which is right for either distribution.
        static const int scale_mode = [] {
            const char * e = getenv("GGML_Q2_0_G128_SCALE");
            if (e && !std::strcmp(e, "amax"))    return 1;
            if (e && !std::strcmp(e, "absmean")) return 0;
            return 2;   // lsq -- MEASURED strictly better than both, see the table below
        }();
        // MEASURED 2026-08-22, 12 chunks, identical corpus, this fork's own build:
        //                       absmean      amax        lsq        reference
        //   TriLM_830M (ternary)  4708.05     15.12      4.1237     f16 4.0306, tq2_0 4.0325
        //   Llama-3.2-1B (Gauss) 179201.14  533559.43  77664.59     Q8_0 4.6547
        // lsq wins on BOTH distributions: 1141x on ternary, 2.3x on Gaussian. amax -- the rule
        // mainline TQ2_0 uses -- wins on ternary but is 3x WORSE than absmean on Gaussian, because
        // there it is set by a single outlier and crushes the bulk toward zero. Copying mainline
        // would therefore have fixed one model class and broken the other.
        //
        // Q1_0_G128 needs NO equivalent change, and that is provable rather than assumed: its codes
        // are +-1 with no zero, so sum(q*q) = 128 and sum(x*q) = sum|x|, making d* = sum|x|/128 --
        // absmean IS already the least-squares optimum there. The defect is specific to having a
        // zero code, which drags absmean below the magnitude of the non-zero weights it represents.

        float sum = 0, amax = 0;
        for (int j = 0; j < QK2_0_G128; ++j) {
            const float a = std::abs(x[j]);
            sum += a;
            if (a > amax) amax = a;
        }
        float d = scale_mode == 1 ? amax : sum / QK2_0_G128;
        std::memset(y[ib].qs, 0, QK2_0_G128/4);
        float id = d > 0 ? 1/d : 0.f;
        int q[QK2_0_G128];
        for (int j = 0; j < QK2_0_G128; ++j) {
            int t = (int)std::lround(x[j]*id);
            if (t < -1) t = -1;
            if (t >  1) t =  1;
            q[j] = t;
        }
        if (scale_mode == 2) {
            // Least-squares optimal magnitude for the codes just chosen, then re-code once.
            float num = 0, den = 0;
            for (int j = 0; j < QK2_0_G128; ++j) { num += x[j]*q[j]; den += (float)(q[j]*q[j]); }
            if (den > 0 && num > 0) {
                d = num/den;
                id = 1/d;
                for (int j = 0; j < QK2_0_G128; ++j) {
                    int t = (int)std::lround(x[j]*id);
                    if (t < -1) t = -1;
                    if (t >  1) t =  1;
                    q[j] = t;
                }
                num = den = 0;
                for (int j = 0; j < QK2_0_G128; ++j) { num += x[j]*q[j]; den += (float)(q[j]*q[j]); }
                if (den > 0 && num > 0) d = num/den;
            }
        }
        y[ib].d = GGML_FP32_TO_FP16(d);
        for (int j = 0; j < QK2_0_G128; ++j) {
            const uint8_t code = (uint8_t)(q[j] + 1);        // -1/0/+1 -> 0/1/2
            y[ib].qs[j/4] |= (uint8_t)(code << (2*(j%4)));  // LSB-first within the byte
        }
        x += QK2_0_G128;
    }
}
void quantize_row_q2_0_g128_ref(const float * x, block_q2_0_g128 * y, int64_t k) {
    quantize_row_q2_0_g128(x, y, k);
}
void dequantize_row_q2_0_g128(const block_q2_0_g128 * x, float * y, int64_t k) {
    assert(k % QK2_0_G128 == 0);
    int nb = k / QK2_0_G128;
    for (int ib = 0; ib < nb; ++ib) {
        float d = GGML_FP16_TO_FP32(x[ib].d);
        for (int i = 0; i < QK2_0_G128/4; ++i) {
            const uint8_t bits = x[ib].qs[i];
            *y++ = d * (float)((int)((bits >> 0) & 3) - 1);
            *y++ = d * (float)((int)((bits >> 2) & 3) - 1);
            *y++ = d * (float)((int)((bits >> 4) & 3) - 1);
            *y++ = d * (float)((int)((bits >> 6) & 3) - 1);
        }
    }
}
void repack_q1_0_g128_r8(int nrows, int n_per_row, const block_q1_0_g128 * x, block_q1_0_g128_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows % QK1_0_G128_R8_ROWS == 0);
    GGML_ASSERT(n_per_row % QK1_0_G128 == 0);
    constexpr uint8_t k_mask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const int nblock = n_per_row/QK1_0_G128;
    const block_q1_0_g128 * xr[QK1_0_G128_R8_ROWS];
    for (int row = 0; row < nrows; row += QK1_0_G128_R8_ROWS) {
        for (int k = 0; k < QK1_0_G128_R8_ROWS; ++k) xr[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            std::memset(&y[ib], 0, sizeof(y[ib]));
            for (int k = 0; k < QK1_0_G128_R8_ROWS; ++k) y[ib].d[k] = xr[k][ib].d;
            for (int l = 0; l < 4; ++l) {
                auto qy = y[ib].qs + 4*QK1_0_G128_R8_ROWS*l;
                for (int k = 0; k < QK1_0_G128_R8_ROWS; ++k) {
                    auto qx = xr[k][ib].qs + 4*l;
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            if (qx[i] & k_mask[j+0]) qy[4*k+j] |= (1 << (2*i+0));
                        }
                        for (int j = 0; j < 4; ++j) {
                            if (qx[i] & k_mask[j+4]) qy[4*k+j] |= (1 << (2*i+1));
                        }
                    }
                }
            }
        }
        x += QK1_0_G128_R8_ROWS*nblock;
        y += nblock;
    }
}
void dequantize_row_q1_0_g128_r8(const block_q1_0_g128_r8 * x, float * y, int64_t n) {
    const int n_per_row = (int)(n/QK1_0_G128_R8_ROWS);
    GGML_ASSERT(n_per_row % QK1_0_G128 == 0);
    const int nblock = n_per_row/QK1_0_G128;
    for (int r = 0; r < QK1_0_G128_R8_ROWS; ++r) {
        float * yr = y + (int64_t)r*n_per_row;
        for (int ib = 0; ib < nblock; ++ib) {
            const float d = GGML_FP16_TO_FP32(x[ib].d[r]);
            for (int l = 0; l < 4; ++l) {
                auto qx = x[ib].qs + 32*l;
                for (int i = 0; i < 4; ++i) {
                    uint8_t mask1 = 1 << (2*i+0);
                    uint8_t mask2 = 1 << (2*i+1);
                    for (int j = 0; j < 4; ++j) {
                        yr[(int64_t)ib*QK1_0_G128 + 32*l + 8*i + j + 0] = qx[4*r + j] & mask1 ? d : -d;
                    }
                    for (int j = 0; j < 4; ++j) {
                        yr[(int64_t)ib*QK1_0_G128 + 32*l + 8*i + j + 4] = qx[4*r + j] & mask2 ? d : -d;
                    }
                }
            }
        }
    }
}
// FORK FIX of ik_llama #2541 (still so at ik main cdf232cc1, where this function carries "TODO: fix this"): ik copied the
// plain Q1_0_G128 bytes tile-wise into the R8 struct, but dequantize_row_q1_0_g128_r8 and the AVX2/AVX512/NEON kernels read
// the BIT-INTERLEAVED layout that repack_q1_0_g128_r8 writes. MEASURED (scratch control, 8 rows x 512, gaussian input):
// dequantizing ik's output disagreed with the plain Q1_0_G128 dequantization on 1940 of 4096 elements. Quantize each row
// with the plain quantizer, then repack, so the bytes are exactly what the load-time repack produces.
void quantize_row_q1_0_g128_r8(const float * x, void * vy, int64_t n) {
    auto y = (block_q1_0_g128_r8 *)vy;
    const int n_per_row = (int)(n/QK1_0_G128_R8_ROWS);
    GGML_ASSERT(n_per_row % QK1_0_G128 == 0);
    const int nblock = n_per_row/QK1_0_G128;
    std::vector<block_q1_0_g128> tmp((size_t)QK1_0_G128_R8_ROWS*nblock);
    for (int r = 0; r < QK1_0_G128_R8_ROWS; ++r) {
        quantize_row_q1_0_g128(x + (int64_t)r*n_per_row, tmp.data() + (size_t)r*nblock, n_per_row);
    }
    repack_q1_0_g128_r8(QK1_0_G128_R8_ROWS, n_per_row, tmp.data(), y, false);
}
void quantize_row_q1_0_g128_r8_ref(const float * x, block_q1_0_g128_r8 * y, int64_t n) {
    quantize_row_q1_0_g128_r8(x, (void *)y, n);
}
void repack_pq2_0_r8(int nrows, int n_per_row, const block_pq2_0 * GGML_RESTRICT x, block_pq2_0_r8 * GGML_RESTRICT y, bool online) {
    GGML_UNUSED(online);
    GGML_ASSERT(nrows % QK_PQ2_0_R8_ROWS == 0);
    const int nblock = n_per_row/QK_PQ2_0;
    const block_pq2_0 * xr[QK_PQ2_0_R8_ROWS];
    for (int row = 0; row < nrows; row += QK_PQ2_0_R8_ROWS) {
        for (int k = 0; k < QK_PQ2_0_R8_ROWS; ++k) xr[k] = x + (size_t)k*nblock;
        for (int ib128 = 0; ib128 < nblock; ++ib128) {
            auto yb = y + ib128;
            memset(yb, 0, sizeof(*yb));
            for (int k = 0; k < QK_PQ2_0_R8_ROWS; ++k) yb->d[k] = xr[k][ib128].d;
            for (int ib32 = 0; ib32 < QK_PQ2_0/32; ++ib32) {
                for (int l = 0; l < 8; ++l) {
                    for (int k = 0; k < QK_PQ2_0_R8_ROWS; ++k) {
                        for (int i = 0; i < 4; ++i) {
                            int p = 32*ib32 + 4*l + i;
                            int q = (xr[k][ib128].qs[p>>2] >> (2*(p&3))) & 3;
                            yb->qs[64*ib32 + 32*(l/4) + 4*k + i] |= (q << 2*(l%4));
                        }
                    }
                }
            }
        }
        x += (size_t)QK_PQ2_0_R8_ROWS*nblock;
        y += nblock;
    }
}
void dequantize_row_pq2_0_r8(const block_pq2_0_r8 * x, float * GGML_RESTRICT y, int64_t n) {
    const int n_per_row = (int)(n/QK_PQ2_0_R8_ROWS);
    GGML_ASSERT(n_per_row % QK_PQ2_0 == 0);
    const int nblock = n_per_row/QK_PQ2_0;
    for (int r = 0; r < QK_PQ2_0_R8_ROWS; ++r) {
        float * yr = y + (int64_t)r*n_per_row;
        for (int ib128 = 0; ib128 < nblock; ++ib128) {
            const float d = GGML_FP16_TO_FP32(x[ib128].d[r]);
            for (int ib32 = 0; ib32 < QK_PQ2_0/32; ++ib32) {
                for (int l = 0; l < 8; l += 4) {
                    for (int i = 0; i < 4; ++i) {
                        int q = x[ib128].qs[64*ib32 + 32*(l/4) + 4*r + i];
                        yr[(int64_t)ib128*QK_PQ2_0 + 32*ib32 + 4*l + i +  0] = d * (((q >> 0) & 3) - 1);
                        yr[(int64_t)ib128*QK_PQ2_0 + 32*ib32 + 4*l + i +  4] = d * (((q >> 2) & 3) - 1);
                        yr[(int64_t)ib128*QK_PQ2_0 + 32*ib32 + 4*l + i +  8] = d * (((q >> 4) & 3) - 1);
                        yr[(int64_t)ib128*QK_PQ2_0 + 32*ib32 + 4*l + i + 12] = d * (((q >> 6) & 3) - 1);
                    }
                }
            }
        }
    }
}
void quantize_row_pq2_0_r8(const float * x, void * vy, int64_t n) {
    auto y = (block_pq2_0_r8 *)vy;
    const int n_per_row = (int)(n/QK_PQ2_0_R8_ROWS);
    GGML_ASSERT(n_per_row % QK_PQ2_0 == 0);
    const int nblock = n_per_row/QK_PQ2_0;
    std::vector<block_pq2_0> tmp((size_t)QK_PQ2_0_R8_ROWS*nblock);
    for (int r = 0; r < QK_PQ2_0_R8_ROWS; ++r) {
        quantize_row_pq2_0_ref(x + (int64_t)r*n_per_row, tmp.data() + (size_t)r*nblock, n_per_row);
    }
    repack_pq2_0_r8(QK_PQ2_0_R8_ROWS, n_per_row, tmp.data(), y, false);
}
void quantize_row_pq2_0_r8_ref(const float * x, block_pq2_0_r8 * y, int64_t n) { quantize_row_pq2_0_r8(x, (void *)y, n); }
namespace {
void dequant_block_ptq1_0(const block_ptq1_0 & x, uint8_t * y) {

    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    static const size_t ptq1_0_stages[2] = {16, 8};

    size_t j = 0;
    for (size_t s = 0; s < 2; ++s) {
        const size_t c = ptq1_0_stages[s];
        for (; j + c <= sizeof(x.qs); j += c) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < c; ++m) {
                    uint8_t q = x.qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = xi;
                }
            }
        }
    }
    for (size_t n = 0; n < 4; ++n) {
        for (size_t h = 0; h < sizeof(x.qh); ++h) {
            uint8_t q = x.qh[h] * pow3[n];
            int16_t xi = ((uint16_t) q * 3) >> 8;
            *y++ = xi;
        }
    }
}
}
void repack_ptq1_0_r8(int nrows, int n_per_row, const block_ptq1_0 * GGML_RESTRICT x, block_ptq1_0_r8 * GGML_RESTRICT y, bool online) {
    GGML_UNUSED(online);
    GGML_ASSERT(nrows % QK_PTQ1_0_R8_ROWS == 0);
    constexpr int kn4 = QK_PTQ1_0/sizeof(uint32_t);
    const int nblock = n_per_row/QK_PTQ1_0;
    const block_ptq1_0 * xr[QK_PTQ1_0_R8_ROWS];
    uint32_t quants[kn4*QK_PTQ1_0_R8_ROWS];
    uint32_t quants_r[kn4*QK_PTQ1_0_R8_ROWS];
    for (int row = 0; row < nrows; row += QK_PTQ1_0_R8_ROWS) {
        for (int k = 0; k < QK_PTQ1_0_R8_ROWS; ++k) xr[k] = x + (size_t)k*nblock;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < QK_PTQ1_0_R8_ROWS; ++k) {
                y[ib].d[k] = xr[k][ib].d;
                dequant_block_ptq1_0(xr[k][ib], (uint8_t *)(quants + k*kn4));
            }
            for (int j = 0; j < kn4; ++j) {
                for (int k = 0; k < QK_PTQ1_0_R8_ROWS; ++k) quants_r[QK_PTQ1_0_R8_ROWS*j + k] = quants[k*kn4 + j];
            }
            for (int j = 0; j < QK_PTQ1_0_R8_ROWS*QK_PTQ1_0/QK_IQ1BN; ++j) {
                IQ1BNQuantizer::pack_1bn((const int8_t *)quants_r + QK_IQ1BN*j, y[ib].qs[j]);
            }
        }
        x += (size_t)QK_PTQ1_0_R8_ROWS*nblock;
        y += nblock;
    }
}
namespace {
// ik main's PTQ1_0_R8 dequantizer calls dequantize_row_iq1_bn() on RAW packed blocks. In this fork that function was
// fixed (see its comment) to honour the iq1_bn row contract -- a 2-byte fp16 scale at the row start, skipped before the
// blocks -- so handing it raw blocks reads the first two bytes of ql[] as a scale and shifts every block by two bytes:
// test-rowint-dequant-bounds measured ptq1_0_r8 rel_rms 11759.56 against a 0.556 baseline (CUDA session 2026-10-01).
// PTQ1_0_R8 keeps its per-row scales in x[ib].d[], so the packed blocks carry NO row header: decode them header-less
// and unscaled here (values in {-1, 0, +1}); the caller applies d[k].
void dequant_iq1bn_blocks_noscale(const block_iq1_bn * x, float * y, int64_t k) {
    GGML_ASSERT(k % QK_IQ1BN == 0);
    const int nblock = (int)(k / QK_IQ1BN);
    for (int i = 0; i < nblock; ++i) {
        const uint8_t extra = x[i].extra;
        const uint8_t * ql = x[i].ql;
        for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
            for (int kk = 0; kk < 3; ++kk) {
                for (int j = 0; j < 5; ++j) {
                    uint8_t v = ql[kk]*IQ1BNQuantizer::k_mult[j];
                    int8_t vs = ((v + (v >> 1)) >> 7);
                    *y++ = (float)(vs - 1);
                }
            }
            ql += 3;
            uint8_t v = extra*IQ1BNQuantizer::k_mult[i16];
            int8_t vs = ((v + (v >> 1)) >> 7);
            *y++ = (float)(vs - 1);
        }
    }
}
}
void dequantize_row_ptq1_0_r8(const block_ptq1_0_r8 * x, float * GGML_RESTRICT y, int64_t n) {
    const int n_per_row = (int)(n/QK_PTQ1_0_R8_ROWS);
    GGML_ASSERT(n_per_row % QK_PTQ1_0 == 0);
    const int nblock = n_per_row/QK_PTQ1_0;
    float ftmp[QK_PTQ1_0*QK_PTQ1_0_R8_ROWS];
    for (int ib = 0; ib < nblock; ++ib) {
        dequant_iq1bn_blocks_noscale(x[ib].qs, ftmp, QK_PTQ1_0*QK_PTQ1_0_R8_ROWS);
        for (int k = 0; k < QK_PTQ1_0_R8_ROWS; ++k) {
            float d = GGML_FP16_TO_FP32(x[ib].d[k]);
            float * yr = y + (int64_t)k*n_per_row + ib*QK_PTQ1_0;
            for (int j = 0; j < QK_PTQ1_0/4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    yr[4*j + i] = d * ftmp[32*j + 4*k + i];
                }
            }
        }
    }
}
void quantize_row_ptq1_0_r8(const float * x, void * vy, int64_t n) {
    auto y = (block_ptq1_0_r8 *)vy;
    const int n_per_row = (int)(n/QK_PTQ1_0_R8_ROWS);
    GGML_ASSERT(n_per_row % QK_PTQ1_0 == 0);
    const int nblock = n_per_row/QK_PTQ1_0;
    std::vector<block_ptq1_0> tmp((size_t)QK_PTQ1_0_R8_ROWS*nblock);
    for (int r = 0; r < QK_PTQ1_0_R8_ROWS; ++r) {
        quantize_row_ptq1_0_ref(x + (int64_t)r*n_per_row, tmp.data() + (size_t)r*nblock, n_per_row);
    }
    repack_ptq1_0_r8(QK_PTQ1_0_R8_ROWS, n_per_row, tmp.data(), y, false);
}
void quantize_row_ptq1_0_r8_ref(const float * x, block_ptq1_0_r8 * y, int64_t n) { quantize_row_ptq1_0_r8(x, (void *)y, n); }

// ik: extern since ks-encoder.cu (GPU KS encoder) repacks IQ5_KS rows through it (moved out of
// iqk_quantize_shared.h, which two TUs include)
void repack_iq5_ks(int nrows, int n_per_row, const block_iq5_ks * x, block_iq5_ks_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    const block_iq5_ks * x4[4];
    uint8_t L[QK_K];
    char * cy = (char *)y;
    const char * cx = (const char *)x;
    for (int row = 0; row < nrows; row += 4) {
        float * dptr = (float *)cy;
        block_iq5_ks_r4 * y = (block_iq5_ks_r4 *)(dptr + 4);
        for (int k = 0; k < 4; ++k) {
            auto dk = (const float *)(cx + k*row_size);
            dptr[k] = dk[0];
            x4[k] = (const block_iq5_ks *)(dk + 1);
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                convert_iq5_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);  //  0....3 +  8...11 from each row
                        y[ibl].qs[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);  // 16...19 + 24...27 from each row
                        y[ibl].qs[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);  //  4....7 + 12...15 from each row
                        y[ibl].qs[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);  // 20...23 + 28...31 from each row
                        y[ibl].qh[16*ib+4*k+i   ] = ((L[32*ib+i+ 0] >> 4) << 0) | ((L[32*ib+i+ 8] >> 4) << 1) | ((L[32*ib+i+16] >> 4) << 2) | ((L[32*ib+i+24] >> 4) << 3)
                                                  | ((L[32*ib+i+ 4] >> 4) << 4) | ((L[32*ib+i+12] >> 4) << 5) | ((L[32*ib+i+20] >> 4) << 6) | ((L[32*ib+i+28] >> 4) << 7);
                    }
                }
            }
        }
        cx += 4*row_size;
        cy += 4*row_size;
    }
}

// ik main (cea2d61ff) table accessors for the GPU KS encoder (ggml-cuda/ks-encoder.cu). In ik they sit in
// iqk_quantize.cpp; here the IQ3/IQ5 tables are file-local to this TU and ggml-cuda links ggml-base, so they live here.
int iqk_ks_index(int type, const int8_t ** index) {
    if (type == GGML_TYPE_IQ4_KS) { *index = iq4nl_index; return 241; }
    if (type == GGML_TYPE_IQ3_KS) { *index = iq3nl_index; return 111; }
    if (type == GGML_TYPE_IQ5_KS) { *index = iq5nl_index; return 247; }
    return 0;
}

int iqk_ks_simple_index(int type, const int8_t ** index) {
    if (type == GGML_TYPE_IQ4_KS) { *index = k_simple_index_iq4nl; return 256; }
    if (type == GGML_TYPE_IQ3_KS) { *index = k_simple_index_iq3nl; return 128; }
    if (type == GGML_TYPE_IQ5_KS) { *index = k_simple_index_iq5nl; return 256; }
    return 0;
}
