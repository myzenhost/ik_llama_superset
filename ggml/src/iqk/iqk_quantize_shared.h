//
// HALF 2 of S69-GGML-BASE-DLL-CANNOT-LINK-CIRCULAR-ACROSS-A-DLL-BOUNDARY-2026-09-15-01.
//
// Textual header shared by iqk_quantize.cpp (ggml-cpu) and iqk_quantize_base.cpp (ggml-base).
// Holds the internal-linkage helpers (anonymous namespace / types / templates) that BOTH translation
// units need after the type_traits[] .to_float / .from_float_ref implementations moved to ggml-base.
// Each including translation unit gets its own private copy (internal linkage), so no symbol is
// exported from here. Bodies are the original ones, moved without edits.
// Include AFTER the preamble of the including .cpp (ggml-common.h with GGML_COMMON_IMPL_C, iqk_quantize.h, iqk_config.h, <vector> ...).
//
#pragma once
// The int8_t/int16_t/uint32_t tables below need <cstdint> whatever the includer's preamble did; included here so the
// header's own fixed-width types resolve (a standalone `#include` of this file reported int8_t undeclared at :120+).
#include <cstdint>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif


// External-linkage functions now defined in iqk_quantize_base.cpp (ggml-base) and still called from iqk_quantize.cpp (ggml-cpu).
size_t quantize_iq1_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const quantize_user_data *);

namespace {
inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
// Nearest IQ5_K/IQ5_KS non-linear value. Moved verbatim from iqk_quantize_base.cpp: the scalar reference quantizer
// (ggml-base) and the AVX2 bulk quantizer (iqk_quantize.cpp, ggml-cpu) both encode with it.
const int8_t iq5nl_index[248] = {
     0,  0,  0,  0,  0,  0, 32,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 33, 33,  2,  2,  2,  2,  2,  2,  2,  2,  2, 34, 34,  3,  3,
     3,  3,  3,  3,  3,  3, 35, 35,  4,  4,  4,  4,  4,  4,  4, 36, 36,  5,  5,  5,  5,  5,  5,  5, 37, 37,  6,  6,  6,  6,  6,  6,
     6, 38,  7,  7,  7,  7,  7,  7, 39, 39,  8,  8,  8,  8,  8, 40, 40,  9,  9,  9,  9,  9, 41, 41, 10, 10, 10, 10, 10, 42, 11, 11,
    11, 11, 11, 43, 12, 12, 12, 12, 12, 44, 13, 13, 13, 13, 13, 45, 14, 14, 14, 14, 14, 46, 15, 15, 15, 15, 47, 47, 16, 16, 16, 16,
    48, 17, 17, 17, 17, 17, 49, 18, 18, 18, 18, 18, 50, 19, 19, 19, 19, 19, 51, 20, 20, 20, 20, 20, 52, 21, 21, 21, 21, 21, 53, 53,
    22, 22, 22, 22, 22, 54, 54, 23, 23, 23, 23, 23, 23, 55, 24, 24, 24, 24, 24, 24, 24, 56, 25, 25, 25, 25, 25, 25, 25, 57, 57, 26,
    26, 26, 26, 26, 26, 26, 58, 58, 27, 27, 27, 27, 27, 27, 27, 27, 59, 28, 28, 28, 28, 28, 28, 28, 28, 28, 60, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 61, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 62, 31, 31, 31, 31, 31, 31
};
inline int best_index_iq5nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 247) return ix < 0 ? 0 : 31;
    ix = iq5nl_index[ix];
    return ix < 32 ? ix : x - values[ix-32] < values[ix-31] - x ? ix-32 : ix-31;
}
typedef void (*quantize_func_t)(const float * src, void * qdata, int n_per_row, const float * imatrix, const quantize_user_data * user_data);
struct QHelper {
    QHelper(const float * imatrix, const quantize_user_data * user_data, int n_per_row, int block_size) :
        m_imatrix(imatrix), m_user_data(user_data), m_n_per_row(n_per_row), m_block_size(block_size) {
        if (m_imatrix) {
            m_weight.resize(m_n_per_row);
        }
    }
    const float * row_weights(const float * x) {
        constexpr float kEps  = 1e-9f;
        constexpr float kEps2 = kEps*kEps;
        if (!m_imatrix) return m_imatrix;
        int nblock = m_n_per_row / m_block_size;
        for (int ib = 0; ib < nblock; ++ib) {
            auto wb_in = m_imatrix + ib*m_block_size;
            auto xb = x + ib*m_block_size;
            auto wb = m_weight.data() + ib*m_block_size;
            float sumw2 = 0, sumx2 = 0, sumwx = 0;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = wb_in[j];
                sumw2 += wb[j]*wb[j];
                sumx2 += xb[j]*xb[j];
                sumwx += wb[j]*std::abs(xb[j]);
            }
            if (sumw2 > m_block_size*kEps2 && sumx2 > m_block_size*kEps2 && sumwx > m_block_size*kEps2) continue;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = kEps;
            }
        }
        return m_weight.data();
    }
    template <typename Func>
    void quantize(int nrows, const float * src, void * dst, int row_size, const Func& qfunc) {
        auto cdst = (char *)dst;
        for (int row = 0; row < nrows; ++row) {
            auto weights = row_weights(src);
            qfunc(src, cdst, m_n_per_row, weights, m_user_data);
            src  += m_n_per_row;
            cdst += row_size;
        }
    }
private:
    const float * m_imatrix;
    const quantize_user_data * m_user_data;
    const int     m_n_per_row;
    const int     m_block_size;
    std::vector<float> m_weight;
};
float make_qx_quants(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax-1, l));
        L[i] = l + nmax;
        sumlx += qw[i]*x[i]*l;
        suml2 += qw[i]*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax-1, l));
            sumlx += qw[i]*x[i]*l;
            suml2 += qw[i]*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}
struct IQ1BNQuantizer {
    int8_t L[QK_IQ1BN];
    void quantize_one_row_1bn(const float * src, block_iq1_bn * y, int n_per_row, const float * imatrix);
    void quantize_one_row_2bn(const float * src, block_iq2_bn * y, int n_per_row, const float * imatrix);
    static inline float row_max(int n_per_row, const float * src) {
        float max_in_row = 0;
        for (int j = 0; j < n_per_row; ++j) {
            float ax = fabsf(src[j]);
            max_in_row = std::max(max_in_row, ax);
        }
        return max_in_row;
    }
    static void pack_1bn(const int8_t * q, block_iq1_bn & y);
    // The Makefile has issues dwaling with this?
    //static constexpr uint8_t k_mult[5] = {81, 27, 9, 3, 1};
    static const uint8_t k_mult[5];
};
const uint8_t IQ1BNQuantizer::k_mult[5] = {81, 27, 9, 3, 1};
void IQ1BNQuantizer::pack_1bn(const int8_t * q, block_iq1_bn & y) {
    static const int k_nb[6] = {1, 3, 9, 27, 81, 243};
    std::memset(&y, 0, sizeof(block_iq1_bn));
    int v13 = 0;
    for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
        for (int k = 0; k < 3; ++k) {
            int idx = 0;
            for (int j = 0; j < 5; ++j) {
                idx += k_nb[j]*q[16*i16 + 5*k + j];
            }
            idx = (256*idx + k_nb[5] - 1)/k_nb[5];
            y.ql[3*i16 + k] = idx;
        }
        v13 += k_nb[i16]*q[16*i16 + 15];
    }
    y.extra = (256*v13 + k_nb[5] - 1)/k_nb[5];
}
void IQ1BNQuantizer::quantize_one_row_1bn(const float * src, block_iq1_bn * y, int n_per_row, const float * imatrix) {

    //static const int k_nb[6] = {1, 3, 9, 27, 81, 243};
    (void)imatrix;

    const int nblock = n_per_row/QK_IQ1BN;

    ggml_half * dptr = (ggml_half *)y;
    y = (block_iq1_bn *)(dptr + 1);

    float max = 0;
    for (int j = 0; j < n_per_row; ++j) max = std::max(max, fabsf(src[j]));
    ggml_half d = GGML_FP32_TO_FP16(max);
    std::memcpy(dptr, &d, sizeof(d));

    float thresh = 0.5f*max;

    for (int ib = 0; ib < nblock; ++ib) {
        std::memset(&y[ib], 0, sizeof(block_iq1_bn));
        auto xb = src + ib*QK_IQ1BN;
        for (int j = 0; j < QK_IQ1BN; ++j) {
            float v = xb[j];
            L[j] = fabsf(v) < thresh ? 1 : v < 0 ? 0 : 2;
        }
        pack_1bn(L, y[ib]);
        //int v13 = 0;
        //for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
        //    for (int k = 0; k < 3; ++k) {
        //        int idx = 0;
        //        for (int j = 0; j < 5; ++j) {
        //            float v = xb[16*i16 + 5*k + j];
        //            int q = fabsf(v) < thresh ? 1 : v < 0 ? 0 : 2;
        //            idx += k_nb[j]*q;
        //        }
        //        idx = (256*idx + k_nb[5] - 1)/k_nb[5];
        //        y[ib].ql[3*i16 + k] = idx;
        //    }
        //    float v = xb[16*i16 + 15];
        //    int q = fabsf(v) < thresh ? 1 : v < 0 ? 0 : 2;
        //    v13 += k_nb[i16]*q;
        //}
        //y[ib].extra = (256*v13 + k_nb[5] - 1)/k_nb[5];
    }
}
void IQ1BNQuantizer::quantize_one_row_2bn(const float * src, block_iq2_bn * y, int n_per_row, const float * imatrix) {

    (void)imatrix;

    const int nblock = n_per_row/QK_IQ1BN;

    constexpr int Nj = QK_IQ1BN/4;

    float max = 0;
    for (int j = 0; j < n_per_row; ++j) max = std::max(max, fabsf(src[j]));

    float * dptr = (float *)y;
    *dptr = max;
    y = (block_iq2_bn *)(dptr + 1);
    float thresh = 0.5f*max;

    for (int ib = 0; ib < nblock; ++ib) {
        auto xb = src + QK_IQ1BN*ib;
        for (int j = 0; j < QK_IQ1BN; ++j) {
            L[j] = fabsf(xb[j]) < thresh ? 1 : xb[j] < 0 ? 0 : 2;
        }
        for (int j = 0; j < Nj; ++j) {
            y[ib].qs[j] = L[j] | (L[j + Nj] << 2) | (L[j + 2*Nj] << 4) | (L[j + 3*Nj] << 6);
        }
    }
}
inline int best_index_iq2nl(const int8_t * values, float x) {
    int idx = x < values[1] ? 0 : x > values[2] ? 2 : 1;
    return x - values[idx] < values[idx+1] - x ? idx : idx + 1;
}
const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}
// ik main (cea2d61ff): simple-rounding index table for the AVX2 IQ4 scale search (iqk_quantize.cpp) and for
// iqk_ks_simple_index() (iqk_quantize_base.cpp, the GPU KS encoder's table source); both TUs include this header.
static const int8_t k_simple_index_iq4nl[256] = {
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5,
     5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,10,10,10,10,10,10,10,10,
    10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,
    12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,
    14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
};
inline int best_index_mxfp4(float d, const int8_t * values, float x) {
    float best = std::abs(x - d*values[0]);
    int index = 0;
    for (int j = 1; j < 16; ++j) {
        float diff = std::abs(x - d*values[j]);
        if (diff < best) { best = diff; index = j; }
    }
    return index;
}
static void quantize_row_mxfp4_impl(int n_per_row, const float * x, char * cy,
        [[maybe_unused]] float * weight,
        const int8_t * values,
        const float * quant_weights,
        [[maybe_unused]] const int ntry,
        const bool search = false) {

    GGML_ASSERT(n_per_row % QK_MXFP4 == 0);

    block_mxfp4 * y = (block_mxfp4 *)cy;

    for (int ib = 0; ib < n_per_row/QK_MXFP4; ++ib) {
        memset(&y[ib], 0, sizeof(block_mxfp4));
        const float * xb = x + ib*QK_MXFP4;
        float amax = 0;
        for (int j = 0; j < QK_MXFP4; ++j) {
            float ax = fabsf(xb[j]);
            amax = std::max(amax, ax);
        }
        if (!amax) {
            continue;
        }
        uint8_t e = (uint8_t) (floorf(log2f(amax)) - 2 + 127);
        if (search) {
            // S69-BURN-PICK-FP4-BLOCK-SCALE-CODE-SEARCH (2026-10-05), MXFP4 half: the exponent of the OCP floor rule is a search
            // variable, not a fact. Evaluate e-1, e, e+1, quantise the block against each DECODED scale and keep the code with the
            // smallest importance-weighted squared error; the nominal code goes first and only a strictly smaller error replaces it,
            // so the result is never worse than the formula under the same weights. Decode and format are unchanged.
            // Measured in simulation on real weights (temp/_agA_1005_fp4_scale_real_weights.txt): -5.5% block error.
            const float * wb = quant_weights ? quant_weights + ib*QK_MXFP4 : nullptr;
            float best_err = INFINITY;
            uint8_t best_e = e;
            const int cand[3] = { (int) e, (int) e - 1, (int) e + 1 };
            for (int ci = 0; ci < 3; ++ci) {
                if (cand[ci] < 1 || cand[ci] > 254) {
                    continue;
                }
                const float dc = GGML_E8M0_TO_FP32_HALF((uint8_t) cand[ci]);
                float err = 0.0f;
                for (int j = 0; j < QK_MXFP4; ++j) {
                    const int   q  = best_index_mxfp4(dc, values, xb[j]);
                    const float df = xb[j] - dc*values[q];
                    err += (wb ? wb[j] : 1.0f) * df * df;
                }
                if (err < best_err) {
                    best_err = err;
                    best_e   = (uint8_t) cand[ci];
                }
            }
            e = best_e;
        }
        const float d = GGML_E8M0_TO_FP32_HALF(e);
        y[ib].e = e;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            uint8_t v0 = best_index_mxfp4(d, values, xb[j]);
            uint8_t v1 = best_index_mxfp4(d, values, xb[j+QK_MXFP4/2]);
            y[ib].qs[j] = v0 | (v1 << 4);
        }
    }
}
}
static void repack_iq4_nl(int nrows, int n_per_row, const block_iq4_nl * x, block_iq4_nl_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK4_NL == 0);
    int nblock = n_per_row/QK4_NL;
    const block_iq4_nl * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 4; ++k) y[ib].d[k] = x4[k][ib].d;
            for (int k = 0; k < 4; ++k) for (int i = 0; i < 4; ++i) {
                y[ib].qs[4*k+i+ 0] = (x4[k][ib].qs[i+0] & 0xf) | ((x4[k][ib].qs[i+ 8] & 0x0f) << 4);  //  0....3 +  8...11 from each row
                y[ib].qs[4*k+i+16] = (x4[k][ib].qs[i+0] >>  4) | ((x4[k][ib].qs[i+ 8] & 0xf0));       // 16...19 + 24...27 from each row
                y[ib].qs[4*k+i+32] = (x4[k][ib].qs[i+4] & 0xf) | ((x4[k][ib].qs[i+12] & 0x0f) << 4);  //  4....7 + 12...15 from each row
                y[ib].qs[4*k+i+48] = (x4[k][ib].qs[i+4] >>  4) | ((x4[k][ib].qs[i+12] & 0xf0));       // 20...23 + 28...31 from each row
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
static inline void convert_q5_0(const block_q5_0& x, uint8_t * L) {
    uint32_t qh;
    memcpy(&qh, x.qh, sizeof(qh));

    for (int j = 0; j < QK5_0/2; ++j) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

        L[j          ] = (x.qs[j] & 0x0F) | xh_0;
        L[j + QK4_0/2] = (x.qs[j] >>   4) | xh_1;
    }
}
static void repack_q5_0(int nrows, int n_per_row, const block_q5_0 * x, block_q5_0_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK5_0 == 0);
    int nblock = n_per_row/QK5_0;
    const block_q5_0 * x4[4];
    uint8_t L[QK5_0];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            std::memset(y[ib].qh, 0, QK5_0/2);
            for (int k = 0; k < 4; ++k) {
                y[ib].d[k] = x4[k][ib].d;
                convert_q5_0(x4[k][ib], L);
                for (int l = 0; l < 4; ++l) {
                    int l1 = 4*(l/2) + 16*(l%2), l2 = l1 + 8;
                    for (int i = 0; i < 4; ++i) {
                        y[ib].qs[4*k+i+16*l] = (L[i + l1] & 0xf) | ((L[i + l2] & 0xf) << 4);
                        y[ib].qh[4*k+i] |= ((L[i + l1] >> 4) | ((L[i + l2] >> 4) << 4)) << l;
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
static inline void convert_q6_0(const block_q6_0& x, uint8_t * L) {

    for (int j = 0; j < QK6_0/2; ++j) {
        const uint8_t h = x.qh[j%(QK6_0/4)] >> 4*(j/(QK6_0/4));
        L[j          ] = (x.qs[j] & 0x0F) | ((h << 4) & 0x30);
        L[j + QK6_0/2] = (x.qs[j] >>   4) | ((h << 2) & 0x30);
    }
}
static void repack_q6_0(int nrows, int n_per_row, const block_q6_0 * x, block_q6_0_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK5_0 == 0);
    int nblock = n_per_row/QK6_0;
    const block_q6_0 * x4[4];
    uint8_t L[QK6_0];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            std::memset(y[ib].qh, 0, QK6_0);
            for (int k = 0; k < 4; ++k) {
                y[ib].d[k] = x4[k][ib].d;
                convert_q6_0(x4[k][ib], L);
                for (int l = 0; l < 4; ++l) {
                    int l1 = 4*(l/2) + 16*(l%2), l2 = l1 + 8;
                    for (int i = 0; i < 4; ++i) {
                        y[ib].qs[4*k+i+16*l] = (L[i + l1] & 0xf) | ((L[i + l2] & 0xf) << 4);
                        y[ib].qh[4*k+i+16*(l%2)] |= ((L[i + l1] >> 4) | ((L[i + l2] >> 4) << 4)) << 2*(l/2);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
static void repack_iq4_xs(int nrows, int n_per_row, const block_iq4_xs * x, block_iq4_xs_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq4_xs * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 8; ++k) {
                y[ibl].d[k] = x8[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    uint8_t sl = (x8[k][ibl].scales_l[ib/2] >> 4*(ib%2)) & 0xf;
                    uint8_t sh = (x8[k][ibl].scales_h >> 2*ib) & 3;
                    int i = 8*ib + k;
                    y[ibl].scales_l[i%32] |= (sl << 4*(i/32));
                    y[ibl].scales_h[i%16] |= (sh << 2*(i/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[128*ib+4*k+i+ 0] = (x8[k][ibl].qs[16*ib+i+0] & 0xf) | ((x8[k][ibl].qs[16*ib+i+ 4] & 0xf) << 4);
                        y[ibl].qs[128*ib+4*k+i+32] = (x8[k][ibl].qs[16*ib+i+8] & 0xf) | ((x8[k][ibl].qs[16*ib+i+12] & 0xf) << 4);
                        y[ibl].qs[128*ib+4*k+i+64] = (x8[k][ibl].qs[16*ib+i+0] >>  4) | ((x8[k][ibl].qs[16*ib+i+ 4] >>  4) << 4);
                        y[ibl].qs[128*ib+4*k+i+96] = (x8[k][ibl].qs[16*ib+i+8] >>  4) | ((x8[k][ibl].qs[16*ib+i+12] >>  4) << 4);
                    }
                }
            }
        }
        x += 8*nblock;
        y += nblock;
    }
}
namespace {
void repack_iq2_bn(int nrows, int n_per_row, const char * x, char * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_IQ1BN == 0);
    int nblock = n_per_row/QK_IQ1BN;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_BN, n_per_row);
    const uint8_t * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        float * dr4 = (float *)(y + 4*row*row_size);
        for (int k = 0; k < 4; ++k) {
            const float * dptr = (const float *)(x + (row + k)*row_size);
            dr4[k] = *dptr;
            x4[k] = (const uint8_t *)(dptr + 1);
        }
        uint8_t * y4 = (uint8_t *)(dr4 + 4);
        //std::memset(y4, 0, n_per_row);
        for (int ib = 0; ib < nblock; ++ib) {
            //  0...3 from rows 0...3 go to 1st 2 bits of  0...15
            // 16..19 from rows 0...3 go to 1st 2 bits of 16...31
            // 32..35 from rows 0...3 go to 1st 2 bits of 32...47
            // 48..51 from rows 0...3 go to 1st 2 bits of 48...63
            //  4...7 from rows 0...3 go to 2nd 2 bits of  0...15
            // 20..23 from rows 0...3 go to 2nd 2 bits of 16...31
            // 36..39 from rows 0...3 go to 2nd 2 bits of 32...47
            // 52..55 from rows 0...3 go to 2nd 2 bits of 48...63
            //  8..11 from rows 0...3 go to 3rd 2 bits of  0...15
            // 24..27 from rows 0...3 go to 3rd 2 bits of 16...31
            // 40..43 from rows 0...3 go to 3rd 2 bits of 32...47
            // 56..59 from rows 0...3 go to 3rd 2 bits of 48...63
            // 12..15 from rows 0...3 go to 4th 2 bits of  0...15
            // 28..31 from rows 0...3 go to 4th 2 bits of 16...31
            // 44..47 from rows 0...3 go to 4th 2 bits of 32...47
            // 60..63 from rows 0...3 go to 4th 2 bits of 48...63
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                    y4[64*ib + 4*k + i + 16*l] = (((x4[k][16*ib + i +  0] >> 2*l) & 3) << 0) |
                                                 (((x4[k][16*ib + i +  4] >> 2*l) & 3) << 2) |
                                                 (((x4[k][16*ib + i +  8] >> 2*l) & 3) << 4) |
                                                 (((x4[k][16*ib + i + 12] >> 2*l) & 3) << 6);
                    //y4[64*ib + 4*k + i +  0] |= (x4[k][16*ib + i] >> 0) & 3;
                    //y4[64*ib + 4*k + i + 16] |= (x4[k][16*ib + i] >> 2) & 3;
                    //y4[64*ib + 4*k + i + 32] |= (x4[k][16*ib + i] >> 4) & 3;
                    //y4[64*ib + 4*k + i + 48] |= (x4[k][16*ib + i] >> 6) & 3;
                    //y4[64*ib + 4*k + i +  0] |= ((x4[k][16*ib + i + 4] >> 0) & 3) << 2;
                    //y4[64*ib + 4*k + i + 16] |= ((x4[k][16*ib + i + 4] >> 2) & 3) << 2;
                    //y4[64*ib + 4*k + i + 32] |= ((x4[k][16*ib + i + 4] >> 4) & 3) << 2;
                    //y4[64*ib + 4*k + i + 48] |= ((x4[k][16*ib + i + 4] >> 6) & 3) << 2;
                    //y4[64*ib + 4*k + i +  0] |= ((x4[k][16*ib + i + 8] >> 0) & 3) << 4;
                    //y4[64*ib + 4*k + i + 16] |= ((x4[k][16*ib + i + 8] >> 2) & 3) << 4;
                    //y4[64*ib + 4*k + i + 32] |= ((x4[k][16*ib + i + 8] >> 4) & 3) << 4;
                    //y4[64*ib + 4*k + i + 48] |= ((x4[k][16*ib + i + 8] >> 6) & 3) << 4;
                    //y4[64*ib + 4*k + i +  0] |= ((x4[k][16*ib + i + 12] >> 0) & 3) << 6;
                    //y4[64*ib + 4*k + i + 16] |= ((x4[k][16*ib + i + 12] >> 2) & 3) << 6;
                    //y4[64*ib + 4*k + i + 32] |= ((x4[k][16*ib + i + 12] >> 4) & 3) << 6;
                    //y4[64*ib + 4*k + i + 48] |= ((x4[k][16*ib + i + 12] >> 6) & 3) << 6;
                }
            }
        }
    }
}
inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
inline void convert_q4_k(const block_q4_K& x, uint8_t * L, uint8_t * Ld, uint8_t * Lm) {
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        get_scale_min_k4(2*ib64+0, x.scales, Ld[2*ib64+0], Lm[2*ib64+0]);
        get_scale_min_k4(2*ib64+1, x.scales, Ld[2*ib64+1], Lm[2*ib64+1]);
        for (int j = 0; j < 32; ++j) {
            L[64*ib64+j+ 0] = x.qs[32*ib64+j] & 0xf;
            L[64*ib64+j+32] = x.qs[32*ib64+j] >>  4;
        }
    }
}
}
static void repack_q4_k(int nrows, int n_per_row, const block_q4_K * x, block_q4_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q4_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/32], Lm[QK_K/32];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                convert_q4_k(x4[k][ibl], L, Ld, Lm);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales_l[4*ib+k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4);
                    uint8_t h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2);
                    y[ibl].scales_h[(4*ib+k)%16] |= (h << 4*((4*ib+k)/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = L[32*ib+i+ 0] | (L[32*ib+i+ 8] << 4);
                        y[ibl].qs[64*ib+4*k+i+16] = L[32*ib+i+16] | (L[32*ib+i+24] << 4);
                        y[ibl].qs[64*ib+4*k+i+32] = L[32*ib+i+ 4] | (L[32*ib+i+12] << 4);
                        y[ibl].qs[64*ib+4*k+i+48] = L[32*ib+i+20] | (L[32*ib+i+28] << 4);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
namespace {
inline void convert_q6_k(const block_q6_K& x, uint8_t * L) {
    const uint8_t * ql = x.ql;
    const uint8_t * qh = x.qh;

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            L[n + l +  0] = (ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            L[n + l + 32] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            L[n + l + 64] = (ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4);
            L[n + l + 96] = (ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4);
        }
        ql += 64;
        qh += 32;
    }
}
}
static void repack_q6_k(int nrows, int n_per_row, const block_q6_K * x, block_q6_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q6_K * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                convert_q6_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[8*ib+k+0] = x4[k][ibl].scales[2*ib+0];
                    y[ibl].scales[8*ib+k+4] = x4[k][ibl].scales[2*ib+1];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].ql[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);
                        y[ibl].qh[32*ib+4*k+i+ 0] = (L[32*ib+i+ 0] >> 4) | ((L[32*ib+i+ 8] >> 4) << 2) | ((L[32*ib+i+ 4] >> 4) << 4) | ((L[32*ib+i+12] >> 4) << 6);
                        y[ibl].qh[32*ib+4*k+i+16] = (L[32*ib+i+16] >> 4) | ((L[32*ib+i+24] >> 4) << 2) | ((L[32*ib+i+20] >> 4) << 4) | ((L[32*ib+i+28] >> 4) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
namespace {
inline void convert_q5_k(const block_q5_K& x, uint8_t * L, uint8_t * Ld, uint8_t * Lm) {
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        get_scale_min_k4(2*ib64+0, x.scales, Ld[2*ib64+0], Lm[2*ib64+0]);
        get_scale_min_k4(2*ib64+1, x.scales, Ld[2*ib64+1], Lm[2*ib64+1]);
        for (int j = 0; j < 32; ++j) {
            L[64*ib64+j+ 0] = (x.qs[32*ib64+j] & 0xf) | (((x.qh[j] >> (2*ib64+0)) & 1) << 4);
            L[64*ib64+j+32] = (x.qs[32*ib64+j] >>  4) | (((x.qh[j] >> (2*ib64+1)) & 1) << 4);
        }
    }
}
}
static void repack_q5_k(int nrows, int n_per_row, const block_q5_K * x, block_q5_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q5_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/32], Lm[QK_K/32];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                convert_q5_k(x4[k][ibl], L, Ld, Lm);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales_l[4*ib+k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4);
                    uint8_t h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2);
                    y[ibl].scales_h[(4*ib+k)%16] |= (h << 4*((4*ib+k)/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] >> 4) << 0) | ((L[32*ib+i+ 8] >> 4) << 1) | ((L[32*ib+i+ 4] >> 4) << 2) | ((L[32*ib+i+12] >> 4) << 3) |
                                                    ((L[32*ib+i+16] >> 4) << 4) | ((L[32*ib+i+24] >> 4) << 5) | ((L[32*ib+i+20] >> 4) << 6) | ((L[32*ib+i+28] >> 4) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
namespace {
inline void convert_q3_k(const block_q3_K& x, uint8_t * L, uint8_t * Ld) {
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    memcpy(aux, x.scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(Ld, aux, 16);

    const uint8_t * q = x.qs;
    const uint8_t * hm = x.hmask;
    uint8_t m = 1;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 32; ++l) {
                *L++ = ((q[l] >> shift) & 3) + ((hm[l] & m) ? 4 : 0);
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}
}
static void repack_q3_k(int nrows, int n_per_row, const block_q3_K * x, block_q3_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q3_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/16];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                convert_q3_k(x4[k][ibl], L, Ld);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    int is = 8*ib+k;
                    y[ibl].scales_l[is%32] |= (Ld[2*ib+0] & 0xf) << 4*(is/32);
                    y[ibl].scales_h[is%16] |= (Ld[2*ib+0] >>  4) << 2*(is/16);
                    is += 4;
                    y[ibl].scales_l[is%32] |= (Ld[2*ib+1] & 0xf) << 4*(is/32);
                    y[ibl].scales_h[is%16] |= (Ld[2*ib+1] >>  4) << 2*(is/16);
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0]  >> 2) << 0) | ((L[32*ib+i+ 4]  >> 2) << 1) | ((L[32*ib+i+ 8]  >> 2) << 2) | ((L[32*ib+i+12]  >> 2) << 3)
                                                  | ((L[32*ib+i+16]  >> 2) << 4) | ((L[32*ib+i+20]  >> 2) << 5) | ((L[32*ib+i+24]  >> 2) << 6) | ((L[32*ib+i+28]  >> 2) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
namespace {
inline void convert_q2_k(const block_q2_K& x, uint8_t * L) {

    const uint8_t * qs = x.qs;
    for (int n = 0; n < QK_K; n += 128) {
        for (int j = 0; j < 32; ++j) {
            L[n + j +  0] = (qs[j] >> 0) & 0x3;
            L[n + j + 32] = (qs[j] >> 2) & 0x3;
            L[n + j + 64] = (qs[j] >> 4) & 0x3;
            L[n + j + 96] = (qs[j] >> 6) & 0x3;
        }
        qs += 32;
    }
}
}
static void repack_q2_k(int nrows, int n_per_row, const block_q2_K * x, block_q2_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q2_K * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                for (int ib = 0; ib < QK_K/16; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                }
                convert_q2_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
namespace {
template <typename Block>
inline void convert_iq5_k(const Block& x, uint8_t * L) {
    const uint8_t * qs = x.qs;
    const uint8_t * qh = x.qh;
    int shift = 0;
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        for (int j = 0; j < 16; ++j) {
            L[j+ 0] = (qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4);
            L[j+16] = (qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4);
            L[j+32] = (qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3);
            L[j+48] = (qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3);
        }
        L  += 64;
        qs += 32;
        shift += 2;
        if (shift == 8) { qh += 32; shift = 0; }
    }
}
}
static void repack_iq5_k(int nrows, int n_per_row, const block_iq5_k * x, block_iq5_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq5_k * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].extra, 0, 8);
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto extra = x4[k][ibl].extra;
                convert_iq5_k(x4[k][ibl], L);
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
        x += 4*nblock;
        y += nblock;
    }
}
// repack_iq5_ks: extern (iqk_quantize.h, ggml-cuda/ks-encoder.cu); its single definition is in
// iqk_quantize_base.cpp -- a body here would be defined once per including TU.
namespace {
inline void convert_iq3_k(const block_iq3_k& x, uint8_t * L) {
    const uint8_t * qs = x.qs;
    const uint8_t * qh = x.qh;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        int shift_l = 2*(ib32%4);
        int shift_h = ib32%8;
        for (int j = 0; j < 16; ++j) {
            L[j+ 0] = ((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2);
            L[j+16] = ((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2);
        }
        L += 32;
        if (shift_l == 6) qs += 32;
    }
}
}
static void repack_iq3_k(int nrows, int n_per_row, const block_iq3_k * x, block_iq3_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq3_k * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].extra, 0, 8);
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/32);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto extra = x4[k][ibl].extra;
                uint16_t sh  = x4[k][ibl].scales_h;
                convert_iq3_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    if (extra & 1) y[ibl].extra[k+0] |= (1 << ib);
                    if (extra & 2) y[ibl].extra[k+4] |= (1 << ib);
                    extra >>= 2;
                    uint8_t sl1 = x4[k][ibl].scales_l[ib] & 0xf;
                    uint8_t sl2 = x4[k][ibl].scales_l[ib] >>  4;
                    uint8_t sh1 = (sh >> 0) & 1;
                    uint8_t sh2 = (sh >> 1) & 1;
                    sh >>= 2;
                    int i = 8*ib + k;
                    y[ibl].scales_l[i%32] |= (sl1 << 4*(i/32));
                    y[ibl].scales_h[i%8 ] |= (sh1 << (i/8));
                    i += 4;
                    y[ibl].scales_l[i%32] |= (sl2 << 4*(i/32));
                    y[ibl].scales_h[i%8 ] |= (sh2 << (i/8));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0]  >> 2) << 0) | ((L[32*ib+i+ 4]  >> 2) << 1) | ((L[32*ib+i+ 8]  >> 2) << 2) | ((L[32*ib+i+12]  >> 2) << 3)
                                                  | ((L[32*ib+i+16]  >> 2) << 4) | ((L[32*ib+i+20]  >> 2) << 5) | ((L[32*ib+i+24]  >> 2) << 6) | ((L[32*ib+i+28]  >> 2) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
namespace {
inline void convert_iq2_k(const block_iq2_k& x, uint8_t * L) {
    const uint8_t * qs = x.qs;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        int shift_l = 2*(ib32%4);
        for (int j = 0; j < 16; ++j) {
            L[j+ 0] = ((qs[j+ 0] >> shift_l) & 3);
            L[j+16] = ((qs[j+16] >> shift_l) & 3);
        }
        L += 32;
        if (shift_l == 6) qs += 32;
    }
}
}
static void repack_iq2_k(int nrows, int n_per_row, const block_iq2_k * x, block_iq2_k_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_k * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].extra, 0, 8);
            std::memset(y[ibl].scales, 0, QK_K/8);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto extra = x4[k][ibl].extra;
                convert_iq2_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    if (extra & 1) y[ibl].extra[k+0] |= (1 << ib);
                    if (extra & 2) y[ibl].extra[k+4] |= (1 << ib);
                    extra >>= 2;
                    uint8_t sl1 = x4[k][ibl].scales[ib] & 0xf;
                    uint8_t sl2 = x4[k][ibl].scales[ib] >>  4;
                    int i = 8*ib + k;
                    y[ibl].scales[i%32] |= (sl1 << 4*(i/32));
                    i += 4;
                    y[ibl].scales[i%32] |= (sl2 << 4*(i/32));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
namespace {
inline uint8_t scrambled_sign(uint8_t s) {
    static const uint8_t k_table[128] = {
        0x00, 0x7f, 0x7e, 0x01, 0x7c, 0x03, 0x02, 0x7d, 0x78, 0x07, 0x06, 0x79, 0x04, 0x7b, 0x7a, 0x05,
        0x70, 0x0f, 0x0e, 0x71, 0x0c, 0x73, 0x72, 0x0d, 0x08, 0x77, 0x76, 0x09, 0x74, 0x0b, 0x0a, 0x75,
        0x60, 0x1f, 0x1e, 0x61, 0x1c, 0x63, 0x62, 0x1d, 0x18, 0x67, 0x66, 0x19, 0x64, 0x1b, 0x1a, 0x65,
        0x10, 0x6f, 0x6e, 0x11, 0x6c, 0x13, 0x12, 0x6d, 0x68, 0x17, 0x16, 0x69, 0x14, 0x6b, 0x6a, 0x15,
        0x40, 0x3f, 0x3e, 0x41, 0x3c, 0x43, 0x42, 0x3d, 0x38, 0x47, 0x46, 0x39, 0x44, 0x3b, 0x3a, 0x45,
        0x30, 0x4f, 0x4e, 0x31, 0x4c, 0x33, 0x32, 0x4d, 0x48, 0x37, 0x36, 0x49, 0x34, 0x4b, 0x4a, 0x35,
        0x20, 0x5f, 0x5e, 0x21, 0x5c, 0x23, 0x22, 0x5d, 0x58, 0x27, 0x26, 0x59, 0x24, 0x5b, 0x5a, 0x25,
        0x50, 0x2f, 0x2e, 0x51, 0x2c, 0x53, 0x52, 0x2d, 0x28, 0x57, 0x56, 0x29, 0x54, 0x2b, 0x2a, 0x55,
    };
    return k_table[s];
}
}
static void repack_iq2_xxs(int nrows, int n_per_row, const block_iq2_xxs * x, block_iq2_xxs_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_xxs * x4[4];
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto ysas = (uint32_t *)y[ibl].sas;
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    std::memcpy(aux32, x4[k][ibl].qs + 4*ib, 2*sizeof(uint32_t));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[16*ib+4*k+i] = aux8[i];
                    }
                    uint8_t scale = aux32[1] >> 28;
                    uint8_t s1 = (scrambled_sign((aux32[1] >>  0) & 127) << 1) | ((scale >> 0) & 1);
                    uint8_t s2 = (scrambled_sign((aux32[1] >>  7) & 127) << 1) | ((scale >> 1) & 1);
                    uint8_t s3 = (scrambled_sign((aux32[1] >> 14) & 127) << 1) | ((scale >> 2) & 1);
                    uint8_t s4 = (scrambled_sign((aux32[1] >> 21) & 127) << 1) | ((scale >> 3) & 1);
                    aux32[1] = uint32_t(s1) | (uint32_t(s2) << 8) | (uint32_t(s3) << 16) | (uint32_t(s4) << 24);
                    ysas[4*ib+k] = aux32[1];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
static void repack_iq2_xs(int nrows, int n_per_row, const block_iq2_xs * x, block_iq2_xs_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_xs * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 4; ++i) {
                        uint16_t v = x4[k][ibl].qs[4*ib+i];
                        uint8_t s = v >> 9;
                        y[ibl].qs[16*ib+4*k+i] = (v & 511) | (scrambled_sign(s) << 9);
                    }
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
static void repack_iq2_s(int nrows, int n_per_row, const block_iq2_s * x, block_iq2_s_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_s * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                auto signs = x4[k][ibl].qs + QK_K/8;
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[16*ib+4*k+i] = x4[k][ibl].qs[4*ib+i];
                        y[ibl].signs[16*ib+4*k+i] = signs[4*ib+i];
                    }
                    y[ibl].qh[4*ib+k] = x4[k][ibl].qh[ib];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
static void repack_iq3_xxs(int nrows, int n_per_row, const block_iq3_xxs * x, block_iq3_xxs_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq3_xxs * x4[4];
    uint32_t aux32;
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto ysas = (uint32_t *)y[ibl].sas;
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto xsas = x4[k][ibl].qs + QK_K/4;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 8; ++i) {
                        y[ibl].qs[32*ib+8*k+i] = x4[k][ibl].qs[8*ib+i];
                    }
                    std::memcpy(&aux32, xsas + 4*ib, 4);
                    uint8_t scale = aux32 >> 28;
                    uint8_t s1 = (scrambled_sign((aux32 >>  0) & 127) << 1) | ((scale >> 0) & 1);
                    uint8_t s2 = (scrambled_sign((aux32 >>  7) & 127) << 1) | ((scale >> 1) & 1);
                    uint8_t s3 = (scrambled_sign((aux32 >> 14) & 127) << 1) | ((scale >> 2) & 1);
                    uint8_t s4 = (scrambled_sign((aux32 >> 21) & 127) << 1) | ((scale >> 3) & 1);
                    aux32 = uint32_t(s1) | (uint32_t(s2) << 8) | (uint32_t(s3) << 16) | (uint32_t(s4) << 24);
                    ysas[4*ib+k] = aux32;
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
static void repack_iq3_s(int nrows, int n_per_row, const block_iq3_s * x, block_iq3_s_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq3_s * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales, 0, QK_K/16);
            std::memset(y[ibl].signs,  0, QK_K/2);
            std::memset(y[ibl].qh,     0, QK_K/8);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/64; ++ib) {
                    int j = 8*ib + k;
                    y[ibl].scales[(j+0)%16] |= ((x4[k][ibl].scales[ib] & 0xf) << 4*((j+0)/16));
                    y[ibl].scales[(j+4)%16] |= ((x4[k][ibl].scales[ib] >>  4) << 4*((j+4)/16));
                }
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].qh[4*ib+k] = x4[k][ibl].qh[ib]; // leave ot like this?
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+k+8*i+0] = x4[k][ibl].qs[8*ib+i+0];
                        y[ibl].qs[32*ib+k+8*i+4] = x4[k][ibl].qs[8*ib+i+4];
                    }
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].signs[16*ib+4*k+i] = (((x4[k][ibl].signs[4*ib+0] >> i) & 1) << 0) | (((x4[k][ibl].signs[4*ib+0] >> (4+i)) & 1) << 1) |
                                                    (((x4[k][ibl].signs[4*ib+1] >> i) & 1) << 2) | (((x4[k][ibl].signs[4*ib+1] >> (4+i)) & 1) << 3) |
                                                    (((x4[k][ibl].signs[4*ib+2] >> i) & 1) << 4) | (((x4[k][ibl].signs[4*ib+2] >> (4+i)) & 1) << 5) |
                                                    (((x4[k][ibl].signs[4*ib+3] >> i) & 1) << 6) | (((x4[k][ibl].signs[4*ib+3] >> (4+i)) & 1) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}
namespace {
template <int block_size, int group_size, int num_bits, bool is_abs = false, bool is_int = false>
class QuantizerIQKT {
    static_assert(group_size == 8 || group_size == 4);
    static_assert(block_size >= 8 && block_size%8 == 0);
public:
    constexpr static int kSuperBlockSize = QK_K;
    constexpr static int kBlockSize = block_size;
    constexpr static int kGroupSize = group_size;
    constexpr static int kNg = kBlockSize/kGroupSize;
    constexpr static int kNblock = kSuperBlockSize/kBlockSize;
    constexpr static int kNumVal = 1 << num_bits; // i.e, 16 bits per group of 8
    constexpr static float kScale = is_int ? 1.f : 31.75f;
    constexpr static bool kVerbose = false;

    QuantizerIQKT(int num_clusters, int num_neighbours, int offset = 4096);
    const float * values() const { return m_values.data(); }
    const std::vector<std::vector<int>>& in_cluster() const { return m_in_cluster; }
    const float * mid() const { return m_mid; }

    inline void find_best_match(float d, const float * xb, const float * weight, int * best_idx) const;
    inline std::pair<float, float> find_best_scale(const float * xb, const float * weight, const int * best_idx) const;
    inline float find_best_inverse_scale(const float * xb, const float * weight, const int * best_idx) const;

    static inline void set_values(uint32_t i, float * result, float scale, int offset = 4096) {
        uint32_t x = i + offset;
        if constexpr (is_int) {
            constexpr uint32_t ka = 0xCBAC1FED;
            uint32_t s;
            auto i8 = (const int8_t *)&s;
            for (int k = 0; k < kGroupSize; ++k) {
                x = ka*x;
                s = x & 0x3f3f3f3f;
                if constexpr (is_abs) {
                    result[k] = scale*std::abs(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
                } else {
                    result[k] = scale*(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
                }
            }
        } else {
            constexpr uint32_t ka = 89226354;
            constexpr uint32_t kb = 64248484;
            constexpr uint32_t kmask = 0x8fff8fff;
            constexpr uint32_t km32 = 0x3b603b60;
            for (int k = 0; k < kGroupSize; ++k) {
                x = ka*x + kb;
                uint32_t s = (x & kmask) ^ km32;
                float val = GGML_FP16_TO_FP32(s & 65535) + GGML_FP16_TO_FP32(s >> 16);
                if constexpr (is_abs) result[k] = scale*std::abs(val);
                else result[k] = scale*val;
            }
        }
    }

    static inline int bin4(float x) {
        if constexpr (is_abs) {
            return x < 16.f ? 0 : x < 32.f ? 1 : x < 64.f ? 2 : 3;
        } else {
            return x < -24.f ? 0 : x < 0.0f ? 1 : x < 24.f ? 2 : 3;
        }
    }
    static inline int bin5(float x) {
        if constexpr (is_abs) {
            return x < 11.2f ? 0 : x < 24.f ? 1 : x < 39.f ? 2 : x < 58.f ? 3 : 4;
        } else {
            return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
        }
    }
    inline int bin3(int idim, float x) const { return x < m_mid[2*idim+0] ? 0 : x < m_mid[2*idim+1] ? 1 : 2; }

    // n_per_row < nblock*kSuperBlockSize only for a KT tail row quantised as a zero-padded block: the padding then
    // gets weight 0 and the statistics use the n real values (upstream ik_llama). Default = whole blocks, unchanged.
    static inline void set_weights(float sigma2_scale, int nblock, const float * x, const float * imatrix, float * row_weights,
            int n_per_row = INT_MAX) {
        constexpr float kEps2   = 1e-14f;
        constexpr float kWeight = 1e-4f;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*kSuperBlockSize;
            float * wbl = row_weights + ibl*kSuperBlockSize;
            const int n = std::min(kSuperBlockSize, n_per_row - ibl*kSuperBlockSize);
            for (int j = n; j < kSuperBlockSize; ++j) wbl[j] = 0;

            float sumx2 = 0;
            for (int j = 0; j < n; ++j) sumx2 += xbl[j]*xbl[j];
            if (sumx2 < kEps2*n) {
                // all x in th super block are (almost) zero
                for (int j = 0; j < n; ++j) wbl[j] = kWeight;
                continue;
            }
            const float sigma2 = sigma2_scale*sumx2/n;

            if (imatrix) {
                for (int ib = 0; ib < n/kBlockSize; ++ib) {
                    const float * qw = imatrix + ibl*kSuperBlockSize + ib*kBlockSize;
                    const float * xb = xbl + ib*kBlockSize;
                    float * wb = wbl + ib*kBlockSize;
                    float sumwx = 0, sumw2 = 0, sumx2 = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        wb[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                        sumwx += wb[j]*std::abs(xb[j]);
                        sumw2 += wb[j]*wb[j];
                        sumx2 += xb[j]*xb[j];
                    }
                    if (sumx2 < kEps2 || sumw2 < kEps2 || sumwx < kEps2) {
                        for (int j = 0; j < kBlockSize; ++j) wb[j] = kWeight;
                    }
                }
            } else {
                for (int j = 0; j < n; ++j) wbl[j] = 0.25f*sigma2 + xbl[j]*xbl[j];
            }
        }
    }
private:
    static std::vector<float> cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid);
    static std::vector<std::vector<int>> finalize_clusters(int num_neighbours, const std::vector<float>& points, const std::vector<float>& clusters,
            std::vector<std::vector<float>>& c_values);
    std::vector<float> m_values;
    std::vector<float> m_clusters;
    std::vector<std::vector<int>> m_in_cluster;
    std::vector<std::vector<float>> m_c_values;
    float m_mid[4*kGroupSize];
};
template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::QuantizerIQKT(int num_clusters, int num_neighbours, int offset) {
    m_values.resize(kNumVal*kGroupSize);
    float * data = m_values.data();
    for (int i = 0; i < kNumVal; ++i) {
        set_values(i, data, kScale, offset);
        data += kGroupSize;
    }
    if (num_clusters == 0) return;
    // Make 128 clusters.
    // Note: we get a slightly better result by using 64 clusters
    //       at the expense of almost doubling the quantization time.
    m_clusters = cluster_points(m_values, num_clusters, 200, m_mid);
    GGML_ASSERT(!m_clusters.empty());
    m_in_cluster = finalize_clusters(num_neighbours, m_values, m_clusters, m_c_values);
}
template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::vector<std::vector<int>> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::finalize_clusters(int num_neighbours,
        const std::vector<float>& values, const std::vector<float>& clusters, std::vector<std::vector<float>>& c_values) {
    int ncluster = clusters.size()/kGroupSize;
    std::vector<std::vector<int>> p_in_cluster(ncluster);
    std::vector<int> which_cluster(num_neighbours*kNumVal);
    std::vector<int> ibest(num_neighbours);
    std::vector<float> best(num_neighbours);
    for (int ip = 0; ip < kNumVal; ++ip) {
        auto vp = values.data() + ip*kGroupSize;
        for (int j = 0; j < num_neighbours; ++j) {
            best[j] = INFINITY; ibest[j] = -1;
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            auto vc = clusters.data() + ic*kGroupSize;
            float dist2 = 0;
            for (int k = 0; k < kGroupSize; ++k) {
                float d = vp[k] - vc[k]; dist2 += d*d;
            }
            for (int j = 0; j < num_neighbours; ++j) {
                if (dist2 < best[j]) {
                    for (int k = num_neighbours-1; k > j; --k) {
                        best[k] = best[k-1]; ibest[k] = ibest[k-1];
                    }
                    best[j] = dist2; ibest[j] = ic;
                    break;
                }
            }
        }
        for (int j = 0; j < num_neighbours; ++j) {
            if (ibest[j] < 0) {
                printf("Oops: ibest[%d] = %d\n", j, ibest[j]);
            }
            GGML_ASSERT(ibest[j] >= 0);
            p_in_cluster[ibest[j]].push_back(ip);
        }
        std::memcpy(which_cluster.data() + num_neighbours*ip, ibest.data(), num_neighbours*sizeof(int));
    }
    std::vector<std::pair<float, int>> extra;
    extra.reserve(kNumVal);
    for (int ic = 0; ic < ncluster; ++ic) {
        auto& points = p_in_cluster[ic];
        if (!points.empty() && points.size()%8 == 0) continue;
        extra.clear();
        auto vc = clusters.data() + ic*kGroupSize;
        for (int ip = 0; ip < kNumVal; ++ip) {
            bool can_add = true;
            for (int j = 0; j < num_neighbours; ++j) {
                if (which_cluster[num_neighbours*ip+j] == ic) { can_add = false; break; }
            }
            if (!can_add) continue;
            auto vp = values.data() + ip*kGroupSize;
            float dist2 = 0;
            for (int k = 0; k < kGroupSize; ++k) {
                float d = vp[k] - vc[k]; dist2 += d*d;
            }
            extra.push_back(std::make_pair(dist2, ip));
        }
        std::sort(extra.begin(), extra.end());
        int nadd = 8*((points.size()+7)/8) - points.size();
        for (int i = 0; i < nadd; ++i) points.push_back(extra[i].second);
        GGML_ASSERT(points.size()%8 == 0);
    }
    auto min = p_in_cluster.front().size(), max = p_in_cluster.front().size();
    for (auto& points : p_in_cluster) {
        min = std::min(min, points.size());
        max = std::max(max, points.size());
    }
    c_values.resize(p_in_cluster.size());
    for (int i = 0; i < int(p_in_cluster.size()); ++i) {
        auto& points = p_in_cluster[i];
        c_values[i].resize(points.size()*kGroupSize);
        auto ptr = c_values[i].data();
        for (auto j : points) {
            std::memcpy(ptr, values.data() + j*kGroupSize, kGroupSize*sizeof(float));
            ptr += kGroupSize;
        }
    }

    if (kVerbose) {
        printf("%s: prepared %d clusters\n", __func__, ncluster);
        printf("    min number of points in a cluster: %d\n", int(min));
        printf("    max number of points in a cluster: %d\n", int(max));
    }
    return p_in_cluster;
}
template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::vector<float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid) {
    constexpr int ndim = kGroupSize;
    GGML_ASSERT(points.size() % ndim == 0);
    int npoint = points.size() / ndim;
    GGML_ASSERT(npoint >= 2*ncluster);
    std::vector<std::pair<float, float>> range(ndim, std::make_pair(INFINITY, -INFINITY));
    double Fo = 0;
    for (int i = 0; i < npoint; ++i) {
        auto v = points.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) {
            Fo += v[k]*v[k];
            range[k].first  = std::min(range[k].first, v[k]);
            range[k].second = std::max(range[k].second, v[k]);
        }
    }
    if (kVerbose) printf("%s (ndim = %d, npoint = %d): Fo = %g\n", __func__, ndim, npoint, Fo/points.size());
    if constexpr (is_abs) {
        std::vector<int> P(npoint);
        for (int idim = 0; idim < ndim; ++idim) {
            for (int ip = 0; ip < npoint; ++ip) P[ip] = points[ip*ndim+idim];
            std::sort(P.begin(), P.end());
            if (ndim == 8 && ncluster == 6561) {
                mid[2*idim + 0] = P[npoint/3];
                mid[2*idim + 1] = P[2*npoint/3];
            } else {
                mid[idim] = npoint%2 == 0 ? 0.5f*(P[npoint/2] + P[npoint/2-1]) : P[npoint/2];
                if (kVerbose) printf("%s: mid[%d] = %g\n", __func__, idim, mid[idim]);
            }
        }
    } else {
        for (int k = 0; k < ndim; ++k) mid[k] = 0.5f*(range[k].first + range[k].second);
    }
    std::vector<float> sump(ncluster*ndim);
    std::vector<int> counts(ncluster);
    std::vector<float> result(ncluster*ndim);
    if (ndim == 8 && (ncluster == 256 || ncluster == 6561)) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            uint16_t u = 0;
            if (ncluster == 256) {
                for (int k = 0; k < ndim; ++k) if (vp[k] > mid[k]) u |= (1 << k);
            } else {
                int s = 1;
                for (int k = 0; k < ndim; ++k) {
                    int bin = vp[k] < mid[2*k+0] ? 0 : vp[k] < mid[2*k+1] ? 1 : 2;
                    u += s*bin; s *= 3;
                }
            }
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[ndim*u + k] += vp[k];
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) {
                printf("%s: Oops. Cluster %d has no points\n", __func__, ic);
                GGML_ABORT("fatal error");
            }
            for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
        }
        return result;
    }
    else if (ndim == 4 && (ncluster == 256 || ncluster == 625)) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            uint16_t u = 0;
            if (ncluster == 256) {
                for (int k = 0; k < ndim; ++k) u |= (bin4(vp[k]) << 2*k);
            } else {
                int s = 1;
                for (int k = 0; k < ndim; ++k) { u += s*bin5(vp[k]); s *= 5; }
            }
            if (u >= int(counts.size())) {
                printf("Oops: u = %u, vp = %g, %g, %g, %g\n", u, vp[0], vp[1], vp[2], vp[3]);
                u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < ndim; ++k) {
                        auto bin = bin4(vp[k]); u |= (bin << 2*k);
                        printf(" bin[%d] = %d, u = %u", k, bin, u);
                    }
                } else {
                    for (int k = 0; k < ndim; ++k) printf(" bin[%d] = %d", k, bin5(vp[k]));
                }
                printf("\n");
                GGML_ABORT("fatal error");
            }
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[ndim*u + k] += vp[k];
        }
        int nzero = 0;
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) {
                ++nzero;
                printf("%s: Oops. Cluster %d has no points: ", __func__, ic);
                for (int k = 0; k < ndim; ++k) {
                    int l = (ic >> 2*k) & 3;
                    printf(" %d", l);
                }
                printf("\n");
            } else {
                for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
            }
        }
        if (nzero > 0) printf("%s: %d out of %d clusters dir not have any points\n", __func__, nzero, ncluster);
        return result;
    }
    std::mt19937 rndm(1234);
    float scale = 1.f/4294967296.f;
    for (int i = 0; i < ncluster; ++i) {
        auto v = result.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) v[k] = range[k].first + (range[k].second - range[k].first)*scale*rndm();
    }
    std::vector<int> which_cluster(npoint, -1);
    double Flast = Fo;
    for (int iter = 0; iter < niter; ++iter) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        int nchanged = 0;
        double F = 0;
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            float best = INFINITY; int ibest = -1;
            for (int ic = 0; ic < ncluster; ++ic) {
                auto vc = result.data() + ndim*ic;
                float dist2 = 0;
                for (int k = 0; k < ndim; ++k) {
                    float d = vp[k] - vc[k]; dist2 += d*d;
                }
                if (dist2 < best) {
                    best = dist2; ibest = ic;
                }
            }
            if (ibest < 0) {
                printf("Oops(iteration %d) - failed to find cluster for point", iter);
                for (int k = 0; k < ndim; ++k) printf(" %g", vp[k]);
                printf("\nHave %d clusters\n", ncluster);
            }
            GGML_ASSERT(ibest >= 0);
            F += best;
            if (which_cluster[ip] != ibest) ++nchanged;
            which_cluster[ip] = ibest;
            ++counts[ibest];
            auto vc = sump.data() + ndim*ibest;
            for (int k = 0; k < ndim; ++k) vc[k] += vp[k];
        }
        if (nchanged == 0) break;
        for (int ic = 0; ic < ncluster; ++ic) {
            float norm = counts[ic] > 0 ? 1.f/counts[ic] : 0.f;
            auto vc = sump.data() + ndim*ic;
            auto r  = result.data() + ndim*ic;
            for (int k = 0; k < ndim; ++k) r[k] = vc[k]*norm;
        }
        if (kVerbose) printf("%s(iteration %d): F = %g, nchanged = %d\n", __func__, iter+1, F/points.size(), nchanged);
        if (iter > 1 && Flast/F - 1 < 1e-6) break;
        Flast = F;
    }
    int nzero = 0;
    for (int ic = 0; ic < ncluster; ++ic) {
        if (!counts[ic]) ++nzero;
    }
    if (nzero > 0) printf("%s: there are %d empty clusters\n", __func__, nzero);
    return result;
}
using QuantizerIQ1KT = QuantizerIQKT<32, 8, 13, false, true>;
const QuantizerIQ1KT& iq1kt_quantizer() {
    static std::mutex mutex;
    static std::unique_ptr<QuantizerIQ1KT> quantizer;
    std::lock_guard<std::mutex> lock(mutex);
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ1KT>(256, 32);
    return *quantizer;
}
using QuantizerIQ2KT = QuantizerIQKT<32, 8, 16, false, true>;
const QuantizerIQ2KT& iq2kt_quantizer() {
    static std::mutex mutex;
    static std::unique_ptr<QuantizerIQ2KT> quantizer;
    std::lock_guard<std::mutex> lock(mutex);
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ2KT>(256, 8);
    return *quantizer;
}
using QuantizerIQ3KT = QuantizerIQKT<32, 8, 16, true, true>;
const QuantizerIQ3KT& iq3kt_quantizer() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ3KT> quantizer;
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ3KT>(256, 8);
    return *quantizer;
}
using QuantizerIQ4KT = QuantizerIQKT<32, 4, 15, false, true>;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
