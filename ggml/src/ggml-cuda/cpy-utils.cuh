#pragma once

#include "ggml-common.h"

template<typename src_t, typename dst_t>
static __device__ __forceinline__ void convert_flt(const src_t * src, dst_t * dst) {
    if constexpr (std::is_same_v<src_t, dst_t>) {
        *dst = *src;
    } else {
        *dst = float(*src);
    }
}

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_0/2; ++j) {
        const float v0 = x[0       + j];
        const float v1 = x[QK4_0/2 + j];
        const float x0 = v0*id;
        const float x1 = v1*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));
        float q0 = xi0 - 8;
        float q1 = xi1 - 8;
        float w0 = v0*v0;
        float w1 = v1*v1;
        sumqx += w0*q0*v0 + w1*q1*v1;
        sumq2 += w0*q0*q0 + w1*q1*q1;

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
    if (sumq2 > 0) {
        y->d = sumqx/sumq2;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    // First pass, as before: nearest grid index at the max-derived scale, then the least-squares
    // scale for those indices (weights x^2).
    uint8_t L[QK4_NL];
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL; ++j) {
        L[j] = best_index_int8(16, kvalues_iq4nl, x[j]*id);
        const float q = kvalues_iq4nl[L[j]];
        const float w = x[j]*x[j];
        sumqx += w*q*x[j];
        sumq2 += w*q*q;
    }
    if (sumq2 > 0) {
        d = sumqx/sumq2;
        // Then the same refinement this fork's CPU row quantiser runs (ggml-quants.c,
        // quantize_row_iq4_nl_impl with ntry = -1, i.e. quantize_row_iq4_nl = from_float): repeatedly
        // move the one index whose step toward its residual is cheapest, while that improves the
        // weighted fit. Upstream's CPU quantiser has no such step, which is why upstream's CUDA copy
        // did not either; this fork's does, so CUDA quantised the same data differently --
        // test-backend-ops CPY f32->iq4_nl ERR 1.4e-3 and SET_ROWS ->iq4_nl ERR 1.2e-4, both over
        // their tolerances, on every other type clean.
        float best = d*sumqx;
        for (int iter = 0; iter < 32*QK4_NL; ++iter) {
            float min_step = INFINITY;
            int best_j = -1, dir = 0;
            for (int j = 0; j < QK4_NL; ++j) {
                const float g = d * x[j]*x[j] * (x[j] - d*kvalues_iq4nl[L[j]]);
                if (g > 0 && L[j] < 15) {
                    const float step = (kvalues_iq4nl[L[j]+1] - kvalues_iq4nl[L[j]])/g;
                    if (step < min_step) { min_step = step; best_j = j; dir = 1; }
                } else if (g < 0 && L[j] > 0) {
                    const float step = (kvalues_iq4nl[L[j]-1] - kvalues_iq4nl[L[j]])/g;
                    if (step < min_step) { min_step = step; best_j = j; dir = -1; }
                }
            }
            if (best_j < 0) break;
            const float w  = x[best_j]*x[best_j];
            const float q0 = kvalues_iq4nl[L[best_j]];
            const float q1 = kvalues_iq4nl[L[best_j] + dir];
            const float new_sumqx = sumqx + w*x[best_j]*(q1 - q0);
            const float new_sumq2 = sumq2 + w*(q1*q1 - q0*q0);
            if (new_sumq2 > 0 && new_sumqx*new_sumqx > best*new_sumq2) {
                sumqx = new_sumqx; sumq2 = new_sumq2;
                d = sumqx/sumq2; best = d*sumqx;
                L[best_j] += dir;
            } else {
                break;
            }
        }
    }
    for (int j = 0; j < QK4_NL/2; ++j) {
        y->qs[j] = L[j] | (L[QK4_NL/2 + j] << 4);
    }
    y->d = d;
}

static __device__ void quantize_f32_q6_0_block(const float * __restrict__ xi, block_q6_0 * __restrict__ y) {

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK6_0; ++j) {
        const float v  = xi[j];
        const float av = fabsf(xi[j]);
        if (amax < av) {
            amax = av;
            vmax = v;
        }
    }

    const float d  = vmax / -32;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;
    memset(y->qh, 0, QK6_0/4);

    for (int j = 0; j < QK6_0/2; ++j) {
        const float x0 = xi[0       + j]*id;
        const float x1 = xi[QK6_0/2 + j]*id;

        const uint8_t xi0 = min(63, (int8_t)(x0 + 32.5f));
        const uint8_t xi1 = min(63, (int8_t)(x1 + 32.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        const uint8_t h = (xi0 >> 4) | ((xi1 >> 4) << 2);
        y->qh[j%(QK6_0/4)] |= (h << 4*(j/(QK6_0/4)));
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q6_0(const char * cxi, char * cdsti) {
    quantize_f32_q6_0_block((const float *)cxi, (block_q6_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_flt(const char * cxi, char * cdsti) {
    convert_flt((const src_t *)cxi, (dst_t *)cdsti);
}
