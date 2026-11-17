// attn-bf16.h -- AMX-BF16 attention for up to 16 query rows that share one K/V head (decode with GQA, or a small
// prefill block). bf16 K/V rows, f32 queries, f16 mask, online softmax in f32, both products on the AMX tiles.
//
// Row S69-OV-PICK-AMX-BF16-ATTENTION-DECODE-GEMV-AND-PREFILL-2026-10-06-01. Spec read in full:
// openvinotoolkit/openvino src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/executor_pa.cpp (Apache-2.0,
// b0086aa9c6c5) :139-304 (VNNI pair-interleave packing of K/V blocks, zero padding of a partial block BEFORE packing,
// :223-225), :709-796 (brgemm per query-row count, AMX only when S % 32 == 0 and Q/K/V bf16). Own code, no code copied.
// HARNESS-CONSIDERED: a CPU attention kernel inside the engine; no harness capability is involved.
//
// Layout facts the kernel relies on (Intel AMX: a B tile row holds K-dimension PAIRS, 16 columns x 2 bf16 = 64 bytes):
//   scores  S[m][n] = sum_d Q[m][d] K[n][d]: A = Q (16 rows x 32 d), B row r = { K[n0+j][d0+2r], K[n0+j][d0+2r+1] : j<16 }
//   output  O[m][c] = sum_n P[m][n] V[n][c]: A = P (16 rows x 32 keys), B row r = { V[n0+2r][c0+j], V[n0+2r+1][c0+j] : j<16 }
// K therefore needs a transpose at 32-bit (pair) granularity, V only a pair interleave of adjacent tokens.
//
// Tile configuration: every tile is 16 rows x 64 bytes, which is EXACTLY the shape amx/mmq.cpp configures for tiles
// 4..7, and only tiles 4..7 are used here. mmq.cpp loads its configuration once per thread (thread_local flag), so a
// kernel that loaded a different palette in the same thread would silently corrupt the next quantized matmul; this one
// loads mmq's full table, so whichever of the two loads first, both see the shapes they expect.
//
// Preconditions (checked by can_run): D % 32 == 0, Dv % 16 == 0, n_kv % 32 == 0, 1 <= M <= 16.
#pragma once

#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#include <immintrin.h>
#define GGML_AMX_ATTN_BF16 1
#endif

namespace ggml_amx_attn {

inline bool can_run(int M, int D, int Dv, int n_kv) {
    return M >= 1 && M <= 16 && D > 0 && D % 32 == 0 && Dv > 0 && Dv % 16 == 0 && n_kv > 0 && n_kv % 32 == 0;
}

// Scratch bytes the kernel needs: Q bf16 (16 x D), P bf16 (16 x 32), B tile (16 x 32 bf16), scores f32 (2 x 16 x 16),
// out tile f32 (16 x 16), accumulators f32 (16 x Dv), plus alignment slack.
inline size_t scratch_size(int D, int Dv) {
    return (size_t) 16 * D * 2 + 16 * 32 * 2 + 16 * 32 * 2 + 2 * 16 * 16 * 4 + 16 * 16 * 4 + (size_t) 16 * Dv * 4 + 4 * 64;
}

// IEEE half -> float, exact (self-contained so the header needs nothing from ggml-impl.h).
inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t u;
    if (exp == 0x1f) {
        u = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        u = sign | ((exp + 112) << 23) | (man << 13);
    } else if (man == 0) {
        u = sign;
    } else {
        exp = 113;
        while ((man & 0x400u) == 0) {
            man <<= 1;
            --exp;
        }
        u = sign | (exp << 23) | ((man & 0x3ffu) << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t) ((u >> 16) | 0x40);   // quiet NaN
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

#if defined(GGML_AMX_ATTN_BF16)

struct tile_cfg {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

// Same table as amx/mmq.cpp ggml_tile_config_init (see the header comment for why it must be identical).
inline void load_tile_config() {
    alignas(64) tile_cfg tc = {};
    tc.palette_id = 1;
    tc.rows[0] = 8;   tc.colsb[0] = 64;
    tc.rows[1] = 8;   tc.colsb[1] = 64;
    tc.rows[2] = 16;  tc.colsb[2] = 32;
    tc.rows[3] = 16;  tc.colsb[3] = 32;
    tc.rows[4] = 16;  tc.colsb[4] = 64;
    tc.rows[5] = 16;  tc.colsb[5] = 64;
    tc.rows[6] = 16;  tc.colsb[6] = 64;
    tc.rows[7] = 16;  tc.colsb[7] = 64;
    _tile_loadconfig(&tc);
}

static inline char * align64(char * p) {
    return (char *) (((uintptr_t) p + 63) & ~(uintptr_t) 63);
}

// out[m][0..Dv) = softmax(scale * softcap(Q[m] . K^T) + mask[m]) . V, sinks[m] (if given) joining the softmax as one
// extra logit with no value. q rows: f32, stride_q floats. K/V rows: bf16, strides in bytes. mask rows: f16, stride_m
// bytes (0 = one row shared by all M queries), may be null. out rows: f32, stride_out floats.
inline void attn_rows(int M, int D, int Dv, int n_kv,
                      const float * q, size_t stride_q,
                      const ggml_bf16_t * K, size_t stride_k,
                      const ggml_bf16_t * V, size_t stride_v,
                      const ggml_fp16_t * mask, size_t stride_m,
                      float scale, float softcap, const float * sinks,
                      float * out, size_t stride_out, void * scratch) {
    char * p = align64((char *) scratch);
    uint16_t * qb   = (uint16_t *) p;  p = align64(p + (size_t) 16 * D * 2);
    uint16_t * pb   = (uint16_t *) p;  p = align64(p + 16 * 32 * 2);
    uint16_t * bt   = (uint16_t *) p;  p = align64(p + 16 * 32 * 2);
    float    * sc   = (float *) p;     p = align64(p + 2 * 16 * 16 * 4);
    float    * ot   = (float *) p;     p = align64(p + 16 * 16 * 4);
    float    * acc  = (float *) p;

    // Q -> bf16, rows M..15 zero (a tile always has 16 rows; zero rows produce zero scores that are never read)
    std::memset(qb, 0, (size_t) 16 * D * 2);
    for (int m = 0; m < M; ++m) {
        const float * qm = q + (size_t) m * stride_q;
        for (int d = 0; d < D; ++d) {
            qb[(size_t) m * D + d] = f32_to_bf16_rne(qm[d]);
        }
    }
    std::memset(pb, 0, 16 * 32 * 2);
    std::memset(acc, 0, (size_t) 16 * Dv * 4);
    float row_max[16], row_sum[16];
    for (int m = 0; m < 16; ++m) {
        row_max[m] = -INFINITY;
        row_sum[m] = 0.0f;
    }

    load_tile_config();
    const uint16_t * Kb = (const uint16_t *) K;
    const uint16_t * Vb = (const uint16_t *) V;
    for (int n0 = 0; n0 < n_kv; n0 += 32) {
        // ---- scores for keys n0 .. n0+31, two 16-key C tiles
        for (int g = 0; g < 2; ++g) {
            _tile_zero(6);
            for (int d0 = 0; d0 < D; d0 += 32) {
                for (int r = 0; r < 16; ++r) {
                    for (int j = 0; j < 16; ++j) {
                        const uint16_t * kr = (const uint16_t *) ((const char *) Kb + (size_t) (n0 + 16 * g + j) * stride_k);
                        bt[r * 32 + 2 * j + 0] = kr[d0 + 2 * r + 0];
                        bt[r * 32 + 2 * j + 1] = kr[d0 + 2 * r + 1];
                    }
                }
                _tile_loadd(4, qb + d0, D * 2);
                _tile_loadd(5, bt, 64);
                _tile_dpbf16ps(6, 4, 5);
            }
            _tile_stored(6, sc + g * 256, 64);
        }
        // ---- scale, softcap, mask, online softmax per row; P -> bf16
        for (int m = 0; m < M; ++m) {
            const ggml_fp16_t * mr = mask ? (const ggml_fp16_t *) ((const char *) mask + (size_t) m * stride_m) : nullptr;
            float s[32];
            float bmax = -INFINITY;
            for (int j = 0; j < 32; ++j) {
                float x = sc[(j / 16) * 256 + m * 16 + (j % 16)] * scale;
                if (softcap > 0.0f) {
                    x = softcap * tanhf(x);
                }
                if (mr) {
                    x += fp16_to_f32(mr[n0 + j]);
                }
                s[j] = x;
                bmax = x > bmax ? x : bmax;
            }
            const float new_max = bmax > row_max[m] ? bmax : row_max[m];
            const float alpha   = row_max[m] == -INFINITY ? 0.0f : expf(row_max[m] - new_max);
            float bsum = 0.0f;
            for (int j = 0; j < 32; ++j) {
                const float e = new_max == -INFINITY ? 0.0f : expf(s[j] - new_max);
                bsum += e;
                pb[m * 32 + j] = f32_to_bf16_rne(e);
            }
            row_sum[m] = row_sum[m] * alpha + bsum;
            row_max[m] = new_max;
            float * am = acc + (size_t) m * Dv;
            if (alpha != 1.0f) {
                for (int c = 0; c < Dv; ++c) {
                    am[c] *= alpha;
                }
            }
        }
        // ---- O += P . V for this block, 16 output columns per tile
        _tile_loadd(4, pb, 64);
        for (int c0 = 0; c0 < Dv; c0 += 16) {
            for (int r = 0; r < 16; ++r) {
                const uint16_t * v0 = (const uint16_t *) ((const char *) Vb + (size_t) (n0 + 2 * r + 0) * stride_v);
                const uint16_t * v1 = (const uint16_t *) ((const char *) Vb + (size_t) (n0 + 2 * r + 1) * stride_v);
                for (int j = 0; j < 16; ++j) {
                    bt[r * 32 + 2 * j + 0] = v0[c0 + j];
                    bt[r * 32 + 2 * j + 1] = v1[c0 + j];
                }
            }
            _tile_zero(6);
            _tile_loadd(5, bt, 64);
            _tile_dpbf16ps(6, 4, 5);
            _tile_stored(6, ot, 64);
            for (int m = 0; m < M; ++m) {
                float * am = acc + (size_t) m * Dv + c0;
                for (int j = 0; j < 16; ++j) {
                    am[j] += ot[m * 16 + j];
                }
            }
        }
    }
    // tiles are left configured: mmq.cpp keeps its own once-per-thread flag and expects this exact palette

    for (int m = 0; m < M; ++m) {
        float l = row_sum[m];
        if (sinks) {
            const float sk = sinks[m];
            if (row_max[m] == -INFINITY) {
                l = 1.0f;   // only the sink: every value weight is 0
            } else if (sk > row_max[m]) {
                const float a = expf(row_max[m] - sk);
                l = l * a + 1.0f;
                for (int c = 0; c < Dv; ++c) {
                    acc[(size_t) m * Dv + c] *= a;
                }
            } else {
                l += expf(sk - row_max[m]);
            }
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float * om = out + (size_t) m * stride_out;
        for (int c = 0; c < Dv; ++c) {
            om[c] = acc[(size_t) m * Dv + c] * inv;
        }
    }
}

#endif  // GGML_AMX_ATTN_BF16

} // namespace ggml_amx_attn
