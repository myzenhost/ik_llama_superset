//
// TurboQuant KV constants and per-element decode for the SYCL backend.
//
// The tables below are the ONE SYCL copy. They must stay byte-identical to the CPU copy in ggml-turbo-quant.c, the CUDA copy in
// ggml-cuda/turbo-quant.cuh and the Vulkan copy in ggml-vulkan/vulkan-shaders/turbo_tables.glsl: a block written by one backend
// is read by another (the KV cache is written on whichever device the scheduler picks), and a diverging table decodes silently
// to the wrong vector. scripts/check-turbo-tables-consistent.py parses this file too (ctest test-turbo-tables-consistent) and its
// --selftest nudges each table to prove the checker fails.
//
// Block layouts (ggml-common.h): turbo2 34 B, turbo3 50 B, turbo4 66 B (the 4-bit PolarQuant layout, TURBO4_USE_4BIT=1), all 128
// elements, stored in the ROTATED basis: value = centroid[idx] * norm. Decoding here does NOT undo the rotation; the attention
// wrapper in fattn.cpp rotates Q once before the kernel and the finished output once after it (see ggml_sycl_turbo_wht).
//
// DESIGN SOURCE: TheTom/llama-cpp-turboquant bcb85fc3ae85 ggml/src/ggml-sycl/turbo-quant.hpp (MIT, read in full 2026-10-05). The decode
// helpers are the same few lines as the CUDA ones in this tree (turbo-quant.cuh turbo{2,3,4}_dequant_element); the tip's
// duplicated 64-wide sign tables are left out because this fork rotates in 128-element groups only.
//
// GATE-CLAIM-OK: a header; it declares no resource unavailable.

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

#include "common.hpp"

#if !TURBO4_USE_4BIT
#error "SYCL TurboQuant assumes the 4-bit turbo4_0 layout (TURBO4_USE_4BIT=1, 66-byte block)"
#endif

static constexpr float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static constexpr float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

static constexpr float TURBO_CENTROIDS_3BIT[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};

static constexpr float TURBO_MID_3BIT[7] = {
    -0.154496f, -0.092804f, -0.044243f, 0.0f,
     0.044243f,  0.092804f,  0.154496f
};

static constexpr float TURBO_CENTROIDS_4BIT[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f
};

static constexpr float TURBO_MID_4BIT[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f,
    -0.070671f, -0.046174f, -0.022824f,  0.000000f,
     0.022824f,  0.046174f,  0.070671f,  0.097164f,
     0.127026f,  0.162947f,  0.212203f
};

// WHT sign arrays (seed 42, 128 elements): forward rotation = signs1 -> butterflies -> (1/sqrt(128)) * signs2.
static constexpr float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static constexpr float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// Per-element decode in the ROTATED basis (centroid * block norm).
inline float turbo2_dequant_element(const block_turbo2_0 * __restrict__ x, int j, float norm) {
    const uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}

inline float turbo3_dequant_element(const block_turbo3_0 * __restrict__ x, int j, float norm) {
    const uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    const uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    const uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT[idx] * norm;
}

inline float turbo4_dequant_element(const block_turbo4_0 * __restrict__ x, int j, float norm) {
    const uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}
