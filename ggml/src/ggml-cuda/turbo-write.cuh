// Device entry points for the TurboQuant KV writer (see turbo-write.h for the algorithm and its provenance).
// Signature matches the cpy_f32_q / k_set_rows_quant quantize_func contract: (const float * x, block_type * y),
// x pointing at the 128 consecutive floats of one group, y at the destination block.
//
// The tables are the per-TU __constant__ arrays from turbo-quant.cuh (byte-identical to the host copies in
// ggml-turbo-quant.c; enforced by temp/research/turbo/verify_rotation_algebra.py). Including this header
// therefore pulls those tables into the including TU -- cpy.cu and set-rows.cu only.

#pragma once

#include "turbo-quant.cuh"
#include "turbo-write.h"

static __device__ void quantize_f32_turbo3_0_group(const float * __restrict__ x, block_turbo3_0 * __restrict__ y) {
    static_assert(QK_TURBO3 == TURBO_WRITE_GROUP, "turbo3 writer assumes one 128-element block per rotation group");
    const turbo_write_tables t = { TURBO_WHT_SIGNS1, TURBO_WHT_SIGNS2, TURBO_MID_3BIT, TURBO_CENTROIDS_3BIT };
    const float norm = turbo_write_group<3>(x, y->qs, y->signs, t);
    y->norm = __float2half(norm);
}

static __device__ void quantize_f32_turbo4_0_group(const float * __restrict__ x, block_turbo4_0 * __restrict__ y) {
    static_assert(QK_TURBO4 == TURBO_WRITE_GROUP, "turbo4 writer assumes one 128-element block per rotation group");
    const turbo_write_tables t = { TURBO_WHT_SIGNS1, TURBO_WHT_SIGNS2, TURBO_MID_4BIT, TURBO_CENTROIDS_4BIT };
    const float norm = turbo_write_group<4>(x, y->qs, nullptr, t);
    y->norm = __float2half(norm);
}

static __device__ void quantize_f32_turbo2_0_group(const float * __restrict__ x, block_turbo2_0 * __restrict__ y) {
    static_assert(QK_TURBO2 == TURBO_WRITE_GROUP, "turbo2 writer assumes one 128-element block per rotation group");
    const turbo_write_tables t = { TURBO_WHT_SIGNS1, TURBO_WHT_SIGNS2, TURBO_MID_2BIT, TURBO_CENTROIDS_2BIT };
    const float norm = turbo_write_group<2>(x, y->qs, nullptr, t);
    y->norm = __float2half(norm);
}
