//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

// Include guard: fattn-common.cuh includes this header, so any TU that includes both (lightning-indexer.cu does)
// saw ggml_cuda_cast and the to_t_cuda_t typedefs defined twice. Every other header here already has one.
#pragma once

#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

// TurboQuant KV types (turbo2/3/4): block dequant to the ROTATED basis (centroid * norm), defined in turbo-convert.cu.
// nullptr for any other type. ggml_get_to_fp16_cuda / ggml_get_to_fp32_cuda forward to these.
to_fp16_cuda_t ggml_cuda_get_turbo_to_fp16(ggml_type type);
to_t_cuda_t<float> ggml_cuda_get_turbo_to_fp32(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

// Number of values a K/V view spans, from its first byte to its last. The FA launchers convert a
// quantised K/V to f16 with ONE contiguous to_fp16 call and then scale the view's own strides by
// bs*sizeof(half)/ts, so the copy must cover every byte those strides can reach. That is
// ggml_nelements() only for a contiguously allocated view; a KV-cache view that steps over more rows
// than it exposes (test-backend-ops kv_view, a cache with more slots than n_kv per head/sequence)
// spans more, and converting ggml_nelements() values converted the wrong rows and left the tail of
// the scaled index range uninitialised -- NaN for every non-f16 K/V type in test-backend-ops.
static inline int64_t ggml_cuda_fattn_f16_span(const ggml_tensor * t) {
    const size_t  ts = ggml_type_size(t->type);
    const int64_t bs = ggml_blck_size(t->type);
    GGML_ASSERT(t->nb[0] == ts && "FA f16 conversion needs K/V rows to be contiguous");
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        GGML_ASSERT((t->ne[i] == 1 || t->nb[i] % ts == 0) && "FA f16 conversion needs block-aligned K/V strides");
    }
    const size_t nbytes = ggml_nbytes(t);
    GGML_ASSERT(nbytes % ts == 0);
    return (int64_t) (nbytes/ts)*bs;
}

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    } else if constexpr(std::is_same_v<dst_t, nv_bfloat16>) {
        return __float2bfloat16(float(x));
    } else if constexpr(std::is_same_v<src_t, nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, half2>) {
        return __float22half2_rn(x);
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, nv_bfloat162>) {
        // bypass compile error on cuda 12.0.1
#ifdef GGML_USE_HIP
        return __float22bfloat162_rn(x);
#else
        return {x.x, x.y};
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}

