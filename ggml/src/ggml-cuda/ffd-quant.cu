// GGML_OP_FFD_QUANT_K, CUDA. Contract: ggml.h (ggml_ffd_quant_k); CPU reference: ggml-cpu/ops.cpp.
// Quantiser from qluoluo/faster-flash-decoding @ ca09458 (Apache-2.0) quantized_cache.py:111-192, reimplemented.
// BENCH-PENDING: no speed is claimed; "faster" is only the upstream repo name.
//
// One launch, grid (n_idx, n_head_kv): the CUDA block for write index i re-quantises the whole `block`-cell block
// holding that cell, for one KV head. An index whose block equals the previous index's block exits at once (a
// prompt ubatch writes consecutive cells, so this removes nearly all repeats); any repeat that survives recomputes
// the SAME values from the same K, so concurrent identical stores are benign. Each thread owns one output byte =
// 4 channels, so no intra-block synchronisation is needed. Arithmetic is pinned to IEEE round-to-nearest
// (__fdiv_rn / __fadd_rn) and rintf (round-half-even) so the result matches the CPU reference bit for bit even
// when the file is compiled with --use_fast_math.

#include "ffd-quant.cuh"

static __device__ __forceinline__ float ffd_to_f32(float x)       { return x; }
static __device__ __forceinline__ float ffd_to_f32(half x)        { return __half2float(x); }
static __device__ __forceinline__ float ffd_to_f32(nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
static __global__ void k_ffd_quant_k(
        const T * __restrict__ k, const int64_t * __restrict__ idx, uint8_t * __restrict__ k2, float * __restrict__ ks,
        const int n_idx, const int D, const int block, const int64_t kv_size, const int64_t ns, const int64_t n_blk,
        const int64_t k_nb1, const int64_t k2_nb1, const int64_t k2_nb2, const int64_t k2_nb3,
        const int64_t ks_nb1, const int64_t ks_nb2, const int64_t ks_nb3) {
    const int     i = blockIdx.x;
    const int64_t h = blockIdx.y;
    const int64_t r = idx[i];
    if (r < 0 || r >= kv_size*ns) {
        return;
    }
    const int64_t s = r/kv_size;
    const int64_t b = (r % kv_size)/block;
    if (i > 0) {
        const int64_t rp = idx[i - 1];
        if (rp >= 0 && rp < kv_size*ns && rp/kv_size == s && (rp % kv_size)/block == b) {
            return;                                   // same block as the previous index: already covered
        }
    }
    const int64_t c0 = b*block;
    const int64_t c1 = min(c0 + (int64_t) block, kv_size);

    for (int j = threadIdx.x; j < D/4; j += blockDim.x) {
        float am[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int64_t c = c0; c < c1; ++c) {
            const T * kr = (const T *) ((const char *) k + (s*kv_size + c)*k_nb1) + h*D + 4*j;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                am[e] = fmaxf(am[e], fabsf(ffd_to_f32(kr[e])));
            }
        }
        float sc[4];
        float * so = (float *) ((char *) ks + b*ks_nb1 + h*ks_nb2 + s*ks_nb3) + 4*j;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            sc[e] = fmaxf(__fdiv_rn(am[e], 1.5f), 1e-8f);
            so[e] = sc[e];
        }
        for (int64_t c = c0; c < c1; ++c) {
            const T * kr = (const T *) ((const char *) k + (s*kv_size + c)*k_nb1) + h*D + 4*j;
            uint8_t byte = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float v = rintf(__fadd_rn(__fdiv_rn(ffd_to_f32(kr[e]), sc[e]), 1.5f));
                const int   q = (int) fminf(fmaxf(v, 0.0f), 3.0f);
                byte |= (uint8_t) (q << (2*e));
            }
            k2[c*k2_nb1 + h*k2_nb2 + s*k2_nb3 + j] = byte;
        }
    }
}

bool ggml_cuda_ffd_quant_k_supported(const ggml_tensor * op) {
    const ggml_tensor * k2  = op->src[0];
    const ggml_tensor * k   = op->src[1];
    const ggml_tensor * idx = op->src[2];
    const ggml_tensor * ks  = op->src[3];
    const int D = ggml_get_op_params_i32(op, 0);
    return (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) &&
           idx->type == GGML_TYPE_I64 && ggml_is_contiguous(idx) &&
           k2->type == GGML_TYPE_I8 && k2->nb[0] == 1 && ks->type == GGML_TYPE_F32 && ks->nb[0] == sizeof(float) &&
           k->nb[0] == ggml_type_size(k->type) && D % 4 == 0 && D/4 <= 1024;
}

void ggml_cuda_op_ffd_quant_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k2  = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * idx = dst->src[2];
    const ggml_tensor * ks  = dst->src[3];

    const int D     = ggml_get_op_params_i32(dst, 0);
    const int block = ggml_get_op_params_i32(dst, 1);
    const int n_idx = (int) idx->ne[0];
    const int64_t H = k2->ne[2];
    if (n_idx == 0 || H == 0) {
        return;
    }

    const dim3 grid(n_idx, (unsigned) H);
    const int  nthreads = min(D/4, 256);
    cudaStream_t stream = ctx.stream();

#define FFD_QUANT_LAUNCH(T)                                                                                     \
    k_ffd_quant_k<T><<<grid, nthreads, 0, stream>>>(                                                            \
        (const T *) k->data, (const int64_t *) idx->data, (uint8_t *) k2->data, (float *) ks->data,            \
        n_idx, D, block, k2->ne[1], k2->ne[3], ks->ne[1],                                                       \
        k->nb[1], k2->nb[1], k2->nb[2], k2->nb[3], ks->nb[1], ks->nb[2], ks->nb[3])

    switch (k->type) {
        case GGML_TYPE_F32:  FFD_QUANT_LAUNCH(float);       break;
        case GGML_TYPE_F16:  FFD_QUANT_LAUNCH(half);        break;
        case GGML_TYPE_BF16: FFD_QUANT_LAUNCH(nv_bfloat16); break;
        default: GGML_ABORT("ffd_quant_k: unsupported K type");
    }
#undef FFD_QUANT_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
