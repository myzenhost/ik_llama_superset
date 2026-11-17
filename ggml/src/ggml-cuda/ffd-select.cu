// GGML_OP_FFD_SELECT, CUDA. Contract: ggml.h (ggml_ffd_select); CPU reference: ggml-cpu/ops.cpp.
// Design source: qluoluo/FFD @ ca09458 (Apache-2.0), reimplemented, not copied.
//
// Two launches:
//   k_ffd_block_max  grid (n_blocks, n_tokens, n_head_kv*ns). The threads of one CUDA block walk the KV
//                    positions of one selector block, score each against the 2-bit K for every query head
//                    of the GQA group (q and the block's per-channel scale staged in shared memory), and
//                    reduce to one max per group head -> scratch [ns][hkv][t][g][b].
//   k_ffd_decide     grid (n_tokens, n_head_kv, ns); threads over blocks. th_g = max(first, last) - delta,
//                    keep if any g reaches th_g; fully masked -> prune; any other last block -> keep.

#include "ffd-select.cuh"

#include <cfloat>

#define FFD_THREADS 128

static __device__ __forceinline__ float ffd_warp_max(float v) {
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off, WARP_SIZE));
    }
    return v;
}

static __global__ void k_ffd_block_max(
        const float * __restrict__ q, const uint8_t * __restrict__ k2, const float * __restrict__ ks,
        const half * __restrict__ mask, float * __restrict__ bmax,
        const int D, const int G, const int n_kv, const int block, const int n_blocks,
        const int n_tokens, const int n_head_kv, const float scale,
        const int64_t q_nb1, const int64_t q_nb2, const int64_t q_nb3,
        const int64_t k2_nb1, const int64_t k2_nb2, const int64_t k2_nb3,
        const int64_t ks_nb1, const int64_t ks_nb2, const int64_t ks_nb3,
        const int64_t m_nb1, const int64_t m_nb2, const int64_t m_nb3, const int m_ne2, const int m_ne3) {
    extern __shared__ float smem[];
    float * sq  = smem;           // [G][D]
    float * ssc = smem + G*D;     // [D]
    float * red = ssc + D;        // [G][FFD_THREADS/WARP_SIZE]

    const int b   = blockIdx.x;
    const int t   = blockIdx.y;
    const int hkv = blockIdx.z % n_head_kv;
    const int s   = blockIdx.z / n_head_kv;
    const int tid = threadIdx.x;

    for (int i = tid; i < G*D; i += FFD_THREADS) {
        const int g = i / D, c = i % D;
        sq[i] = *(const float *) ((const char *) q + t*q_nb1 + (int64_t)(hkv*G + g)*q_nb2 + s*q_nb3 + c*sizeof(float));
    }
    for (int c = tid; c < D; c += FFD_THREADS) {
        ssc[c] = *(const float *) ((const char *) ks + b*ks_nb1 + hkv*ks_nb2 + s*ks_nb3 + c*sizeof(float));
    }
    __syncthreads();

    // per-thread running max per group head; G is small (bounded in supports_op)
    float vmax[16];
#pragma unroll
    for (int g = 0; g < 16; ++g) vmax[g] = -INFINITY;

    const int j0 = b*block;
    const int j1 = min(n_kv, j0 + block);
    for (int j = j0 + tid; j < j1; j += FFD_THREADS) {
        const uint8_t * kr = k2 + j*k2_nb1 + hkv*k2_nb2 + s*k2_nb3;
        for (int g = 0; g < G; ++g) {
            float mv = 0.0f;
            if (mask) {
                const half * mr = (const half *) ((const char *) mask + t*m_nb1 + ((hkv*G + g) % m_ne2)*m_nb2 + (s % m_ne3)*m_nb3);
                mv = __half2float(mr[j]);
                if (mv == -INFINITY) continue;
            }
            const float * qg = sq + g*D;
            float dot = 0.0f;
            for (int c4 = 0; c4 < D/4; ++c4) {
                const uint8_t byte = kr[c4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int c = 4*c4 + e;
                    dot += qg[c]*(((float) ((byte >> (2*e)) & 3) - 1.5f)*ssc[c]);
                }
            }
            vmax[g] = fmaxf(vmax[g], scale*dot + mv);
        }
    }

    const int warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    for (int g = 0; g < G; ++g) {
        const float w = ffd_warp_max(vmax[g]);
        if (lane == 0) red[g*(FFD_THREADS/WARP_SIZE) + warp] = w;
    }
    __syncthreads();
    if (tid < G) {
        float m = -INFINITY;
        for (int w = 0; w < FFD_THREADS/WARP_SIZE; ++w) m = fmaxf(m, red[tid*(FFD_THREADS/WARP_SIZE) + w]);
        bmax[((((int64_t) s*n_head_kv + hkv)*n_tokens + t)*G + tid)*n_blocks + b] = m;
    }
}

static __global__ void k_ffd_decide(
        const float * __restrict__ bmax, char * __restrict__ dst,
        const int G, const int n_blocks, const int n_tokens, const int n_head_kv, const float delta,
        const int64_t nb1, const int64_t nb2, const int64_t nb3) {
    const int t = blockIdx.x, hkv = blockIdx.y, s = blockIdx.z;
    const float * bm = bmax + (((int64_t) s*n_head_kv + hkv)*n_tokens + t)*G*n_blocks;
    int32_t * out = (int32_t *) (dst + t*nb1 + hkv*nb2 + s*nb3);
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        int32_t keep = 0;
        for (int g = 0; g < G && !keep; ++g) {
            const float * bg = bm + (int64_t) g*n_blocks;
            const float v = bg[b];
            if (v == -INFINITY) continue;
            const float th = fmaxf(bg[0], bg[n_blocks - 1]) - delta;
            keep = (b == n_blocks - 1 || v >= th) ? 1 : 0;
        }
        out[b] = keep;
    }
}

bool ggml_cuda_ffd_select_supported(const ggml_tensor * op) {
    const ggml_tensor * q  = op->src[0];
    const ggml_tensor * k2 = op->src[1];
    const int64_t G = q->ne[2]/k2->ne[2];
    const int64_t D = q->ne[0];
    // register array vmax[16]; shared memory (G+1)*D + G*warps floats must fit the default 48 KiB
    return G <= 16 && ((G + 1)*D + G*(FFD_THREADS/WARP_SIZE))*sizeof(float) <= 48*1024 &&
           ggml_is_contiguous(op) && q->nb[0] == sizeof(float) && k2->nb[0] == 1 && op->src[2]->nb[0] == sizeof(float);
}

void ggml_cuda_op_ffd_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q  = dst->src[0];
    const ggml_tensor * k2 = dst->src[1];
    const ggml_tensor * ks = dst->src[2];
    const ggml_tensor * m  = dst->src[3];

    const int D         = (int) q->ne[0];
    const int n_tokens  = (int) q->ne[1];
    const int ns        = (int) q->ne[3];
    const int n_kv      = (int) k2->ne[1];
    const int n_head_kv = (int) k2->ne[2];
    const int G         = (int) (q->ne[2]/k2->ne[2]);
    const int n_blocks  = (int) dst->ne[0];

    const int32_t block = ggml_get_op_params_i32(dst, 0);
    const float   scale = ggml_get_op_params_f32(dst, 1);
    const float   delta = ggml_get_op_params_f32(dst, 2);

    if (n_blocks == 0 || n_tokens == 0 || n_head_kv == 0 || ns == 0) {
        return;  // empty output; a zero-sized grid is an invalid launch configuration
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> bmax(ctx.pool(), (size_t) ns*n_head_kv*n_tokens*G*n_blocks);

    const size_t shmem = ((size_t) (G + 1)*D + (size_t) G*(FFD_THREADS/WARP_SIZE))*sizeof(float);
    const dim3 grid1(n_blocks, n_tokens, n_head_kv*ns);
    k_ffd_block_max<<<grid1, FFD_THREADS, shmem, stream>>>(
        (const float *) q->data, (const uint8_t *) k2->data, (const float *) ks->data,
        m ? (const half *) m->data : nullptr, bmax.get(),
        D, G, n_kv, block, n_blocks, n_tokens, n_head_kv, scale,
        q->nb[1], q->nb[2], q->nb[3], k2->nb[1], k2->nb[2], k2->nb[3], ks->nb[1], ks->nb[2], ks->nb[3],
        m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, m ? (int) m->ne[2] : 1, m ? (int) m->ne[3] : 1);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid2(n_tokens, n_head_kv, ns);
    k_ffd_decide<<<grid2, 128, 0, stream>>>(bmax.get(), (char *) dst->data, G, n_blocks, n_tokens, n_head_kv, delta,
                                            dst->nb[1], dst->nb[2], dst->nb[3]);
    CUDA_CHECK(cudaGetLastError());
}
