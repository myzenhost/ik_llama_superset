// Logging fallbacks for GGML_LOG_* macros (not available in ggml CUDA context)
#ifndef GGML_LOG_INFO
#include <cstdio>
#define GGML_LOG_INFO(...) fprintf(stderr, __VA_ARGS__)
#endif
#ifndef GGML_LOG_WARN
#include <cstdio>
#define GGML_LOG_WARN(...) fprintf(stderr, __VA_ARGS__)
#endif

#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

// ─── CUDA kernel ──────────────────────────────────────────────────────────────
//
// Templated on direction and group_size (128 or 64).
// One block per group, group_size threads per block.
// direction: 0 = forward (signs1 → WHT → signs2), 1 = inverse (signs2 → WHT → signs1)
//
// When head_dim is not a multiple of group_size, only the full groups
// within each head are processed.  Tail elements are left unchanged (identity).
//
// Algorithm mirrors the CPU implementation in ggml-cpu/ops.cpp:
//   1. Apply s_first elementwise
//   2. Radix-2 Hadamard butterfly (log2(group_size) stages, in-place)
//   3. Normalize by 1/sqrt(group_size) and apply s_second elementwise
//
// InnerQ scale_inv: when non-null, applies per-channel inverse scaling for
// Q/V equalization. For forward (Q rotation): multiply BEFORE signs+WHT.
// For inverse (V un-rotation): multiply AFTER WHT+signs.

template <int direction, int group_size>
static __global__ void k_turbo_wht_f32(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        const float * __restrict__ scale_inv,
                                        int64_t n_groups,
                                        int64_t head_dim,
                                        int64_t groups_per_head) {
    static_assert(group_size == 128 || group_size == 64 || group_size == 32, "group_size must be 128, 64, or 32");

    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;  // 0 .. group_size-1

    // Map group index to position in the tensor:
    // each head has groups_per_head full groups, then a gap of tail elements.
    const int64_t head_idx     = g / groups_per_head;
    const int64_t grp_in_head  = g % groups_per_head;
    const int64_t base         = head_idx * head_dim + grp_in_head * group_size;

    __shared__ float x[group_size];

    // Load from global memory
    x[t] = src[base + t];
    __syncthreads();

    // InnerQ forward: apply scale_inv BEFORE signs+WHT (for Q pre-rotation)
    if (direction == 0 && scale_inv != nullptr) {
        x[t] *= scale_inv[t % group_size];
        __syncthreads();
    }

    // Apply first sign array
    if (group_size == 128) {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    } else if (group_size == 64) {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1_64[t] : TURBO_WHT_SIGNS2_64[t];
    } else if (direction == 0) {
        // group_size == 32 has ONE sign vector S (TQ_WEIGHT_SIGNS): forward = H(S x)/sqrt(32), so the inverse is
        // S H(y)/sqrt(32) -- signs AFTER the butterfly (applied below). This path used to apply S BEFORE the
        // butterfly in both directions ("same for forward and inverse"), which made "inverse" a second forward:
        // round trip error 2.55 on unit-range data, measured 2026-09-27 by tests/test-turbo-wht.cpp. Forward output
        // is unchanged by the fix; no graph in src/ called direction 1 with group 32.
        x[t] *= TQ_WEIGHT_SIGNS[t];
    }
    __syncthreads();

    // WHT butterfly — log2(group_size) stages.
    // In stage h, threads where (t % (2h)) < h read x[t] and x[t+h],
    // then write x[t] = a+b and x[t+h] = a-b.  Each active thread
    // owns a disjoint pair, so no intra-stage conflicts exist.
#define WHT_STAGE(h) \
    if (t % (2*(h)) < (h)) { float a = x[t], b = x[t+(h)]; x[t] = a+b; x[t+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    if (group_size >= 64) { WHT_STAGE(32) }
    if (group_size == 128) { WHT_STAGE(64) }
#undef WHT_STAGE

    // Normalize and apply second sign array, write to output
    constexpr float inv_sqrt = (group_size == 128) ? 0.08838834764831845f :
                               (group_size == 64)  ? 0.125f :
                                                     0.17677669529663688f; // 1/sqrt(32)
    float result;
    if (group_size == 128) {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
    } else if (group_size == 64) {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2_64[t] : TURBO_WHT_SIGNS1_64[t]);
    } else {
        // group_size == 32: forward applied S before the butterfly; inverse applies it here, after
        result = x[t] * inv_sqrt * ((direction == 0) ? 1.0f : TQ_WEIGHT_SIGNS[t]);
    }

    // InnerQ inverse: apply scale_inv AFTER WHT+signs (for V un-rotation)
    if (direction == 1 && scale_inv != nullptr) {
        result *= scale_inv[t % group_size];
    }

    dst[base + t] = result;
}

// ─── Fast path: group_size == 128 (design from TheTom/llama-cpp-turboquant, MIT; written here, not pasted) ───
//
// One group per warp; lane t holds elements 4t..4t+3 as a float4. Stages h=1,2 pair elements inside the
// lane; stages h=4..64 pair lane t with lane t^(h/4) in the same slot (h >= 4 leaves the low two index
// bits unchanged), so no shared memory and no block barrier is needed.
//
// BIT-IDENTICAL to k_turbo_wht_f32<direction, 128>: same stage order (1,2,4,...,64), same pairing, the
// lower index forms a+b and the upper a-b (so the shuffled operand is subtracted FROM on the upper lane),
// same multiply order (x * inv_sqrt, then the sign), and flipping the sign bit equals a multiply by -1.0f
// because every sign entry is exactly +-1.0f (scripts/check-turbo-tables-consistent.py enforces both that
// and the packed words). Modelled on the host in float32 against the original before this was written:
// 160/160 bit-identical cases, and a flipped packed bit is caught 20/20.

static __device__ __forceinline__ float turbo_wht_sign_flip(float x, unsigned bit) {
    return __uint_as_float(__float_as_uint(x) ^ (bit << 31));
}

template <int direction, int warps_per_block>
static __global__ void k_turbo_wht_f32_fast(const float * __restrict__ src,
                                            float * __restrict__ dst,
                                            const float * __restrict__ scale_inv,
                                            int64_t n_groups,
                                            int64_t head_dim,
                                            int64_t groups_per_head) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // A group is a whole warp, so this return never splits one; the full-mask shuffles below depend on that.
    const int64_t g = (int64_t) blockIdx.x * warps_per_block + warp;
    if (g >= n_groups) return;

    const int64_t head_idx    = g / groups_per_head;
    const int64_t grp_in_head = g % groups_per_head;
    const int64_t base        = head_idx * head_dim + grp_in_head * 128;

    float4 v = *((const float4 *) (src + base) + lane);

    // InnerQ forward: scale before signs+WHT, as in the generic kernel.
    if (direction == 0 && scale_inv != nullptr) {
        const float4 s = *((const float4 *) scale_inv + lane);
        v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
    }

    // Lane t's four elements share word t>>3; their bits are the nibble at 4*(t&7).
    {
        const unsigned * s = (direction == 0) ? TURBO_WHT_SIGNBITS1 : TURBO_WHT_SIGNBITS2;
        const unsigned nib = s[lane >> 3] >> (4 * (lane & 7));
        v.x = turbo_wht_sign_flip(v.x, (nib     ) & 1u);
        v.y = turbo_wht_sign_flip(v.y, (nib >> 1) & 1u);
        v.z = turbo_wht_sign_flip(v.z, (nib >> 2) & 1u);
        v.w = turbo_wht_sign_flip(v.w, (nib >> 3) & 1u);
    }

    // h = 1: pairs (x,y) and (z,w)
    { const float a = v.x, b = v.y; v.x = a + b; v.y = a - b;
      const float c = v.z, d = v.w; v.z = c + d; v.w = c - d; }

    // h = 2: pairs (x,z) and (y,w)
    { const float a = v.x, b = v.z; v.x = a + b; v.z = a - b;
      const float c = v.y, d = v.w; v.y = c + d; v.w = c - d; }

    // h = 4..64: partner is lane t^m (m = h/4), same slot; the upper lane forms partner - self.
#define WHT_SHFL_STAGE(m)                                       \
    {                                                           \
        const float px = __shfl_xor_sync(0xffffffff, v.x, (m)); \
        const float py = __shfl_xor_sync(0xffffffff, v.y, (m)); \
        const float pz = __shfl_xor_sync(0xffffffff, v.z, (m)); \
        const float pw = __shfl_xor_sync(0xffffffff, v.w, (m)); \
        const bool hi = (lane & (m)) != 0;                      \
        v.x = hi ? px - v.x : v.x + px;                         \
        v.y = hi ? py - v.y : v.y + py;                         \
        v.z = hi ? pz - v.z : v.z + pz;                         \
        v.w = hi ? pw - v.w : v.w + pw;                         \
    }

    WHT_SHFL_STAGE(1)    // h = 4
    WHT_SHFL_STAGE(2)    // h = 8
    WHT_SHFL_STAGE(4)    // h = 16
    WHT_SHFL_STAGE(8)    // h = 32
    WHT_SHFL_STAGE(16)   // h = 64
#undef WHT_SHFL_STAGE

    {
        constexpr float inv_sqrt = 0.08838834764831845f;  // 1/sqrt(128)
        const unsigned * s = (direction == 0) ? TURBO_WHT_SIGNBITS2 : TURBO_WHT_SIGNBITS1;
        const unsigned nib = s[lane >> 3] >> (4 * (lane & 7));
        v.x = turbo_wht_sign_flip(v.x * inv_sqrt, (nib     ) & 1u);
        v.y = turbo_wht_sign_flip(v.y * inv_sqrt, (nib >> 1) & 1u);
        v.z = turbo_wht_sign_flip(v.z * inv_sqrt, (nib >> 2) & 1u);
        v.w = turbo_wht_sign_flip(v.w * inv_sqrt, (nib >> 3) & 1u);
    }

    // InnerQ inverse: scale after WHT+signs, as in the generic kernel.
    if (direction == 1 && scale_inv != nullptr) {
        const float4 s = *((const float4 *) scale_inv + lane);
        v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
    }

    *((float4 *) (dst + base) + lane) = v;
}

// ─── Simple copy kernel for tail elements (identity pass-through) ────────────

static __global__ void k_turbo_wht_copy_tail(const float * __restrict__ src,
                                              float * __restrict__ dst,
                                              int64_t n_heads,
                                              int64_t head_dim,
                                              int64_t tail_offset,
                                              int tail_size) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_heads * tail_size) return;

    const int64_t head_idx  = i / tail_size;
    const int64_t tail_elem = i % tail_size;
    const int64_t offset    = head_idx * head_dim + tail_offset + tail_elem;
    dst[offset] = src[offset];
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * scale_tensor = dst->src[1];  // InnerQ scale_inv (may be NULL)

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction;
    int group_size;
    // op_params is an int32_t ARRAY: `op_params + sizeof(int)` is element 4, not byte 4. ggml_turbo_wht writes
    // direction to [0] and group_size to [1] (ggml.c). Reading [4] (always 0) tripped the assert below on every
    // call -- found 2026-09-27 by tests/test-turbo-wht.cpp, the first test ever to run this op on CUDA.
    direction  = dst->op_params[0];
    group_size = dst->op_params[1];

    const int64_t head_dim        = src->ne[0];
    const int64_t n_heads         = ggml_nelements(src) / head_dim;

    GGML_ASSERT(group_size == 32 || group_size == 64 || group_size == 128);
    const int64_t groups_per_head = head_dim / group_size;
    const int     tail_size       = (int)(head_dim % group_size);
    const int64_t n_groups        = groups_per_head * n_heads;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;
    const float * scale_inv_ptr = scale_tensor ? (const float *) scale_tensor->data : nullptr;

    cudaStream_t stream = ctx.stream();

    // Process full groups
    if (n_groups > 0) {
        dim3 blocks(n_groups);
        // The fast kernel covers the 128-wide shape the KV cache uses; it needs float4 indexing and 16-byte
        // aligned src/dst/scale_inv. GGML_CUDA_TURBO_WHT_NO_FAST forces the generic kernel, so the two can be
        // compared bit for bit on the GPU (tests/test-turbo-wht.cpp). The env var is read once per process.
        static const bool fast_disabled = getenv("GGML_CUDA_TURBO_WHT_NO_FAST") != nullptr;
        const bool fast_ok =
            !fast_disabled &&
            group_size == 128 &&
            (head_dim % 4) == 0 &&
            (((uintptr_t) src_ptr | (uintptr_t) dst_ptr) % 16) == 0 &&
            (scale_inv_ptr == nullptr || ((uintptr_t) scale_inv_ptr % 16) == 0);
        if (fast_ok) {
            constexpr int warps = 4;   // 1 warp per block underfeeds the SMs; 2 and up saturate (TheTom's measurement, to be re-measured here)
            const int64_t n_blocks = (n_groups + warps - 1) / warps;
            if (direction == 0) {
                k_turbo_wht_f32_fast<0, warps><<<(int) n_blocks, warps*32, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32_fast<1, warps><<<(int) n_blocks, warps*32, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else if (group_size == 128) {
            dim3 threads(128);
            if (direction == 0) {
                k_turbo_wht_f32<0, 128><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 128><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else if (group_size == 64) {
            dim3 threads(64);
            if (direction == 0) {
                k_turbo_wht_f32<0, 64><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 64><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        } else {
            dim3 threads(32);
            if (direction == 0) {
                k_turbo_wht_f32<0, 32><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            } else {
                k_turbo_wht_f32<1, 32><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
            }
        }
    }

    // Pass through tail elements unchanged (no rotation)
    // Not needed for 64-aligned dims but kept for completeness
    if (tail_size > 0) {
        const int64_t total_tail = n_heads * tail_size;
        const int block_sz = 256;
        const int n_blocks = (int)((total_tail + block_sz - 1) / block_sz);
        k_turbo_wht_copy_tail<<<n_blocks, block_sz, 0, stream>>>(
            src_ptr, dst_ptr, n_heads, head_dim, groups_per_head * group_size, tail_size);
    }
}

// ─── Backend-internal rotations for the tensor-core flash-attention route (fattn.cu) ──────────────────────
//
// The vec kernels rotate Q and un-rotate the output INSIDE the kernel (fattn-turbo.cuh). The MMA kernels have
// no turbo code, so ggml_cuda_fattn_turbo_via_mma rotates Q into a scratch before the launch and un-rotates the
// finished output after it, using the SAME kernels as the GGML_OP_TURBO_WHT op above (bit-identical to the op,
// pinned by tests/test-turbo-wht.cpp). A flash-attention Q is a permuted view (token stride != head stride), so
// it is first copied to a contiguous scratch; the rotation then runs in place on that scratch.

static __global__ void k_copy_f32_strided(const char * __restrict__ src, float * __restrict__ dst,
                                          int64_t ne0, int64_t ne1, int64_t ne2, int64_t n,
                                          size_t nb1, size_t nb2, size_t nb3) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t i0 = i % ne0;
    int64_t r        = i / ne0;
    const int64_t i1 = r % ne1;
    r /= ne1;
    const int64_t i2 = r % ne2;
    const int64_t i3 = r / ne2;
    dst[i] = *(const float *) (src + i1*nb1 + i2*nb2 + i3*nb3 + i0*sizeof(float));
}

static void turbo_wht_inplace_contiguous(ggml_backend_cuda_context & ctx, const ggml_tensor * like, float * data, int direction) {
    GGML_ASSERT(like->ne[0] % 128 == 0);
    ggml_tensor view = *like;                  // shape / type / nb carry over; only data, op and params change
    view.type = GGML_TYPE_F32;
    view.nb[0] = sizeof(float);
    view.nb[1] = view.nb[0] * view.ne[0];
    view.nb[2] = view.nb[1] * view.ne[1];
    view.nb[3] = view.nb[2] * view.ne[2];
    view.data = data;
    view.src[0] = nullptr;
    ggml_tensor out = view;
    out.src[0] = &view;
    out.src[1] = nullptr;                      // no InnerQ scale: the vec path applies none either
    out.op_params[0] = direction;
    out.op_params[1] = 128;
    ggml_cuda_turbo_wht(ctx, &out);            // in place: every 128-group is read then written by one block / warp
}

void ggml_cuda_turbo_rotate_forward_to(ggml_backend_cuda_context & ctx, const ggml_tensor * q, float * dst) {
    GGML_ASSERT(q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float));
    const int64_t n = ggml_nelements(q);
    if (n > 0) {
        const int block_sz = 256;
        const int64_t n_blocks = (n + block_sz - 1) / block_sz;
        k_copy_f32_strided<<<(unsigned) n_blocks, block_sz, 0, ctx.stream()>>>((const char *) q->data, dst, q->ne[0], q->ne[1], q->ne[2], n, q->nb[1], q->nb[2], q->nb[3]);
    }
    turbo_wht_inplace_contiguous(ctx, q, dst, /*direction=*/0);
}

void ggml_cuda_turbo_rotate_inverse_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    turbo_wht_inplace_contiguous(ctx, t, (float *) t->data, /*direction=*/1);
}
