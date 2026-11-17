//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"
#include "fattn-vec-common.cuh"

// Currently llvm with the amdgcn target does not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template<int Dk, int Dv, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool sparse_v = false> // Dk, Dv == K-, V-head size
#ifndef GGML_USE_HIP
__launch_bounds__(Dk, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_vec_ext_f32(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int2 * __restrict__ KV_min_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const float sparse_v_thr,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {

    // Skip unused kernel variants for faster compilation:
    if constexpr (Dk == Dv || (Dk == 192 && Dv == 128) || (Dk == 576 && Dv == 512)) {
    if (use_logit_softcap && !(Dk == 128 || Dk == 256)) {
        NO_DEVICE_CODE;
        return;
    }
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f32_t vec_dot_KQ = get_vec_dot_KQ_f32<Dk>(type_K);

    // TurboQuant K/V: rotated-domain KV. K needs Q rotated once into the same basis (and NOT
    // quantised to q8_1 -- the turbo decode is float); V needs one cooperative inverse rotation of
    // the finished output vector. See fattn-turbo.cuh for the algebra and the measurements.
    constexpr bool turbo_K = turbo_fattn_type_supported(type_K);
    constexpr bool turbo_V = turbo_fattn_type_supported(type_V);

    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16 && !turbo_K;
    constexpr dequantize_1_f32_t dequantize_1_v = get_dequantize_1_f32(type_V);

    // STEP 4 GUARD. Before this, an unhandled K/V type fell through the constexpr selector chains
    // to `nullptr` -- a perfectly valid constexpr -- so the instantiation COMPILED CLEAN and then
    // dereferenced a null device function pointer at run time. These turn that silent crash class
    // into a build error naming the missing case.
    static_assert(vec_dot_KQ != nullptr,
        "no vec_dot_KQ case for this K type: add one to get_vec_dot_KQ_f32 rather than letting the "
        "selector fall through to nullptr (which compiles and then dereferences null at run time)");
    static_assert(dequantize_1_v != nullptr,
        "no dequantize_1 case for this V type: add one to get_dequantize_1_f32 rather than letting "
        "the selector fall through to nullptr (which compiles and then dereferences null at run time)");
    static_assert(!turbo_fattn_type_is_turbo(type_K) || turbo_K,
        "this turbo K type cannot be used by the fused fattn path here: the TU did not #define "
        "GGML_CUDA_FATTN_TURBO before including fattn-vec.cuh -- see fattn-turbo.cuh");
    static_assert(!turbo_fattn_type_is_turbo(type_V) || turbo_V,
        "this turbo V type cannot be used by the fused fattn path here: the TU did not #define "
        "GGML_CUDA_FATTN_TURBO before including fattn-vec.cuh -- see fattn-turbo.cuh");
    static_assert(!turbo_K || Dk % QK_TURBO3 == 0,
        "turbo K needs a head size that is a whole number of 128-element rotation groups");
    static_assert(!turbo_V || (Dv % QK_TURBO3 == 0 && Dk == Dv),
        "turbo V needs Dv a whole number of 128-element rotation groups, and one thread per output "
        "element (Dk == Dv) so the block can invert the rotation cooperatively");

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half  * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);
    const float * sinksf = (const float *) (sinks);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(Dk % (2*WARP_SIZE) == 0, "Dk not divisible by 2*WARP_SIZE == 64.");
    static_assert(Dv % (2*WARP_SIZE) == 0, "Dv not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = Dk / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < Dk);

    __shared__ float KQ[ncols*Dk];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*Dk + tid] = -FLT_MAX/2.0f;
    }

    float kqmax[ncols];
    float kqsum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
        kqsum[j] = 0.0f;
    }

    __shared__ float kqmax_shared[ncols][WARP_SIZE];
    __shared__ float kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ float maskf_shared[ncols*Dk];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskf_shared[j*Dk + tid] = 0.0f;
    }

    __syncthreads();

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
    float2  Q_f2[ncols][Dk/(2*WARP_SIZE)];
    int    Q_i32[ncols][(Dk/(int)sizeof(int) + WARP_SIZE - 1)/WARP_SIZE];
    float2  Q_ds[ncols][Dk/QK8_1 == 0 ? 1 : Dk/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*Dk];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + Dk/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < Dk/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
            // Full WARP_SIZE-wide int32 chunks: ni == WARP_SIZE so quantize_q8_1_to_shared's lane
            // guards fold away -> byte-identical to the original unguarded loop for aligned Dk.
            constexpr int Dk_i32  = Dk/(int)sizeof(int);
            constexpr int Dk_full = (Dk_i32/WARP_SIZE)*WARP_SIZE;
#pragma unroll
            for (int i0 = 0; i0 < Dk_full; i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2, WARP_SIZE>(Q_f + 4*i0, scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
            }
            // Ragged tail: only compiled for Dk whose int32 count is not a multiple of WARP_SIZE
            // (Dk=576 -> 16 valid lanes, Dk=192 -> 16 valid lanes). `if constexpr` drops this
            // entirely for aligned Dk (128/256), so the aligned path stays byte-identical.
            if constexpr (Dk_i32 % WARP_SIZE != 0) {
                constexpr int i0 = Dk_full;
                constexpr int ni = Dk_i32 - Dk_full;
                quantize_q8_1_to_shared<float2, ni>(Q_f + 4*i0, scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*Dk];
            float2 * tmp_q_ds  = (float2 *) (tmp_q_i32 + Dk/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                // Ragged-tail guard (mirrors mainline fattn-vec.cuh:169). Aligned Dk: predicate
                // folds to always-true -> byte-identical. Dk=576 ragged tail: OOB lanes (i >= 144)
                // load 0 rather than reading the trailing shared region (tmp_q_i32[144..159] is now
                // never written by the guarded quantize, so it is stale) and feed 0 to the dot.
                if (i0 + WARP_SIZE <= Dk/(int)sizeof(int) || i < Dk/(int)sizeof(int)) {
                    Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                    Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
                } else {
                    Q_i32[j][i0/WARP_SIZE] = 0;
                    Q_ds[j][i0/WARP_SIZE]  = make_float2(0.0f, 0.0f);
                }
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < Dk/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_f2[j][i0/WARP_SIZE]    = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_f2[j][i0/WARP_SIZE].x *= scale;
                Q_f2[j][i0/WARP_SIZE].y *= scale;
            }
        }
    }

    // Scratch for the two turbo rotations. They never overlap in time: the Q rotation happens once
    // before the KV loop, the V un-rotation once after it, so one buffer serves both. Sized 1 when
    // no turbo type is in play so nothing is spent on the ordinary paths.
    constexpr int turbo_smem_n = (turbo_K || turbo_V) ? (Dk > Dv ? Dk : Dv) : 1;
    __shared__ float turbo_smem[turbo_smem_n];

    // ---- STEP 2: rotate Q ONCE into the turbo basis -------------------------------------------
    // (R q).(R k) == q.k, so rotating Q here lets every K read stay a raw per-element decode.
    if constexpr (turbo_K) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            // Every warp built an identical copy of Q_f2; warp 0 publishes it.
            if (threadIdx.y == 0) {
#pragma unroll
                for (int i0 = 0; i0 < Dk/2; i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;
                    turbo_smem[2*i + 0] = Q_f2[j][i0/WARP_SIZE].x;
                    turbo_smem[2*i + 1] = Q_f2[j][i0/WARP_SIZE].y;
                }
            }
            __syncthreads();

            turbo_fattn_rotate_shared<turbo_smem_n, 0>(turbo_smem, tid);

#pragma unroll
            for (int i0 = 0; i0 < Dk/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                Q_f2[j][i0/WARP_SIZE].x = turbo_smem[2*i + 0];
                Q_f2[j][i0/WARP_SIZE].y = turbo_smem[2*i + 1];
            }
            __syncthreads();
        }
    }

    float VKQ[ncols] = {0.0f};

    const int k_VKQ_max = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].y : ne11;
    const int first_y = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].x : 0;

    K     += (first_y + blockIdx.y*Dk) * nb11;
    V     += (first_y + blockIdx.y*Dk) * nb21;
    maskh += (first_y + blockIdx.y*Dk);
    for (int k_VKQ_0 = first_y + blockIdx.y*Dk; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*Dk,
             // Increment pointers after each loop:
             K += gridDim.y*Dk*nb11, V += gridDim.y*Dk*nb21, maskh += gridDim.y*Dk) {

        // Calculate KQ tile and keep track of new maximum KQ values:

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskf_shared[j*Dk + tid] = slope*__half2float(maskh[j*ne11 + tid]);
            }
            __syncthreads();
        }

        float kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < Dk; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > Dk && i_KQ >= Dk) || (FATTN_KQ_STRIDE % Dk != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_f2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += maskf_shared[j*Dk + i_KQ];

                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], sum);

                if (threadIdx.x == 0) {
                    KQ[j*Dk + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = (FATTN_KQ_STRIDE % Dk != 0 && k_VKQ_0 + tid >= ne11) ? 0.0f : expf(KQ[j*Dk + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*Dk + tid] = val;

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();

        // Accumulate V over all Dk scored KV positions of this block (not Dv). For asymmetric MLA
        // head sizes (Dk=576 K, Dv=512 V) the K/score loop above covers Dk KV rows, so the V loop and
        // the V pointer stride must also step Dk KV rows or K and V desync (= garbage decode on sm_60,
        // which uses this vec kernel for MLA -fa 1 batch=1). Dk==Dv leaves every other case unchanged.
#pragma unroll
        for (int k = 0; k < Dk; ++k) {
            if (FATTN_KQ_STRIDE % Dk != 0 && k_VKQ_0 + k >= ne11) {
                break;
            }

            // Sparse V (opt-in): KQ is in shared memory, so the branch is block-uniform (see fattn-vec-f16.cuh).
            if (sparse_v && sparse_v_thr > 0.0f) {
                bool keep = false;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    keep = keep || KQ[j*Dk + k] >= sparse_v_thr;
                }
                if (!keep) {
                    continue;
                }
            }

            // For asymmetric Dk>Dv the V row is only Dv wide, so threads tid>=Dv have no V element
            // (their VKQ lane is discarded at output anyway). Read 0 to avoid stepping past the row.
            const float V_ki = tid < Dv ? dequantize_1_v(V + k*nb21, tid) : 0.0f;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_ki*KQ[j*Dk + k];
            }
        }

        __syncthreads();
    }

    if (sinksf && blockIdx.y == 0) {
        const float sink = sinksf[head];

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = fmaxf(kqmax[j], sink);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(sink - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale;

            if (tid == 0) {
                kqsum[j] += val;
            }

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();
    }

    // ---- STEP 3: un-rotate the FINISHED V accumulation, ONCE -----------------------------------
    // Every V read above was a per-element decode in the rotated domain, so VKQ currently holds
    // R*(sum_i a_i v_i). sum_i a_i (R v_i) == R (sum_i a_i v_i) by linearity, so one inverse
    // rotation here recovers the un-rotated result -- and it is the only place a whole head-dim
    // vector is block-visible (thread `tid` owns output element `tid`, and turbo requires Dk==Dv).
    //
    // Placed BEFORE the kqsum normalisation and before the gridDim.y>1 partial writeback, both of
    // which are linear in VKQ, so the rotation commutes with them either way.
    if constexpr (turbo_V) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            __syncthreads();
            turbo_smem[tid] = VKQ[j];
            __syncthreads();

            turbo_fattn_rotate_shared<turbo_smem_n, 1>(turbo_smem, tid);

            VKQ[j] = turbo_smem[tid];
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum(kqsum[j_VKQ]);

        float dst_val = VKQ[j_VKQ];
        if (gridDim.y == 1) {
            // kqsum == 0 only for a fully masked row: write 0 like the CPU path, not 0/0 = NaN.
            dst_val = kqsum[j_VKQ] > 0.0f ? dst_val / kqsum[j_VKQ] : 0.0f;
        }
        if (tid < Dv) {
            dst[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*Dv + tid] = dst_val;
        }
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

template <int Dk, int Dv, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool sparse_v = false>
void ggml_cuda_flash_attn_ext_vec_f32_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = Dk/WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap, sparse_v>;
    constexpr bool need_f16_K = (Dk != 128 && Dk != 256) && type_K == GGML_TYPE_F16;
    constexpr bool need_f16_V = (Dv != 64 && Dv != 128 && Dv != 256) && type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    // MLA (Dv=512): the KV cache is padded to FATTN_KQ_STRIDE (256), not to Dv, so use 256 as the KQ-row
    // granularity. Passing Dv=512 asserts n_kv % 512 == 0, which fails for odd multiples of 256.
    constexpr int kq_row_granularity = (Dv == 512) ? FATTN_KQ_STRIDE : Dv;
    launch_fattn<Dv, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, kq_row_granularity, need_f16_K, need_f16_V);
}

template <int Dk, int Dv, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f32_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (Q->ne[1] == 1 || GGML_CUDA_CC_IS_NVIDIA(cc)) {
        constexpr int cols_per_block = 1;
        // Sparse V is a template switch, not only a runtime test: a runtime branch in the V loop changed the
        // compiled default path (olmo3 q8_0-K/turbo2-V PPL 15.4298 -> 15.4377, MEASURED 2026-10-17). NVIDIA
        // always dispatches here, so only cols_per_block 1 carries the sparse variants.
        const bool sparse_v = ggml_cuda_fattn_vec_sparse_v_threshold() > 0.0f;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            if (sparse_v) {
                ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap, true>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            }
        } else {
            constexpr bool use_logit_softcap = true;
            if (sparse_v) {
                ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap, true>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            }
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block = 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    // both logit_softcap variants of the 8 column kernel in one module overflow the 16 bit branch
    // offset of the AMDGPU backend, but only for this instance. Any block width covers any
    // Q->ne[1], see the single column NVIDIA case above
    constexpr int cols_per_block = (Dk == 128 && type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) ? 4 : 8;
#else
    constexpr int cols_per_block = 8;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f32_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_F32_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f32_case                     \
    <D, D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_VEC_F32_CASE_DKDV(Dk, Dv, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f32_case                     \
    <Dk, Dv, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_F16,  GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F32_CASE_DKDV(192, 128, GGML_TYPE_F16, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE_DKDV(192, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F32_CASE_DKDV(576, 512, GGML_TYPE_F16, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE_DKDV(576, 512, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);

// TurboQuant KV. These extern template declarations are LOAD-BEARING, not tidiness: without them
// fattn-vec-f32.cu implicitly instantiates the turbo cases it dispatches, and that TU has not
// defined GGML_CUDA_FATTN_TURBO -- so the kernel's step-4 static_asserts fire on it. The explicit
// instantiation lives in the turbo instance files, which do opt in.
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);
