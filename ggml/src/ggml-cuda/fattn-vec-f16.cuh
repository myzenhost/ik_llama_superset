//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"
#include "fattn-vec-common.cuh"

template<int Dk, int Dv, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool sparse_v = false> // Dk, Dv == K, V head sizes
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(Dk, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_vec_ext_f16(
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
#if defined(FP16_AVAILABLE)

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

    constexpr vec_dot_KQ_f16_t vec_dot_KQ = get_vec_dot_KQ_f16<Dk>(type_K);

    // TurboQuant K/V: rotated-domain KV. K needs Q rotated once into the same basis (and NOT
    // quantised to q8_1 -- the turbo decode is float); V needs one cooperative inverse rotation of
    // the finished output vector. See fattn-turbo.cuh for the algebra and the measurements.
    constexpr bool turbo_K = turbo_fattn_type_supported(type_K);
    constexpr bool turbo_V = turbo_fattn_type_supported(type_V);

    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && type_K != GGML_TYPE_BF16 && !turbo_K;
    constexpr dequantize_1_f16_t dequantize_1_v = get_dequantize_1_f16(type_V);

    // STEP 4 GUARD. Before this, an unhandled K/V type fell through the constexpr selector chains
    // to `nullptr` -- a perfectly valid constexpr -- so the instantiation COMPILED CLEAN and then
    // dereferenced a null device function pointer at run time. These turn that silent crash class
    // into a build error naming the missing case.
    static_assert(vec_dot_KQ != nullptr,
        "no vec_dot_KQ case for this K type: add one to get_vec_dot_KQ_f16 rather than letting the "
        "selector fall through to nullptr (which compiles and then dereferences null at run time)");
    static_assert(dequantize_1_v != nullptr,
        "no dequantize_1 case for this V type: add one to get_dequantize_1_f16 rather than letting "
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

    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(Dk % (2*WARP_SIZE) == 0, "Dk not divisible by 2*WARP_SIZE == 64.");
    static_assert(Dv % (2*WARP_SIZE) == 0, "Dv not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = Dk / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < Dk);

    __shared__ half KQ[ncols*Dk];
    half2 * KQ2 = (half2 *) KQ;

    half kqmax[ncols];
    half kqsum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -HALF_MAX_HALF;
        kqsum[j] = 0.0f;
    }

    __shared__ half kqmax_shared[ncols][WARP_SIZE];
    __shared__ half kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ half maskh_shared[ncols*Dk];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskh_shared[j*Dk + tid] = 0.0f;
    }

    __syncthreads();

    half2 VKQ[ncols] = {{0.0f, 0.0f}};

    // Scratch for the two turbo rotations. They never overlap in time: the Q rotation happens once
    // before the KV loop, the V un-rotation once after it, so one buffer serves both. Sized 1 when
    // no turbo type is in play so nothing is spent on the ordinary paths.
    constexpr int turbo_smem_n = (turbo_K || turbo_V) ? (Dk > Dv ? Dk : Dv) : 1;
    __shared__ float turbo_smem[turbo_smem_n];

    const int k_VKQ_max = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].y : ne11;
    const int first_y = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].x : 0;
    if (first_y + blockIdx.y*Dk < k_VKQ_max) {

    // Convert Q to half2 (f16 K) or q8_1 (quantized K) and store in registers:
    half2  Q_h2[ncols][Dk/(2*WARP_SIZE)];
    int   Q_i32[ncols][Dk/(sizeof(int)*QK8_1) == 0 ? 1 : Dk/(sizeof(int)*QK8_1)];
    half2  Q_ds[ncols][Dk/QK8_1 == 0 ? 1 : Dk/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int   * tmp_q_i32 = (int   *) &KQ[j*Dk];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + Dk/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < Dk/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_half2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<half2>(Q_f + 4*i0, scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int   * tmp_q_i32 = (int   *) &KQ[j*Dk];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + Dk/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
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

                const float2 tmp = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
            }
        }
    }

    // ---- STEP 2: rotate Q ONCE into the turbo basis -------------------------------------------
    // (R q).(R k) == q.k, so rotating Q here lets every K read stay a raw per-element decode.
    // The butterfly runs in fp32 in shared memory even for this fp16 kernel; only the result is
    // narrowed back into Q_h2, which is the same precision Q already had.
    if constexpr (turbo_K) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            // Every warp built an identical copy of Q_h2; warp 0 publishes it.
            if (threadIdx.y == 0) {
#pragma unroll
                for (int i0 = 0; i0 < Dk/2; i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;
                    const float2 q = __half22float2(Q_h2[j][i0/WARP_SIZE]);
                    turbo_smem[2*i + 0] = q.x;
                    turbo_smem[2*i + 1] = q.y;
                }
            }
            __syncthreads();

            turbo_fattn_rotate_shared<turbo_smem_n, 0>(turbo_smem, tid);

#pragma unroll
            for (int i0 = 0; i0 < Dk/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                Q_h2[j][i0/WARP_SIZE] = make_half2(turbo_smem[2*i + 0], turbo_smem[2*i + 1]);
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*Dk + tid] = -HALF_MAX_HALF;
    }
    __syncthreads();

    const int k_VKQ_max = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].y : ne11;
    const int first_y = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].x : 0;
    //if (KV_min_max && threadIdx.x == 0 && head == 0) {
    //    printf("gridDims = %u, %u, %u, ncols = %d, head = %d, blockIdx.x = %d, blockIdx.y = %d, bounds = %d, %d, ne11 = %d, nb11 = %d, blockIdx.y*Dk = %d\n", gridDim.x, gridDim.y, gridDim.z, ncols, head, blockIdx.x, blockIdx.y, KV_min_max[sequence*gridDim.x + blockIdx.x].x, KV_min_max[sequence*gridDim.x + blockIdx.x].y, ne11, nb11, blockIdx.y*Dk);
    //}
    K     += (first_y + blockIdx.y*Dk) * nb11;
    V     += (first_y + blockIdx.y*Dk) * nb21;
    maskh += (first_y + blockIdx.y*Dk);
    for (int k_VKQ_0 = first_y + blockIdx.y*Dk; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*Dk,
             // Increment pointers after each loop:
             K += gridDim.y*Dk*nb11, V += gridDim.y*Dk*nb21, maskh += gridDim.y*Dk) {

        // For unknown reasons using a half array of size 1 for kqmax_new causes a performance regression,
        // see https://github.com/ggerganov/llama.cpp/pull/7061 .
        // Therefore this variable is defined twice but only used once (so that the compiler can optimize out the unused variable).
        half kqmax_new = kqmax[0];
        half kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskh_shared[j*Dk + tid] = slopeh*maskh[j*ne11 + tid];
            }
            __syncthreads();
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < Dk; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > Dk && i_KQ >= Dk) || (FATTN_KQ_STRIDE % Dk != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                half sum = vec_dot_KQ(K + i_KQ*nb11, Q_h2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum((float)sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += maskh_shared[j*Dk + i_KQ];

                if (ncols == 1) {
                    kqmax_new        = ggml_cuda_hmax(kqmax_new,        sum);
                } else {
                    kqmax_new_arr[j] = ggml_cuda_hmax(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ[j*Dk + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = ncols == 1 ? kqmax_new : kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(KQ[j*Dk + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*Dk + tid] = val;

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();

        // Accumulate V over all Dk scored KV positions of this block (not Dv). For asymmetric MLA
        // head sizes (Dk=576 K, Dv=512 V) the K/score loop above covers Dk KV rows, so the V loop and
        // the V pointer stride must also step Dk KV rows or K and V desync (= garbage decode on sm_60,
        // which uses this vec kernel for MLA -fa 1 batch=1). Dk==Dv leaves every other case unchanged.
#pragma unroll
        for (int k0 = 0; k0 < Dk; k0 += 2) {
            if (FATTN_KQ_STRIDE % Dk != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            // Sparse V (opt-in, ggml_cuda_fattn_vec_sparse_v_threshold): KQ2 is in shared memory, so every thread of
            // the block reads the same weights and takes the same branch -- no divergence, on any V type.
            if (sparse_v && sparse_v_thr > 0.0f) {
                bool keep = false;
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const float2 w = __half22float2(KQ2[j*(Dk/2) + k0/2]);
                    keep = keep || w.x >= sparse_v_thr || w.y >= sparse_v_thr;
                }
                if (!keep) {
                    continue;
                }
            }

            // For asymmetric Dk>Dv the V row is only Dv wide, so threads tid>=Dv have no V element
            // (their VKQ lane is discarded at output anyway). Read 0 to avoid stepping past the row.
            half2 V_k = make_half2(0.0f, 0.0f);
            if (tid < Dv) {
                reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k0 + 0)*nb21, tid);
                reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k0 + 1)*nb21, tid);
            }
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k*KQ2[j*(Dk/2) + k0/2];
            }
        }

        __syncthreads();
    }
    }

    if (sinksf && blockIdx.y == 0) {
        const half sink = __float2half(sinksf[head]);

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = fmaxf(kqmax[j], sink);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(sink - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale;

            if (tid == 0) {
                kqsum[j] += val;
            }

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();
    }

    // ---- STEP 3: un-rotate the FINISHED V accumulation, ONCE -----------------------------------
    // Every V read above was a per-element decode in the rotated domain, so VKQ currently holds
    // R*(sum_i a_i v_i). sum_i a_i (R v_i) == R (sum_i a_i v_i) by linearity, so one inverse
    // rotation here recovers the un-rotated result -- and it is the only place a whole head-dim
    // vector is block-visible (thread `tid` owns output element `tid`, and turbo requires Dk==Dv).
    //
    // VKQ is a half2 whose two halves are both partial sums of the SAME output element (the V loop
    // packs two KV rows per half2, not two dims -- the writeback below adds them). Collapse them
    // first, rotate the scalar, and put the result back in the low half with a zero high half so
    // that writeback is unchanged. Placed before the kqsum normalisation and the gridDim.y>1
    // partial writeback, both of which are linear in VKQ, so the rotation commutes with them.
    if constexpr (turbo_V) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            __syncthreads();
            turbo_smem[tid] = __half2float(__low2half(VKQ[j]) + __high2half(VKQ[j]));
            __syncthreads();

            turbo_fattn_rotate_shared<turbo_smem_n, 1>(turbo_smem, tid);

            VKQ[j] = make_half2(turbo_smem[tid], 0.0f);
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum((float)kqsum[j]);
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
        kqsum[j_VKQ] = warp_reduce_sum((float)kqsum[j_VKQ]);

        half dst_val = (__low2half(VKQ[j_VKQ]) + __high2half(VKQ[j_VKQ]));
        if (gridDim.y == 1) {
            // kqsum == 0 only for a fully masked row: write 0 like the CPU path, not 0/0 = NaN.
            dst_val = (float) kqsum[j_VKQ] > 0.0f ? dst_val / kqsum[j_VKQ] : __float2half(0.0f);
        }
        dst[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*Dv + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int Dk, int Dv, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap, bool sparse_v = false>
void ggml_cuda_flash_attn_ext_vec_f16_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = Dk/WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f16<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap, sparse_v>;
    constexpr bool need_f16_K = Dk != 128 && Dk != 192 && Dk != 256 && type_K == GGML_TYPE_F16; // only an f16 K is re-laid-out as f16 (a bf16/quantised K is read in its own format)
    constexpr bool need_f16_V = Dv != 64 && Dv != 128 && Dv != 256 && type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<Dv, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, Dv, need_f16_K, need_f16_V);
}

template <int Dk, int Dv, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

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
                ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap, true>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            }
        } else {
            constexpr bool use_logit_softcap = true;
            if (sparse_v) {
                ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap, true>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
            }
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block = 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 8;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_F16_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f16_case                     \
    <D, D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_VEC_F16_CASE_DKDV(Dk, Dv, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f16_case                     \
    <Dk, Dv, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

//extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16,  GGML_TYPE_IQ4_NL);
//extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL);
//extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_IQ4_NL);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL);
//extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16,  GGML_TYPE_IQ4_NL);
//extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16,  GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F16_CASE_DKDV(192, 128, GGML_TYPE_F16, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE_DKDV(192, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);

// TurboQuant KV. These extern template declarations are LOAD-BEARING, not tidiness: without them
// fattn-vec-f16.cu implicitly instantiates the turbo cases it dispatches, and that TU has not
// defined GGML_CUDA_FATTN_TURBO -- so the kernel's step-4 static_asserts fire on it. The explicit
// instantiation lives in the turbo instance files, which do opt in.
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO2_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);
extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO4_0);
