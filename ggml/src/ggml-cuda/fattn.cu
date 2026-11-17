//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16-interface.cuh"
#include "fattn-vec-f32-interface.cuh"
#include "fattn-wmma-f16-interface.cuh"
#include "fattn-mma-f16-interface.cuh"
#include "fattn-new-mma.cuh"
#include "fattn.cuh"
#include "fattn-force.h"
#include "dsa_attn.cuh"
// Only for turbo_fattn_type_is_turbo(); without GGML_CUDA_FATTN_TURBO defined this header brings in
// two constexpr predicates over the type enum and nothing else -- no turbo tables, no device code.
#include "fattn-turbo.cuh"
#include "convert.cuh"
#include "turbo-wht.cuh"

#include <cstdint>

// Sparse V threshold for the CUDA vec flash-attention kernels (declared in fattn-vec-common.cuh).
float ggml_cuda_fattn_vec_sparse_v_threshold() {
    static const float thr = [] {
        const char * e = getenv("GGML_CUDA_FA_SPARSE_V");
        if (!e || !*e || (e[0] == '0' && e[1] == '\0')) {
            return 0.0f;
        }
        float v = (e[0] == '1' && e[1] == '\0') ? 1e-3f : (float) atof(e);
        if (!(v > 0.0f) || v >= 1.0f) {
            GGML_LOG_WARN("ggml_cuda_fattn_vec_sparse_v_threshold: GGML_CUDA_FA_SPARSE_V=%s is not a threshold in (0,1); sparse V stays off\n", e);
            return 0.0f;
        }
        GGML_LOG_INFO("ggml_cuda_fattn_vec_sparse_v_threshold: sparse V enabled in the CUDA vec flash-attention kernels, threshold %g\n", v);
        return v;
    }();
    return thr;
}

#define FATTN_KQ_STRIDE 256

static inline bool mma_better_than_turing(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) > GGML_CUDA_CC_TURING;
}

// On GPUs without fp16 MMA (Pascal/sm_60), the MLA absorbed head sizes 576/512 are asymmetric, so the f16
// vector kernel (which requires K==V) rejects them and decode falls back to the CPU. Route MLA decode
// (batch <= 8) to the f32 vector kernel to keep it on the GPU. Single-sourced so the dispatch and the
// is_supported check cannot drift.
static inline bool is_pascal_mla_absorbed_decode(const ggml_tensor * Q, const ggml_tensor * K, const ggml_tensor * V) {
    return Q->ne[1] <= 8 && K->ne[0] == 576 && V->ne[0] == 512;
}

// The MLA instantiations of the mma_new kernel (576/512, 320/256: is_mla in fattn-new-mma.cu) never
// read V. They take V as the last DV columns of K (V_h2 = K_h2 + (DKQ/2 - DV/2)) and reuse K's
// shared-memory tile for it (reusable_cutoff), which is exactly how build_deepseek2.cpp lays the
// cache out: V is a view of the same cache at offset row_size(n_embd_head_qk_rope). Any other V --
// an independent tensor, or a view of K at offset 0 as upstream's MLA graph builds it -- was
// silently replaced by K's tail: test-backend-ops ERR ~2 on every 576/512 and 320/256 case. Compared
// through view_src/view_offs because supports_op may run before the tensors are allocated.
static bool ggml_cuda_fattn_v_is_k_tail(const ggml_tensor * K, const ggml_tensor * V) {
    if (V->type != K->type) return false;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (V->ne[i] != K->ne[i] || V->nb[i] != K->nb[i]) return false;
    }
    const ggml_tensor * k_root = K->view_src ? K->view_src : K;
    const ggml_tensor * v_root = V->view_src ? V->view_src : V;
    if (k_root != v_root) return false;
    const size_t k_offs = K->view_src ? K->view_offs : 0;
    const size_t v_offs = V->view_src ? V->view_offs : 0;
    return v_offs == k_offs + ggml_row_size(K->type, K->ne[0] - V->ne[0]);
}

// Every precondition ggml_cuda_flash_attn_ext_mma_new (fattn-new-mma.cu) asserts or aborts on, in
// one place, so the dispatch and ggml_cuda_fattn_is_supported cannot drift from the launcher. Both
// used to route by head size alone: a masked-out or ALiBi (max_bias != 0) case, a multi-sequence Q,
// or an unlisted GQA ratio reached GGML_ASSERT(use_gqa_opt) / GGML_ABORT("Not implemented") and
// killed the process (test-backend-ops FLASH_ATTN_EXT, fattn-new-mma.cu:2239).
static bool ggml_cuda_fattn_mma_new_can_run(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if (!mask || max_bias != 0.0f) return false;
    if (Q->ne[3] != 1)             return false;
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) return false;

    const int gqa = Q->ne[2] / K->ne[2];
    const int64_t dq = Q->ne[0], dk = K->ne[0], dv = V->ne[0];
    if (dk == 128 && dq == 128 && dv == 128) return gqa == 12 || gqa == 6 || gqa == 10;
    if (dk == 256 && dq == 256 && dv == 256) return gqa == 6;
    if (dk == 192 && dq == 192 && dv == 128) return true;
    if (dk == 192 && dq == 192 && dv == 192) return gqa == 1;
    if (dk == 320 && dq == 320 && dv == 256) return gqa % 16 == 0 && ggml_cuda_fattn_v_is_k_tail(K, V);
    if (dk == 512 && dq == 512 && dv == 512) return gqa % 2 == 0;
    if (dk == 576 && dq == 576 && dv == 512) return gqa % 4 == 0 && ggml_cuda_fattn_v_is_k_tail(K, V);
    return false;
}

// ---------------------------------------------------------------------------------------------------------------
// TurboQuant KV on the tensor-core (MMA) kernel, for PREFILL-sized batches. OPT-IN: GGML_TURBO_MMA_PREFILL=<n>
// routes a turbo K/V flash attention with at least <n> queries through the f16 MMA kernel; unset or 0 keeps every
// turbo K/V on the vec kernels (the previous behaviour). It stays opt-in until tests/test-turbo-fa.cpp and
// test-backend-ops FLASH_ATTN_EXT pass on a CUDA device with the variable set and a prefill A/B is recorded (the GPU
// is not available to the agent that wrote this; row S69-G-TURBO-MMA-PREFILL-GPU-VERIFY-2026-10-05-01).
//
// HOW IT IS EXACT. K and V are stored ROTATED: k' = R k, v' = R v with R orthogonal. Then (R q).k' = q.k and
// sum_i p_i v'_i = R (sum_i p_i v_i). So: (1) rotate Q forward into a scratch, (2) let launch_fattn_mma convert the
// turbo K/V to f16 in the rotated basis (convert.cu -> turbo-convert.cu) and run the ordinary f16 MMA kernel, (3)
// rotate the finished output back in place. K and V are handled independently: Q is rotated only when K is turbo and
// the output only when V is turbo, so f16 x turbo and turbo x f16 mixes are right too.
//
// DESIGN CHOICE (the spec of S69-THETOM-FATTN-MMA-TURBO-TENSORCORE-PREFILL-ADOPT-2026-10-05-01, options (a)/(b)):
// TheTom's tip rotates Q in the GRAPH (llama-graph.cpp, which this fork has no counterpart of: zero ggml_turbo_wht
// calls under src/) and dequantises K/V tiles INSIDE a modified MMA kernel (fattn-mma-turbo.cuh + fattn-mma-f16.cuh
// load tiles) for GQA-packed DECODE. This route is the other end of the trade: no change to the MMA kernel, the
// rotation is a thin pre/post kernel pair (option (b)), and the cost is one f16 expansion of the K/V span per call,
// which a prefill batch amortises and a single-token decode does not. Decode on this path would be slower than vec,
// so the threshold is a query count, not a flag.
// ---------------------------------------------------------------------------------------------------------------
static int ggml_cuda_turbo_mma_prefill_min_q() {
    static const int v = []{
        const char * s = getenv("GGML_TURBO_MMA_PREFILL");
        return s ? atoi(s) : 0;
    }();
    return v;
}

static bool ggml_cuda_fattn_turbo_mma_prefill_applies(const ggml_tensor * dst, int cc) {
    const int min_q = ggml_cuda_turbo_mma_prefill_min_q();
    if (min_q <= 0) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (Q->ne[1] < min_q || Q->type != GGML_TYPE_F32) {
        return false;
    }
    // The route is the f16 MMA kernel of ggml_cuda_flash_attn_ext below; it needs the same hardware and the same shapes.
    if (!new_mma_available(cc) || !ggml_cuda_fattn_mma_f16_is_supported(ggml_cuda_get_device(), dst)) {
        return false;
    }
    // Whole 128-element rotation groups (turbo blocks), and a converter for each side.
    if (Q->ne[0] % 128 != 0 || K->ne[0] != Q->ne[0] || V->ne[0] != Q->ne[0]) {
        return false;
    }
    return ggml_get_to_fp16_cuda(K->type) != nullptr && ggml_get_to_fp16_cuda(V->type) != nullptr;
}

static void ggml_cuda_fattn_turbo_via_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const bool rotate_q       = turbo_fattn_type_is_turbo(dst->src[1]->type);
    const bool unrotate_out   = turbo_fattn_type_is_turbo(dst->src[2]->type);

    ggml_cuda_pool_alloc<float> q_rot(ctx.pool());
    ggml_tensor Ql;
    ggml_tensor local_dst = *dst;
    if (rotate_q) {
        q_rot.alloc(ggml_nelements(Q));
        ggml_cuda_turbo_rotate_forward_to(ctx, Q, q_rot.get());
        Ql = *Q;
        Ql.data  = q_rot.get();
        Ql.nb[0] = sizeof(float);
        Ql.nb[1] = Ql.nb[0] * Ql.ne[0];
        Ql.nb[2] = Ql.nb[1] * Ql.ne[1];
        Ql.nb[3] = Ql.nb[2] * Ql.ne[2];
        local_dst.src[0] = &Ql;
    }
    ggml_cuda_flash_attn_ext_mma_f16(ctx, &local_dst);
    if (unrotate_out) {
        ggml_cuda_turbo_rotate_inverse_inplace(ctx, dst);
    }
}

// Pascal has no other kernel for head size 512 (the Gemma-4 global layers), in prefill or decode.
static inline bool is_pascal_hs512(const int cc, const ggml_tensor * K, const ggml_tensor * V) {
    return cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_VOLTA && K->ne[0] == 512 && V->ne[0] == 512;
}

static __global__ void k_index_to_mask(const int * __restrict__ idx, half * __restrict__ mask, const int n_idx, const int s1, const int n_kv) {
    idx  += int64_t(s1)*blockIdx.x;
    mask += int64_t(n_kv)*blockIdx.x;
    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) mask[j] = __float2half(-INFINITY);
    __syncthreads();
    for (int j = threadIdx.x; j < n_idx; j += blockDim.x) {
        if (idx[j] >= 0) mask[idx[j]] = __float2half(0.0f);
    }
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[GGML_FLASH_ATTN_EXT_N_SWA_SLOT];

    ggml_cuda_pool_alloc<half> idx_mask(ctx.pool());
    ggml_tensor idx_dst, Mi;
    if (dst->src[5]) {
        if (ggml_cuda_flash_attn_ext_mma_new_sparse(ctx, dst)) {
            return;
        }
        if (!mask) {
            const ggml_tensor * kv_idx = dst->src[5];
            Mi = *kv_idx;
            Mi.type  = GGML_TYPE_F16;
            Mi.ne[0] = K->ne[1];
            Mi.nb[0] = sizeof(half);
            Mi.nb[1] = Mi.ne[0]*Mi.nb[0];
            Mi.nb[2] = Mi.ne[1]*Mi.nb[1];
            Mi.nb[3] = Mi.ne[2]*Mi.nb[2];
            Mi.data  = idx_mask.alloc(ggml_nelements(&Mi));
            k_index_to_mask<<<ggml_nrows(&Mi), 256, 0, ctx.stream()>>>((const int *)kv_idx->data, (half *)Mi.data,
                    kv_idx->ne[0], kv_idx->nb[1]/sizeof(int), Mi.ne[0]);
            CUDA_CHECK(cudaGetLastError());
            idx_dst = *dst;
            idx_dst.src[3] = &Mi;
            dst  = &idx_dst;
            mask = &Mi;
        }
        if (ggml_cuda_dsa_attn_ext(ctx, dst)) {
            return;
        }
    }

    ggml_tensor local_dst, Kl, Vl, Ml;
    if (n_swa > 0) {
        int ntokens = std::max(FATTN_KQ_STRIDE, int(Q->ne[1]));
        int nton = FATTN_KQ_STRIDE*((ntokens + n_swa + FATTN_KQ_STRIDE - 1)/FATTN_KQ_STRIDE);
        int first = K->ne[1] - nton;
        local_dst = *dst;
        local_dst.op_params[GGML_FLASH_ATTN_EXT_N_SWA_SLOT] = 0;
        if (first > 0) {
            local_dst = *dst;
            Kl = *K; Kl.ne[1] = nton; Kl.data = (char *)K->data + K->nb[1]*first;
            Vl = *V; Vl.ne[1] = nton; Vl.data = (char *)V->data + V->nb[1]*first;
            Ml = *mask; Ml.ne[0] = nton; Ml.data = (char *)mask->data + mask->nb[0]*first;
            local_dst.src[1] = &Kl;
            local_dst.src[2] = &Vl;
            local_dst.src[3] = &Ml;
            local_dst.op_params[GGML_FLASH_ATTN_EXT_N_SWA_SLOT] = 0;
            dst = &local_dst;
        }
        dst = &local_dst;
    }

    // TurboQuant KV is handled ONLY by the vec kernels (see fattn-turbo.cuh): they are the ones that
    // rotate Q once and inverse-rotate the finished V vector once. The tile/wmma/mma kernels have no
    // turbo decode at all, and ggml_is_quantized() is true for the turbo types, so the generic
    // quantized-KV logic further down would hand them to an mma path that cannot read them. Route
    // explicitly, before any of that. ggml_cuda_fattn_is_supported() gates the same condition, so a
    // combination the vec kernels do not instantiate never reaches here in the first place.
    if (turbo_fattn_type_is_turbo(K->type) || turbo_fattn_type_is_turbo(V->type)) {
        if (ggml_cuda_fattn_turbo_mma_prefill_applies(dst, cc)) {
            ggml_cuda_fattn_turbo_via_mma(ctx, dst);
            return;
        }
        if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    // On AMD the tile kernels perform poorly, use the vec kernel instead:
    if (cc >= GGML_CUDA_CC_OFFSET_AMD) {
        if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    // GGML_CUDA_FATTN_FORCE=vec|tile|mma: measurement override. Honoured only when the family is legal
    // for this node (the same predicates the static dispatch below uses); otherwise declined and the
    // static dispatch runs unchanged. Unset = today's behaviour. See fattn-force.h.
    {
        static const fattn_force_kind force_want = fattn_force_parse(getenv("GGML_CUDA_FATTN_FORCE"));
        if (force_want != FATTN_FORCE_NONE) {
            const bool legal_vec  = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0 && Q->ne[1] <= 8;
            const bool legal_tile = ggml_cuda_fattn_tile_new_is_supported(ctx.device, dst);
            const bool legal_mma  = new_mma_available(cc) && ggml_cuda_fattn_mma_new_can_run(dst);
            switch (fattn_force_decide(force_want, legal_vec, legal_tile, legal_mma)) {
                case FATTN_FORCE_VEC:
                    if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
                        ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                    } else {
                        ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                    }
                    return;
                case FATTN_FORCE_TILE: ggml_cuda_flash_attn_ext_tile_new(ctx, dst); return;
                case FATTN_FORCE_MMA:  ggml_cuda_flash_attn_ext_mma_new(ctx, dst);  return;
                default: break;
            }
        }
    }

    if (is_pascal_hs512(cc, K, V)) {
        ggml_cuda_flash_attn_ext_tile_new(ctx, dst);
        return;
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (is_pascal_mla_absorbed_decode(Q, K, V)) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            return;
        }
        // With GQA the vec kernel reads every K/V row once per Q head. The tile kernel packs 4 or 8 Q heads per K/V head.
        if (cc == GGML_CUDA_CC_PASCAL && Q->ne[1] == 1 && (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0) && V->type == K->type &&
            ggml_cuda_fattn_tile_new_is_supported(ctx.device, dst)) {
            ggml_cuda_flash_attn_ext_tile_new(ctx, dst);
            return;
        }
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                // P100 (sm_60/GGML_CUDA_CC_PASCAL): the fp16 vec kernel accumulates the online-softmax
                // denominator and the P*V product in fp16, flipping ~3-4% of decode top-1 tokens
                // vs fp32 (llama.cpp#25593). Decode is bandwidth-bound on P100 so vec_f32 is free.
                if (cc == GGML_CUDA_CC_PASCAL && Q->ne[1] <= 8) {  // decode only (batch<=8); D=256 prefill stays vec_f16
                    ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                }
            } else if (Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            } else if (ctx.fabsum) {
                ggml_cuda_flash_attn_ext_tile_f16_fabsum(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    int gqa_ratio = Q->ne[2] / K->ne[2];

    if (new_mma_available(cc) && K->ne[0] == 128 && V->ne[0] == 128 && Q->ne[0] == 128 && Q->ne[1] == 1 &&
            (Q->ne[2] / K->ne[2] == 12 || Q->ne[2] / K->ne[2] == 6 || Q->ne[2] / K->ne[2] == 10) &&
            ggml_cuda_fattn_mma_new_can_run(dst)) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    // CHERRY-PICK 555330fbb (upstream ikawrakow/ik_llama.cpp, not Thireus -- see
    // S69_ENGINE_THIREUS_2026-09-26.md): extends this head_size=256 branch from gqa_ratio==6 only
    // to also accept gqa_ratio==12, matching fattn-new-mma.cu's ggml_cuda_flash_attn_ext_mma_new
    // now handling gqa_ratio % 6 == 0. ggml_cuda_fattn_mma_new_can_run(dst) is OUR fork-only
    // capability guard (absent upstream) and is preserved here.
    if (new_mma_available(cc) && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[0] == 256 && Q->ne[1] == 1 &&
            (Q->ne[2] / K->ne[2] == 6 || Q->ne[2] / K->ne[2] == 12) &&
            ggml_cuda_fattn_mma_new_can_run(dst)) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for GGML_CUDA_CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < GGML_CUDA_CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        return;
    }

    //
    // It turns out the new new MMA implementation is slower than the
    // previous MMA implementation.
    // Hence, we use it only for DeepSeek with MLA enabled, where head sizes are 576, 512,
    // so no other implementation works.
    //

    if (new_mma_available(cc) &&
            ((K->ne[0] == 576 && V->ne[0] == 512) ||
             (K->ne[0] == 320 && V->ne[0] == 256) ||
             (K->ne[0] == 512 && V->ne[0] == 512) ||
             (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc))) &&
            ggml_cuda_fattn_mma_new_can_run(dst)) {
        //printf("Using ggml_cuda_flash_attn_ext_mma_new\n");
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    //
    // We need this because I haven't adapted new MMA kernels to work for different
    // K and V head sizes.
    // We also need it if the new MMA is not available
    //
    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
        return;
    }

    // As mentioned above, the new-new MMA is slower then the new MMA.
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
    //ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
}

// V is K (or a view of the same storage at the same offset). The runtime test in
// ggml_cuda_flash_attn_ext_mma_new_sparse is `V->data == K->data`; supports_op can run before the
// tensors are allocated (data == nullptr on both), so compare through view_src/view_offs as well.
static bool ggml_cuda_fattn_v_aliases_k(const ggml_tensor * K, const ggml_tensor * V) {
    if (V == K) return true;
    if (K->data && V->data) return K->data == V->data;
    const ggml_tensor * k_root = K->view_src ? K->view_src : K;
    const ggml_tensor * v_root = V->view_src ? V->view_src : V;
    return k_root == v_root && (K->view_src ? K->view_offs : 0) == (V->view_src ? V->view_offs : 0);
}

// Mirror of the accept conditions of ggml_cuda_flash_attn_ext_mma_new_sparse (fattn-new-mma.cu) -- the
// kernel that ggml_cuda_flash_attn_ext tries FIRST when src[5] (kv_idx) is set. Keep the two in step.
static bool ggml_cuda_fattn_sparse_new_mma_would_run(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * kv_idx = dst->src[5];
    if (!new_mma_available(cc) || Q->ne[1] < 8) return false;
    if (Q->ne[0] == 256 && V->ne[0] == 256) return true;
    if (Q->ne[0] != 512 || V->ne[0] != 512 || !ggml_cuda_fattn_v_aliases_k(K, V)) return false;
    if (ggml_is_quantized(K->type) && 2*Q->ne[1]*kv_idx->ne[0] < K->ne[1]) return false;
    return true;
}

bool ggml_cuda_fattn_is_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // ggml_flash_attn_ext_rows (ggml-org 1b6935b22, per-slice K/V row list): none of this fork's CUDA FA kernels reads
    // it -- they take src[5] as the ik per-query/indexer list -- so answer false and let the scheduler run the node
    // elsewhere. llama_context::resolve_fused_ops probes the op per backend, so on CUDA the graph keeps the dense
    // unified-cache mask (the pre-port behaviour) until a CUDA kernel exists (row S69-FAROWS-CUDA-KERNEL).
    if (ggml_flash_attn_ext_get_kv_rows(dst) != nullptr) {
        GGML_UNUSED(device);
        return false;
    }

    // kv_idx (src[5]) WITHOUT a mask: the DeepSeek-V4 CSA/HCA prefill form (ik #2779). The dispatch in
    // ggml_cuda_flash_attn_ext tries the sparse new-MMA kernel first; if it declines it builds a dense F16
    // mask [K->ne[1], rows] from the list (k_index_to_mask) and continues as if the caller had passed that
    // mask. This function used to ask the question about the maskless dst, and every shape check below
    // that needs a mask (ggml_cuda_fattn_mma_new_can_run: `!mask` -> false) declined it, so hs=512 V=K
    // lists never reached the sparse kernel on CUDA (test-backend-ops FLASH_ATTN_EXT kv_idx cases,
    // MEASURED 2026-11-13 at c654c5bdb, row S69-DRIFT-1110-DSV4-DENSE-MASK-DROP-CHAIN-2026-11-10-01).
    // Answer for the path that will actually run: sparse if it takes the shape, else the dense fallback.
    if (dst->src[5] && !mask) {
        const int cc_s = ggml_cuda_info().devices[device].cc;
        if (ggml_cuda_fattn_sparse_new_mma_would_run(cc_s, dst)) {
            return true;
        }
        ggml_tensor Mi = *dst->src[5];
        Mi.type     = GGML_TYPE_F16;
        Mi.ne[0]    = K->ne[1];
        Mi.nb[0]    = sizeof(half);
        Mi.nb[1]    = Mi.ne[0]*Mi.nb[0];
        Mi.nb[2]    = Mi.ne[1]*Mi.nb[1];
        Mi.nb[3]    = Mi.ne[2]*Mi.nb[2];
        Mi.data     = nullptr;
        Mi.buffer   = nullptr;
        Mi.view_src = nullptr;
        Mi.view_offs = 0;
        ggml_tensor dense = *dst;
        dense.src[3] = &Mi;
        return ggml_cuda_fattn_is_supported(device, &dense);
    }

    // UNIVERSAL PRECONDITION, checked before any variant is chosen.
    //
    // Every flash-attention variant asserts this same condition, in three separate places:
    //     fattn-common.cuh:764, fattn-mma-f16.cuh:1291, fattn-vec-common.cuh:885
    //         GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.")
    // so it is not a property of one kernel but a precondition of the whole family. Asserting it
    // inside the kernel KILLS THE PROCESS on a shape ggml could simply have routed around;
    // reporting it here makes supports_op tell the truth, and ggml falls back to the non-FA
    // attention path -- slower, correct, alive.
    //
    // MEASURED 2026-09-01: llama-audio ASR aborted at fattn-mma-f16.cuh:1291 while encoding audio,
    // exit 3221226505 (0xC0000409). The audio encoder's K length is 750 and FATTN_KQ_STRIDE is
    // 256; 750 % 256 = 238. The SAME binary with CUDA hidden transcribed the clip exactly
    // ("Concord returned to its place amidst the tents." against a LibriSpeech ground truth of
    // "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"), so nothing was wrong but the routing.
    //
    // THIS IS THE SECOND INSTANCE OF THE CLASS IN THIS FORK. _build_consolidated.bat already
    // records the first: "supports_op returning true for GGML_OP_DELTA_NET whose KDA kernel covers
    // only head_dim 64/128 (0xC0000409)" -- a capability check that answered a question the kernel
    // could not honour. check_capability_proxy.py exists because of that one.
    if (K != nullptr && (K->ne[1] % FATTN_KQ_STRIDE) != 0) {
        return false;
    }

    // Was devices[ggml_cuda_get_device()].cc -- whichever device happened to be current when the
    // scheduler asked. It is the device being ASKED ABOUT now, which is the question supports_op
    // is actually posing and differs on a multi-GPU host with unlike cards.
    const int cc = ggml_cuda_info().devices[device].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[GGML_FLASH_ATTN_EXT_N_SWA_SLOT];

    // TurboQuant KV: answer for the kernel that will actually run it (the vec one -- see the
    // matching branch in ggml_cuda_flash_attn_ext), not for whichever kernel the generic logic below
    // would have picked. Answering for a different kernel than the one dispatched is exactly the
    // capability-proxy class this file's long comment above documents twice over.
    if (turbo_fattn_type_is_turbo(K->type) || turbo_fattn_type_is_turbo(V->type)) {
        return precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)
                   ? ggml_cuda_fattn_vec_f16_is_supported(device, dst)
                   : ggml_cuda_fattn_vec_f32_is_supported(device, dst);
    }

    if (cc >= GGML_CUDA_CC_OFFSET_AMD) {
        return precision == GGML_PREC_DEFAULT ? ggml_cuda_fattn_vec_f16_is_supported(device, dst)
                                              : ggml_cuda_fattn_vec_f32_is_supported(device, dst);
    }

    if (is_pascal_hs512(cc, K, V)) {
        return ggml_cuda_fattn_tile_new_is_supported(device, dst);
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            return ggml_cuda_fattn_vec_f32_is_supported(device, dst);
        } else {
            return ggml_cuda_fattn_tile_f32_is_supported(device, dst);
        }
    }

    if (!fp16_mma_available(cc)) {
        if (is_pascal_mla_absorbed_decode(Q, K, V)) {
            return ggml_cuda_fattn_vec_f32_is_supported(device, dst);
        }
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                if (cc == GGML_CUDA_CC_PASCAL && Q->ne[1] <= 8) {  // decode only (batch<=8); D=256 prefill stays vec_f16
                    return ggml_cuda_fattn_vec_f32_is_supported(device, dst);
                }
                return ggml_cuda_fattn_vec_f16_is_supported(device, dst);
            } else {
                return ggml_cuda_fattn_tile_f16_is_supported(device, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                return ggml_cuda_fattn_vec_f32_is_supported(device, dst);
            } else {
                return ggml_cuda_fattn_tile_f32_is_supported(device, dst);
            }
        }
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for GGML_CUDA_CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < GGML_CUDA_CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        return ggml_cuda_fattn_vec_f32_is_supported(device, dst);
    }

    // Same shape test as the dispatch above, then the launcher's own preconditions. A 192/128 case
    // mma_new cannot take falls through to wmma there, so it does here; the others have no other
    // kernel and are declined, which lets the scheduler run them on the CPU instead of aborting.
    if (new_mma_available(cc) &&
            ((K->ne[0] == 576 && V->ne[0] == 512) ||
             (K->ne[0] == 320 && V->ne[0] == 256) ||
             (K->ne[0] == 512 && V->ne[0] == 512) ||
             (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        if (ggml_cuda_fattn_mma_new_can_run(dst)) return true;
        if (K->ne[0] == 192 && V->ne[0] == 128) return ggml_cuda_fattn_wmma_f16_is_supported(device, dst);
        return false;
    }

    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        return ggml_cuda_fattn_wmma_f16_is_supported(device, dst);
    }

    return ggml_cuda_fattn_mma_f16_is_supported(device, dst);
}
