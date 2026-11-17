#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cfloat>
#include <cmath>
#include <initializer_list>

// Kernel body, gating variants, group mask, NaN guard and the (value, lowest expert index) tie-break are the
// kernel of upstream 977c48a25 ggml-cuda/topk-moe.cu (PRs #27621, #29181, #30441, DSV4.1 #31086), merged here by hand
// (row S69-UPMERGE-TOPK-MOE-CU-UPSTREAM-KERNEL-DROPPED-GATING-NONPOW2-TIEBREAK-2026-11-06-01). The merge of
// 2026-11-03 kept the fork's older kernel whole, which lost: the tie-break (the fork's butterfly used strict `>`, so on an
// exact tie two lanes each kept their own candidate and BOTH wrote weights[k]/ids[k] and set their expert to -inf),
// the NaN guard, non-power-of-2 expert counts (288, 576), sigmoid / sqrt-softplus / delayed-softmax gating, DeepSeek
// expert groups and the clamp/scale epilogue. The fork keeps what upstream never had: the legacy entry
// ggml_cuda_op_topk_moe(ctx, softmax, weights, ids, bias) used by the ik-style positional fusion in ggml-cuda.cu
// (fused experts bias ik #1170, fused sum_rows+div ik #984), now on this one kernel.
// Dropped from upstream on purpose: PDL (ggml_cuda_pdl_sync / ggml_cuda_kernel_launch) and the MUSA row clamp -- the
// fork launches with <<<>>> and builds no MUSA.

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    int  n_groups;
    int  n_group_used;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : log1pf(expf(vals[i]))) : -INFINITY;
    }
}

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
// Set -inf for experts outside the top n_group_used groups. Group score is the sum of its top-2 experts (DeepSeek V3).
template <int n_experts, int experts_per_thread>
__device__ void group_mask_warp_inplace(float (&vals)[experts_per_thread], const int n_groups, const int n_group_used, const int lane) {
    const int n_per_group = n_experts / n_groups;

    float score = -INFINITY; // lane g holds the score of group g
    for (int g = 0; g < n_groups; g++) {
        float top1 = -INFINITY;
        float top2 = -INFINITY;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if ((i * WARP_SIZE + lane) / n_per_group == g) {
                top2 = fmaxf(top2, fminf(top1, vals[i]));
                top1 = fmaxf(top1, vals[i]);
            }
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            const float o1 = __shfl_xor_sync(0xFFFFFFFF, top1, offset, WARP_SIZE);
            const float o2 = __shfl_xor_sync(0xFFFFFFFF, top2, offset, WARP_SIZE);
            top2 = fmaxf(fminf(top1, o1), fmaxf(top2, o2));
            top1 = fmaxf(top1, o1);
        }
        if (lane == g) {
            score = top1 + top2;
        }
    }

    // rank of each group, lower index wins ties
    int rank = 0;
    for (int g = 0; g < n_groups; g++) {
        const float other = __shfl_sync(0xFFFFFFFF, score, g, WARP_SIZE);
        rank += other > score || (other == score && g < lane);
    }
    const auto keep = __ballot_sync(0xFFFFFFFFULL, lane < n_groups && rank < n_group_used);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int g = (i * WARP_SIZE + lane) / n_per_group;
        if (g >= n_groups || !((keep >> g) & 1)) {
            vals[i] = -INFINITY;
        }
    }
}
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              const float *         bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
            if (__isnanf(selection_wt[i / WARP_SIZE])) {
                selection_wt[i / WARP_SIZE] = -FLT_MAX;
            }
        }
    }

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
    // without bias, masking wt is safe: the matcher ensures the kept groups hold all used experts
    if (config.n_groups > 0) {
        if constexpr (has_bias) {
            group_mask_warp_inplace<n_experts>(selection_wt, config.n_groups, config.n_group_used, threadIdx.x);
        } else {
            group_mask_warp_inplace<n_experts>(wt, config.n_groups, config.n_group_used, threadIdx.x);
        }
    }
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

// Expert counts the kernel is instantiated for: powers of 2 up to 512 plus the non-power-of-2 counts of supported
// models (288 StepFun 3.7, 576). Every n > 32 must be a multiple of WARP_SIZE (experts_per_thread = n / 32).
static bool topk_moe_n_expert_supported(const int64_t n_expert) {
    if (n_expert <= 0) {
        return false;
    }
    if ((n_expert & (n_expert - 1)) == 0) {
        return n_expert <= 512;
    }
    return n_expert == 288 || n_expert == 576;
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 const float *               bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();

#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_ASSERT(config.n_groups == 0 && "expert groups fusion is not supported on this HIP version");
#endif

#define TOPK_MOE_CASE(N) \
    case N: \
        topk_moe_cuda<N, has_bias><<<grid_dims, block_dims, 0, stream>>>( \
            logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config); \
        break;

    switch (n_expert) {
        TOPK_MOE_CASE(1)
        TOPK_MOE_CASE(2)
        TOPK_MOE_CASE(4)
        TOPK_MOE_CASE(8)
        TOPK_MOE_CASE(16)
        TOPK_MOE_CASE(32)
        TOPK_MOE_CASE(64)
        TOPK_MOE_CASE(128)
        TOPK_MOE_CASE(256)
        TOPK_MOE_CASE(288) // StepFun 3.7
        TOPK_MOE_CASE(512)
        TOPK_MOE_CASE(576)
        default:
            GGML_ASSERT(false && "fatal error");
    }
#undef TOPK_MOE_CASE
}

static void topk_moe_run(ggml_backend_cuda_context & ctx,
                         const ggml_tensor *         logits,
                         ggml_tensor *               weights,
                         ggml_tensor *               ids,
                         const ggml_tensor *         bias,
                         const int                   n_expert_used,
                         const bool                  with_norm,
                         const float                 clamp_val,
                         const float                 scale_val,
                         const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    if (bias) {
        GGML_ASSERT(logits->ne[0] == bias->ne[0] && ggml_nrows(bias) == 1 && bias->type == GGML_TYPE_F32);
    }

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    const float * bias_d    = bias ? (const float *) bias->data : nullptr;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                                   scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                                    scale_val, config);
    }
}

// Upstream form (977c48a25).
void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;
    const int   n_expert_used = weights->ne[1];
    const bool  with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_run(ctx, logits, weights, ids, bias, n_expert_used, with_norm, clamp_val, scale_val, args);
}

// Fork's legacy entry: `softmax` is the SOFT_MAX node, `weights` the last fused node (DIV when the selected sum
// normalisation is fused, no clamp), `bias` the selection bias (choose by softmax+bias, weight by softmax).
void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         softmax,
                           ggml_tensor *               weights,
                           ggml_tensor *               ids,
                           ggml_tensor *               bias) {
    ggml_cuda_topk_moe_args args;
    args.softmax = true;

    if (weights->op == GGML_OP_DIV) {
        topk_moe_run(ctx, softmax->src[0], weights, ids, bias, (int) weights->ne[0], /*with_norm=*/true,
                     /*clamp_val=*/-INFINITY, 1.0f, args);
    } else {
        topk_moe_run(ctx, softmax->src[0], weights, ids, bias, (int) weights->ne[1], /*with_norm=*/false,
                     /*clamp_val=*/-INFINITY, 1.0f, args);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (!topk_moe_n_expert_supported(n_expert)) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}

// Fork's legacy check (ik-style graphs: SOFT_MAX node + the last fused node).
bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights) {
    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        return false;
    }

    if (scale != 1.0f || max_bias != 0.0f) {
        return false;
    }

    // don't fuse when masks or sinks are present
    if (softmax->src[1] || softmax->src[2]) {
        return false;
    }

    return topk_moe_n_expert_supported(softmax->ne[0]);
}
