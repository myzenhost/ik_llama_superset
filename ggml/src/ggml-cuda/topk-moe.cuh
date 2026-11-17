#pragma once

#include "common.cuh"
#include "ggml.h"

// Rows that one CUDA block handles (upstream 977c48a25 topk-moe.cuh; the fork's pre-merge value was 4).
#define TOPK_MOE_ROWS_PER_BLOCK 8

// What the fused router computes, filled by ggml_cuda_topk_moe_fusion() in ggml-cuda.cu
// (port of upstream 977c48a25 ggml-cuda/ggml-cuda.cu:3197-3358, same field names).
struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
    int  n_groups{};     // > 0: pick experts only from the top n_group_used expert groups
    int  n_group_used{};
};

// Fork's original entry (ik #789/#983/#984/#1170 graphs: SOFT_MAX [-> ADD bias] -> ARGSORT -> VIEW -> GET_ROWS [-> RESHAPE -> SUM_ROWS -> DIV]).
// `logits` is the SOFT_MAX node (the kernel reads logits->src[0]); `weights` is the last fused node; DIV means
// normalise by the selected sum (no clamp). The bias is the selection bias: experts are CHOSEN by softmax+bias,
// the weights are the unbiased softmax.
void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               top_k,
                           ggml_tensor *               bias = nullptr);

// Upstream form. `logits` is the gating op's src[0].
void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);
