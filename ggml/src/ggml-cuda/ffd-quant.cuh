#pragma once

#include "common.cuh"

// GGML_OP_FFD_QUANT_K on CUDA -- same contract as the CPU reference (ggml.h, ggml_ffd_quant_k).
void ggml_cuda_op_ffd_quant_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_ffd_quant_k_supported(const ggml_tensor * op);
