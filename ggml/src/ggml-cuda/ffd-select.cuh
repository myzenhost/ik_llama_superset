#pragma once

#include "common.cuh"

// GGML_OP_FFD_SELECT on CUDA -- same contract as the CPU reference (ggml.h, ggml_ffd_select).
void ggml_cuda_op_ffd_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_ffd_select_supported(const ggml_tensor * op);
