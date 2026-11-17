#pragma once

#include "common.cuh"

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Backend-internal rotations for attention kernels that read turbo K/V as f16 (tensor-core path, fattn.cu).
// Rotate the (possibly permuted) F32 tensor q forward, in 128-element groups along dim 0, into the contiguous
// scratch `dst` (ggml_nelements(q) floats). q->ne[0] must be a multiple of 128.
void ggml_cuda_turbo_rotate_forward_to(ggml_backend_cuda_context & ctx, const ggml_tensor * q, float * dst);
// Apply the INVERSE rotation in place to the contiguous F32 tensor t (a flash-attention output).
void ggml_cuda_turbo_rotate_inverse_inplace(ggml_backend_cuda_context & ctx, ggml_tensor * t);
