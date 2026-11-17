#pragma once

#include "common.hpp"

// GGML_OP_TURBO_WHT: Walsh-Hadamard rotation of the TurboQuant KV basis, 128-element groups, contiguous F32 in and out.
void ggml_sycl_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// The shapes ggml_sycl_turbo_wht answers for (group_size 128, contiguous F32). supports_op uses this so a graph the kernel cannot
// honour is declined instead of aborting mid-request.
bool ggml_sycl_turbo_wht_supported(const ggml_tensor * op);

// Backend-internal rotations used by the flash-attention wrapper (fattn.cpp), the SYCL counterpart of fattn.cu's
// ggml_cuda_fattn_turbo_via_mma path and of the Vulkan wrapper:
//   forward_to       rotate the (possibly permuted) F32 tensor q forward, in 128-element groups along dim 0, into the contiguous
//                    scratch dst (ggml_nelements(q) floats). q->ne[0] must be a multiple of 128.
//   inverse_inplace  apply the inverse rotation in place to the contiguous F32 tensor t (a flash-attention output).
void ggml_sycl_turbo_rotate_forward_to(ggml_backend_sycl_context & ctx, const ggml_tensor * q, float * dst);
void ggml_sycl_turbo_rotate_inverse_inplace(ggml_backend_sycl_context & ctx, ggml_tensor * t);
