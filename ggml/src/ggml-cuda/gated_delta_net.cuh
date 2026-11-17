#pragma once
// ADDED 2026-09-08. This header DEFINES types, so a second inclusion in one translation
// unit is a hard error -- nvcc proved it when new mmf.cuh / mmq-vec-dot.cuh paths reached
// it alongside mmq.cuh. Most .cuh here are guard-less and that is FINE: 90 of 122 have no
// guard, and re-including a declaration is legal. Only 2 of those 90 define types; this is
// one of them. Measured, not assumed.
#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);
