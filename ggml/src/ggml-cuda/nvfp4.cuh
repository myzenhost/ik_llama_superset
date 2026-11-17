#pragma once

#include "common.cuh"

// Native Blackwell (sm_120a / sm_121a) W4A4 NVFP4 matmul on the mma.sync block-scaled FP4 tensor-core instruction.
//
// SOURCE: ik_llama.cpp b71fe5368 (#2868) + c7ea52b33 (#2880), ggml/src/ggml-cuda/nvfp4.cu/.cuh, read in full (MIT, same
// licence as this tree). Adapted to this fork, where the weights are GGML_TYPE_NVFP4 (id 40, mainline layout, no row
// header) and the row scale ik keeps in a 4-byte row header is the constant 1.0f (the per-TENSOR weight scale of an NVFP4
// weight is a graph MUL after the mul_mat, as for every other NVFP4 path here). The kernels take the weight row layout as
// a runtime { row_bytes, header } pair, so GGML_TYPE_NVFP4_KS (header 4, input scale from the checkpoint) is one predicate
// away once that type lands (row S69-DRIFT-1110-W4A4-BLACKWELL-NVFP4-MXFP4, deferred-on-prereq NVFP4_KS).
//
// OPT-IN: GGML_CUDA_NVFP4_PREFILL=native (aliases: mma, w4a4). GGML_CUDA_NVFP4_PREFILL=1 keeps selecting the cuBLASLt
// prefill GEMM in nvfp4-prefill.cu; 0/unset selects neither. The activation is quantised to E2M1 with a per-row dynamic
// global scale, which is lossy by construction, so this is never on by default for a type-40 weight.
//
// Cases covered here that the cuBLASLt path cannot do: MUL_MAT_ID (MoE, per-expert tiles over the expert-sorted row list),
// fused up/gate + SiLU / SWIGLU_OAI (+ expert biases) epilogue, no repacked copy of the weight (the kernel reads the ggml
// block layout in place), and activation quantisation shared by consecutive MUL_MATs that read the same src1.

bool ggml_cuda_nvfp4_available(int cc);          // compiled for this compute capability (needs -DCMAKE_CUDA_ARCHITECTURES ...120a)
bool ggml_cuda_nvfp4_native_selected();          // GGML_CUDA_NVFP4_PREFILL=native|mma|w4a4, read per call
// Defined in ggml-cuda.cu (the buffer-type predicates are static there): the tensor lives in an ordinary CUDA device buffer,
// not a cuda-split one, so its rows are at nb[1] strides on this device.
bool ggml_cuda_tensor_on_plain_cuda_buffer(const ggml_tensor * t);

// src0 is a weight this path can read, src1 an F32 activation. The _id form accepts a 3-D (per-expert) src0.
bool ggml_cuda_can_mul_mat_nvfp4_id(const ggml_tensor * src0, const ggml_tensor * src1, int cc);
bool ggml_cuda_can_mul_mat_nvfp4(const ggml_tensor * src0, const ggml_tensor * src1, int cc);
bool ggml_cuda_nvfp4_same_input_scale(const ggml_tensor * a, const ggml_tensor * b);

// Dense MUL_MAT. Also runs the MUL_MATs that directly follow in `cgraph` and read the same src1 with a compatible
// weight (one activation quantisation for q/k/v or up/gate); returns the index of the last node it consumed.
int ggml_cuda_mul_mat_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const ggml_cgraph * cgraph, int node_n);

// Dense fused up/gate + SiLU (GGML_UNARY_OP_SILU, optional clamp `limit`), dst = silu(gate*x)*(up*x), one kernel.
bool ggml_cuda_can_up_gate_nvfp4(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_up_gate_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, ggml_tensor * dst, float limit);

// Activation quantised once (E2M1 values, one UE4M3 scale per 16, one F32 scale per row), reused by several GEMMs.
struct nvfp4_input {
    int k, n;
    ggml_cuda_pool_alloc<uint8_t> q, s;
    ggml_cuda_pool_alloc<float> r;

    nvfp4_input(ggml_backend_cuda_context & ctx);
    nvfp4_input(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float scale);
    void quantize(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float scale);
    void mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, float * dst);
};

// Activation scale to pass to nvfp4_input for this weight: < 0 means "dynamic, per row" (amax/(6*448)).
float ggml_cuda_nvfp4_input_scale(const ggml_tensor * src0);

// Expert-indexed GEMM over ids_src/ids_dst/bounds as produced by compute_row_ids (mmq_id.cuh). With `fused`, dst receives
// glu(up, gate) of one row per (token, expert); `gate` null means src0 holds [gate rows; up rows] (merged up_gate_exps).
// alpha > 0 selects SWIGLU_OAI (limit is its clamp), alpha == 0 the SiLU form (limit < 1e-6 means unclamped).
void ggml_cuda_gemm_nvfp4_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const int32_t * ids_src, const int32_t * ids_dst, const int32_t * bounds,
        const nvfp4_input & input, bool fused = false, const ggml_tensor * gate = nullptr, float alpha = 0.0f, float limit = 0.0f);

// "Was it FIRED" counters (a path that is compiled and selected but never reached looks identical to one that works).
// ggml-cuda.h: ggml_backend_cuda_nvfp4_native_stats.
