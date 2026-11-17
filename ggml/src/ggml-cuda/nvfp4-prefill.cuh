#pragma once

#include "common.cuh"

// Native block-scaled FP4 (NVFP4 W4A4) prefill GEMM for Blackwell, through cuBLASLt.
// Row S69-FASTLLM-PICK-SM120-NATIVE-NVFP4-FP8-PREFILL-2026-10-05-01.
//
// WHAT. For a MUL_MAT whose src0 is NVFP4 and whose src1 has at least GGML_CUDA_NVFP4_PREFILL_MIN_M rows
// (default 128, i.e. prompt processing, not decode), quantise the F32 activations to NVFP4 (E2M1 values, one
// UE4M3 scale per 16, one F32 global scale per row), repack the weights into the layout cuBLASLt wants, run
// cublasLtMatmul with the VEC16_UE4M3 block-scale mode on the FP4 tensor cores, then apply the row scale.
//
// DEFAULT OFF. Activation quantisation is lossy by construction (E2M1 has one mantissa bit). Enable with
// GGML_CUDA_NVFP4_PREFILL=1. Every precondition that fails returns false and the caller takes the ordinary
// dequantise + cuBLAS path, so the flag can only change a result by the activation rounding it names.
//
// The per-TENSOR scale of an NVFP4 weight is NOT applied here: it is a graph MUL after the mul_mat, exactly as for
// every other NVFP4 path in this backend (ggml-common.h:255-272, convert.cu dequantize_block_nvfp4).
//
// Design source: ztxz16/fastllm (Apache-2.0) docs/cuda-native-prefill.md and
// src/devices/cuda/linear/fastllm-native-lowbit-prefill.cu, read in full; the code here is written from that
// description and from the cuBLASLt documentation, no code copied.
bool ggml_cuda_mul_mat_nvfp4_prefill(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Persistent repacked-weight cache (S69-FASTLLM-PICK-NVFP4-PREFILL-REPACK-CACHE-AUTOTUNE-2026-10-27-01). The repack of a
// weight into the cuBLASLt operand layout depends only on the weight, so the first prefill call for a tensor held in a
// WEIGHTS buffer keeps the repacked copy (same size as the weight) and later calls reuse it. Bounded per device by
// GGML_CUDA_NVFP4_PREFILL_CACHE_MB (default 2048, 0 = off); a weight that does not fit repacks per call as before.
// The CUDA buffer interface calls this wherever the bytes behind [base, base+size) can change or disappear (free, set,
// memset, copy-into, clear); every cached entry overlapping the range is dropped.
void ggml_cuda_nvfp4_prefill_cache_invalidate(const void * base, size_t size);
