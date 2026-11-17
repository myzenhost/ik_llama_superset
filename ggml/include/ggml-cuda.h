#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// The backend registry entry point. ggml/src/ggml-backend-reg.cpp:121 calls this under
// GGML_USE_CUDA and includes only this header, so without the line below that file fails with
// `error C3861: 'ggml_backend_cuda_reg': identifier not found` -- measured, not inferred.
//
// GGML_BACKEND_API, NOT the GGML_API used by every other declaration in this file, and the
// difference is load-bearing rather than stylistic. The two macros key off DIFFERENT defines,
// and this symbol crosses a library boundary that the others do not: it is DEFINED in
// ggml-cuda.dll and CALLED from ggml.dll. Measured from the generated build graph:
//
//   translation unit          GGML_API expands to    GGML_BACKEND_API expands to
//   ggml-cuda.cu              dllimport  (wrong --   dllexport  (correct -- it defines it)
//     (GGML_BACKEND_BUILD,     on the TU that
//      no GGML_BUILD)          defines it)
//   ggml-backend-reg.cpp      dllexport  (wrong --   dllimport  (correct -- it imports it)
//     (GGML_BUILD, no          it is the caller)
//      GGML_BACKEND_BUILD)
//
// So GGML_API would be wrong in BOTH directions, which is why ggml-vulkan.h:27 and
// ggml-cpu.h:152 declare their own reg entry points with GGML_BACKEND_API too. Declaring it
// here also retires the local forward declaration ggml-cuda.cu carried near its top; keeping
// both would be a dllimport/dllexport collision (C2375) in one translation unit.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// backend API
GGML_BACKEND_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params, const void * model);

GGML_BACKEND_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
GGML_BACKEND_API GGML_CALL bool ggml_backend_cuda_is_integrated(int device);

GGML_BACKEND_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API GGML_CALL void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

GGML_BACKEND_API GGML_CALL void ggml_backend_cuda_invalidate_graphs(const void * model);

// S69-FASTLLM-PICK-SM120-NATIVE-NVFP4-FP8-PREFILL: how often the native FP4 prefill GEMM ran (handled) and how often it was
// enabled and prefill-sized but refused (declined, the caller then used dequantize + cuBLAS). Both stay 0 with the flag off.
GGML_BACKEND_API GGML_CALL void ggml_backend_cuda_nvfp4_prefill_stats(unsigned long long * handled, unsigned long long * declined);
// S69-FASTLLM-PICK-NVFP4-PREFILL-REPACK-CACHE-AUTOTUNE-2026-10-27-01: repacked-weight cache hits, weight repacks performed
// (cache fills plus per-call repacks), and cuBLASLt plans whose algorithm was picked by timing. All 0 with the prefill flag off.
GGML_BACKEND_API GGML_CALL void ggml_backend_cuda_nvfp4_prefill_cache_stats(unsigned long long * cache_hits, unsigned long long * repacks, unsigned long long * tuned_plans);
// S69-DRIFT-1110-W4A4-BLACKWELL (ik b71fe5368): fire counters of the native mma.sync W4A4 NVFP4 kernels (GGML_CUDA_NVFP4_PREFILL=native):
// dense GEMMs, expert-indexed (MUL_MAT_ID) GEMMs, fused up/gate GLU kernels. All 0 with the flag off or a build without sm_120a.
GGML_BACKEND_API GGML_CALL void ggml_backend_cuda_nvfp4_native_stats(unsigned long long * dense, unsigned long long * id, unsigned long long * glu);

// S69 DiffusionGemma stage-1 dense device sampler. Runs argmax + entropy + a multinomial draw over
// per-position canvas logits held in a DEVICE tensor, so a denoising step never downloads the full
// [n_vocab x canvas] logits. Returns false (caller falls back to the host path) for a host tensor,
// a non-F32/non-contiguous tensor, or a build without CUDA. Called by llama_diffusion_device_sample.
GGML_BACKEND_API GGML_CALL bool ggml_backend_cuda_diffusion_sample(
        struct ggml_tensor * logits,
        const float        * u_host,
        int                * argmax_host,
        float              * entropy_host,
        int                * sampled_host,
        int                  n_tokens,
        float                inv_temp);

// Encode IQ*_KT / IQ*_KS rows on a CUDA device (llama-quantize --cuda-quantize). Thireus d9e286846 / ik #2682, MIT.
GGML_BACKEND_API GGML_CALL size_t ggml_cuda_quantize(int device, enum ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice, const float * imatrix);
#ifdef  __cplusplus
}
#endif
