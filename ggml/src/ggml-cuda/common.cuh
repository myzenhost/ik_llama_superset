#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>

#if defined(GGML_USE_HIP)
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#else
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#if defined(GGML_USE_MUSA)
#define GGML_COMMON_DECL_MUSA
#define GGML_COMMON_IMPL_MUSA
#endif
#endif
#include "ggml-common.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#elif defined(GGML_USE_MUSA)
#include "vendors/musa.h"
#else
#include "vendors/cuda.h"
#endif // defined(GGML_USE_HIP)

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define CUDART_HMAX   11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)
#define CUDART_HMASK  12000 // CUDA 12.0, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_PASCAL     600
#define MIN_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define GGML_CUDA_CC_VOLTA      700
#define GGML_CUDA_CC_TURING     750
#define GGML_CUDA_CC_AMPERE     800
#define GGML_CUDA_CC_AMPERE_GA10X 860 // upstream 977c48a25 common.cuh:55; gated-delta-net-mma.cu:404 keys its smem budget on it
#define GGML_CUDA_CC_ADA_LOVELACE 890
#define GGML_CUDA_CC_HOPPER     900
// While BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core instructions available to 1200 family, see
// https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms
#define GGML_CUDA_CC_BLACKWELL  1200
#define GGML_CUDA_CC_DGX_SPARK  1210
#define GGML_CUDA_CC_RUBIN      1300
#define GGML_CUDA_CC_OFFSET_AMD 1000000
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_RDNA1      (GGML_CUDA_CC_OFFSET_AMD + 1010)
#define GGML_CUDA_CC_RDNA2      (GGML_CUDA_CC_OFFSET_AMD + 1030)
#define GGML_CUDA_CC_RDNA3      (GGML_CUDA_CC_OFFSET_AMD + 1100)
#define GGML_CUDA_CC_RDNA3_5    (GGML_CUDA_CC_OFFSET_AMD + 1150) // AI 370, AI Max 395 laptops.
#define GGML_CUDA_CC_RDNA4      (GGML_CUDA_CC_OFFSET_AMD + 1200) // RX 9000
// GGML_CUDA_CC_OFFSET_AMD (1000000) is BELOW GGML_CUDA_CC_OFFSET_MTHREADS (0x100000 = 1048576), so testing only
// against the MTHREADS offset classified every AMD cc in [1000000, 1048576) as NVIDIA. An NVIDIA cc is 100*major + 10*minor
// and never comes near either offset. Fix from ik_llama #2339 (its spelling: cc < CC_OFFSET_AMD).
#define GGML_CUDA_CC_IS_NVIDIA(cc)   (cc < GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_AMD(cc)   (cc >= GGML_CUDA_CC_OFFSET_AMD)

// AMD GCN/CDNA, wave size is 64
#define GGML_CUDA_CC_GCN4       (GGML_CUDA_CC_OFFSET_AMD + 0x803)  // Tonga, Fiji, Polaris, minimum for fast fp16
#define GGML_CUDA_CC_VEGA       (GGML_CUDA_CC_OFFSET_AMD + 0x900)  // Vega56/64, minimum for fp16 dual issue
#define GGML_CUDA_CC_VEGA20     (GGML_CUDA_CC_OFFSET_AMD + 0x906)  // MI50/Radeon VII, minimum for dp4a
#define GGML_CUDA_CC_CDNA1      (GGML_CUDA_CC_OFFSET_AMD + 0x908)  // MI100, minimum for MFMA, acc registers
#define GGML_CUDA_CC_CDNA2      (GGML_CUDA_CC_OFFSET_AMD + 0x90a)  // MI210 (gfx90a), minimum acc register renaming
#define GGML_CUDA_CC_CDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x942)  // MI300
#define GGML_CUDA_CC_CDNA4      (GGML_CUDA_CC_OFFSET_AMD + 0x950)  // MI350X/MI355X

#define GGML_CUDA_CC_IS_RDNA(cc)    (cc >= GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_RDNA1(cc)   (cc >= GGML_CUDA_CC_RDNA1 && cc < GGML_CUDA_CC_RDNA2)
#define GGML_CUDA_CC_IS_RDNA2(cc)   (cc >= GGML_CUDA_CC_RDNA2 && cc < GGML_CUDA_CC_RDNA3)
#define GGML_CUDA_CC_IS_RDNA3_0(cc) (cc >= GGML_CUDA_CC_RDNA3 && cc < GGML_CUDA_CC_RDNA3_5)
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_RDNA3(cc)   (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc))
#define GGML_CUDA_CC_IS_RDNA4(cc)   (cc >= GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_GCN(cc)     (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_CDNA1)
#define GGML_CUDA_CC_IS_CDNA(cc)    (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_RDNA1)
#define GGML_CUDA_CC_IS_CDNA1(cc)   (cc >= GGML_CUDA_CC_CDNA1 && cc < GGML_CUDA_CC_CDNA2)
#define GGML_CUDA_CC_IS_CDNA2(cc)   (cc >= GGML_CUDA_CC_CDNA2 && cc < GGML_CUDA_CC_CDNA3)
#define GGML_CUDA_CC_IS_CDNA3(cc)   (cc >= GGML_CUDA_CC_CDNA3 && cc < GGML_CUDA_CC_CDNA4)
#define GGML_CUDA_CC_IS_CDNA4(cc)   (cc >= GGML_CUDA_CC_CDNA4 && cc < GGML_CUDA_CC_RDNA1)

// Moore Threads
#define MUSART_HMASK 40300 // MUSA rc4.3, min. ver. for half2 -> uint mask comparisons

#define GGML_CUDA_CC_QY1 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x210) // MTT S80, MTT S3000
#define GGML_CUDA_CC_QY2 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x220) // MTT S4000
#define GGML_CUDA_CC_PH1 (GGML_CUDA_CC_OFFSET_MTHREADS + 0x310) // MTT S5000

#define GGML_CUDA_CC_IS_MTHREADS(cc) (cc >= GGML_CUDA_CC_OFFSET_MTHREADS && cc < GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_QY1(cc)      (cc >= GGML_CUDA_CC_QY1 && cc < GGML_CUDA_CC_QY2)
#define GGML_CUDA_CC_IS_QY2(cc)      (cc >= GGML_CUDA_CC_QY2 && cc < GGML_CUDA_CC_PH1)
#define GGML_CUDA_CC_IS_PH1(cc)      (cc >= GGML_CUDA_CC_PH1)

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#    define GGML_CUDA_USE_CUB
#endif  // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070

// PDL host-side support (cudaLaunchKernelEx) requires CUDART >= 11.8.
// However, this has been bugged in CTK < 12.3 for MSVC builds, see
// https://github.com/ggml-org/llama.cpp/pull/22522#discussion_r3302393293
// __CUDA_ARCH__  is undefined in host passes; GPU arch check happens in device-side code.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && \
    (CUDART_VERSION >= 12030 || (!(defined(_MSC_VER) && !defined(__clang__)) && CUDART_VERSION >= 11080))
#    define GGML_CUDA_USE_PDL
#endif  // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (CUDART_VERSION >= 12030 || (!(defined(_MSC_VER) && !defined(__clang__)) && CUDART_VERSION >= 11080))

static __device__ __forceinline__ void ggml_cuda_pdl_sync() {
#if defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    cudaGridDependencySynchronize();
#endif // defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

static __device__ __forceinline__ void ggml_cuda_pdl_lc() {
#if defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
    cudaTriggerProgrammaticLaunchCompletion();
#endif // defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER
}

#define GGML_CUDA_MAX_STREAMS 8

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define CUDA_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)

#if CUDART_VERSION >= 12000 || defined(GGML_USE_MUSA)
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        return cublasGetStatusString(err);
    }
#else
    static const char * cublas_get_error_str(const cublasStatus_t err) {
        switch (err) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            default: return "unknown error";
        }
    }
#endif // CUDART_VERSION >= 12000

#define CUBLAS_CHECK(err) CUDA_CHECK_GEN(err, CUBLAS_STATUS_SUCCESS, cublas_get_error_str)

#ifdef GGML_USE_NCCL
#define NCCL_CHECK(err) CUDA_CHECK_GEN(err, ncclSuccess, ncclGetErrorString)
#endif // GGML_USE_NCCL

#if !defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)
static const char * cu_get_error_str(CUresult err) {
    const char * err_str;
    cuGetErrorString(err, &err_str);
    return err_str;
}
#define CU_CHECK(err) CUDA_CHECK_GEN(err, CUDA_SUCCESS, cu_get_error_str)
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes)                                                       \
        do {                                                                                                   \
            static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = { false };                         \
            const int   id                                                = ggml_cuda_get_device();            \
            if (!shared_memory_limit_raised[id]) {                                                             \
                CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes)); \
                shared_memory_limit_raised[id] = true;                                                         \
            }                                                                                                  \
        } while (0)
#else
#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes) \
        do {                                             \
            GGML_UNUSED(nbytes);                         \
        } while (0)
#endif // !(defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#if CUDART_VERSION >= 11100 || defined(GGML_USE_MUSA)
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif // CUDART_VERSION >= 11100

#ifdef GGML_CUDA_F16
typedef half dfloat; // dequantize float
typedef half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
#endif // GGML_CUDA_F16

#if (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))
#define GGML_USE_VMM
#endif // (!defined(GGML_USE_HIP) && !defined(GGML_CUDA_NO_VMM)) || (defined(GGML_USE_HIP) && !defined(GGML_HIP_NO_VMM))

#if (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif // (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL

#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif // defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#define FP16_MMA_AVAILABLE
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define INT8_MMA_AVAILABLE
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)
#define FLASH_ATTN_AVAILABLE
#endif // !defined(GGML_CUDA_NO_FA) && !(defined(GGML_USE_MUSA) && __MUSA_ARCH__ < 220)

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

// Bulk asynchronous copies (the 1D form of TMA: cp.async.bulk + mbarrier complete_tx), sm_90 and newer.
// sm_120/sm_120a (RTX 50) accept these; the tensor-map (cuTensorMapEncode) form is not needed for them.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER && \
    defined(CUDART_VERSION) && CUDART_VERSION >= 12000
#define TMA_BULK_AVAILABLE
#endif // TMA_BULK_AVAILABLE

#ifdef __CUDA_ARCH_LIST__
constexpr bool ggml_cuda_has_arch_impl(int) {
    return false;
}

template<class ... Archs>
constexpr bool ggml_cuda_has_arch_impl(const int arch, const int first, Archs... rest) {
    return arch == first || ggml_cuda_has_arch_impl(arch, rest...);
}

constexpr bool ggml_cuda_has_arch(const int arch) {
    return ggml_cuda_has_arch_impl(arch, __CUDA_ARCH_LIST__);
}

constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur) {
    if (cur == 0) {
        GGML_ABORT("ggml was not compiled with any CUDA arch <= %d", arch);
    }
    return cur;
}

template<class ... Archs>
constexpr int ggml_cuda_highest_compiled_arch_impl(const int arch, const int cur, const int first, Archs... rest) {
    if (first <= arch && first > cur) {
        return ggml_cuda_highest_compiled_arch_impl(arch, first, rest...);
    } else {
        return ggml_cuda_highest_compiled_arch_impl(arch, cur, rest...);
    }
}

constexpr int ggml_cuda_highest_compiled_arch(const int arch) {
    return ggml_cuda_highest_compiled_arch_impl(arch, 0, __CUDA_ARCH_LIST__);
}
#else
static constexpr bool ggml_cuda_has_arch(const int) {
    return true;
}

static int ggml_cuda_highest_compiled_arch(const int arch) {
    return arch;
}
#endif // __CUDA_ARCH_LIST__

static constexpr bool fast_fp16_available(const int cc) {
    return cc >= GGML_CUDA_CC_PASCAL && cc != 610;
}

static constexpr bool fp16_mma_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && cc >= GGML_CUDA_CC_VOLTA;
}

static constexpr bool int8_mma_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && cc >= GGML_CUDA_CC_TURING;
}

// Volta technically had FP16 tensor cores but they work very differently compared to Turing and later.
static bool new_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

static bool cp_async_available(const int cc) {
    return cc < GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
#define AMD_MFMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
#define AMD_WMMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(RDNA4)

// The Volta instructions are in principle available on Turing or newer but they are effectively unusable:
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#    define BLACKWELL_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL

// NOTE: fork keeps FP16_MMA_AVAILABLE/INT8_MMA_AVAILABLE/CP_ASYNC_AVAILABLE (above, GGML_CUDA_CC_VOLTA/GGML_CUDA_CC_TURING/GGML_CUDA_CC_AMPERE
// gated) as the live compile-time gates for its own kernels; VOLTA/TURING/AMPERE/BLACKWELL_MMA_AVAILABLE above are
// upstream's newer, more granular names -- both sets are referenced by other files in this directory, so both stay.

static bool fp16_available(const int cc) {
    return ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_PASCAL ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fast_fp16_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_PASCAL && cc != 610) || GGML_CUDA_CC_IS_AMD(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

// To be used for feature selection of external libraries, e.g. cuBLAS.
static bool fp16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA) ||
        GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc) ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);
}

static bool bf16_mma_hardware_available(const int cc) {
    return (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) ||
        GGML_CUDA_CC_IS_CDNA(cc) || cc >= GGML_CUDA_CC_RDNA3 ||
        (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_PH1);
}

static bool fp32_mma_hardware_available(const int cc) {
    return GGML_CUDA_CC_IS_CDNA(cc);
}

static bool amd_mfma_available(const int cc) {
#if !defined(GGML_HIP_NO_MMQ_MFMA)
    return GGML_CUDA_CC_IS_CDNA(cc);
#else
    return false;
#endif //!defined(GGML_HIP_NO_MMQ_MFMA)
}

static bool amd_wmma_available(const int cc) {
    return (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
}

static bool volta_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_VOLTA;
}

static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING;
}

static bool ampere_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
}

// GGML_CUDA_NO_TF32=1: true F32 math wherever the backend has a choice -- CUBLAS_PEDANTIC_MATH on every cuBLAS handle
// and no TF32 tensor-core kernel for F32 weights (MMF takes F32 through TF32 mma on Ampere and newer). Read once.
// DISABLE-REVERSIBLE: GGML_CUDA_NO_TF32 (default unset = TF32 on, as before)
static bool ggml_cuda_no_tf32() {
    static const bool no_tf32 = [] { const char * e = getenv("GGML_CUDA_NO_TF32"); return e != nullptr && e[0] != '\0' && e[0] != '0'; }();
    return no_tf32;
}

static bool blackwell_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL &&
           ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_RUBIN;
}

// Checks whether the tensor's base data pointer and higher-dimensional strides are byte-aligned to `alignment` bytes.
static bool ggml_cuda_is_aligned(const ggml_tensor * tensor, const size_t alignment) {
    GGML_ASSERT(tensor != nullptr);
    return (reinterpret_cast<uintptr_t>(tensor->data) % alignment) == 0 &&
           tensor->nb[1] % alignment == 0 &&
           tensor->nb[2] % alignment == 0 &&
           tensor->nb[3] % alignment == 0;
}

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
    return 64;
#else
    return 32;
#endif // defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#ifdef GGML_USE_HIP
    return 16;
#else
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    return 16;
#else
    return 8;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
#endif // GGML_USE_HIP
}

[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
           file_name, line, function_name, arch);
    GGML_UNUSED(arch_list);
#else
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. ggml-cuda.cu was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning

#if defined(GGML_USE_MUSA)
    __builtin_unreachable();
#endif // defined(GGML_USE_MUSA)
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))
#else
#define NO_DEVICE_CODE //GGML_ABORT("NO_DEVICE_CODE not valid in host code.")
#endif // __CUDA_ARCH__

static __device__ __forceinline__ void ggml_cuda_syncwarp() {
#ifndef GGML_USE_HIP
    __syncwarp();
#endif // GGML_USE_HIP
}

// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    return __reduce_add_sync(0xffffffff, x);
#else
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, offset, width);
        a.y += __shfl_xor_sync(0xffffffff, a.y, offset, width);
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#ifdef FP16_AVAILABLE
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, offset, width));
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
    // `if constexpr`, not `if`. Two copies of this function existed -- here, and identically in
    // fattn-vec-common.cuh:748 and fattn-new-mma.cu:1792 -- and they collided in every TU that
    // included both, 76 errors. The fattn copies differed in exactly two ways: they used
    // `if constexpr`, and they replaced ggml_cuda_get_physical_warp_size() with WARP_SIZE, leaving
    // the original call commented out beside it.
    //
    // The `if constexpr` is the part that was actually needed: with a runtime `if`, BOTH branches
    // must compile for every instantiated width, and the shuffle loop does not for all of them.
    // The WARP_SIZE substitution looks like collateral damage from getting there -- it is wrong on
    // GFX9/GFX8, where the physical warp is 64 and WARP_SIZE is not.
    //
    // So the two are unified here keeping BOTH correct halves: constexpr selection AND the physical
    // warp size. That is only sound because ggml_cuda_get_physical_warp_size() is itself
    // `constexpr __device__` and its body is decided entirely by preprocessor guards, so the
    // condition is a constant expression. Verified by compiling, not by reading: see the commit.
    if constexpr (width == ggml_cuda_get_physical_warp_size()) {
        return __all_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) && x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __any_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

template<typename T, int width = WARP_SIZE>
static __device__ __forceinline__ T warp_prefix_inclusive_sum(T x) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const T t = __shfl_up_sync(0xffffffff, x, offset, width);
        if (lane_id >= offset) {
            x += t;
        }
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_prefix_inclusive_sum(float2 a) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const float t_x = __shfl_up_sync(0xffffffff, a.x, offset, width);
        const float t_y = __shfl_up_sync(0xffffffff, a.y, offset, width);
        if (lane_id >= offset) {
            a.x += t_x;
            a.y += t_y;
        }
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_prefix_inclusive_sum(half2 a) {
#ifdef FP16_AVAILABLE
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const half2 t = __shfl_up_sync(0xffffffff, a, offset, width);
        if (lane_id >= offset) {
            a = __hadd2(a, t);
        }
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

enum class block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T>
struct block_reduce_policy;

template <typename T, typename... Ts>
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
inline constexpr bool ggml_cuda_dependent_false_v = false;

template <typename T> struct block_reduce_policy<block_reduce_method::SUM, T> {
    static __device__ T reduce(T val) {
        if constexpr(is_any<T, float, float2, half2, int>) {
            return warp_reduce_sum(val);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.0f;
        } else if constexpr (std::is_same_v<T, float2>) {
            return make_float2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, half2>) {
            return make_half2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, int>) {
            return 0;
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T> struct block_reduce_policy<block_reduce_method::MAX, T> {
    static __device__ T reduce(T val) {
        if constexpr (is_any<T, float, half2>) {
            return warp_reduce_max(val);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return -INFINITY;
        } else if constexpr (std::is_same_v<T, half2>) {
            return make_half2(-INFINITY, -INFINITY);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};

template <block_reduce_method reduce_method_t, const unsigned int block_size_template = 0, typename T>
static __device__ T block_reduce(T val, [[maybe_unused]] T * shared_vals) {
    // for multi-warp reductions, callers must not reuse shared_vals until all reads from this invocation have completed
    val                           = block_reduce_policy<reduce_method_t, T>::reduce(val);
    const unsigned int block_size = block_size_template == 0 ? blockDim.x : block_size_template;
    if (block_size > WARP_SIZE) {
        assert((block_size <= 1024) && (block_size % WARP_SIZE) == 0);
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        __syncthreads();
        val = block_reduce_policy<reduce_method_t, T>::sentinel();
        if (lane_id < (static_cast<int>(block_size) / WARP_SIZE)) {
            val = shared_vals[lane_id];
        }
        return block_reduce_policy<reduce_method_t, T>::reduce(val);
    }

    return val;
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half a, const half b) {
#ifdef FP16_AVAILABLE

#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX
    return __float2half(fmaxf(__half2float(a), __half2float(b)));
#else
    return __hmax(a, b);
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && CUDART_VERSION < CUDART_HMAX

#else
   NO_DEVICE_CODE;
   GGML_UNUSED(b);
   return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 a, const half2 b) {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    return half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
#elif CUDART_VERSION >= CUDART_HMAX
    return __hmax2(a, b);
#else
    half2 ret;
    reinterpret_cast<half&>(ret.x) = __float2half(fmaxf( __low2float(a),  __low2float(b)));
    reinterpret_cast<half&>(ret.y) = __float2half(fmaxf(__high2float(a), __high2float(b)));
    return ret;
#endif
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#if (!(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL) || (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
#pragma unroll
   for (int offset = width/2; offset > 0; offset >>= 1) {
       x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, offset, width));
   }
   return x;
#else
   GGML_UNUSED(x);
   NO_DEVICE_CODE;
#endif // (!(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) && __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL) || (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
}

#if (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || \
    (defined(MUSART_VERSION) && MUSART_VERSION < MUSART_HMASK)
static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float( __low2half(a)) > float( __low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}
#endif // (defined(CUDART_VERSION) && CUDART_VERSION < CUDART_HMASK) || (defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)) || (defined(MUSART_VERSION) && MUSART_VERSION < MUSART_HMASK)

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx1010__) || defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

#else // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

#if __CUDA_ARCH__ >= MIN_CC_DP4A || defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= MIN_CC_DP4A || defined(GGML_USE_MUSA)
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A || defined(GGML_USE_MUSA)

#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
}

#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
// look up the 8 nibbles of q4 in a 16 entry table. HIP emulates __byte_perm() with an 8 byte union
// and four dynamically indexed byte loads, which end up in scratch, so the __byte_perm() version in
// get_int_from_table_16() is an order of magnitude slower here. v_perm_b32 does the same in one
// instruction, but takes one selector byte per output byte where prmt takes one nibble
// (ik_llama #2339; guard spelled GGML_USE_HIP here, the fork's spelling)
static __device__ __forceinline__ int2 ggml_cuda_perm_table_16(const int & q4, const int8_t * values) {
    const uint32_t * values32 = (const uint32_t *)values;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    // indices 0-7 come from the lower half of the table, 8-15 from the upper half
    const uint32_t v_even_low  = __builtin_amdgcn_perm(values32[1], values32[0], q_even & 0x07070707);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values32[1], values32[0], q_odd  & 0x07070707);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values32[3], values32[2], q_even & 0x07070707);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values32[3], values32[2], q_odd  & 0x07070707);

    // bit 3 of each index picks the half, i.e. byte 0-3 or byte 4-7 of the pair above
    const uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    const uint32_t mask_odd  = 0x03020100 | ((q_odd  & 0x08080808) >> 1);

    return make_int2(__builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
                     __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
}
#endif // defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

// kvalues_iq4nl WAS defined here, with the note "move to ggml-common.h eventually". That move
// HAPPENED -- the 2026-09-08 merge put the table in ggml-common.h:2463 so upstream's ggml-cpu
// sources could see it (28 sites had failed with `'kvalues_iq4nl': undeclared identifier`) -- but
// this copy was left behind, so both definitions reached every CUDA TU:
//     error: variable "kvalues_iq4nl" has already been defined
//            (previous definition at line 2463 of ggml-common.h)
// This file includes ggml-common.h at line 23, and under GGML_COMMON_IMPL_CUDA the table expands
// to `static const __device__ int8_t kvalues_iq4nl[16]` (ggml-common.h:1005) -- same storage class,
// same device placement. The two value lists were compared element by element and are IDENTICAL,
// which is the only reason deleting one is safe rather than merely convenient.
// Removed rather than #ifdef'd: a second spelling of a shared table is how they drift apart.

// NOTE(merge): fork's dfloat2-based dequantize_kernel_t is kept, not upstream's float2 version.
// PRIOR-ART-NONE beyond a direct check: dmmv.cu (clean, not part of this merge) feeds dfloat2
// through this typedef, so switching to float2 here would break that file, which is outside this
// task's scope (ggml-cuda/dmmv.cu is not in the unmerged set).
typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, dfloat2 & v);

template<typename dst_t>
using dequantize_kq_t = void (*)(const void * vx, const int64_t ib, dst_t * y, const int tid);

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x*u.x;
    acc += v.y*u.y;
}

#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
#define V_DOT2_F32_F16_AVAILABLE
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
#ifdef V_DOT2_F32_F16_AVAILABLE
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v*u);
    acc += tmp.x + tmp.y;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // FAST_FP16_AVAILABLE
#endif // V_DOT2_F32_F16_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v*u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right instructions.
//     If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
#if CUDART_VERSION >= 12080
    const nv_bfloat16 e = __nv_cvt_e8m0_to_bf16raw(x);
    return (float) e;
#else
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
#endif // CUDART_VERSION >= 12050
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
#if defined(GGML_USE_HIP) && defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
    // ROCm does not support fp8 in software on devices with fp8 hardware,
    // but CDNA3 supports only e4m3_fnuz (no inf).
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __hip_fp8_e4m3_fnuz xf = *reinterpret_cast<const __hip_fp8_e4m3_fnuz *>(&bits);
    return static_cast<float>(xf) / 2;
#else
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
#else
    if (x == 0 || (x == 0x7F && x != 0xFF)) { // Convert NaN to 0.0f
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return static_cast<float>(raw / 2);
#endif // defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
#endif // defined(GGML_USE_HIP) && defined(CDNA3) && defined(FP8_AVAILABLE) && HIP_VERSION >= 60200000
}

static __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_ue4m3(float x) {
#if defined(BLACKWELL_MMA_AVAILABLE) // This is used for NVFP4 subblock scale quantizations only
    if (!(x > 0.0f)) {
        return 0;
    }
    const __nv_fp8_e4m3 xf(x);
    return xf.__x;
#else
     NO_DEVICE_CODE; // Used only for NVFP4 Scales for Activations, only for Blackwell
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

__device__ __forceinline__ uint8_t ggml_cuda_float_to_fp4_e2m1(float x, float e) {
    const uint8_t sign_bit = (x < 0.0f) << 3;
    float         ax       = fabsf(x) * e;

    // Positive LUT
    static constexpr float pos_lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

    int   best_i   = 0;
    float best_err = fabsf(ax - pos_lut[0]);

#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float err = fabsf(ax - pos_lut[i]);
        if (err < best_err) {
            best_err = err;
            best_i   = i;
        }
    }

    return static_cast<uint8_t>(best_i | sign_bit);
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return powf(base, exph);
}

// NOTE(merge): iq1bn_fp8_to_float dropped -- PRIOR-ART-NONE: grep -rlw across ggml/src/ggml-cuda/
// finds zero callers (CPU-side ggml.c/iqk has its own copy); upstream also drops it here.
//
// NOTE(merge): GGML_TYPE_Q1_0 / GGML_TYPE_Q2_0 (upstream's plain, non-G128 types, enum 400/401,
// renumbered clear of the fork's own 41/42) were once blocked here on missing QK1_0/QK2_0 constants;
// ggml-common.h defines them now and the specializations sit below, with dequant, get_rows and
// mmvq (template-instances/mmvq-instance-q1_0.cu, -q2_0.cu). No MMQ tile loader yet: batched
// matmuls on these types go through dequant -> cuBLAS.
template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_F16> {
    static constexpr int qk = 1;
    static constexpr int qr = 1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = QI4_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q1_0_G128> {
    // See the Q2_0_G128 block below for why qr == 1 is required rather than stylistic.
    // qi is ints of QUANT DATA per block and the MMVQ driver divides by it:
    // 128 weights * 1 bit = 16 bytes = 4 ints. The qk/(4*qr) convention would give 32 and
    // is only correct for 4-bit types.
    static constexpr int qk = QK1_0_G128;
    static constexpr int qr = 1;
    static constexpr int qi = QK1_0_G128 / 32;
};

// Upstream plain Q1_0 / Q2_0 (enum 400/401). Same sequential layout as the G128 pair, so the same
// qr == 1 / qi == ints-of-quant-data reasoning applies: Q1_0 = 128 x 1 bit = 16 B = 4 ints,
// Q2_0 = 64 x 2 bit = 16 B = 4 ints. QK1_0/QK2_0 are defined in ggml-common.h.
template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q1_0> {
    static constexpr int qk = QK1_0;
    static constexpr int qr = 1;
    static constexpr int qi = QK1_0 / 32;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_0> {
    static constexpr int qk = QK2_0;
    static constexpr int qr = 1;
    static constexpr int qi = QK2_0 / 16;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_0_G128> {
    // qr == 1 is REQUIRED, not stylistic: dequantize_block() derives
    // `y_offset = qr == 1 ? 1 : qk/2`, so qr == 2 would write the pair half a block apart. That
    // suits the 4-bit nibble types; this layout is SEQUENTIAL (byte b holds elements 4b..4b+3),
    // so the pair must land adjacent.
    // qi is the count of 32-bit ints of QUANT DATA per block, and the MMVQ driver divides by it:
    // `kbx = tid / (qi/vdr)` and `kqs = vdr * (tid % (qi/vdr))`. It was qk/(4*qr) == 32 while this
    // type had no MMVQ instance -- the qk/(4*qr) convention holds for 4-bit types (2 weights per
    // byte) and is simply wrong at 2 bits. Corrected when the kernel was written, per the note
    // that used to stand here: 2 bits/weight -> qk/4 bytes -> qk/16 ints == 8.
    // The dequant->cuBLAS path is unaffected: dequantize_block_cuda takes qk and qr explicitly.
    static constexpr int qk = QK2_0_G128;
    static constexpr int qr = 1;
    static constexpr int qi = QK2_0_G128 / 16;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_1> {
    static constexpr int qk = QK4_1;
    static constexpr int qr = QR4_1;
    static constexpr int qi = QI4_1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;
    static constexpr int qr = QR5_0;
    static constexpr int qi = QI5_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;
    static constexpr int qr = QR5_1;
    static constexpr int qi = QI5_1;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q6_0> {
    static constexpr int qk = QK6_0;
    static constexpr int qr = QR6_0;
    static constexpr int qi = QI6_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_MXFP4> {
    static constexpr int qk = QK_MXFP4;
    static constexpr int qr = QR_MXFP4;
    static constexpr int qi = QI_MXFP4;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_NVFP4> {
    static constexpr int qk = QK_NVFP4;
    static constexpr int qr = QR_NVFP4;
    static constexpr int qi = QI_NVFP4;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XXS;
    static constexpr int qi = QI2_XXS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XS;
    static constexpr int qi = QI2_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_S;
    static constexpr int qi = QI2_S;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_XXS;
    static constexpr int qi = QI3_XXS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_S;
    static constexpr int qi = QI1_S;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S_R4> {
    static constexpr int qk = 32;
    static constexpr int qr = 2;
    static constexpr int qi = 4;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_M> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_M;
    static constexpr int qi = QI1_M;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_BN> {
    static constexpr int qk = QK_IQ1BN;
    static constexpr int qr = QR1_BN;
    static constexpr int qi = QI1_BN;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQ2_0> {
    // Mainline ternary 2-bit, 256-element super-block with an fp16 scale in the block.
    // Same shape as IQ2_KS: qi/vdr = 32/4 = 8 threads per super-block, one per q8_1 block.
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQ1_0> {
    // Mainline ternary 1.69 bpw, 256-element super-block: qs[48] base-3 (5 trits/byte) + qh[4]
    // (4 trits/byte) + an fp16 scale. The PACKING differs completely from TQ2_0, but the
    // super-block SHAPE is identical, so the same traits apply: qi/vdr = 32/4 = 8 threads per
    // super-block, one per q8_1 block.
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_BN> {
    static constexpr int qk = QK_IQ1BN;
    static constexpr int qr = QR1_BN;
    static constexpr int qi = QI1_BN;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_PQ2_0> {
    static constexpr int qk = QK_PQ2_0;
    static constexpr int qr = PQR2_0;
    static constexpr int qi = PQI2_0;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int qk = QK4_NL;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
};

// NOTE(merge): fork's own GGML_TYPE_MXFP4 placeholder (QK4_NL/QR4_NL/QI4_NL, same values as
// IQ4_NL above) is dropped here in favor of upstream's dedicated QK_MXFP4/QR_MXFP4/QI_MXFP4
// specialization added earlier in this table (right after GGML_TYPE_Q8_0) -- those constants
// already exist in ggml-common.h (verified) and are more correct than the IQ4_NL stand-in;
// keeping both would be a duplicate template specialization (compile error).
template<>
struct ggml_cuda_type_traits<GGML_TYPE_NVFP4_KS> {
    static constexpr int qk = QK_NVFP4;
    static constexpr int qr = QR_NVFP4;
    static constexpr int qi = QI_NVFP4;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_KS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_KT> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_KT> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_KT> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_KT> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_KL> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_KS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_KS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_KS_R4> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_KS_R16> {
    static constexpr int qk = 32;   // block = 16 rows x 32 columns
    static constexpr int qr = 2;
    static constexpr int qi = 8;    // one lane per block (vdr = 8)
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_KS_R16> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = 16;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ5_KS_R16> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = 16;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_KSS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_XS;
    static constexpr int qi = QI5_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ5_KS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_XS;
    static constexpr int qi = QI5_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ5_KS_R4> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_XS;
    static constexpr int qi = QI5_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_XS;
    static constexpr int qi = QI6_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_S;
    static constexpr int qi = QI3_S;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_K_R4> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_K_R4> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_K_R4> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ5_K_R4> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_XS;
    static constexpr int qi = QI5_XS;
};


#define QR_TQ 4
#define QI_TQ (QK_TQK/QR_TQ)

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQ2_T> {
    static constexpr int qk = QK_TQK;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQK6> {
    static constexpr int qk = QK_TQK;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQK7> {
    static constexpr int qk = QK_TQK;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQP5> {
    static constexpr int qk = QK_TQP;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQP6> {
    static constexpr int qk = QK_TQP;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_TQP7> {
    static constexpr int qk = QK_TQP;
    static constexpr int qr = QR_TQ;
    static constexpr int qi = QI_TQ;
};


//////////////////////

struct ggml_backend_cuda_context;

struct ggml_cuda_device_info {
    int device_count;           // number of (possibly virtual) devices exposed to the rest of ggml
    int physical_device_count;  // number of physical CUDA devices actually present

    struct cuda_device_info {
        int     cc;                             // compute capability
        int     nsm;                            // number of streaming multiprocessors
        size_t  smpb;                           // max. shared memory per block
        size_t  smpbo;                          // max. shared memory per block (with opt-in)
        bool    integrated;                     // Device is integrated as opposed to discrete
        bool    vmm;                            // virtual memory support
        size_t  vmm_granularity;                // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;                      // Number of threads in a dispatch
        bool    supports_cooperative_launch;    // whether cooperative launch is supported
        int     physical_device;                // backing physical CUDA device for this (virtual) device
        int     physical_share_count;           // number of (virtual) devices sharing this device physical GPU
        int     virtual_index;                  // index of this (virtual) device among those sharing its physical GPU
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};

    std::unordered_map<const void *, std::array<ggml_backend_cuda_context *, GGML_CUDA_MAX_DEVICES>> all_ctx;
    //ggml_backend_cuda_context * all_ctx[GGML_CUDA_MAX_DEVICES] = { nullptr };
#ifdef GGML_USE_NCCL
    ncclComm_t nccl_coms[GGML_CUDA_MAX_DEVICES];
    bool have_nccl;
#endif
};

const ggml_cuda_device_info & ggml_cuda_info();

void ggml_cuda_set_device(int device);
int ggml_cuda_get_device();

struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool & pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_cuda_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc &&) = delete;
};


// backend interface

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // events for synchronizing multiple GPUs
};


#if (CUDART_VERSION >= 12000) && defined(GGML_CUDA_USE_GRAPHS)
#define USE_CUDA_GRAPH
#endif

// NOTE(merge): fork's real ggml_cuda_graph definition lives in ggml-cuda/graph.cuh (already clean,
// not part of this merge) and is #include'd directly by ggml-cuda.cu / cpy.cu; keep this a forward
// declare only, matching fork's original -- do NOT paste upstream's node_props-based redefinition
// here, it would collide with graph.cuh's definition (fork's ggml-cuda.cu -- forbidden file --
// dereferences fork's own field names: cpy_dest_ptrs, disable_due_to_too_many_updates,
// ggml_graph_properties, graph_cpynode_index, use_cpy_indirection -- which only graph.cuh provides).
//
// Upstream's ggml_cuda_concurrent_event / ggml_cuda_stream_context (and the any_cuda_graph_enabled /
// any_cuda_graph_has_instance helpers that exist only to serve them) are dropped here:
// PRIOR-ART-NONE: grep -rlw "concurrent_stream_context|stream_context|ggml_cuda_stream_context|
// ggml_cuda_concurrent_event" across ggml/src/ggml-cuda/**/*.cu(h) and ggml/src/ggml-cuda.cu found
// exactly one hit -- ggml/src/ggml-cuda/ggml-cuda.cu, which is upstream's OWN copy of the root file
// living inside this directory, excluded from the build per this task's general method note. Zero
// call sites in anything the fork actually compiles. DISABLE-REVERSIBLE: re-add verbatim from
// unsloth/master's ggml-cuda/common.cuh (`git show :3:ggml/src/ggml-cuda/common.cuh`) the moment a
// live caller appears -- most likely if/when the concurrent-stream-fusion feature is adopted.
// A FORWARD DECLARATION IS NOT ENOUGH HERE, and this used to be one.
// ggml_backend_cuda_context::cuda_graph() below is an inline member function: its body is compiled
// wherever this header is included, and it dereferences ggml_cuda_graph (->last_used_time) and
// constructs one (std::make_unique). Both need the COMPLETE type. With only the forward
// declaration, every CUDA TU that includes common.cuh and is not cpy.cu -- the sole includer of
// graph.cuh -- failed with
//     error: pointer or reference to incomplete type "ggml_cuda_graph" is not allowed
// at the two lines below. MEASURED on cumsum.cu. The member map is fine with an incomplete type
// (std::unique_ptr permits that); it is the function BODY that is not.
//
// graph.cuh is included HERE rather than at the top of this file on purpose: it uses CUDA_CHECK
// (defined at common.cuh:155) and std::vector, so hoisting it above those would simply move the
// error. It includes only ggml.h itself, so there is no cycle.
#include "graph.cuh"

struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;
    cudaEvent_t compute_event = nullptr;
    bool p2p_enabled = false;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
    void * cublas_workspaces[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
    size_t cublas_workspace_sizes[GGML_CUDA_MAX_DEVICES] = {0};

    int curr_stream_no = 0;

    int   fusion = GGML_CUDA_FUSION;
    int   offload_batch_size = GGML_CUDA_MIN_BATCH_OFFLOAD;
    int   offload_batch_size_per_byte = -1;
    int   mmq_id_thresh = 32;
    float fa_offset = 0.6931f; // ln(2)
    bool  fabsum = false;      // opt-in: bound the fp16 accumulation error in the tile FA kernel
#ifdef USE_CUDA_GRAPH
    bool use_cuda_graph = true;

    ggml_cuda_graph * cur_graph = nullptr;

    // Map from first_node_ptr to cuda_graph - allows multiple graphs per context
    // when the computation is split across CPU/GPU (e.g., with --n-cpu-moe)
    std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;

    int64_t last_graph_eviction_sweep = 0;

    // Set by the legacy pool when it returns device memory to the driver (cudaFree): a captured graph may still
    // reference the freed buffer, so every captured graph must be re-captured. The pool only SETS this flag;
    // ggml_backend_cuda_graph_compute clears cuda_graphs at its next entry, because pool frees happen mid-evaluation
    // while cur_graph (read by cpy.cu) still points into cuda_graphs -- clearing there would dangle it.
    // RESEARCH-DONE: upstream 13786fbb3 (https://github.com/ggml-org/llama.cpp/pull/26771, read in full, 106-line
    // diff) clears ctx.cuda_graphs directly inside the pool ("captured CUDA graphs can still use freed buffers, so
    // they must be captured again"); deferred here for the fork's cur_graph. Row S69-UPMERGE-LOSS-CUDA-POOL-LEG-GRAPH-INVALIDATE-2026-11-06-01.
    bool cuda_graphs_need_reset = false;

    // Per-context CUDA graph counters, printed at context destruction when GGML_CUDA_GRAPH_STATS is
    // set. They are how "this path runs without graphs" is measured instead of inferred from a
    // kernel-launch count (row S69-SERVER-SPEC-PATH-NO-CUDA-GRAPHS-2026-10-18-01).
    // RESEARCH-DONE: upstream llama.cpp master ggml-cuda.cu (sha256 68f09269...7777) replaced the
    // sticky too-many-updates switch-off with a warmup state machine; `legacy_off` below counts the
    // old policy only, reachable solely through GGML_CUDA_GRAPH_LEGACY=1 for A/B.
    struct {
        uint64_t computes   = 0; // graph_compute calls
        uint64_t launches   = 0; // cudaGraphLaunch calls
        uint64_t captures   = 0; // stream captures (first capture and every re-capture)
        uint64_t direct     = 0; // calls evaluated node by node
        uint64_t resets     = 0; // warmed-up graphs whose properties changed
        uint64_t legacy_off = 0; // legacy policy only: graphs switched off for too many updates
        uint64_t created    = 0; // cache entries created
        uint64_t evicted    = 0; // cache entries evicted by the sweep
    } graph_stats;

    ggml_cuda_graph * cuda_graph(const void * first_node_ptr) {
        const int64_t time_now = ggml_time_us();

        // sweep every 5s, evicting cuda graphs unused for >=10s
        if (time_now - last_graph_eviction_sweep >= 5'000'000) {
            last_graph_eviction_sweep = time_now;
            for (auto it = cuda_graphs.begin(); it != cuda_graphs.end(); ) {
                if (time_now - it->second->last_used_time >= 10'000'000) {
                    it = cuda_graphs.erase(it);
                    graph_stats.evicted++;
                } else {
                    ++it;
                }
            }
        }

        auto it = cuda_graphs.find(first_node_ptr);
        if (it == cuda_graphs.end()) {
            it = cuda_graphs.emplace(first_node_ptr, std::make_unique<ggml_cuda_graph>()).first;
            graph_stats.created++;
        }
        it->second->last_used_time = time_now;
        return it->second.get();
    }

    // Re-added 2026-10-18 for mean.cu (upstream aa064b2eb #14313), the live caller the note above
    // waits for; upstream b9590 common.cuh:1433-1450. mean.cu picks its CUB single-row threshold
    // from these. Upstream's ggml_cuda_graph::is_enabled() is !(disable_due_to_gpu_arch || env);
    // the fork's graph struct has no such method, so the same test is spelled out here.
    bool any_cuda_graph_enabled() const {
        static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
        for (const auto & [key, graph] : cuda_graphs) {
            if (graph && !(graph->disable_due_to_gpu_arch || disable_cuda_graphs_due_to_env)) {
                return true;
            }
        }
        return false;
    }

    // Check if any CUDA graph has an instance for this context
    bool any_cuda_graph_has_instance() const {
        for (const auto & [key, graph] : cuda_graphs) {
            if (graph && graph->instance != nullptr) {
                return true;
            }
        }
        return false;
    }
#endif // USE_CUDA_GRAPH

    const void * model;
    void * copy_buffer = nullptr;
    size_t copy_size   = 0;

    explicit ggml_backend_cuda_context(int device, const void * model);

    ~ggml_backend_cuda_context();

    cudaStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
        }
        return streams[device][stream];
    }

    cudaStream_t stream() { return stream(device, curr_stream_no); }

    // NOTE(merge): fork's cublas_handle(int device) overload is kept -- root ggml-cuda.cu,
    // indexer_topk.cu and latent_attn.cu (fork-only, must-keep archs) all call it with an explicit
    // device argument. Body updated to the 2-D [device][curr_stream_no] shape + workspace setup
    // upstream adds (needed once cublas_handles became 2-D, see the struct union above).
    cublasHandle_t cublas_handle(int device) {
        if (cublas_handles[device][curr_stream_no] == nullptr) {
            ggml_cuda_set_device(device);
            CUBLAS_CHECK(cublasCreate(&cublas_handles[device][curr_stream_no]));
            // TF32 tensor-op math is the default (fast F32 GEMMs, ~1e-3 relative rounding). GGML_CUDA_NO_TF32=1 selects CUBLAS_PEDANTIC_MATH instead: true F32
            // GEMMs, for validating the CUDA backend against the CPU one. Read when a handle is created, so it can be set at runtime before the first CUDA op;
            // NVIDIA_TF32_OVERRIDE cannot be (the CUDA libraries read it at load, before main). Unset = no change in behaviour.
            // DISABLE-REVERSIBLE: GGML_CUDA_NO_TF32 (default unset = TF32 on, as before)
            CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device][curr_stream_no], ggml_cuda_no_tf32() ? CUBLAS_PEDANTIC_MATH : CUBLAS_TF32_TENSOR_OP_MATH));
            CUBLAS_CHECK(cublasSetStream(cublas_handles[device][curr_stream_no], stream(device, curr_stream_no)));
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (CUBLAS_VER_MAJOR > 11 || (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR >= 2))
            if (cublas_workspace_sizes[device] == 0) {
                const int cc = ggml_cuda_info().devices[device].cc;
                cublas_workspace_sizes[device] = (cc >= GGML_CUDA_CC_HOPPER) ? 32 * 1024 * 1024 : 4 * 1024 * 1024;
            }
            CUDA_CHECK(cudaMalloc(&cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device]));
            CUBLAS_CHECK(cublasSetWorkspace(cublas_handles[device][curr_stream_no], cublas_workspaces[device][curr_stream_no], cublas_workspace_sizes[device]));
#endif
        }
        return cublas_handles[device][curr_stream_no];
    }

    cublasHandle_t cublas_handle() {
        return cublas_handle(device);
    }

    // pool -- kept 1-D (fork's shape): NOT part of the orchestrator's explicit union spec for this
    // struct, and widening it to [DEVICES][STREAMS] like cublas_handles would require touching
    // ctx.pool()/ctx.pool(device) call sites across many other files, out of scope here.
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES];

    // ctx is passed so the legacy pool can flag captured CUDA graphs stale when it frees device memory (upstream #26771)
    static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(ggml_backend_cuda_context & ctx, int device);

    ggml_cuda_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(*this, device);
        }
        return *pools[device];
    }

    ggml_cuda_pool & pool() {
        return pool(device);
    }
};

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device);

struct ggml_cuda_mm_fusion_args_host {
    const ggml_tensor * x_bias = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * gate_bias = nullptr;
    const ggml_tensor * x_scale = nullptr;
    const ggml_tensor * gate_scale = nullptr;
    ggml_glu_op glu_op;
};
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias = nullptr;
    const void * gate = nullptr;
    const void * gate_bias = nullptr;
    const void * x_scale = nullptr;
    const void * gate_scale = nullptr;
    ggml_glu_op glu_op;
};

struct ggml_cuda_kernel_launch_params {
    dim3 block_nums;
    dim3 block_dims;
    size_t shmem;
    cudaStream_t stream;

    // size_t shmem
    ggml_cuda_kernel_launch_params(const dim3& block_nums_, const dim3& block_dims_, const size_t shmem_, const cudaStream_t stream_)
        : block_nums(block_nums_), block_dims(block_dims_), shmem(shmem_), stream(stream_) {}

    // Some call sites pass ints instead of the required size_t. This 2nd constructor casts int->size_t to avoid these -Wnarrowing warnings.
    ggml_cuda_kernel_launch_params(const dim3& block_nums_, const dim3& block_dims_, const int shmem_, const cudaStream_t stream_)
        : block_nums(block_nums_), block_dims(block_dims_), shmem((size_t)shmem_), stream(stream_) {}
};

#if defined(GGML_CUDA_USE_PDL)
struct ggml_cuda_pdl_config {
    cudaLaunchAttribute attr;
    cudaLaunchConfig_t  cfg;

    ggml_cuda_pdl_config(const ggml_cuda_kernel_launch_params & params) {
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;

        cfg = {};
        cfg.gridDim          = params.block_nums;
        cfg.blockDim         = params.block_dims;
        cfg.dynamicSmemBytes = params.shmem;
        cfg.stream           = params.stream;
        cfg.attrs            = &attr;
        cfg.numAttrs         = 1;
    }

    // Delete due to &attr
    ggml_cuda_pdl_config(const ggml_cuda_pdl_config&) = delete;
    ggml_cuda_pdl_config& operator=(const ggml_cuda_pdl_config&) = delete;
    ggml_cuda_pdl_config& operator=(ggml_cuda_pdl_config&&) = delete;

};

static bool ggml_cuda_kernel_can_use_pdl(const void * kernel) {
    const int device = ggml_cuda_get_device();

    struct cache_key {
        int          device;
        const void * kernel;

        bool operator==(const cache_key & other) const { return device == other.device && kernel == other.kernel; }
    };

    struct cache_key_hash {
        // MurmurHash3 mixing function for better hash distribution (vs. just std::hash which in some implementations simply returns the identity)
        static size_t hash_mix(size_t x) {
            std::uint64_t       y = x;
            const std::uint64_t m = 0xe9846af9b1a615d;

            y ^= y >> 32;
            y *= m;
            y ^= y >> 32;
            y *= m;
            y ^= y >> 28;

            return static_cast<size_t>(y);
        }

        size_t operator()(const cache_key & key) const {
            // Use a nonzero seed to avoid mapping all-zero keys to zero
            size_t h = 42;
            h        = hash_mix(h + key.device);
            h        = hash_mix(h + reinterpret_cast<size_t>(key.kernel));
            return h;
        }
    };

    static std::mutex                                          cache_mutex;
    static std::unordered_map<cache_key, bool, cache_key_hash> cache;

    const cache_key             key = { device, kernel };
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto                  it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    cudaFuncAttributes attr = {};
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));

    // PDL device-side primitives are emitted only for PTX versions >= 90.
    // We have to guard on a loaded kernel's PTX version so a kernel forward-JIT'ed
    // from pre-Hopper PTX to a Hopper-or-newer GPU does not opt into PDL.
    const bool can_use_pdl = attr.ptxVersion >= 90;
    cache.emplace(key, can_use_pdl);
    return can_use_pdl;
}

#endif //defined(GGML_CUDA_USE_PDL)

// PDL and __restrict__ need to be mutually exclusive, see https://github.com/ggml-org/llama.cpp/pull/24030
# if (defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER)
# define GGML_CUDA_RESTRICT
# else
# define GGML_CUDA_RESTRICT __restrict__
# endif // defined(GGML_CUDA_USE_PDL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_HOPPER

template<typename Kernel, typename... Args>
static __inline__ void ggml_cuda_kernel_launch(Kernel kernel, const ggml_cuda_kernel_launch_params & launch_params, Args&&... args) {
#if defined(GGML_CUDA_USE_PDL)

    static const bool env_pdl_enabled = []() {
        const char * env = getenv("GGML_CUDA_PDL");
        return env == nullptr || std::atoi(env) != 0;
    }();

    if (env_pdl_enabled && ggml_cuda_kernel_can_use_pdl(reinterpret_cast<const void *>(kernel))) {
        auto pdl_cfg = ggml_cuda_pdl_config(launch_params);

        CUDA_CHECK(cudaLaunchKernelEx(&pdl_cfg.cfg, kernel, std::forward<Args>(args)... ));
        return;
    }
#endif //defined(GGML_CUDA_USE_PDL)

    kernel<<<launch_params.block_nums, launch_params.block_dims, launch_params.shmem, launch_params.stream>>>(std::forward<Args>(args)... );
    CUDA_CHECK(cudaGetLastError());
}
