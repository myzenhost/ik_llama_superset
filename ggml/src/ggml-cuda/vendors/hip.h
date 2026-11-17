#pragma once

// ROCm 6 and later provide __shfl_sync() and friends as templates that static_assert on the
// width of the mask, so the shims below have to replace them rather than shadow them
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#ifdef __HIP_PLATFORM_AMD__
// for rocblas_initialize()
#include "rocblas/rocblas.h"
#endif // __HIP_PLATFORM_AMD__

#ifdef GGML_USE_NCCL
#include <rccl/rccl.h>
#endif // GGML_USE_NCCL

#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_16BF HIPBLAS_R_16B
#define CUDA_R_32F  HIPBLAS_R_32F
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hipMemAllocationGranularityRecommended
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_CHECK(fn) {hipError_t err = fn; if(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_up_sync(mask, var, laneMask, width) __shfl_up(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define __all_sync(mask, var) __all(var)
#define __any_sync(mask, var) __any(var)
#define cublasStrsmBatched hipblasStrsmBatched
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define cublasStrsmBatched hipblasStrsmBatched
// hipBLAS has no equivalent of the cuBLAS math mode, TF32 is not a thing here
#define cublasMath_t int
#define CUBLAS_DEFAULT_MATH 0
#define cublasGetMathMode(handle, mode) (*(mode) = CUBLAS_DEFAULT_MATH, CUBLAS_STATUS_SUCCESS)
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasSgemmBatched hipblasSgemmBatched
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasStatus_t hipblasStatus_t
#define cublasOperation_t hipblasOperation_t
#define cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define CUresult hipError_t
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define CUDA_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventDestroy hipEventDestroy
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
// cudaFuncSetAttribute() is a template over the kernel type, hipFuncSetAttribute() takes a const void *
#define cudaFuncSetAttribute(func, attr, value) hipFuncSetAttribute((const void *)(func), attr, value)
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaMallocManaged hipMallocManaged
#define cudaMemAdvise hipMemAdvise
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice
#define cuDeviceGet hipDeviceGet
#define CUdevice hipDevice_t
#define CUdeviceptr hipDeviceptr_t
#define cuMemUnmap hipMemUnmap
#define CUmemAccessDesc hipMemAccessDesc
#define cuMemAddressFree hipMemAddressFree
#define cuMemRelease hipMemRelease
#define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
#define cuMemCreate hipMemCreate
#define cuMemAddressReserve hipMemAddressReserve
#define cuMemMap hipMemMap
#define cuMemSetAccess hipMemSetAccess
#define cuMemGetAllocationGranularity hipMemGetAllocationGranularity
#define CUmemAllocationProp hipMemAllocationProp
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamFireAndForget hipStreamFireAndForget
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphNode_t hipGraphNode_t
#define cudaKernelNodeParams hipKernelNodeParams
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure
#define cudaGraphExecUpdateResult hipGraphExecUpdateResult
#define cudaGraphNodeType hipGraphNodeType
#define cudaGraphNodeTypeKernel hipGraphNodeTypeKernel
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphKernelNodeSetParams hipGraphKernelNodeSetParams
#define cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#define cudaGraphKernelNodeGetParams hipGraphKernelNodeGetParams
#define cudaGraphNodeGetType hipGraphNodeGetType
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphExecUpdate hipGraphExecUpdate
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaGraph_t hipGraph_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define __trap() do { abort(); __builtin_unreachable(); } while(0)
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

#if HIP_VERSION >= 60500000
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#define cublasComputeType_t hipblasComputeType_t
#define cudaDataType_t hipDataType
#else
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define cublasComputeType_t hipblasDatatype_t
#define cudaDataType_t hipblasDatatype_t
#endif // HIP_VERSION >= 6050000

#if !defined(__HIP_PLATFORM_AMD__)
#error "The HIP backend supports only AMD targets"
#endif // !defined(__HIP_PLATFORM_AMD__)

#define __CUDA_ARCH__ 1300

#if defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#define GCN5
#endif // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#if defined(__gfx803__)
#define GCN4
#endif // defined(__gfx803__)

#if defined(GCN5) || defined(GCN4)
#define GCN
#endif // defined(GCN5) || defined(GCN4)

#if defined(__gfx950__)
#define CDNA4
#endif // defined(__gfx950__)

#if defined(__gfx942__)
#define CDNA3
#endif // defined(__gfx942__)

#if defined(__gfx90a__)
#define CDNA2
#endif // defined(__gfx90a__)

#if defined(__gfx908__)
#define CDNA1
#endif // defined(__gfx908__)

#if defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#define CDNA // For the entire family
#endif // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#if defined(__GFX12__)
#define RDNA4
#endif // defined(__GFX12__)

#if defined(__GFX11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define RDNA3
#endif

#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define RDNA3_5
#endif // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#if defined(RDNA3) && !defined(RDNA3_5)
#define RDNA3_0
#endif // defined(RDNA3) && !defined(RDNA3_5)

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#if defined(__gfx1010__) || defined(__gfx1012__)
#define RDNA1
#endif // defined(__gfx1010__) || defined(__gfx1012__)

#if defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#define RDNA // For the entire family
#endif // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

#if HIP_VERSION >= 60300000
#include <hip/hip_fp8.h>
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
#define FP8_AVAILABLE
#endif // HIP_VERSION >= 60300000

// hipBLAS spells a half as an unsigned short, so the pointers need a cast the cuBLAS call does not
// (ik_llama #2339: needed by ggml-cuda/dsa_attn.cu's cublasHgemmStridedBatched on HIP)
static inline hipblasStatus_t ggml_cuda_hgemm_strided_batched(
        hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k,
        const half * alpha, const half * AP, int lda, long long strideA,
        const half * BP, int ldb, long long strideB,
        const half * beta, half * CP, int ldc, long long strideC, int batchCount) {
    return hipblasHgemmStridedBatched(handle, transA, transB, m, n, k,
            (const hipblasHalf *)alpha, (const hipblasHalf *)AP, lda, strideA,
            (const hipblasHalf *)BP, ldb, strideB,
            (const hipblasHalf *)beta, (hipblasHalf *)CP, ldc, strideC, batchCount);
}
#define cublasHgemmStridedBatched ggml_cuda_hgemm_strided_batched

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    // do some small modifications to a and b to make the subtraction not underflow
    const unsigned int a_large = a | 0x80808080;
    const unsigned int b_small = b & 0x7f7f7f7f;
    const unsigned int result_low_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    const unsigned int should_flip_high_1bit = (a ^ ~b) & 0x80808080;

    return result_low_7bits ^ should_flip_high_1bit;
}

static __device__ __forceinline__ unsigned int __vsub2(const unsigned int a, const unsigned int b) {
    const unsigned int lo = (a - b) & 0x0000ffffu;
    const unsigned int hi = ((a & 0xffff0000u) - (b & 0xffff0000u)) & 0xffff0000u;
    return hi | lo;
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const unsigned int x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    const unsigned int ne_low_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    const unsigned int ne_high_1bit = x & 0x80808080;
    const unsigned int ne_any_bit = ne_low_7bits | ne_high_1bit;

    return (ne_any_bit >> 7) * 0xff;
}

// v_perm_b32 with selector bytes in [0, 7]: 0-3 pick bytes of lo, 4-7 pick bytes of hi.
static __device__ __forceinline__ uint32_t ggml_hip_perm(const uint32_t hi, const uint32_t lo, const uint32_t sel) {
#if __has_builtin(__builtin_amdgcn_is_invocable)
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_perm)) {
        return __builtin_amdgcn_perm(hi, lo, sel);
    }
    const uint64_t src = ((uint64_t) hi << 32) | lo;
    uint32_t res = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t byte_sel = (sel >> (8 * i)) & 0x07;
        res |= (uint32_t) ((src >> (8 * byte_sel)) & 0xFF) << (8 * i);
    }
    return res;
#else
    return __builtin_amdgcn_perm(hi, lo, sel);
#endif // __has_builtin(__builtin_amdgcn_is_invocable)
}

#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 50600000
// __shfl_xor() for half2 was added in ROCm 5.6
static __device__ __forceinline__ half2 __shfl_xor(half2 var, int laneMask, int width) {
    typedef union half2_b32 {
        half2 val;
        int   b32;
    } half2_b32_t;
    half2_b32_t tmp;
    tmp.val = var;
    tmp.b32 = __shfl_xor(tmp.b32, laneMask, width);
    return tmp.val;
}
#endif // defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 50600000
