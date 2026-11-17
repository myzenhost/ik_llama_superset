// Simplified API for asynchronous data loading.

#pragma once

#include "common.cuh"

static __device__ __forceinline__ unsigned int ggml_cuda_cvta_generic_to_shared(void * generic_ptr) {
#ifdef CP_ASYNC_AVAILABLE
    return __cvta_generic_to_shared(generic_ptr);
#else
    GGML_UNUSED(generic_ptr);
    NO_DEVICE_CODE;
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// Copies data from global to shared memory, cg == cache global.
// Both the src and dst pointers must be aligned to 16 bit.
// Shared memory uses 32 bit addressing, the pointer is passed as unsigned int.
// Generic pointers can be converted to 32 bit shared memory pointers using __cvta_generic_to_shared.
// Only the 16 bit copy is exposed because 4 and 8 bit copies did not yield performance improvements.
template <int preload>
static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256, "bad preload");
#ifdef CP_ASYNC_AVAILABLE
#if CUDART_VERSION >= 11040
    if (preload == 256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else
#endif // CUDART_VERSION >= 11040
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    }
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Makes each thread wait until its asynchronous data copies are done.
// This does NOT provide any additional synchronization.
// In particular, when copying data with multiple warps a call to __syncthreads will be needed.
static __device__ __forceinline__ void cp_async_wait_all() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_all;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// ---------------------------------------------------------------------------------------------------------------
// Bulk asynchronous copies (1D TMA), sm_90+. One thread issues a copy of a whole contiguous span; completion is
// counted in BYTES on an mbarrier that lives in shared memory, so the waiting threads may differ from the issuing
// thread and each consumer waits on the barrier instead of a per-thread cp.async.wait.
//
// Protocol for one barrier, one phase:
//   1. once per block: tma_mbar_init(bar, 1) by one thread, then tma_mbar_init_fence() + __syncthreads().
//   2. per phase: ONE thread calls tma_mbar_arrive_expect_tx(bar, total_bytes) (the single expected arrival),
//      any threads issue tma_bulk_g2s(dst, src, bytes, bar) whose bytes sum to total_bytes.
//   3. every consumer calls tma_mbar_wait(bar, phase) and then flips its phase bit (phase ^= 1).
// If the destination shared memory was READ through ordinary loads earlier, the block must __syncthreads() and
// the issuing thread must call tma_fence_proxy_async() before the next tma_bulk_g2s into it (generic-proxy
// reads vs async-proxy writes).
// Requirements (PTX ISA, cp.async.bulk): dst and src 16-byte aligned, bytes a non-zero multiple of 16.
// tma_bulk_ok() checks those at run time so callers can fall back to cp_async_cg_16.
// ---------------------------------------------------------------------------------------------------------------

static __device__ __forceinline__ bool tma_bulk_ok(const void * src, const unsigned int dst, const unsigned int bytes) {
    return bytes != 0 && bytes % 16 == 0 && (reinterpret_cast<uintptr_t>(src) % 16) == 0 && dst % 16 == 0;
}

static __device__ __forceinline__ void tma_mbar_init(const unsigned int bar, const unsigned int count) {
#ifdef TMA_BULK_AVAILABLE
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : : "r"(bar), "r"(count) : "memory");
#else
    GGML_UNUSED(bar); GGML_UNUSED(count);
    NO_DEVICE_CODE;
#endif // TMA_BULK_AVAILABLE
}

// Makes the mbarrier initialisation visible to the async proxy (required before the first complete_tx).
static __device__ __forceinline__ void tma_mbar_init_fence() {
#ifdef TMA_BULK_AVAILABLE
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
#else
    NO_DEVICE_CODE;
#endif // TMA_BULK_AVAILABLE
}

static __device__ __forceinline__ void tma_fence_proxy_async() {
#ifdef TMA_BULK_AVAILABLE
    asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
#else
    NO_DEVICE_CODE;
#endif // TMA_BULK_AVAILABLE
}

static __device__ __forceinline__ void tma_mbar_arrive_expect_tx(const unsigned int bar, const unsigned int bytes) {
#ifdef TMA_BULK_AVAILABLE
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" : : "r"(bar), "r"(bytes) : "memory");
#else
    GGML_UNUSED(bar); GGML_UNUSED(bytes);
    NO_DEVICE_CODE;
#endif // TMA_BULK_AVAILABLE
}

static __device__ __forceinline__ void tma_bulk_g2s(const unsigned int dst, const void * src, const unsigned int bytes,
                                                    const unsigned int bar) {
#ifdef TMA_BULK_AVAILABLE
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        : : "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
#else
    GGML_UNUSED(dst); GGML_UNUSED(src); GGML_UNUSED(bytes); GGML_UNUSED(bar);
    NO_DEVICE_CODE;
#endif // TMA_BULK_AVAILABLE
}

// Spins until the barrier's phase with the given parity has completed.
static __device__ __forceinline__ void tma_mbar_wait(const unsigned int bar, const unsigned int phase) {
#ifdef TMA_BULK_AVAILABLE
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "TMA_WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra TMA_WAIT_%=;\n"
        "}\n" : : "r"(bar), "r"(phase) : "memory");
#else
    GGML_UNUSED(bar); GGML_UNUSED(phase);
    NO_DEVICE_CODE;
#endif // TMA_BULK_AVAILABLE
}
