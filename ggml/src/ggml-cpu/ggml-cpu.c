#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "ggml-cpu-isa-guard.h"
#include "ggml-cpu.h"   // CPU-side declarations (ggml_graph_plan/compute, ggml_cpu_has_*, ggml_get/set_*) moved here, per upstream
#include "common.h"
// The fork's own quantisers and vec_dots -- quantize_row_iq2_k, vec_dot_iq2_k_q8_k and ~100 more.
// quants.h above declares only the upstream set, which is why type_traits_cpu could not name the
// fork's types: the names were not in scope here, so the table silently had no entry for them and
// every one of those types yielded a NULL vec_dot at run time. ggml.c reaches the same
// declarations for its own table; this file needs them for the same reason.
#include "../iqk/iqk_quantize.h"
// ...and ggml/src/ggml-quants.h, which is NOT the "quants.h" included above. Two headers with
// nearly the same name sit on either side of the base/cpu boundary: ggml-cpu/quants.h declares
// the CPU backend's own set, while ggml/src/ggml-quants.h declares the fork's Q6_0 and IQ1_BN
// entry points (quantize_row_q6_0, ggml_vec_dot_q6_0_q8_0, quantize_row_iq1_bn,
// ggml_vec_dot_iq1_bn_q8_K64). Including only the first left exactly those four undeclared.
#include "../ggml-quants.h"
// runtime AVX2 scale search for the ggml-base IQK quantizers (registered in ggml_cpu_init)
#include "../iqk/iqk_scale_search.h"

#if GGML_USE_IQK_MULMAT
// iqk's tiled GEMM entry points (iqk_mul_mat_4d). The pre-split ggml.c mul_mat handed every
// eligible matrix to it first; the split CPU backend never did, so this fork's main CPU speed path
// was reachable only one dot product at a time through the per-type vec_dot wrappers.
#include "../iqk/iqk_mul_mat.h"

// Mirror of ggml.c's runtime kill-switch (static there, so not linkable from here). Same variable,
// same semantics: GGML_NO_IQK_GEMM=1 makes mul_mat skip the iqk GEMM and take the generic path,
// on the same binary -- which is what makes an iqk-vs-generic A/B measurable.
// DISABLE-REVERSIBLE: GGML_NO_IQK_GEMM (default unset = iqk GEMM ON).
static bool ggml_cpu_iqk_gemm_disabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * e = getenv("GGML_NO_IQK_GEMM");
        cached = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}
#endif
#include "tiled/tiled.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h> // _mm_getcsr/_mm_setcsr for the opt-in flush-denormals mode
#endif
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"

// Same-binary A/B switch for llamafile's tinyBLAS, shaped like GGML_NO_IQK_GEMM above: with
// GGML_NO_LLAMAFILE=1 mul_mat skips both llamafile_sgemm attempts and runs the per-row vec_dot.
// Added to bisect S69-FORK-GENERIC-Q8-0-PATH-DEVIATES-0-47PCT-FROM-EXACT-2026-09-26-01, where the
// generic path's kernel could only be inferred, not observed.
// DISABLE-REVERSIBLE: GGML_NO_LLAMAFILE (default unset = llamafile ON, unchanged behaviour).
static bool ggml_cpu_llamafile_disabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * e = getenv("GGML_NO_LLAMAFILE");
        cached = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

// Signed FP8 needs its own table: UE4M3 ignores the sign, halves values for NVFP4, and maps NaN to zero.
float ggml_table_f32_f8_e4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

// F32 scratch rows reserved PER THREAD for converting a non-F32, non-vec_dot_type src1 (e.g. an F16 KV view) to vec_dot_type.
// iqk_quantize_any dequantizes one whole row-interleaved group (num_rows(vec_dot_type): 8, or 16 with AVX-512) into it at once,
// so one row per thread overran the next thread's slot and the end of wdata (heap corruption, test-mul-mat-f16-src1 n=8).
#define GGML_CPU_SRC1_SCRATCH_ROWS 16

// TurboQuant KV types: quantisers / dequantisers live in ggml-turbo-quant.c (libggml-base); the vec_dots below are defined after the table.
void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k);
void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k);
static void ggml_vec_dot_turbo3_0_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
static void ggml_vec_dot_turbo2_0_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
static void ggml_vec_dot_turbo4_0_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
#if defined(__x86_64__) && (defined(__AVX2__) || defined(__AVX__))
        // the CPU backend's AVX quantiser, not ggml-base's scalar one -- see arch/x86/quants.c
        .from_float               = quantize_row_q8_0_cpu,
#else
        .from_float               = quantize_row_q8_0,
#endif
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_F8_E4M3] = {
        .from_float               = quantize_row_f8_e4m3,
        .vec_dot                  = ggml_vec_dot_f8_e4m3_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    // ik 30903876b: NVFP4 blocks + fp32 row scale. vec_dot_nvfp4_ks_q8_0 (iqk_quantize.cpp) routes to the iqk kernels.
    [GGML_TYPE_NVFP4_KS] = {
        .from_float               = quantize_row_nvfp4_ks,
        .vec_dot                  = vec_dot_nvfp4_ks_q8_0,
#if GGML_USE_IQK_MULMAT
#if defined __AVX2__
        .vec_dot_type             = GGML_TYPE_Q8_2_X4,
#else
        .vec_dot_type             = GGML_TYPE_Q8_0_X4,
#endif
#else
        .vec_dot_type             = GGML_TYPE_Q8_0,
#endif
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_XS] = {
        .from_float               = NULL,
#if defined(__AVX2__)
        .vec_dot                  = ggml_vec_dot_iq1_xs_q8_K_avx2,
#else
        .vec_dot                  = ggml_vec_dot_iq1_xs_q8_K,
#endif
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_XXS] = {
        .from_float               = NULL,
#if defined(__AVX2__)
        .vec_dot                  = ggml_vec_dot_iq1_xxs_q8_K_avx2,
#else
        .vec_dot                  = ggml_vec_dot_iq1_xxs_q8_K,
#endif
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_XXXS] = {
        .from_float               = NULL,
#if defined(__AVX2__)
        .vec_dot                  = ggml_vec_dot_iq1_xxxs_q8_K_avx2,
#else
        .vec_dot                  = ggml_vec_dot_iq1_xxxs_q8_K,
#endif
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
#if defined(__AVX2__)
        // the plain name links to ggml-base, compiled without -mavx2 = the scalar loop (as TQ2_0 below)
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K_avx2,
#else
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
#endif
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP6_E2M3] = {
        .from_float               = quantize_row_mxfp6_e2m3,
        .vec_dot                  = ggml_vec_dot_mxfp6_e2m3_q8_0_cpu,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP6_E3M2] = {
        .from_float               = quantize_row_mxfp6_e3m2,
        .vec_dot                  = ggml_vec_dot_mxfp6_e3m2_q8_0_cpu,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP8_E4M3] = {
        .from_float               = quantize_row_mxfp8_e4m3,
        .vec_dot                  = ggml_vec_dot_mxfp8_e4m3_q8_0_cpu,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP8_E5M2] = {
        .from_float               = quantize_row_mxfp8_e5m2,
        .vec_dot                  = ggml_vec_dot_mxfp8_e5m2_q8_0_cpu,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ3_1S] = {
        .from_float               = quantize_row_tq3_1s,
        .vec_dot                  = ggml_vec_dot_tq3_1s_q8_0_cpu,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ4_1S] = {
        .from_float               = quantize_row_tq4_1s,
        .vec_dot                  = ggml_vec_dot_tq4_1s_q8_0_cpu,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
#if defined(__AVX2__)
        // the plain name links to ggml-base, compiled without -mavx2 = the scalar loop
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K_avx2,
#else
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
#endif
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },

    // ---------------------------------------------------------------------------
    // RESTORED 2026-09-15: the fork's own quant types, lost in the per-backend merge.
    //
    // MEASURED: an IQ2_K model on CPU segfaulted here. ggml-cpu.c:1182 reads
    // type_traits_cpu[type].vec_dot with no null check, and this table had 31
    // designated entries against ggml.c's base table's 99 -- so every type the merge
    // left out yielded a NULL function pointer that was then called. Worse,
    // ggml_backend_cpu_device_supports_op returns true for them regardless, because
    // its MUL_MAT case short-circuits on `src1->type == GGML_TYPE_F32` before
    // vec_dot_type is ever consulted -- so the backend advertised support for types it
    // could not compute. (The same function DOES guard GGML_OP_CPY against seven IQ
    // types, commented 'missing type_traits.from_float'; the shape was known here.)
    //
    // GENERATED, NOT TYPED, from ggml.c's own type_traits[] -- every .vec_dot is
    // carried with the .vec_dot_type it is paired with THERE. That pairing is the part
    // a human retyping 51 entries gets silently wrong, and a mismatched pair produces
    // wrong numbers rather than a crash. Generator: temp/_gen_cpu_traits_entries.py.
    //
    // 19 further types were deliberately NOT given entries because ggml.c's table gives
    // them no vec_dot either: the integer types, the Q8_*_X4 / Q8_K* intermediates, the
    // I2_S (the TURBO types got a row on 2026-10-07: from_float + an f32 vec_dot, see below). An entry
    // without a vec_dot would be worse than none -- it replaces a loud missing case with a quiet NULL.
    // ---------------------------------------------------------------------------
    // TurboQuant KV rows. The stored blocks are in the ROTATED basis (centroid * norm, no inverse rotation), so a K row's dot product is
    // only meaningful against a Q that was rotated with GGML_OP_TURBO_WHT (the graph does that, see llama-turbo-graph.h). vec_dot_type
    // is F32: the flash-attention kernel hands the f32 Q row straight to the vec_dot, which dequantises K in L1-sized chunks.
    [GGML_TYPE_TURBO3_0] = {
        .from_float               = (ggml_from_float_t) quantize_row_turbo3_0_ref,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_turbo3_0_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_TURBO2_0] = {
        .from_float               = (ggml_from_float_t) quantize_row_turbo2_0_ref,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_turbo2_0_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_TURBO4_0] = {
        .from_float               = (ggml_from_float_t) quantize_row_turbo4_0_ref,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_turbo4_0_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_0] = {
        .from_float               = quantize_row_q6_0,
        .vec_dot                  = ggml_vec_dot_q6_0_q8_0,
        // Q8_0, not Q8_0_X4 (FIXED 2026-09-26): the vec_dot's fallback reads plain block_q8_0,
        // exactly as Q5_0's does. iqk takes neither on x86 (it wants Q8_2_X4 for legacy quants).
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K_R4] = {
        .from_float               = quantize_row_q2_k_r4,
        .vec_dot                  = vec_dot_q2_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K_R4] = {
        .from_float               = quantize_row_q3_k_r4,
        .vec_dot                  = vec_dot_q3_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K_R4] = {
        .from_float               = quantize_row_q4_k_r4,
        .vec_dot                  = vec_dot_q4_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_K_R4] = {
        .from_float               = quantize_row_q5_k_r4,
        .vec_dot                  = vec_dot_q5_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K32,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K_R4] = {
        .from_float               = quantize_row_q6_k_r4,
        .vec_dot                  = vec_dot_q6_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K_R8] = {
        .from_float               = quantize_row_q8_k_r8,
        .vec_dot                  = vec_dot_q8_k_r8_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K_R16] = {
        .from_float               = quantize_row_q8_k_r16,
        .vec_dot                  = vec_dot_q8_k_r16_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XXS_R4] = {
        .from_float               = quantize_row_iq2_xxs_r4,
        .vec_dot                  = vec_dot_iq2_xxs_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS_R4] = {
        .from_float               = quantize_row_iq2_xs_r4,
        .vec_dot                  = vec_dot_iq2_xs_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS_R4] = {
        .from_float               = quantize_row_iq3_xxs_r4,
        .vec_dot                  = vec_dot_iq3_xxs_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S_R4] = {
        .from_float               = quantize_row_iq3_s_r4,
        .vec_dot                  = vec_dot_iq3_s_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S_R4] = {
        .from_float               = quantize_row_iq2_s_r4,
        .vec_dot                  = vec_dot_iq2_s_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S_R4] = {
        .from_float               = quantize_row_iq1_s_r4,
        .vec_dot                  = vec_dot_iq1_s_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M_R4] = {
        .from_float               = quantize_row_iq1_m_r4,
        .vec_dot                  = vec_dot_iq1_m_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_BN] = {
        .from_float               = quantize_row_iq1_bn,
        .vec_dot                  = ggml_vec_dot_iq1_bn_q8_K64,
        .vec_dot_type             = GGML_TYPE_Q8_K64,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_BN] = {
        .from_float               = quantize_row_iq2_bn,
        .vec_dot                  = vec_dot_iq2_bn_q8_K64,
        .vec_dot_type             = GGML_TYPE_Q8_K64,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_BN_R4] = {
        .from_float               = quantize_row_iq2_bn_r4,
        .vec_dot                  = vec_dot_iq2_bn_r4_q8_K64,
        .vec_dot_type             = GGML_TYPE_Q8_K64,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4_R8] = {
        .from_float               = quantize_row_mxfp4_r8,
        .vec_dot                  = vec_dot_mxfp4_r8_q8_2_x4,
        .vec_dot_type             = GGML_TYPE_Q8_2_X4,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_KS] = {
        .from_float               = quantize_row_iq4_ks,
        .vec_dot                  = vec_dot_iq4_ks_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_KS_R4] = {
        .from_float               = quantize_row_iq4_ks_r4,
        .vec_dot                  = vec_dot_iq4_ks_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ5_KS_R4] = {
        .from_float               = quantize_row_iq5_ks_r4,
        .vec_dot                  = vec_dot_iq5_ks_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_KSS] = {
        .from_float               = quantize_row_iq4_kss,
        .vec_dot                  = vec_dot_iq4_kss_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ5_KS] = {
        .from_float               = quantize_row_iq5_ks,
        .vec_dot                  = vec_dot_iq5_ks_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_KV] = {
        .from_float               = quantize_row_q8_KV,
        .vec_dot                  = vec_dot_q8_KV_q8_KV,
        .vec_dot_type             = GGML_TYPE_Q8_KV,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_KV_R8] = {
        .from_float               = quantize_row_q8_KV_r8,
        .vec_dot                  = vec_dot_q8_KV_r8_q8_KV,
        .vec_dot_type             = GGML_TYPE_Q8_KV,
        .nrows                    = 1,
    },
    [GGML_TYPE_BF16_R16] = {
        .from_float               = (ggml_from_float_t) ggml_fp32_to_bf16_row,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_K] = {
        .from_float               = quantize_row_iq2_k,
        .vec_dot                  = vec_dot_iq2_k_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_K_R4] = {
        .from_float               = quantize_row_iq2_k_r4,
        .vec_dot                  = vec_dot_iq2_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_KS] = {
        .from_float               = quantize_row_iq2_ks,
        .vec_dot                  = vec_dot_iq2_ks_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_KT] = {
        .from_float               = quantize_row_iq1_kt,
        .vec_dot                  = vec_dot_iq1_kt_q8_k,
        .vec_dot_type             = IQK_KT_ACTIVATION_TYPE,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_KT] = {
        .from_float               = quantize_row_iq2_kt,
        .vec_dot                  = vec_dot_iq2_kt_q8_k,
        .vec_dot_type             = IQK_KT_ACTIVATION_TYPE,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_KT] = {
        .from_float               = quantize_row_iq3_kt,
        .vec_dot                  = vec_dot_iq3_kt_q8_k,
        .vec_dot_type             = IQK_KT_ACTIVATION_TYPE,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_KT] = {
        .from_float               = quantize_row_iq4_kt,
        .vec_dot                  = vec_dot_iq4_kt_q8_k,
        .vec_dot_type             = IQK_KT_ACTIVATION_TYPE,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0_G128] = {
        .from_float               = quantize_row_q1_0_g128,
        .vec_dot                  = vec_dot_q1_0_g128_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0_X4,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0_G128] = {
        .from_float               = quantize_row_q2_0_g128,
        .vec_dot                  = vec_dot_q2_0_g128_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0_X4,
        .nrows                    = 1,
    },
    // ik_llama #2541. ik keeps ONE traits table in ggml.c; this tree keeps the CPU half here, and the merge left the
    // type with no vec_dot / vec_dot_type / from_float on the CPU side (a mul_mat on it crashed, MEASURED by a scratch
    // control: access violation in the R8 graph, plain Q1_0_G128 fine).
    [GGML_TYPE_Q1_0_G128_R8] = {
        .from_float               = quantize_row_q1_0_g128_r8,
        .vec_dot                  = vec_dot_q1_0_g128_r8_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K128,
        .nrows                    = 1,
    },
    // ik_llama #2550 (Prism ternary), same omission and same reason as Q1_0_G128_R8 above: the merge brought the base-table
    // rows in ggml.c and left these four CPU-side rows out (caught by the fork-lint cpu-traits-parity check at commit time).
    // vec_dot / vec_dot_type mirror ik's ggml.c rows; the R8 types dot through iqk_mul_mat on a Q8_K128 activation.
    [GGML_TYPE_PQ2_0] = {
        .from_float               = quantize_row_pq2_0,
        // q8_0 (32-element activation blocks), not ik's q8_K: tests/test-pq2-activation-noise.cpp measured the q8_K route's
        // activation noise at 2.5x the q8_0 route on outlier-bearing activations with an exact kernel decode (KLD 0.0172 vs the
        // 0.0037 of PTQ1_0, which already uses q8_0). Divergence from ik on purpose.
        // the AVX2 copy built in this backend (quants.c); the ggml-quants.c one is in ggml-base, built without AVX2
        .vec_dot                  = ggml_cpu_vec_dot_pq2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_PTQ1_0] = {
        .from_float               = quantize_row_ptq1_0,
        .vec_dot                  = ggml_vec_dot_ptq1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_PQ2_0_R8] = {
        .from_float               = quantize_row_pq2_0_r8,
        .vec_dot                  = vec_dot_pq2_0_r8_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K128,
        .nrows                    = 1,
    },
    [GGML_TYPE_PTQ1_0_R8] = {
        .from_float               = quantize_row_ptq1_0_r8,
        .vec_dot                  = vec_dot_ptq1_0_r8_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K128,
        .nrows                    = 1,
    },
    // ik_llama #2480 (slice S16), same omission and same reason.
    [GGML_TYPE_IQ4_KS_R16] = {
        .from_float               = quantize_row_iq4_ks_r16,
        .vec_dot                  = vec_dot_iq4_ks_r16_q8_0,
#if defined __AVX2__
        .vec_dot_type             = GGML_TYPE_Q8_2_X4,
#else
        .vec_dot_type             = GGML_TYPE_Q8_0_X4,
#endif
        .nrows                    = 1,
    },
    // Gyro trellis types (ik df93e4cc9 #2812), decode-only: from_float NULL as in ggml.c. TQ2_T's vec_dot_type must equal
    // what iqk_set_kernels_ktquants accepts (Q8_2_X4 under IQK + AVX2, ggml.c type_traits); the others use the generic
    // Q8_0 vec_dot (ggml-quants.c).
    [GGML_TYPE_TQ2_T] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_tq2_t_q8_0,
#if GGML_USE_IQK_MULMAT && defined __AVX2__
        .vec_dot_type             = GGML_TYPE_Q8_2_X4,
#else
        .vec_dot_type             = GGML_TYPE_Q8_0,
#endif
        .nrows                    = 1,
    },
    [GGML_TYPE_TQK6] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_tqk6_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQK7] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_tqk7_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQP5] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_tqp5_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQP6] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_tqp6_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQP7] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_tqp7_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    // ik 1b18524bd (#2703). Must equal iqk_set_kernels_legacy_quants expected_typeB for IQ5_KS_R16
    // (iqk_gemm_legacy_quants.cpp): Q8_K with HAVE_FANCY_SIMD (AVX512 VNNI/VL/BW/DQ, iqk_config.h), else Q8_2_X4
    // on AVX2, Q8_0_X4 elsewhere (NEON). ik's ggml.c spells the first branch #ifdef HAVE_FANCY_SIMD, which ggml.c
    // never defines, so here the condition is written out the way IQ3_KS_R16 below does.
    [GGML_TYPE_IQ5_KS_R16] = {
        .from_float               = quantize_row_iq5_ks_r16,
        .vec_dot                  = vec_dot_iq5_ks_r16_q8_0,
#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
        .vec_dot_type             = GGML_TYPE_Q8_K,
#elif defined __AVX2__
        .vec_dot_type             = GGML_TYPE_Q8_2_X4,
#else
        .vec_dot_type             = GGML_TYPE_Q8_0_X4,
#endif
        .nrows                    = 1,
    },
    // ik_llama #2622: must equal iqk_gemm_legacy_quants.cpp expected_typeB for IQ3_KS_R16
    // (HAVE_FANCY_SIMD ? Q8_2_X4 : Q8_0_X4); HAVE_FANCY_SIMD's condition from iqk_config.h.
    [GGML_TYPE_IQ3_KS_R16] = {
        .from_float               = quantize_row_iq3ks_r16,
        .vec_dot                  = vec_dot_iq3ks_r16_q8_2,
#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
        .vec_dot_type             = GGML_TYPE_Q8_2_X4,
#else
        .vec_dot_type             = GGML_TYPE_Q8_0_X4,
#endif
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_K] = {
        .from_float               = quantize_row_iq3_k,
        .vec_dot                  = vec_dot_iq3_k_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_KS] = {
        .from_float               = quantize_row_iq3_ks,
        .vec_dot                  = vec_dot_iq3_ks_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_KL] = {
        .from_float               = quantize_row_iq2_kl,
        .vec_dot                  = vec_dot_iq2_kl_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_K] = {
        .from_float               = quantize_row_iq4_k,
        .vec_dot                  = vec_dot_iq4_k_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_K_R4] = {
        .from_float               = quantize_row_iq4_k_r4,
        .vec_dot                  = vec_dot_iq4_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_K_R4] = {
        .from_float               = quantize_row_iq3_k_r4,
        .vec_dot                  = vec_dot_iq3_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ5_K] = {
        .from_float               = quantize_row_iq5_k,
        .vec_dot                  = vec_dot_iq5_k_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ5_K_R4] = {
        .from_float               = quantize_row_iq5_k_r4,
        .vec_dot                  = vec_dot_iq5_k_r4_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ6_K] = {
        .from_float               = quantize_row_iq6_k,
        .vec_dot                  = vec_dot_iq6_k_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL_R4] = {
        .from_float               = quantize_row_iq4_nl_r4,
        .vec_dot                  = vec_dot_iq4_nl_r4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS_R8] = {
        .from_float               = quantize_row_iq4_xs_r8,
        .vec_dot                  = vec_dot_iq4_xs_r8_q8_k,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0_R8] = {
        .from_float               = quantize_row_q4_0_r8,
        .vec_dot                  = vec_dot_q4_0_r8_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    // ik b9c47ba67 (#2785). Generic-path row mirrors Q4_0_R8 above; the iqk GEMM picks its own activation
    // (Q8_2_X4 on AVX2) through ggml_cpu_iqk_act_type / iqk_gemm_type_b, as for every other legacy _R8 type.
    [GGML_TYPE_Q2_0_R8] = {
        .from_float               = quantize_row_q2_0_r8,
        .vec_dot                  = vec_dot_q2_0_r8_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0_R8] = {
        .from_float               = quantize_row_q8_0_r8,
        .vec_dot                  = vec_dot_q8_0_r8_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_0_R4] = {
        .from_float               = quantize_row_q5_0_r4,
        .vec_dot                  = vec_dot_q5_0_r4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_0_R4] = {
        .from_float               = quantize_row_q6_0_r4,
        .vec_dot                  = vec_dot_q6_0_r4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },

    // ---- ACTIVATION TYPES. from_float ONLY, and that is the whole point of them. -------------
    // RESTORED 2026-09-15, second pass. These 5 are named as a .vec_dot_type by 21 of the weight
    // entries above and had no row here at all, so ggml-cpu.c:1616
    //     ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float;
    // loaded NULL and mul_mat called it. gdb: rip=0x0 inside ggml_compute_forward_mul_mat, on
    // iq2_kt, whose vec_dot_type is Q8_0_X4.
    //
    // They were omitted by the generator that built the block above, which skipped any type ggml.c
    // gives no .vec_dot. That rule is correct for a WEIGHT type and backwards for an ACTIVATION
    // type: these are never the weight side of a matmul, so they have no vec_dot by construction,
    // and skipping them on that ground created exactly the silent NULL the rule existed to
    // prevent -- one struct member over.
    //
    // from_float and NOTHING ELSE, modelled on this table's own [GGML_TYPE_Q8_K] row, which is
    // activation-only, already compiles here, and is why iq2_k (vec_dot_type Q8_K) passed the
    // sweep in the same run that iq2_kt crashed. Giving these a vec_dot or an nrows would be
    // inventing a weight-side capability they do not have.
    // Q8_2_X4 joined this list only after the four *_KT types were repointed at it, which is
    // itself the argument for the checker that found it: temp/_scope_vecdottype_from_float_gap.py
    // reported it within seconds of the repoint, naming all 5 weight types that now depend on
    // it. Changing a .vec_dot_type can create a NEW null here, so that check belongs after
    // every such change, not once.
    [GGML_TYPE_Q8_2_X4] = {
        .from_float               = quantize_row_q8_2_x4,   // IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT, MXFP4_R8
    },
    [GGML_TYPE_Q8_0_X4] = {
        .from_float               = quantize_row_q8_0_x4,   // 13 weight types, every *_KT among them
    },
    [GGML_TYPE_Q8_K16] = {
        .from_float               = quantize_row_q8_K16,   // IQ2_BN_R4
    },
    [GGML_TYPE_Q8_K32] = {
        .from_float               = quantize_row_q8_K32,   // IQ4_XS_R8, Q4_K_R4, Q5_K_R4
    },
    [GGML_TYPE_Q8_K64] = {
        .from_float               = quantize_row_q8_K64,   // IQ1_BN, IQ2_BN
    },
    [GGML_TYPE_Q8_K128] = {
        .from_float               = quantize_row_q8_K128,   // IQ1_M_R4, IQ1_S_R4
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

#if GGML_USE_IQK_MULMAT
// The activation type the iqk GEMM reads for weight type `type`, which is NOT always
// type_traits_cpu[type].vec_dot_type: that field names what the GENERIC vec_dot reads (Q8_K for
// Q4_K/Q5_K/Q6_K, Q8_0 for the legacy quants), while iqk's x86 kernels for those read Q8_2_X4.
// MEASURED 2026-09-26: with vec_dot_type passed, iqk declined every K-quant GEMM -- a Q4_K_M
// qwen3moe ran 114.6 s per 512-token pass against upstream's 8.02, every worker in
// ggml_vec_dot_q4_K_q8_K. GGML_TYPE_COUNT = iqk takes no quantized activation for this type;
// callers then quantize to vec_dot_type as before.
static enum ggml_type ggml_cpu_iqk_act_type(enum ggml_type type, int64_t ne00) {
    // float weights keep their own activation type: nothing to probe, and a lax selector could
    // otherwise pair them with a quantized activation they do not read
    if (!ggml_is_quantized(type)) {
        return GGML_TYPE_COUNT;
    }
    const int t = iqk_gemm_type_b((int) type, (long) ne00, (int) type_traits_cpu[type].vec_dot_type);
    if (t < 0 || t >= GGML_TYPE_COUNT || type_traits_cpu[t].from_float == NULL) {
        return GGML_TYPE_COUNT;
    }
    return (enum ggml_type) t;
}
#endif

// F32 src1 -> packed rows of type t, WHOLE ROWS per thread (the grouped activation types iqk reads
// pack blocks in fours, so a row must not be split across threads). Row (i11, i12, i13) lands at
// ((i13*ne12 + i12)*ne11 + i11)*ggml_row_size(t, ne10).
static void ggml_cpu_quantize_src1_rows(const struct ggml_tensor * src1, enum ggml_type t, char * wdata,
        int ith, int nth) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    ggml_from_float_t const from_float = type_traits_cpu[t].from_float;
    GGML_ASSERT(from_float != NULL);
    const int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    const size_t nbw1 = ggml_row_size(t, ne10);
    const size_t nbw2 = nbw1*ne11;
    const size_t nbw3 = nbw2*ne12;
    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((const float *)((const char *) src1->data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1]),
                           (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                           ne10);
            }
        }
    }
}

// Activation types whose quantizer packs blocks in FOURS from the start of the pointer it is given
// (quantize_row_q8_0_x4 and kin write y4[i/4].d[i%4] for the first 4*(nb/4) blocks and plain blocks for
// the tail). The generic src1 conversion below cuts each row at ARBITRARY block boundaries, one slice
// per thread; a slice that does not start on a multiple of four blocks gets the wrong group layout
// and a tail of plain blocks in the middle of the row. Such types are converted a whole row per thread.
// MEASURED 2026-10-16 (tests/test-bonsai-scalar-vecdot.cpp): Q1_0_G128 / Q2_0_G128 (vec_dot_type
// Q8_0_X4) through the generic path returned non-finite outputs at 2+ threads and were exact at 1.
static bool ggml_cpu_act_type_is_grouped(enum ggml_type t) {
    return t == GGML_TYPE_Q8_0_X4 || t == GGML_TYPE_Q8_1_X4 || t == GGML_TYPE_Q8_2_X4;
}

// Bytes of wdata the src1 copy needs when it may be held in EITHER the generic vec_dot_type or the
// iqk activation type (a declined iqk call re-quantizes in place).
static size_t ggml_cpu_src1_wsize(enum ggml_type src0_type, const struct ggml_tensor * src1) {
    const enum ggml_type vec_dot_type = type_traits_cpu[src0_type].vec_dot_type;
    size_t cur = src1->type != vec_dot_type ? ggml_row_size(vec_dot_type, ggml_nelements(src1)) : 0;
#if GGML_USE_IQK_MULMAT
    if (src1->type == GGML_TYPE_F32) {
        const enum ggml_type t = ggml_cpu_iqk_act_type(src0_type, src1->ne[0]);
        if (t != GGML_TYPE_COUNT) {
            cur = MAX(cur, ggml_row_size(t, ggml_nelements(src1)));
        }
    }
#endif
    return cur;
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
    bool src1_quantized = false;   // set when the iqk block below has already filled wdata
#if GGML_USE_IQK_MULMAT
    // IQK GEMM -- RESTORED 2026-09-26 from the pre-split ggml.c ggml_compute_forward_mul_mat, which
    // tried it before anything else (so it sits ahead of llamafile_sgemm here too). MEASURED before
    // this: Qwen3.8-27B CPU perplexity ran 13+ min without finishing one 512-token chunk, every
    // worker in ggml_vec_dot_iq3_xxs_q8_K.
    // Exclusions carried over VERBATIM with their reasons (see the long note at the old site in
    // ggml.c): iqk_mul_mat_4d returns TRUE and answers with garbage for TQ1_0 / TQ2_0 (and DID for
    // Q2_0_G128 in the authoring tree). Q2_0_G128 is no longer excluded HERE (dense MUL_MAT only):
    // iqk_gemm_1bit.cpp now carries a Q2_0_G128 x Q8_2_X4 AVX2 kernel, checked against the
    // dequantiser by tests/test-iqk-q2-0-g128.cpp. The MoE (moe_iqk_ok) and fused up/gate
    // (ggml_cpu_up_gate_iqk_ok) sites below stay excluded: nothing has exercised the type there.
    // The old "ne00 != 4" shape stopgap is GONE: its root cause (the float GEMM kernel reading a
    // full k_step before checking the row is that long) is fixed where it lives, in
    // iqk_set_kernels_float, which now declines ne00 < k_step for BOTH sites below -- the stopgap
    // only ever guarded site 1 (S69-FORK-IQK-MULMAT-NE00-4-ROOT-CAUSE-01).
    // NOT carried over: the old second site's in-op fusion of the FOLLOWING mul_mats that share src1
    // -- it returned a node index this executor cannot consume. A throughput item, not correctness.
    // TQ2_0 LEFT THIS LIST 2026-10-16: iqk now has a TQ2_0 x Q8_K GEMM on x86
    // (mul_mat_tq2_0_q8_K, iqk_gemm_1bit.cpp) and declines TQ2_0 itself where it has none
    // (k_iqk_has_tq2_0_kernel, iqk_mul_mat.cpp), which falls through to the vec_dot below.
    // Excluding it here ran one vec_dot per row x column: 121.7 vs 24.0 s/pass against IQ2_BN.
    // TQ1_0 LEFT IT 2026-10-17 on the same terms: mul_mat_tq1_0_q8_K (iqk_gemm_1bit.cpp) on x86,
    // k_iqk_has_tq1_0_kernel declining it elsewhere (row S69-TQ1-0-CPU-SCALAR-VEC-DOT-AND-NO-IQK-GEMM-2026-10-17-01).
    const bool iqk_ok = dst->type == GGML_TYPE_F32 && !ggml_cpu_iqk_gemm_disabled()
                     && ggml_iqk_supports_type(src0->type, GGML_IQK_SITE_DENSE);   // the per-site set (ggml.c)
    // site 1: src1 as it is (F32 activations against F32/F16/BF16 weights, and any pair iqk takes)
    if (iqk_ok) {
        if (iqk_mul_mat_4d(ne01, ne11, ne00,
                    ne02, ne03, ne12, ne13, nb02, nb03, nb12, nb13, nb2/sizeof(float), nb3/sizeof(float),
                    src0->type, src0->data, nb01,
                    src1->type, src1->data, nb11,
                    (float *)dst->data, nb1/sizeof(float), ith, nth)) {
            return;
        }
    }
    // site 2: quantize src1 into the type iqk READS (ggml_cpu_iqk_act_type -- Q8_2_X4 for the
    // K-quants and legacy quants on x86, not their generic vec_dot_type), then the GEMM. WHOLE ROWS
    // per thread, exactly as the old path did -- the generic block below splits each row's blocks
    // across threads, and the grouped activation types iqk reads pack blocks in fours.
    if (iqk_ok && src1->type != vec_dot_type && src1->type == GGML_TYPE_F32) {
        const enum ggml_type iqk_type = ggml_cpu_iqk_act_type(src0->type, ne00);
        const enum ggml_type act_type = iqk_type != GGML_TYPE_COUNT ? iqk_type : vec_dot_type;
        char * wdata = params->wdata;
        const size_t nbw1 = ggml_row_size(act_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;
        GGML_ASSERT(params->wsize >= ne13*nbw3);
        ggml_cpu_quantize_src1_rows(src1, act_type, wdata, ith, nth);
        ggml_barrier(params->threadpool);
        if (iqk_mul_mat_4d(ne01, ne11, ne00,
                    ne02, ne03, ne12, ne13, nb02, nb03, nbw2, nbw3, nb2/sizeof(float), nb3/sizeof(float),
                    src0->type, src0->data, nb01,
                    act_type, wdata, nbw1,
                    (float *)dst->data, nb1/sizeof(float), ith, nth)) {
            return;
        }
        // declined (before reading B, on every thread alike). If wdata holds vec_dot_type it is
        // already in the row layout the paths below read; otherwise the generic block re-quantizes
        // it, and nothing reads wdata until the barrier after that.
        src1_quantized = act_type == vec_dot_type;
    }
#endif

    // upstream tiled K-quant / IQ GEMM (tiled.h) executes the full op when it applies. It runs after the iqk GEMM above,
    // this fork's measured CPU speed path, so it takes only what iqk declined (or everything with GGML_NO_IQK_GEMM=1);
    // it converts src1 itself (wdata from offset 0, ggml_compute_forward_mul_mat_tiled_impl).
    if (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        return;
    }

#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (!params->use_ref && src1_cont && !ggml_cpu_llamafile_disabled()) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

#if GGML_USE_IQK_MULMAT
    // src1 neither F32 nor vec_dot_type -- e.g. an F16 KV-cache view against a quantized weight, which is
    // what the MLA prompt path builds (build_deepseek2: wkv_b x kv_cache_nope). The pre-split ggml.c
    // converted it through a per-thread F32 row (iqk_quantize_any); the split backend lost that branch,
    // so supports_op refused the node and the scheduler aborted (deepseek-v2 unloadable on CPU).
    // wdata = [src1 as vec_dot_type rows, the layout every path below reads][nth F32 scratch rows];
    // the planner reserves both (GGML_OP_MUL_MAT in ggml_graph_plan). The barrier below orders it.
    if (src1->type != vec_dot_type && src1->type != GGML_TYPE_F32 && !src1_quantized) {
        const size_t nbw3 = ggml_row_size(vec_dot_type, ne10)*ne11*ne12;
        char * work_buffer = (char *) params->wdata + ne13*nbw3 + ith*ne10*GGML_CPU_SRC1_SCRATCH_ROWS*sizeof(float);
        GGML_ASSERT(params->wsize >= ne13*nbw3 + nth*ne10*GGML_CPU_SRC1_SCRATCH_ROWS*sizeof(float));
        GGML_ASSERT(from_float != NULL && ggml_get_type_traits(src1->type)->to_float != NULL);
        iqk_quantize_any(src1->type, vec_dot_type, ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                src1->data, params->wdata, work_buffer, ggml_get_type_traits(src1->type)->to_float, from_float, ith, nth);
        src1_quantized = true;
    }
#endif

    if (src1->type != vec_dot_type && !src1_quantized) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        const bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        const int64_t bs = ggml_blck_size(vec_dot_type);

        if (ggml_cpu_act_type_is_grouped(vec_dot_type)) {
            ggml_cpu_quantize_src1_rows(src1, vec_dot_type, wdata, ith, nth);
        } else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    const void * src1_row  = (const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    void       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    const int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    if (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) src1_row, (float *) wdata_row, ne10_block_size);
                    } else if (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((const ggml_bf16_t *) src1_row, (float *) wdata_row, ne10_block_size);
                    } else {
                        from_float((const float *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (!params->use_ref && src1->type != vec_dot_type && !ggml_cpu_llamafile_disabled()) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    const size_t src1_wsize = ggml_cpu_src1_wsize(type, src1);
    if (src1_wsize > 0) {
        incr_ptr_aligned(&wdata_cur, src1_wsize, sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    char * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

#if GGML_USE_IQK_MULMAT
    // IQK MoE GEMM -- RESTORED 2026-09-26 from the pre-split ggml.c mul_mat_id (its per-expert
    // iqk_mul_mat_moe site), companion to the dense restore in ggml_compute_forward_mul_mat.
    // Same exclusions, same reasons: iqk claims TQ1_0 / TQ2_0 / Q2_0_G128 and answers with garbage.
    // (Q2_0_G128 now has a DENSE kernel -- see ggml_compute_forward_mul_mat -- but stays excluded
    // here: the MoE row-mapping path has not been exercised for it, and iqk_mul_mat_moe declines it.)
    // src1 must be addressable as contiguous rows of row_size, which is what iqk is handed.
    const bool moe_iqk_ok = ne13 == 1 && dst->type == GGML_TYPE_F32 && !ggml_cpu_iqk_gemm_disabled()
                         && (src1_cont || src1->type != vec_dot_type)
                         // Q2_0_G128 and Q2_0 (401) out: dense kernel only (test-iqk-q2-0*), MoE
                         // path unexercised. TQ1_0 / TQ2_0: declined callee-side, see the dense site.
                         && ggml_iqk_supports_type(type, GGML_IQK_SITE_MOE);
    // the type the iqk path reads (see ggml_cpu_iqk_act_type); what wdata holds is tracked in q_type
    const enum ggml_type iqk_type = moe_iqk_ok ? ggml_cpu_iqk_act_type(type, ne00) : GGML_TYPE_COUNT;
    const enum ggml_type act_type = iqk_type != GGML_TYPE_COUNT ? iqk_type : vec_dot_type;
#else
    const bool moe_iqk_ok = false;
    const enum ggml_type act_type = vec_dot_type;
#endif
    bool moe_use_iqk = moe_iqk_ok;
    enum ggml_type q_type = src1->type != vec_dot_type ? act_type : vec_dot_type;

    if (src1->type != vec_dot_type && moe_iqk_ok) {
        // WHOLE ROWS per thread, as the old path did -- iqk's grouped activation types pack blocks
        // in fours, and the block-split quantizer below cuts rows at arbitrary block boundaries.
        ggml_cpu_quantize_src1_rows(src1, act_type, params->wdata, ith, nth);
    } else if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        if (ggml_cpu_act_type_is_grouped(vec_dot_type)) {
            ggml_cpu_quantize_src1_rows(src1, vec_dot_type, wdata, ith, nth);
        } else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    // SER (smart expert reduction) marks unused expert slots with an out-of-range id. RESTORED
    // 2026-09-26 from the pre-split ggml.c: such a row is ZEROED (else stale values reach later
    // ops as NaNs) and skipped when grouping. This file had only assert(i02 >= 0 && i02 < n_as),
    // which release builds drop -- an id of -1 then wrote matrix_rows out of bounds.
    for (int64_t iid1 = ith; iid1 < ids->ne[1]; iid1 += nth) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
            if (i02 < 0 || i02 >= n_as) {
                memset((char *) dst->data + id*nb1 + iid1*nb2, 0, ne0*sizeof(float));
            }
        }
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                if (i02 < 0 || i02 >= n_as) {
                    continue;   // SER-disabled slot, zeroed above
                }

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

#if GGML_USE_IQK_MULMAT
        // iqk_mul_mat_moe partitions this expert's rows over ith/nth itself and reads the row
        // mapping {i1 = expert slot, i2 = token} -- the same struct this file groups into, at this
        // file's per-expert stride. It declines deterministically (the answer depends on the type
        // pair and this expert's row count, identical on every thread), so every thread takes the
        // same branch here and the barriers below pair up.
        if (moe_use_iqk) {
            if (iqk_mul_mat_moe(nr0, nr1, ne00, ne11,
                    src0->type, src0_cur, nb01,
                    q_type, (const char *) wdata, ggml_row_size(q_type, ne10),
                    (float *) dst->data, nb1, nb2,
                    matrix_rows + cur_a*ids->ne[0]*ids->ne[1], ith, nth)) {
                continue;
            }
            if (q_type != vec_dot_type) {
                // wdata holds the iqk activation type, which the generic vec_dot cannot read:
                // wait until no thread is still inside an earlier expert's iqk call, re-quantize
                // into vec_dot_type, and finish on the generic path
                ggml_barrier(params->threadpool);
                ggml_cpu_quantize_src1_rows(src1, vec_dot_type, params->wdata, ith, nth);
                ggml_barrier(params->threadpool);
                q_type = vec_dot_type;
                moe_use_iqk = false;
            }
        }
#endif
        // upstream tiled takes over if profitable for this expert (see tiled.h). After the iqk attempt above: iqk first, and
        // only here is wdata guaranteed to hold vec_dot_type rows, the layout tiled reads (it re-quantized on an iqk decline).
        if (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            continue;
        }

        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

// ggml_compute_forward_mul_mat_id_up_gate (MOE_FUSED_UP_GATE) and
// ggml_compute_forward_mul_mat_up_gate (FUSED_UP_GATE)
//
// PORTED 2026-09-26 from the pre-split ggml.c. ggml-cpu never had these cases, so every MoE graph
// built with the fused up/gate op (the default: cparams.fused_moe_up_gate) had no CPU kernel.
//
// dst = act(gate.x + gate_b) * (up.x + up_b), with op_params[0] = the unary op and op_params[1] =
// a clamp limit (0 = none). The iqk kernel is tried first, exactly as before. What is NEW: the old
// op ended `if (!iqk_moe_fused_up_gate(...)) GGML_ABORT(...)` -- it had no generic path, which is
// why the graph builder must refuse the fusion for TQ1_0/TQ2_0/Q2_0_G128. Here an iqk decline
// falls through to a vec_dot path with the same arithmetic (MulMat::mul_mat_up_gate_NxM:
// activation, then min(limit), up clamped to +-limit, or 1 + clamp(up, +-7) for SWIGLU_OAI), so a
// type iqk cannot pair with this backend's vec_dot_type is slower instead of fatal. That path is
// also what GGML_NO_IQK_GEMM=1 runs, which makes the two arms an A/B of one binary.
// NOT ported (throughput only): GGML_EXPERT_CHUNKING and the moe_expert_prefetch hook.

static inline float ggml_cpu_up_gate_combine(int op, float limit, float up, float gate) {
    float t;
    switch (op) {
        case GGML_UNARY_OP_SILU:
            t = gate/(1.0f + expf(-gate));
            break;
        case GGML_UNARY_OP_GELU:
            t = 0.5f*gate*(1.0f + tanhf(0.79788456080286535587989211986876f*gate*(1.0f + 0.044715f*gate*gate)));
            break;
        case GGML_UNARY_OP_RELU:
            t = gate > 0.0f ? gate : 0.0f;
            break;
        case GGML_UNARY_OP_SWIGLU_OAI:
            {
                const float x = MIN(gate, 7.0f);
                t = x/(1.0f + expf(-1.702f*x));
            } break;
        default:
            GGML_ABORT("up_gate: unsupported unary op %d", op);
    }
    if (limit > 1e-6f) {
        t = MIN(t, limit);
    }
    if (op == GGML_UNARY_OP_SWIGLU_OAI) {
        up = 1.0f + MAX(MIN(up, 7.0f), -7.0f);
    } else if (limit > 1e-6f) {
        up = MAX(-limit, MIN(limit, up));
    }
    return up*t;
}

// Generic fallback for one weight pair: nr0 output rows split over threads, nr1 activation rows.
// rows == NULL: activation row r is wdata + r*row_size and lands in dst + r*nb1 (dense).
// rows != NULL: row r is {i1 = expert slot, i2 = token}; activation row (i1 % ne11) + i2*ne11,
// destination i1*nb1 + i2*nb2 -- the addressing iqk's DataInfo uses for the same mapping.
static void ggml_cpu_up_gate_generic(
        int ith, int nth, int64_t nr0, int64_t nr1, int64_t ne00, int64_t ne11, int op, float limit,
        enum ggml_type type, const char * up, const char * gate, size_t nb01,
        const float * up_b, const float * gate_b,
        const char * wdata, size_t row_size,
        char * dst, size_t nb1, size_t nb2, const struct mmid_row_mapping * rows) {

    ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;
    GGML_ASSERT(vec_dot != NULL);

    const int64_t dr  = (nr0 + nth - 1)/nth;
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr0);

    for (int64_t r = 0; r < nr1; ++r) {
        const char * y;
        float * d;
        if (rows) {
            const int64_t i1 = rows[r].i1;
            const int64_t i2 = rows[r].i2;
            y = wdata + (i1 % ne11 + i2*ne11)*row_size;
            d = (float *) (dst + i1*nb1 + i2*nb2);
        } else {
            y = wdata + r*row_size;
            d = (float *) (dst + r*nb1);
        }
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            float u, g;
            vec_dot(ne00, &u, 0, up   + ir*nb01, 0, y, 0, 1);
            vec_dot(ne00, &g, 0, gate + ir*nb01, 0, y, 0, 1);
            if (up_b)   u += up_b[ir];
            if (gate_b) g += gate_b[ir];
            d[ir] = ggml_cpu_up_gate_combine(op, limit, u, g);
        }
    }
}

static bool ggml_cpu_up_gate_iqk_ok(enum ggml_type type) {
#if GGML_USE_IQK_MULMAT
    // same exclusions as the mul_mat / mul_mat_id iqk sites
    // (TQ1_0 / TQ2_0 / Q2_0_G128, + Q2_0 401: dense kernel only, fused up/gate unexercised)
    return !ggml_cpu_iqk_gemm_disabled()
        && ggml_iqk_supports_type(type, GGML_IQK_SITE_CPU_UP_GATE);
#else
    GGML_UNUSED(type);
    return false;
#endif
}

static void ggml_compute_forward_mul_mat_id_up_gate(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    GGML_ASSERT(!dst->src[1] || dst->src[0]->type == dst->src[1]->type);
    GGML_ASSERT(!dst->src[1] || ggml_are_same_shape(dst->src[0], dst->src[1]));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const struct ggml_tensor * src1   = dst->src[2];
    const struct ggml_tensor * ids    = dst->src[3];
    const struct ggml_tensor * up_b   = dst->src[4];
    const struct ggml_tensor * gate_b = dst->src[5];
    const struct ggml_tensor * src0_1 = dst->src[0];
    const struct ggml_tensor * src0_2 = dst->src[1];
    const struct ggml_tensor * src0   = src0_1; // so GGML_TENSOR_BINARY_OP_LOCALS works

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float   = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);
    GGML_ASSERT(ne13 == 1);
    // both kernels address src1 as packed rows of row_size
    GGML_ASSERT(src1->type != vec_dot_type || ggml_is_contiguous(src1));

    const size_t nb41 = up_b ? up_b->nb[1] : 0;
    const size_t nb51 = up_b && gate_b ? gate_b->nb[1] : 0;

    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;
    const size_t src1_wsize = ggml_cpu_src1_wsize(type, src1);
    if (src1_wsize > 0) {
        incr_ptr_aligned(&wdata_cur, src1_wsize, sizeof(int64_t));
    }
    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));
    const int64_t rows_per_expert = ids->ne[0]*ids->ne[1];
    struct mmid_row_mapping * matrix_rows = // [n_as][rows_per_expert]
        incr_ptr_aligned(&wdata_cur, n_as*rows_per_expert*sizeof(struct mmid_row_mapping), sizeof(int64_t));
    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    // iqk reads its own activation type (ggml_cpu_iqk_act_type); the generic path reads
    // vec_dot_type. q_type records which one wdata currently holds.
    bool use_iqk = ggml_cpu_up_gate_iqk_ok(type);
#if GGML_USE_IQK_MULMAT
    const enum ggml_type iqk_type = use_iqk ? ggml_cpu_iqk_act_type(type, ne00) : GGML_TYPE_COUNT;
#else
    const enum ggml_type iqk_type = GGML_TYPE_COUNT;
#endif
    enum ggml_type q_type = vec_dot_type;
    if (src1->type != vec_dot_type) {
        q_type = iqk_type != GGML_TYPE_COUNT ? iqk_type : vec_dot_type;
        GGML_ASSERT(from_float != NULL);
        ggml_cpu_quantize_src1_rows(src1, q_type, params->wdata, ith, nth);
    }

    GGML_ASSERT(ids->ne[1] == dst->ne[2]);
    // SER: an out-of-range id is an unused slot -- zero its row (else stale values become NaNs
    // later) and leave it out of the grouping
    for (int64_t iid1 = ith; iid1 < ids->ne[1]; iid1 += nth) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
            if (i02 < 0 || i02 >= n_as) {
                memset((char *) dst->data + id*nb1 + iid1*nb2, 0, ne0*sizeof(float));
            }
        }
    }

    if (ith == 0) {
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                if (i02 < 0 || i02 >= n_as) {
                    continue;
                }
                matrix_rows[i02*rows_per_expert + matrix_row_counts[i02]] = (struct mmid_row_mapping) {id, (int32_t) iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    ggml_barrier(params->threadpool);

    const float  limit    = *(const float *)(dst->op_params + 1);
    const int    unary_op = dst->op_params[0];
    const char * wdata    = (src1->type == vec_dot_type) ? (const char *) src1->data : (const char *) params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    const int64_t nr0     = src0_2 ? ne01 : ne01/2;

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) {
            continue;
        }

        const char *src0_1_cur, *src0_2_cur, *up_b_cur = NULL, *gate_b_cur = NULL;
        if (src0_2) {
            src0_1_cur = (const char *) src0_1->data + cur_a*nb02;
            src0_2_cur = (const char *) src0_2->data + cur_a*nb02;
            up_b_cur   = up_b   ? (const char *)up_b->data   + cur_a*nb41 : NULL;
            gate_b_cur = gate_b ? (const char *)gate_b->data + cur_a*nb51 : NULL;
        } else {
            // merged up_gate tensor: gate is the first half of each expert, up the second
            src0_2_cur = (const char *) src0_1->data + cur_a*nb02;
            src0_1_cur = src0_2_cur + nb02/2;
            if (up_b) {
                GGML_ASSERT(!gate_b);
                gate_b_cur = (const char *)up_b->data + cur_a*nb41;
                up_b_cur   = gate_b_cur + nb41/2;
            }
        }

        const struct mmid_row_mapping * rows = matrix_rows + cur_a*rows_per_expert;

#if GGML_USE_IQK_MULMAT
        // a decline is decided before any work, by the type pair and this expert's row count --
        // identical on every thread, so the barriers below pair up
        if (use_iqk) {
            if (iqk_moe_fused_up_gate(nr0, cne1, ne00, ne11, unary_op,
                            type, src0_1_cur, src0_2_cur, nb01,
                            q_type, wdata, ggml_row_size(q_type, ne10),
                            up_b_cur, gate_b_cur,
                            (float *) dst->data, nb1, nb2,
                            rows, limit, ith, nth)) {
                continue;
            }
            if (q_type != vec_dot_type) {
                // the generic vec_dot cannot read the iqk activation type: wait for every thread to
                // leave earlier experts' iqk calls, re-quantize, finish generically
                ggml_barrier(params->threadpool);
                ggml_cpu_quantize_src1_rows(src1, vec_dot_type, params->wdata, ith, nth);
                ggml_barrier(params->threadpool);
                q_type  = vec_dot_type;
                use_iqk = false;
            }
        }
#else
        GGML_UNUSED(use_iqk);
#endif
        ggml_cpu_up_gate_generic(ith, nth, nr0, cne1, ne00, ne11, unary_op, limit,
                type, src0_1_cur, src0_2_cur, nb01,
                (const float *) up_b_cur, (const float *) gate_b_cur,
                wdata, row_size, (char *) dst->data, nb1, nb2, rows);
    }
}

static void ggml_compute_forward_mul_mat_up_gate(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    GGML_ASSERT(dst->src[0]->type == dst->src[1]->type);
    GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst->src[1]));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const struct ggml_tensor * src1   = dst->src[2];
    const struct ggml_tensor * src0_1 = dst->src[0];
    const struct ggml_tensor * src0_2 = dst->src[1];
    const struct ggml_tensor * src0   = src0_1; // so GGML_TENSOR_BINARY_OP_LOCALS works

    GGML_ASSERT(ggml_is_quantized(src0_1->type) && src0_1->type == src0_2->type);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float   = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);
    GGML_ASSERT(ne13 == 1);
    // the old op quantized every ne12 slice but multiplied only the first; say so instead
    GGML_ASSERT(ne12 == 1);

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(from_float != NULL);

    char * wdata = params->wdata;

    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    GGML_ASSERT(params->wsize >= ggml_cpu_src1_wsize(type, src1));

    const float limit    = *(const float *)(dst->op_params + 1);
    const int   unary_op = dst->op_params[0];

#if GGML_USE_IQK_MULMAT
    const enum ggml_type iqk_type = ggml_cpu_up_gate_iqk_ok(type) ? ggml_cpu_iqk_act_type(type, ne00) : GGML_TYPE_COUNT;
    const enum ggml_type act_type = iqk_type != GGML_TYPE_COUNT ? iqk_type : vec_dot_type;
    ggml_cpu_quantize_src1_rows(src1, act_type, wdata, ith, nth);
    ggml_barrier(params->threadpool);
    if (ggml_cpu_up_gate_iqk_ok(type) &&
        iqk_moe_fused_up_gate(ne01, ne11, ne00, ne11, unary_op,
                         type, src0_1->data, src0_2->data, nb01,
                         act_type, (const char *) wdata, ggml_row_size(act_type, ne10),
                         NULL, NULL,
                         (float *) dst->data, nb1, nb2,
                         NULL, limit, ith, nth)) {
        return;
    }
    if (act_type != vec_dot_type) {
        // declined before reading B on every thread; re-quantize for the generic vec_dot
        ggml_cpu_quantize_src1_rows(src1, vec_dot_type, wdata, ith, nth);
        ggml_barrier(params->threadpool);
    }
#else
    ggml_cpu_quantize_src1_rows(src1, vec_dot_type, wdata, ith, nth);
    ggml_barrier(params->threadpool);
#endif
    ggml_cpu_up_gate_generic(ith, nth, ne01, ne11, ne00, ne11, unary_op, limit,
            type, (const char *) src0_1->data, (const char *) src0_2->data, nb01,
            NULL, NULL, wdata, row_size, (char *) dst->data, nb1, nb2, NULL);
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    // OPT-IN OP TRACE, added 2026-09-13. Set GGML_CPU_TRACE_OPS=1 to print every op this function
    // sees and what it did with it.
    //
    // WHY IT EXISTS: a build of this tree computed ggml_sum(ggml_add(4,3)) = 0 while returning
    // GGML_STATUS_SUCCESS from both the scheduler path and ggml_graph_compute_with_ctx. Every
    // instrument OUTSIDE this function said the setup was fine -- inputs read back exactly,
    // n_nodes=2, ggml_is_empty=false, ne=[1,1,1,1], plan n_threads=1 -- so the silence had to be
    // in here, and there was no way to see it without rebuilding with a printf. There is now.
    //
    // The three early exits below are indistinguishable from outside: each one skips the op and
    // leaves the destination untouched, and the graph still reports success. The trace names which
    // one fired, which is the difference between "a kernel computed the wrong answer" and "no
    // kernel ran at all".
    static int ggml_cpu_trace_ops = -1;
    if (ggml_cpu_trace_ops < 0) {
        const char * e = getenv("GGML_CPU_TRACE_OPS");
        ggml_cpu_trace_ops = (e && *e && *e != '0') ? 1 : 0;
    }

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        if (ggml_cpu_trace_ops) {
            fprintf(stderr, "[cpu-trace] SKIP-EMPTY op=%s name=%s ith=%d\n",
                    ggml_op_name(tensor->op), tensor->name, params->ith);
        }
        return;
    }

    // extra_buffer op?
    const bool ggml_cpu_extra_claimed = ggml_cpu_extra_compute_forward(params, tensor);
    if (ggml_cpu_trace_ops) {
        fprintf(stderr, "[cpu-trace] op=%s name=%s ith=%d nth=%d extra_claimed=%s\n",
                ggml_op_name(tensor->op), tensor->name, params->ith, params->nth,
                ggml_cpu_extra_claimed ? "TRUE (op skipped by an extra-buffer trait)" : "false");
    }
    if (ggml_cpu_extra_claimed) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_FMA:
            {
                ggml_compute_forward_fma(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_FUSED_RMS_NORM:
            {
                ggml_compute_forward_fused_rms_norm(params, tensor);
            } break;
        case GGML_OP_GATED_GROUP_RMS_NORM:
            {
                ggml_compute_forward_gated_group_rms_norm(params, tensor);
            } break;
        case GGML_OP_FUSED_MUL_UNARY:
            {
                ggml_compute_forward_fused_mul_unary(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_MOE_FUSED_UP_GATE:
            {
                ggml_compute_forward_mul_mat_id_up_gate(params, tensor);
            } break;
        case GGML_OP_FUSED_UP_GATE:
            {
                ggml_compute_forward_mul_mat_up_gate(params, tensor);
            } break;
        case GGML_OP_MULTI_ADD:
            {
                ggml_compute_forward_multi_add(params, tensor);
            } break;
        case GGML_OP_REDUCE:
            {
                ggml_compute_forward_reduce(params, tensor);
            } break;
        case GGML_OP_FUSED_NORM:
            {
                ggml_compute_forward_fused_norm(params, tensor);
            } break;
        case GGML_OP_ROPE_CACHE:
            {
                ggml_compute_forward_rope_cache(params, tensor);
            } break;
        case GGML_OP_ROPE_FAST:
            {
                ggml_compute_forward_rope_fast(params, tensor);
            } break;
        case GGML_OP_ARGSORT_THRESH:
            {
                ggml_compute_forward_argsort_thresh(params, tensor);
            } break;
        case GGML_OP_GROUPED_TOPK:
            {
                ggml_compute_forward_grouped_topk(params, tensor);
            } break;
        case GGML_OP_SINKHORN:
            {
                ggml_compute_forward_sinkhorn(params, tensor);
            } break;
        case GGML_OP_HC_PRE:
            {
                ggml_compute_forward_hc_pre(params, tensor);
            } break;
        case GGML_OP_HC_POST:
            {
                ggml_compute_forward_hc_post(params, tensor);
            } break;
        case GGML_OP_DS4_COMP:
            {
                ggml_compute_forward_ds4_comp(params, tensor);
            } break;
        case GGML_OP_LATENT_ATTN:
            {
                ggml_compute_forward_latent_attn(params, tensor);
            } break;
        case GGML_OP_INDEXER_TOPK:
            {
                // iqk's fused indexer top-k synchronizes its phases itself, through the barrier it
                // is handed (the old ggml.c passed ggml_barrier + its shared state the same way)
#if GGML_USE_IQK_MULMAT
                if (!iqk_indexer_topk(tensor, params->wdata, (barrier_t) ggml_barrier, (void *) params->threadpool,
                                      params->ith, params->nth)) {
                    GGML_ABORT("iqk_indexer_topk declined");
                }
#else
                GGML_ABORT("INDEXER_TOPK needs the iqk build");
#endif
            } break;
        case GGML_OP_MAP_UNARY:
            {
                ggml_compute_forward_map_unary(params, tensor);
            } break;
        case GGML_OP_MAP_BINARY:
            {
                ggml_compute_forward_map_binary(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1_F32:
            {
                ggml_compute_forward_map_custom1_f32(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM2_F32:
            {
                ggml_compute_forward_map_custom2_f32(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM3_F32:
            {
                ggml_compute_forward_map_custom3_f32(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFTCAP:
            {
                ggml_compute_forward_softcap(params, tensor);
            } break;
        case GGML_OP_SOFT_CAP_MAX:
            {
                ggml_compute_forward_softcap_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } break;
        case GGML_OP_COL2IM_3D:
            {
                ggml_compute_forward_col2im_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_MASK_TO_IDX:
            {
                ggml_compute_forward_mask_to_idx(params, tensor);
            } break;
        case GGML_OP_DELTA_NET:
            {
                ggml_compute_forward_delta_net(params, tensor);
            } break;
        case GGML_OP_MUL_MULTI_ADD:
            {
                ggml_compute_forward_mul_multi_add(params, tensor);
            } break;
        case GGML_OP_HADAMARD:
            {
                ggml_compute_forward_hadamard(params, tensor);
            } break;
        case GGML_OP_TURBO_WHT:
            {
                ggml_compute_forward_turbo_wht(params, tensor);
            } break;
        case GGML_OP_FUSED_RMS_RMS_ADD:
            {
                ggml_compute_forward_fused_rms_rms_add(params, tensor);
            } break;
        case GGML_OP_BLEND:
            {
                ggml_compute_forward_blend(params, tensor);
            } break;
        case GGML_OP_MASK_TOPK:
            {
                ggml_compute_forward_mask_topk(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } break;
        case GGML_OP_FFD_SELECT:
            {
                ggml_compute_forward_ffd_select(params, tensor);
            } break;
        case GGML_OP_FFD_QUANT_K:
            {
                ggml_compute_forward_ffd_quant_k(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } break;
        case GGML_OP_TOP_K_MASK:
            {
                ggml_compute_forward_top_k_mask(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

// ik afdbc5e13 (#2565): pin worker `thread_n` to its assigned logical CPU (no-op without affinity)
static void set_cpu_thread_affinity(const struct ggml_cplan * cplan, int thread_n) {
    if (cplan == NULL || cplan->cpu_affinity == NULL || cplan->n_cpu_affinity <= 0) {
        return;
    }

    if (thread_n == 0 && cplan->n_threads > cplan->n_cpu_affinity) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "warning: n_threads (%d) exceeds the %d CPUs in the affinity list, threads are stacked\n",
                    cplan->n_threads, cplan->n_cpu_affinity);
        }
    }

    const int cpu = cplan->cpu_affinity[thread_n % cplan->n_cpu_affinity];
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        return;
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);

    const int rv = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
static void set_cpu_thread_affinity(const struct ggml_cplan * cplan, int thread_n) { UNUSED(cplan); UNUSED(thread_n); }
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_TOP_K_MASK:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_SQRT_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                // routed to the GLU kernels in ops.cpp, which split rows by ith/nth like the GLU case
                case GGML_UNARY_OP_SWIGLU:
                case GGML_UNARY_OP_SWIGLU_OAI:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_FMA:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_FUSED_RMS_NORM:
        case GGML_OP_GATED_GROUP_RMS_NORM:
        case GGML_OP_FUSED_MUL_UNARY:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MOE_FUSED_UP_GATE:
        case GGML_OP_FUSED_UP_GATE:
        case GGML_OP_MULTI_ADD:
        case GGML_OP_REDUCE:
        case GGML_OP_FUSED_NORM:
        case GGML_OP_ROPE_CACHE:
        case GGML_OP_ROPE_FAST:
        case GGML_OP_ARGSORT_THRESH:
        case GGML_OP_GROUPED_TOPK:
        case GGML_OP_SINKHORN:
        case GGML_OP_HC_PRE:
        case GGML_OP_HC_POST:
        case GGML_OP_DS4_COMP:
        case GGML_OP_LATENT_ATTN:
        case GGML_OP_INDEXER_TOPK:
        case GGML_OP_OUT_PROD:
        case GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFTCAP:
        case GGML_OP_SOFT_CAP_MAX:
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_COL2IM_3D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_MASK_TO_IDX:
        case GGML_OP_DELTA_NET:
        case GGML_OP_MUL_MULTI_ADD:
        case GGML_OP_HADAMARD:
        case GGML_OP_TURBO_WHT:
        case GGML_OP_FUSED_RMS_RMS_ADD:
        case GGML_OP_BLEND:
        case GGML_OP_MASK_TOPK:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_LIGHTNING_INDEXER:
        case GGML_OP_FFD_SELECT:
        case GGML_OP_FFD_QUANT_K:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
        case GGML_OP_MAP_UNARY:
        case GGML_OP_MAP_BINARY:
        case GGML_OP_MAP_CUSTOM1_F32:
        case GGML_OP_MAP_CUSTOM2_F32:
        case GGML_OP_MAP_CUSTOM3_F32:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// A Windows thread belongs to ONE processor group of at most 64 logical CPUs, so a global CPU
// index i (the index the --cpu-mask / --cpu-range / strict-cpu pool use) is mapped to
// (group, bit) through a table of per-group sizes built once.
#define GGML_WIN_MAX_PROC_GROUPS 16

static int32_t ggml_win_group_size[GGML_WIN_MAX_PROC_GROUPS];
static int32_t ggml_win_group_count = 0;

static void ggml_win_group_table_init(void) {
    int32_t n = (int32_t) GetActiveProcessorGroupCount();
    if (n > GGML_WIN_MAX_PROC_GROUPS) {
        n = GGML_WIN_MAX_PROC_GROUPS;
    }
    for (int32_t g = 0; g < n; g++) {
        ggml_win_group_size[g] = (int32_t) GetActiveProcessorCount((WORD) g);
    }
    ggml_win_group_count = n;
}

// global CPU index -> (group, bit in group); false when the CPU does not exist
bool ggml_win_cpu_to_group(const int32_t * sizes, int32_t n_groups, int32_t cpu, int32_t * group, int32_t * bit) {
    int32_t base = 0;
    for (int32_t g = 0; g < n_groups; g++) {
        if (cpu < base + sizes[g]) {
            *group = g;
            *bit   = cpu - base;
            return true;
        }
        base += sizes[g];
    }
    return false;
}

bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE h = GetCurrentThread();

    // the group table is a read-only snapshot of the machine; two racing initialisers write the same values
    if (ggml_win_group_count == 0) {
        ggml_win_group_table_init();
    }

    // a thread has one group at a time: pin to the group of the FIRST set bit and OR in every
    // set bit of that group; set bits in other groups are reported, not silently dropped
    GROUP_AFFINITY ga;
    memset(&ga, 0, sizeof(ga));
    bool have_group = false;
    bool other_group = false;
    bool missing_cpu = false;

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (!mask[i]) {
            continue;
        }
        int32_t g = 0, b = 0;
        if (!ggml_win_cpu_to_group(ggml_win_group_size, ggml_win_group_count, i, &g, &b)) {
            missing_cpu = true;
            continue;
        }
        if (!have_group) {
            ga.Group   = (WORD) g;
            have_group = true;
        }
        if ((WORD) g != ga.Group) {
            other_group = true;
            continue;
        }
        ga.Mask |= ((KAFFINITY) 1) << b;
    }

    if (missing_cpu) {
        fprintf(stderr, "warn: thread-affinity mask names a CPU that does not exist on this machine\n");
    }
    if (other_group) {
        fprintf(stderr, "warn: thread-affinity mask spans several processor groups; a thread can only use one, pinned to the group of the first set CPU\n");
    }
    if (!have_group) {
        return false;
    }

    return SetThreadGroupAffinity(h, &ga, NULL) != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

// Declared as GGML_BACKEND_API in ggml/include/ggml-cpu.h:71 (and identically in upstream's own
// ggml-cpu.h) but NEVER defined -- not here, not upstream, not in any commit since the declaration
// was introduced by 42c76d135 "Threadpool: take 2 (#8672)" in Aug 2024. Every other member of the
// ggml_threadpool_* family (new/free/pause/resume) is defined in this file; only the getter was
// missing, so the header advertised an API that could not link. Implemented rather than deleted:
// the declaration is public, the value is right here, and a 3-line accessor removes a latent
// unresolved-symbol for any consumer that takes the header at its word.
int ggml_threadpool_get_n_threads(struct ggml_threadpool * threadpool) {
    return threadpool ? threadpool->n_threads : 0;
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#if defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type) || node->src[0]->type == GGML_TYPE_F8_E4M3) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        cur = ggml_cpu_src1_wsize(node->src[0]->type, node->src[1]);
#if GGML_USE_IQK_MULMAT
                        // non-F32, non-vec_dot_type src1: GGML_CPU_SRC1_SCRATCH_ROWS F32 scratch rows per thread after the converted copy
                        // (iqk_quantize_any converts a whole row-interleaved group -- up to 16 rows -- through this scratch at once)
                        if (node->src[1]->type != GGML_TYPE_F32 &&
                            node->src[1]->type != type_traits_cpu[node->src[0]->type].vec_dot_type) {
                            cur += ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * GGML_CPU_SRC1_SCRATCH_ROWS * n_tasks;
                        }
#endif
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const int n_as = src0->ne[2];
                        // src1 (either the vec_dot_type or the iqk activation copy)
                        const size_t src1_wsize = ggml_cpu_src1_wsize(src0->type, src1);
                        if (src1_wsize > 0) {
                            cur += src1_wsize + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } break;
                case GGML_OP_MOE_FUSED_UP_GATE:
                    {
                        // mirrors the incr_ptr_aligned layout in ggml_compute_forward_mul_mat_id_up_gate
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[2];
                        const struct ggml_tensor * ids  = node->src[3];
                        const int n_as = src0->ne[2];
                        const size_t src1_wsize = ggml_cpu_src1_wsize(src0->type, src1);
                        if (src1_wsize > 0) {
                            cur += src1_wsize + sizeof(int64_t);
                        }
                        cur += n_as*sizeof(int64_t) + sizeof(int64_t);
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                    } break;
                case GGML_OP_FUSED_UP_GATE:
                    {
                        cur = ggml_cpu_src1_wsize(node->src[0]->type, node->src[2]);
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type) || node->src[0]->type == GGML_TYPE_F8_E4M3 ||
                            node->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SET_ROWS:
                    {
                        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_SOFT_CAP_MAX:   // same per-thread row scratch as SOFT_MAX
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In
                        const int64_t ne13 = node->src[1]->ne[3]; // Batch

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12 * ne13;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                // The next three are from ggml.c's ggml_graph_plan as of 3c949f339 -- dropped with
                // that whole function when upstream's backend split merged in; ported with their ops.
                case GGML_OP_LATENT_ATTN:
                    {
                        // per thread: [P + dense-or-gathered rows] f32 scores + one [Dk] dequant row
                        const int64_t Dk = node->src[1]->ne[0];
                        const int64_t N  = node->src[1]->ne[1];
                        const int64_t P  = node->src[2] ? node->src[2]->ne[1] : 0;
                        const int mode = node->op_params[4];
                        GGML_ASSERT(mode == 0 || mode == 1);
                        const int64_t rows = mode == 0 ? N : node->src[5]->ne[0];
                        cur = sizeof(float)*(P + rows + Dk)*n_tasks;
                    } break;
                case GGML_OP_DS4_COMP:
                    {
                        if (node->op_params[0] == 0) {
                            const struct ggml_tensor * idx = node->src[2];
                            const int ratio = idx->ne[0] / (2*node->ne[1]);
                            cur = 32*(4*ratio + 3)*sizeof(float)*n_tasks;
                        }
                    } break;
#if GGML_USE_IQK_MULMAT
                case GGML_OP_INDEXER_TOPK:
                    {
                        cur = iqk_idx_topk_work_buffer_size(node, n_tasks);
                    } break;
#endif
                case GGML_OP_MASK_TO_IDX:
                    {
                        // ik #2779: the top-k form (src[1] = topk) sorts each row's surviving top-k columns in a
                        // per-thread int32 buffer of topk->ne[0] entries (ops.cpp ggml_compute_forward_mask_topk_to_idx)
                        if (node->src[1]) {
                            cur = sizeof(int32_t)*node->src[1]->ne[0]*n_tasks;
                        }
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
#if GGML_USE_IQK_MULMAT
                        // the iqk FA tried first in ggml_compute_forward_flash_attn_ext (ops.cpp)
                        cur = MAX(cur, iqk_fa_work_buffer_size(node, n_tasks));
#endif
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = ggml_get_op_params_i32(node, 0);
                        const int64_t per_thread = 3 * S_v + (K > 1 ? S_v * S_v : 0);   // delta, dexp, gk (+ snapshot state): see ops.cpp
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                case GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        const int64_t ne10 = node->src[1]->ne[0];
                        cur += sizeof(float)*ne10*n_tasks;
                    } break;
                case GGML_OP_FFD_SELECT:
                    {
                        // per thread: one dequantised K row (D) + a [G][n_blocks] running block max,
                        // padded by a cache line -- must match the slice arithmetic in ops.cpp
                        const int64_t D  = node->src[0]->ne[0];
                        const int64_t G  = node->src[0]->ne[2]/node->src[1]->ne[2];
                        const int64_t nb = node->ne[0];
                        cur += sizeof(float)*(D + G*nb + CACHE_LINE_SIZE_F32)*n_tasks;
                    } break;
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

static int ggml_cpu_try_fuse_ops(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params,
        const struct ggml_cplan * cplan) {

    if (ggml_cpu_disable_fusion || cplan->use_ref) {
        return 0;
    }

    struct ggml_tensor * node = cgraph->nodes[node_n];

    if (node->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        const enum ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
            struct ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
            const struct ggml_tensor * mul_w = (mul_node->src[0] == node)
                ? mul_node->src[1] : mul_node->src[0];
            if (node->src[0]->type  == GGML_TYPE_F32 &&
                mul_node->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == node->ne[0]   &&
                mul_w->nb[0]        == sizeof(float)) {

                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
                return 1;
            }
        }
    }

    return 0;
}

// ---- opt-in flush-denormals (--cpu-flush-denormals); default OFF, so the default path is unchanged ----
// PRIOR-ART-NONE: no FTZ/DAZ on canonical (pickaxe over all branches 2026-11-16: only miniaudio + wip snapshots).
// MEASURED: ggml-cpu/quants.c:1459 records the denormal assist cost (170 vs 90 ns/block); BENCH-PENDING for the end-to-end effect.
static atomic_int g_cpu_flush_denormals = 0;

void ggml_cpu_set_flush_denormals(bool enable) {
    atomic_store_explicit(&g_cpu_flush_denormals, enable ? 1 : 0, memory_order_relaxed);
}

bool ggml_cpu_get_flush_denormals(void) {
    return atomic_load_explicit(&g_cpu_flush_denormals, memory_order_relaxed) != 0;
}

#define GGML_FP_MODE_NONE (~(uint64_t)0)

// returns the previous control word to hand to ggml_cpu_fp_mode_leave, or GGML_FP_MODE_NONE when nothing was changed
static inline uint64_t ggml_cpu_fp_mode_enter(void) {
    if (!ggml_cpu_get_flush_denormals()) {
        return GGML_FP_MODE_NONE;
    }
#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
    const unsigned int old = _mm_getcsr();
    _mm_setcsr(old | 0x8040u); // FTZ (bit 15) | DAZ (bit 6)
    return (uint64_t) old;
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    uint64_t old;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(old));
    const uint64_t fz = old | (1ull << 24); // FPCR.FZ
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fz));
    return old;
#else
    return GGML_FP_MODE_NONE; // no FTZ control on this target: accepted flag is a no-op
#endif
}

static inline void ggml_cpu_fp_mode_leave(uint64_t old) {
    if (old == GGML_FP_MODE_NONE) {
        return;
    }
#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
    _mm_setcsr((unsigned int) old);
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("msr fpcr, %0" : : "r"(old));
#endif
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

    const uint64_t fp_mode_saved = ggml_cpu_fp_mode_enter();

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state->ith);
#endif
    set_cpu_thread_affinity(cplan, state->ith); // ik #2565: explicit affinity after NUMA placement

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    // Part of the GGML_CPU_TRACE_OPS instrumentation. ggml_graph_compute was proven to be ENTERED
    // with n_nodes=2 while ggml_compute_forward was never called once, so the loss is between the
    // two: either this thread function does not run, or its loop iterates zero times. Printing
    // both the arrival and the loop bounds tells them apart, and the bounds are read from
    // tp->cgraph -- which is NOT necessarily the graph the caller passed.
    {
        static int trace_th = -1;
        if (trace_th < 0) {
            const char * e = getenv("GGML_CPU_TRACE_OPS");
            trace_th = (e && *e && *e != '0') ? 1 : 0;
        }
        if (trace_th) {
            fprintf(stderr, "[cpu-trace] thread ith=%d ARRIVED cgraph=%p n_nodes=%d abort=%d\n",
                    state->ith, (const void *) cgraph, cgraph ? cgraph->n_nodes : -1,
                    (int) atomic_load_explicit(&tp->abort, memory_order_relaxed));
        }
    }

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    ggml_cpu_fp_mode_leave(fp_mode_saved);

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

// TurboQuant vec_dot against an f32 operand (the CPU flash-attention K path): dequantise to the rotated basis in fixed-size chunks
// (no allocation per call, both operands stay in L1) and dot. Every turbo row dequant is a pure per-block loop (ggml-turbo-quant.c), so
// chunking on block boundaries gives the same bytes as a whole-row dequant.
#define GGML_TURBO_DOT_CHUNK 256

static void ggml_vec_dot_turbo_f32_impl(enum ggml_type type_x, int n, float * GGML_RESTRICT s,
                                        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const struct ggml_type_traits * trx = ggml_get_type_traits(type_x);

    const int64_t blk = trx->blck_size;
    GGML_ASSERT(n % blk == 0);
    GGML_ASSERT(blk <= GGML_TURBO_DOT_CHUNK);

    const int64_t chunk = (GGML_TURBO_DOT_CHUNK / blk) * blk;

    const char  * px = (const char  *) vx;
    const float * y  = (const float *) vy;

    float xb[GGML_TURBO_DOT_CHUNK];
    float sum = 0.0f;

    for (int64_t i = 0; i < n; i += chunk) {
        const int64_t nc = MIN(chunk, n - i);
        trx->to_float(px, xb, nc);
        for (int64_t j = 0; j < nc; j++) {
            sum += xb[j] * y[i + j];
        }
        px += (nc / blk) * trx->type_size;
    }

    *s = sum;
}

#define GGML_TURBO_VEC_DOT(NAME, TYPE)                                                                         \
static void NAME(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx,        \
                 const void * GGML_RESTRICT vy, size_t by, int nrc) {                                          \
    GGML_ASSERT(nrc == 1);                                                                                     \
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by);                                                         \
    ggml_vec_dot_turbo_f32_impl(TYPE, n, s, vx, vy);                                                           \
}
GGML_TURBO_VEC_DOT(ggml_vec_dot_turbo3_0_f32, GGML_TYPE_TURBO3_0)
GGML_TURBO_VEC_DOT(ggml_vec_dot_turbo2_0_f32, GGML_TYPE_TURBO2_0)
GGML_TURBO_VEC_DOT(ggml_vec_dot_turbo4_0_f32, GGML_TYPE_TURBO4_0)
#undef GGML_TURBO_VEC_DOT

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    // Part of the GGML_CPU_TRACE_OPS instrumentation (see ggml_compute_forward above). This entry
    // is traced as well because the per-op trace printing NOTHING has two very different meanings:
    // this function was never reached, or it was reached and iterated no nodes. Only printing both
    // tells them apart, and on the run that prompted this the per-op trace was silent.
    {
        static int trace_gc = -1;
        if (trace_gc < 0) {
            const char * e = getenv("GGML_CPU_TRACE_OPS");
            trace_gc = (e && *e && *e != '0') ? 1 : 0;
        }
        if (trace_gc) {
            fprintf(stderr, "[cpu-trace] ggml_graph_compute ENTERED n_nodes=%d n_threads=%d\n",
                    ggml_graph_n_nodes(cgraph), cplan ? cplan->n_threads : -1);
        }
    }

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#elif defined(__VXE__) || defined(__VXE2__)
    for (; i + 7 < n; i += 8) {
        const uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        const uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme2(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    return 1;
#else
    return 0;
#endif
}

// instruction sets the HOST supports (CPUID + OS state), as "AVX2 FMA ..." -- printed next to the compiled set in system_info
const char * ggml_cpu_host_isa(void) {
    static char buf[96];
    ggml_isa_guard_format(ggml_isa_guard_host_mask(), buf, sizeof(buf));
    return buf;
}

void ggml_cpu_init(void) {
    // ISA guard: a binary compiled for an instruction set the host lacks dies later with an unexplained illegal-instruction
    // fault inside a kernel; stop here and name the flag instead.
    {
        const uint32_t compiled = ggml_isa_guard_compiled_mask();
        const char * missing = ggml_isa_guard_first_missing_name(compiled, ggml_isa_guard_host_mask());
        if (missing != NULL) {
            char have[96], need[96];
            ggml_isa_guard_format(ggml_isa_guard_host_mask(), have, sizeof(have));
            ggml_isa_guard_format(compiled, need, sizeof(need));
            fprintf(stderr, "ggml_cpu_init: this binary was compiled for [%s] but the host CPU lacks %s (host has: [%s]). "
                            "Rebuild without that instruction set (GGML_NATIVE=OFF / lower -march) or run on a CPU that has it.\n",
                    need, missing, have);
            GGML_ABORT("CPU lacks an instruction set this binary was compiled for: %s", missing);
        }
    }

    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
                ggml_table_f32_f8_e4m3[i] = ggml_f8_e4m3_to_fp32(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            const char * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);
        }

        // the IQ3_K / IQ3_KS / IQ5_K / IQ5_KS / IQ6_K quantizers live in ggml-base (no arch flags); hand them this
        // backend's AVX2 scale search (iqk/iqk_scale_search.h) when GGML_IQK_AVX2_SCALE_SEARCH=1.
        // OPT-IN, measured 2026-10-17 on TriLM_99M (wikitext-2, 40 x 512): the AVX2 search quantizes 1.4-2.3x
        // faster but PPL moved IQ3_KS 54.77 -> 64.84, IQ3_K 50.70 -> 51.63, IQ6_K 44.23 -> 44.43 (IQ5_K/IQ5_KS
        // flat), and ik's AVX2 IQ5_K/IQ6_K weigh by x^2 where the scalar search adds sigma2. The default also keeps
        // from_float_ref host-independent, which tests/test-iqk-golden relies on.
        {
            const char * env = getenv("GGML_IQK_AVX2_SCALE_SEARCH");
            if (env != NULL && atoi(env) == 1) {
                iqk_register_scale_search_cpu();
            }
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
