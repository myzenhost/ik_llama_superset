//
// Copyright (C) 2023-2025 The ggml authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct ggml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct ggml_context * ctx = ggml_init(params);
//
//       struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//
//       ggml_set_param(ctx, x); // x is an input variable
//
//       struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//       struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
//       struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
//       struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct ggml_cgraph * gf = ggml_new_graph(ctx);
//       ggml_build_forward_expand(gf, f);
//
//       // set the input variable and parameter values
//       ggml_set_f32(x, 2.0f);
//       ggml_set_f32(a, 3.0f);
//       ggml_set_f32(b, 4.0f);
//
//       ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
//
//       printf("f = %f\n", ggml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the ggml_graph_compute() function.
//
// The ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the ggml_used_mem() function to find out how much memory was
// actually needed.
//
// The ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the ggml_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - ggml_permute()
//   - ggml_conv_1d_1s()
//   - ggml_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct ggml_tensor)
//
// The tensors are stored in memory via the ggml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct ggml_tensor * c = ggml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The ggml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       const int nx = 2;
//       const int ny = 3;
//
//       struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nx, ny);
//
//       for (int y = 0; y < ny; y++) {
//           for (int x = 0; x < nx; x++) {
//               *(float *) ((char *) a->data + y*a->nb[1] + x*a->nb[0]) = x + y;
//           }
//       }
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as ggml_get_f32_1d() and ggml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (ggml_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BUILD
#            define GGML_API __declspec(dllexport)
#        else
#            define GGML_API __declspec(dllimport)
#        endif
#    else
#        define GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GGML_API
#endif

#ifdef GGML_MULTIPLATFORM
#    if defined(_WIN32)
#        define GGML_CALL
#    else
#        define GGML_CALL __attribute__((__ms_abi__))
#    endif
#else
#    define GGML_CALL
#endif

// TODO: support for clang
#ifdef __GNUC__
#    define GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define GGML_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define GGML_DEPRECATED(func, hint) func
#endif

#ifndef __GNUC__
#    define GGML_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__)
#    define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_FILE_VERSION 1

#define GGML_QNT_VERSION        2    // bump this on quantization format changes
#define GGML_QNT_VERSION_FACTOR 1000 // do not change this

#define GGML_MAX_DIMS           4
#define GGML_MAX_PARAMS         2048
#ifndef GGML_MAX_CONTEXTS
// Maximum number of model contexts (e.g., for model shards).
// Default raised 64 -> 2048 (S69-THIREUS-PICK-GGML-MAX-CONTEXTS-BUILD-DEFAULT, 2026-10-05): per-tensor-shard GGUF recipes
// (Thireus GGUF-Tool-Suite) open one ggml context per shard. The cost is one 'used' flag plus a ggml_context (~100 bytes)
// per slot in a process-global array (~200 KB) and a linear scan in ggml_init over that array; on Windows the loader raises
// _setmaxstdio to match (src/llama-model-loader.cpp, GGML_MAX_CONTEXTS > 512 branch) so every shard file can stay open.
// Override with -DGGML_MAX_CONTEXTS=<value> in CMake.
#define GGML_MAX_CONTEXTS       2048
#endif
#ifndef GGML_MAX_SRC
// For the machines with 11+ GPUs use -DGGML_MAX_SRC=N
#define GGML_MAX_SRC            16
#endif
#ifndef GGML_MAX_NAME
#define GGML_MAX_NAME           64
#endif
#define GGML_MAX_OP_PARAMS      64
// Adopted from upstream ggml.h:225. Must match the value upstream uses, because it sizes
// ggml_threadpool_params.cpumask AND is the byte count memset/memcmp'd over it.
#define GGML_MAX_N_THREADS      512
#define GGML_DEFAULT_N_THREADS  4
#define GGML_DEFAULT_GRAPH_SIZE 2048
#if UINTPTR_MAX == 0xFFFFFFFF
    #define GGML_MEM_ALIGN 4
#else
    #define GGML_MEM_ALIGN 16
#endif

#define GGML_EXIT_SUCCESS 0
#define GGML_EXIT_ABORTED 1

#define GGML_ROPE_TYPE_NORMAL 0 // the default interleaved layout; upstream spells the 0 case
#define GGML_ROPE_TYPE_NEOX   2
#define GGML_ROPE_TYPE_MROPE  8
#define GGML_ROPE_TYPE_VISION 24
#define GGML_ROPE_TYPE_IMROPE 40 // binary: 101000

#define GGML_MROPE_SECTIONS   4

#define GGUF_MAGIC "GGUF"

#define GGUF_VERSION 3

#define GGUF_DEFAULT_ALIGNMENT 32

#define GGML_UNUSED(x) (void)(x)
#ifdef __CUDACC__
template<typename... Args>
__host__ __device__ constexpr inline void ggml_unused_vars_impl(Args&&...) noexcept {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)
#else
#define GGML_UNUSED_VARS(...) do { (void)sizeof((__VA_ARGS__, 0)); } while(0)
#endif // __CUDACC__

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ifndef NDEBUG
#define GGML_UNREACHABLE() do { fprintf(stderr, "statement should be unreachable\n"); abort(); } while(0)
#elif defined(__GNUC__)
#define GGML_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define GGML_UNREACHABLE() __assume(0)
#else
#define GGML_UNREACHABLE() ((void) 0)
#endif

#ifdef __cplusplus
#define GGML_NORETURN [[noreturn]]
#elif defined(_MSC_VER)
#define GGML_NORETURN __declspec(noreturn)
#else
#define GGML_NORETURN _Noreturn
#endif

#define GGML_ABORT(...) ggml_abort(__FILE__, __LINE__, __VA_ARGS__)
#define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)

// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_UNUSED(prefix##2);
#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_UNUSED(prefix##3);

#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_TERNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne2, src2, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb2, src2, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS01 \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

#ifdef  __cplusplus
extern "C" {
#endif

    GGML_NORETURN GGML_ATTRIBUTE_FORMAT(3, 4)
    GGML_API void ggml_abort(const char * file, int line, const char * fmt, ...);

    enum ggml_status {
        GGML_STATUS_ALLOC_FAILED = -2,
        GGML_STATUS_FAILED = -1,
        GGML_STATUS_SUCCESS = 0,
        GGML_STATUS_ABORTED = 1,
    };

    // get ggml_status name string
    GGML_API GGML_CALL const char * ggml_status_to_string(enum ggml_status status);

    // ieee 754-2008 half-precision float16
    // todo: make this not an integral type
    typedef uint16_t ggml_fp16_t;
    GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t);
    GGML_API ggml_fp16_t ggml_fp32_to_fp16(float);
    GGML_API void        ggml_fp16_to_fp32_row(const ggml_fp16_t *, float *, int64_t);
    GGML_API void        ggml_fp32_to_fp16_row(const float *, ggml_fp16_t *, int64_t);

    // google brain half-precision bfloat16
    typedef struct { uint16_t bits; } ggml_bf16_t;
    GGML_API ggml_bf16_t ggml_fp32_to_bf16(float);
    GGML_API float       ggml_bf16_to_fp32(ggml_bf16_t);  // consider just doing << 16
    GGML_API void        ggml_bf16_to_fp32_row(const ggml_bf16_t *, float *, int64_t);
    GGML_API void        ggml_fp32_to_bf16_row_ref(const float *, ggml_bf16_t *, int64_t);
    GGML_API void        ggml_fp32_to_bf16_row(const float *, ggml_bf16_t *, int64_t);

    typedef struct { uint8_t bits; } ggml_fp8_e4m3_t;

    struct ggml_object;
    struct ggml_context;

    // NOTE: always add types at the end of the enum to keep backward compatibility
    enum ggml_type {
        GGML_TYPE_F32       = 0,
        GGML_TYPE_F16       = 1,
        GGML_TYPE_Q4_0      = 2,
        GGML_TYPE_Q4_1      = 3,
        // GGML_TYPE_Q4_2   = 4, support has been removed
        // GGML_TYPE_Q4_3   = 5, support has been removed
        GGML_TYPE_Q5_0      = 6,
        GGML_TYPE_Q5_1      = 7,
        GGML_TYPE_Q8_0      = 8,
        GGML_TYPE_Q8_1      = 9,
        GGML_TYPE_Q2_K      = 10,
        GGML_TYPE_Q3_K      = 11,
        GGML_TYPE_Q4_K      = 12,
        GGML_TYPE_Q5_K      = 13,
        GGML_TYPE_Q6_K      = 14,
        GGML_TYPE_Q8_K      = 15,
        GGML_TYPE_IQ2_XXS   = 16,
        GGML_TYPE_IQ2_XS    = 17,
        GGML_TYPE_IQ3_XXS   = 18,
        GGML_TYPE_IQ1_S     = 19,
        GGML_TYPE_IQ4_NL    = 20,
        GGML_TYPE_IQ3_S     = 21,
        GGML_TYPE_IQ2_S     = 22,
        GGML_TYPE_IQ4_XS    = 23,
        GGML_TYPE_I8        = 24,
        GGML_TYPE_I16       = 25,
        GGML_TYPE_I32       = 26,
        GGML_TYPE_I64       = 27,
        GGML_TYPE_F64       = 28,
        GGML_TYPE_IQ1_M     = 29,
        GGML_TYPE_BF16      = 30,
        GGML_TYPE_Q4_0_4_4  = 31,
        GGML_TYPE_Q4_0_4_8  = 32,
        GGML_TYPE_Q4_0_8_8  = 33,
        GGML_TYPE_TQ1_0     = 34,  // mainline ternary 1.6875bpw (TriLM/BitNet); additive
        GGML_TYPE_TQ2_0     = 35,  // mainline ternary 2.0625bpw (TriLM/BitNet); additive
        GGML_TYPE_I2_S      = 36,  // So we are able to consume MS BitNet I2_S quants
        GGML_TYPE_MXFP4     = 39,  // so we are compatible with mainline
        // GGML_TYPE_NVFP4 = 40 is declared below (fork's mainline adoption); ik 30903876b declared it here.
        GGML_TYPE_Q1_0_G128 = 41,  // Bonsai 1-bit quants
        GGML_TYPE_Q2_0_G128 = 42,  // Bonsai 2-bit ternary quants (group 128)
        GGML_TYPE_F8_E4M3 = 43,    // OCP FP8 E4M3 (mainline 68d20b884, same id so mainline GGUF files load); 43 verified free in this tree
        // TurboQuant KV cache types. Ids 44-46 verified FREE in this tree by computing
        // effective values for every enum member, so existing quantized files stay readable.
        GGML_TYPE_TURBO3_0 = 44, // TurboQuant 3-bit KV cache: PolarQuant
        GGML_TYPE_TURBO4_0 = 45, // TurboQuant 4-bit KV cache: PolarQuant
        GGML_TYPE_TURBO2_0 = 46, // TurboQuant 2-bit KV cache: PolarQuant (no QJL)
        // Unsloth IQ1 narrow-grid types: IQ1_S's 8-element ternary codebook with a narrower index.
        // SOURCE-READ: unslothai/llama.cpp branch iq1-narrow-grids-b10775 commit c86ed2699
        // ggml/include/ggml.h (MIT) -- same ids 64-66, so GGUFs written by their quantizer load here.
        // Ids 64-66 verified FREE in this tree (no member in 47..96).
        GGML_TYPE_IQ1_XS   = 64, // 1.4375 bpw, 1024-entry grid, 10-bit index
        GGML_TYPE_IQ1_XXS  = 65, // 1.3125 bpw,  512-entry grid,  9-bit index
        GGML_TYPE_IQ1_XXXS = 66, // 1.1875 bpw,  256-entry grid,  8-bit index
        // OCP Microscaling (MX) block-scaled FP6/FP8, 32 elements + one E8M0 scale per block.
        // Ids 70-73: contiguous free run in 67..96, clear of 41/42 (Q1_0/Q2_0 remap in gguf.cpp), 44-46 and 64-66.
        GGML_TYPE_MXFP6_E2M3   = 70, // 25 B / 32 elem, 6.25 bpw, no Inf/NaN
        GGML_TYPE_MXFP6_E3M2   = 71, // 25 B / 32 elem, 6.25 bpw, no Inf/NaN
        GGML_TYPE_MXFP8_E4M3   = 72, // 33 B / 32 elem, 8.25 bpw, NaN (S.1111.111), no Inf
        GGML_TYPE_MXFP8_E5M2   = 73, // 33 B / 32 elem, 8.25 bpw, Inf + NaN
        // TurboQuant WHT-rotated Lloyd-Max WEIGHT types (TheTom/llama-cpp-turboquant ca28b4bc0, MIT). Their files write
        // these as 45 / 46, which are OUR TURBO4_0 / TURBO2_0: gguf.cpp retypes an on-disk 45 / 46 to 74 / 75 when the
        // tensor's byte span fits this geometry and not ours (same mechanism as the upstream Q2_0 id-42 remap).
        // Ids 74-75: parsed-enum free run 74..96 (47-63, 67-69 also free; 41/42, 44-46, 64-66, 70-73 taken).
        GGML_TYPE_TQ3_1S       = 74, // 16 B / 32 elem, 4.0 bpw: 2 fp16 half-block scales + 3-bit Lloyd-Max indices
        GGML_TYPE_TQ4_1S       = 75, // 20 B / 32 elem, 5.0 bpw: 2 fp16 half-block scales + 4-bit Lloyd-Max indices
        //
        //
        GGML_TYPE_Q8_0_X4 = 97,
        GGML_TYPE_Q8_1_X4 = 98,
        GGML_TYPE_Q8_2_X4 = 99,
        GGML_TYPE_Q6_0    = 133,
        GGML_TYPE_IQ1_BN  = 134,
        GGML_TYPE_IQ2_BN  = 135,
        GGML_TYPE_Q8_K64  = 136,
        GGML_TYPE_IQ2_K   = 137,
        GGML_TYPE_IQ3_K   = 138,
        GGML_TYPE_IQ4_K   = 139,
        GGML_TYPE_IQ5_K   = 140,
        GGML_TYPE_IQ6_K   = 141,
        // Prism ternary (ids match PrismML-Eng/llama.cpp)
        GGML_TYPE_PQ2_0   = 142,
        GGML_TYPE_PTQ1_0  = 143,
        GGML_TYPE_IQ4_KS  = 144,
        GGML_TYPE_IQ2_KS  = 145,
        GGML_TYPE_IQ4_KSS = 146,
        GGML_TYPE_Q8_K16  = 147,
        GGML_TYPE_Q8_K32  = 148,
        GGML_TYPE_Q8_KR8  = 149,
        GGML_TYPE_Q8_K128 = 150,
        GGML_TYPE_Q8_KV   = 151,
        GGML_TYPE_IQ5_KS  = 152,
        GGML_TYPE_IQ2_KT  = 153,
        GGML_TYPE_IQ3_KT  = 154,
        GGML_TYPE_IQ4_KT  = 155,
        GGML_TYPE_IQ3_KS  = 156,
        GGML_TYPE_IQ2_KL  = 157,
        GGML_TYPE_IQ1_KT  = 158,

        GGML_TYPE_Q1_0_G128_R8 = 159, // 8-row repacked Q1_0_G128 (Bonsai), vec_dot_type = Q8_K128
        GGML_TYPE_PQ2_0_R8   = 160, // 8-row repacked PQ2_0 (Prism ternary)
        GGML_TYPE_PTQ1_0_R8  = 161, // 8-row repacked PTQ1_0 (Prism ternary, base-3)
        GGML_TYPE_NVFP4_KS   = 168, // ik id 168 (ik 30903876b): NVFP4 blocks + one fp32 row scale; ids 162-167 are ik's TQ* trellis types the fork lacks

        GGML_TYPE_TQK6       = 162, // trellis-coded K=6, group 128 (Agention; on-disk id 145, remapped at load)
        GGML_TYPE_TQK7       = 163, // trellis-coded K=7, group 128 (Agention; on-disk id 146, remapped at load)
        GGML_TYPE_TQ2_T      = 164, // trellis-coded 2.125 bpw (Agention; on-disk id 144, remapped at load)
        GGML_TYPE_TQP5       = 165, // pair trellis K=5, group 128 (Agention; on-disk id 147, remapped at load)
        GGML_TYPE_TQP6       = 166, // pair trellis K=6, group 128 (Agention; on-disk id 148, remapped at load)
        GGML_TYPE_TQP7       = 167, // pair trellis K=7, group 128 (Agention; on-disk id 149, remapped at load)

        GGML_TYPE_Q4_0_R8    = 202,
        GGML_TYPE_Q5_0_R4    = 206,
        GGML_TYPE_Q8_0_R8    = 208,
        GGML_TYPE_Q2_K_R4    = 210,
        GGML_TYPE_Q3_K_R4    = 211,
        GGML_TYPE_Q4_K_R4    = 212,
        GGML_TYPE_Q5_K_R4    = 213,
        GGML_TYPE_Q6_K_R4    = 214,
        GGML_TYPE_IQ2_XXS_R4 = 216,
        GGML_TYPE_IQ2_XS_R4  = 217,
        GGML_TYPE_IQ3_XXS_R4 = 218,
        GGML_TYPE_IQ1_S_R4   = 219,
        GGML_TYPE_IQ4_NL_R4  = 220,
        GGML_TYPE_IQ3_S_R4   = 221,
        GGML_TYPE_IQ2_S_R4   = 222,
        GGML_TYPE_IQ4_XS_R8  = 223,
        GGML_TYPE_IQ1_M_R4   = 229,
        GGML_TYPE_BF16_R16   = 230,
        GGML_TYPE_Q6_0_R4    = 233,
        GGML_TYPE_IQ2_BN_R4  = 335,
        GGML_TYPE_IQ2_K_R4   = 337,
        GGML_TYPE_IQ3_K_R4   = 338,
        GGML_TYPE_IQ4_K_R4   = 339,
        GGML_TYPE_IQ5_K_R4   = 340,
        GGML_TYPE_IQ4_KS_R4  = 344,
        GGML_TYPE_IQ4_KS_R16 = 345,
        GGML_TYPE_IQ3_KS_R16 = 346,
        GGML_TYPE_IQ5_KS_R16 = 347,
        GGML_TYPE_IQ5_KS_R4  = 352,
        GGML_TYPE_MXFP4_R8   = 353,
        GGML_TYPE_Q2_0_R8    = 354, // 8-row repacked Q2_0
        GGML_TYPE_Q8_0_R16   = 396,
        GGML_TYPE_Q8_K_R16   = 397,
        GGML_TYPE_Q8_KV_R8   = 398,
        GGML_TYPE_Q8_K_R8    = 399,
        GGML_TYPE_NVFP4        = 40, // adopted from upstream -- upstream's own number, free in ours
        GGML_TYPE_Q1_0         = 400, // adopted from upstream -- renumbered: upstream uses 41, which is OUR Q1_0_G128 in files on disk
        GGML_TYPE_Q2_0         = 401, // adopted from upstream -- renumbered: upstream uses 42, which is OUR Q2_0_G128 in files on disk
        GGML_TYPE_COUNT,
    };

    // [TAG_GGML_PREC]
    // this enum is used to declare the allowed numerical precision/data-types types that can be used during the compute of an op
    // the declared types can be:
    //  - result accumulation type
    //  - source tensor data representation type
    //  - etc.
    // the precision parameters are stored as ggml_tensor.op_params to the respective ops
    enum ggml_prec {
        GGML_PREC_UNDEFINED = 0,
        GGML_PREC_DEFAULT   = 0,  // note: deprecated, use GGML_PREC_UNDEFINED
        GGML_PREC_F32       = 10,
        GGML_PREC_BF16      = 15,
        GGML_PREC_F16       = 20,
        GGML_PREC_Q8        = 30,
        GGML_PREC_Q4        = 40,
    };

    enum ggml_backend_type {
        GGML_BACKEND_TYPE_CPU = 0,
        GGML_BACKEND_TYPE_GPU = 10,
        GGML_BACKEND_TYPE_GPU_SPLIT = 20,
    };

    // model file types
    enum ggml_ftype {
        GGML_FTYPE_UNKNOWN        = -1,
        GGML_FTYPE_ALL_F32        = 0,
        GGML_FTYPE_MOSTLY_F16     = 1,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0    = 2,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1    = 3,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_FTYPE_MOSTLY_Q8_0    = 7,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_0    = 8,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_1    = 9,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q2_K    = 10, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q3_K    = 11, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_K    = 12, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_K    = 13, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q6_K    = 14, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XS  = 16, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_S   = 18, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_NL  = 19, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_S   = 20, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_S   = 21, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_XS  = 22, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_M   = 23, // except 1d tensors
        GGML_FTYPE_MOSTLY_BF16    = 24, // except 1d tensors
        GGML_FTYPE_MOSTLY_MXFP4   = 25, // except 1d tensors, using 26 to be compatible with mainline
        GGML_FTYPE_MOSTLY_Q4_0_4_4 = 26, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0_4_8 = 27, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0_8_8 = 28, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q1_0_128 = 29, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q2_0_128 = 30, // except 1d tensors (Bonsai 2-bit ternary, g128)
        // ggml_ftype is INTERNAL -- a gguf stores the llama ftype, not this -- so these take the
        // next free values here rather than mainline's. The llama-level values DO match mainline.
        GGML_FTYPE_MOSTLY_TQ1_0    = 31, // except 1d tensors (mainline ternary 1.6875 bpw)
        GGML_FTYPE_MOSTLY_TQ2_0    = 32, // except 1d tensors (mainline ternary 2.0625 bpw)
        GGML_FTYPE_MOSTLY_IQ1_XS   = 33, // except 1d tensors (Unsloth IQ1 narrow grid, 1.4375 bpw)
        GGML_FTYPE_MOSTLY_IQ1_XXS  = 34, // except 1d tensors (Unsloth IQ1 narrow grid, 1.3125 bpw)
        GGML_FTYPE_MOSTLY_IQ1_XXXS = 35, // except 1d tensors (Unsloth IQ1 narrow grid, 1.1875 bpw)

        //
        GGML_FTYPE_MOSTLY_Q6_0    = 127, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_BN  = 128, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_BN  = 129, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_K   = 130, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_K   = 131, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_K   = 132, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ5_K   = 133, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ6_K   = 134, // except 1d tensors
        // depricated: GGML_FTYPE_MOSTLY_IQ2_TN  = 135, // except 1d tensors
        // depricated: GGML_FTYPE_MOSTLY_IQ1_TN  = 136, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_KS  = 137, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_KS  = 138, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_KSS = 139, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q8_KV   = 140, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ5_KS  = 141, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_KT  = 142, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_KT  = 143, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_KT  = 144, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_KS  = 145, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_KL  = 146, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_KT  = 147, // except 1d tensors
        GGML_FTYPE_MOSTLY_PQ2_0   = 148, // except 1d tensors (Prism ternary)
        GGML_FTYPE_MOSTLY_PTQ1_0  = 149, // except 1d tensors (Prism ternary)
        GGML_FTYPE_MOSTLY_TQK6    = 150, // except 1d tensors (trellis, group-128)
        GGML_FTYPE_MOSTLY_TQK7    = 151, // except 1d tensors (trellis, group-128)
        GGML_FTYPE_MOSTLY_TQ2_T   = 152, // except 1d tensors (trellis, group-128)
        GGML_FTYPE_MOSTLY_TQP5    = 153, // except 1d tensors (trellis, group-128)
        GGML_FTYPE_MOSTLY_TQP6    = 154, // except 1d tensors (trellis, group-128)
        GGML_FTYPE_MOSTLY_TQP7    = 155, // except 1d tensors (trellis, group-128)
        GGML_FTYPE_MOSTLY_NVFP4_KS = 156, // ik id (ik 30903876b)
                                         //
        GGML_FTYPE_MOSTLY_Q4_0_R8   = 202, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q8_0_R8   = 207, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_0_R4   = 208, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q2_K_R4   = 210, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q3_K_R4   = 211, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_K_R4   = 212, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_K_R4   = 213, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q6_K_R4   = 214, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XXS_R4= 215, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_XS_R4 = 216, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_XXS_R4= 217, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_S_R4  = 218, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_NL_R4 = 219, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_S_R4  = 220, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_S_R4  = 221, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_XS_R8 = 222, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ1_M_R4  = 223, // except 1d tensors
        GGML_FTYPE_MOSTLY_BF16_R16  = 224, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q6_0_R4   = 227, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_BN_R4 = 329, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ2_K_R4  = 330, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_K_R4  = 331, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_K_R4  = 332, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ5_K_R4  = 333, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_KS_R4 = 337, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ4_KS_R16= 338, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ3_KS_R16= 339, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ5_KS_R16= 340, // except 1d tensors
        GGML_FTYPE_MOSTLY_IQ5_KS_R4 = 341, // except 1d tensors
        GGML_FTYPE_MOSTLY_MXFP4_R8  = 342, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q2_0_R8   = 343, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q8_K_R16  = 397, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q8_KV_R8  = 398, // except 1d tensors
        GGML_FTYPE_MOSTLY_Q8_K_R8   = 399, // except 1d tensors
        GGML_FTYPE_MOSTLY_F8_E4M3   = 402, // except 1d tensors (upstream 29; relocated, fork ftype numbers differ)
    };

    // available tensor operations:
    enum ggml_op {
        GGML_OP_NONE = 0,

        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_ADD_ID,
        GGML_OP_ADD1,
        GGML_OP_ACC,
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_LOG,
        GGML_OP_SUM,
        GGML_OP_SUM_ROWS,
        GGML_OP_MEAN,
        GGML_OP_ARGMAX,
        GGML_OP_REPEAT,
        GGML_OP_REPEAT_BACK,
        GGML_OP_CONCAT,
        GGML_OP_SILU_BACK,
        GGML_OP_NORM, // normalize
        GGML_OP_RMS_NORM,
        GGML_OP_RMS_NORM_BACK,
        GGML_OP_GROUP_NORM,
        GGML_OP_FUSED_RMS_NORM,
        GGML_OP_FUSED_MUL_UNARY,
        GGML_OP_MULTI_ADD,

        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT_ID,
        GGML_OP_OUT_PROD,
        GGML_OP_FUSED_UP_GATE,
        GGML_OP_MOE_FUSED_UP_GATE,
        GGML_OP_MUL_MULTI_ADD,
        GGML_OP_HADAMARD,

        GGML_OP_SCALE,
        GGML_OP_SET,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_GET_ROWS_BACK,
        GGML_OP_SET_ROWS,
        GGML_OP_DIAG,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_DIAG_MASK_ZERO,
        GGML_OP_SOFT_MAX,
        GGML_OP_SOFT_MAX_BACK,
        GGML_OP_ROPE,
        GGML_OP_ROPE_BACK,
        GGML_OP_ROPE_CACHE,
        GGML_OP_ROPE_FAST,
        GGML_OP_CLAMP,
        GGML_OP_CONV_TRANSPOSE_1D,
        GGML_OP_IM2COL,
        GGML_OP_CONV_TRANSPOSE_2D,
        GGML_OP_POOL_1D,
        GGML_OP_POOL_2D,
        GGML_OP_UPSCALE, // nearest interpolate
        GGML_OP_PAD,
        GGML_OP_ARANGE,
        GGML_OP_TIMESTEP_EMBEDDING,
        GGML_OP_ARGSORT,
        GGML_OP_ARGSORT_THRESH,
        GGML_OP_GROUPED_TOPK,
        GGML_OP_LEAKY_RELU,
        GGML_OP_SOFTCAP,
        GGML_OP_SOFT_CAP_MAX,

        GGML_OP_CONV_2D,
        GGML_OP_CONV_2D_DW,

        GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_FLASH_ATTN_BACK,
        GGML_OP_SSM_CONV,
        GGML_OP_SSM_SCAN,
        GGML_OP_WIN_PART,
        GGML_OP_WIN_UNPART,
        GGML_OP_GET_REL_POS,
        GGML_OP_ADD_REL_POS,
        GGML_OP_UNARY,

        GGML_OP_CUMSUM,
        GGML_OP_L2_NORM,
        GGML_OP_TRI,
        GGML_OP_FILL,
        GGML_OP_SOLVE_TRI,
        GGML_OP_DELTA_NET,

        GGML_OP_MAP_UNARY,
        GGML_OP_MAP_BINARY,

        GGML_OP_MAP_CUSTOM1_F32,
        GGML_OP_MAP_CUSTOM2_F32,
        GGML_OP_MAP_CUSTOM3_F32,

        GGML_OP_MAP_CUSTOM1,
        GGML_OP_MAP_CUSTOM2,
        GGML_OP_MAP_CUSTOM3,

        GGML_OP_CROSS_ENTROPY_LOSS,
        GGML_OP_CROSS_ENTROPY_LOSS_BACK,

        GGML_OP_GLU,

        GGML_OP_REDUCE,
        GGML_OP_FAKE_CPY,
        GGML_OP_FUSED_NORM,
        GGML_OP_FUSED_RMS_RMS_ADD,
        GGML_OP_BLEND,
        GGML_OP_INDEXER_TOPK,
        GGML_OP_MASK_TOPK,
        GGML_OP_SINKHORN,
        GGML_OP_HC_PRE,
        GGML_OP_HC_POST,
        GGML_OP_MASK_TO_IDX,
        GGML_OP_LATENT_ATTN,
        GGML_OP_DS4_COMP,

        GGML_OP_TURBO_WHT,  // Walsh-Hadamard Transform for TurboQuant KV rotation

        // --- added for the upstream mtmd port (S69) ---------------------------------------
        // Appended at the END, immediately before GGML_OP_COUNT, so no existing op's numeric
        // value moves. Op indices are serialised in nothing we persist, but several of our
        // backends switch on them, and renumbering would silently repoint every one.
        GGML_OP_SIN,
        GGML_OP_COS,
        GGML_OP_ROLL,
        GGML_OP_PAD_REFLECT_1D,
        GGML_OP_COL2IM_1D,

        // --- upstream ops the merge left behind (added 2026-09-08) --------------------
        // The merge resolved this enum to the fork's side wholesale, so upstream's 17
        // newer ops never arrived -- while the upstream SOURCES that switch on them did.
        // Their CPU implementations are already present in ggml-cpu/ops.cpp; only the
        // names were missing. Appended at the END so no existing op's value moves.
        GGML_OP_COUNT_EQUAL,
        GGML_OP_IM2COL_BACK,
        GGML_OP_IM2COL_3D,
        GGML_OP_CONV_3D,
        GGML_OP_POOL_2D_BACK,
        GGML_OP_TOP_K,
        GGML_OP_RWKV_WKV6,
        GGML_OP_GATED_LINEAR_ATTN,
        GGML_OP_RWKV_WKV7,
        GGML_OP_GATED_DELTA_NET,
        GGML_OP_LIGHTNING_INDEXER,
        GGML_OP_DSV4_HC_PRE,
        GGML_OP_DSV4_HC_POST,
        GGML_OP_DSV4_HC_COMB,
        GGML_OP_OPT_STEP_ADAMW,
        GGML_OP_OPT_STEP_SGD,
        GGML_OP_CUSTOM,
        // FFD (arXiv 2609.00097) decode block selector -- appended after CUSTOM so no existing value moves.
        GGML_OP_FFD_SELECT,
        // FFD 2-bit K shadow maintenance (feeds GGML_OP_FFD_SELECT) -- appended, no existing value moves.
        GGML_OP_FFD_QUANT_K,
        // Mamba-2 gated group RMSNorm, rms_norm_group(y * silu(gate)) * w in one node
        // (ggml_gated_group_rms_norm below) -- appended, no existing value moves.
        GGML_OP_GATED_GROUP_RMS_NORM,
        // upstream top-k sparse attention mask (ggml_top_k_mask) -- appended, no existing value moves.
        GGML_OP_TOP_K_MASK,
        // fused multiply-add c + a*b (ggml_fma, ik #2794) -- appended, no existing value moves.
        GGML_OP_FMA,
        // overlap-ADD inverse of im2col_3d (ggml_col2im_3d) -- appended, no existing value moves.
        GGML_OP_COL2IM_3D,

        GGML_OP_COUNT,
    };

    enum ggml_unary_op {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP,
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU_ERF,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
        GGML_UNARY_OP_SWIGLU,
        GGML_UNARY_OP_SWIGLU_OAI,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_EXP,
        GGML_UNARY_OP_SOFTPLUS,
        GGML_UNARY_OP_SQRT_SOFTPLUS,

        // Upstream unary ops the merge left out (added 2026-09-08). Measured by set:
        // HEAD 20 enumerators == the fork parent's, upstream 23, upstream-only 6,
        // fork-only LOST 0. Their compute paths are already in ggml-cpu/ops.cpp, which
        // is what could not compile. Appended before COUNT so no existing value moves.
        GGML_UNARY_OP_EXPM1,
        GGML_UNARY_OP_FLOOR,
        GGML_UNARY_OP_CEIL,
        GGML_UNARY_OP_ROUND,
        GGML_UNARY_OP_TRUNC,
        GGML_UNARY_OP_XIELU,

        GGML_UNARY_OP_COUNT,
    };

    enum ggml_object_type {
        GGML_OBJECT_TYPE_TENSOR,
        GGML_OBJECT_TYPE_GRAPH,
        GGML_OBJECT_TYPE_WORK_BUFFER
    };

    enum ggml_log_level {
        GGML_LOG_LEVEL_NONE  = 0,
        GGML_LOG_LEVEL_DEBUG = 1,
        GGML_LOG_LEVEL_INFO  = 2,
        GGML_LOG_LEVEL_WARN  = 3,
        GGML_LOG_LEVEL_ERROR = 4,
        GGML_LOG_LEVEL_CONT  = 5, // continue previous log
    };

    enum ggml_tensor_flag {
        GGML_TENSOR_FLAG_INPUT  = 1,
        GGML_TENSOR_FLAG_OUTPUT = 2,
        GGML_TENSOR_FLAG_PARAM  = 4,
        GGML_TENSOR_FLAG_LOSS   = 8, // ...defines loss for numerical optimization (multiple loss tensors add up)
        // Adopted from upstream 2026-09-03 (ik_llama merge). Purely additive:
        // a new bit, no existing value changes, so no ABI impact.
        GGML_TENSOR_FLAG_COMPUTE = 16, // ...must be computed
    };

    enum ggml_tri_type {
        GGML_TRI_TYPE_LOWER,
        GGML_TRI_TYPE_UPPER,
        GGML_TRI_TYPE_LOWER_DIAG,
        GGML_TRI_TYPE_UPPER_DIAG,
    };

    // ggml object
    struct ggml_object {
        size_t offs;
        size_t size;

        struct ggml_object * next;

        enum ggml_object_type type;

        char padding[4];
    };

    static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

    // Adopted from upstream 2026-09-08. ggml-cpu.c:1263 reads GGML_HINT_SRC0_IS_HADAMARD
    // and nothing in this fork declared it.
    enum ggml_op_hint {
        GGML_HINT_NONE             = 0,
        GGML_HINT_SRC0_IS_HADAMARD = 1,
    };

    // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type         type;

        enum ggml_backend_type __backend;

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        // RE-ENABLED 2026-09-08, and widened to upstream's 8. ggml-backend-meta.cpp:429
        // computes `GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding)` to cache every
        // meaningful byte of a tensor and skip the trailing filler, so the member has to
        // EXIST for that translation unit to compile. Upstream uses 8; this struct is
        // 8-aligned either way, so the size is unchanged and only sizeof(padding) -- the
        // number of trailing bytes that cache deliberately skips -- differs.
        // 16, not upstream's 8. The assert in ggml.c ('ggml_tensor size must be a multiple of
        // GGML_MEM_ALIGN') is what settled the number: this fork's struct carries an
        // extra `grad` pointer upstream does not, so it was already aligned without
        // padding and adding 8 pushed it off the 16-byte multiple. Copy upstream's
        // INTENT (pad to alignment), not upstream's literal size.
        char padding[24];
    };

    static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);

    // Abort callback
    // If not NULL, called before ggml computation
    // If it returns true, the computation is aborted
    typedef bool (*ggml_abort_callback)(void * data);

    // NOTE: struct ggml_cplan -- the compute plan prepared for ggml_graph_compute(), see
    // https://github.com/ggerganov/ggml/issues/287 -- moved to ggml-cpu.h with the rest of the
    // CPU-side API, following upstream commit 9f409893 (#10144).

    enum ggml_cgraph_eval_order {
        GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
        GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
        GGML_CGRAPH_EVAL_ORDER_COUNT
    };

    typedef uint32_t ggml_bitset_t;

    struct ggml_hash_set {
        size_t size;
        ggml_bitset_t * used;
        struct ggml_tensor ** keys;
    };

    // computation graph
    struct ggml_cgraph {
        int size;
        int n_nodes;
        int n_leafs;
        int n_batch;

        struct ggml_tensor ** nodes;
        struct ggml_tensor ** grads;
        struct ggml_tensor ** grad_accs;
        struct ggml_tensor ** leafs;
        // Adopted from upstream 2026-09-03 (ik_llama merge): number of uses of each
        // tensor, indexed by HASH TABLE SLOT (not by node index). Required by
        // ggml_node_get_use_count / ggml_can_fuse_ext in ggml-impl.h.
        int32_t * use_counts;

        struct ggml_hash_set visited_hash_set;

        enum ggml_cgraph_eval_order order;

        // Adopted from upstream 2026-09-08. An optional identifier used to recognise the
        // same graph across rebuilds when two non-zero values match; 0 means unset and is
        // ignored. ggml-backend.cpp and ggml-backend-meta.cpp both read it.
        uint64_t uid;
    };

    // scratch buffer
    struct ggml_scratch {
        size_t offs;
        size_t size;
        void * data;
    };

    struct ggml_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };

    // numa strategies

    //
    // GUID
    //

    // GUID types
    typedef uint8_t ggml_guid[16];
    typedef ggml_guid * ggml_guid_t;

    GGML_API bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);

    // misc

    GGML_API void    ggml_time_init(void); // call this once at the beginning of the program
    GGML_API int64_t ggml_time_ms(void);
    GGML_API int64_t ggml_time_us(void);
    GGML_API int64_t ggml_cycles(void);
    GGML_API int64_t ggml_cycles_per_ms(void);

    // accepts a UTF-8 path, even on Windows
    GGML_API FILE *  ggml_fopen(const char * fname, const char * mode);


    GGML_API void    ggml_print_object (const struct ggml_object * obj);
    GGML_API void    ggml_print_objects(const struct ggml_context * ctx);

    // per-tensor activation (input) scale of an NVFP4_KS weight; stored in the last op_params slot, 0 = none
    GGML_API void  ggml_set_input_scale(struct ggml_tensor * tensor, float scale);
    GGML_API float ggml_get_input_scale(const struct ggml_tensor * tensor);
    // one MXFP4 row -> one NVFP4_KS row; returns false if a nonzero block fell outside the UE4M3 range (requantized)
    GGML_API bool  ggml_mxfp4_to_nvfp4_ks(const void * src, void * dst, int n_per_row);

    GGML_API GGML_CALL int64_t ggml_nelements   (const struct ggml_tensor * tensor);
    GGML_API GGML_CALL int64_t ggml_nrows       (const struct ggml_tensor * tensor);
    GGML_API GGML_CALL size_t  ggml_nbytes      (const struct ggml_tensor * tensor);
    GGML_API           size_t  ggml_nbytes_pad  (const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN

    // TODO: remove the following from the public API to avoid unnecessary assumptions about data layout
    GGML_API GGML_CALL int64_t ggml_blck_size(enum ggml_type type);
    GGML_API GGML_CALL int64_t ggml_row_blck_size(enum ggml_type type);         // granularity of a ROW length (32 for the KT tail types, else blck_size)
    GGML_API GGML_CALL size_t  ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
    GGML_API GGML_CALL size_t  ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row

    GGML_DEPRECATED(
    GGML_API double ggml_type_sizef(enum ggml_type type), // ggml_type_size()/ggml_blck_size() as float
    "use ggml_row_size() instead");

    GGML_API GGML_CALL const char * ggml_type_name(enum ggml_type type);
    GGML_API GGML_CALL const char * ggml_op_name  (enum ggml_op   op);
    GGML_API           const char * ggml_op_symbol(enum ggml_op   op);

    GGML_API GGML_CALL bool ggml_is_noop(const struct ggml_tensor * tensor);

    GGML_API           const char * ggml_unary_op_name(enum ggml_unary_op op);
    GGML_API GGML_CALL const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name

    GGML_API GGML_CALL size_t  ggml_element_size(const struct ggml_tensor * tensor);

    GGML_API GGML_CALL bool    ggml_is_quantized(enum ggml_type type);

    // Which weight types each iqk call site may hand to iqk. Coverage differs PER SITE (a dense
    // GEMM kernel does not imply the MoE row-mapped or fused up/gate path was exercised), so this
    // is a (type, site) question, not one deny-list. Every site that used to carry its own
    // `type != ...` chain asks here; tests/test-iqk-site-types.cpp pins each site's set.
    // Row S69-FORK-CARRY-IQK-SUPPORTS-TYPE-PREDICATE-01.
    enum ggml_iqk_site {
        GGML_IQK_SITE_DENSE = 0,          // ggml-cpu.c ggml_compute_forward_mul_mat (iqk_mul_mat_4d)
        GGML_IQK_SITE_MOE,                // ggml-cpu.c ggml_compute_forward_mul_mat_id (moe_iqk_ok)
        GGML_IQK_SITE_CPU_UP_GATE,        // ggml-cpu.c ggml_cpu_up_gate_iqk_ok (fused up/gate kernels)
        GGML_IQK_SITE_UP_GATE_OP,         // ggml.c ggml_moe_up_gate / ggml_fused_up_gate constructors
        GGML_IQK_SITE_GRAPH_FFN_UP_GATE,  // llama-build-context.cpp dense FFN fused up/gate node
        GGML_IQK_SITE_GRAPH_MOE_UP_GATE,  // llama-build-context.cpp MoE can_use_fmoe
        GGML_IQK_SITE_COUNT,
    };
    // true = the site may take `type`. Pure: no env, no CPU features (the call sites keep those).
    GGML_API bool ggml_iqk_supports_type(enum ggml_type type, enum ggml_iqk_site site);

    // TODO: temporary until model loading of ggml examples is refactored
    GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);

    GGML_API GGML_CALL bool ggml_is_transposed(const struct ggml_tensor * tensor);
    GGML_API GGML_CALL bool ggml_is_permuted  (const struct ggml_tensor * tensor);
    GGML_API GGML_CALL bool ggml_is_empty     (const struct ggml_tensor * tensor);
    GGML_API           bool ggml_is_view      (const struct ggml_tensor * tensor); // upstream API (merge 2026-11-04)
    GGML_API           bool ggml_is_scalar    (const struct ggml_tensor * tensor);
    GGML_API           bool ggml_is_vector    (const struct ggml_tensor * tensor);
    GGML_API           bool ggml_is_matrix    (const struct ggml_tensor * tensor);
    GGML_API           bool ggml_is_3d        (const struct ggml_tensor * tensor);
    GGML_API           int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars

    GGML_API GGML_CALL bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
    GGML_API GGML_CALL bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
    GGML_API GGML_CALL bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
    GGML_API GGML_CALL bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2

    // returns whether the tensor elements are allocated as one contiguous block of memory (no gaps, but permutation ok)
    GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);

    // true for tensor that is stored in memory as CxWxHxN and has been permuted to WxHxCxN
    GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);

    // true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
    GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);

    GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
    GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);

    GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);

    // use this to compute the memory overhead of a tensor
    GGML_API size_t ggml_tensor_overhead(void);

    GGML_API bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);

    // main

    GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);
    GGML_API void                  ggml_free(struct ggml_context * ctx);
    // ADDED 2026-09-09 (slice 52). The DEFINITION is already in this tree at
    // ggml/src/ggml.c:29380 and is byte-identical to upstream's; only this line was
    // lost. ggml/src/ggml-backend-meta.cpp:2001 calls it and could not compile:
    // "'ggml_reset' was not declared in this scope". Behaviour without a contract.
    GGML_API void                  ggml_reset(struct ggml_context * ctx);

    GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);

    GGML_API size_t  ggml_set_scratch (struct ggml_context * ctx, struct ggml_scratch scratch);
    GGML_API bool    ggml_get_no_alloc(struct ggml_context * ctx);
    GGML_API void    ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);

    GGML_API void *  ggml_get_mem_buffer     (const struct ggml_context * ctx);
    GGML_API size_t  ggml_get_mem_size       (const struct ggml_context * ctx);
    GGML_API size_t  ggml_get_max_tensor_size(const struct ggml_context * ctx);

    GGML_API struct ggml_tensor * ggml_new_tensor(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int    n_dims,
            const int64_t *ne);

    GGML_API struct ggml_tensor * ggml_new_tensor_1d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0);

    GGML_API struct ggml_tensor * ggml_new_tensor_2d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1);

    GGML_API struct ggml_tensor * ggml_new_tensor_3d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    GGML_API struct ggml_tensor * ggml_new_tensor_4d(
            struct ggml_context * ctx,
            enum   ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);


    GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
    GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);

    // Context tensor enumeration and lookup
    GGML_API struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
    GGML_API struct ggml_tensor * ggml_get_next_tensor (const struct ggml_context * ctx, struct ggml_tensor * tensor);
    GGML_API struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);

    GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);

    // Converts a flat index into coordinates
    GGML_API void    ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);





    GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);
    GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);

    GGML_API GGML_CALL enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);

    GGML_API const char *         ggml_get_name   (const struct ggml_tensor * tensor);
    GGML_API struct ggml_tensor * ggml_set_name   (      struct ggml_tensor * tensor, const char * name);
    GGML_ATTRIBUTE_FORMAT(2, 3)
    GGML_API struct ggml_tensor * ggml_format_name(      struct ggml_tensor * tensor, const char * fmt, ...);

    //
    // operations on tensors with backpropagation
    //

    GGML_API struct ggml_tensor * ggml_dup(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_dup_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_add(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_add_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_add_cast(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            enum   ggml_type      type);

    // dst[i0, i1, i2] = a[i0, i1, i2] + b[i0, ids[i1, i2]]
    // an id of -1 adds nothing: dst[i0, i1, i2] = a[i0, i1, i2]
    GGML_API struct ggml_tensor * ggml_add_id(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * ids);

    GGML_API struct ggml_tensor * ggml_add1(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_add1_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_multi_add(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int n_experts);

    GGML_API struct ggml_tensor * ggml_mul_multi_add(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // Source may be F32, F16, or a supported quantized type; output is always F32.
    GGML_API struct ggml_tensor * ggml_hadamard(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n);

    GGML_API struct ggml_tensor * ggml_signed_hadamard(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * signs,
            int                   n, int before);

    // dst = a
    // view(dst, nb1, nb2, nb3, offset) += b
    // return dst
    GGML_API struct ggml_tensor * ggml_acc(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_acc_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_sub(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_sub_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_mul(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_mul_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_fused_mul_unary(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            enum ggml_unary_op    op);

    GGML_API struct ggml_tensor * ggml_fused_mul_unary_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            enum ggml_unary_op    op);

    GGML_API struct ggml_tensor * ggml_div(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_div_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_sqr(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sqr_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sqrt(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sqrt_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_log(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_log_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_softplus(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_softplus_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sqrt_softplus(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sqrt_softplus_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // return scalar
    GGML_API struct ggml_tensor * ggml_sum(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
    GGML_API struct ggml_tensor * ggml_sum_rows(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);
    GGML_API struct ggml_tensor * ggml_sum_rows_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   dim);

    GGML_API struct ggml_tensor * ggml_cumsum(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // mean along rows
    GGML_API struct ggml_tensor * ggml_mean(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // argmax along rows
    GGML_API struct ggml_tensor * ggml_argmax(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);


    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    GGML_API struct ggml_tensor * ggml_repeat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // repeat a to specified shape
    GGML_API struct ggml_tensor * ggml_repeat_4d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
                       int64_t    ne0,
                       int64_t    ne1,
                       int64_t    ne2,
                       int64_t    ne3);

    // sums repetitions in a into shape of b
    GGML_API struct ggml_tensor * ggml_repeat_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // concat a and b along dim
    // used in stable-diffusion
// TurboQuant Walsh-Hadamard Transform for KV cache compression (unified ik_llama build)
    // direction: 0 = forward (pre-quantize rotation), 1 = inverse (post-dequantize rotation)
    // group_size: WHT group size (32, 64, or 128); scale_inv: per-channel InnerQ scaling (may be NULL)
    GGML_API struct ggml_tensor * ggml_turbo_wht(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,          // F32 tensor to transform
            struct ggml_tensor  * scale_inv,  // InnerQ scale_inv (may be NULL)
            int                   direction,
            int                   group_size);

    GGML_API struct ggml_tensor * ggml_concat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   dim);
    GGML_API struct ggml_tensor * ggml_concat_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * result,
            int                   dim);

    GGML_API struct ggml_tensor * ggml_abs(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_abs_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sgn(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sgn_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_neg(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_neg_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_step(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_step_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_tanh(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_tanh_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_elu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_elu_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_relu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_leaky_relu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a, float negative_slope, bool inplace);

    GGML_API struct ggml_tensor * ggml_relu_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sigmoid(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_sigmoid_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_gelu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_gelu_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // GELU using erf (error function) when possible
    // some backends may fallback to approximation based on Abramowitz and Stegun formula
    GGML_API struct ggml_tensor * ggml_gelu_erf(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_gelu_erf_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_gelu_quick(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_gelu_quick_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_silu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_silu_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_swiglu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_swiglu_oai(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 alpha,
            float                 limit);

    GGML_API struct ggml_tensor * ggml_swiglu_clamp(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 limit);

    // c + a*b. a and b must be broadcastable to c
    GGML_API struct ggml_tensor * ggml_fma(
            struct ggml_context * ctx,
            struct ggml_tensor  * c,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // c += a*b. a and b must be broadcastable to c
    GGML_API struct ggml_tensor * ggml_fma_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * c,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    //
    // *** from mainline begin ***
    //
    enum ggml_glu_op {
        GGML_GLU_OP_REGLU,
        GGML_GLU_OP_GEGLU,
        GGML_GLU_OP_SWIGLU,
        GGML_GLU_OP_SWIGLU_OAI,
        GGML_GLU_OP_GEGLU_ERF,
        GGML_GLU_OP_GEGLU_QUICK,
        GGML_GLU_OP_SWIGLU_CLAMP,

        GGML_GLU_OP_COUNT,
    };
    GGML_API const char * ggml_glu_op_name(enum ggml_glu_op op);
    GGML_API enum ggml_glu_op ggml_get_glu_op(const struct ggml_tensor * tensor);

    GGML_API struct ggml_tensor * ggml_glu(
            struct ggml_context * ctx,
             struct ggml_tensor * a,
             enum ggml_glu_op     op,
             bool                 swapped);

    GGML_API struct ggml_tensor * ggml_glu_split(
            struct ggml_context * ctx,
             struct ggml_tensor * a,
             struct ggml_tensor * b,
             enum ggml_glu_op     op);

    GGML_API struct ggml_tensor * ggml_reglu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_reglu_swapped(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_reglu_split(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_geglu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_geglu_swapped(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_geglu_split(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_swiglu_swapped(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_swiglu_split(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_geglu_erf(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_geglu_erf_swapped(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_geglu_erf_split(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_geglu_quick(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_geglu_quick_swapped(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_geglu_quick_split(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    //
    // *** from mainline end ***
    //


    // a - x
    // b - dy
    GGML_API struct ggml_tensor * ggml_silu_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // hardswish(x) = x * relu6(x + 3) / 6
    GGML_API struct ggml_tensor * ggml_hardswish(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // hardsigmoid(x) = relu6(x + 3) / 6
    GGML_API struct ggml_tensor * ggml_hardsigmoid(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_exp(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_exp_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // normalize along rows
    GGML_API struct ggml_tensor * ggml_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_norm_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_rms_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_rms_norm_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_fused_rms_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_fused_rms_norm_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 eps);

    // a must be contiguous. b must have one row and b->ne[0] must be eqal a->ne[0]
    // a and b must be contiguous
    // b must be GGML_TYPE_F32
    // a must be GGML_TYPE_F32 || GGML_TYPE_F16 || GGML_TYPE_BF16 || GGML_TYPE_Q8_0
    GGML_API struct ggml_tensor * ggml_fused_grouped_rms_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 eps,
            int                   ngroups);

    // Mamba-2 / Nemotron-H / Granite-4.0-H gated group RMSNorm (RMSNormGated, norm_before_gate=False),
    // one node instead of silu + mul + rms_norm + mul. For every row r of a and every group g of
    // n_group contiguous, equally sized groups of that row (gs = a->ne[0]/n_group, c = g*gs + j):
    //     t[c]   = silu(gate[r, c]) * a[r, c]
    //     scale  = 1 / sqrt(sum_j t[g*gs + j]^2 / gs + eps)
    //     out[r, c] = (t[c] * scale) * w[c]
    // a, gate: F32, same shape, elements of a row contiguous (nb[0] == 4); rows may be strided, so
    //          gate can be a view of a fused in-projection without a ggml_cont.
    // w:       F32, contiguous, ggml_nelements(w) == a->ne[0] (a {gs, n_group} tensor is accepted as is).
    // result:  F32, contiguous, same shape as a.
    // In F32 the intermediate cast of the reference (Modular MAX gated_group_rmsnorm.mojo:32-38,
    // cast(t*scale, y.dtype) before the fp32 weight multiply) is the identity.
    GGML_API struct ggml_tensor * ggml_gated_group_rms_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * gate,
            struct ggml_tensor  * w,
            float                 eps,
            int                   n_group);

    GGML_API struct ggml_tensor * ggml_fused_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_fused_norm_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 eps);

    // group normalize along ne0*ne1*n_groups
    // used in stable-diffusion
    GGML_API struct ggml_tensor * ggml_group_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_groups,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_group_norm_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_groups,
            float                 eps);

    // l2 normalize along rows
    GGML_API struct ggml_tensor * ggml_l2_norm(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_l2_norm_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_fused_rms_rms_add(
            struct ggml_context * ctx,
            struct ggml_tensor  * x1,
            struct ggml_tensor  * c1,
            struct ggml_tensor  * x2,
            struct ggml_tensor  * c2,
            float                 eps);

    // a - x
    // b - dy
    GGML_API struct ggml_tensor * ggml_rms_norm_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 eps);

    // [TAG_GGML_PREC]
    // set the minimum required accumulator type for the implementation to use during the compute
    // for example:
    //  - GGML_PREC_F32  - requires accumulation of the results in F32
    //  - GGML_PREC_BF16 - can accumulate the results in BF16, F32
    //  - GGML_PREC_F16  - can accumulate the results in F16, F32
    //  - GGML_PREC_Q8   - not allowed
    //  - GGML_PREC_Q4   - not allowed
    //
    // return false on faliure
    GGML_API bool ggml_prec_set_acc(
            struct ggml_tensor * a,
            enum ggml_prec       prec);

    // [TAG_GGML_PREC]
    // set the smallest rank that the implementation can use to internally convert the src[idx] data to
    // ranks in decreasing order:
    //  - GGML_PREC_F32  - GGML_TYPE_F32
    //  - GGML_PREC_BF16 - GGML_TYPE_BF16
    //  - GGML_PREC_F16  - GGML_TYPE_F16,
    //  - GGML_PREC_Q8   - GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q8_K, etc.
    //  - GGML_PREC_Q4   - GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_NVFP4, GGML_TYPE_MXFP4, etc.
    //
    // for example:
    //   - ggml_prec_set_src(a, GGML_PREC_Q8, 1):
    //     - allows the implementation to quantize F32, BF16, F16 data of src[1] down to GGML_TYPE_Q8_0
    //     - cannot quantize it down to GGML_TYPE_Q4_0 or GGML_TYPE_NVFP4
    //   - ggml_prec_set_src(a, GGML_PREC_Q4, 1):
    //     - allows the implementation to quantize F32, BF16, F16 data of src[1] down to 4-bit datatypes such as GGML_TYPE_Q4_K, GGML_TYPE_NVFP4 etc.
    //
    // return false on faliure
    GGML_API bool ggml_prec_set_src(
            struct ggml_tensor * a,
            enum ggml_prec       prec,
            int                  idx);

    // [TAG_GGML_PREC]
    // get the accumulator / src[idx] precision set by the functions above
    // return GGML_PREC_UNDEFINED when the op does not carry the requested precision
    GGML_API enum ggml_prec ggml_prec_get_acc(
            const struct ggml_tensor * a);

    GGML_API enum ggml_prec ggml_prec_get_src(
            const struct ggml_tensor * a,
            int                        idx);

    // A: k columns, n rows => [ne03, ne02, n, k]
    // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
    // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
    GGML_API struct ggml_tensor * ggml_mul_mat(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_mul_mat_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * result);

    // change the precision of a matrix multiplication
    // set to GGML_PREC_F32 for higher precision (useful for phi-2)
    GGML_DEPRECATED(GGML_API void ggml_mul_mat_set_prec(
            struct ggml_tensor * a,
            enum ggml_prec       prec),
        "use ggml_prec_set_acc() instead");

    // indirect matrix multiplication
    // an id of -1 skips the slot and zeroes the matching row of the result
    //
    // [TAG_MUL_MAT_ID_DUPLICATE_ID]
    // note: this op has undefined behaviour if a given `a` is referenced more than once for a given `b`
    //       (i.e. duplicate experts per token are not guaranteed to be supported in all backends)
    // TODO: add backend support: https://github.com/ggml-org/llama.cpp/pull/29998#issuecomment-6153742533
    GGML_API struct ggml_tensor * ggml_mul_mat_id(
            struct ggml_context * ctx,
            struct ggml_tensor  * as,
            struct ggml_tensor  * b,
            struct ggml_tensor  * ids);

    // MoE up + gate + unary
    GGML_API struct ggml_tensor * ggml_moe_up_gate(
            struct ggml_context * ctx,
            struct ggml_tensor  * as_up,
            struct ggml_tensor  * as_gate,
            struct ggml_tensor  * b,
            struct ggml_tensor  * ids,
            enum ggml_unary_op    op);

    GGML_API struct ggml_tensor * ggml_moe_up_gate_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a_up,
            struct ggml_tensor  * a_gate,
            struct ggml_tensor  * b,
            struct ggml_tensor  * ids,
            struct ggml_tensor  * a_up_b,
            struct ggml_tensor  * a_gate_b,
            enum ggml_unary_op    op);

    GGML_API struct ggml_tensor * ggml_fused_up_gate(
            struct ggml_context * ctx,
            struct ggml_tensor  * up,
            struct ggml_tensor  * gate,
            struct ggml_tensor  * b,
            enum ggml_unary_op    op);

    // A: m columns, n rows,
    // B: p columns, n rows,
    // result is m columns, p rows
    GGML_API struct ggml_tensor * ggml_out_prod(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    GGML_API struct ggml_tensor * ggml_scale(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 s);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_scale_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 s);

    // x = s * a + b
    GGML_API struct ggml_tensor * ggml_scale_bias(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 s,
        float                 b);

    GGML_API struct ggml_tensor * ggml_scale_bias_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        float                 s,
        float                 b);

    GGML_API struct ggml_tensor * ggml_softcap(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 s_before,
            float                 s_after);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_softcap_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 s_before,
            float                 s_after);

    GGML_API struct ggml_tensor * ggml_softcap_max(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * mask,
            float                 scale,
            float                 max_bias,
            float                 s_before,
            float                 s_after);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_softcap_max_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * mask,
            float                 scale,
            float                 max_bias,
            float                 s_before,
            float                 s_after);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    GGML_API struct ggml_tensor * ggml_set(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    GGML_API struct ggml_tensor * ggml_set_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_set_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_set_1d_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    GGML_API struct ggml_tensor * ggml_set_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    GGML_API struct ggml_tensor * ggml_set_2d_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // a -> b, return view(b)
    GGML_API struct ggml_tensor * ggml_cpy(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_cast(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum   ggml_type      type);

    // make contiguous
    GGML_API struct ggml_tensor * ggml_cont(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // make contiguous, with new shape
    GGML_API struct ggml_tensor * ggml_cont_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0);

    GGML_API struct ggml_tensor * ggml_cont_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    GGML_API struct ggml_tensor * ggml_cont_3d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    GGML_API struct ggml_tensor * ggml_cont_4d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0);

    GGML_API struct ggml_tensor * ggml_reshape_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    GGML_API struct ggml_tensor * ggml_reshape_3d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    GGML_API struct ggml_tensor * ggml_reshape_4d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    GGML_API struct ggml_tensor * ggml_reshape_4d_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum ggml_type        type,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);


    // offset in bytes
    GGML_API struct ggml_tensor * ggml_view_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_view_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_view_3d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_view_4d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                nb3,
            size_t                offset);

    GGML_API struct ggml_tensor * ggml_permute(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
    GGML_API struct ggml_tensor * ggml_transpose(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // supports 3D: a->ne[2] == b->ne[1]
    GGML_API struct ggml_tensor * ggml_get_rows(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_get_rows_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * c);

    GGML_API struct ggml_tensor * ggml_get_rows_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            bool                  same_type,
            bool                  dim0);

    // a TD  [n_embd, ne1,    ne2,    ne3]
    // b TS  [n_embd, n_rows, ne02,   ne03] | ne02 == ne2, ne03 == ne3
    // c I64 [n_rows, ne11,   ne12,   1]    | c[i] in [0, ne1)
    //
    // undefined behavior if destination rows overlap
    //
    // broadcast:
    //   ne2 % ne11 == 0
    //   ne3 % ne12 == 0
    //
    // return view(a)
    GGML_API struct ggml_tensor * ggml_set_rows(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,  // destination
            struct ggml_tensor  * b,  // source
            struct ggml_tensor  * c); // row indices

    GGML_API struct ggml_tensor * ggml_diag(
        struct ggml_context     * ctx,
        struct ggml_tensor      * a);

    // set elements above the diagonal to -INF
    GGML_API struct ggml_tensor * ggml_diag_mask_inf(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_diag_mask_inf_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_past);

    // set elements above the diagonal to 0
    GGML_API struct ggml_tensor * ggml_diag_mask_zero(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_diag_mask_zero_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   n_past);

    GGML_API struct ggml_tensor * ggml_soft_max(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_soft_max_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // fused soft_max(a*scale + mask*(ALiBi slope))
    // mask is optional
    // max_bias = 0.0f for no ALiBi
    GGML_API struct ggml_tensor * ggml_soft_max_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * mask,
            float                 scale,
            float                 max_bias);

    GGML_API void ggml_soft_max_add_sinks(
            struct ggml_tensor * a,
            struct ggml_tensor * sinks);


    // ADDED 2026-09-09 (slice 53) -- see the note beside the definitions in
    // ggml.c. Back-op constructors whose kernels already ship here.
    GGML_API struct ggml_tensor * ggml_soft_max_ext_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        float                 scale,
        float                 max_bias);
    GGML_API struct ggml_tensor * ggml_rope_multi_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        struct ggml_tensor  * c,
        int                   n_dims,
        int                   sections[4],
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);
    GGML_API struct ggml_tensor * ggml_im2col_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int64_t             * ne,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1,
        bool                  is_2D);

    GGML_API struct ggml_tensor * ggml_soft_max_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_soft_max_back_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // rotary position embedding
    // if mode & 1 == 1, skip n_past elements (NOT SUPPORTED)
    // if mode & 2 == 1, GPT-NeoX style
    //
    // b is an int32 vector with size a->ne[2], it contains the positions

    // PORTED FROM UPSTREAM 2026-09-09 (279b4bfaf^2) -- CONSTRUCTORS ONLY.
    // The COMPUTE for every one of these is ALREADY in this tree: the enum, the ggml-cpu kernel and
    // the ggml-cuda kernel are all present and were verified present before these were added. What
    // was missing is the front door -- the small function that builds the node and stamps the enum.
    // Backends dispatch on the ENUM (ggml_compute_forward_<enum>), never on the constructor name,
    // which is why searching for these names found nothing in ggml-cpu/ggml-cuda and made the gap
    // look like a kernel port. It was not.
    // ggml_ssm_scan: THAT SEPARATE CHANGE HAPPENED, 2026-09-20. This note used to say it was not
    // ported here because it was a signature change with existing callers rather than an
    // addition. That was correct, and it was done on its own: the declaration below now takes
    // ids and K.
    // RESEARCH-AFTER-RESULT: ggml/src/ggml-cpu/ops.cpp:9912-9937, read in full once the arity
    // mismatch was in hand. ggml_compute_forward_ssm_scan_f32 already binds
    // `const ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}` and
    // `const int64_t K = ggml_get_op_params_i32(dst, 0)`, then asserts `GGML_ASSERT(K >= 1)` and
    // `ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst)`. So this was never a kernel
    // port: every backend already consumed both. Only the constructor kept the old shape, and
    // because it never called ggml_set_op_params, K read 0 and that same assert fired on the
    // first compute -- measured at ops.cpp:9937 with op_params[0] = 0. Build-valid, run-time
    // dead.
    GGML_API struct ggml_tensor * ggml_lightning_indexer(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * weights,
        struct ggml_tensor  * mask);

    // FFD decode block selector (design from qluoluo/faster-flash-decoding @ ca09458, Apache-2.0,
    // ffd_core/kernels/paged_decode_kernel.py:14-478; reimplemented, not copied).
    // BENCH-PENDING: no speed is claimed for this op; "faster" above is only the upstream repo name.
    // Scores every block of `block` KV positions against a 2-bit copy of K and returns, per
    // (block, query token, KV head, stream), 1 = keep / 0 = prune.
    //   q        F32 [D, n_tokens, n_head,    ns]   (flash_attn_ext layout)
    //   k2       I8  [D/4, n_kv,   n_head_kv, ns]   2-bit codes, 4 per byte, element j at bits 2*(j%4)
    //   k2_scale F32 [D, n_blocks, n_head_kv, ns]   per-block per-channel scale; k ~= (code - 1.5)*scale
    //   mask     F16 [n_kv, >= n_tokens] or NULL    added to the score, as flash_attn_ext does
    //   result   I32 [n_blocks, n_tokens, n_head_kv, ns], n_blocks = ceil(n_kv/block)
    // score = scale*q.k~ + mask, in NATURAL-LOG units. For each query head h of the GQA group:
    // th_h = max(block-max of the first block, block-max of the last block) - delta. A block is kept
    // when ANY head of the group reaches its th_h. A fully masked block is always pruned (exact); any
    // other last block is always kept. delta must be >= 0 (a NaN delta fails that assert). delta =
    // +INFINITY keeps every block that is not fully masked, so the selector then reproduces dense
    // attention exactly.
    // NOTE: upstream FFD applies delta in LOG2 units (scores * 1/ln2 before th = max - delta); its
    // delta=5 equals 5*ln(2) = 3.466 here.
    GGML_API struct ggml_tensor * ggml_ffd_select(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k2,
        struct ggml_tensor  * k2_scale,
        struct ggml_tensor  * mask,
        int                   block,
        float                 scale,
        float                 delta);

    // FFD K shadow maintenance: re-quantise, IN PLACE, every `block`-cell block that k_idxs touched, so the 2-bit
    // copy ggml_ffd_select reads stays in step with the K cache. Quantiser from qluoluo/faster-flash-decoding @ ca09458
    // (Apache-2.0) ffd_core/modeling/quantized_cache.py:111-192 (quantize_symmetric_blocks), reimplemented, not copied.
    // BENCH-PENDING: no speed is claimed for this op; "faster" is only the upstream repo name.
    //   k        F16/BF16/F32 [D*n_head_kv, kv_size*ns]  K rows (cells) as cpy_k/set_rows writes them; row r is
    //                                                     stream r/kv_size, cell r%kv_size. Pass the cpy_k RESULT so
    //                                                     this op is ordered after the write by a real src edge.
    //   k_idxs   I64 [n]                                  rows written this ubatch (the set_rows indices)
    //   k2       I8  [D/4, kv_size, n_head_kv, ns]        full shadow, written in place (the result is a view of it)
    //   k2_scale F32 [D, ceil(kv_size/block), n_head_kv, ns]   written in place
    // For every touched (stream, block), head h, channel d, over the cells of the block (the last may be short):
    //   scale = max(absmax_c |K| / 1.5, 1e-8)
    //   code  = clamp(round-half-even(K/scale + 1.5), 0, 3), 4 codes per byte, element d at bits 2*(d%4)
    // Blocks no index touches are not written. Consume the RESULT (not k2 itself) in ggml_ffd_select so the selector is
    // ordered after this op; k2_scale is written by the same op, so that one edge orders both.
    GGML_API struct ggml_tensor * ggml_ffd_quant_k(
        struct ggml_context * ctx,
        struct ggml_tensor  * k,
        struct ggml_tensor  * k_idxs,
        struct ggml_tensor  * k2,
        struct ggml_tensor  * k2_scale,
        int                   head_dim,
        int                   block);
    GGML_API struct ggml_tensor * ggml_dsv4_hc_pre(
            struct ggml_context * ctx,
            struct ggml_tensor  * x,
            struct ggml_tensor  * weights);
    GGML_API struct ggml_tensor * ggml_dsv4_hc_comb(
            struct ggml_context * ctx,
            struct ggml_tensor  * mixes,
            struct ggml_tensor  * scale,
            struct ggml_tensor  * base,
            float                 eps,
            int32_t               n_iter);
    // hc_pre with a per-element gate (Qwen3.8-Flash-Next): gate [n_embd, hc, n_tokens]
    //   result[i, t] = scale*sum_h x[i, h, t]*sigmoid(gate[i, h, t])
    GGML_API struct ggml_tensor * ggml_dsv4_hc_pre_gated(
            struct ggml_context * ctx,
            struct ggml_tensor  * x,
            struct ggml_tensor  * gate,
            float                 scale);

    GGML_API struct ggml_tensor * ggml_dsv4_hc_post(
            struct ggml_context * ctx,
            struct ggml_tensor  * x,
            struct ggml_tensor  * residual,
            struct ggml_tensor  * post,
            struct ggml_tensor  * comb);
    GGML_API struct ggml_tensor * ggml_xielu(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float alpha_n,
            float alpha_p,
            float beta,
            float eps);
    GGML_API struct ggml_tensor * ggml_rope_ext_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a, // gradients of ggml_rope result
            struct ggml_tensor  * b, // positions
            struct ggml_tensor  * c, // freq factors
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);
    GGML_API struct ggml_tensor * ggml_gated_delta_net(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            struct ggml_tensor  * g,
            struct ggml_tensor  * beta,
            struct ggml_tensor  * state,
            int64_t               K);

    // upstream API (merge 2026-11-04): builders for the CEIL / FLOOR unary ops this fork already computes
    GGML_API struct ggml_tensor * ggml_ceil(struct ggml_context * ctx, struct ggml_tensor * a);
    GGML_API struct ggml_tensor * ggml_ceil_inplace(struct ggml_context * ctx, struct ggml_tensor * a);
    GGML_API struct ggml_tensor * ggml_floor(struct ggml_context * ctx, struct ggml_tensor * a);
    GGML_API struct ggml_tensor * ggml_floor_inplace(struct ggml_context * ctx, struct ggml_tensor * a);

    GGML_API struct ggml_tensor * ggml_round(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_gated_linear_attn(
            struct ggml_context * ctx,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            struct ggml_tensor  * q,
            struct ggml_tensor  * g,
            struct ggml_tensor  * state,
            float scale);

    GGML_API struct ggml_tensor * ggml_rwkv_wkv6(
            struct ggml_context * ctx,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            struct ggml_tensor  * r,
            struct ggml_tensor  * tf,
            struct ggml_tensor  * td,
            struct ggml_tensor  * state);

    GGML_API struct ggml_tensor * ggml_rwkv_wkv7(
            struct ggml_context * ctx,
            struct ggml_tensor  * r,
            struct ggml_tensor  * w,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * state);

    GGML_API struct ggml_tensor * ggml_conv_1d_dw_ph(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,   // convolution kernel
            struct ggml_tensor  * b,   // data
            int                   s0,  // stride
            int                   d0);

    GGML_API struct ggml_tensor * ggml_rope(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   n_dims,
            int                   mode);

    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_rope_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   n_dims,
            int                   mode);

    // custom RoPE
    // c is freq factors (e.g. phi3-128k), (optional)
    GGML_API struct ggml_tensor * ggml_rope_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * c,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);

    GGML_API struct ggml_tensor * ggml_rope_multi(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * c,
            int                   n_dims,
            int                   sections[GGML_MROPE_SECTIONS],
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);


    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_rope_ext_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * c,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);

    GGML_API struct ggml_tensor * ggml_rope_multi_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * c,
            int                   n_dims,
            int                   sections[GGML_MROPE_SECTIONS],
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);


    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_rope_custom(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow),
        "use ggml_rope_ext instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_rope_custom_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow),
        "use ggml_rope_ext_inplace instead");

    // compute correction dims for YaRN RoPE scaling
    GGML_CALL void ggml_rope_yarn_corr_dims(
        int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]);

    // Same, with HF's `truncate` switch. truncate=true is ggml_rope_yarn_corr_dims (floor/ceil of the
    // correction range, HF default and every pre-2026 YaRN model); truncate=false keeps the range
    // fractional (HF rope_parameters["truncate"] == false: gpt-oss, Kimi-style configs).
    GGML_API GGML_CALL void ggml_rope_yarn_corr_dims_ext(
        int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, bool truncate, float dims[2]);

    // Per-node YaRN truncate flag, stored in the (otherwise unused) n_ctx slot op_params[3], bit 0 set =
    // NO truncation, so a node nobody touched (all zero) keeps the historical floor/ceil. Backends read it
    // with ggml_rope_yarn_truncate(). Mutates and returns `a` (a GGML_OP_ROPE / ROPE_BACK / ROPE_CACHE node).
    GGML_API struct ggml_tensor * ggml_rope_set_yarn_truncate(struct ggml_tensor * a, bool truncate);
    GGML_API GGML_CALL bool       ggml_rope_yarn_truncate(const struct ggml_tensor * rope_node);

    // Context-wide default stamped on every rope node created AFTER the call (the graph builders create
    // ~470 rope nodes, so a per-model switch cannot be threaded through each call). Default true.
    GGML_API void ggml_set_rope_yarn_truncate(struct ggml_context * ctx, bool truncate);

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    GGML_API struct ggml_tensor * ggml_rope_back(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * c,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);

    GGML_API struct ggml_tensor * ggml_rope_cache(
            struct ggml_context * ctx,
            struct ggml_tensor  * b,
            struct ggml_tensor  * c,
            int                   ne0,
            int                   n_dims,
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);

    GGML_API struct ggml_tensor * ggml_rope_fast(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // clamp
    // in-place, returns view(a)
    GGML_API struct ggml_tensor * ggml_clamp(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 min,
            float                 max);

    GGML_API struct ggml_tensor * ggml_im2col(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                  s0,
            int                  s1,
            int                  p0,
            int                  p1,
            int                  d0,
            int                  d1,
            bool                 is_2D,
            enum ggml_type       dst_type);

    GGML_API struct ggml_tensor * ggml_conv_depthwise_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                  s0,
            int                  s1,
            int                  p0,
            int                  p1,
            int                  d0,
            int                  d1);

    GGML_API struct ggml_tensor * ggml_conv_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   s0,  // stride
            int                   p0,  // padding
            int                   d0); // dilation

    // conv_1d with padding = half
    // alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
    GGML_API struct ggml_tensor* ggml_conv_1d_ph(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   s,
            int                   d);

    GGML_API struct ggml_tensor * ggml_conv_transpose_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   s0,
            int                   p0,
            int                   d0);

    GGML_API struct ggml_tensor * ggml_conv_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1,
            int                   d0,
            int                   d1);


    // kernel size is a->ne[0] x a->ne[1]
    // stride is equal to kernel size
    // padding is zero
    // example:
    // a:     16   16    3  768
    // b:   1024 1024    3    1
    // res:   64   64  768    1
    // used in sam
    GGML_API struct ggml_tensor * ggml_conv_2d_sk_p0(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    // kernel size is a->ne[0] x a->ne[1]
    // stride is 1
    // padding is half
    // example:
    // a:      3    3    256  256
    // b:     64   64    256    1
    // res:   64   64    256    1
    // used in sam
    GGML_API struct ggml_tensor * ggml_conv_2d_s1_ph(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);

    GGML_API struct ggml_tensor * ggml_conv_transpose_2d_p0(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   stride);

    // 2D convolution taking the kernel as `a` and the data as `b`, returning F32.
    // a: [KW, KH, IC, OC], b: [W, H, IC, N] -> [OW, OH, OC, N]
    // Upstream added this as a dedicated fused op; here it is expressed with our existing
    // im2col + mul_mat, which computes the same result. See ggml.c for the tradeoff.
    GGML_API struct ggml_tensor * ggml_conv_2d_direct(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,  // convolution kernel
            struct ggml_tensor  * b,  // data
            int                   s0, // stride dimension 0
            int                   s1, // stride dimension 1
            int                   p0, // padding dimension 0
            int                   p1, // padding dimension 1
            int                   d0, // dilation dimension 0
            int                   d1); // dilation dimension 1

    // depthwise 1D convolution (via im2col and mul_mat)
    // a: [K, 1, C], b: [T, C] -> [T_out, C, 1]
    GGML_API struct ggml_tensor * ggml_conv_1d_dw(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,  // convolution kernel
            struct ggml_tensor  * b,  // data
            int                   s0, // stride
            int                   p0, // padding
            int                   d0); // dilation

    // depthwise (via im2col and mul_mat)
    GGML_API struct ggml_tensor * ggml_conv_2d_dw(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,  // convolution kernel
            struct ggml_tensor  * b,  // data
            int                  s0,  // stride dimension 0
            int                  s1,  // stride dimension 1
            int                  p0,  // padding dimension 0
            int                  p1,  // padding dimension 1
            int                  d0,  // dilation dimension 0
            int                  d1); // dilation dimension 1

    // Depthwise 2D convolution
    // may be faster than ggml_conv_2d_dw, but not available in all backends
    // a:   KW    KH    1    C    convolution kernel
    // b:   W     H     C    N    input data
    // res: W_out H_out C    N
    GGML_API struct ggml_tensor * ggml_conv_2d_dw_direct(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            int                   stride0,
            int                   stride1,
            int                   pad0,
            int                   pad1,
            int                   dilation0,
            int                   dilation1);

    enum ggml_op_pool {
        GGML_OP_POOL_MAX,
        GGML_OP_POOL_AVG,
        GGML_OP_POOL_COUNT,
    };

    GGML_API struct ggml_tensor * ggml_pool_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum ggml_op_pool     op,
            int                   k0, // kernel size
            int                   s0, // stride
            int                   p0); // padding

    // the result will have 2*p0 padding for the first dimension
    // and 2*p1 padding for the second dimension
    GGML_API struct ggml_tensor * ggml_pool_2d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum ggml_op_pool     op,
            int                   k0,
            int                   k1,
            int                   s0,
            int                   s1,
            float                 p0,
            float                 p1);

    // ADDED 2026-09-09 (slice 53). Placed HERE, after ggml_pool_2d, rather
    // than with its three siblings above: its enum ggml_op_pool parameter needs
    // that enum to be COMPLETE at the point of declaration, and it is defined
    // below them.
    GGML_API struct ggml_tensor * ggml_pool_2d_back(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * af,
        enum ggml_op_pool     op,
        int                   k0,
        int                   k1,
        int                   s0,
        int                   s1,
        float                 p0,
        float                 p1);

    enum ggml_scale_mode {
        GGML_SCALE_MODE_NEAREST  = 0,
        GGML_SCALE_MODE_BILINEAR = 1,
        GGML_SCALE_MODE_BICUBIC  = 2,

        GGML_SCALE_MODE_COUNT
    };

    enum ggml_scale_flag {
        GGML_SCALE_FLAG_ALIGN_CORNERS = (1 << 8),

        // Area-averaged ("antialiased") resampling, matching PIL / torchvision
        // `antialias=True`. When DOWN-scaling, the interpolation kernel is widened to
        // 1/scale so every source pixel that falls inside a destination pixel's footprint
        // contributes to it. Without this, downsampling point-samples the source and
        // aliases: a 1024x1024 photo reduced to 14x14 patches keeps whatever happened to
        // land on the sample points and discards the rest.
        //
        // Ignored when UP-scaling (support <= 1), which is what upstream does too, so the
        // flag is safe to leave set in a mode constant that is used for both directions --
        // and mtmd does exactly that (DEFAULT_INTERPOLATION_MODE in examples/mtmd/clip-graph.h).
        GGML_SCALE_FLAG_ANTIALIAS     = (1 << 9)
    };

    // nearest interpolate
    // multiplies ne0 and ne1 by scale factor
    // used in stable-diffusion
    GGML_API struct ggml_tensor * ggml_upscale(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   scale_factor,
            enum ggml_scale_mode  mode);

    // interpolate
    // interpolate scale to specified dimensions
    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_upscale_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   ne0,
            int                   ne1,
            int                   ne2,
            int                   ne3,
            enum ggml_scale_mode  mode),
        "use ggml_interpolate instead");

    // Move the dim window that an ALREADY-BUILT rope node rotates, so two ropes with different
    // position streams can cover disjoint halves of the same head (clip.cpp's 2D rope, gemma4v,
    // minimax-m3). Mutates and returns `a`; `a` must be a GGML_OP_ROPE node.
    GGML_API struct ggml_tensor * ggml_rope_set_offset(
            struct ggml_tensor  * a,
            int                   offset);

    // Up- or downsamples the input to the specified size.
    // 2D scale modes (eg. bilinear) are applied to the first two dimensions.
    GGML_API struct ggml_tensor * ggml_interpolate(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            uint32_t              mode); // ggml_scale_mode [ | ggml_scale_flag...]


    // pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
    GGML_API struct ggml_tensor * ggml_pad(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                  p0,
            int                  p1,
            int                  p2,
            int                  p3);

    // pad each dimension with zeros, giving the LOW and HIGH side of each dimension
    // separately. ggml_pad(a, p0, p1, p2, p3) is ggml_pad_ext(a, 0, p0, 0, p1, 0, p2, 0, p3).
    GGML_API struct ggml_tensor * ggml_pad_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   lp0,
            int                   rp0,
            int                   lp1,
            int                   rp1,
            int                   lp2,
            int                   rp2,
            int                   lp3,
            int                   rp3);

    // pad dimension 0 by REFLECTING the interior, excluding the edge sample itself:
    // [a b c d] with p0 = p1 = 2 -> [c b | a b c d | c b]. Requires p0, p1 < ne0.
    GGML_API struct ggml_tensor * ggml_pad_reflect_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   p0,
            int                   p1);

    // circularly shift the contents of each dimension; elements shifted off one end
    // reappear at the other. Positive shift moves data toward higher indices.
    GGML_API struct ggml_tensor * ggml_roll(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   shift0,
            int                   shift1,
            int                   shift2,
            int                   shift3);

    // element-wise sine / cosine
    GGML_API struct ggml_tensor * ggml_sin(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    GGML_API struct ggml_tensor * ggml_cos(
            struct ggml_context * ctx,
            struct ggml_tensor  * a);

    // Overlap-ADD inverse of im2col for 1D, i.e. the scatter half of a transposed convolution.
    // a is the GEMM result laid out [K*OC, T]; element (k + K*oc, t) is added into
    // out[t*s0 + k, oc]. Output is [(T-1)*s0 + K - 2*p0, OC].
    // Contributions overlap when s0 < K -- they are summed, which is what makes this the
    // adjoint of im2col rather than a plain reshape.
    GGML_API struct ggml_tensor * ggml_col2im_1d(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   s0,  // stride
            int                   OC,  // number of output channels
            int                   p0); // padding trimmed from each end

    // Scatter-ADD adjoint of GGML_OP_IM2COL_3D (what IM2COL_3D's backward needs).
    // cols is the im2col_3d result layout [IC*KD*KH*KW, OW, OH, N*OD], F32, contiguous; element
    // (iic*KD*KH*KW + ikd*KH*KW + ikh*KW + ikw, iow, ioh, in*OD + iod) is ADDED into
    // out[iow*s0 + ikw*d0 - p0, ioh*s1 + ikh*d1 - p1, iod*s2 + ikd*d2 - p2, in*IC + iic] when that
    // position lies inside the image; positions in the padding are discarded. Output is F32
    // [IW, IH, ID, N*IC]. IW/IH/ID are the forward image extents (not derivable from cols when
    // the strides do not divide evenly), and OW/OH/OD of cols must equal the forward
    // ggml_calc_conv_output_size of those extents.
    GGML_API struct ggml_tensor * ggml_col2im_3d(
            struct ggml_context * ctx,
            struct ggml_tensor  * cols,
            int64_t               IW,
            int64_t               IH,
            int64_t               ID,
            int64_t               N,
            int64_t               IC,
            int64_t               KW,
            int64_t               KH,
            int64_t               KD,
            int s0, int s1, int s2,
            int p0, int p1, int p2,
            int d0, int d1, int d2);

    // Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
    // timesteps: [N,]
    // return: [N, dim]
    GGML_API struct ggml_tensor * ggml_timestep_embedding(
            struct ggml_context * ctx,
            struct ggml_tensor  * timesteps,
            int                   dim,
            int                   max_period);

    // convert matrix to triangular form by zeroing values outside selected half
    GGML_API struct ggml_tensor * ggml_tri(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum ggml_tri_type    type);

    // fill tensor with constant c
    GGML_API struct ggml_tensor * ggml_fill(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 c);

    GGML_API struct ggml_tensor * ggml_fill_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 c);

    // Overwrite values in a with c for the indeces stored in b
    GGML_API struct ggml_tensor * ggml_blend(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            float                 c);

    GGML_API struct ggml_tensor * ggml_indexer_mask(
            struct ggml_context * ctx,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * topk);

    // n_visible is I32 [1, n_rows]; row i of the result is 0 in its first n_visible[i] columns, -INFINITY after
    GGML_API struct ggml_tensor * ggml_visible_mask(
            struct ggml_context * ctx,
            struct ggml_tensor  * n_visible,
            int64_t               n_kv,
            enum ggml_type        type);


    // sort rows
    enum ggml_sort_order {
        GGML_SORT_ORDER_ASC,
        GGML_SORT_ORDER_DESC,
    };

    GGML_API struct ggml_tensor * ggml_argsort(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            enum ggml_sort_order  order);

    GGML_API struct ggml_tensor * ggml_argsort_thresh(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   min_entries,
            float                 threshold);

    GGML_API struct ggml_tensor * ggml_arange(
            struct ggml_context * ctx,
            float                 start,
            float                 stop,
            float                 step);

    // top k elements per row: dedicated GGML_OP_TOP_K, contiguous I32 [k, ...] result
    // note: the resulting top k indices are in no particular order
    GGML_API struct ggml_tensor * ggml_top_k(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   k);

    // argsort(DESC) + k-wide VIEW (sorted, NOT contiguous: nb[1] = ne[0]*4). Same as upstream's
    // ggml_argsort_top_k and the fork's former ggml_top_k. Use this for MoE routing (the CUDA/Vulkan/
    // Metal/SYCL fusions match ARGSORT -> VIEW -> GET_ROWS) and wherever sorted indices are needed.
    GGML_API struct ggml_tensor * ggml_argsort_top_k(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   k);
    GGML_API struct ggml_tensor * ggml_top_k_thresh(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   k,
            int                   min_entries,
            float                 thresh);
    GGML_API struct ggml_tensor * ggml_grouped_topk(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   num_groups,
            int                   num_top_groups,
            int                   nk,
            int                   topk_experts);

#define GGML_KQ_MASK_PAD 64

    // q:    [n_embd, n_batch,     n_head,    1]
    // k:    [n_embd, n_kv,        n_head_kv, 1]
    // v:    [n_embd, n_kv,        n_head_kv, 1] !! not transposed !!
    // mask: [n_kv,   n_batch_pad, 1,         1] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!
    // res:  [n_embd, n_head,      n_batch,   1] !! permuted !!
    GGML_API struct ggml_tensor * ggml_flash_attn_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            struct ggml_tensor  * mask,
            float                 scale,
            float                 max_bias,
            float                 softcap);

    // This fork's backend hint, stored in ggml_flash_attn_ext op_params slot 5 (slot 4 is upstream's n_kv_max below):
    //   n_swa > 0  the sliding window: backends may skip the K/V rows older than it (CUDA FA, CPU iqk FA)
    //   0          none
    //   GGML_FLASH_ATTN_EXT_IQK_DISABLED  request the generic CPU implementation instead of IQK FA
    #define GGML_FLASH_ATTN_EXT_IQK_DISABLED (-1)
    #define GGML_FLASH_ATTN_EXT_N_SWA_SLOT   5

    GGML_API void ggml_flash_attn_ext_set_n_swa(
            struct ggml_tensor * a,
            int32_t              n_swa);

    GGML_API int32_t ggml_flash_attn_ext_get_n_swa(
            const struct ggml_tensor * a);

    // ggml_flash_attn_ext_rows keeps ggml-org's layout (the K/V row list is src[5]), but this fork already
    // uses src[5] of FLASH_ATTN_EXT for the ik per-query / indexer row list (DSA, DSV4), with or without a
    // mask, so the two are told apart by op_params slot 6 (set by ggml_flash_attn_ext_rows, 0 otherwise)
    #define GGML_FLASH_ATTN_EXT_KV_ROWS_SLOT 6

    // the per-slice K/V row list of a ggml_flash_attn_ext_rows node, NULL for any other FLASH_ATTN_EXT node
    GGML_API struct ggml_tensor * ggml_flash_attn_ext_get_kv_rows(
            const struct ggml_tensor * a);

    // same as ggml_flash_attn_ext, but each q slice attends to its own list of K/V rows
    // q:       [n_embd_k, n_batch, n_head,    ne3]
    // k:       [n_embd_k, n_rows,  n_head_kv, 1  ] shared by all q slices
    // v:       [n_embd_v, n_rows,  n_head_kv, 1  ]
    // mask:    [n_kv,     n_batch, ne32,      ne3]
    // kv_rows: [n_kv,     ne3] I32, mask column i of slice i3 uses row kv_rows[i, i3] of k and v
    // negative rows are padding, their mask entries must be -inf
    GGML_API struct ggml_tensor * ggml_flash_attn_ext_rows(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * kv_rows,
            float                 scale,
            float                 max_bias,
            float                 logit_softcap);

    GGML_DEPRECATED(GGML_API void ggml_flash_attn_ext_set_prec(
            struct ggml_tensor * a,
            enum ggml_prec       prec),
        "use ggml_prec_set_acc() instead");

    GGML_API enum ggml_prec ggml_flash_attn_ext_get_prec(
            const struct ggml_tensor * a);

    // Use finite mask entries as a sparse K/V set. Set 0 to disable.
    // n_kv_max must bound the number of finite entries in every mask row.
    GGML_API void ggml_flash_attn_ext_set_n_kv_max(
            struct ggml_tensor * a,
            int32_t              n_kv_max);

    GGML_API void ggml_flash_attn_ext_add_sinks(
            struct ggml_tensor * a,
            struct ggml_tensor * sinks);

    // Latent attention over a packed K/V cache with an independently-visible K/V prefix.
    // The value vector of cache row n is cache[dv_off .. dv_off+dv, n] (MLA "absorbed"
    // layout: openPangu packs [ckv|k_pe] -> dv_off=0; DeepSeek/GLM_DSA packs [k_pe|ckv]
    // rope-first -> dv_off=64). The prefix K/V rows are always visible (bias 0); the mask
    // applies to the cache segment only.
    //
    // q:        [Dk, T, H]   F32, contiguous
    // cache:    [Dk, N]      F32 / F16: row stride nb[1] arbitrary (windowed views ok);
    //                        Q8_0: rows must be packed (nb[1] == row size)
    // prefix_k: [Dk, P]      F32 / F16, contiguous; NULL iff P == 0
    // prefix_v: [P,  Dv]     F32 / F16, contiguous (value-transposed: ne0 = P); NULL iff P == 0
    // mask:     [N,  >=T]    F32, additive, cache segment only; NULL = unmasked cache
    // returns:  [Dv, T, H]   F32, contiguous
    // Precondition: every query column must have at least one visible position (a prefix row,
    // or a cache row not masked to -inf). Violating it is undefined: a fully-masked column with
    // P == 0 yields NaN on CUDA and on the CPU dense path, and zeros on the CPU indexed path, so
    // callers must not depend on the value. Inference-only: a gradient-bearing input is rejected.
    GGML_API struct ggml_tensor * ggml_latent_attn_prefix_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * cache,
            struct ggml_tensor  * prefix_k,
            struct ggml_tensor  * prefix_v,
            struct ggml_tensor  * mask,
            int                   dv,
            int                   dv_off,
            float                 scale,
            float                 max_bias);

    // Indexed latent attention over absolute cache row ids shared across heads.
    // indices: [topk, T] I32, contiguous; mask remains [N, >=T] and is gathered by index.
    // Preconditions: each index must lie in [0, N); an out-of-range id aborts on CPU and is
    // undefined behavior (an out-of-bounds device read) on CUDA. The visible-position and
    // inference-only preconditions above apply here as well.
    GGML_API struct ggml_tensor * ggml_latent_attn_indexed_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * cache,
            struct ggml_tensor  * prefix_k,
            struct ggml_tensor  * prefix_v,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * indices,
            int                   dv,
            int                   dv_off,
            float                 scale,
            float                 max_bias);

    // TODO: needs to be adapted to ggml_flash_attn_ext
    GGML_API struct ggml_tensor * ggml_flash_attn_back(
           struct ggml_context * ctx,
           struct ggml_tensor  * q,
           struct ggml_tensor  * k,
           struct ggml_tensor  * v,
           struct ggml_tensor  * d,
           bool                  masked);

    GGML_API struct ggml_tensor * ggml_ssm_conv(
            struct ggml_context * ctx,
            struct ggml_tensor  * s,
            struct ggml_tensor  * x,
            struct ggml_tensor  * c,
            struct ggml_tensor  * sq,
            struct ggml_tensor  * saved_steps);

    GGML_API struct ggml_tensor * ggml_ssm_scan(
            struct ggml_context * ctx,
            struct ggml_tensor  * s,
            struct ggml_tensor  * x,
            struct ggml_tensor  * dt,
            struct ggml_tensor  * A,
            struct ggml_tensor  * B,
            struct ggml_tensor  * C,
            struct ggml_tensor  * ids,
            int                   K);
    // ids {n_seqs} I32, and K = number of chunks (K >= 1; K > 1 also returns rollback
    // snapshots). Both are what every backend in this tree ALREADY consumes -- ids as src[6],
    // K as op_params[0]. The previous 8-parameter form never set op_params, so K read 0 and the
    // CPU kernel's own GGML_ASSERT(K >= 1) fired on the first compute: measured 2026-09-20 at
    // ggml/src/ggml-cpu/ops.cpp:9937. Pass K = 1 for the plain single-chunk scan.

    // partition into non-overlapping windows with padding if needed
    // example:
    // a:   768   64   64    1
    // w:    14
    // res: 768   14   14    25
    // used in sam
    GGML_API struct ggml_tensor * ggml_win_part(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   w);

    // reverse of ggml_win_part
    // used in sam
    GGML_API struct ggml_tensor * ggml_win_unpart(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   w0,
            int                   h0,
            int                   w);

    GGML_API struct ggml_tensor * ggml_unary(
            struct ggml_context * ctx,
             struct ggml_tensor * a,
             enum ggml_unary_op op);

    GGML_API struct ggml_tensor * ggml_unary_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        enum ggml_unary_op op);

    // used in sam
    GGML_API struct ggml_tensor * ggml_get_rel_pos(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   qh,
            int                   kh);

    // used in sam
    GGML_API struct ggml_tensor * ggml_add_rel_pos(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * pw,
            struct ggml_tensor  * ph);

    GGML_API struct ggml_tensor * ggml_add_rel_pos_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * pw,
            struct ggml_tensor  * ph);

    // Solve Ax = B where A is triangular
    GGML_API struct ggml_tensor * ggml_solve_tri(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            bool                  left,
            bool                  lower,
            bool                  uni);

    GGML_API struct ggml_tensor * ggml_delta_net(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            struct ggml_tensor  * g,
            struct ggml_tensor  * beta,
            struct ggml_tensor  * state,
            struct ggml_tensor  * saved_steps);

    GGML_API struct ggml_tensor * ggml_indexer_topk(
            struct ggml_context * ctx,
            struct ggml_tensor  * k,
            struct ggml_tensor  * q,
            struct ggml_tensor  * w,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * blk_idx,
            enum ggml_unary_op    op,
            int                   n_top_k,
            int                   group_size,
            bool                  select_groups,
            struct ggml_tensor  * top_k_out);

    // Sinkhorn normalization of a flat [S*S, T] batch of S x S matrices into
    // doubly-stochastic form: softmax over columns, then column normalization,
    // then (n_iters - 1) rounds of row + column normalization (ends on columns).
    // The flat input is row-major (column index fastest). eps, when non-zero, is
    // added to the softmax output and to every normalization sum before dividing.
    // With output_transposed the result is [S, S, T] with ne0 = row, ne1 = column
    // (ready for out[c] = sum_r m[r,c] * residual[r] consumers); otherwise the
    // bare input layout (ne0 = column) is kept.
    GGML_API struct ggml_tensor * ggml_sinkhorn(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   S,
            int                   n_iters,
            float                 eps,
            bool                  output_transposed);

    GGML_API struct ggml_tensor * ggml_hc_pre(
            struct ggml_context * ctx,
            struct ggml_tensor  * x,
            struct ggml_tensor  * scale,
            struct ggml_tensor  * bias,
            int                   S,
            int                   n_iters,
            float                 eps);

    GGML_API struct ggml_tensor * ggml_hc_post(
            struct ggml_context * ctx,
            struct ggml_tensor  * x,
            struct ggml_tensor  * post,
            struct ggml_tensor  * res,
            struct ggml_tensor  * comb);

    GGML_API struct ggml_tensor * ggml_mask_to_index(
            struct ggml_context * ctx,
            struct ggml_tensor  * mask,
            int                   max_row_size);

    // ggml_mask_to_index(concat(mask, ggml_indexer_mask(ggml_visible_mask(n_visible), topk), 0), max_row_size) without the concat
    GGML_API struct ggml_tensor * ggml_mask_to_index_topk(
            struct ggml_context * ctx,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * topk,
            struct ggml_tensor  * n_visible,
            int                   max_row_size);

    // ggml_mask_to_index(ggml_indexer_mask(mask, topk), max_row_size) without the dense mask
    GGML_API struct ggml_tensor * ggml_indexer_mask_to_index(
            struct ggml_context * ctx,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * topk,
            int                   max_row_size);

    GGML_API struct ggml_tensor * ggml_ds4_comp(
            struct ggml_context * ctx,
            struct ggml_tensor  * state,
            struct ggml_tensor  * score,
            struct ggml_tensor  * idx,
            int                   ratio,
            int                   type);

    // top-k sparse attention mask: keep the mask cells of the live picks and the extra cells, set the rest to -inf
    //
    // mask:   [n_kv,    m1,  m2,  m3]  f16 or f32, broadcast to the rows of top_k
    // top_k:  [n_top,   ne1, ne2, ne3] i32, picks of each row
    // score:  [n_pick,  ne1, ne2, ne3] f32, can be NULL: a pick is live when its score is above -inf, else every pick is live
    // blocks: [n_blk,   n_pick]        i32, can be NULL: the member cells of each pick, else a pick is a cell
    // extra:  [n_extra, ne1, ne2, ne3] i32, can be NULL: cells of each row kept without a pick
    // res:    [n_kv,    ne1, ne2, ne3] same type as mask
    //
    // res[c, i] = mask[c, i % m] when c is a cell of a live pick of row i or an extra cell of row i, else -inf
    // picks and cells out of range are skipped
    //
    GGML_API struct ggml_tensor * ggml_top_k_mask(
            struct ggml_context * ctx,
            struct ggml_tensor  * mask,
            struct ggml_tensor  * top_k,
            struct ggml_tensor  * score,
            struct ggml_tensor  * blocks,
            struct ggml_tensor  * extra);

    // custom operators

    typedef void (*ggml_unary_op_f32_t) (const int, float *, const float *);
    typedef void (*ggml_binary_op_f32_t)(const int, float *, const float *, const float *);

    typedef void (*ggml_custom1_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *);
    typedef void (*ggml_custom2_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *);
    typedef void (*ggml_custom3_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *);

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_unary_f32(
            struct ggml_context        * ctx,
            struct ggml_tensor         * a,
                   ggml_unary_op_f32_t   fun),
        "use ggml_map_custom1 instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_unary_inplace_f32(
            struct ggml_context        * ctx,
            struct ggml_tensor         * a,
                   ggml_unary_op_f32_t   fun),
        "use ggml_map_custom1_inplace instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_binary_f32(
            struct ggml_context         * ctx,
            struct ggml_tensor          * a,
            struct ggml_tensor          * b,
                   ggml_binary_op_f32_t   fun),
        "use ggml_map_custom2 instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_binary_inplace_f32(
            struct ggml_context         * ctx,
            struct ggml_tensor          * a,
            struct ggml_tensor          * b,
                   ggml_binary_op_f32_t   fun),
        "use ggml_map_custom2_inplace instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom1_f32(
            struct ggml_context          * ctx,
            struct ggml_tensor           * a,
                   ggml_custom1_op_f32_t   fun),
        "use ggml_map_custom1 instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom1_inplace_f32(
            struct ggml_context          * ctx,
            struct ggml_tensor           * a,
                   ggml_custom1_op_f32_t   fun),
        "use ggml_map_custom1_inplace instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom2_f32(
            struct ggml_context          * ctx,
            struct ggml_tensor           * a,
            struct ggml_tensor           * b,
                   ggml_custom2_op_f32_t   fun),
        "use ggml_map_custom2 instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom2_inplace_f32(
            struct ggml_context          * ctx,
            struct ggml_tensor           * a,
            struct ggml_tensor           * b,
                   ggml_custom2_op_f32_t   fun),
        "use ggml_map_custom2_inplace instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom3_f32(
            struct ggml_context          * ctx,
            struct ggml_tensor           * a,
            struct ggml_tensor           * b,
            struct ggml_tensor           * c,
                   ggml_custom3_op_f32_t   fun),
        "use ggml_map_custom3 instead");

    GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_map_custom3_inplace_f32(
            struct ggml_context          * ctx,
            struct ggml_tensor           * a,
            struct ggml_tensor           * b,
            struct ggml_tensor           * c,
                   ggml_custom3_op_f32_t   fun),
        "use ggml_map_custom3_inplace instead");

    // custom operators v2

    typedef void (*ggml_custom1_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata);
    typedef void (*ggml_custom2_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata);
    typedef void (*ggml_custom3_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata);

    // Adopted from upstream 2026-09-03 (ik_llama merge): the n-ary custom
    // op used by ggml_custom_4d. ik_llama had 1/2/3-ary forms only, and
    // merged internal code referencing this one did not compile.
    typedef void (*ggml_custom_op_t)(struct ggml_tensor * dst , int ith, int nth, void * userdata);

    #define GGML_N_TASKS_MAX -1

    GGML_API struct ggml_tensor * ggml_map_custom1(
            struct ggml_context   * ctx,
            struct ggml_tensor    * a,
            ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_API struct ggml_tensor * ggml_map_custom1_inplace(
            struct ggml_context   * ctx,
            struct ggml_tensor    * a,
            ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_API struct ggml_tensor * ggml_map_custom2(
            struct ggml_context   * ctx,
            struct ggml_tensor    * a,
            struct ggml_tensor    * b,
            ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_API struct ggml_tensor * ggml_map_custom2_inplace(
            struct ggml_context   * ctx,
            struct ggml_tensor    * a,
            struct ggml_tensor    * b,
            ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_API struct ggml_tensor * ggml_map_custom3(
            struct ggml_context   * ctx,
            struct ggml_tensor    * a,
            struct ggml_tensor    * b,
            struct ggml_tensor    * c,
            ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    GGML_API struct ggml_tensor * ggml_map_custom3_inplace(
            struct ggml_context   * ctx,
            struct ggml_tensor    * a,
            struct ggml_tensor    * b,
            struct ggml_tensor    * c,
            ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    // loss function

    GGML_API struct ggml_tensor * ggml_cross_entropy_loss(
            struct ggml_context         * ctx,
            struct ggml_tensor          * a,
            struct ggml_tensor          * b);

    GGML_API struct ggml_tensor * ggml_cross_entropy_loss_back(
            struct ggml_context         * ctx,
            struct ggml_tensor          * a,
            struct ggml_tensor          * b,
            struct ggml_tensor          * c);

    //
    // automatic differentiation
    //

    GGML_API void ggml_set_param(struct ggml_tensor * tensor);
    GGML_API void ggml_set_loss (struct ggml_tensor * tensor);

    GGML_API struct ggml_tensor * ggml_graph_get_grad    (const struct ggml_cgraph * cgraph, const struct ggml_tensor * node);
    GGML_API struct ggml_tensor * ggml_graph_get_grad_acc(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node);

    // ADDED 2026-09-09 (slice 50). The KERNELS for these ops already ship on the
    // CPU, CUDA, Vulkan, Metal, SYCL and CANN backends and the enum entries are in
    // enum ggml_op above -- what was missing was any way to BUILD a node of them.
    // Same shape as commit 670dd0efb, which ported 12 other missing constructors.
    // Bodies taken from ancestor revisions of this repository, not retyped.
    GGML_API void ggml_set_loss(struct ggml_tensor * tensor);
    GGML_API struct ggml_tensor * ggml_count_equal(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b);
    GGML_API struct ggml_tensor * ggml_opt_step_adamw(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * grad,
            struct ggml_tensor  * m,
            struct ggml_tensor  * v,
            struct ggml_tensor  * adamw_params);
    GGML_API struct ggml_tensor * ggml_opt_step_sgd(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * grad,
            struct ggml_tensor  * params);


    GGML_API void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);

    // Append `tensor`'s dependencies to the graph so that their ORDER is pinned now, rather
    // than being decided later when some consumer happens to pull them in. clip.cpp uses this
    // to emit q, k and v adjacently, which reduces the number of backend splits.
    GGML_API void ggml_build_forward_order (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);

    // Append every tensor in `outs` to the graph, then make the one at `idx` the graph's
    // effective output. All branches are built, so the graph TOPOLOGY does not depend on idx --
    // that is the point: a graph reused across calls keeps a constant shape while the selected
    // branch varies. See ggml.c for what "effective output" means on our base.
    // Returns the SELECTED tensor, so the caller can chain from it -- matching upstream
    // (279b4bfaf^2:ggml/include/ggml.h:2795). It used to return void here, and the merged
    // src/llama-graph.cpp:2376 assigns the result, which is a hard error. An existing caller that
    // ignores the return value is unaffected.
    GGML_API struct ggml_tensor * ggml_build_forward_select (struct ggml_cgraph * cgraph, struct ggml_tensor ** outs, int n, int idx);
    GGML_API void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * cgraph, struct ggml_tensor ** grad_accs);
    // FORK-ONLY: ggml_build_backward_expand that returns false (and writes why into `reason`, if given) instead of
    // aborting when a node on the gradient path is an inplace op or has no backward pass. On false the graph is
    // half-built and must be discarded. test-backend-ops grad uses it to report such cases as "not supported".
    GGML_API bool ggml_build_backward_expand_checked(struct ggml_context * ctx, struct ggml_cgraph * cgraph,
                                                     struct ggml_tensor ** grad_accs, char * reason, size_t reason_size);

    // graph allocation in a context
    GGML_API struct ggml_cgraph * ggml_new_graph         (struct ggml_context * ctx); // size = GGML_DEFAULT_GRAPH_SIZE, grads = false
    GGML_API struct ggml_cgraph * ggml_new_graph_custom  (struct ggml_context * ctx, size_t size, bool grads);
    GGML_API struct ggml_cgraph * ggml_graph_dup         (struct ggml_context * ctx, struct ggml_cgraph * cgraph, bool force_grads);
    GGML_API struct ggml_cgraph   ggml_graph_view        (struct ggml_cgraph * cgraph, int i0, int i1);
    GGML_API void                 ggml_graph_cpy         (struct ggml_cgraph * src, struct ggml_cgraph * dst);
    GGML_API void                 ggml_graph_reset       (struct ggml_cgraph * cgraph);  // zero grads
    GGML_API void                 ggml_graph_clear       (struct ggml_cgraph * cgraph);

    GGML_API size_t ggml_graph_overhead(void);
    GGML_API size_t ggml_graph_overhead_custom(size_t size, bool grads);

    GGML_API int                   ggml_graph_size(struct ggml_cgraph* cgraph);
    GGML_API struct ggml_tensor* ggml_graph_node(struct ggml_cgraph* cgraph, int i); // if i < 0, returns nodes[n_nodes + i]
    GGML_API struct ggml_tensor** ggml_graph_nodes(struct ggml_cgraph* cgraph);
    GGML_API int                   ggml_graph_n_nodes(struct ggml_cgraph* cgraph);


    // NOTE: the comments that stood here described ggml_graph_plan, ggml_graph_compute and
    // ggml_graph_compute_with_ctx, which now live in ggml-cpu.h along with their declarations.
    // Upstream's commit 9f409893 (#10144) removed these same three lines for the same reason.

    GGML_API struct ggml_tensor * ggml_graph_get_tensor(const struct ggml_cgraph * cgraph, const char * name);

    GGML_API void                 ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);
    GGML_API struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);

    // print info and performance information for the graph
    GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);

    // dump the graph into a file using the dot format
    GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);

    // build gradient checkpointing backward graph gb for gf using provided checkpoints
    // gb_tmp will contain original backward graph with rewritten backward process nodes,
    // but without the second forward pass nodes.
    //
    // optimization
    //

    // optimization methods

    // linesearch methods

    // optimization return values

    typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    // (ported verbatim from upstream ggml/include/ggml.h during the R6 merge)
    GGML_API void ggml_log_get(ggml_log_callback * log_callback, void ** user_data);
    GGML_API void ggml_log_set(ggml_log_callback   log_callback, void *  user_data);

    // optimization parameters
    //
    //



    // optimize the function defined by the tensor f

    // initialize optimizer context

    // continue optimizing the function defined by the tensor f

    // continue optimizing the function defined by the tensor f

    //
    // tensor flags
    //
    // Adopted from upstream 2026-09-08 alongside enum ggml_op_hint.
    GGML_API void ggml_mul_mat_set_hint(struct ggml_tensor * a, enum ggml_op_hint hint);

    GGML_API void ggml_set_input(struct ggml_tensor * tensor);
    GGML_API void ggml_set_output(struct ggml_tensor * tensor);

    //
    // quantization
    //

    // - ggml_quantize_init can be called multiple times with the same type
    //   it will only initialize the quantization tables for the first call or after ggml_quantize_free
    //   automatically called by ggml_quantize_chunk for convenience
    //
    // - ggml_quantize_free will free any memory allocated by ggml_quantize_init
    //   call this at the end of the program to avoid memory leaks
    //
    // note: these are thread-safe
    //
    GGML_API void ggml_quantize_init(enum ggml_type type);
    GGML_API void ggml_quantize_free(void);

    // some quantization type cannot be used without an importance matrix
    GGML_API bool ggml_quantize_requires_imatrix(enum ggml_type type);

    // per-type multiplier applied to the block scale by the quantiser (upstream ik_llama API).
    // defaults reproduce the constants the quantisers used to hard-code; 1.0 for every other type.
#define GGML_HAS_QUANTIZE_FUDGE_FACTOR
    GGML_API void  ggml_set_quantize_fudge_factor(enum ggml_type type, float fudge);
    GGML_API float ggml_get_quantize_fudge_factor(enum ggml_type type);

    struct quantize_user_data {
        bool  symmetric_q4_0;
        bool  slow_iq2_ks;
    };

    // calls ggml_quantize_init internally (i.e. can allocate memory)
    // GGML_CXX_DEFAULT: C++ callers may omit user_data (upstream's 7-argument spelling); C callers pass it
#ifdef __cplusplus
#define GGML_CXX_DEFAULT(v) = v
#else
#define GGML_CXX_DEFAULT(v)
#endif
    GGML_API size_t ggml_quantize_chunk(
            enum ggml_type   type,
               const float * src,
                      void * dst,
                   int64_t   start,
                   int64_t   nrows,
                   int64_t   n_per_row,
               const float * imatrix,
               const struct quantize_user_data * user_data GGML_CXX_DEFAULT(nullptr));

    //
    // system info
    //

    GGML_API int ggml_cpu_has_metal      (void);
    GGML_API int ggml_cpu_has_blas       (void);
    GGML_API int ggml_cpu_has_cuda       (void);
    GGML_API int ggml_cpu_has_vulkan     (void);
    GGML_API int ggml_cpu_has_gpublas    (void);
    GGML_API int ggml_cpu_has_sycl       (void);
    GGML_API int ggml_cpu_has_rpc        (void);
    GGML_API int ggml_cpu_has_cann       (void);

    //
    // Internal types and functions exposed for tests and benchmarks
    //

#ifdef  __cplusplus
// restrict not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif
    typedef void (*ggml_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
    typedef void (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);
    typedef void (*ggml_from_float_to_mat_t)
                                     (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t nr, int64_t k, int64_t bs);
    typedef void (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                       const void * GGML_RESTRICT y, size_t by, int nrc);
    typedef void (*ggml_gemv_t)     (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x,
                                       const void * GGML_RESTRICT y, int nr, int nc);
    typedef void (*ggml_gemm_t)     (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x,
                                       const void * GGML_RESTRICT y, int nr, int nc);

    // The tag is what lets `struct ggml_type_traits` name this type. Upstream declares a
    // SEPARATE 7-field struct under that name; this fork's struct has all 7 of those fields plus
    // ten more, so tagging the fork's superset lets upstream's call sites compile against the
    // fork's data with no second table and no reinterpretation. It is deliberately NOT a cast:
    // the fork interleaves from_float between to_float and from_float_ref, so the two layouts
    // diverge at offset 6 and a cast would silently hand out the wrong function pointer.
    typedef struct ggml_type_traits {
        const char             * type_name;
        int64_t                  blck_size;
        int64_t                  blck_size_interleave; // interleave elements in blocks
        size_t                   type_size;
        bool                     is_quantized;
        ggml_to_float_t          to_float;
        ggml_from_float_t        from_float;
        ggml_from_float_t        from_float_ref;
        ggml_from_float_to_mat_t from_float_to_mat;
        ggml_vec_dot_t           vec_dot;
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
        int64_t                  ncols; // number of columns to process simultaneously
        ggml_gemv_t              gemv;
        ggml_gemm_t              gemm;
        int64_t                  row_meta_size;
    } ggml_type_traits_t;

    GGML_API ggml_type_traits_t ggml_internal_get_type_traits(enum ggml_type type);

    // Upstream's spelling of the same lookup, returning a pointer into the static
    // traits table rather than a copy. Both exist because both are used: this merged
    // tree has ~25 call sites on the by-value form above and 40+ on this one,
    // including 11 in ggml/src/ggml-cpu/ops.cpp, which is always built.
    GGML_API const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type);

    typedef struct {
        int                   n_device;
        int                   split_dim;
        struct ggml_tensor *  tensor;
        struct ggml_tensor ** splits;
    } ggml_split_tensor_t;

    GGML_API struct ggml_tensor * ggml_reduce(
            struct ggml_context         * ctx,
            struct ggml_tensor         ** a,
            int                           n,
            enum ggml_op                  op);

    GGML_API struct ggml_tensor * ggml_fake_cpy(
            struct ggml_context         * ctx,
            struct ggml_tensor          * dst,
            struct ggml_tensor          * src);

    // ---------------------------------------------------------------------------------------
    // Threadpool parameters. Adopted verbatim from upstream ggml.h (2921-2946). This fork's
    // ggml-cpu.h was taken from upstream wholesale and uses ggml_threadpool_t and
    // struct ggml_threadpool_params, which nothing in this tree declared -- so ggml-cpu.h could
    // not compile. The three helpers are defined in ggml.c, ported from upstream's ggml.c.
    // ---------------------------------------------------------------------------------------

    enum ggml_sched_priority {
        GGML_SCHED_PRIO_LOW = -1,
        GGML_SCHED_PRIO_NORMAL,
        GGML_SCHED_PRIO_MEDIUM,
        GGML_SCHED_PRIO_HIGH,
        GGML_SCHED_PRIO_REALTIME
    };

    // threadpool params
    // Use ggml_threadpool_params_default() or ggml_threadpool_params_init() to populate the defaults
    struct ggml_threadpool_params {
        bool                cpumask[GGML_MAX_N_THREADS]; // mask of cpu cores (all-zeros means use default affinity settings)
        int                 n_threads;                   // number of threads
        enum ggml_sched_priority prio;                   // thread priority
        uint32_t            poll;                        // polling level (0 - no polling, 100 - aggressive polling)
        bool                strict_cpu;                  // strict cpu placement
        bool                paused;                      // start in paused state
    };

    struct ggml_threadpool;     // forward declaration, see ggml.c

    typedef struct ggml_threadpool * ggml_threadpool_t;

    GGML_API struct ggml_threadpool_params ggml_threadpool_params_default(int n_threads);
    GGML_API void                          ggml_threadpool_params_init   (struct ggml_threadpool_params * p, int n_threads);
    GGML_API bool                          ggml_threadpool_params_match  (const struct ggml_threadpool_params * p0, const struct ggml_threadpool_params * p1);

#ifdef  __cplusplus
}

//
// upstream-name compatibility (C++ only)
//
// ggml_ssm_conv changed CONTRACT, not just spelling. Ours is the older stateful Mamba-1 form
//     ggml_ssm_conv(ctx, s, x, c, sq, saved_steps)
// which carries conv state across steps, needs an I32 sequence-id tensor, and returns a single
// packed 1-D buffer holding the output AND the updated states. Upstream reduced it to a pure
// function of a pre-padded sequence:
//     ggml_ssm_conv(ctx, sx, c)   sx [T_pad, d_inner, n_s], c [d_conv, d_inner]
//                                 -> [d_inner, n_t, n_s], n_t = T_pad - d_conv + 1
// with the caller prepending the previous window into sx itself -- which is exactly what
// conformer.cpp / gemma4a.cpp / granite-speech.cpp do with their pad/roll/pad sequence.
//
// That reduced form is a stride-1, zero-padding DEPTHWISE 1-D convolution followed by a
// transpose, so it is expressed with ops we already have rather than by touching the existing
// six-argument kernel, which keeps working untouched for every current caller. This is an
// ARITY overload; at most one member of an overload set may have C linkage, and that is the
// original.
static inline struct ggml_tensor * ggml_ssm_conv(
        struct ggml_context * ctx,
        struct ggml_tensor  * sx,   // [T_pad, d_inner, n_s]
        struct ggml_tensor  * c) {  // [d_conv, d_inner]
    const int64_t d_conv  = c->ne[0];
    const int64_t d_inner = c->ne[1];

    GGML_ASSERT(sx->ne[1] == d_inner);
    GGML_ASSERT(sx->ne[3] == 1);   // upstream's sx is 3-D

    struct ggml_tensor * k = ggml_reshape_3d(ctx, c, d_conv, 1, d_inner);

    // The single-sequence composite loses a batch dimension, because ggml_mul_mat takes dims 2
    // and 3 of the result from the KERNEL argument. So each sequence is convolved on its own and
    // the results are concatenated on dim 2. This was asserted n_s == 1 until 2026-09-25 (every
    // mtmd caller encodes one utterance), which aborted test-backend-ops' test_ssm_conv (n_s = 4)
    // at graph BUILD -- and with it every case after it. n_s == 1 builds exactly as before.
    struct ggml_tensor * out = NULL;
    for (int64_t s = 0; s < sx->ne[2]; ++s) {
        struct ggml_tensor * sx_s = sx->ne[2] == 1 ? sx
            : ggml_cont(ctx, ggml_view_3d(ctx, sx, sx->ne[0], sx->ne[1], 1, sx->nb[1], sx->nb[2],
                                          (size_t) s * sx->nb[2]));

        // stride 1, no padding, no dilation -- sx is already padded by the caller
        struct ggml_tensor * y = ggml_conv_1d_dw(ctx, k, sx_s, 1, 0, 1);   // [n_t, d_inner, 1]

        // upstream returns channels in dim 0
        struct ggml_tensor * y_s = ggml_cont(ctx,
                ggml_transpose(ctx, ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1])));   // [d_inner, n_t]
        out = out ? ggml_concat(ctx, out, y_s, 2) : y_s;
    }
    return out;
}

#endif
