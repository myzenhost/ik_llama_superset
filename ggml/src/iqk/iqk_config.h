//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#if defined IQK_IMPLEMENT
#undef IQK_IMPLEMENT
#endif

#if defined __AVX2__ || defined __ARM_FEATURE_DOTPROD
#define IQK_IMPLEMENT
#endif

// The iqk sources moved INTO the CPU backend library when this fork adopted upstream's split
// build (ggml/src/ggml-cpu/CMakeLists.txt:125 lists iqk/iqk_mul_mat.cpp and its siblings). That
// changed which build symbol is in scope, and this macro was still testing only the old one.
//
// ggml_add_backend_library() gives a backend target GGML_BACKEND_BUILD and GGML_BACKEND_SHARED --
// NOT GGML_BUILD, which now belongs to ggml-base alone. So with BUILD_SHARED_LIBS=ON the iqk
// declarations resolved to __declspec(dllimport) inside the very library that DEFINES them:
//     iqk_mul_mat.cpp(503): error C2491: 'iqk_mul_mat': definition of dllimport function not allowed
// 7 diagnostics across iqk_mul_mat.cpp, iqk_flash_attn.cpp and iqk_kda.cpp, none of which is a
// defect in those files.
//
// Accepting either symbol keeps this correct wherever the sources are compiled: dllexport when
// built as part of ggml-base (GGML_BUILD) or as part of a backend library (GGML_BACKEND_BUILD),
// dllimport only for an actual outside consumer. Mirrors ggml-backend.h's GGML_BACKEND_API.
#if defined(GGML_SHARED) || defined(GGML_BACKEND_SHARED)
#    if defined(_WIN32) && !defined(__MINGW32__)
#        if defined(GGML_BUILD) || defined(GGML_BACKEND_BUILD)
#            define IQK_API __declspec(dllexport)
#        else
#            define IQK_API __declspec(dllimport)
#        endif
#    else
#        define IQK_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define IQK_API
#endif

#ifdef _MSC_VER
#define IQK_NOINLINE __declspec(noinline)
#define IQK_ALWAYS_INLINE inline
#if !defined __x86_64__ && defined _M_X64
#define __x86_64__
#endif
#else
#define IQK_NOINLINE __attribute__((__noinline__))
#define IQK_ALWAYS_INLINE __attribute__((__always_inline__))
#endif

#if defined __x86_64__
#if defined HAVE_FANCY_SIMD
    #undef HAVE_FANCY_SIMD
#endif
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    #define HAVE_FANCY_SIMD
#endif
#if defined HAVE_VNNI256
    #undef HAVE_VNNI256
#endif
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    #define HAVE_VNNI256
#endif
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    #define ggml_mm256_dpbusd_epi32 _mm256_dpbusd_epi32
    #define ggml_mm256_dpwssd_epi32 _mm256_dpwssd_epi32
    #define ggml_mm_dpbusd_epi32    _mm_dpbusd_epi32
#elif defined(__AVXVNNI__)
    #define ggml_mm256_dpbusd_epi32 _mm256_dpbusd_avx_epi32
    #define ggml_mm256_dpwssd_epi32 _mm256_dpwssd_avx_epi32
    #define ggml_mm_dpbusd_epi32    _mm_dpbusd_avx_epi32
#endif
#endif

