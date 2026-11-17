//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stdint.h>
#include <stddef.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"

// PLACED BEFORE the "#ifdef __cplusplus" guard ON PURPOSE. Inside it, the C preprocessor skips
// the whole region, so ggml-cpu.c -- a .c file, and one of the two users this macro exists to
// keep in agreement -- still saw it as undeclared. Measured: ggml-cpu.c:635 error
// "IQK_KT_ACTIVATION_TYPE undeclared here (not in a function)".
// MOVED HERE from iqk_quantize.cpp: ggml-cpu.c names this macro in type_traits_cpu and could not
// see a definition living in a .cpp. The whole point of the macro is that BOTH users share one
// definition, so it has to be in the header both of them already include.
// THE ACTIVATION TYPE THE KT (trellis) KERNELS REQUIRE, named ONCE so the two places that must
// agree cannot drift apart again. Those two places are this file's iqk_mul_mat calls and
// .vec_dot_type in ggml-cpu.c's type_traits_cpu; in one session they disagreed three different
// ways (Q8_2_X4, Q8_0_X4, Q8_K), each time producing wrong numbers rather than an error.
//
// The value is taken from iqk_set_kernels_ktquants(), which is what actually decides whether the
// call succeeds: its __x86_64__ arm accepts ONLY Q8_2_X4 and its "#else // !__x86_64__" arm
// accepts ONLY Q8_0_X4. Before this macro these four wrappers passed GGML_TYPE_Q8_K, which
// NEITHER arm accepts on ANY architecture, so they always returned without writing *s.
//
// The condition is __x86_64__, matching the SELECTOR -- not __AVX2__, which is what ggml.c's
// base table switches on. They are not the same test: a non-AVX2 x86_64 build takes the
// selector's x86 arm while ggml.c's table would name the other type. That is a latent defect in
// the tree, recorded here rather than inherited silently. Only the x86_64 arm is measured on this
// host.
#if defined __x86_64__ || defined _M_X64
#define IQK_KT_ACTIVATION_TYPE GGML_TYPE_Q8_2_X4
#else
#define IQK_KT_ACTIVATION_TYPE GGML_TYPE_Q8_0_X4
#endif

#ifdef __cplusplus
#define GGML_RESTRICT
extern "C" {
#else
#define GGML_RESTRICT restrict
#endif

struct quantize_user_data;

GGML_API void   quantize_row_iq2_k_ref(const float * GGML_RESTRICT x, block_iq2_k  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq2_k(const block_iq2_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq2_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq3_k_ref(const float * GGML_RESTRICT x, block_iq3_k  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq3_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq3_k(const block_iq3_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq3_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq3_ks_ref(const float * GGML_RESTRICT x, block_iq3_ks  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq3_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq3_ks(const block_iq3_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq3_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API size_t quantize_iq3ks_r16(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   quantize_row_iq3ks_r16_ref(const float * GGML_RESTRICT x, block_iq3_ks_r16  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq3ks_r16(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void   dequantize_row_iq3ks_r16(const block_iq3_ks_r16  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq3ks_r16_q8_2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_k_ref(const float * GGML_RESTRICT x, block_iq4_k  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_k(const block_iq4_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq5_k_ref(const float * GGML_RESTRICT x, block_iq5_k  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq5_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq5_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
// Bulk IQ5_K / IQ5_KS quantizers that ggml_quantize_chunk calls (defined in iqk_quantize.cpp, ggml-cpu). With __AVX2__ they run
// the AVX2 scale search of ik_llama #2608; otherwise, or with env GGML_IQK_QUANT_REFERENCE set to anything but "0", they call the
// scalar reference quantize_iq5_k / quantize_iq5_ks (iqk_quantize_base.cpp, ggml-base), which from_float_ref keeps using.
GGML_API size_t quantize_iq5_k_bulk(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API size_t quantize_iq5_ks_bulk(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
// Same for the interleaved IQ5_K_R4 / IQ5_KS_R4: each group of 4 rows takes the fast row search, then the R4 repack; the
// reference (and the non-AVX2 / GGML_IQK_QUANT_REFERENCE fallback) is quantize_iq5_k_r4 / quantize_iq5_ks_r4.
GGML_API size_t quantize_iq5_k_r4_bulk(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API size_t quantize_iq5_ks_r4_bulk(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
// Rows quantized by an AVX2 bulk fast path since process start (test hook for the dispatch contract).
GGML_API int64_t iqk_quantize_fast_path_rows(void);
GGML_API void   dequantize_row_iq5_k(const block_iq5_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq5_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq6_k_ref(const float * GGML_RESTRICT x, block_iq6_k  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq6_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq6_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq6_k(const block_iq6_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq6_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_ks_ref(const float * GGML_RESTRICT x, block_iq4_ks  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_ks(const block_iq4_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_kss_ref(const float * GGML_RESTRICT x, block_iq4_kss  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_kss(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_kss(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_kss(const block_iq4_kss  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_kss_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_mxfp4_q8_0_x4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_nvfp4_ks_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_nvfp4_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_nvfp4_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_nvfp4_ks(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_nvfp4_ks_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq2_ks_ref(const float * GGML_RESTRICT x, block_iq2_ks  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq2_ks(const block_iq2_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq2_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq2_kl_ref(const float * GGML_RESTRICT x, block_iq2_kl  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_kl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_kl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq2_kl(const block_iq2_kl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq2_kl_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq1_kt_ref(const float * GGML_RESTRICT x, block_iq1_kt  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq1_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq1_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq1_kt(const block_iq1_kt  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq1_kt_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, block_iq2_kt  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq2_kt(const block_iq2_kt  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq2_kt_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq3_kt_ref(const float * GGML_RESTRICT x, block_iq3_kt  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq3_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq3_kt(const block_iq3_kt  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq3_kt_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, block_iq4_kt  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_kt(const block_iq4_kt  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_kt_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq5_ks_ref(const float * GGML_RESTRICT x, block_iq5_ks  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq5_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq5_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq5_ks(const block_iq5_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq5_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_nl_r4_ref(const float * GGML_RESTRICT x, block_iq4_nl_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_nl_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_nl_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_nl_r4(const block_iq4_nl_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_nl_r4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q4_0_r8_ref(const float * GGML_RESTRICT x, block_iq4_nl_r8  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q4_0_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q4_0_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q4_0_r8(const block_iq4_nl_r8  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q4_0_r8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q2_0_r8_ref(const float * GGML_RESTRICT x, block_q2_0_r8  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q2_0_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q2_0_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q2_0_r8(const block_q2_0_r8  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q2_0_r8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q8_0_r8_ref(const float * GGML_RESTRICT x, block_q8_0_r8  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q8_0_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q8_0_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q8_0_r8(const block_q8_0_r8  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q8_0_r8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q5_0_r4_ref(const float * GGML_RESTRICT x, block_q5_0_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q5_0_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q5_0_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q5_0_r4(const block_q5_0_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q5_0_r4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q6_0_r4_ref(const float * GGML_RESTRICT x, block_q6_0_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q6_0_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q6_0_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q6_0_r4(const block_q6_0_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q6_0_r4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_xs_r8_ref(const float * GGML_RESTRICT x, block_iq4_xs_r8 * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_xs_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_xs_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_xs_r8(const block_iq4_xs_r8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_xs_r8_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq2_bn_ref (const float * GGML_RESTRICT x, block_iq2_bn  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_bn (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void   dequantize_row_iq2_bn (const block_iq2_bn  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_bn (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   vec_dot_iq2_bn_q8_K64(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq2_bn_r4_ref (const float * GGML_RESTRICT x, block_iq2_bn  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_bn_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void   dequantize_row_iq2_bn_r4(const block_iq2_bn * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_bn_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   vec_dot_iq2_bn_r4_q8_K64(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q3_k_r4_ref(const float * GGML_RESTRICT x, block_q3_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q3_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q3_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q3_k_r4(const block_q3_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q3_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q2_k_r4_ref(const float * GGML_RESTRICT x, block_q2_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q2_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q2_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q2_k_r4(const block_q2_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q2_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q4_k_r4_ref(const float * GGML_RESTRICT x, block_q4_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q4_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q4_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q4_k_r4(const block_q4_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q4_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q5_k_r4_ref(const float * GGML_RESTRICT x, block_q5_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q5_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q5_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q5_k_r4(const block_q5_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q5_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q6_k_r4_ref(const float * GGML_RESTRICT x, block_q6_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q6_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q6_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q6_k_r4(const block_q6_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q6_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq5_k_r4_ref(const float * GGML_RESTRICT x, block_iq5_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq5_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq5_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq5_k_r4(const block_iq5_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq5_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_k_r4_ref(const float * GGML_RESTRICT x, block_iq4_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_k_r4(const block_iq4_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq3_k_r4_ref(const float * GGML_RESTRICT x, block_iq3_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq3_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq3_k_r4(const block_iq3_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq3_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq2_k_r4_ref(const float * GGML_RESTRICT x, block_iq2_k_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq2_k_r4(const block_iq2_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq2_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq4_ks_r4_ref(const float * GGML_RESTRICT x, block_iq4_ks_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_ks_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_ks_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_ks_r4(const block_iq4_ks_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_ks_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void   repack_iq4_ks(int nrows, int n_per_row, const block_iq4_ks * GGML_RESTRICT x, block_iq4_ks_r4 * GGML_RESTRICT y, bool online);

GGML_API void   quantize_row_iq4_ks_r16_ref(const float * GGML_RESTRICT x, block_iq4_ks_r16  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq4_ks_r16(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_ks_r16(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq4_ks_r16(const block_iq4_ks_r16 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq4_ks_r16_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void   repack_iq4_ks_r16(int nrows, int n_per_row, const block_iq4_ks * GGML_RESTRICT x, block_iq4_ks_r16 * GGML_RESTRICT y, bool online, int n_per_row_stored);

GGML_API void   quantize_row_iq5_ks_r4_ref(const float * GGML_RESTRICT x, block_iq5_ks_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq5_ks_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq5_ks_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq5_ks_r4(const block_iq5_ks_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq5_ks_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void   repack_iq5_ks(int nrows, int n_per_row, const block_iq5_ks * GGML_RESTRICT x, block_iq5_ks_r4 * GGML_RESTRICT y, bool online);

GGML_API void   quantize_row_iq5_ks_r16_ref(const float * GGML_RESTRICT x, block_iq5_ks_r16  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq5_ks_r16(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq5_ks_r16(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API size_t quantize_iq5_ks_r16_bulk(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq5_ks_r16(const block_iq5_ks_r16 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq5_ks_r16_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void   repack_iq5_ks_r16(int nrows, int n_per_row, const block_iq5_ks * GGML_RESTRICT x, block_iq5_ks_r16 * GGML_RESTRICT y, bool online, int n_per_row_stored);

GGML_API void   quantize_row_iq2_xxs_r4_ref(const float * GGML_RESTRICT x, block_iq2_xxs_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_xxs_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_xxs_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq2_xxs_r4(const block_iq2_xxs_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq2_xxs_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq2_xs_r4_ref(const float * GGML_RESTRICT x, block_iq2_xs_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_xs_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_xs_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq2_xs_r4(const block_iq2_xs_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq2_xs_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq2_s_r4_ref(const float * GGML_RESTRICT x, block_iq2_s_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq2_s_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_s_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq2_s_r4(const block_iq2_s_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq2_s_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq3_xxs_r4_ref(const float * GGML_RESTRICT x, block_iq3_xxs_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq3_xxs_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_xxs_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq3_xxs_r4(const block_iq3_xxs_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq3_xxs_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq3_s_r4_ref(const float * GGML_RESTRICT x, block_iq3_s_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq3_s_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_s_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq3_s_r4(const block_iq3_s_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq3_s_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq1_s_r4_ref(const float * GGML_RESTRICT x, block_iq1_s_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq1_s_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq1_s_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq1_s_r4(const block_iq1_s_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq1_s_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_iq1_m_r4_ref(const float * GGML_RESTRICT x, block_iq1_m_r4  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_iq1_m_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq1_m_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_iq1_m_r4(const block_iq1_m_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_iq1_m_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q8_k_r8_ref(const float * GGML_RESTRICT x, block_q8_k_r8  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q8_k_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q8_k_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q8_k_r8(const block_q8_k_r8  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q8_k_r8_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q8_k_r16_ref(const float * GGML_RESTRICT x, block_q8_k_r16  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q8_k_r16(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q8_k_r16(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q8_k_r16(const block_q8_k_r16  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q8_k_r16_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q8_KV_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q8_KV(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q8_KV(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q8_KV(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q8_KV_q8_KV(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q8_KV_r8_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q8_KV_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q8_KV_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q8_KV_r8(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q8_KV_r8_q8_KV(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q1_0_g128_ref(const float * GGML_RESTRICT x, block_q1_0_g128  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q1_0_g128(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q1_0_g128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q1_0_g128(const block_q1_0_g128  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q1_0_g128_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_q2_0_g128_ref(const float * GGML_RESTRICT x, block_q2_0_g128  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q2_0_g128(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q2_0_g128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q2_0_g128(const block_q2_0_g128  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q2_0_g128_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);


GGML_API void   quantize_row_q1_0_g128_r8_ref(const float * GGML_RESTRICT x, block_q1_0_g128_r8  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_q1_0_g128_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_q1_0_g128_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_q1_0_g128_r8(const block_q1_0_g128_r8  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_q1_0_g128_r8_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void   repack_q1_0_g128_r8(int nrows, int n_per_row, const block_q1_0_g128 * GGML_RESTRICT x, block_q1_0_g128_r8 * GGML_RESTRICT y, bool online);

GGML_API void   repack_pq2_0_r8(int nrows, int n_per_row, const block_pq2_0 * GGML_RESTRICT x, block_pq2_0_r8 * GGML_RESTRICT y, bool online);
GGML_API void   dequantize_row_pq2_0_r8(const block_pq2_0_r8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_pq2_0_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_pq2_0_r8_ref(const float * GGML_RESTRICT x, block_pq2_0_r8 * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_pq2_0_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   vec_dot_pq2_0_r8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   repack_ptq1_0_r8(int nrows, int n_per_row, const block_ptq1_0 * GGML_RESTRICT x, block_ptq1_0_r8 * GGML_RESTRICT y, bool online);
GGML_API void   dequantize_row_ptq1_0_r8(const block_ptq1_0_r8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_ptq1_0_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_ptq1_0_r8_ref(const float * GGML_RESTRICT x, block_ptq1_0_r8 * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_ptq1_0_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   vec_dot_ptq1_0_r8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void   quantize_row_mxfp4_r8_ref(const float * GGML_RESTRICT x, block_mxfp4_r8  * GGML_RESTRICT y, int64_t k);
GGML_API void   quantize_row_mxfp4_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_mxfp4_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix, const struct quantize_user_data * use_data);
GGML_API void   dequantize_row_mxfp4_r8(const block_mxfp4_r8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   vec_dot_mxfp4_r8_q8_2_x4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void iqk_quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
GGML_API void quantize_row_q8_K64_ref(const float * GGML_RESTRICT x, block_q8_K64 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K128(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K16(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K32(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_KR8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_2_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void iqk_quantize_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

GGML_API void repack_f32_bf16_r16 (const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row);
GGML_API void repack_bf16_bf16_r16(const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row);

GGML_API void iqk_repack_tensor(struct ggml_tensor * tensor);
GGML_API bool iqk_modify_tensor(struct ggml_tensor * tensor);

int iqk_repacked_type(const struct ggml_tensor * tensor); // int instead of ggml_type so we don't need to include ggml.h
GGML_API bool iqk_should_modify_tensor(const struct ggml_tensor * tensor);

// So we can re-pack Microsoft's BitNet I2_S quants
GGML_API void dequantize_row_ms_i2s(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

typedef void (*to_float_t)  (const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
typedef void (*from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);
void iqk_quantize_any(int from_type, int to_type,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                      uint64_t nb0, uint64_t nb1, uint64_t nb2, uint64_t nb3,
                      const void * GGML_RESTRICT x, void * GGML_RESTRICT y, void * work_buffer,
                      to_float_t to_float, from_float_t from_float, int ith, int nth);

GGML_API bool iqk_validate_tensor(const struct ggml_tensor * src);

GGML_API int  iqk_kt_codebook(int type, int bank, const int ** offsets, const int ** points, const float ** values, const float ** mid);
GGML_API void iqk_kt_finish_row(int type, const void * ypad, void * qrow, int64_t n_per_row);
GGML_API void iqk_kt_pad_row(const float * src, float * dst, int64_t n_per_row);
GGML_API int  iqk_ks_index(int type, const int8_t ** index);
GGML_API int  iqk_ks_simple_index(int type, const int8_t ** index);

#ifdef __cplusplus
}
#endif
