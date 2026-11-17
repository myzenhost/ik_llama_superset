//
// TurboQuant KV support for the flash-attention *vec* kernels.
//
// WHY THIS FILE EXISTS
// --------------------
// A turbo KV block stores its 128 elements ROTATED: the writer L2-normalises the 128-group,
// applies R = diag(signs2) . H_128 . diag(signs1) (an orthogonal fast Walsh-Hadamard rotation),
// then quantises the rotated coefficients to 2/3-bit Lloyd-Max centroids with ONE fp16 norm per
// 128-element block (ggml-turbo-quant.c: quantize_row_turbo{2,3}_0_ref).
//
// The standing holding in ggml/src/CMakeLists.txt read that as blocking turbo in the vec kernels,
// because "no per-element dequantize_1 can ever be correct" -- one thread holding one element
// cannot invert a rotation that mixes all 128.
//
// THAT HOLDING IS REFUTED, and the refutation is what this file implements. R is orthogonal, so
//     (R q) . (R k) == q . k                      -> the K side needs NO un-rotation at all,
//                                                     only Q rotated ONCE into the same basis.
//     sum_i a_i (R v_i) == R (sum_i a_i v_i)      -> the V side may accumulate entirely in the
//                                                     ROTATED domain and be un-rotated ONCE, on
//                                                     the finished head-dim vector.
// Both identities are measured, with negative controls, in
//     temp/_agAA_turbo_per_element_dequant_challenge.py
// re-run 2026-09-03 by the implementing agent before any of this was written:
//     per-element path vs full-dequant turbo attention   rel L2 6.194e-16   (mechanism correct)
//     omit the Q rotation                                rel L2 0.7957      (control fires)
//     omit the final inverse rotation                    rel L2 1.4505      (control fires)
//     one flipped writer/reader sign                     rel L2 0.2371      (control fires)
// and again on this device code by temp/_agAC_turbo_fattn_kernel_proof.cu.
//
// So the per-element decode below is deliberately rotation-free: it is the SAME decode
// dequantize_row_turbo3_0 performs before its group-wide inverse, and it is correct precisely
// because the two rotations are hoisted out to the two places where a whole 128-vector is
// block-visible.
//
// TURBO4 IS IN SCOPE FROM 2026-10-05 (commit dd765fbf0 on engine/g-turbo-backends-1005).
//   It was excluded on a diagnosed mechanism: the CPU turbo4 was a 3-bit + QJL layout whose two inverse sites
//   applied a dense LCG+Gram-Schmidt basis that is not the FWHT's transpose (FWHT^T.FWHT vs I = 2.220e-16, but the
//   composition the CPU performed, dense^T.FWHT vs I = 1.197, round-trip error 1.408; task #278). That layout is
//   retired: TURBO4_USE_4BIT now defaults to 1, and quantize_row_turbo4_0_ref writes the same FWHT-rotated,
//   norm-corrected centroid block as turbo2/3 (4-bit indices, 66-byte block). tests/test-turbo3-roundtrip.cpp pins
//   the new contract (turbo4 relative error 0.009138, cosine 0.995431; the retired layout measured 1.957019 / 0.005235
//   on the same test), so the per-element rotated-domain decode below is exactly dequantize_row_turbo4_0 minus the
//   group-wide inverse, as for turbo2/3.
//   WRITER: the CUDA KV writer carries a turbo4 case as well (turbo-write.h NBITS=4, set-rows.cu / cpy.cu cpy_blck_f32_turbo4),
//   byte-identical to quantize_row_turbo4_0_ref in tests/test-turbo-write-host.cpp.
//
// MECHANISM-TESTED: the fused path in this header is measured against a full-dequant reference on
// the GPU, with a wrong-rotation control that must explode, by temp/_agAC_turbo_fattn_kernel_proof.cu.

// OPT-IN PER TRANSLATION UNIT, and the reason is MEASURED, not stylistic.
//
// fattn-vec-common.cuh is included by every fattn-vec instance TU. Pulling turbo-quant.cuh in
// unconditionally put the turbo __constant__ tables (signs 128 and 64, the 2/3/4-bit centroids and
// midpoints, the TQ weight tables) and the InnerQ __device__ globals into ~130 TUs that never touch
// them. temp/_agAC_nonturbo_codegen_regression.sh measured the effect on the untouched
// q8_0-q8_0 instance: the PTX INSTRUCTION STREAM was identical, but the SASS moved every kernel
// parameter's constant-bank address (c[0x4][0x98] -> c[0x4][0xc8]) because the unused tables sit
// ahead of the parameters in bank 4. Same instructions, different addresses -- functionally fine,
// but it means "the ordinary path is unchanged" could no longer be stated as byte-identity, and it
// duplicates ~2.5 KB of constant data per TU because the tables have internal linkage.
//
// So a TU opts in by defining GGML_CUDA_FATTN_TURBO before including this header; the turbo
// instance files do. A TU that has NOT opted in still sees the two predicates below (they are
// plain constexpr over an enum and cost nothing), so the vec kernels' static_asserts still fire --
// instantiating a turbo type in a non-opted-in TU is a BUILD ERROR naming the missing opt-in,
// never a silent nullptr device function pointer.

#pragma once

#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// Which KV types this fused path handles. Always available; never pulls in the tables.
// ---------------------------------------------------------------------------------------------

static constexpr __host__ __device__ bool turbo_fattn_type_supported(ggml_type t) {
#ifdef GGML_CUDA_FATTN_TURBO
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
#else
    return (void) t, false;
#endif
}

// Any turbo type at all, including turbo4 -- used to give turbo4 a clear, named rejection instead
// of letting it look like an ordinary unknown type.
static constexpr __host__ __device__ bool turbo_fattn_type_is_turbo(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

#ifdef GGML_CUDA_FATTN_TURBO

#include "turbo-quant.cuh"

// ---------------------------------------------------------------------------------------------
// Cooperative 128-group rotation over a shared-memory vector.
//
// One thread per element: thread `t` in [0, NELEM) owns x[t]. NELEM must be a multiple of 128
// because that is the turbo block/rotation group size (QK_TURBO3 == QK_TURBO2 == 128).
// DIR == 0 forward (signs1 -> FWHT -> signs2), DIR == 1 inverse (signs2 -> FWHT -> signs1);
// R is orthogonal so the inverse is R^T, which is exactly the forward run with the sign arrays
// swapped. This mirrors k_turbo_wht_f32 in turbo-wht.cu -- same butterfly, same normalisation.
//
// The caller must __syncthreads() before calling (x must be fully written) and after (x is only
// valid once every stage has completed). Every __syncthreads() below is reached by ALL threads of
// the block: the divergent `if` guards only the butterfly's read-modify-write, never the barrier.
// ---------------------------------------------------------------------------------------------

template <int NELEM, int DIR>
static __device__ __forceinline__ void turbo_fattn_rotate_shared(float * x, const int t) {
    static_assert(NELEM % QK_TURBO3 == 0, "turbo rotation groups are 128 elements wide");
    static_assert(DIR == 0 || DIR == 1, "DIR must be 0 (forward) or 1 (inverse)");

    float * g = x + (t / 128) * 128;  // this thread's 128-element group
    const int l = t % 128;            // lane within the group

    g[l] *= (DIR == 0) ? TURBO_WHT_SIGNS1[l] : TURBO_WHT_SIGNS2[l];
    __syncthreads();

#define TURBO_FATTN_WHT_STAGE(h)                                                    \
    if ((l % (2*(h))) < (h)) {                                                      \
        const float a = g[l], b = g[l + (h)];                                       \
        g[l] = a + b; g[l + (h)] = a - b;                                           \
    }                                                                               \
    __syncthreads();

    TURBO_FATTN_WHT_STAGE( 1)
    TURBO_FATTN_WHT_STAGE( 2)
    TURBO_FATTN_WHT_STAGE( 4)
    TURBO_FATTN_WHT_STAGE( 8)
    TURBO_FATTN_WHT_STAGE(16)
    TURBO_FATTN_WHT_STAGE(32)
    TURBO_FATTN_WHT_STAGE(64)
#undef TURBO_FATTN_WHT_STAGE

    const float s = (DIR == 0) ? TURBO_WHT_SIGNS2[l] : TURBO_WHT_SIGNS1[l];
    g[l] = g[l] * 0.08838834764831845f /* 1/sqrt(128) */ * s;
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------
// Per-element decode, ROTATED DOMAIN. Exactly the dequantize_1_* contract: one element, one block
// norm, no inverse rotation. Correct only because both rotations are hoisted (see header comment).
// ---------------------------------------------------------------------------------------------

template <typename T>
static __device__ __forceinline__ T dequantize_1_turbo3_0(const void * __restrict__ vx, const int64_t i) {
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;

    const int64_t ib = i / QK_TURBO3;
    const int     j  = (int)(i % QK_TURBO3);

    return (T) turbo3_dequant_element(&x[ib], j, __half2float(x[ib].norm));
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_turbo2_0(const void * __restrict__ vx, const int64_t i) {
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;

    const int64_t ib = i / QK_TURBO2;
    const int     j  = (int)(i % QK_TURBO2);

    return (T) turbo2_dequant_element(&x[ib], j, __half2float(x[ib].norm));
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_turbo4_0(const void * __restrict__ vx, const int64_t i) {
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;

    const int64_t ib = i / QK_TURBO4;
    const int     j  = (int)(i % QK_TURBO4);

    return (T) turbo4_dequant_element(&x[ib], j, __half2float(x[ib].norm));
}

// ---------------------------------------------------------------------------------------------
// vec_dot_KQ for turbo K.
//
// Q reaches here ALREADY ROTATED (the kernel rotates it once, in shared memory, before the KV
// loop) and NOT quantised to q8_1 -- turbo decode produces floats, so there is no integer dot to
// use. Layout therefore matches the f16-K path: thread threadIdx.x holds the pair at half2/float2
// index k_KQ_0 + threadIdx.x, i.e. elements 2*k and 2*k+1. Returns this thread's partial; the
// caller warp-reduces, exactly as for every other K type.
// ---------------------------------------------------------------------------------------------

#define TURBO_FATTN_DEFINE_VEC_DOT(SUFFIX, BLOCK_T, QK, DEQUANT_ELEM)                              \
template <typename T, int Dk>                                                                      \
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_##SUFFIX(                                 \
    const char * __restrict__ K_c, const void * __restrict__ Q_v,                                  \
    const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {                             \
                                                                                                   \
    static_assert(Dk % (QK) == 0, "turbo K needs a head size that is a whole number of blocks");    \
                                                                                                   \
    const BLOCK_T * K_b = (const BLOCK_T *) K_c;                                                    \
    GGML_UNUSED(Q_q8);                                                                              \
    GGML_UNUSED(Q_ds_v);                                                                            \
                                                                                                    \
    float sum = 0.0f;                                                                               \
                                                                                                    \
    _Pragma("unroll")                                                                               \
    for (int k_KQ_0 = 0; k_KQ_0 < Dk/2; k_KQ_0 += WARP_SIZE) {                                      \
        const int k  = k_KQ_0 + threadIdx.x;   /* pair index */                                     \
        const int e0 = 2*k;                    /* first element of the pair */                      \
        const int ib = e0 / (QK);                                                                   \
        const int j0 = e0 % (QK);                                                                   \
                                                                                                    \
        const float nrm = __half2float(K_b[ib].norm);                                               \
        const float k0  = DEQUANT_ELEM(&K_b[ib], j0,     nrm);                                      \
        const float k1  = DEQUANT_ELEM(&K_b[ib], j0 + 1, nrm);                                      \
                                                                                                    \
        float2 q;                                                                                   \
        if (std::is_same<T, half>::value) {                                                         \
            q = __half22float2(((const half2 *) Q_v)[k_KQ_0/WARP_SIZE]);                            \
        } else {                                                                                    \
            q = ((const float2 *) Q_v)[k_KQ_0/WARP_SIZE];                                           \
        }                                                                                           \
                                                                                                    \
        sum += k0*q.x + k1*q.y;                                                                     \
    }                                                                                               \
                                                                                                    \
    return (T) sum;                                                                                 \
}

TURBO_FATTN_DEFINE_VEC_DOT(turbo3_0, block_turbo3_0, QK_TURBO3, turbo3_dequant_element)
TURBO_FATTN_DEFINE_VEC_DOT(turbo2_0, block_turbo2_0, QK_TURBO2, turbo2_dequant_element)
TURBO_FATTN_DEFINE_VEC_DOT(turbo4_0, block_turbo4_0, QK_TURBO4, turbo4_dequant_element)

#undef TURBO_FATTN_DEFINE_VEC_DOT

#else // !GGML_CUDA_FATTN_TURBO

// NAME-LOOKUP STUB, and it exists for a measured reason.
//
// The vec kernels call turbo_fattn_rotate_shared inside `if constexpr (turbo_K)`. A discarded
// if-constexpr branch inside a template is still PARSED, and this name is not dependent on the
// enclosing template parameters for lookup purposes, so a TU that has not opted in failed to
// compile with "identifier turbo_fattn_rotate_shared is undefined" -- on the ordinary q8_0, q4_0,
// f16 and hs576 instances, none of which will ever call it.
//
// The static_assert condition is written in terms of DIR precisely so it is DEPENDENT: it does not
// fire on parsing, only if something actually instantiates this stub. So a genuine mistake -- a
// turbo path reaching here without the opt-in -- is a build error naming the cause, while the
// discarded branch costs nothing.
template <int NELEM, int DIR>
static __device__ __forceinline__ void turbo_fattn_rotate_shared(float *, const int) {
    static_assert(DIR < 0,
        "turbo_fattn_rotate_shared was actually instantiated in a translation unit that did not "
        "#define GGML_CUDA_FATTN_TURBO before including fattn-vec.cuh");
}

#endif // GGML_CUDA_FATTN_TURBO
