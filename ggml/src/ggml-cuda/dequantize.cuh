#pragma once

#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = kvalues_iq4nl[vui & 0xF];
    v.y = kvalues_iq4nl[vui >>  4];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x = v.x * d;
    v.y = v.y * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q1_0_g128(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    // Bonsai 1-bit, group 128. Eight weights per byte, LSB-first, elements SEQUENTIAL.
    // Value is d when the bit is set and -d when clear -- taken verbatim from
    // dequantize_row_q1_0_g128 in iqk/iqk_quantize.cpp, not inferred.
    // REQUIRES qr == 1; see dequantize_q2_0_g128 below for what qr == 2 would do here.
    const block_q1_0_g128 * x = (const block_q1_0_g128 *) vx;

    const dfloat d = x[ib].d;

    const int i0 = iqs;
    const int i1 = iqs + 1;      // adjacent, per qr == 1
    const int b0 = (x[ib].qs[i0 >> 3] >> (i0 & 7)) & 1;
    const int b1 = (x[ib].qs[i1 >> 3] >> (i1 & 7)) & 1;

    v.x = b0 ? d : -d;
    v.y = b1 ? d : -d;
}

static __device__ __forceinline__ void dequantize_q2_0_g128(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    // Bonsai ternary, group 128. Four 2-bit codes per byte, LSB-first, elements SEQUENTIAL:
    // byte b holds elements 4b..4b+3. Value is d * (code - 1), so codes 0/1/2 mean -1/0/+1 --
    // the same shape as the -16 offset above, with 1 instead of 16.
    //
    // REQUIRES qr == 1 in ggml_cuda_type_traits, and that is not a detail. dequantize_block()
    // writes v.x to y[iybs+iqs] and v.y to y[iybs+iqs+y_offset] where
    // `y_offset = qr == 1 ? 1 : qk/2`. With qr == 2 the pair lands HALF A BLOCK APART, which is
    // right for the 4-bit nibble layouts above and WRONG for a sequential one -- it would scatter
    // every weight to the wrong column and produce fluent nonsense rather than an error.
    //
    // The unpack itself is the sequence validated against the F16 parent before any of this was
    // written (temp/research/validate_q2_unpack.py, 64/64 blocks, 8192 codes reconstructed
    // exactly): shift right by 2*(index mod 4), mask to 2 bits.
    //
    // CARRIED 2026-08-21 from portable/ik_llama/source, which had the whole Bonsai CUDA arm the
    // consolidated fork lacked. Independently corroborated here: the CPU path for this exact
    // layout measured Mean KLD 0.000399 and Same top p 98.333% against its own F16 parent, with
    // a 0.000121 noise floor -- so the unpack shape below is the one already shown correct.
    const block_q2_0_g128 * x = (const block_q2_0_g128 *) vx;

    const dfloat d = x[ib].d;

    const int i0 = iqs;
    const int i1 = iqs + 1;      // adjacent, per qr == 1
    const int c0 = (x[ib].qs[i0 >> 2] >> (2*(i0 & 3))) & 3;
    const int c1 = (x[ib].qs[i1 >> 2] >> (2*(i1 & 3))) & 3;

    v.x = c0;
    v.y = c1;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {1.0f, 1.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 1.0f) * d;
    v.y = (v.y - 1.0f) * d;
#endif // GGML_CUDA_F16
}

// Upstream's plain Q1_0 (128 x 1 bit) and Q2_0 (64 x 2 bit) -- ggml enum 400/401 here. Their element
// order and value rule are the G128 ones above (sequential, qr == 1; Q1_0: +-d, Q2_0: (code-1)*d,
// ggml-quants.c dequantize_row_q1_0/q2_0); only the Q2_0 block is 64 weights where G128 is 128.
static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const dfloat d = x[ib].d;

    const int i0 = iqs;
    const int i1 = iqs + 1;
    const int b0 = (x[ib].qs[i0 >> 3] >> (i0 & 7)) & 1;
    const int b1 = (x[ib].qs[i1 >> 3] >> (i1 & 7)) & 1;

    v.x = b0 ? d : -d;
    v.y = b1 ? d : -d;
}

static __device__ __forceinline__ void dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const dfloat d = x[ib].d;

    const int i0 = iqs;
    const int i1 = iqs + 1;
    const int c0 = (x[ib].qs[i0 >> 2] >> (2*(i0 & 3))) & 3;
    const int c1 = (x[ib].qs[i1 >> 2] >> (2*(i1 & 3))) & 3;

    v.x = c0;
    v.y = c1;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {1.0f, 1.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 1.0f) * d;
    v.y = (v.y - 1.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q6_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q6_0 * x = (const block_q6_0 *) vx;

    const dfloat d = x[ib].d;

    const uint8_t h = x[ib].qh[iqs%8] >> 4*(iqs/8);
    v.x = ((x[ib].qs[iqs] & 0xf) | ((h & 0x3) << 4));
    v.y = ((x[ib].qs[iqs] >>  4) | ((h & 0xc) << 2));

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {32.0f, 32.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 32.0f) * d;
    v.y = (v.y - 32.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_nvfp4_ks(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const float * dptr = (const float *) vx;
    const block_nvfp4 * x = (const block_nvfp4 *)(dptr + 1) + ib/4;

    const float d = dptr[0]*ggml_cuda_ue4m3_to_fp32(x->d[ib%4]) /* ggml_cuda_ue4m3_to_fp32 already returns raw*0.5 in this fork; ik's explicit 0.5f dropped */;

    const int vui = x->qs[8*(ib%4) + iqs];

    v.x = d*kvalues_mxfp4[vui & 0xF];
    v.y = d*kvalues_mxfp4[vui >>  4];
}
