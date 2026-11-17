#include "../iqk_mmvq_templates.cuh"

// CARRIED 2026-08-21 from portable/ik_llama/source, verbatim. This type already ran on the
// device here via dequant->cuBLAS (4/4 factual anchors, 11.65 tok/s, graph splits 947 -> 2), but
// the reference build that HAS this instance served the same model at 93.85 tok/s -- an 8.06x
// gap that is exactly the missing MMVQ kernel.

// Bonsai Q2_0_G128: ternary codes {0,1,2} in 2-bit slots, LSB-first, 4 codes per byte, one fp16
// scale per 128 weights. Dequantized value is d * (code - 1).
//
// vdr == 2, so each thread owns ints [iqs, iqs+1] of the block == 8 bytes == 32 codes, which is
// EXACTLY one q8_1 block (QK8_1 == 32). iqs is therefore always even and the alignment is exact,
// which is what lets the offset term below be applied once per call rather than pro-rated.
// (qi == 8 for this type: 128 weights * 2 bits = 32 bytes = 8 ints. See ggml_cuda_type_traits.)
__device__ __forceinline__ void vec_dot_q2_0_g128_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_q2_0_g128 * bq = (const block_q2_0_g128 *)vbq + kbx;

    const uint8_t    * qs  = bq->qs + 4*iqs;
    const block_q8_1 * bq8 = bq8_1 + iqs/2;

    int sumi = 0;
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        // Expand ONE packed byte into four CONSECUTIVE codes, one per byte lane.
        //   lane0 = b&3, lane1 = (b>>2)&3, lane2 = (b>>4)&3, lane3 = (b>>6)&3
        // Masking the packed int directly -- the IQ2_BN approach, `vl & 0x03030303` -- would
        // instead select codes STRIDED BY 4, and those line up with no contiguous run of q8
        // bytes under this sequential layout. That stride is the whole reason the IQ2_BN masks
        // cannot be reused here, and reusing them was the first thing that looked correct.
        // v is int, NOT uint32_t: CUDA declares both __dp4a(int,int,int) and
        // __dp4a(unsigned,unsigned,unsigned), so a uint32_t first argument against an int second
        // is ambiguous and fails to compile. The mask caps the value at 0x03030303, well inside
        // int32, so the cast is not a truncation.
        const uint32_t b = qs[k];
        const int v = (int)((b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303);
        sumi = __dp4a(v, ((const int *)bq8->qs)[k], sumi);
    }
#else
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t b = qs[k];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi += int((b >> (2*j)) & 3) * bq8->qs[4*k + j];
        }
    }
#endif

    // sum_i d*(code_i - 1) * (ds.x*q8_i) = d * (ds.x*sum(code*q8) - ds.x*sum(q8))
    // and ds.y IS ds.x*sum(q8) for the block, so the -1 offset costs one subtraction.
    const float2 ds = __half22float2(bq8->ds);
    *result += __half2float(bq->d) * (ds.x * sumi - ds.y);
}

void mul_mat_vec_q2_0_g128_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_Q2_0_G128, 2, vec_dot_q2_0_g128_q8_1>(args, stream);
}
