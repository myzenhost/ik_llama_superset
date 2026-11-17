#include "../iqk_mmvq_templates.cuh"

// CARRIED 2026-08-21 from portable/ik_llama/source, verbatim. Without this instance the type
// cannot be routed to CUDA at all: ggml_cuda_op_mul_mat_vec_q_impl aborts for any type with no
// case, and adding the routing entry without the kernel crashed Bonsai-27B-Q1_0 at load with
// `ggml-cuda\mmvq.cu:140: fatal error`. The reference build that HAS this file serves the same
// model at 105.53 tok/s where ours managed 3.96.

// Bonsai Q1_0_G128: one bit per weight, 8 weights per byte, LSB-first, one fp16 scale per 128.
// Value is d when the bit is set and -d when clear -- taken from dequantize_row_q1_0_g128 in
// iqk/iqk_quantize.cpp (`x[ib].qs[i] & k_mask[j] ? d : -d`), not inferred.
//
// vdr == 1: qi == 4 for this type (128 bits = 16 bytes = 4 ints), so each thread owns ONE int
// == 4 bytes == 32 weights == exactly one q8_1 block. iqs indexes both the int and the q8 block.
__device__ __forceinline__ void vec_dot_q1_0_g128_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_q1_0_g128 * bq = (const block_q1_0_g128 *)vbq + kbx;

    const uint8_t    * qs  = bq->qs + 4*iqs;
    const block_q8_1 * bq8 = bq8_1 + iqs;

    int sumi = 0;
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        // One byte covers EIGHT weights, so it takes two dp4a: the low nibble against q8 bytes
        // 8k..8k+3 and the high nibble against 8k+4..8k+7. Expanding a nibble to four byte lanes
        // is the 1-bit form of the Q2_0_G128 trick -- shifts 7/14/21 instead of 6/12/18.
        // vlo/vhi are int, NOT uint32_t: CUDA declares both __dp4a(int,int,int) and
        // __dp4a(unsigned,unsigned,unsigned), so an unsigned first argument against the int
        // second argument is ambiguous and fails to compile. The mask caps these at 0x01010101,
        // well inside int32, so the cast is not a truncation.
        const uint32_t b   = qs[k];
        const uint32_t lo  = b & 0xF;
        const uint32_t hi  = (b >> 4) & 0xF;
        const int vlo = (int)((lo | (lo << 7) | (lo << 14) | (lo << 21)) & 0x01010101);
        const int vhi = (int)((hi | (hi << 7) | (hi << 14) | (hi << 21)) & 0x01010101);
        sumi = __dp4a(vlo, ((const int *)bq8->qs)[2*k + 0], sumi);
        sumi = __dp4a(vhi, ((const int *)bq8->qs)[2*k + 1], sumi);
    }
#else
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t b = qs[k];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sumi += int((b >> j) & 1) * bq8->qs[8*k + j];
        }
    }
#endif

    // value = d*(2*bit - 1), so the dot is d*(2*ds.x*sum(bit*q8) - ds.x*sum(q8)) and ds.y IS
    // ds.x*sum(q8). The thread covers exactly one q8_1 block, so ds.y is subtracted once.
    const float2 ds = __half22float2(bq8->ds);
    *result += __half2float(bq->d) * (2.0f * ds.x * sumi - ds.y);
}

void mul_mat_vec_q1_0_g128_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_Q1_0_G128, 1, vec_dot_q1_0_g128_q8_1>(args, stream);
}
