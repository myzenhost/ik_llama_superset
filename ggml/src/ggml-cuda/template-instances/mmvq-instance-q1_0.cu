#include "../iqk_mmvq_templates.cuh"

// Upstream plain Q1_0 (ggml enum 400 here): 128 weights per block = one fp16 scale + 16 bytes of
// sign bits, LSB-first; value = +d when the bit is set, -d when clear (ggml-quants.c
// dequantize_row_q1_0). The block layout is byte-identical to block_q1_0_g128, so this is the
// Q1_0_G128 kernel (mmvq-instance-q1_0_g128.cu) retargeted at the plain type; vdr == 1, qi == 4.
__device__ __forceinline__ void vec_dot_q1_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_q1_0 * bq = (const block_q1_0 *)vbq + kbx;

    const uint8_t    * qs  = bq->qs + 4*iqs;
    const block_q8_1 * bq8 = bq8_1 + iqs;

    int sumi = 0;
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        // One byte = eight weights = two dp4a (low nibble against q8 bytes 8k..8k+3, high nibble
        // against 8k+4..8k+7); a nibble is spread to four byte lanes by shifts 7/14/21.
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

    // value = d*(2*bit - 1): dot = d*(2*ds.x*sum(bit*q8) - ds.y)
    const float2 ds = __half22float2(bq8->ds);
    *result += __half2float(bq->d) * (2.0f * ds.x * sumi - ds.y);
}

void mul_mat_vec_q1_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_Q1_0, 1, vec_dot_q1_0_q8_1>(args, stream);
}
