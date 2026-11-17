#include "../iqk_mmvq_templates.cuh"

// Upstream plain Q2_0 (ggml enum 401 here): 64 weights per block = one fp16 scale + 16 bytes of 2-bit
// codes, LSB-first, 4 codes per byte, sequential; value = d * (code - 1) (ggml-quants.c
// dequantize_row_q2_0). Flash-Next GSQ-RCO files carry these tensors (53 per file), and without this
// kernel they ran off the device: the scheduler copied each expert tensor device-to-host every token.
//
// Same arithmetic as vec_dot_q2_0_g128_q8_1, with a 64-weight block: qi == 4 ints, vdr == 2, so a
// thread owns ints [iqs, iqs+1] == 8 bytes == 32 codes == exactly one q8_1 block, iqs in {0, 2} and
// the q8_1 block for a thread is iqs/2 within the two that cover this 64-weight block.
__device__ __forceinline__ void vec_dot_q2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_q2_0 * bq = (const block_q2_0 *)vbq + kbx;

    const uint8_t    * qs  = bq->qs + 4*iqs;
    const block_q8_1 * bq8 = bq8_1 + iqs/2;

    int sumi = 0;
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        // One packed byte -> four CONSECUTIVE codes, one per byte lane (see the G128 instance for why
        // masking the packed int directly would pair the wrong q8 bytes). v is int for __dp4a.
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

    // sum_i d*(code_i - 1) * (ds.x*q8_i) = d * (ds.x*sum(code*q8) - ds.y), ds.y = ds.x*sum(q8)
    const float2 ds = __half22float2(bq8->ds);
    *result += __half2float(bq->d) * (ds.x * sumi - ds.y);
}

void mul_mat_vec_q2_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_Q2_0, 2, vec_dot_q2_0_q8_1>(args, stream);
}
