#include "../iqk_mmvq_templates.cuh"

// TQ1_0 x q8_1 mul_mat_vec -- the FIRST CUDA kernel for this type in any tree we know of.
// llama.cpp PR #11183 (compilade), which added TQ2_0 GPU support, says outright: "TQ1_0 is out of
// scope of this PR, but GPU support for it will also come eventually". It has not. So this is
// written from the CPU reference (dequantize_row_tq1_0 / quantize_row_tq1_0_ref in ggml-quants.c,
// both read in full), not ported.
//
// A LOCAL 2-byte-safe int load, for the same reason mmvq-instance-tq2_0.cu carries one: the iqk
// header chain cannot include vecdotq.cuh (duplicate get_int_from_table_16 definitions), so
// get_int_b2 is unreachable here. See that file's comment block for the two rejected alternatives.
//
// AND THE ALIGNMENT HAZARD IS REAL FOR THIS TYPE TOO, by the same arithmetic that crashed the host
// on 2026-09-01 with TQ2_0: sizeof(block_tq1_0) = qs[48] + qh[4] + ggml_half = 54 bytes. 54 % 4 == 2,
// so consecutive blocks start at 0, 54, 108, 162, ... and every ODD block is 2-mod-4. A plain
// (const int *) cast would be a 4-byte load at a 2-byte address -- a hard "misaligned address"
// fault, not a wrong answer. Every load below goes through this helper. qs starts at block offset 0
// and qh at offset 48, both even, so 2-byte alignment holds for both arrays.
static __device__ __forceinline__ int tq1_0_get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    return (int)x16[2*i32 + 0] | ((int)x16[2*i32 + 1] << 16);
}

// Decode FOUR consecutive trits that share one digit position, entirely in 32-bit integer
// arithmetic -- no per-element loop, no lookup table, and no SIMD-video intrinsic dependency.
//
// The CPU reference for one element is exactly two lines:
//     uint8_t q  = qs[byte] * pow3[n];        // NOTE: truncated to 8 bits, deliberately
//     int16_t xi = ((uint16_t) q * 3) >> 8;   // -> 0,1,2
// Base-3 packing: five trits ride in one byte because 3^5 = 243 < 256, and the n-th trit is
// recovered by multiplying by 3^n and taking the top of the byte. The 8-bit truncation is part of
// the format, not a C accident -- it is what discards the digits already consumed.
//
// Doing that four-lanes-at-a-time needs both steps to stay inside their own byte lane:
//
//   STEP 1, the multiply. Split into even and odd lanes so each product owns a full 16-bit field.
//   pow3[n] <= 81, so a lane product is at most 255*81 = 20655 = 0x50AF -- it fits in 16 bits and
//   cannot carry into the neighbouring field. Masking the result with 0x00FF00FF then keeps the LOW
//   byte of each field, which IS the `uint8_t` truncation the reference performs.
//
//   STEP 2, the (q*3)>>8. Here the mask must come BEFORE the shift, and that is the one subtle
//   line. q*3 <= 765 = 0x2FD, so each field's answer sits in its bits [9:8] / [25:24] -- i.e. in
//   the HIGH byte of the field. Masking with 0xFF00FF00 first selects exactly those high bytes,
//   and only then does >>8 bring them down to the lane's own byte position. Masking AFTER the
//   shift instead would read field 2's answer out of bits that belong to the next lane and
//   silently return zero for it.
//
// Result: four codes in {0,1,2} in four bytes, ready for dp4a against four int8 activations.
static __device__ __forceinline__ int tq1_0_decode4(const int v, const int p) {
    const int even = ((v      & 0x00FF00FF) * p) & 0x00FF00FF;  // lanes 0,2 -> byte 0,2
    const int odd  = (((v>>8) & 0x00FF00FF) * p) & 0x00FF00FF;  // lanes 1,3 -> byte 0,2
    const int xe   = ((even * 3) & 0xFF00FF00) >> 8;            // lanes 0,2 in place
    const int xo   = ((odd  * 3) & 0xFF00FF00) >> 8;            // lanes 1,3, still at byte 0,2
    return xe | (xo << 8);
}

// One thread owns ONE q8_1 block (32 elements) of a 256-element super-block; qi/vdr = 32/4 = 8
// threads per super-block, the same shape TQ2_0 uses.
//
// THE ELEMENT MAP, inverted from dequantize_row_tq1_0's three write loops rather than guessed.
// sizeof(qs) is 48 and 48 % 32 == 16, so the reference walks qs in a 32-byte group then a 16-byte
// tail, and finally qh:
//     elements   0..159   n = i/32,       m = i%32,       byte qs[m]         (5 trits/byte)
//     elements 160..239   e = i-160, n = e/16, m = e%16,  byte qs[32+m]      (5 trits/byte)
//     elements 240..255   e = i-240, n = e/4,  j = e%4,   byte qh[j]         (4 trits/byte)
// so, per q8_1 block:
//     ib 0..4  one digit position (n = ib) across all 32 bytes qs[0..31]
//     ib 5     qs[32..47] twice, digits n=0 then n=1
//     ib 6     qs[32..47] twice, digits n=2 then n=3
//     ib 7     MIXED: qs[32..47] at digit n=4 for its first 16 elements, then the four qh bytes at
//              digits n=0..3 for its last 16. This block is the reason a uniform loop cannot work.
// In every case four CONSECUTIVE output elements share one digit position and come from four
// CONSECUTIVE bytes, which is what makes the 4-lane decode above legal.
//
// Codes are {0,1,2} meaning {-1,0,+1}. As in TQ2_0 and iq2_bn, do not subtract per element:
// sum((q-1)*y) == sum(q*y) - sum(y), and q8_1's ds.y already carries sum(y) scaled. One thread
// covers exactly one whole q8_1 block, so the -1 offset costs exactly one ds.y, no fraction.
__device__ __forceinline__ void vec_dot_tq1_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs, float * result) {

    const block_tq1_0 * bq1 = (const block_tq1_0 *)vbq + kbx;

    const int ib = iqs/4;                 // 0..7 -- which q8_1 block this thread owns
    const int * y = (const int *)bq8_1[ib].qs;   // block_q8_1 is 36 bytes, always 4-byte aligned

    const int p3[5] = {1, 3, 9, 27, 81};

    int sumi = 0;

    if (ib < 5) {
        // Digit position n = ib, across the whole 32-byte group qs[0..31].
        const int p = p3[ib];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            sumi = ggml_cuda_dp4a(tq1_0_decode4(tq1_0_get_int_b2(bq1->qs, k), p), y[k], sumi);
        }
    } else if (ib < 7) {
        // The 16-byte tail qs[32..47], read twice at two consecutive digit positions.
        const int n0 = 2*(ib - 5);        // ib 5 -> digits 0,1 ; ib 6 -> digits 2,3
        const int pa = p3[n0];
        const int pb = p3[n0 + 1];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int v = tq1_0_get_int_b2(bq1->qs + 32, k);
            sumi = ggml_cuda_dp4a(tq1_0_decode4(v, pa), y[k],     sumi);   // elements  0..15
            sumi = ggml_cuda_dp4a(tq1_0_decode4(v, pb), y[k + 4], sumi);   // elements 16..31
        }
    } else {
        // The mixed block. First half: the tail bytes at the last digit position n = 4.
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int v = tq1_0_get_int_b2(bq1->qs + 32, k);
            sumi = ggml_cuda_dp4a(tq1_0_decode4(v, p3[4]), y[k], sumi);
        }
        // Second half: the FOUR qh bytes, re-read at each of four digit positions. For a fixed n
        // the four j values are four consecutive output elements AND four consecutive bytes, so
        // one load feeds one dp4a -- the same shape as everywhere else, just transposed.
        const int vh = tq1_0_get_int_b2(bq1->qh, 0);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            sumi = ggml_cuda_dp4a(tq1_0_decode4(vh, p3[n]), y[4 + n], sumi);
        }
    }

    const float  d  = __half2float(bq1->d);
    const float2 ds = __half22float2(bq8_1[ib].ds);
    *result += d * (sumi * ds.x - ds.y);
}

void mul_mat_vec_tq1_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_TQ1_0, 4, vec_dot_tq1_0_q8_1>(args, stream);
}
