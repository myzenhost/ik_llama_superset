#include "../iqk_mmvq_templates.cuh"
// A LOCAL 2-byte-safe int load, rather than vecdotq.cuh's get_int_b2. Both alternatives were tried
// and the compiler rejected both, which is why this is here:
//
//   1. calling get_int_b2 without an include:
//        mmvq-instance-tq2_0.cu(44): error: identifier "get_int_b2" is undefined
//      iqk_mmvq_templates.cuh pulls in only iqk_mmvq.cuh, iqk_cuda_common.h and mmvq-args.h. It is
//      mmq.cuh that includes vecdotq.cuh, which is why every mmq-instance-*.cu may call these
//      helpers and no mmvq instance can.
//
//   2. adding #include "../vecdotq.cuh" here:
//        vecdotq.cuh(1243): error: function "get_int_from_table_16_shift" has already been defined
//        vecdotq.cuh: function "get_int_from_table_16" has already been defined
//      The iqk headers carry their OWN copies of those functions, so vecdotq.cuh and the iqk chain
//      cannot both be included in one translation unit. That is a pre-existing property of this
//      divergent fork, not something to "fix" in passing on the way to a ternary kernel.
//
// Four lines, no header surgery, and nothing shared is perturbed -- iqk_mmvq_templates.cuh is
// included by 49 instances and widening it for one type's alignment bug would rebuild and re-risk
// all of them. Named distinctly so it cannot collide if the headers are ever reconciled.
static __device__ __forceinline__ int tq2_0_get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x;   // 2-byte alignment is all block_tq2_0 gives us
    return (int)x16[2*i32 + 0] | ((int)x16[2*i32 + 1] << 16);
}

// TQ2_0 x q8_1 mul_mat_vec.
//
// Layout, from dequantize_row_tq2_0: a 256-element super-block stores 64 bytes; byte qs[j+m]
// (j in {0,32}, m in 0..31) holds four 2-bit codes at bit offsets 0,2,4,6, and those four codes
// are elements j*4 + l*32 + m for l = 0..3. Mapping that onto q8_1's 32-element blocks: q8_1 block
// b takes its m-th element from qs[32*(b>>2) + m] at shift 2*(b&3). One thread owns one block.
//
// Codes are {0,1,2} meaning {-1,0,+1}. Rather than decode, use sum((q-1)*y) == sum(q*y) - sum(y):
// dp4a accumulates sum(q*y) on the raw codes and q8_1's ds.y supplies sum(y) already scaled. This
// is the same deferral our vec_dot_iq2_bn_q8_1 uses, and it is why no unpack step is needed.
__device__ __forceinline__ void vec_dot_tq2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs, float * result) {

    const block_tq2_0 * bq2 = (const block_tq2_0 *)vbq + kbx;

    const int ib   = iqs/4;              // 0..7 -- which q8_1 block this thread owns
    const int shift = 2*(ib & 3);        // bit offset of this block's plane within each byte
    const uint8_t * qs = bq2->qs + 32*(ib >> 2);              // 32 bytes -> 8 ints

    int sumi = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        // get_int_b2, NOT a plain (const int *) cast, AND THIS IS THE WHOLE BUG THAT CRASHED THE
        // HOST. sizeof(block_tq2_0) is 66: qs[QK_K/4] = 64 bytes plus a 2-byte ggml_half scale. So
        // consecutive blocks start at byte offsets 0, 66, 132, 198 ... and `(const block_tq2_0 *)
        // vbq + kbx` lands 2-mod-4 for every ODD kbx. Casting that to `const int *` and
        // dereferencing is a 4-byte load at a 2-byte address, which on CUDA is a hard
        // "misaligned address" fault -- not a wrong answer, a dead context.
        //
        // MEASURED 2026-09-01: with the plain cast, -ngl 99 on TriLM_830M-tq2_0 aborted at
        // ggml-cuda.cu:145 (CUDA error) and took the host down with it -- hundreds of nvlddmkm
        // id-13 FECS exceptions in one second, then id-153 TDR, then bugcheck 0x133
        // DPC_WATCHDOG_VIOLATION. Exactly half the blocks were misaligned.
        //
        // WHY IQ2_BN NEVER SHOWED THIS, which is why modelling this kernel on it was not enough:
        // block_iq2_bn is QK_IQ2BN/4 = 16 bytes with NO in-block scale, so every block is 4-byte
        // aligned and a plain int cast is always legal there. The 2-byte ggml_half is what breaks
        // the alignment, and it is the one structural difference between the two types.
        // get_int_b2 loads two uint16_t and recombines, which needs only 2-byte alignment -- the
        // same helper q4_0 (18 bytes), q5_0 and q6_0 already use in this tree for the same reason.
        const int vi = (tq2_0_get_int_b2(qs, k) >> shift) & 0x03030303;
        // The q8_1 side keeps the plain int cast, and that is CORRECT, not an oversight:
        // block_q8_1 is 32 int8 plus a half2 = 36 bytes, a multiple of 4, so every block starts
        // 4-byte aligned. The asymmetry between these two lines is the entire lesson of this file.
        sumi = ggml_cuda_dp4a(vi, ((const int *)bq8_1[ib].qs)[k], sumi);
    }

    const float  d  = __half2float(bq2->d);
    const float2 ds = __half22float2(bq8_1[ib].ds);
    // ds.x = d_y, ds.y = sum(y) already scaled by d_y. One thread covers a whole q8_1 block, so the
    // -1 offset contributes exactly one ds.y, with no fractional correction.
    *result += d * (sumi * ds.x - ds.y);
}

void mul_mat_vec_tq2_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_TQ2_0, 4, vec_dot_tq2_0_q8_1>(args, stream);
}
