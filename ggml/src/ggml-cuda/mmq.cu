//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "mmq.cuh"

void ggml_cuda_op_mul_mat_q(ggml_backend_cuda_context & ctx, enum ggml_type type, const mmq_args & args) {
    auto stream = ctx.stream();
    switch (type) {
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_PQ2_0:
            mul_mat_q_case<GGML_TYPE_PQ2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_0:
            mul_mat_q_case<GGML_TYPE_Q6_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q1_0_G128:
            mul_mat_q_case<GGML_TYPE_Q1_0_G128>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0_G128:
            mul_mat_q_case<GGML_TYPE_Q2_0_G128>(ctx, args, stream);
            break;
        case GGML_TYPE_TQ2_0:
            mul_mat_q_case<GGML_TYPE_TQ2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_q_case<GGML_TYPE_IQ1_M>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S_R4:
            mul_mat_q_case<GGML_TYPE_IQ1_S_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4_KS:
            mul_mat_q_case<GGML_TYPE_NVFP4_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            mul_mat_q_case<GGML_TYPE_IQ2_KL>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_q_case<GGML_TYPE_IQ3_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_KS_R16:
            mul_mat_q_case<GGML_TYPE_IQ3_KS_R16>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            mul_mat_q_case<GGML_TYPE_IQ4_KSS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            mul_mat_q_case<GGML_TYPE_IQ4_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            mul_mat_q_case<GGML_TYPE_IQ4_KS_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KS_R16:
            mul_mat_q_case<GGML_TYPE_IQ4_KS_R16>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_KS_R16:
            mul_mat_q_case<GGML_TYPE_IQ5_KS_R16>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            mul_mat_q_case<GGML_TYPE_IQ4_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_KT:
            mul_mat_q_case<GGML_TYPE_IQ1_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            mul_mat_q_case<GGML_TYPE_IQ2_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            mul_mat_q_case<GGML_TYPE_IQ3_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            mul_mat_q_case<GGML_TYPE_IQ5_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            mul_mat_q_case<GGML_TYPE_IQ5_KS_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            mul_mat_q_case<GGML_TYPE_IQ2_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_K:
            mul_mat_q_case<GGML_TYPE_IQ2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_q_case<GGML_TYPE_IQ3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_K:
            mul_mat_q_case<GGML_TYPE_IQ4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_K:
            mul_mat_q_case<GGML_TYPE_IQ5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ6_K:
            mul_mat_q_case<GGML_TYPE_IQ6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_K_R4:
            mul_mat_q_case<GGML_TYPE_IQ2_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_K_R4:
            mul_mat_q_case<GGML_TYPE_IQ3_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_K_R4:
            mul_mat_q_case<GGML_TYPE_IQ4_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_K_R4:
            mul_mat_q_case<GGML_TYPE_IQ5_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_TQ2_T:
            mul_mat_q_case<GGML_TYPE_TQ2_T>(ctx, args, stream);
            break;
        case GGML_TYPE_TQK6:
            mul_mat_q_case<GGML_TYPE_TQK6>(ctx, args, stream);
            break;
        case GGML_TYPE_TQK7:
            mul_mat_q_case<GGML_TYPE_TQK7>(ctx, args, stream);
            break;
        case GGML_TYPE_TQP5:
            mul_mat_q_case<GGML_TYPE_TQP5>(ctx, args, stream);
            break;
        case GGML_TYPE_TQP6:
            mul_mat_q_case<GGML_TYPE_TQP6>(ctx, args, stream);
            break;
        case GGML_TYPE_TQP7:
            mul_mat_q_case<GGML_TYPE_TQP7>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, [[maybe_unused]] cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t nb01 = ggml_row_size(src0->type, ne00);

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const mmq_args args = {src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, nb01, src1_padded_row_size, src1_ncols, ne11, nrows_dst};

    ggml_cuda_op_mul_mat_q(ctx, src0->type, args);

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}

static bool ggml_cuda_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11);

// Always compiled in; one line per (type, path) pair, first occurrence only. This exists because
// an A/B is only as good as its proof that the two arms differ.
bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11) {
    const bool r = ggml_cuda_should_use_mmq_impl(type, cc, ne11);
    if (getenv("GGML_CUDA_TRACE_MMQ") != nullptr) {
        static bool seen[GGML_TYPE_COUNT][2] = {};
        const int ti = (int) type;
        if (ti >= 0 && ti < GGML_TYPE_COUNT && !seen[ti][r ? 1 : 0]) {
            seen[ti][r ? 1 : 0] = true;
            fprintf(stderr, "[cuda-mmq] %-14s mat-MAT path = %s\n",
                    ggml_type_name(type), r ? "MMQ (q8_1 integer dot)" : "dequant->cuBLAS");
            fflush(stderr);
        }
    }
    return r;
}

static bool ggml_cuda_should_use_mmq_impl(enum ggml_type type, int cc, int64_t ne11) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q1_0_G128:
        case GGML_TYPE_Q2_0_G128:
        // TQ2_0 gains MMQ 2026-09-02. This predicate is the gate: without the case,
        // load_tiles_tq2_0 is dead code and batched mul_mats keep taking the dequant-to-cuBLAS
        // path. Added only AFTER the loader and its DECL_MMQ_CASE instance compile.
        // RESEARCH-DONE, in-tree and re-read rather than recalled:
        //   * ggml-cuda.cu, the GGML_TYPE_TQ2_0 supports_op comment block, records the measured
        //     cost of listing a type whose consuming path is not ready -- "MEASURED with the
        //     kernel present and this case absent: tg64 28.93 -> 29.03, pp128 395.70 -> 409.43",
        //     i.e. the kernel never executed at all.
        //   * commit 7b1784a7 ("cuda: fix the TQ2_0 mmvq misaligned load, and enable the routing
        //     that makes it run") is the repair for the opposite ordering error on this very
        //     type: routing declared against a kernel that then faulted at ggml-cuda.cu:145 and
        //     took the host down with bugcheck 0x133.
        //   * llama.cpp PR #11183 (compilade) is the upstream source for the technique used by
        //     load_tiles_tq2_0: it removed its own TQ2_0 mmq dp4a because re-using the Q8_0 one
        //     "allows avoiding to repeatedly unpack the 2-bit values to 8-bit and instead only
        //     do it once per tile", and measured TQ2_0 tg128 262.27 t/s vs Q4_0 215.75 on a 3090.
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_NVFP4_KS:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_KS_R16:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ4_KS_R16:
        case GGML_TYPE_IQ5_KS_R16:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
        case GGML_TYPE_TQ2_T:
        case GGML_TYPE_TQK6:
        case GGML_TYPE_TQK7:
        case GGML_TYPE_TQP5:
        case GGML_TYPE_TQP6:
        case GGML_TYPE_TQP7:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (int8_mma_available(cc)) {
        return true;
    }
    if (type == GGML_TYPE_IQ1_S_R4) {
        return false;
    }

    if (cc < MIN_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (cc < GGML_CUDA_CC_OFFSET_AMD) {
        // On Volta, large-batch quantized matmuls otherwise fall back through
        // fp16 cuBLAS temporaries. Keep using MMQ for pre-Turing NVIDIA.
        return cc < GGML_CUDA_CC_TURING || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    return cc < GGML_CUDA_CC_RDNA3 || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
