//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "getrows.cuh"
#include "dequantize.cuh"
#include "dequantize-kq.cuh"

// Grid layout follows upstream getrows.cu: x = gathered row i10 (x allows 2^31-1 blocks), y = column
// blocks and z = (i11,i12), both CLAMPED to 65535 and strided. The old layout put i10 in grid.y, so
// gathering more than 65535 rows failed to launch ("CUDA error: invalid argument", measured with
// test-backend-ops GET_ROWS n=256 m=80000 r=70000 on CUDA0).

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, /*int64_t ne02, int64_t ne03,*/
            /*int64_t ne10,*/ int64_t ne11, int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12/*, size_t s13*/) {

    const int64_t i10 = blockIdx.x;
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        const int64_t i11 = z / ne12;
        const int64_t i12 = z % ne12;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t i00 = 2*((int64_t)blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += 2*(int64_t)gridDim.y*blockDim.x) {
            const int ib = i00/qk; // block index
            const int iqs = (i00%qk)/qr; // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            dfloat2 v;
            if (i01 >= 0 && i01 < ne01) {
                dequantize_kernel(src0_row, ib, iqs, v);
            } else {
                v.x = v.y = 0;
            }

            dst_row[iybs + iqs + 0]        = v.x;
            dst_row[iybs + iqs + y_offset] = v.y;
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, /*int64_t ne02, int64_t ne03,*/
            /*int64_t ne10,*/ int64_t ne11, int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12/*, size_t s13*/) {

    const int64_t i10 = blockIdx.x;
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        const int64_t i11 = z / ne12;
        const int64_t i12 = z % ne12;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i00 = (int64_t)blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += (int64_t)gridDim.y*blockDim.x) {
            dst_row[i00] = i01 >= 0 && i01 < ne01 ? dst_t(src0_row[i00]) : dst_t(0);
        }
    }
}

// GET_ROWS for the k-quants, i-quants and mxfp4: one thread block dequantizes one (dst row, QK_K
// super-block) pair through the shared dequantizers in dequantize-kq.cuh. Algorithm after llama.cpp
// c5a4a0bb83 (MIT, #25962) k_get_rows_kq; grid layout and the out-of-range-index zero fill follow this
// file's k_get_rows (x = gathered row, y = super-blocks, z = (i11,i12), y and z clamped and strided).
template<typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void k_get_rows_kq(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01,
            int64_t ne11, int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12) {

    const int64_t nsb = ne00/QK_K; // super-blocks per row
    const int64_t i10 = blockIdx.x;
    for (int64_t z = blockIdx.z; z < ne11*ne12; z += gridDim.z) {
        const int64_t i11 = z / ne12;
        const int64_t i12 = z % ne12;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            if (i01 >= 0 && i01 < ne01) {
                dequantize_kq(src0_row, ib, dst_row + ib*QK_K, threadIdx.x);
            } else {
                for (int j = threadIdx.x; j < QK_K; j += blockDim.x) {
                    dst_row[ib*QK_K + j] = ggml_cuda_cast<dst_t>(0.0f);
                }
            }
        }
    }
}

template<int block_dim, dequantize_kq_t<float> dequantize_kq>
static void get_rows_cuda_kq(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                             const void * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nsb = ne00/QK_K;

    const dim3 block_dims(block_dim, 1, 1);
    const dim3 block_nums(ne10, MIN(nsb, (int64_t) UINT16_MAX), MIN(ne11*ne12, (int64_t) UINT16_MAX));

    // strides in elements
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);

    k_get_rows_kq<float, dequantize_kq><<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, ne01,
            ne11, ne12,
            s1, s2, s3,
            nb01, nb02, nb03,
            s10, s11, s12);

    GGML_UNUSED(dst);
}

template<int qk, int qr, dequantize_kernel_t dq>
static void get_rows_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                            const void * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int64_t block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, (int64_t) UINT16_MAX), MIN(ne11*ne12, (int64_t) UINT16_MAX));

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, ne01, /*ne02, ne03,*/
            /*ne10,*/ ne11, ne12, /*ne13,*/
            /* s0,*/ s1, s2, s3,
            /* nb00,*/ nb01, nb02, nb03,
            s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

template<typename src0_t>
static void get_rows_cuda_float(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                const src0_t * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int64_t block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, (int64_t) UINT16_MAX), MIN(ne11*ne12, (int64_t) UINT16_MAX));

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    k_get_rows_float<<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, ne01, /*ne02, ne03,*/
            /*ne10,*/ ne11, ne12, /*ne13,*/
            /* s0,*/ s1, s2, s3,
            /* nb00,*/ nb01, nb02, nb03,
            s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

template <typename data_t>
static __global__ void k_get_rows_dim0(int n,
        size_t nb01, size_t nb02, size_t nb03, size_t nb11, size_t nb12, size_t nb13, size_t nb1, size_t nb2, size_t nb3,
        const data_t * __restrict__ src, const int * __restrict__ idx, data_t * __restrict__ dst) {
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;
    src += nb01*i1 + nb02*i2 + nb03*i3;
    idx += nb11*i1 + nb12*i2 + nb13*i3;
    dst +=  nb1*i1 +  nb2*i2 +  nb3*i3;

    for (int j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[idx[j]];
}

template <typename data_t>
static __global__ void k_get_rows_dim1(int n,
        size_t nb01, size_t nb02, size_t nb03, size_t nb11, size_t nb12, size_t nb1, size_t nb2, size_t nb3,
        const data_t * __restrict__ src, const int * __restrict__ idx, data_t * __restrict__ dst) {
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;
    idx += nb11*i2 + nb12*i3;
    src += nb01*idx[i1] + nb02*i2 + nb03*i3;
    dst +=  nb1*i1 +  nb2*i2 +  nb3*i3;

    for (int j = threadIdx.x; j < n; j += blockDim.x) dst[j] = src[j];
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    if (dst->type != GGML_TYPE_F32 || dst->op_params[0] == 1) {
        constexpr int k_block_size = 512;
        GGML_ASSERT(src0->type == dst->type);
        if (dst->op_params[0] == 1) {
            //printf("%s(%s, %s) - dim0\n", __func__, src0->name, ggml_type_name(src0->type));
            GGML_ASSERT(src0->ne[1] == src1->ne[1] && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]);
            GGML_ASSERT(src0->ne[0] >= src1->ne[0]);
            GGML_ASSERT(dst->ne[0] == src1->ne[0]);
            auto type_size = ggml_type_size(dst->type);
            GGML_ASSERT(type_size == 2 || type_size == 4);
            dim3 grid(dst->ne[1], dst->ne[2], dst->ne[3]);
            if (type_size == 2) {
                k_get_rows_dim0<<<grid, k_block_size, 0, ctx.stream()>>>(dst->ne[0],
                        src0->nb[1]/type_size, src0->nb[2]/type_size, src0->nb[3]/type_size,
                        src1->nb[1]/type_size, src1->nb[2]/type_size, src1->nb[3]/type_size,
                         dst->nb[1]/type_size,  dst->nb[2]/type_size, dst->nb[3]/type_size,
                        (const uint16_t *)src0->data, (const int *)src1->data, (uint16_t *)dst->data);
            } else {
                k_get_rows_dim0<<<grid, k_block_size, 0, ctx.stream()>>>(dst->ne[0],
                        src0->nb[1]/type_size, src0->nb[2]/type_size, src0->nb[3]/type_size,
                        src1->nb[1]/type_size, src1->nb[2]/type_size, src1->nb[3]/type_size,
                         dst->nb[1]/type_size,  dst->nb[2]/type_size,  dst->nb[3]/type_size,
                        (const uint32_t *)src0->data, (const int *)src1->data, (uint32_t *)dst->data);
            }
        } else {
            //printf("%s(%s, %s) - dim1\n", __func__, src0->name, ggml_type_name(src0->type));
            GGML_ASSERT(src0->ne[2] == src1->ne[1]);
            GGML_ASSERT(src0->ne[3] == src1->ne[2]);
            GGML_ASSERT(src0->ne[3] == 1);
            GGML_ASSERT(dst->ne[0] == src0->ne[0] && dst->ne[1] == src1->ne[0] && dst->ne[2] == src1->ne[1] && dst->ne[2]);

            auto row_size = ggml_row_size(dst->type, dst->ne[0]);
            GGML_ASSERT(row_size % 2 == 0);
            auto type_size = row_size % 4 == 0 ? 4 : 2;
            dim3 grid(dst->ne[1], dst->ne[2], dst->ne[3]);
            if (type_size == 2) {
                k_get_rows_dim1<<<grid, k_block_size, 0, ctx.stream()>>>(row_size/type_size,
                        src0->nb[1]/type_size, src0->nb[2]/type_size, src0->nb[3]/type_size,
                        src1->nb[1]/type_size, src1->nb[2]/type_size,
                         dst->nb[1]/type_size,  dst->nb[2]/type_size, dst->nb[3]/type_size,
                        (const uint16_t *)src0->data, (const int *)src1->data, (uint16_t *)dst->data);
            } else {
                k_get_rows_dim1<<<grid, k_block_size, 0, ctx.stream()>>>(row_size/type_size,
                        src0->nb[1]/type_size, src0->nb[2]/type_size, src0->nb[3]/type_size,
                        src1->nb[1]/type_size, src1->nb[2]/type_size,
                         dst->nb[1]/type_size,  dst->nb[2]/type_size,  dst->nb[3]/type_size,
                        (const uint32_t *)src0->data, (const int *)src1->data, (uint32_t *)dst->data);
            }
        }
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32));

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) src1_d;

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float(src0, src1, dst, (const half *)src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            get_rows_cuda_float(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        // Bonsai ternary quants. FOUND BY a peer-coverage check, not by a crash: GGML_TYPE_Q4_0
        // appears here and ours did not, and the consequence is silent -- GGML_SCHED_DEBUG showed
        // GET_ROWS on token_embd.weight (322 MB, type Q2_0_G128) assigned to CPU while the rest of
        // the graph ran on the device. No assert, no error, just the embedding lookup off-GPU.
        // The dequantize functions and QK/QR traits already exist for the convert path, so this is
        // a dispatch gap rather than a missing kernel. qr == 1 for both, per the sequential layout.
        case GGML_TYPE_Q1_0_G128:
            get_rows_cuda<QK1_0_G128, 1, dequantize_q1_0_g128>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q2_0_G128:
            get_rows_cuda<QK2_0_G128, 1, dequantize_q2_0_g128>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda<QK1_0, 1, dequantize_q1_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q2_0:
            get_rows_cuda<QK2_0, 1, dequantize_q2_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda<QK4_0, QR4_0, dequantize_q4_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda<QK4_1, QR4_1, dequantize_q4_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda<QK5_0, QR5_0, dequantize_q5_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda<QK5_1, QR5_1, dequantize_q5_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda<QK8_0, QR8_0, dequantize_q8_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_NVFP4_KS:
            get_rows_cuda<QK_NVFP4/4, QR_NVFP4, dequantize_nvfp4_ks>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        // k-quants, i-quants and mxfp4 (mirrors llama.cpp c5a4a0bb83): 32 threads for q4_K and for the
        // 32-value sub-block types, 64 threads for the other k-quants (the convert.cu thread layouts).
        case GGML_TYPE_Q2_K:
            get_rows_cuda_kq<64, dequantize_q2_K<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_cuda_kq<64, dequantize_q3_K<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_kq<32, dequantize_q4_K<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_kq<64, dequantize_q5_K<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_cuda_kq<64, dequantize_q6_K<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_cuda_kq<32, dequantize_iq2_xxs<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_cuda_kq<32, dequantize_iq2_xs<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_cuda_kq<32, dequantize_iq2_s<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_cuda_kq<32, dequantize_iq3_xxs<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_cuda_kq<32, dequantize_iq3_s<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_cuda_kq<32, dequantize_iq1_s<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_cuda_kq<32, dequantize_iq1_m<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_cuda_kq<32, dequantize_iq4_nl<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_cuda_kq<32, dequantize_iq4_xs<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_MXFP4:
            get_rows_cuda_kq<32, dequantize_mxfp4<float>>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
            break;
    }
}

// GET_ROWS_BACK. Ported from ggml-org/llama.cpp @95887577ab5fead779581a7030a83c7752ff3234 (MIT),
// ggml/src/ggml-cuda/getrows.cu: k_get_rows_back_float (lines 132-158) and ggml_cuda_op_get_rows_back
// (lines 461-490). getrows.cuh already declared the entry point and CUDA_GET_ROWS_BACK_BLOCK_SIZE; the
// definition was missing. One thread per column; each destination row sums the gradient rows whose
// index equals it. grid.y is clamped to 65535 and the kernel strides over the destination rows.
template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);
    GGML_ASSERT(ne0 == ne00);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, MIN(ne1, (int64_t) UINT16_MAX), 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10, ne1);
}
