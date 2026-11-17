#include "pad.cuh"

// GGML_OP_PAD stores eight int32 op_params: the LOW and HIGH pad of each dimension, in the
// order lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3 (see ggml_pad_ext() in ggml.c). The kernels
// below take the four LOW pads and read the source at index i - lp, so ggml_pad(), which
// writes zeros into every low slot, keeps its exact previous behaviour.

static __global__ void pad_f32(const float * x, float * dst, const int ne0, const int ne00, const int ne01, const int ne02, const int ne03,
        const int lp0, const int lp1, const int lp2, const int ne1, const int ne23) {
    // blockIdx.z: idx of ne2*ne3, aka ne02*ne03
    // blockIdx.y: idx of ne1
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    // upstream #30147: gridDim.y and gridDim.z are capped at 65535, so blocks stride over larger ne1 and ne2*ne3
    const int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    for (int i1 = blockIdx.y; i1 < ne1; i1 += gridDim.y) {
        for (int i23 = blockIdx.z; i23 < ne23; i23 += gridDim.z) {
            const int64_t offset_dst = nidx + (int64_t) i1 * ne0 + (int64_t) i23 * ne0 * ne1;

            // source index, shifted back by the low pad. lp3 is not needed here: this path is only
            // taken with ne3 == ne03 == 1, which forces lp3 == 0.
            const int s0 = nidx - lp0;
            const int s1 = i1   - lp1;
            const int s2 = i23  - lp2;

            if (s0 >= 0 && s0 < ne00 && s1 >= 0 && s1 < ne01 && s2 >= 0 && s2 < ne02*ne03) {
                const int64_t offset_src = s0 + (int64_t) s1 * ne00 + (int64_t) s2 * ne00 * ne01;
                dst[offset_dst] = x[offset_src];
            } else {
                dst[offset_dst] = 0.0f;
            }
        }
    }
}

static void pad_f32_cuda(const float * x, float * dst,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int ne0, const int ne1, const int ne2, const int ne3,
    const int lp0, const int lp1, const int lp2, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, std::min(ne1, 65535), std::min(ne2*ne3, 65535));
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(x, dst, ne0, ne00, ne01, ne02, ne03, lp0, lp1, lp2, ne1, ne2*ne3);
}

template <int dim>
static __global__ void pad_f32_nc(const char * cx, float * dst, int nelem,
        int ne0, int ne1, int ne2, int ne3, int ne00, int ne01, int ne02, int ne03,
        size_t nb00, size_t nb01, size_t nb02, size_t nb03,
        int lp0, int lp1, int lp2, int lp3) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    int ii = i;
    int i3 = ii/(ne0*ne1*ne2); ii -= i3*ne0*ne1*ne2;
    int i2 = ii/(ne0*ne1    ); ii -= i2*ne0*ne1;
    int i1 = ii/(ne0        );
    int i0 = ii - i1*ne0;

    // Shift each index back by that dimension's low pad. The specialised <0..4> instantiations
    // below are only dispatched when every low pad is zero, in which case s* == i* exactly and
    // the unchecked dimensions stay as safe as they were before.
    const int s0 = i0 - lp0;
    const int s1 = i1 - lp1;
    const int s2 = i2 - lp2;
    const int s3 = i3 - lp3;

    // signed offset: s* can be negative inside the low pad, where the value is never
    // dereferenced but the address is still formed.
    const int64_t off = (int64_t)s0*(int64_t)nb00 + (int64_t)s1*(int64_t)nb01
                      + (int64_t)s2*(int64_t)nb02 + (int64_t)s3*(int64_t)nb03;
    const char * src = cx + off;

    if constexpr (dim == 0) {
        dst[i] = s0 >= 0 && s0 < ne00 ? *(const float *)src : 0.0f;
    }
    else if constexpr (dim == 1) {
        dst[i] = s1 >= 0 && s1 < ne01 ? *(const float *)src : 0.0f;
    }
    else if constexpr (dim == 2) {
        dst[i] = s2 >= 0 && s2 < ne02 ? *(const float *)src : 0.0f;
    }
    else if constexpr (dim == 3) {
        dst[i] = s3 >= 0 && s3 < ne03 ? *(const float *)src : 0.0f;
    }
    else if constexpr (dim == 4) {
        dst[i] = *(const float *)src;
    }
    else {
        dst[i] = s0 >= 0 && s0 < ne00 && s1 >= 0 && s1 < ne01 &&
                 s2 >= 0 && s2 < ne02 && s3 >= 0 && s3 < ne03 ? *(const float *)src : 0.0f;
    }
}

static __device__ __forceinline__ int pad_wrap(const int i, const int n) {
    const int r = i % n;
    return r < 0 ? r + n : r;
}

// Circular padding (op_params[8] != 0, ggml_pad_ext(..., circular)): every destination index reads
// the source at (i - low pad) wrapped into [0, ne), in each dimension -- the CPU's
// ggml_wrap_around (ggml-cpu/ops.cpp). The kernels above only zero-fill, and this op never read
// the flag: test-backend-ops PAD circular=1 failed 7 of 7 (ERR ~0.33).
static __global__ void pad_f32_circular(const char * cx, float * dst, const int nelem,
        const int ne0, const int ne1, const int ne2, const int ne00, const int ne01, const int ne02, const int ne03,
        const size_t nb00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int lp0, const int lp1, const int lp2, const int lp3) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    int ii = i;
    const int i3 = ii/(ne0*ne1*ne2); ii -= i3*ne0*ne1*ne2;
    const int i2 = ii/(ne0*ne1);     ii -= i2*ne0*ne1;
    const int i1 = ii/ne0;
    const int i0 = ii - i1*ne0;
    const int s0 = pad_wrap(i0 - lp0, ne00);
    const int s1 = pad_wrap(i1 - lp1, ne01);
    const int s2 = pad_wrap(i2 - lp2, ne02);
    const int s3 = pad_wrap(i3 - lp3, ne03);
    dst[i] = *(const float *)(cx + s0*nb00 + s1*nb01 + s2*nb02 + s3*nb03);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // LOW pads, op_params slots 0/2/4/6 (slots 1/3/5/7 are the high pads, which are implied by
    // the destination size and so never read).
    const int lp0 = dst->op_params[0];
    const int lp1 = dst->op_params[2];
    const int lp2 = dst->op_params[4];
    const int lp3 = dst->op_params[6];
    const bool has_low_pad = lp0 != 0 || lp1 != 0 || lp2 != 0 || lp3 != 0;

    if (dst->op_params[8] != 0) { // circular
        constexpr int kBlockSize = 256;
        const int nelem = ggml_nelements(dst);
        pad_f32_circular<<<(nelem + kBlockSize - 1)/kBlockSize, kBlockSize, 0, stream>>>(
                (const char *)src0->data, dst_d, nelem, dst->ne[0], dst->ne[1], dst->ne[2],
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], lp0, lp1, lp2, lp3);
        return;
    }

    // Fast path for a contiguous source. It folds ne2 and ne3 into one grid dimension and so
    // only works for 3D tensors -- it used to ASSERT that, which meant a contiguous 4D source
    // with ne3 > 1 aborted the process even though supports_op had just claimed the op. A
    // contiguous tensor is a special case of a strided one, so send those to the general
    // kernel below instead of dying. Same for a non-zero lp3, which this path cannot express.
    if (ggml_is_contiguous(src0) && src0->ne[3] == 1 && dst->ne[3] == 1 && lp3 == 0) {
        pad_f32_cuda(src0_d, dst_d,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                lp0, lp1, lp2, stream);
        return;
    }

    int npad = 0; int pad_dim = -1;
    for (int i = 0; i < 4; ++i) {
        if (dst->ne[i] > src0->ne[i]) {
            ++npad; pad_dim = i;
        }
    }

    constexpr int kBlockSize = 256;
    int nelem = ggml_nelements(dst);
    int nblock = (nelem + kBlockSize - 1)/kBlockSize;

    // A non-zero low pad means the padded dimension has valid source on BOTH sides of the
    // window, so the single-axis fast paths (which only bounds-check one dimension) no longer
    // hold. Fall back to the fully checked instantiation; it is still a single GPU kernel, which
    // is the whole point -- previously this case was declined and ran on the CPU mid-graph.
    if (has_low_pad) {
        pad_f32_nc<-1><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], lp0, lp1, lp2, lp3);
        return;
    }

    if (npad == 0) {
        pad_f32_nc<4><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], 0, 0, 0, 0);
    } else if (npad == 1) {
        if (pad_dim == 0) {
            pad_f32_nc<0><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], 0, 0, 0, 0);
        } else if (pad_dim == 1) {
            pad_f32_nc<1><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], 0, 0, 0, 0);
        } else if (pad_dim == 2) {
            pad_f32_nc<2><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], 0, 0, 0, 0);
        } else if (pad_dim == 3) {
            pad_f32_nc<3><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], 0, 0, 0, 0);
        } else {
            GGML_ABORT("Fatal error");
        }
    } else {
        pad_f32_nc<-1><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], 0, 0, 0, 0);
    }
}
