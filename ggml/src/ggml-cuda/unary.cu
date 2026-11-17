//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "unary.cuh"
#include "convert.cuh" // ggml_cuda_cast (convert.cuh:67), used by the merged upstream unary templates; upstream unary.cuh pulls it in

// ik's element-wise kernels below are F32-only; an F16 tensor is routed to upstream's
// ggml_cuda_op_unary<op> template (defined further down), which handles F32 and F16.
static void ggml_cuda_op_unary_f16_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

static __global__ void gelu_f32(const float * x, float * dst, const int k) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    float xi = x[i];
    dst[i] = 0.5f*xi*(1.0f + tanhf(SQRT_2_OVER_PI*xi*(1.0f + GELU_COEF_A*xi*xi)));
}

static __global__ void gelu_quick_f32(const float * x, float * dst, int k) {
    const float GELU_QUICK_COEF = -1.702f;
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = x[i] * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x[i])));
}

static __global__ void silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}

static __global__ void scaled_silu_f32(const float * x, float * dst, const int k, float scale) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    float sx = scale*x[i];
    dst[i] = sx / (1.0f + expf(-sx));
}

#if 0
static __global__ void swiglu_f32(const float * x, float * dst, const int k, const int ne0, const int64_t nb1) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    const int row = i/ne0;
    const int idx = i%ne0;
    const int j   = row*nb1 + idx;
    dst[i] = x[j] * x[j + ne0] / (1.0f + expf(-x[j]));
}
#endif

static __global__ void fused_mul_silu_f32(const float * x, const float * y, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * y[i] / (1.0f + expf(-x[i]));
}

static __global__ void fused_mul_silu_f32(const float * x, const float * y, float * dst, const int k, float limit) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    float g = x[i] / (1.0f + expf(-x[i]));
    g = min(g, limit);
    dst[i] = g * max(-limit, min(limit, y[i]));
}

static __global__ void fused_mul_silu_f32(int ne0, const float * x, const float * y, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    int row = i / ne0;
    dst[i] = x[row] * y[i] / (1.0f + expf(-x[row]));
}

static __global__ void fused_mul_sigmoid_f32(int ne0, const float * x, const float * y, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    int row = i / ne0;
    dst[i] = y[i] / (1.0f + expf(-x[row]));
}

static __global__ void fused_mul_sigmoid_f32(const float * x, const float * y, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = y[i] / (1.0f + expf(-x[i]));
}

static __global__ void fused_mul_silu_f32(int ne0, const float * x, const float * y, float * dst, const int k, float limit) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    int row = i / ne0;
    float g = x[row] / (1.0f + expf(-x[row]));
    g = min(g, limit);
    dst[i] = g * max(-limit, min(limit, y[i]));
}

static __global__ void fused_mul_silu_f32(const float * x, float * dst, const int k, const int ne0) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    int row = i / ne0;
    int j   = i % ne0;
    auto x_row = x + 2*row*ne0;
    // Note: gate is at the beginning of each row, up is offset by ne0
    dst[i] = x_row[j] * x_row[j + ne0] / (1.0f + expf(-x_row[j]));
}

static __global__ void fused_mul_silu_f32(const float * x, float * dst, const int k, const int ne0, float limit) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    int row = i / ne0;
    int j   = i % ne0;
    auto x_row = x + 2*row*ne0;
    // Note: gate is at the beginning of each row, up is offset by ne0
    float g = x_row[j] / (1.0f + expf(-x_row[j]));
    g = min(g, limit);
    dst[i] = max(-limit, min(limit, x_row[j + ne0])) * g;
}

static __global__ void fused_mul_relu_f32(const float * x, const float * y, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0) * y[i];
}

static __global__ void fused_mul_relu_f32(const float * x, float * dst, const int k, const int ne0) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    int row = i / ne0;
    int j   = i % ne0;
    auto x_row = x + 2*row*ne0;
    dst[i] = fmaxf(x_row[j], 0) * x_row[j + ne0];
}

static __global__ void fused_mul_gelu_f32(const float * x, const float * y, float * dst, const int k) {
    constexpr float GELU_COEF_A    = 0.044715f;
    constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    float xi = x[i];
    dst[i] = 0.5f*xi*y[i]*(1.0f + tanhf(SQRT_2_OVER_PI*xi*(1.0f + GELU_COEF_A*xi*xi)));
}

static __global__ void fused_mul_gelu_f32(const float * x, float * dst, const int k, const int ne0) {
    constexpr float GELU_COEF_A    = 0.044715f;
    constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    int row = i / ne0;
    int j   = i % ne0;
    auto x_row = x + 2*row*ne0;
    float xi = x_row[j];
    dst[i] = 0.5f*xi*x_row[j+ne0]*(1.0f + tanhf(SQRT_2_OVER_PI*xi*(1.0f + GELU_COEF_A*xi*xi)));
}

static __global__ void tanh_f32(const float * x, float * dst, int k) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = tanhf(x[i]);
}

static __global__ void relu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0);
}

static __global__ void sigmoid_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = 1.0f / (1.0f + expf(-x[i]));
}

static __global__ void scaled_sigmoid_f32(const float * x, float * dst, const int k, float scale) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = 1.0f / (1.0f + expf(-scale*x[i]));
}

static __global__ void biased_sigmoid_f32(const float * x, const float * bias, float * dst, float * dst_biased, const int k, const int ncols) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = 1.0f / (1.0f + expf(-x[i]));
    dst_biased[i] = dst[i] + bias[i % ncols];
}

static __global__ void hardsigmoid_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
}

static __global__ void hardswish_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
}

static __global__ void leaky_relu_f32(const float * x, float * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = fmaxf(x[i], 0) + fminf(x[i], 0.0f) * negative_slope;
}

static __global__ void sqr_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = x[i] * x[i];
}

static __global__ void sqrt_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = sqrtf(x[i]);
}

static void gelu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void gelu_quick_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    gelu_quick_f32<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void silu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

#if 0
static void swiglu_f32_cuda(const float * x, float * dst, const int k, const int64_t ne0, const int64_t nb1, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    swiglu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k, ne0, nb1);
}
#endif

static void fused_mul_silu_f32_cuda(const float * x, const float * y, float * dst, const int k, float limit, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    if (limit < 1e-6f) {
        fused_mul_silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
    } else {
        fused_mul_silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, y, dst, k, limit);
    }
}

static void fused_mul_silu_f32_cuda(int ne0, const float * x, const float * y, float * dst, const int k, float limit, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    if (limit < 1e-6f) {
        fused_mul_silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(ne0, x, y, dst, k);
    } else {
        fused_mul_silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(ne0, x, y, dst, k, limit);
    }
}

static void fused_mul_sigmoid_f32_cuda(int ne0, const float * x, const float * y, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    fused_mul_sigmoid_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(ne0, x, y, dst, k);
}

static void fused_mul_relu_f32_cuda(const float * x, const float * y, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    fused_mul_relu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

static void fused_mul_sigmoid_f32_cuda(const float * x, const float * y, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    fused_mul_sigmoid_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

static void fused_mul_gelu_f32_cuda(const float * x, const float * y, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    fused_mul_gelu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, y, dst, k);
}

static void fused_mul_silu_f32_cuda(const float * x, float * dst, const int k, const int ne0, float limit, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    if (limit < 1e-6f) {
        fused_mul_silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k, ne0);
    } else {
        fused_mul_silu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k, ne0, limit);
    }
}

static void fused_mul_relu_f32_cuda(const float * x, float * dst, const int k, const int ne0, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    fused_mul_relu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k, ne0);
}

static void fused_mul_gelu_f32_cuda(const float * x, float * dst, const int k, const int ne0, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    fused_mul_gelu_f32<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(x, dst, k, ne0);
}

static void tanh_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_TANH_BLOCK_SIZE - 1) / CUDA_TANH_BLOCK_SIZE;
    tanh_f32<<<num_blocks, CUDA_TANH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void relu_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void sigmoid_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    sigmoid_f32<<<num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void scaled_sigmoid_f32_cuda(const float * x, float * dst, const int k, float scale, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    scaled_sigmoid_f32<<<num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k, scale);
}

static void scaled_silu_f32_cuda(const float * x, float * dst, const int k, float scale, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    scaled_silu_f32<<<num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k, scale);
}

static void biased_sigmoid_f32_cuda(const float * x, const float * bias, float * dst, float * dst_biased, const int k, const int ncols, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    biased_sigmoid_f32<<<num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream>>>(x, bias, dst, dst_biased, k, ncols);
}

static void hardsigmoid_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSIGMOID_BLOCK_SIZE - 1) / CUDA_HARDSIGMOID_BLOCK_SIZE;
    hardsigmoid_f32<<<num_blocks, CUDA_HARDSIGMOID_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void hardswish_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_HARDSWISH_BLOCK_SIZE - 1) / CUDA_HARDSWISH_BLOCK_SIZE;
    hardswish_f32<<<num_blocks, CUDA_HARDSWISH_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void leaky_relu_f32_cuda(const float * x, float * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_f32<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

static void sqr_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQR_BLOCK_SIZE - 1) / CUDA_SQR_BLOCK_SIZE;
    sqr_f32<<<num_blocks, CUDA_SQR_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

static void sqrt_f32_cuda(const float * x, float * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SQRT_BLOCK_SIZE - 1) / CUDA_SQRT_BLOCK_SIZE;
    sqrt_f32<<<num_blocks, CUDA_SQRT_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    silu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

#if 0
void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[0] == src0->ne[0]/2);

    swiglu_f32_cuda(src0_d, dst_d, ggml_nelements(dst), dst->ne[0], src0->nb[1]/sizeof(float), stream);
}
#endif

void ggml_fused_mul_unary(ggml_backend_cuda_context & ctx, ggml_unary_op op,
        int64_t nelements, const float * src0_d, const float * src1_d, float * dst_d, float limit) {

    cudaStream_t stream = ctx.stream();

    switch (op) {
        case GGML_UNARY_OP_SILU: fused_mul_silu_f32_cuda(src0_d, src1_d, dst_d, nelements, limit, stream); break;
        case GGML_UNARY_OP_RELU: fused_mul_relu_f32_cuda(src0_d, src1_d, dst_d, nelements, stream); break;
        case GGML_UNARY_OP_GELU: fused_mul_gelu_f32_cuda(src0_d, src1_d, dst_d, nelements, stream); break;
        case GGML_UNARY_OP_SIGMOID: fused_mul_sigmoid_f32_cuda(src0_d, src1_d, dst_d, nelements, stream); break;
        default: GGML_ASSERT(false);
    }
}

void ggml_fused_mul_unary(ggml_backend_cuda_context & ctx, ggml_unary_op op,
        int64_t nelements, int64_t ne0, const float * src0_d, float * dst_d, float limit) {

    cudaStream_t stream = ctx.stream();

    switch (op) {
        case GGML_UNARY_OP_SILU: fused_mul_silu_f32_cuda(src0_d, dst_d, nelements, ne0, limit, stream); break;
        case GGML_UNARY_OP_RELU: fused_mul_relu_f32_cuda(src0_d, dst_d, nelements, ne0, stream); break;
        case GGML_UNARY_OP_GELU: fused_mul_gelu_f32_cuda(src0_d, dst_d, nelements, ne0, stream); break;
        default: GGML_ASSERT(false);
    }
}

void ggml_cuda_op_fused_mul_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    ggml_unary_op op = (ggml_unary_op)dst->op_params[0];
    float limit = *(const float *)(dst->op_params + 1);
    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src1) {
        GGML_ASSERT(ggml_are_same_shape(src1, dst));
        if (!ggml_are_same_shape(src0, src1)) {
            GGML_ASSERT(src0->ne[0] == 1 && src0->ne[1] == src1->ne[1] && src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]);
            if (op == GGML_UNARY_OP_SILU) {
                fused_mul_silu_f32_cuda(src1->ne[0], (const float *)src0->data, (const float *)src1->data, (float *)dst->data,
                        ggml_nelements(dst), limit, ctx.stream());
            }
            else if (op == GGML_UNARY_OP_SIGMOID) {
                fused_mul_sigmoid_f32_cuda(src1->ne[0], (const float *)src0->data, (const float *)src1->data, (float *)dst->data,
                        ggml_nelements(dst), ctx.stream());
            } else {
                GGML_ABORT("Fatal error");
            }
            return;
        }
        GGML_ASSERT(ggml_are_same_shape(src0, src1));
        ggml_fused_mul_unary(ctx, op, ggml_nelements(dst), (const float *)src0->data, (const float *)src1->data, (float *)dst->data, limit);
    } else {
        GGML_ASSERT(src0->ne[0] == 2*dst->ne[0] && src0->ne[1] == dst->ne[1] && src0->ne[2] == dst->ne[2] && src0->ne[3] == dst->ne[3]);
        ggml_fused_mul_unary(ctx, op, ggml_nelements(dst), dst->ne[0], (const float *)src0->data, (float *)dst->data, limit);
    }
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    gelu_quick_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    tanh_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    relu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sigmoid_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

bool ggml_cuda_op_scale_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto src = dst->src[0];
    if (src->op != GGML_OP_SCALE) return false;
    auto src0 = src->src[0];
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    auto op = (ggml_unary_op)dst->op_params[0];
    if (op != GGML_UNARY_OP_SILU && op != GGML_UNARY_OP_SIGMOID) return false;
    float scale;
    memcpy(&scale, src->op_params, sizeof(scale));
    if (op == GGML_UNARY_OP_SILU) {
        scaled_silu_f32_cuda((const float *)src0->data, (float *)dst->data, ggml_nelements(src0), scale, ctx.stream());
    } else {
        scaled_sigmoid_f32_cuda((const float *)src0->data, (float *)dst->data, ggml_nelements(src0), scale, ctx.stream());
    }
    return true;
}

void ggml_cuda_op_biased_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_ADD);
    GGML_ASSERT(dst->src[0]->op == GGML_OP_UNARY);
    const ggml_tensor * src0 = dst->src[0]->src[0];
    const ggml_tensor * bias = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->type == GGML_TYPE_F32);
    GGML_ASSERT(bias->ne[0] == src0->ne[0]);
    GGML_ASSERT(ggml_nrows(bias) == 1);

    biased_sigmoid_f32_cuda(src0_d, (const float *)bias->data, (float *)dst->src[0]->data, dst_d, ggml_nelements(src0), src0->ne[0], stream);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardsigmoid_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    hardswish_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    leaky_relu_f32_cuda(src0_d, dst_d, ggml_nelements(src0), negative_slope, stream);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqr_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    if (src0->type == GGML_TYPE_F16) {
        ggml_cuda_op_unary_f16_route(ctx, dst);
        return;
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    sqrt_f32_cuda(src0_d, dst_d, ggml_nelements(src0), stream);
}

template <typename T>
static __global__ void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float alpha, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];
    xi = fminf(xi, limit);
    gi = fmaxf(fminf(gi, limit), -limit);

    float out_glu = xi / (1.0f + expf(-xi * alpha));
    out_glu = out_glu * (1.0f + gi);

    dst[i] = out_glu;
}

template <typename T>
static void swiglu_oai_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    swiglu_oai_kernel<<<num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1, alpha, limit);
}

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = false; //ggml_get_op_params_i32(dst, 1);
    const float * op_params = (const float *)dst->op_params;
    const float alpha = op_params[2];
    const float limit = op_params[3];

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_cuda(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc,
            src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

void ggml_swiglu_oai_cuda_f32(const float * x, const float * g, float * dst, const int64_t k, const int64_t n,
        const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    swiglu_oai_cuda(x, g, dst, k, n, o0, o1, alpha, limit, stream);
}

// =========================================

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

static __device__ __forceinline__ float op_sqrt_softplus(float x) {
    return (x > 20.0f) ? sqrtf(x) : sqrtf(log1pf(expf(x)));
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_gelu(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return x / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;
    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_relu_sqr(float x) {
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    // roundf, not round: with nvcc 13.2 + MSVC 14.51 host headers, round(float) in device code rounded halfway cases to
    // even (-148.5 -> -148) while roundf rounds them away from zero (-149) like the CPU backend's roundf. Measured on 601
    // inputs -150..150 step 0.5: round differed on 150, roundf on 0, with and without -use_fast_math
    // (temp/arex1014/_round_probe_1018.cu). test-backend-ops ROUND f16 failed 3 cases on the GPU-backend before this.
    return roundf(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[i])));
}

template <float (*op)(float), typename T>
static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    unary_op_kernel<op><<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(x, dst, k);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    int64_t rem = i;
    const int64_t i0 = rem % ne00; rem /= ne00;
    const int64_t i1 = rem % ne01; rem /= ne01;
    const int64_t i2 = rem % ne02;
    const int64_t i3 = rem / ne02;
    const size_t src_byte_offset = i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    const T * src_ptr = (const T *)((const char *)x + src_byte_offset);

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(*src_ptr)));
}

template <float (*op)(float), typename T>
static void unary_cuda_strided(const T * x, T * dst, const int k,const int64_t ne00,const int64_t ne01,const int64_t ne02,const size_t nb00,const size_t nb01,const size_t nb02,const size_t nb03, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel_strided<op, T>, launch_params, x, dst, k, ne00,ne01,ne02,nb00,nb01,nb02,nb03);
}

template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    if (ggml_is_contiguous(src0)) {
        if (src0->type == GGML_TYPE_F16) {
            unary_cuda<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            unary_cuda<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), stream);
        } else {
            unary_cuda<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), stream);
        }
    } else {
        if (src0->type == GGML_TYPE_F16) {
            unary_cuda_strided<op>((const half *)src0_d, (half *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else if (src0->type == GGML_TYPE_BF16) {
            unary_cuda_strided<op>((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        } else {
            unary_cuda_strided<op>((const float *)src0_d, (float *)dst_d, ggml_nelements(src0),
                src0->ne[0], src0->ne[1], src0->ne[2],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], stream);
        }
    }
}

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_abs>(ctx, dst);
}

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sgn>(ctx, dst);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_neg>(ctx, dst);
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_step>(ctx, dst);
}

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_exp>(ctx, dst);
}

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_softplus>(ctx, dst);
}

static __device__ __forceinline__ float op_softplus_scaled(float x, float s_before, float s_after) {
    x *= s_before;
    float r = (x > 20.0f) ? x : log1pf(expf(x));
    return r*s_after;
}

template <typename T>
static __global__ void scaled_softplus_kernel(const T * x, T * dst, const int k, float s_before, float s_after) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = (T)op_softplus_scaled((float)x[i], s_before, s_after);
}

template <typename T>
static __global__ void scaled_sigmoid_kernel(const T * x, T * dst, const int k, float s_before, float s_after) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    dst[i] = (T)(s_after / (1.0f + expf(-s_before * (float)x[i])));
}

bool ggml_cuda_op_scale_unary_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->op != GGML_OP_SCALE) return false;
    auto unary = dst->src[0];
    if (unary->op != GGML_OP_UNARY) return false;
    auto unary_op = (ggml_unary_op)unary->op_params[0];
    if (unary_op != GGML_UNARY_OP_SOFTPLUS && unary_op != GGML_UNARY_OP_SIGMOID) return false;
    auto src = unary->src[0];
    if (src->op != GGML_OP_SCALE) return false;
    auto src0 = src->src[0];
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) return false;
    if (src0->type != dst->type) return false;
    if (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) return false;

    float s_before, s_after;
    memcpy(&s_before, src->op_params, sizeof(float));
    memcpy(&s_after, dst->op_params, sizeof(float));

    auto nelem = ggml_nelements(src0);
    const int num_blocks = (nelem + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    if (unary_op == GGML_UNARY_OP_SOFTPLUS) {
        if (src0->type == GGML_TYPE_F16) {
            scaled_softplus_kernel<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream()>>>((const half *)src0->data, (half *)dst->data, nelem, s_before, s_after);
        } else {
            scaled_softplus_kernel<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream()>>>((const float *)src0->data, (float *)dst->data, nelem, s_before, s_after);
        }
    } else {
        if (src0->type == GGML_TYPE_F16) {
            scaled_sigmoid_kernel<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream()>>>((const half *)src0->data, (half *)dst->data, nelem, s_before, s_after);
        } else {
            scaled_sigmoid_kernel<<<num_blocks, CUDA_NEG_BLOCK_SIZE, 0, ctx.stream()>>>((const float *)src0->data, (float *)dst->data, nelem, s_before, s_after);
        }
    }
    return true;
}

void ggml_cuda_op_sqrt_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqrt_softplus>(ctx, dst);
}

// === gated ops

template <float (*op)(float), typename T>
static __global__ void unary_gated_op_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ggml_cuda_pdl_sync();
    dst[i] = ggml_cuda_cast<T>(op(ggml_cuda_cast<float>(x[j0])) * ggml_cuda_cast<float>(g[j1]));
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    unary_gated_op_kernel<op><<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1);
}
template <float (*op)(float)>
void ggml_cuda_op_unary_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (half *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        nv_bfloat16 * src0_p = (nv_bfloat16 *) src0_d;
        nv_bfloat16 * src1_p = (nv_bfloat16 *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (nv_bfloat16 *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(nv_bfloat16), src1_o / sizeof(nv_bfloat16), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream);
    }
}

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_relu>(ctx, dst);
}

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu>(ctx, dst);
}

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_silu>(ctx, dst);
}

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst);
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sin>(ctx, dst);
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_cos>(ctx, dst);
}

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_log>(ctx, dst);
}

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_elu>(ctx, dst);
}

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_floor>(ctx, dst);
}

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_ceil>(ctx, dst);
}

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_round>(ctx, dst);
}

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_trunc>(ctx, dst);
}

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_expm1>(ctx, dst);
}

static __global__ void k_fused_softplus(int ne0, int nelem, const float * a, const float * b, const float * c, float * dst) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    int i0 = i % ne0;
    dst[i] = c[i0] * op_softplus(a[i] + b[i0]);
}

void ggml_cuda_fused_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int kBlockSize = 256;
    auto m = dst;
    auto u = dst->src[0];
    auto a = u->src[0];
    GGML_ASSERT(m->op == GGML_OP_MUL);
    GGML_ASSERT(a->op == GGML_OP_ADD);
    GGML_ASSERT(u->op == GGML_OP_UNARY && (ggml_unary_op)u->op_params[0] == GGML_UNARY_OP_SOFTPLUS);
    GGML_ASSERT(ggml_nrows(m->src[1]) == 1 && m->src[1]->ne[0] == m->src[0]->ne[0]);
    GGML_ASSERT(ggml_nrows(a->src[1]) == 1 && a->src[1]->ne[0] == a->src[0]->ne[0]);
    GGML_ASSERT(a->type == GGML_TYPE_F32 && u->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F32);
    GGML_ASSERT(a->src[0]->type == GGML_TYPE_F32 && a->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a->src[0]));

    int nelem = ggml_nelements(a->src[0]);
    int nblock = (nelem + kBlockSize - 1)/kBlockSize;
    k_fused_softplus<<<nblock, kBlockSize, 0, ctx.stream()>>>(a->src[0]->ne[0], nelem,
            (const float *)a->src[0]->data, (const float *)a->src[1]->data, (const float *)m->src[1]->data, (float *)dst->data);
}

static __global__ void k_fused_mul_exp_mul(int ne0, int nelem, const float * x, const float * y, float * dst) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    int i0 = i % ne0;
    dst[i] = y[i0] * expf(x[i] * y[i0]);
}

void ggml_cuda_fused_mul_exp_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int kBlockSize = 256;
    auto m2 = dst;
    auto u  = dst->src[0];
    auto m1 = u->src[0];
    GGML_ASSERT(m1->src[0]->type == GGML_TYPE_F32 && m1->src[1]->type == GGML_TYPE_F32 && m2->type == GGML_TYPE_F32);
    GGML_ASSERT(m1->src[1] == m2->src[1]);
    GGML_ASSERT(ggml_is_contiguous(m1->src[0]) && ggml_is_contiguous(m1->src[1]));
    GGML_ASSERT(u->op == GGML_OP_UNARY && (ggml_unary_op)u->op_params[0] == GGML_UNARY_OP_EXP);

    auto nelem = ggml_nelements(m1->src[0]);
    auto ne0   = ggml_nelements(m1->src[1]);
    int nblock = (nelem + kBlockSize - 1)/kBlockSize;

    k_fused_mul_exp_mul<<<nblock, kBlockSize, 0, ctx.stream()>>>(ne0, nelem,
            (const float *)m1->src[0]->data, (const float *)m1->src[1]->data, (float *)dst->data);
}

// swiglu_clamp

template <typename T>
static __global__ void swiglu_clamp_kernel(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    dst[i] = (T) ggml_cuda_op_swiglu_clamp_single((float) gate[j0], (float) up[j1], limit);
}

template <typename T>
static void swiglu_clamp_cuda(const T * gate, const T * up, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_clamp_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(gate, up, dst, k, n, o0, o1, limit);
}

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float limit = ggml_get_op_params_f32(dst, 3);

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (half *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), limit, stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        swiglu_clamp_cuda(src0_p, src1_p, (float *) dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), limit, stream);
    }
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xi = (float) x[i];

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fminf(xi, eps);
    const float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    dst[i] = (T) out;
}

template <typename T>
static void xielu_cuda(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_XIELU_BLOCK_SIZE - 1) / CUDA_XIELU_BLOCK_SIZE;
    xielu_kernel<<<num_blocks, CUDA_XIELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, alpha_n, alpha_p, beta, eps);
}

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16 ||  dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    if (src0->type == GGML_TYPE_F16) {
        xielu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else if (src0->type == GGML_TYPE_BF16) {
        xielu_cuda((const nv_bfloat16 *)src0_d, (nv_bfloat16 *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}

/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return grad * s * (1.0f + x * (1.0f - s));
}

template <class T>
static __global__ void silu_back_kernel(const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_silu_back((float)grad[i], (float)xf[i]);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

/* fused unary + mul */

template <float (*op)(float)>
static void ggml_cuda_op_unary_mul_impl(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    const ggml_tensor * unary_src = unary_node->src[0];
    const ggml_tensor * other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(ggml_is_contiguous_1(unary_src));
    GGML_ASSERT(unary_src->nb[0] == ggml_element_size(unary_src));
    GGML_ASSERT(ggml_is_contiguous_1(other_src));
    GGML_ASSERT(other_src->nb[0] == ggml_element_size(other_src));
    GGML_ASSERT(ggml_are_same_shape(unary_src, other_src));

    GGML_ASSERT(unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16 || unary_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(unary_src->type == other_src->type);
    GGML_ASSERT(unary_src->type == mul_node->type);

    cudaStream_t stream = ctx.stream();

    const int64_t k  = ggml_nelements(mul_node);
    const int64_t nc = unary_src->ne[0];
    const int64_t unary_stride = unary_src->nb[1];
    const int64_t other_stride = other_src->nb[1];

    if (unary_src->type == GGML_TYPE_F16) {
        unary_gated_cuda<op>((const half *) unary_src->data, (const half *) other_src->data,
                             (half *) mul_node->data, k, nc,
                             unary_stride / sizeof(half), other_stride / sizeof(half), stream);
    } else if (unary_src->type == GGML_TYPE_BF16) {
        unary_gated_cuda<op>((const nv_bfloat16 *) unary_src->data, (const nv_bfloat16 *) other_src->data,
                             (nv_bfloat16 *) mul_node->data, k, nc,
                             unary_stride / sizeof(nv_bfloat16), other_stride / sizeof(nv_bfloat16), stream);
    } else {
        unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                             (float *) mul_node->data, k, nc,
                             unary_stride / sizeof(float), other_stride / sizeof(float), stream);
    }
}

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:
            ggml_cuda_op_unary_mul_impl<op_silu>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ggml_cuda_op_unary_mul_impl<op_sigmoid>(ctx, unary_node, mul_node);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            ggml_cuda_op_unary_mul_impl<op_softplus>(ctx, unary_node, mul_node);
            break;
        default:
            GGML_ABORT("Unsupported unary op for fused unary+mul");
    }
}

/* fused relu + sqr */

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node) {
    const ggml_tensor * src = relu_node->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}

/* F16 route for ik's F32-only unary kernels (see the forward declaration at the top) */

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static void ggml_cuda_op_unary_f16_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // SQR / SQRT are their own ggml ops, not GGML_OP_UNARY: their f32 kernels assert F32 while
    // supports_op claimed every type -- test-backend-ops aborted at unary.cu (SQR / SQRT f16).
    if (dst->op == GGML_OP_SQR)  { ggml_cuda_op_unary<op_sqr>(ctx, dst);  return; }
    if (dst->op == GGML_OP_SQRT) { ggml_cuda_op_unary<op_sqrt>(ctx, dst); return; }
    switch (ggml_get_unary_op(dst)) {
        case GGML_UNARY_OP_GELU:        ggml_cuda_op_unary<op_gelu>(ctx, dst);        break;
        case GGML_UNARY_OP_SILU:        ggml_cuda_op_unary<op_silu>(ctx, dst);        break;
        case GGML_UNARY_OP_GELU_QUICK:  ggml_cuda_op_unary<op_gelu_quick>(ctx, dst);  break;
        case GGML_UNARY_OP_TANH:        ggml_cuda_op_unary<op_tanh>(ctx, dst);        break;
        case GGML_UNARY_OP_RELU:        ggml_cuda_op_unary<op_relu>(ctx, dst);        break;
        case GGML_UNARY_OP_SIGMOID:     ggml_cuda_op_unary<op_sigmoid>(ctx, dst);     break;
        case GGML_UNARY_OP_HARDSIGMOID: ggml_cuda_op_unary<op_hardsigmoid>(ctx, dst); break;
        case GGML_UNARY_OP_HARDSWISH:   ggml_cuda_op_unary<op_hardswish>(ctx, dst);   break;
        default: GGML_ABORT("%s: no F16 route for unary op %d", __func__, (int) ggml_get_unary_op(dst));
    }
}
