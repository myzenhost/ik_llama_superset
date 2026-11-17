#include "indexer_topk.cuh"
#include <cfloat>
#include "mmq.cuh"
#include "quantize.cuh"
#include "convert.cuh"
#include "argsort.cuh"
#include "topk-select.cuh"

template <typename mask_t>
static __device__ __forceinline__ float indexer_mask_value(const mask_t * m, int col) {
    return (float)m[col];
}

template <>
__device__ __forceinline__ float indexer_mask_value(const int * m, int col) {
    return col < *m ? 0.0f : -INFINITY;
}

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst, const int ncols, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    int64_t step = ncols*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    for (int i = col; i < ncols; i += blockDim.x) {
        float sum = indexer_mask_value(this_m, i);
        auto this_kq = kq + blockIdx.x * step;
        for (int head = 0; head < nhead; ++head) {
            float relu = (float)this_kq[i];
            relu = relu > 0.0f ? relu : 0.0f;
            sum += relu * this_w[head];
            this_kq += ncols;
        }
        dst[ncols*row + i] = sum;
    }
}

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows_2(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst, const int ncols, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = blockIdx.y*blockDim.x + threadIdx.x;
    if (col >= ncols) {
        return;
    }

    int64_t step = ncols*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    float sum = indexer_mask_value(this_m, col);
    auto this_kq = kq + row * step;
    for (int head = 0; head < nhead; ++head) {
        float relu = (float)this_kq[col];
        relu = relu > 0.0f ? relu : 0.0f;
        sum += relu * this_w[head];
        this_kq += ncols;
    }
    dst[ncols*row + col] = sum;
}

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows_3(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m,
        const int * __restrict__ cell_idx, float * __restrict__ dst, const int ncols, int ncols_kq, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = blockIdx.y*blockDim.x + threadIdx.x;
    if (col >= ncols) {
        return;
    }

    int64_t step = ncols_kq*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    int col_kq = cell_idx[col];

    float sum = indexer_mask_value(this_m, col);
    auto this_kq = kq + row * step;
    for (int head = 0; head < nhead; ++head) {
        float relu = (float)this_kq[col_kq];
        relu = relu > 0.0f ? relu : 0.0f;
        sum += relu * this_w[head];
        this_kq += ncols_kq;
    }
    dst[ncols*row + col] = sum;
}

static __global__ void k_copy_topk(const int * __restrict__ sorted, int * dst, const int ncols, const int n_top_k) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    sorted += int64_t(ncols)*row;
    dst    += int64_t(n_top_k)*row;
    for (int i = col; i < n_top_k; i += blockDim.x) {
        dst[i] = sorted[i];
    }
}

// Exact top-k by radix SELECT instead of a full argsort of n_kv scores (topk-select.cuh, rows
// S69-EXLLAMAV3-PICK-DSATOPKRADIX-2026-09-28-01 and S69-GGML-PICK-INDEXER-TOPK-PARTIAL-SELECT-2026-09-29-01).
// Behind GGML_CUDA_INDEXER_TOPK_SELECT=1 until the device run in row S69-F2-GPU-VERIFY-QUEUE-2026-10-05-01 has compared it
// with the sort path on the 5090; the CPU-emulated kernel body is covered by tests/test-topk-select.cpp, the launch is not.
// Returns false when it does not apply (switch unset, k > TOPK_SELECT_MAX_K, k > n), and the caller sorts as before.
static bool ggml_cuda_indexer_topk_select(cudaStream_t stream, const float * score, int * dst_rows,
                                          int n, int k, int nrows) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_INDEXER_TOPK_SELECT");
        return e != nullptr && *e != '\0' && *e != '0';
    }();
    if (!enabled || !topk_select_supported(n, k) || nrows <= 0) {
        return false;
    }
    k_topk_select<<<nrows, TOPK_SELECT_THREADS, 0, stream>>>(score, dst_rows, n, k, topk_pad_pow2(k), (int64_t) k);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// The top-k of each of nrows score rows of n entries into out (k ints per row, contiguous). ik #2819 calls this
// indexer_select_top_k (its radix select, ik #2767, is not in the fork): here it is the fork's opt-in select, else the
// argsort + copy the fork used inline before. sorted needs n*nrows ints.
static void indexer_select_top_k(ggml_backend_cuda_context & ctx, const float * score, int * sorted, int * out,
                                 int n, int nrows, int k) {
    if (ggml_cuda_indexer_topk_select(ctx.stream(), score, out, n, k, nrows)) {
        return;
    }
    argsort_f32_i32_cuda_cub(ctx.pool(), score, sorted, n, nrows, GGML_SORT_ORDER_DESC, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    k_copy_topk<<<nrows, 256, 0, ctx.stream()>>>(sorted, out, n, k);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void k_indexer_apply_group_keep(float * __restrict__ score, const int * __restrict__ group_ids,
        int n_kv, int n_b, int group_size, int n_top_groups, int64_t ngi) {
    extern __shared__ unsigned int bits[];
    const int row   = blockIdx.x;
    const int nword = (n_b + 31)/32;

    for (int i = threadIdx.x; i < nword; i += blockDim.x) {
        bits[i] = 0u;
    }
    __syncthreads();

    const int * ids = (const int *)((const char *)group_ids + (size_t)row*ngi*sizeof(int));
    for (int j = threadIdx.x; j < n_top_groups; j += blockDim.x) {
        const int b = ids[j];
        if (b >= 0 && b < n_b) {
            atomicOr(&bits[b >> 5], 1u << (b & 31));
        }
    }
    __syncthreads();

    float * s = score + (size_t)row*n_kv;
    for (int col = threadIdx.x; col < n_kv; col += blockDim.x) {
        const int b = col/group_size;
        if ((bits[b >> 5] & (1u << (b & 31))) == 0u) {
            s[col] = -INFINITY;
        }
    }
}

static __global__ void k_indexer_group_max(const float * __restrict__ score, float * __restrict__ group_score,
        int n_kv, int n_b, int group_size) {
    const int row = blockIdx.x;
    const float * s = score + (size_t)row*n_kv;
    float * gs = group_score + (size_t)row*n_b;

    for (int b = threadIdx.x; b < n_b; b += blockDim.x) {
        const float * sb = s + (size_t)b*group_size;
        float mx = -INFINITY;
        for (int i = 0; i < group_size; ++i) {
            mx = fmaxf(mx, sb[i]);
        }
        gs[b] = mx;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int b = n_b - 1; b >= 0; --b) {
            if (gs[b] > -INFINITY) {
                gs[b] = FLT_MAX;
                break;
            }
        }
    }
}

static void indexer_topk_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto op = ggml_unary_op(dst->op_params[0]);
    GGML_ASSERT(op == GGML_UNARY_OP_RELU);
    auto k = dst->src[0];
    auto q = dst->src[1];
    auto w = dst->src[2];
    auto m = dst->src[3];
    // ik #2819: op_params[1] = group size (block for select_groups, else ik's qwen4exp ratio -- the fork has no
    // ratio-aware MMA scorer, ik #2784/#2809, so it is unused here), op_params[2] = select the top groups
    const int  group_size    = (int) dst->op_params[1];
    const bool select_groups = dst->op_params[2] != 0;
    auto c = select_groups ? nullptr : dst->src[4];
    int n_top_k = dst->ne[0];
    int n_kv    = m->type != GGML_TYPE_I32 ? m->ne[0] : c ? c->ne[0] : k->ne[1];
    int n_head  = q->ne[1];
    //if (k->type != GGML_TYPE_F16 && !ggml_is_quantized(k->type)) printf("%s: K is %s?\n", __func__, ggml_type_name(k->type));
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_F32 || ggml_is_quantized(k->type));
    GGML_ASSERT(k->ne[2] == 1 || k->ne[3] == 1);
    GGML_ASSERT(k->ne[0] == q->ne[0]);
    if (c) {
        GGML_ASSERT(c->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nrows(c) == 1);
        GGML_ASSERT(c->ne[0] == n_kv);
        //printf("Using cell_idx: k = %ld x %ld, c = %ld, n_kv = %d\n", k->ne[0], k->ne[1], c->ne[0], n_kv);
    } else {
        GGML_ASSERT(k->ne[1] > n_top_k);
        GGML_ASSERT(k->ne[1] == n_kv);
    }
    GGML_ASSERT(q->ne[2] == m->ne[1]);
    GGML_ASSERT(q->ne[1] == w->ne[0]);
    GGML_ASSERT(q->ne[2] == w->ne[1]);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F32 || m->type == GGML_TYPE_F16 || m->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(w));

    constexpr int k_block_size = 256;

    const int n_b = select_groups ? n_kv/group_size : 0;
    ggml_tensor * top_k_out = select_groups ? dst->src[5] : nullptr;
    if (select_groups) {
        GGML_ASSERT(group_size > 0 && n_kv % group_size == 0);
        GGML_ASSERT(dst->ne[0] <= n_b);
        if (top_k_out) {
            GGML_ASSERT(top_k_out->type == GGML_TYPE_I32);
            GGML_ASSERT(top_k_out->ne[1] == q->ne[2] && top_k_out->ne[0] <= n_kv);
        }
    }
    auto finish_rows = [&](const float * score, int * sorted, int nrows, int first_row) {
        int * out = (int *)((char *)dst->data + first_row*dst->nb[1]);
        if (!select_groups) {
            indexer_select_top_k(ctx, score, sorted, out, n_kv, nrows, dst->ne[0]);
            return;
        }
        ggml_cuda_pool_alloc<float> group_score(ctx.pool(), int64_t(n_b)*nrows);
        k_indexer_group_max<<<nrows, k_block_size, 0, ctx.stream()>>>(score, group_score.get(), n_kv, n_b, group_size);
        CUDA_CHECK(cudaGetLastError());
        indexer_select_top_k(ctx, group_score.get(), sorted, out, n_b, nrows, dst->ne[0]);
        if (top_k_out) {
            const int nword = (n_b + 31)/32;
            k_indexer_apply_group_keep<<<nrows, k_block_size, nword*sizeof(unsigned int), ctx.stream()>>>(
                    const_cast<float *>(score), out, n_kv, n_b, group_size, (int) dst->ne[0],
                    dst->nb[1]/sizeof(int));
            CUDA_CHECK(cudaGetLastError());
            int * out2 = (int *)((char *)top_k_out->data + (size_t)first_row*top_k_out->nb[1]);
            indexer_select_top_k(ctx, score, sorted, out2, n_kv, nrows, top_k_out->ne[0]);
        }
    };

    if ((k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) && q->type == GGML_TYPE_F32) {
        constexpr size_t k_max_buf_size = 1 << 28;
        size_t per_row = size_t(n_kv)*(q->ne[1]*sizeof(half) + sizeof(int) + sizeof(float)) + q->ne[0]*q->ne[1]*sizeof(half);
        int max_rows = (k_max_buf_size + per_row - 1)/per_row;
        max_rows = std::min<int>(max_rows, q->ne[2]);
        int nstep = (q->ne[2] + max_rows - 1)/max_rows;

        ggml_cuda_pool_alloc<float> kq(ctx.pool(), k->ne[1]*q->ne[1]*max_rows);
        ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*max_rows);
        ggml_cuda_pool_alloc<int>   sorted(ctx.pool(), int64_t(n_kv)*max_rows);
        ggml_cuda_pool_alloc<half>  q_f16(ctx.pool(), q->ne[0]*q->ne[1]*max_rows);
        ggml_cuda_pool_alloc<half>  k_f16(ctx.pool());

        auto k_data = (const half *)k->data;
        if (k->type == GGML_TYPE_Q8_0) {
            k_f16.alloc(k->ne[0]*k->ne[1]);
            auto to_fp16_cuda = ggml_get_to_fp16_cuda(k->type);
            GGML_ASSERT(to_fp16_cuda);
            to_fp16_cuda(k->data, k_f16.get(), k->ne[0]*k->ne[1], 1, ctx.stream());
            CUDA_CHECK(cudaGetLastError());
            k_data = k_f16.get();
        }

        auto to_fp16_cuda = ggml_get_to_fp16_cuda(q->type);
        GGML_ASSERT(to_fp16_cuda);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));

        for (int istep = 0; istep < nstep; ++istep) {
            int first_row = max_rows*istep;
            if (first_row >= int(q->ne[2])) break;
            int last_row  = std::min(first_row + max_rows, int(q->ne[2]));
            int nrows     = last_row - first_row;

            to_fp16_cuda((const float *)q->data + q->ne[0]*q->ne[1]*first_row, q_f16.get(), q->ne[0]*q->ne[1]*nrows, 1, ctx.stream());
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(ctx.device), CUBLAS_OP_T, CUBLAS_OP_N,
                    k->ne[1], q->ne[1]*nrows, q->ne[0],
                    &alpha, k_data,       CUDA_R_16F, k->ne[0],
                            q_f16.get(),  CUDA_R_16F, q->ne[0],
                    &beta,  kq.get(),     CUDA_R_32F, k->ne[1],
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            if (c) {
                int nblocks = (n_kv + k_block_size - 1)/k_block_size;
                dim3 grid(nrows, nblocks, 1);
                if (m->type == GGML_TYPE_F32) {
                    k_fused_relu_mul_sum_rows_3<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const float *)((const char *)m->data + first_row*m->nb[1]),
                            (const int *)c->data,
                            score.get(), n_kv, k->ne[1], q->ne[1], m->nb[1]);
                } else if (m->type == GGML_TYPE_I32) {
                    k_fused_relu_mul_sum_rows_3<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const int   *)((const char *)m->data + first_row*m->nb[1]),
                            (const int *)c->data,
                            score.get(), n_kv, k->ne[1], q->ne[1], m->nb[1]);
                } else {
                    k_fused_relu_mul_sum_rows_3<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const half  *)((const char *)m->data + first_row*m->nb[1]),
                            (const int *)c->data,
                            score.get(), n_kv, k->ne[1], q->ne[1], m->nb[1]);
                }
                CUDA_CHECK(cudaGetLastError());
            } else {
                int nblocks = (k->ne[1] + k_block_size - 1)/k_block_size;
                dim3 grid(nrows, nblocks, 1);
                if (m->type == GGML_TYPE_F32) {
                    k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const float *)((const char *)m->data + first_row*m->nb[1]),
                            score.get(), k->ne[1], q->ne[1], m->nb[1]);
                } else if (m->type == GGML_TYPE_I32) {
                    k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const int   *)((const char *)m->data + first_row*m->nb[1]),
                            score.get(), k->ne[1], q->ne[1], m->nb[1]);
                } else {
                    k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const half  *)((const char *)m->data + first_row*m->nb[1]),
                            score.get(), k->ne[1], q->ne[1], m->nb[1]);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            finish_rows(score.get(), sorted.get(), nrows, first_row);
        }

        return;

    }

    constexpr int64_t k_max_work_buffer_elements = 1 << 26;

    int max_rows = k_max_work_buffer_elements / n_kv / n_head;
    if (max_rows < 1) max_rows = 1;
    if (max_rows > q->ne[2]) max_rows = q->ne[2];

    int nstep = (q->ne[2] + max_rows - 1)/max_rows;

    ggml_cuda_pool_alloc<float> kq(ctx.pool(), int64_t(n_kv)*max_rows*n_head);
    ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*max_rows);
    ggml_cuda_pool_alloc<int>   sorted(ctx.pool(), int64_t(n_kv)*max_rows);
    ggml_cuda_pool_alloc<float> k_f32(ctx.pool());
    ggml_cuda_pool_alloc<char>  q_converted(ctx.pool());
    const float * k_data = nullptr;
    int k_ld = k->ne[0];
    auto q_padded = GGML_PAD(q->ne[0], MATRIX_ROW_PADDING);
    if (ggml_is_quantized(k->type)) {
        auto nbytes_q = (size_t)(q_padded/QK8_1) * ((size_t) q->ne[1] * max_rows) * sizeof(block_q8_1);
        nbytes_q += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
        q_converted.alloc(nbytes_q);
    } else if (k->type == GGML_TYPE_F32) {
        k_data = (const float *) k->data;
        k_ld   = k->nb[1]/sizeof(float);
    } else {
        k_f32.alloc(k->ne[0]*k->ne[1]);
        ggml_get_to_fp32_cuda(k->type)(k->data, k_f32.get(), k->ne[1]*k->ne[0], 1, ctx.stream());
        CUDA_CHECK(cudaGetLastError());
        k_data = k_f32.get();
    }

    for (int istep = 0; istep < nstep; ++istep) {
        int first = istep*max_rows;
        int last  = std::min(first + max_rows, int(q->ne[2]));
        int nrows = last - first;
        auto q_data = (const char *)q->data + istep*max_rows*q->nb[2];
        auto m_data = (const char *)m->data + istep*max_rows*m->nb[1];
        auto w_data = (const float *)w->data + first*q->ne[1];
        if (ggml_is_quantized(k->type)) {
            quantize_mmq_q8_1_cuda((const float *)q_data, q_converted.get(), q->ne[0], q->ne[1]*nrows, 1, q_padded, k->type, ctx.stream());
            CUDA_CHECK(cudaGetLastError());
            mmq_args args{(const char *)k->data, q_converted.get(), kq.get(),
                k->ne[0], k->ne[1], int64_t(k->nb[1]),
                q_padded, q->ne[1]*nrows, q->ne[1]*nrows, k->ne[1]};
            ggml_cuda_op_mul_mat_q(ctx, k->type, args);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // I wonder if it makes sense to use CUBLAS. If we did simple dot products we could fuse the
            // relu, mul, sum_rows all in one kernel, avoiding the k*q intermediate result.
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));
            CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(ctx.device), CUBLAS_OP_T, CUBLAS_OP_N,
                    k->ne[1], q->ne[1]*nrows, q->ne[0],
                    &alpha,     k_data,                k_ld,
                       (const float *)q_data, q->ne[0],
                    &beta,      kq.get(),     k->ne[1]));
        }
        if (m->type == GGML_TYPE_F32) {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const float *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        } else if (m->type == GGML_TYPE_I32) {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const int   *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        } else {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const half  *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        }
        CUDA_CHECK(cudaGetLastError());

        finish_rows(score.get(), sorted.get(), nrows, first);
    }

}

void ggml_cuda_op_indexer_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    indexer_topk_impl(ctx, dst);
}

template <typename mask_t>
static __global__ void k_indexer_mask(int ne0, int ne1, int ne2, int ntopk, int ne11,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb1,  size_t nb2,  size_t nb3,
        const mask_t * mask, const int * idx, mask_t * dst) {
    int i1 = blockIdx.x;
    int i3 = i1 / (ne1*ne2); i1 -= i3*ne1*ne2;
    int i2 = i1 / (ne1);     i1 -= i2*ne1;

    auto m = (const mask_t *)((const char *)mask + i1*nb01 + i2*nb02 + i3*nb03);
    auto i = (const int *)((const char *)idx + i1*nb11 + i2*nb12 + i3*nb13);
    auto d = (mask_t *)((char *)dst + i1*nb1 + i2*nb2 + i3*nb3);

    mask_t inf, zero;
    if constexpr (std::is_same_v<mask_t, half>) {
        inf  = __float2half(-INFINITY);
        zero = __float2half(0.0f);
    } else {
        inf  = -INFINITY;
        zero = 0.0f;
    }

    if (i1 < ne11) {
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] = inf;
        __syncthreads();
        for (int j = threadIdx.x; j < ntopk; j += blockDim.x) if (i[j] >= 0) d[i[j]] = zero;  // i < 0 = "no cell" (an unused k-pool tail slot)
        __syncthreads();
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] += m[j];
    } else {
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] = m[j];
    }

}

template <typename mask_t>
static __global__ void k_visible_mask(int ne0, const int * n_visible, mask_t * dst) {
    const int n = n_visible[blockIdx.x];
    dst += int64_t(ne0)*blockIdx.x;
    for (int j = threadIdx.x; j < ne0; j += blockDim.x) dst[j] = j < n ? 0.0f : -INFINITY;
}

void ggml_cuda_op_indexer_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto mask = dst->src[0];
    auto topk = dst->src[1];
    if (mask->type == GGML_TYPE_I32) {
        if (dst->type == GGML_TYPE_F16) {
            k_visible_mask<<<ggml_nrows(dst), 256, 0, ctx.stream()>>>(dst->ne[0], (const int *)mask->data, (half *)dst->data);
        } else {
            k_visible_mask<<<ggml_nrows(dst), 256, 0, ctx.stream()>>>(dst->ne[0], (const int *)mask->data, (float *)dst->data);
        }
        return;
    }
    GGML_ASSERT(mask->ne[0] >= topk->ne[0]);
    GGML_ASSERT(mask->ne[1] >= topk->ne[1] && mask->ne[2] == topk->ne[2] && mask->ne[3] == topk->ne[3]);
    GGML_ASSERT(ggml_are_same_shape(mask, dst));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(topk->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == dst->type);

    int nrows = ggml_nrows(dst);

    if (dst->type == GGML_TYPE_F16) {
        k_indexer_mask<<<nrows, 256, 0, ctx.stream()>>>(dst->ne[0], dst->ne[1], dst->ne[2], topk->ne[0], topk->ne[1],
                mask->nb[1], mask->nb[2], mask->nb[3],
                topk->nb[1], topk->nb[2], topk->nb[3],
                dst->nb[1],  dst->nb[2],  dst->nb[3],
                (const half *)mask->data, (const int *)topk->data, (half *)dst->data);
    } else {
        k_indexer_mask<<<nrows, 256, 0, ctx.stream()>>>(dst->ne[0], dst->ne[1], dst->ne[2], topk->ne[0], topk->ne[1],
                mask->nb[1], mask->nb[2], mask->nb[3],
                topk->nb[1], topk->nb[2], topk->nb[3],
                dst->nb[1],  dst->nb[2],  dst->nb[3],
                (const float *)mask->data, (const int *)topk->data, (float *)dst->data);
    }

}

template <typename mask_t>
static __global__ void k_mask_to_index(int ne00, [[maybe_unused]] int ne0,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb1,  size_t nb2,  size_t nb3,
        const mask_t * __restrict__ mask, int * __restrict__ idx) {
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;

    mask_t zero;
    if constexpr (std::is_same_v<mask_t, half>) {
        zero = __float2half(0.0f);
    } else {
        zero = 0.0f;
    }
    const mask_t * mask_r = (const mask_t *)((const char *)mask + i1*nb01 + i2*nb02 + i3*nb03);
    int * idx_r = (int *)((char *)idx + i1*nb1 + i2*nb2 + i3*nb3);

    // One warp per row, stream compaction by ballot: each 32-column chunk is compacted in column
    // order and appended, so the row comes out ASCENDING -- the same order as the CPU op
    // (ggml-cpu/ops.cpp), which keeps the first ne0 zeros in ascending order. The previous kernel
    // gave each lane a strided subset of columns and a prefix slot per lane, so it emitted the same
    // SET in lane-interleaved order: test-backend-ops compares every node, and this node alone
    // failed every sparse FLASH_ATTN_EXT case (ERR 0.475) while the attention output agreed.
    // Bounded: ne0 = MIN(mask ne0, max_row_size) (ggml_mask_to_index); zeros past ne0 are dropped.
    static_assert(WARP_SIZE == 32, "lane mask arithmetic assumes a 32-wide warp");
    const int lane = threadIdx.x;
    const unsigned lanes_below = (1u << lane) - 1u;
    int base = 0; // uniform across the warp
    for (int j0 = 0; j0 < ne00 && base < ne0; j0 += WARP_SIZE) {
        const int  j  = j0 + lane;
        const bool on = j < ne00 && mask_r[j] == zero;
        const unsigned ballot = __ballot_sync(0xffffffffu, on);
        const int slot = base + __popc(ballot & lanes_below);
        if (on && slot < ne0) {
            idx_r[slot] = j;
        }
        base += __popc(ballot);
    }
    for (int j = min(base, ne0) + lane; j < ne0; j += WARP_SIZE) {
        idx_r[j] = -1;
    }
}

// the list of k_mask_to_index for the mask concat(mask, visible top-k columns), in its (fork) order: ASCENDING column,
// the same order as the CPU op (ggml-cpu/ops.cpp). ik sorts by (column % WARP_SIZE, column / WARP_SIZE) because its
// k_mask_to_index emits lane-interleaved; this fork's k_mask_to_index is ascending (ballot), and test-backend-ops
// compares every node, so the interleaved key made MASK_TO_IDX fail CUDA0 vs CPU (ERR 0.5166, FA kv_idx hs=128 case).
template <typename mask_t>
static __global__ void k_mask_topk_to_index(int ne00, int ne0, int ntopk, int ntok, int P,
        size_t nb01, size_t nb11, size_t nb21, size_t nb1,
        const mask_t * __restrict__ mask, const int * __restrict__ topk, const char * __restrict__ n_visible, int * __restrict__ idx) {
    extern __shared__ uint32_t keys[];
    __shared__ int n_on;

    const int row = blockIdx.x;
    const mask_t * mask_r = (const mask_t *)((const char *)mask + row*nb01);
    int * idx_r = (int *)((char *)idx + row*nb1);

    mask_t zero;
    if constexpr (std::is_same_v<mask_t, half>) {
        zero = __float2half(0.0f);
    } else {
        zero = 0.0f;
    }

    if (threadIdx.x == 0) n_on = 0;
    for (int i = threadIdx.x; i < P; i += blockDim.x) keys[i] = 0xffffffff;
    __syncthreads();

    if (n_visible) {
        for (int j = threadIdx.x; j < ne00; j += blockDim.x) {
            if (mask_r[j] == zero) {
                const int s = atomicAdd(&n_on, 1);
                if (s < P) keys[s] = uint32_t(j);
            }
        }
    }
    if (row < ntok) {
        const int * t = (const int *)((const char *)topk + row*nb11);
        const int n = n_visible ? *(const int *)(n_visible + row*nb21) : 0;
        for (int j = threadIdx.x; j < ntopk; j += blockDim.x) {
            if (n_visible ? t[j] < n : (t[j] >= 0 && t[j] < ne00 && mask_r[t[j]] == zero)) {
                const int c = (n_visible ? ne00 : 0) + t[j];
                const int s = atomicAdd(&n_on, 1);
                if (s < P) keys[s] = uint32_t(c);
            }
        }
    }

    for (int size = 2; size <= P; size *= 2) {
        for (int stride = size/2; stride > 0; stride /= 2) {
            __syncthreads();
            for (int i = threadIdx.x; i < P/2; i += blockDim.x) {
                const int lo = 2*i - (i & (stride - 1));
                const int hi = lo + stride;
                const uint32_t a = keys[lo], b = keys[hi];
                if ((a > b) == ((lo & size) == 0)) { keys[lo] = b; keys[hi] = a; }
            }
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ne0; j += blockDim.x) idx_r[j] = keys[j] == 0xffffffff ? -1 : int(keys[j]);
}

template <typename mask_t>
static void mask_topk_to_index_dense(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * topk, ggml_tensor * dst) {
    ggml_cuda_pool_alloc<mask_t> mask(ctx.pool(), src->ne[0]*dst->ne[1]);
    const size_t nb1 = src->ne[0]*sizeof(mask_t);
    k_indexer_mask<<<dst->ne[1], 256, 0, ctx.stream()>>>(src->ne[0], dst->ne[1], 1, topk->ne[0], topk->ne[1],
            src->nb[1], 0, 0, topk->nb[1], 0, 0, nb1, 0, 0,
            (const mask_t *)src->data, (const int *)topk->data, mask.get());
    k_mask_to_index<<<dst->ne[1], WARP_SIZE, 0, ctx.stream()>>>(src->ne[0], dst->ne[0], nb1, 0, 0, dst->nb[1], 0, 0,
            mask.get(), (int *)dst->data);
}

void ggml_cuda_op_mask_to_index(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    if (auto topk = dst->src[1]) {
        auto vis = dst->src[2];
        const size_t nb21 = vis ? vis->nb[1] : 0;
        const char * vis_data = vis ? (const char *)vis->data : nullptr;
        int P = 1;
        while (P < dst->ne[0]) P *= 2;
        constexpr int max_keys = 8192; // P*4 bytes of shared memory
        if (!vis && P > max_keys) {
            if (src->type == GGML_TYPE_F16) {
                mask_topk_to_index_dense<half>(ctx, src, topk, dst);
            } else {
                mask_topk_to_index_dense<float>(ctx, src, topk, dst);
            }
            return;
        }
        if (src->type == GGML_TYPE_F16) {
            k_mask_topk_to_index<<<dst->ne[1], 256, P*sizeof(uint32_t), ctx.stream()>>>(src->ne[0], dst->ne[0], topk->ne[0], topk->ne[1], P,
                    src->nb[1], topk->nb[1], nb21, dst->nb[1],
                    (const half *)src->data, (const int *)topk->data, vis_data, (int *)dst->data);
        } else {
            k_mask_topk_to_index<<<dst->ne[1], 256, P*sizeof(uint32_t), ctx.stream()>>>(src->ne[0], dst->ne[0], topk->ne[0], topk->ne[1], P,
                    src->nb[1], topk->nb[1], nb21, dst->nb[1],
                    (const float *)src->data, (const int *)topk->data, vis_data, (int *)dst->data);
        }
        return;
    }
    GGML_ASSERT(src->ne[1] == dst->ne[1] && src->ne[2] == dst->ne[2] && src->ne[3] == dst->ne[3]);
    GGML_ASSERT(src->ne[0] >= dst->ne[0]);

    dim3 grid(dst->ne[1], dst->ne[2], dst->ne[3]);
    if (src->type == GGML_TYPE_F16) {
        k_mask_to_index<<<grid, WARP_SIZE, 0, ctx.stream()>>>(src->ne[0], dst->ne[0],
                src->nb[1], src->nb[2], src->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                (const half *)src->data, (int *)dst->data);
    } else {
        k_mask_to_index<<<grid, WARP_SIZE, 0, ctx.stream()>>>(src->ne[0], dst->ne[0],
                src->nb[1], src->nb[2], src->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                (const float *)src->data, (int *)dst->data);
    }

}
