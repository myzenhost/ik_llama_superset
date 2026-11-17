//
// TurboQuant Walsh-Hadamard rotation for the SYCL backend.
//
// One work-group of 128 work-items per 128-element group; the butterflies run in work-group local memory, the same stage order and
// the same multiply order as k_turbo_wht_f32<direction, 128> in ggml-cuda/turbo-wht.cu and as the CPU turbo_cpu_fwht, so the three
// agree to the last bit apart from FMA contraction.
//   forward (direction 0): [x * scale_inv] -> signs1 -> butterflies -> (1/sqrt(128)) * signs2
//   inverse (direction 1): signs2 -> butterflies -> (1/sqrt(128)) * signs1 [-> * scale_inv]
//
// Differences from TheTom/llama-cpp-turboquant bcb85fc3ae85 ggml-sycl/turbo-wht.cpp (MIT, read in full 2026-10-05; rewritten here):
//   * direction and group_size come from op_params[0] and op_params[1]. The tip read group_size from `op_params + sizeof(int)`,
//     which is element 4 of the int32_t array (always 0), so every call tripped its own assert; the same slip was fixed in this
//     fork's CUDA op.
//   * the source may be a permuted view (byte strides along dims 1..3), so a flash-attention Q needs no separate copy kernel.
//   * only group_size 128 (the KV rotation); the tip's 64-wide variant has no counterpart in this fork's tables.
//
// GATE-CLAIM-OK: a kernel source file; it declares no resource unavailable.
//
#include <cstring>

#include "turbo-wht.hpp"
#include "turbo-quant.hpp"

template <int direction>
static void k_turbo_wht_128(const char * __restrict__ src, const size_t nb1, const size_t nb2, const size_t nb3,
                            float * __restrict__ dst, const float * __restrict__ scale_inv,
                            const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t groups_per_row,
                            sycl::local_accessor<float, 1> shared, const sycl::nd_item<1> & item) {
    const int64_t g = item.get_group(0);
    const int     t = item.get_local_id(0);

    const int64_t row = g / groups_per_row;
    const int64_t grp = g % groups_per_row;
    const int64_t i1  = row % ne1;
    const int64_t r2  = row / ne1;
    const int64_t i2  = r2 % ne2;
    const int64_t i3  = r2 / ne2;

    const float * s = (const float *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + grp * 128;
    float *       d = dst + row * ne0 + grp * 128;

    shared[t] = s[t];
    item.barrier(sycl::access::fence_space::local_space);

    if constexpr (direction == 0) {
        if (scale_inv != nullptr) {
            shared[t] *= scale_inv[t];
            item.barrier(sycl::access::fence_space::local_space);
        }
        shared[t] *= TURBO_WHT_SIGNS1[t];
    } else {
        shared[t] *= TURBO_WHT_SIGNS2[t];
    }
    item.barrier(sycl::access::fence_space::local_space);

#define TURBO_WHT_STAGE(h)                                                  \
    if (t % (2 * (h)) < (h)) {                                              \
        const float a = shared[t], b = shared[t + (h)];                     \
        shared[t] = a + b;                                                  \
        shared[t + (h)] = a - b;                                            \
    }                                                                       \
    item.barrier(sycl::access::fence_space::local_space);

    TURBO_WHT_STAGE(1)
    TURBO_WHT_STAGE(2)
    TURBO_WHT_STAGE(4)
    TURBO_WHT_STAGE(8)
    TURBO_WHT_STAGE(16)
    TURBO_WHT_STAGE(32)
    TURBO_WHT_STAGE(64)
#undef TURBO_WHT_STAGE

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    float result;
    if constexpr (direction == 0) {
        result = shared[t] * inv_sqrt_128 * TURBO_WHT_SIGNS2[t];
    } else {
        result = shared[t] * inv_sqrt_128 * TURBO_WHT_SIGNS1[t];
        if (scale_inv != nullptr) {
            result *= scale_inv[t];
        }
    }
    d[t] = result;
}

// Elements past the last whole 128-group of a row are copied through unrotated, as in the CUDA op.
static void k_turbo_wht_copy_tail(const char * __restrict__ src, const size_t nb1, const size_t nb2, const size_t nb3,
                                  float * __restrict__ dst, const int64_t ne0, const int64_t ne1, const int64_t ne2,
                                  const int64_t tail_offset, const int tail_size, const int64_t n_rows,
                                  const sycl::nd_item<1> & item) {
    const int64_t i = item.get_global_linear_id();
    if (i >= n_rows * tail_size) {
        return;
    }
    const int64_t row = i / tail_size;
    const int64_t e   = tail_offset + i % tail_size;
    const int64_t i1  = row % ne1;
    const int64_t r2  = row / ne1;
    const int64_t i2  = r2 % ne2;
    const int64_t i3  = r2 / ne2;
    dst[row * ne0 + e] = ((const float *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3))[e];
}

static void launch_turbo_wht(dpct::queue_ptr stream, const char * src, const size_t nb1, const size_t nb2, const size_t nb3,
                             float * dst, const float * scale_inv, const int64_t ne0, const int64_t ne1, const int64_t ne2,
                             const int64_t ne3, const int direction) {
    GGML_ASSERT(direction == 0 || direction == 1);
    const int64_t n_rows          = ne1 * ne2 * ne3;
    const int64_t groups_per_row  = ne0 / 128;
    const int64_t n_groups        = groups_per_row * n_rows;
    const int     tail_size       = (int) (ne0 % 128);

    if (n_groups > 0) {
        const sycl::range<1> global(n_groups * 128);
        const sycl::range<1> local(128);
        if (direction == 0) {
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> shared(128, cgh);
                cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
                    k_turbo_wht_128<0>(src, nb1, nb2, nb3, dst, scale_inv, ne0, ne1, ne2, groups_per_row, shared, item);
                });
            });
        } else {
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> shared(128, cgh);
                cgh.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
                    k_turbo_wht_128<1>(src, nb1, nb2, nb3, dst, scale_inv, ne0, ne1, ne2, groups_per_row, shared, item);
                });
            });
        }
    }

    if (tail_size > 0) {
        constexpr int block_sz = 256;
        const int64_t total    = n_rows * tail_size;
        const int64_t grid     = (total + block_sz - 1) / block_sz;
        const int64_t tail_off = groups_per_row * 128;
        stream->parallel_for(sycl::nd_range<1>(grid * block_sz, block_sz), [=](sycl::nd_item<1> item) {
            k_turbo_wht_copy_tail(src, nb1, nb2, nb3, dst, ne0, ne1, ne2, tail_off, tail_size, n_rows, item);
        });
    }
}

bool ggml_sycl_turbo_wht_supported(const ggml_tensor * op) {
    const ggml_tensor * src = op->src[0];
    if (src == nullptr || src->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(op)) {
        return false;
    }
    const int direction  = op->op_params[0];
    const int group_size = op->op_params[1];
    if ((direction != 0 && direction != 1) || group_size != 128) {
        return false;
    }
    const ggml_tensor * scale = op->src[1];
    if (scale != nullptr && (scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || scale->ne[0] < group_size)) {
        return false;
    }
    return src->ne[0] > 0;
}

void ggml_sycl_turbo_wht(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(ggml_sycl_turbo_wht_supported(dst));

    // op_params is an int32_t array: direction is [0], group_size [1] (ggml.c ggml_turbo_wht).
    const int direction = dst->op_params[0];

    const ggml_tensor * scale_tensor = dst->src[1];
    const float * scale_inv = scale_tensor ? (const float *) scale_tensor->data : nullptr;

    launch_turbo_wht(ctx.stream(), (const char *) src->data, src->nb[1], src->nb[2], src->nb[3], (float *) dst->data, scale_inv,
                     src->ne[0], src->ne[1], src->ne[2], src->ne[3], direction);
}

void ggml_sycl_turbo_rotate_forward_to(ggml_backend_sycl_context & ctx, const ggml_tensor * q, float * dst) {
    GGML_ASSERT(q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) && q->ne[0] % 128 == 0);
    launch_turbo_wht(ctx.stream(), (const char *) q->data, q->nb[1], q->nb[2], q->nb[3], dst, /*scale_inv=*/nullptr,
                     q->ne[0], q->ne[1], q->ne[2], q->ne[3], /*direction=*/0);
}

void ggml_sycl_turbo_rotate_inverse_inplace(ggml_backend_sycl_context & ctx, ggml_tensor * t) {
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] % 128 == 0);
    // In place is safe: every 128-group is read into local memory by one work-group before any of it is written back.
    launch_turbo_wht(ctx.stream(), (const char *) t->data, t->nb[1], t->nb[2], t->nb[3], (float *) t->data, /*scale_inv=*/nullptr,
                     t->ne[0], t->ne[1], t->ne[2], t->ne[3], /*direction=*/1);
}
