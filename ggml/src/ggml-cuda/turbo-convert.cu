// TurboQuant KV -> f16 / f32 block dequantisation for the CUDA backend.
//
// WHY THIS FILE EXISTS. The tensor-core (MMA) flash-attention kernels read K/V as f16 and launch_fattn_mma
// (fattn-mma-f16.cuh) converts a quantised K/V with ggml_get_to_fp16_cuda(type) before the kernel runs. Until now
// no turbo type had such a converter, which is half of why turbo KV could only reach the vec kernels (the other
// half is the Q rotation, see ggml_cuda_fattn_turbo_via_mma in fattn.cu).
//
// WHAT IT DECODES. A turbo block stores its 128 elements in the ROTATED basis: value = centroid[idx] * norm. This
// file does exactly that and nothing else -- the same per-element helpers the vec kernels use
// (turbo{2,3,4}_dequant_element in turbo-quant.cuh) -- and it does NOT undo the rotation. That is deliberate: the
// attention identity (R q).(R k) = q.k and sum_i p_i (R v_i) = R sum_i p_i v_i needs K and V left in the rotated
// basis, with Q rotated forward and the finished output rotated back. A converter that un-rotated here would
// double-rotate. The CPU dequantiser (dequantize_row_turbo*_0 in ggml-turbo-quant.c) has the same contract, which
// tests/test-turbo3-roundtrip.cpp pins by applying turbo_cpu_fwht_inverse itself.
//
// One CUDA block of 128 threads decodes one turbo block (thread t -> element t), so there is no cross-thread
// state and the result does not depend on the launch shape. The centroid tables are the ones in turbo-quant.cuh,
// checked against the CPU and Vulkan copies by scripts/check-turbo-tables-consistent.py -- there is no copy here.
//
// DESIGN SOURCE: TheTom/llama-cpp-turboquant bcb85fc3ae85 ggml/src/ggml-cuda/convert.cu dequantize_block_cuda<QK_TURBO*,...>
// (MIT) was read for the contract; this is a smaller one-block-per-turbo-block rewrite against OUR launcher signature
// (to_t_cuda_t: x, y, nrows, n_per_row, stream).
//
// GATE-CLAIM-OK: a kernel source file; it declares no resource unavailable.

#include "convert.cuh"
#include "turbo-quant.cuh"

template <typename dst_t>
static __global__ void k_dequantize_turbo3_0(const block_turbo3_0 * __restrict__ x, dst_t * __restrict__ y) {
    const block_turbo3_0 * b = x + blockIdx.x;
    const float norm = __half2float(b->norm);
    y[(int64_t) blockIdx.x * QK_TURBO3 + threadIdx.x] = ggml_cuda_cast<dst_t>(turbo3_dequant_element(b, (int) threadIdx.x, norm));
}

template <typename dst_t>
static __global__ void k_dequantize_turbo2_0(const block_turbo2_0 * __restrict__ x, dst_t * __restrict__ y) {
    const block_turbo2_0 * b = x + blockIdx.x;
    const float norm = __half2float(b->norm);
    y[(int64_t) blockIdx.x * QK_TURBO2 + threadIdx.x] = ggml_cuda_cast<dst_t>(turbo2_dequant_element(b, (int) threadIdx.x, norm));
}

template <typename dst_t>
static __global__ void k_dequantize_turbo4_0(const block_turbo4_0 * __restrict__ x, dst_t * __restrict__ y) {
    const block_turbo4_0 * b = x + blockIdx.x;
    const float norm = __half2float(b->norm);
    y[(int64_t) blockIdx.x * QK_TURBO4 + threadIdx.x] = ggml_cuda_cast<dst_t>(turbo4_dequant_element(b, (int) threadIdx.x, norm));
}

static_assert(QK_TURBO3 == 128 && QK_TURBO2 == 128 && QK_TURBO4 == 128, "one 128-thread CUDA block decodes one turbo block");

template <typename dst_t>
static void dequantize_turbo3_0_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    GGML_ASSERT(k % QK_TURBO3 == 0);
    if (k > 0) k_dequantize_turbo3_0<dst_t><<<(unsigned) (k / QK_TURBO3), QK_TURBO3, 0, stream>>>((const block_turbo3_0 *) vx, y);
}

template <typename dst_t>
static void dequantize_turbo2_0_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    GGML_ASSERT(k % QK_TURBO2 == 0);
    if (k > 0) k_dequantize_turbo2_0<dst_t><<<(unsigned) (k / QK_TURBO2), QK_TURBO2, 0, stream>>>((const block_turbo2_0 *) vx, y);
}

template <typename dst_t>
static void dequantize_turbo4_0_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t nrows, const int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    GGML_ASSERT(k % QK_TURBO4 == 0);
    if (k > 0) k_dequantize_turbo4_0<dst_t><<<(unsigned) (k / QK_TURBO4), QK_TURBO4, 0, stream>>>((const block_turbo4_0 *) vx, y);
}

// nullptr for every non-turbo type, so convert.cu can fall through to its own table.
to_fp16_cuda_t ggml_cuda_get_turbo_to_fp16(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO3_0: return dequantize_turbo3_0_cuda<half>;
        case GGML_TYPE_TURBO2_0: return dequantize_turbo2_0_cuda<half>;
        case GGML_TYPE_TURBO4_0: return dequantize_turbo4_0_cuda<half>;
        default:                 return nullptr;
    }
}

to_fp32_cuda_t ggml_cuda_get_turbo_to_fp32(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO3_0: return dequantize_turbo3_0_cuda<float>;
        case GGML_TYPE_TURBO2_0: return dequantize_turbo2_0_cuda<float>;
        case GGML_TYPE_TURBO4_0: return dequantize_turbo4_0_cuda<float>;
        default:                 return nullptr;
    }
}
