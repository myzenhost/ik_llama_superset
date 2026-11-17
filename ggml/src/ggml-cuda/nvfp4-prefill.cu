// Native block-scaled FP4 (NVFP4 W4A4) prefill GEMM, see nvfp4-prefill.cuh for the contract.
// Row S69-FASTLLM-PICK-SM120-NATIVE-NVFP4-FP8-PREFILL-2026-10-05-01.

#include "nvfp4-prefill.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
#define GGML_CUDA_NVFP4_PREFILL_AVAILABLE 1
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif

// Counters for the "was it FIRED" question (ggml-cuda.h: ggml_backend_cuda_nvfp4_prefill_stats). A path that is
// compiled and enabled but never reached looks identical to one that works, unless something counts.
static std::atomic<unsigned long long> g_nvfp4_prefill_handled{0};
static std::atomic<unsigned long long> g_nvfp4_prefill_declined{0};
// Weight-repack accounting: how many calls reused a cached repacked weight, how many ran the repack kernel, and how many
// plans were timed. A cache that is enabled but never hit looks identical to one that works unless something counts.
static std::atomic<unsigned long long> g_nvfp4_prefill_cache_hits{0};
static std::atomic<unsigned long long> g_nvfp4_prefill_repacks{0};
static std::atomic<unsigned long long> g_nvfp4_prefill_tuned_plans{0};

extern "C" void ggml_backend_cuda_nvfp4_prefill_stats(unsigned long long * handled, unsigned long long * declined) {
    if (handled)  *handled  = g_nvfp4_prefill_handled.load();
    if (declined) *declined = g_nvfp4_prefill_declined.load();
}

extern "C" void ggml_backend_cuda_nvfp4_prefill_cache_stats(unsigned long long * cache_hits, unsigned long long * repacks,
        unsigned long long * tuned_plans) {
    if (cache_hits)  *cache_hits  = g_nvfp4_prefill_cache_hits.load();
    if (repacks)     *repacks     = g_nvfp4_prefill_repacks.load();
    if (tuned_plans) *tuned_plans = g_nvfp4_prefill_tuned_plans.load();
}

#ifdef GGML_CUDA_NVFP4_PREFILL_AVAILABLE

// ---------------------------------------------------------------------------------------------------------------
// Scale layout. cuBLASLt's VEC16_UE4M3 block-scale operand is stored as 128-row x 4-group tiles of 512 bytes: tile
// (row/128, group/4), and inside a tile byte (row%32)*16 + ((row%128)/32)*4 + group%4. `groups` is K/16.
// ---------------------------------------------------------------------------------------------------------------
static __device__ __forceinline__ size_t nvfp4_scale_index(int64_t row, int64_t group, int64_t groups) {
    return (size_t(row >> 7) * size_t(groups >> 2) + size_t(group >> 2)) * 512
         + size_t(row & 31) * 16 + size_t((row & 127) >> 5) * 4 + size_t(group & 3);
}

// One thread = one block_nvfp4 (64 values of one weight row). ggml packs a 16-value sub-block as 8 bytes whose low
// nibble is value j and high nibble value j+8; cuBLASLt wants value 2b in the low nibble and 2b+1 in the high one.
static __global__ void k_nvfp4_repack_weights(const block_nvfp4 * __restrict__ w, uint8_t * __restrict__ qw,
        uint8_t * __restrict__ sw, const int64_t n_blocks, const int nblk_row) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_blocks) {
        return;
    }
    const int64_t row = gid / nblk_row;
    const int     blk = (int) (gid % nblk_row);
    const block_nvfp4 * b = w + gid;

    uint8_t out[32];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const uint8_t * q = b->qs + s * 8;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[s * 8 + i]     = (uint8_t) ((q[2 * i]     & 0x0F) | ((q[2 * i + 1] & 0x0F) << 4));
            out[s * 8 + 4 + i] = (uint8_t) ((q[2 * i]     >> 4)   | ((q[2 * i + 1] >> 4)   << 4));
        }
    }
    uint4 * dstq = reinterpret_cast<uint4 *>(qw + (size_t) row * (size_t) nblk_row * 32 + (size_t) blk * 32);
    dstq[0] = *reinterpret_cast<const uint4 *>(out);
    dstq[1] = *reinterpret_cast<const uint4 *>(out + 16);

    uint32_t sc = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        uint32_t v = b->d[s];
        v = (v == 0x7F || v == 0xFF) ? 0u : v; // NaN scale reads as 0, as ggml_cuda_ue4m3_to_fp32 does
        sc |= v << (8 * s);
    }
    *reinterpret_cast<uint32_t *>(sw + nvfp4_scale_index(row, (int64_t) blk * 4, (int64_t) nblk_row * 4)) = sc;
}

static __device__ __forceinline__ float nvfp4_block_max(float v) {
    __shared__ float warp_max[8];
    for (int d = 16; d > 0; d >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, d));
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) {
        warp_max[wid] = v;
    }
    __syncthreads();
    v = lane < 8 ? warp_max[lane] : 0.0f;
    for (int d = 4; d > 0; d >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, d));
    }
    __syncthreads();
    return __shfl_sync(0xffffffff, v, 0);
}

// Activation quantiser. One CUDA block (256 threads) per row. Row r >= n_rows is padding: zero data, zero scales.
// x_row = qx * sx * gx per 16-group: gx = rowmax / (448*6) keeps every sx = group_max/(6*gx) inside E4M3's 448.
static __global__ void k_nvfp4_quant_act(const float * __restrict__ x, const int64_t ldx, uint8_t * __restrict__ qx,
        uint8_t * __restrict__ sx, float * __restrict__ gx, const int n_rows, const int K) {
    const int row = blockIdx.x;
    const int groups = K / 16;
    if (row >= n_rows) {
        for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
            qx[(size_t) row * (K / 2) + i] = 0;
        }
        for (int g = threadIdx.x; g < groups; g += blockDim.x) {
            sx[nvfp4_scale_index(row, g, groups)] = 0;
        }
        if (threadIdx.x == 0) {
            gx[row] = 0.0f;
        }
        return;
    }
    const float * xr = x + (size_t) row * (size_t) ldx;

    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        m = fmaxf(m, fabsf(xr[k]));
    }
    m = nvfp4_block_max(m);
    const float g = fmaxf(m / (448.0f * 6.0f), 1.0e-30f);
    if (threadIdx.x == 0) {
        gx[row] = g;
    }
    for (int gi = threadIdx.x; gi < groups; gi += blockDim.x) {
        float v[16];
        float mg = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] = xr[gi * 16 + j];
            mg = fmaxf(mg, fabsf(v[j]));
        }
        const __nv_fp8_e4m3 sf(mg / (6.0f * g));
        const float denom = float(sf) * g;
        const float inv   = denom > 0.0f ? 1.0f / denom : 0.0f;
        uint64_t packed = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint64_t b = __nv_cvt_float2_to_fp4x2(make_float2(v[2 * j] * inv, v[2 * j + 1] * inv), __NV_E2M1, cudaRoundNearest);
            packed |= (b & 0xFFull) << (8 * j);
        }
        reinterpret_cast<uint64_t *>(qx + (size_t) row * (K / 2))[gi] = packed;
        sx[nvfp4_scale_index(row, gi, groups)] = sf.__x;
    }
}

// dst[m][n] = src[m][n] * gx[m]   (src may equal dst)
static __global__ void k_nvfp4_scale_rows(const float * __restrict__ src, float * __restrict__ dst, const float * __restrict__ gx, const int64_t N, const int M) {
    const int row = blockIdx.y;
    const int64_t n = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (row >= M || n >= N) {
        return;
    }
    float4 v = *reinterpret_cast<const float4 *>(src + (size_t) row * (size_t) N + n);
    const float g = gx[row];
    v.x *= g; v.y *= g; v.z *= g; v.w *= g;
    *reinterpret_cast<float4 *>(dst + (size_t) row * (size_t) N + n) = v;
}

// ---------------------------------------------------------------------------------------------------------------
// cuBLASLt plan cache. Key (device, padded rows, N, K). A plan owns its descriptors and the first heuristic algo.
// ---------------------------------------------------------------------------------------------------------------
struct nvfp4_plan {
    cublasLtMatmulDesc_t   op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t   algo{};
    std::vector<cublasLtMatmulAlgo_t> cands; // every algorithm the heuristic returned as usable, best-guess first
    bool tuned = false;                      // the timed pick over `cands` has run (or was ruled out)
    ~nvfp4_plan() {
        if (op) cublasLtMatmulDescDestroy(op);
        if (a)  cublasLtMatrixLayoutDestroy(a);
        if (b)  cublasLtMatrixLayoutDestroy(b);
        if (c)  cublasLtMatrixLayoutDestroy(c);
    }
};

static constexpr size_t NVFP4_WORKSPACE_BYTES = 32ull << 20;

struct nvfp4_lt_state {
    std::mutex mtx;
    cublasLtHandle_t handle = nullptr;
    std::map<std::tuple<int64_t, int64_t, int64_t>, std::unique_ptr<nvfp4_plan>> plans; // value null = known-bad shape
};

static nvfp4_lt_state & nvfp4_state_for_device(int device) {
    static std::mutex mtx;
    static std::map<int, std::unique_ptr<nvfp4_lt_state>> states;
    std::lock_guard<std::mutex> lock(mtx);
    auto & s = states[device];
    if (!s) {
        s = std::make_unique<nvfp4_lt_state>();
    }
    return *s;
}

static nvfp4_plan * nvfp4_get_plan(nvfp4_lt_state & st, int64_t Mp, int64_t N, int64_t K, const void * placeholder) {
    const auto key = std::make_tuple(Mp, N, K);
    auto it = st.plans.find(key);
    if (it != st.plans.end()) {
        return it->second.get();
    }
    auto fail = [&]() -> nvfp4_plan * {
        st.plans.emplace(key, nullptr);
        return nullptr;
    };
    if (!st.handle && cublasLtCreate(&st.handle) != CUBLAS_STATUS_SUCCESS) {
        st.handle = nullptr;
        return nullptr; // not cached: a later call may succeed
    }
    auto p = std::make_unique<nvfp4_plan>();
    if (cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return fail();
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) != CUBLAS_STATUS_SUCCESS) return fail();
    const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)) != CUBLAS_STATUS_SUCCESS) return fail();
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)) != CUBLAS_STATUS_SUCCESS) return fail();
    // the heuristic wants a complete descriptor, scale pointers included; real ones are set per call
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &placeholder, sizeof(placeholder)) != CUBLAS_STATUS_SUCCESS) return fail();
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &placeholder, sizeof(placeholder)) != CUBLAS_STATUS_SUCCESS) return fail();
    // y[Mp x N] = x[Mp x K] * w[N x K]^T, written column-major as C(N x Mp) = A^T(N x K) * B(K x Mp)
    if (cublasLtMatrixLayoutCreate(&p->a, CUDA_R_4F_E2M1, K, N,  K) != CUBLAS_STATUS_SUCCESS) return fail();
    if (cublasLtMatrixLayoutCreate(&p->b, CUDA_R_4F_E2M1, K, Mp, K) != CUBLAS_STATUS_SUCCESS) return fail();
    if (cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32F,     N, Mp, N) != CUBLAS_STATUS_SUCCESS) return fail();

    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return fail();
    size_t ws = NVFP4_WORKSPACE_BYTES;
    cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[8];
    int found = 0;
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulAlgoGetHeuristic(st.handle, p->op, p->a, p->b, p->c, p->c, pref, 8, heur, &found);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    if (status != CUBLAS_STATUS_SUCCESS) return fail();
    for (int i = 0; i < found; ++i) {
        if (heur[i].state == CUBLAS_STATUS_SUCCESS) { p->cands.push_back(heur[i].algo); }
    }
    if (p->cands.empty()) return fail();
    p->algo = p->cands[0];
    nvfp4_plan * raw = p.get();
    st.plans.emplace(key, std::move(p));
    return raw;
}

static bool nvfp4_prefill_env_enabled() {
    const char * e = getenv("GGML_CUDA_NVFP4_PREFILL");
    return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
}

static int64_t nvfp4_prefill_min_rows() {
    const char * e = getenv("GGML_CUDA_NVFP4_PREFILL_MIN_M");
    const long v = e ? atol(e) : 0;
    return v > 0 ? v : 128;
}

// ---------------------------------------------------------------------------------------------------------------
// Timed algorithm pick. cublasLtMatmulAlgoGetHeuristic returns a ranked list, but the ranking is a model, not a
// measurement; the first entry is not always the fastest on a given shape. The first call per plan times each usable
// candidate on the real operands (one warm-up and three timed runs each) and keeps the fastest. GGML_CUDA_NVFP4_PREFILL_AUTOTUNE=0
// restores "take the heuristic's first". A stream that is being captured into a CUDA graph is never timed (an event
// sync inside a capture is an error); the plan stays untuned and is timed on the next uncaptured call.
// ---------------------------------------------------------------------------------------------------------------
static bool nvfp4_autotune_enabled() {
    const char * e = getenv("GGML_CUDA_NVFP4_PREFILL_AUTOTUNE");
    return !(e && (e[0] == '0' || e[0] == 'f' || e[0] == 'F'));
}

static void nvfp4_tune_plan(cublasLtHandle_t handle, nvfp4_plan & p, const void * qw, const void * qx, float * out,
        void * workspace, cudaStream_t stream, int64_t Mp, int64_t N, int64_t K) {
    if (p.tuned) {
        return;
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return; // not tuned: try again on an uncaptured call
    }
    p.tuned = true;
    if (!nvfp4_autotune_enabled() || p.cands.size() < 2) {
        return;
    }
    cudaEvent_t begin = nullptr, end = nullptr;
    if (cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
        cudaGetLastError();
        if (begin) cudaEventDestroy(begin);
        return;
    }
    const float alpha = 1.0f, beta = 0.0f;
    float best = 1.0e30f, worst = 0.0f;
    int best_i = 0;
    for (size_t i = 0; i < p.cands.size(); ++i) {
        auto run = [&]() {
            return cublasLtMatmul(handle, p.op, &alpha, qw, p.a, qx, p.b, &beta, out, p.c, out, p.c, &p.cands[i],
                    workspace, NVFP4_WORKSPACE_BYTES, stream);
        };
        if (run() != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        cudaEventRecord(begin, stream);
        bool ok = true;
        for (int r = 0; r < 3 && ok; ++r) {
            ok = run() == CUBLAS_STATUS_SUCCESS;
        }
        cudaEventRecord(end, stream);
        if (cudaEventSynchronize(end) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, begin, end);
        if (ok) {
            worst = std::max(worst, ms);
            if (ms < best) { best = ms; best_i = (int) i; }
        }
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    p.algo = p.cands[best_i];
    g_nvfp4_prefill_tuned_plans.fetch_add(1);
    if (getenv("GGML_CUDA_NVFP4_PREFILL_LOG")) {
        fprintf(stderr, "nvfp4 prefill autotune Mp=%lld N=%lld K=%lld: picked %d of %d, best %.4f ms worst %.4f ms (3 runs)\n",
                (long long) Mp, (long long) N, (long long) K, best_i, (int) p.cands.size(), best / 3.0f, worst / 3.0f);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Persistent repacked-weight cache. The weight repack (ggml block layout -> cuBLASLt operand layout) depends only on
// the weight, so doing it on every call re-reads and re-writes the whole weight each time a prompt chunk goes through a
// layer. The first call for a weight tensor repacks it once into device memory that this file owns; later calls reuse it.
//
// Cost: the repacked copy is the same size as the weight (36 bytes per 64 values either way), so a cached tensor doubles
// its VRAM. GGML_CUDA_NVFP4_PREFILL_CACHE_MB bounds the total per device (default 2048; 0 turns the cache off). A weight that
// does not fit keeps repacking per call. Only tensors in a WEIGHTS-usage buffer are cached: a compute buffer is reused by the
// allocator for different data, so a pointer match there would not mean the same contents.
//
// Staleness: an entry is dropped when the buffer holding the weight is freed, written (set/memset/copy-into) or cleared
// (nvfp4_prefill_cache_invalidate, called from the CUDA buffer interface).
// ---------------------------------------------------------------------------------------------------------------
struct nvfp4_wentry {
    uint8_t * qw = nullptr;
    uint8_t * sw = nullptr;
    size_t    bytes = 0;
    size_t    src_bytes = 0; // extent of the source weight, for range invalidation
};

struct nvfp4_wcache {
    std::mutex mtx;
    std::map<std::tuple<int, uintptr_t, int64_t, int64_t>, nvfp4_wentry> map; // (device, weight data pointer, N, K)
    std::map<int, size_t> used;                                               // bytes held per device
};

// Intentionally never destroyed: at process exit the CUDA context may already be gone and a cudaFree would only error.
static nvfp4_wcache & nvfp4_wcache_get() {
    static nvfp4_wcache * c = new nvfp4_wcache();
    return *c;
}

static size_t nvfp4_cache_budget_bytes() {
    const char * e = getenv("GGML_CUDA_NVFP4_PREFILL_CACHE_MB");
    if (!e || !*e) {
        return (size_t) 2048 << 20;
    }
    const long long v = atoll(e);
    return v > 0 ? (size_t) v << 20 : 0;
}

static void nvfp4_wentry_free(int device, nvfp4_wentry & e) {
    int prev = 0;
    cudaGetDevice(&prev);
    cudaSetDevice(device);
    cudaFree(e.qw);
    cudaFree(e.sw);
    cudaSetDevice(prev);
    e.qw = e.sw = nullptr;
}

void ggml_cuda_nvfp4_prefill_cache_invalidate(const void * base, size_t size) {
    nvfp4_wcache & c = nvfp4_wcache_get();
    std::lock_guard<std::mutex> lock(c.mtx);
    if (c.map.empty()) {
        return;
    }
    const uintptr_t lo = (uintptr_t) base, hi = lo + size;
    for (auto it = c.map.begin(); it != c.map.end();) {
        const uintptr_t p = std::get<1>(it->first);
        if (p < hi && p + it->second.src_bytes > lo) {
            const int device = std::get<0>(it->first);
            c.used[device] -= std::min(c.used[device], it->second.bytes);
            nvfp4_wentry_free(device, it->second);
            it = c.map.erase(it);
        } else {
            ++it;
        }
    }
}

bool ggml_cuda_mul_mat_nvfp4_prefill(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_NVFP4 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = src0->ne[0], N = src0->ne[1], M = src1->ne[1];
    if (M < nvfp4_prefill_min_rows() || !nvfp4_prefill_env_enabled()) {
        return false; // the common decode case exits here, before any counter is touched
    }
    // From here on a refusal is counted: the flag is on and the batch is prefill-sized.
    auto decline = [&]() { g_nvfp4_prefill_declined.fetch_add(1); return false; };

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || (cc / 100 != 10 && cc / 100 != 12)) {
        return decline(); // block-scaled FP4 tensor cores: sm_100 / sm_120 only
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            !ggml_is_contiguous(src0) || !ggml_is_contiguous(dst) || src1->nb[0] != sizeof(float) ||
            src1->nb[1] % sizeof(float) != 0 || src1->ne[0] != K || dst->ne[0] != N || dst->ne[1] != M) {
        return decline();
    }
    // cuBLASLt operand shape rules for the FP4 path: K % 64 so every weight row is whole blocks (and a whole number of
    // 4-group scale tiles), N % 128 so the weight scale tiles are whole.
    if (K % 64 != 0 || N % 128 != 0) {
        return decline();
    }
    if ((uintptr_t) src0->data % 16 != 0 || (uintptr_t) dst->data % 16 != 0 || (uintptr_t) src1->data % 4 != 0) {
        return decline();
    }

    const int id = ctx.device;
    cudaStream_t stream = ctx.stream();
    const int64_t chunk_max = 4096;

    nvfp4_lt_state & st = nvfp4_state_for_device(id);
    std::lock_guard<std::mutex> lock(st.mtx);

    const int64_t Kb = K / 64;
    const size_t qw_bytes = (size_t) N * (size_t) K / 2;
    const size_t sw_bytes = (size_t) N * (size_t) K / 16;
    auto launch_repack = [&](uint8_t * qw_dst, uint8_t * sw_dst) {
        const int64_t n_blocks = N * Kb;
        const int threads = 128;
        k_nvfp4_repack_weights<<<(unsigned) ((n_blocks + threads - 1) / threads), threads, 0, stream>>>(
            (const block_nvfp4 *) src0->data, qw_dst, sw_dst, n_blocks, (int) Kb);
        CUDA_CHECK(cudaGetLastError());
        g_nvfp4_prefill_repacks.fetch_add(1);
    };

    // Repacked weight: from the persistent cache when this tensor lives in a WEIGHTS buffer and fits the budget, else a
    // per-call pool copy (the pre-cache behaviour).
    const uint8_t * qw_ptr = nullptr;
    const uint8_t * sw_ptr = nullptr;
    ggml_cuda_pool_alloc<uint8_t> qw_tmp(ctx.pool(id));
    ggml_cuda_pool_alloc<uint8_t> sw_tmp(ctx.pool(id));
    const size_t cache_budget = nvfp4_cache_budget_bytes();
    // CAPABILITY-GENUINELY-NA: CUDA-kernel cache hygiene, no model capability involved.
    // A stream being captured into a CUDA graph bakes every pointer it uses into the graph, and the graph is replayed later
    // without any of this host code running: a cache entry freed after the capture (weight rewritten or buffer freed) would
    // leave the replayed graph reading freed memory. So a captured call never uses the cache; it repacks into the pool like
    // every other pool buffer in a captured graph (the same address-baking convention the rest of this backend relies on).
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess) {
        cudaGetLastError();
        cap = cudaStreamCaptureStatusActive;
    }
    // GGML_CUDA_NVFP4_PREFILL_CACHE_IN_GRAPHS=1 restores the unguarded behaviour (cache hits inside a capture); it exists so the
    // guard can be A/B tested and costs one getenv per call.
    const char * in_graphs_env = getenv("GGML_CUDA_NVFP4_PREFILL_CACHE_IN_GRAPHS");
    const bool cache_in_graphs = in_graphs_env && *in_graphs_env && *in_graphs_env != '0';
    if ((cap == cudaStreamCaptureStatusNone || cache_in_graphs) && cache_budget > 0 && src0->buffer &&
            ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        nvfp4_wcache & wc = nvfp4_wcache_get();
        std::lock_guard<std::mutex> wlock(wc.mtx);
        const auto wkey = std::make_tuple(id, (uintptr_t) src0->data, N, K);
        auto it = wc.map.find(wkey);
        if (it != wc.map.end()) {
            qw_ptr = it->second.qw;
            sw_ptr = it->second.sw;
            g_nvfp4_prefill_cache_hits.fetch_add(1);
        } else if (wc.used[id] + qw_bytes + sw_bytes <= cache_budget) {
            // Populating needs a stream sync (the entry becomes visible to every stream), which a stream being captured into a
            // CUDA graph cannot do; such a call repacks into the pool and the entry is created on the next uncaptured call.
            if (cap == cudaStreamCaptureStatusNone) {
                nvfp4_wentry e;
                e.bytes = qw_bytes + sw_bytes;
                e.src_bytes = (size_t) N * (size_t) Kb * sizeof(block_nvfp4);
                if (cudaMalloc((void **) &e.qw, qw_bytes) == cudaSuccess && cudaMalloc((void **) &e.sw, sw_bytes) == cudaSuccess) {
                    launch_repack(e.qw, e.sw);
                    if (cudaStreamSynchronize(stream) == cudaSuccess) {
                        wc.used[id] += e.bytes;
                        qw_ptr = e.qw;
                        sw_ptr = e.sw;
                        wc.map.emplace(wkey, e);
                    } else {
                        cudaGetLastError();
                        nvfp4_wentry_free(id, e);
                    }
                } else {
                    cudaGetLastError(); // out of memory for the cache copy: keep the per-call path
                    nvfp4_wentry_free(id, e);
                }
            }
        }
    }
    if (!qw_ptr) {
        qw_tmp.alloc(qw_bytes);
        sw_tmp.alloc(sw_bytes);
        launch_repack(qw_tmp.get(), sw_tmp.get());
        qw_ptr = qw_tmp.get();
        sw_ptr = sw_tmp.get();
    }

    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool(id), NVFP4_WORKSPACE_BYTES);

    for (int64_t m0 = 0; m0 < M; m0 += chunk_max) {
        const int64_t mc = std::min(chunk_max, M - m0);
        const int64_t Mp = (mc + 127) / 128 * 128;

        nvfp4_plan * plan = nvfp4_get_plan(st, Mp, N, K, qw_ptr);
        if (!plan) {
            // earlier chunks of this call already wrote their rows of dst; the fallback recomputes ALL rows of dst
            return decline();
        }

        ggml_cuda_pool_alloc<uint8_t> qx(ctx.pool(id), (size_t) Mp * (size_t) K / 2);
        ggml_cuda_pool_alloc<uint8_t> sx(ctx.pool(id), (size_t) Mp * (size_t) K / 16);
        ggml_cuda_pool_alloc<float>   gx(ctx.pool(id), (size_t) Mp);

        const float * x_chunk = (const float *) ((const char *) src1->data + (size_t) m0 * src1->nb[1]);
        k_nvfp4_quant_act<<<(unsigned) Mp, 256, 0, stream>>>(x_chunk, (int64_t) (src1->nb[1] / sizeof(float)),
                qx.get(), sx.get(), gx.get(), (int) mc, (int) K);
        CUDA_CHECK(cudaGetLastError());

        const void * pa_scale = sw_ptr;
        const void * pb_scale = sx.get();
        if (cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pa_scale, sizeof(pa_scale)) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(plan->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pb_scale, sizeof(pb_scale)) != CUBLAS_STATUS_SUCCESS) {
            return decline();
        }
        float * dst_chunk = (float *) ((char *) dst->data + (size_t) m0 * dst->nb[1]);
        // the matmul writes Mp rows; dst only holds mc, so a short chunk goes through a staging buffer
        ggml_cuda_pool_alloc<float> stage(ctx.pool(id));
        float * mm_out = dst_chunk;
        if (Mp != mc) {
            stage.alloc((size_t) Mp * (size_t) N);
            mm_out = stage.get();
        }
        nvfp4_tune_plan(st.handle, *plan, qw_ptr, qx.get(), mm_out, workspace.get(), stream, Mp, N, K);
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t status = cublasLtMatmul(st.handle, plan->op, &alpha, qw_ptr, plan->a, qx.get(), plan->b, &beta,
                mm_out, plan->c, mm_out, plan->c, &plan->algo, workspace.get(), NVFP4_WORKSPACE_BYTES, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return decline();
        }
        // dst[m][n] = mm_out[m][n] * gx[m]: undoes the per-row activation scale (and copies out of the staging buffer)
        k_nvfp4_scale_rows<<<dim3((unsigned) ((N / 4 + 255) / 256), (unsigned) mc), 256, 0, stream>>>(mm_out, dst_chunk, gx.get(), N, (int) mc);
        CUDA_CHECK(cudaGetLastError());
    }
    g_nvfp4_prefill_handled.fetch_add(1);
    return true;
}

#else // !GGML_CUDA_NVFP4_PREFILL_AVAILABLE

bool ggml_cuda_mul_mat_nvfp4_prefill(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    return false;
}

void ggml_cuda_nvfp4_prefill_cache_invalidate(const void *, size_t) {}

#endif
