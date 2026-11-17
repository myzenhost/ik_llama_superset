#pragma once

// Exact per-row top-k SELECTION (radix select) for the DSA lightning indexer, replacing "full sort of n_kv scores,
// keep the first k".
//
// RESEARCH-DONE: exllamav3 exllamav3_ext/dsa_topk.cu (MIT, fa714d0a) and vLLM csrc/libtorch_stable/persistent_topk.cuh
// (Apache-2.0) read for the row; CCCL cub::DeviceTopK (single-segment) considered and not used for the batched case, see below.
//
// WHAT. For each row of `score` (n floats) pick the k largest and write their column indices, ordered by score
// descending, equal scores by index ascending -- the order a STABLE descending sort would give. The old path
// (indexer_topk.cu: argsort_f32_i32_cuda_cub, then k_copy_topk) sorted all n and copied the first k.
// Cost: 4 histogram passes over the row (8 bits each) + one compaction pass + a k-element bitonic sort, instead of a
// radix sort of n key/value pairs.
//
// ONE SOURCE, TWO EXECUTORS. The kernel body (topk_select_row) is written against a small block context
// (tid / nthreads / sync / shared atomics / exclusive block scan) so the same code runs on the GPU (GpuBlock below,
// compiled only by nvcc) and on the CPU under std::thread (tests/test-topk-select.cpp), where it is checked against a
// stable-sort reference including ties, -inf, NaN and every (n, k) edge. The GPU launch itself is exercised by the
// parent's GPU run (see the row), not by this file.
//
// NUANCES THE ROW TEXT DID NOT SAY:
//   1. Ordered keys: float bits are mapped to an unsigned key whose order is the float order (flip all bits of a
//      negative, the sign bit of a positive). NaN is mapped to key 0, i.e. it ranks BELOW -inf and is selected only if
//      fewer than k non-NaN values exist. cub's float twiddle would rank a positive NaN ABOVE +inf; scores here are
//      relu(...)*w + mask sums and are never NaN in practice, so the choice only matters for malformed input.
//      -0.0 is mapped like +0.0 (they compare equal as floats, so they must tie, ordered by index).
//   2. The k-th value T is found by four MSB-first byte passes. After them `k_rem` is how many of the elements equal
//      to T must be taken; ties at T are resolved by LOWEST INDEX, which needs the compaction pass to walk the row in
//      index order (it does, in chunks of nthreads, with an exclusive scan per chunk).
//   3. Elements strictly above T are exactly k - k_rem in number by construction, so their output slots are known
//      before the pass and the two groups never collide.
//   4. The sort key is (ordered key << 32) | (0xFFFFFFFF - index): descending order on that u64 is "score desc, index
//      asc". Padding slots hold 0, which no real element can have (that would need key 0 AND index 0xFFFFFFFF).
//   5. vLLM clamps a negative or oversized row length before dispatch, because a padded slot with a negative length
//      reads as ~4e9 and hangs a cooperative barrier. Here the row length is a launch argument, not per-row data, and
//      the launcher refuses n < k, k > TOPK_SELECT_MAX_K and k <= 0 (falls back to the sort path).
//   6. cub::DeviceTopK handles ONE segment per call: batched prefill (hundreds of rows) would need one launch sequence
//      per row. A grid of nrows blocks does the whole batch in one launch, which is the case that dominates the
//      indexer cost.

#include <cstdint>
#include <cstring>

#define TOPK_SELECT_MAX_K 4096

#if defined(__CUDACC__)
#    define TOPK_HD __host__ __device__
#    define TOPK_DEV __device__
#else
#    define TOPK_HD
#    define TOPK_DEV
#endif

static TOPK_HD inline uint32_t topk_ordered_key(float x) {
    uint32_t b;
#if defined(__CUDA_ARCH__)
    b = __float_as_uint(x);
#else
    static_assert(sizeof(float) == 4, "float must be 32-bit");
    std::memcpy(&b, &x, 4);
#endif
    if ((b & 0x7fffffffu) > 0x7f800000u) {       // NaN (either sign)
        return 0u;
    }
    if (b == 0x80000000u) {                      // -0.0 compares equal to +0.0, so it must tie with it
        b = 0u;
    }
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// shared state of one block
struct topk_shared {
    uint32_t hist[256];
    uint32_t sel;        // selected byte of the current pass
    uint32_t k_rem;      // how many elements at the current prefix are still needed
    uint32_t gt_base;    // elements strictly above T seen so far
    uint32_t eq_base;    // elements equal to T seen so far
    uint64_t keys[TOPK_SELECT_MAX_K];
};

// Block context concept (both executors implement it):
//   int  tid() const; int nt() const;
//   void sync();                                    // block barrier incl. shared-memory visibility
//   uint32_t atomic_add(uint32_t * shared_ptr, uint32_t v);
//   uint32_t excl_scan(uint32_t v, uint32_t * total);   // exclusive sum over tid; *total = sum of all threads' v
template <class Blk>
static TOPK_HD void topk_select_row(Blk & blk, topk_shared & sh, const float * row, int n, int k,
                                    int * out, int k_pad /* power of two >= k */) {
    const int tid = blk.tid();
    const int nt  = blk.nt();

    uint32_t prefix = 0, known = 0, k_rem = (uint32_t) k;

    // ---- phase 1: radix select of the k-th largest key ------------------------------------------------------
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int b = tid; b < 256; b += nt) {
            sh.hist[b] = 0;
        }
        blk.sync();
        for (int i = tid; i < n; i += nt) {
            const uint32_t key = topk_ordered_key(row[i]);
            if ((key & known) == prefix) {
                blk.atomic_add(&sh.hist[(key >> shift) & 255u], 1u);
            }
        }
        blk.sync();
        if (tid == 0) {
            uint32_t cum = 0, chosen = 0;
            for (int b = 255; b >= 0; --b) {
                if (cum + sh.hist[b] >= k_rem) {
                    chosen = (uint32_t) b;
                    break;
                }
                cum += sh.hist[b];
            }
            sh.sel   = chosen;
            sh.k_rem = k_rem - cum;
        }
        blk.sync();
        prefix |= sh.sel << shift;
        known  |= 0xffu << shift;
        k_rem   = sh.k_rem;
        blk.sync(); // everyone has read sel/k_rem before the next pass rewrites shared state
    }
    const uint32_t T    = prefix;
    const uint32_t c_gt = (uint32_t) k - k_rem;

    // ---- phase 2: compaction in index order -----------------------------------------------------------------
    if (tid == 0) {
        sh.gt_base = 0;
        sh.eq_base = 0;
    }
    for (int j = tid; j < k_pad; j += nt) {
        sh.keys[j] = 0;
    }
    blk.sync();
    for (int base = 0; base < n; base += nt) {
        const int      i     = base + tid;
        const bool     valid = i < n;
        const uint32_t key   = valid ? topk_ordered_key(row[i]) : 0u;
        const bool     isgt  = valid && key > T;
        const bool     iseq  = valid && key == T;
        const uint32_t v     = (isgt ? 1u : 0u) | (iseq ? 0x10000u : 0u); // chunk counts stay < 65536 (nt <= 1024)
        uint32_t       total = 0;
        const uint32_t ex    = blk.excl_scan(v, &total);
        if (isgt) {
            const uint32_t slot = sh.gt_base + (ex & 0xffffu);
            sh.keys[slot] = ((uint64_t) key << 32) | (uint64_t) (0xffffffffu - (uint32_t) i);
        }
        if (iseq) {
            const uint32_t eq_rank = sh.eq_base + (ex >> 16);
            if (eq_rank < k_rem) {
                sh.keys[c_gt + eq_rank] = ((uint64_t) key << 32) | (uint64_t) (0xffffffffu - (uint32_t) i);
            }
        }
        blk.sync();
        if (tid == 0) {
            sh.gt_base += total & 0xffffu;
            sh.eq_base += total >> 16;
        }
        blk.sync();
    }

    // ---- phase 3: bitonic sort of the k selected (u64) descending --------------------------------------------
    for (int size = 2; size <= k_pad; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int j = tid; j < (k_pad >> 1); j += nt) {
                const int      pos     = 2 * j - (j & (stride - 1));
                const int      partner = pos + stride;
                const bool     desc    = (pos & size) == 0;   // this sub-sequence is built descending
                const uint64_t a = sh.keys[pos], b = sh.keys[partner];
                if (desc ? (a < b) : (a > b)) {
                    sh.keys[pos]     = b;
                    sh.keys[partner] = a;
                }
            }
            blk.sync();
        }
    }
    for (int j = tid; j < k; j += nt) {
        out[j] = (int) (0xffffffffu - (uint32_t) (sh.keys[j] & 0xffffffffull));
    }
}

static TOPK_HD inline int topk_pad_pow2(int k) {
    int p = 1;
    while (p < k) {
        p <<= 1;
    }
    return p;
}

// The launcher uses this to decide whether the select path applies.
static TOPK_HD inline bool topk_select_supported(int n, int k) {
    return k > 0 && k <= n && k <= TOPK_SELECT_MAX_K && n > 0;
}

#if defined(__CUDACC__)

#    include <cub/block/block_scan.cuh>

#    define TOPK_SELECT_THREADS 512

struct topk_gpu_block {
    typedef cub::BlockScan<uint32_t, TOPK_SELECT_THREADS> scan_t;
    typename scan_t::TempStorage & scan_tmp;
    __device__ int tid() const { return threadIdx.x; }
    __device__ int nt() const { return TOPK_SELECT_THREADS; }
    __device__ void sync() { __syncthreads(); }
    __device__ uint32_t atomic_add(uint32_t * p, uint32_t v) { return atomicAdd(p, v); }
    __device__ uint32_t excl_scan(uint32_t v, uint32_t * total) {
        uint32_t out;
        scan_t(scan_tmp).ExclusiveSum(v, out, *total);
        __syncthreads(); // the cub temp storage is reused by the next call
        return out;
    }
};

// one block per row; `score` rows are n floats apart, `out` rows are ld_out ints apart
static __global__ void k_topk_select(const float * __restrict__ score, int * __restrict__ out,
                                     int n, int k, int k_pad, int64_t ld_out) {
    __shared__ topk_shared sh;
    __shared__ typename topk_gpu_block::scan_t::TempStorage scan_tmp;
    topk_gpu_block blk{scan_tmp};
    topk_select_row(blk, sh, score + (int64_t) blockIdx.x * n, n, k, out + (int64_t) blockIdx.x * ld_out, k_pad);
}

#endif // __CUDACC__
