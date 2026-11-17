#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

// AVX2 path availability. A TU compiled WITHOUT -mavx2 (the common library is, on every build here: GGML_NATIVE=OFF applies the ISA flags to the
// ggml targets only, measured in compile_commands.json 2026-10-06) used to get the scalar scan only, because the path was guarded by __AVX2__ at
// compile time. GCC/clang x86 builds now compile the AVX2 scan with a function-level target attribute (the way this library already does for
// common_sampler_speculative_top1_avx2) and pick it at run time.
#if defined(__AVX2__)
#define COMMON_TOPK_AVX2_ATTR
#define COMMON_TOPK_AVX2_PATH 1
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define COMMON_TOPK_AVX2_ATTR __attribute__((target("avx2")))
#define COMMON_TOPK_AVX2_PATH 1
#endif
#if defined(COMMON_TOPK_AVX2_PATH)
#include <immintrin.h>
#endif

// Exact top-k over raw logits without building a vocabulary-sized candidate array.
//
// S69-MNN-PICK-TOPK-RAW-LOGIT-THRESHOLD-PREFILTER-2026-10-05-01. Behaviour spec from MNN transformers/llm/engine/src/sampler.cpp
// (Apache-2.0): keep a k-sized sorted window and the running k-th value as a threshold; a block whose maximum does not exceed the
// threshold cannot change the window and is skipped. Written from that description; no code copied.
//
// Result order is the order llama_sort() produces: logit descending, equal logits by the LOWER token id. A later element only enters the
// window when its logit is STRICTLY greater than the current k-th, which is what makes a tie keep the lower id (elements arrive in id
// order). NaN is never inserted (the old path's order with a NaN is undefined). `out` must have room for k elements; returns how many were
// written (min(k, number of non-NaN logits)).
static inline size_t common_topk_scan_scalar(const float * logits, int32_t n, int32_t k, llama_token_data * out) {
    size_t cnt = 0;
    float  thr = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        const float x = logits[i];
        if (x != x) {
            continue;
        }
        if ((int32_t) cnt == k && !(x > thr)) {
            continue;
        }
        // insertion position: first slot whose logit is not >= x (strictly greater elements stay in front)
        size_t pos = cnt < (size_t) k ? cnt : (size_t) k - 1;
        while (pos > 0 && out[pos - 1].logit < x) {
            if (pos < (size_t) k) {
                out[pos] = out[pos - 1];
            }
            --pos;
        }
        out[pos] = llama_token_data{ i, x, 0.0f };
        if ((int32_t) cnt < k) {
            ++cnt;
        }
        if ((int32_t) cnt == k) {
            thr = out[k - 1].logit;
        }
    }
    return cnt;
}

#if defined(COMMON_TOPK_AVX2_PATH)
static inline int common_topk_ctz32(unsigned v) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long idx;
    _BitScanForward(&idx, v);
    return (int) idx;
#else
    return __builtin_ctz(v);
#endif
}

COMMON_TOPK_AVX2_ATTR
static inline size_t common_topk_scan_avx2(const float * logits, int32_t n, int32_t k, llama_token_data * out) {
    size_t cnt = 0;
    float  thr = 0.0f;
    int32_t i = 0;
    auto insert = [&](int32_t idx, float x) {
        size_t pos = cnt < (size_t) k ? cnt : (size_t) k - 1;
        while (pos > 0 && out[pos - 1].logit < x) {
            if (pos < (size_t) k) {
                out[pos] = out[pos - 1];
            }
            --pos;
        }
        out[pos] = llama_token_data{ idx, x, 0.0f };
        if ((int32_t) cnt < k) {
            ++cnt;
        }
        if ((int32_t) cnt == k) {
            thr = out[k - 1].logit;
        }
    };
    // fill phase: the window is not full yet, every non-NaN element enters
    for (; i < n && (int32_t) cnt < k; ++i) {
        if (logits[i] == logits[i]) {
            insert(i, logits[i]);
        }
    }
    // steady phase: 8 logits at a time; a block with no element strictly above the threshold is skipped (NaN compares false, so it is skipped too)
    __m256 vthr = _mm256_set1_ps(thr);
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(logits + i);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(v, vthr, _CMP_GT_OQ));
        if (!mask) {
            continue;
        }
        while (mask) {
            const int j = common_topk_ctz32((unsigned) mask);
            mask &= mask - 1;
            const float x = logits[i + j];
            if (x > thr) {                     // the threshold may have risen inside this block
                insert(i + j, x);
            }
        }
        vthr = _mm256_set1_ps(thr);
    }
    for (; i < n; ++i) {
        const float x = logits[i];
        if (x > thr) {
            insert(i, x);
        }
    }
    return cnt;
}
#endif

// true when the AVX2 scan exists in this build AND the running CPU has AVX2
static inline bool common_topk_have_avx2() {
#if defined(COMMON_TOPK_AVX2_PATH)
#if defined(__AVX2__)
    return true;
#else
    static const bool have = __builtin_cpu_supports("avx2") != 0;
    return have;
#endif
#else
    return false;
#endif
}

static inline size_t common_topk_scan(const float * logits, int32_t n, int32_t k, llama_token_data * out) {
    if (k <= 0 || n <= 0) {
        return 0;
    }
    if (k > n) {
        k = n;
    }
#if defined(__AVX2__)
    return common_topk_scan_avx2(logits, n, k, out);              // compiled with AVX2: no run-time check needed (the same code path as before)
#else
    if (common_topk_have_avx2()) {                                // function-attribute build: pick at run time
#if defined(COMMON_TOPK_AVX2_PATH)
        return common_topk_scan_avx2(logits, n, k, out);
#endif
    }
    return common_topk_scan_scalar(logits, n, k, out);
#endif
}

// Greedy fast path: index of the largest logit, lowest index on a tie (what std::max_element with a `<` comparator returns in
// llama_sample_token_greedy_impl), skipping NaN. Returns -1 when there is no non-NaN logit. With a NaN in the array the old path's answer
// is order-dependent (a NaN at the front wins), so the two agree only on NaN-free input; the caller treats -1 as "use the full path".
static inline int32_t common_argmax_scan_scalar(const float * logits, int32_t n) {
    int32_t best = -1;
    float   bv   = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        const float x = logits[i];
        if (x == x && (best < 0 || x > bv)) {
            best = i;
            bv   = x;
        }
    }
    return best;
}

#if defined(COMMON_TOPK_AVX2_PATH)
COMMON_TOPK_AVX2_ATTR
static inline int32_t common_argmax_scan_avx2(const float * logits, int32_t n) {
    int32_t i = 0;
    int32_t best = -1;
    float   bv = 0.0f;
    for (; i < n && best < 0; ++i) {            // first non-NaN seeds the running maximum
        if (logits[i] == logits[i]) {
            best = i;
            bv   = logits[i];
        }
    }
    __m256 vb = _mm256_set1_ps(bv);
    for (; i + 8 <= n; i += 8) {
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(logits + i), vb, _CMP_GT_OQ));
        if (!mask) {
            continue;
        }
        while (mask) {
            const int j = common_topk_ctz32((unsigned) mask);
            mask &= mask - 1;
            if (logits[i + j] > bv) {           // strictly greater: a tie keeps the lower index
                bv   = logits[i + j];
                best = i + j;
            }
        }
        vb = _mm256_set1_ps(bv);
    }
    for (; i < n; ++i) {
        if (logits[i] > bv) {
            bv   = logits[i];
            best = i;
        }
    }
    return best;
}
#endif

static inline int32_t common_argmax_scan(const float * logits, int32_t n) {
    if (n <= 0) {
        return -1;
    }
#if defined(COMMON_TOPK_AVX2_PATH)
    if (common_topk_have_avx2()) {
        return common_argmax_scan_avx2(logits, n);
    }
#endif
    return common_argmax_scan_scalar(logits, n);
}
