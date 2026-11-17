// llama-kv-xattn.h -- XAttention-style block selection for PREFILL attention: score every (query block, key block) pair from an anti-diagonal strided
// QK estimate, keep per query-block row the blocks that cover `threshold` of the attention mass (first column and the diagonal always kept), emit a
// bool mask [head][q_block][k_block].  Pure host C++, header-only, no ggml dependency (tests/test-kv-xattn.cpp).
// Row: S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01  (step (1) of the row: the estimator + selection; the kernel that SKIPS masked
// tiles is step (2) and is NOT in this file).
//
// SPEC READ FROM (behaviour only, own implementation; Apache-2.0, openvinotoolkit/openvino @ b0086aa9c,
// src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/xattention.hpp, read in full, 368 lines):
//   * for each of `stride` phases i the strided query rows (stride-1-i + m*stride) are multiplied with the key rows repacked at the same stride
//     (i + n*stride) and accumulated: A[m][n] = sum_i q[m*stride + stride-1-i] . k[n*stride + i]  (the sum runs over the anti-diagonals of every
//     stride x stride patch, so one strided cell sees `stride` query-key products that lie on one anti-diagonal);
//   * softmax over the strided key axis with scale 1/sqrt(head_dim)/stride and causal truncation;
//   * each (block/stride)^2 tile is summed to one score per (q block, k block);
//   * per (head, q block): required = threshold * row_sum; the FIRST column and the DIAGONAL block are always selected and are left out of the
//     sort; the rest are taken in descending score while the cumulative mass of the ones already taken is below `required`; blocks right of the
//     diagonal are false.  threshold 1.0 therefore selects every causal block.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace justician_kv {

struct xattn_cfg {
    int   stride    = 8;      // anti-diagonal stride (block must be a multiple of it)
    int   block     = 128;    // selection granularity in tokens
    float threshold = 0.9f;   // fraction of the per-row attention mass to cover, in (0, 1]
};

// Q: [n_head][n_q][head_dim], K: [n_head][n_k][head_dim] row-major.  The queries are the LAST n_q positions of the n_k keys (prefill chunk with n_k >=
// n_q; n_k - n_q must be a multiple of block).  n_q and n_k must be multiples of block.  Returns uint8 mask [n_head][n_q/block][n_k/block]
// (1 = compute this tile), empty on a shape error.  A single query block returns {all ones for the causal prefix} without any estimate.
inline std::vector<uint8_t> xattn_block_mask(const float * Q, const float * K, int n_head, int n_q, int n_k, int head_dim, const xattn_cfg & cfg) {
    std::vector<uint8_t> mask;
    const int B = cfg.block, st = cfg.stride;
    if (n_head <= 0 || head_dim <= 0 || B <= 0 || st <= 0 || B % st != 0 || n_q <= 0 || n_k < n_q || n_q % B != 0 || n_k % B != 0 ||
        (n_k - n_q) % B != 0 || !(cfg.threshold > 0.0f && cfg.threshold <= 1.0f)) {
        return mask;
    }
    const int qb_n = n_q / B, kb_n = n_k / B, off_b = (n_k - n_q) / B;   // query block qb sits at key block qb + off_b
    mask.assign((size_t) n_head * qb_n * kb_n, 0);
    if (qb_n == 1 && kb_n == 1) {                                      // nothing to select: one block, early return
        std::fill(mask.begin(), mask.end(), 1);
        return mask;
    }
    const int sm = n_q / st, sn = n_k / st;                            // strided matrix size
    const int per = B / st;                                            // strided cells per block edge
    const float scale = 1.0f / std::sqrt((float) head_dim) / (float) st;
    const int L = st * head_dim;                                       // length of one strided cell's dot product
    std::vector<float> A((size_t) sm * sn);
    std::vector<float> Qs((size_t) sm * L);
    std::vector<double> tile((size_t) qb_n * kb_n);
    for (int h = 0; h < n_head; ++h) {
        const float * q = Q + (size_t) h * n_q * head_dim;
        const float * k = K + (size_t) h * n_k * head_dim;
        // A[m][n] = sum_i q[m*st + st-1-i] . k[n*st + i]. The st key rows of cell n are contiguous in K (k + n*st*head_dim), so gather the
        // query rows once in the matching (reversed) order and the cell becomes one contiguous dot product of length st*head_dim.
        for (int m = 0; m < sm; ++m) {
            for (int i = 0; i < st; ++i) {
                std::copy(q + (size_t) (m * st + st - 1 - i) * head_dim, q + (size_t) (m * st + st - i) * head_dim,
                          Qs.data() + (size_t) m * L + (size_t) i * head_dim);
            }
        }
        // strided anti-diagonal scores (8 independent partial sums so the compiler can keep them in vector lanes)
        for (int m = 0; m < sm; ++m) {
            const int qb = m / per;
            const int n_allowed = (qb + off_b + 1) * per;              // strided key cells in the causal prefix of this q block (block granularity)
            const float * qs = Qs.data() + (size_t) m * L;
            for (int n = 0; n < sn; ++n) {
                if (n >= n_allowed) {
                    A[(size_t) m * sn + n] = -INFINITY;
                    continue;
                }
                const float * kr = k + (size_t) n * L;
                float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                int d = 0;
                for (; d + 8 <= L; d += 8) {
                    for (int j = 0; j < 8; ++j) {
                        acc[j] += qs[d + j] * kr[d + j];
                    }
                }
                float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
                for (; d < L; ++d) {
                    s += qs[d] * kr[d];
                }
                A[(size_t) m * sn + n] = s * scale;
            }
        }
        // row softmax over the causal strided keys, then tile sums (each exp computed once)
        std::fill(tile.begin(), tile.end(), 0.0);
        for (int m = 0; m < sm; ++m) {
            const int qb = m / per;
            const int n_allowed = (qb + off_b + 1) * per;
            float * a = A.data() + (size_t) m * sn;
            float mx = -INFINITY;
            for (int n = 0; n < n_allowed; ++n) {
                mx = std::max(mx, a[n]);
            }
            double den = 0.0;
            for (int n = 0; n < n_allowed; ++n) {
                a[n] = std::exp(a[n] - mx);
                den += a[n];
            }
            const double inv = 1.0 / den;
            for (int n = 0; n < n_allowed; ++n) {
                tile[(size_t) qb * kb_n + n / per] += (double) a[n] * inv;
            }
        }
        // selection per q block row
        for (int qb = 0; qb < qb_n; ++qb) {
            const int diag = qb + off_b;
            uint8_t * mrow = mask.data() + ((size_t) h * qb_n + qb) * kb_n;
            double row_sum = 0.0;
            for (int kb = 0; kb <= diag; ++kb) {
                row_sum += tile[(size_t) qb * kb_n + kb];
            }
            const double required = (double) cfg.threshold * row_sum;
            mrow[0] = 1;
            mrow[diag] = 1;
            if (cfg.threshold >= 1.0f) {                               // every causal block (no rounding race between cum and row_sum)
                std::fill(mrow, mrow + diag + 1, (uint8_t) 1);
                continue;
            }
            std::vector<int> rest;
            for (int kb = 1; kb < diag; ++kb) {
                rest.push_back(kb);
            }
            std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) { return tile[(size_t) qb * kb_n + a] > tile[(size_t) qb * kb_n + b]; });
            // The always-kept first column and diagonal COUNT toward the covered mass: xattention.hpp:350-357 builds the cumulative sum over the
            // sorted row with the first column (entry 0) and the diagonal (entry 1, swapped in at :340) at its head, zeroing only entry 1's own
            // prefix, so the third entry already starts from first + diagonal.
            double cum = tile[(size_t) qb * kb_n + 0] + (diag != 0 ? tile[(size_t) qb * kb_n + diag] : 0.0);
            for (int kb : rest) {
                if (cum >= required) {
                    break;
                }
                mrow[kb] = 1;
                cum += tile[(size_t) qb * kb_n + kb];
            }
        }
    }
    return mask;
}

} // namespace justician_kv
