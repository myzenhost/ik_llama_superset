#pragma once

// GPTQ error-compensated rounding for llama-quantize over the block-diagonal Hessian sidecar of llama-imatrix
// (--hessian-block N, common/imatrix-hessian.h). Row S69-ncnn-PICK-GPTQ-ERROR-COMPENSATED-ROUNDING-2026-09-29-01.
//
// SOURCE-READ (rule 1.87; BSD-3-Clause, behaviour only, no code copied):
//   Tencent/ncnn @ c6b351b56fbe tools/quantize/ncnnllm2table.cpp:936-1046 make_gptq_qweight -- block scales FIRST, H = sum_s x_s x_s^T,
//   damp H_ii += 0.01 * mean(diag H) (:982-988), upper Cholesky factor of H^-1 (:893-934, :990-996), then per column
//   err = (w_k - q_k) / Hinv_kk and w_j -= err * Hinv_kj for the later columns (:1029-1040, immediate inside a block, lazy across blocks).
//   llmcompressor 0.10.0.2 (Apache-2.0) gptq_quantize.py:153-166: a dead input column (H_ii == 0) gets H_ii = 1.
//
// WHAT IS DIFFERENT HERE, AND WHY. A ggml quant block (32 or 256 values) is quantised as a unit: the type's quantiser chooses the block's
// scales together with its values (imatrix-weighted search). Rounding one column alone at a fixed scale would need a per-type decoder of
// every scale layout. So the unit of the column walk is the type's quant block G (all of it quantised at once by the type's own
// quantiser, exactly as today), and its error is propagated to the not-yet-quantised columns R of the same Hessian block by the exact
// optimal-brain-quantiser update for a jointly fixed group:
//     w_R  -=  (e_G  U_GG^-1)  U_GR,      e_G = w_G - deq(q_G),   U = the upper Cholesky factor of inv(H_damped)  (inv(H) = U^T U).
// For |G| = 1 this is GPTQ's per-column update exactly (e_k / U_kk * U_kR). Derivation: with the columns before G already fixed, the
// remaining inverse is M = U_F^T U_F (F = G u R, U upper), the least-squares response to a fixed change -e_G of w_G is
// -e_G M_GG^-1 M_GR, and M_GG = U_GG^T U_GG, M_GR = U_GG^T U_GR.
// Propagation stops at the Hessian block edge (the sidecar is block-diagonal), so a Hessian block must hold at least two quant blocks
// for anything to move: N >= 2 * blck_size (64 for Q4_0, 512 for the k-quants).

#include "ggml.h"

#include <algorithm>
#include <atomic>   // the llama-quantize driver (gptq_quantize_tensor) splits factors and rows over threads with atomics
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace llama_gptq {

// Types whose row is a plain concatenation of independent quant blocks (no row header, no interleave), so a block can be
// quantised alone and the row assembled block by block. Others are left to the ordinary path.
inline bool type_supported(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// One Hessian block's factor: U (n x n, row-major, upper) with inv(H + damp * mean(diag H) * I) = U^T U.
struct block_factor {
    int64_t             n = 0;
    std::vector<double> U;
    int64_t             n_dead = 0;   // input columns with H_ii == 0 (never active in calibration): H_ii set to 1
};

// In-place lower Cholesky of a (n x n, row-major, symmetric): a = L L^T, L in the lower triangle, upper triangle zeroed.
// Returns false if a is not positive definite.
inline bool cholesky_lower(std::vector<double> & a, int64_t n) {
    for (int64_t j = 0; j < n; ++j) {
        double d = a[j * n + j];
        for (int64_t k = 0; k < j; ++k) d -= a[j * n + k] * a[j * n + k];
        if (!(d > 0.0)) return false;
        const double ljj = std::sqrt(d);
        a[j * n + j] = ljj;
        for (int64_t i = j + 1; i < n; ++i) {
            double s = a[i * n + j];
            for (int64_t k = 0; k < j; ++k) s -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = s / ljj;
        }
        for (int64_t k = j + 1; k < n; ++k) a[j * n + k] = 0.0;
    }
    return true;
}

// H: n x n block, element (i, j) at H[i * ld + j] (the sidecar slab has ld = its block size N).
inline bool factor_block(const float * H, int64_t ld, int64_t n, double damp, block_factor & f) {
    f.n = n;
    f.n_dead = 0;
    std::vector<double> a((size_t) (n * n));
    double mean_diag = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) a[(size_t) (i * n + j)] = 0.5 * ((double) H[i * ld + j] + (double) H[j * ld + i]);
        mean_diag += a[(size_t) (i * n + i)];
    }
    mean_diag /= (double) std::max<int64_t>(1, n);
    if (!(mean_diag > 0.0)) mean_diag = 1.0;
    for (int64_t i = 0; i < n; ++i) {
        if (a[(size_t) (i * n + i)] == 0.0) {
            ++f.n_dead;
            for (int64_t j = 0; j < n; ++j) { a[(size_t) (i * n + j)] = 0.0; a[(size_t) (j * n + i)] = 0.0; }
            a[(size_t) (i * n + i)] = 1.0;
        }
        a[(size_t) (i * n + i)] += damp * mean_diag;
    }
    if (!cholesky_lower(a, n)) return false;                 // a = L
    // Linv (lower) by forward substitution, column by column
    std::vector<double> li((size_t) (n * n), 0.0);
    for (int64_t c = 0; c < n; ++c) {
        li[(size_t) (c * n + c)] = 1.0 / a[(size_t) (c * n + c)];
        for (int64_t i = c + 1; i < n; ++i) {
            double s = 0.0;
            for (int64_t k = c; k < i; ++k) s += a[(size_t) (i * n + k)] * li[(size_t) (k * n + c)];
            li[(size_t) (i * n + c)] = -s / a[(size_t) (i * n + i)];
        }
    }
    // inv(A) = Linv^T Linv
    std::vector<double> inv((size_t) (n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            double s = 0.0;
            for (int64_t k = std::max(i, j); k < n; ++k) s += li[(size_t) (k * n + i)] * li[(size_t) (k * n + j)];
            inv[(size_t) (i * n + j)] = s;
            inv[(size_t) (j * n + i)] = s;
        }
    }
    // inv(A) = L2 L2^T, U = L2^T  -> inv(A) = U^T U with U upper
    if (!cholesky_lower(inv, n)) return false;
    f.U.assign((size_t) (n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) f.U[(size_t) (i * n + j)] = inv[(size_t) (j * n + i)];
    }
    return true;
}

// Quantise one row of K values into dst (ggml_row_size(t, K) bytes). w is a WORKING COPY (it is modified: the propagated errors).
// fac[b] covers columns [b*N, b*N + fac[b].n). propagate = false quantises block by block with no error feedback (the control arm:
// the same block-wise calls, so any difference between the arms is the propagation alone).
inline void quantize_row(ggml_type t, float * w, int64_t K, const float * imatrix, const quantize_user_data * ud,
                         const std::vector<block_factor> & fac, int64_t N, void * dst, bool propagate) {
    const int64_t g  = ggml_blck_size(t);
    const size_t  gb = ggml_row_size(t, g);
    const auto tr = ggml_internal_get_type_traits(t);
    std::vector<float> deq((size_t) g), e((size_t) g);
    for (int64_t b = 0; b * N < K; ++b) {
        const int64_t k0 = b * N;
        const int64_t nb = std::min<int64_t>(N, K - k0);
        const block_factor & f = fac[(size_t) b];
        for (int64_t a = 0; a < nb; a += g) {
            char * qd = (char *) dst + ((k0 + a) / g) * gb;
            ggml_quantize_chunk(t, w + k0 + a, qd, 0, 1, g, imatrix ? imatrix + k0 + a : nullptr, ud);
            if (!propagate || a + g >= nb) continue;
            tr.to_float(qd, deq.data(), g);
            // e' = e U_GG^-1  (row vector times the inverse of an upper triangle: forward substitution)
            for (int64_t i = 0; i < g; ++i) {
                double s = (double) w[k0 + a + i] - (double) deq[(size_t) i];
                for (int64_t m = 0; m < i; ++m) s -= (double) e[(size_t) m] * f.U[(size_t) ((a + m) * f.n + a + i)];
                e[(size_t) i] = (float) (s / f.U[(size_t) ((a + i) * f.n + a + i)]);
            }
            for (int64_t j = a + g; j < nb; ++j) {
                double s = 0.0;
                for (int64_t i = 0; i < g; ++i) s += (double) e[(size_t) i] * f.U[(size_t) ((a + i) * f.n + j)];
                w[k0 + j] -= (float) s;
            }
        }
    }
}

// The quantity GPTQ minimises for one row, restricted to the block-diagonal Hessian: sum_b d_b H_b d_b^T, d = w_orig - deq(q).
// H_b read from the sidecar slab (ld = N). Used by the tests and the tool's per-tensor report.
inline double block_objective(const float * w_orig, const void * q, ggml_type t, int64_t K, const float * slab, int64_t N) {
    std::vector<float> deq((size_t) K);
    ggml_internal_get_type_traits(t).to_float(q, deq.data(), K);
    double tot = 0.0;
    for (int64_t k0 = 0; k0 < K; k0 += N) {
        const int64_t nb = std::min<int64_t>(N, K - k0);
        for (int64_t i = 0; i < nb; ++i) {
            const double di = (double) w_orig[k0 + i] - (double) deq[(size_t) (k0 + i)];
            if (di == 0.0) continue;
            double s = 0.0;
            for (int64_t j = 0; j < nb; ++j) s += (double) slab[(k0 + i) * N + j] * ((double) w_orig[k0 + j] - (double) deq[(size_t) (k0 + j)]);
            tot += di * s;
        }
    }
    return tot;
}

}  // namespace llama_gptq
