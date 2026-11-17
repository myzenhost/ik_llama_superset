// llama-kv-xattn-op.h -- XAttention prefill block selection WIRED into the flash-attention graph (step 2 of the row).
// Row S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01.  Step 1 (the estimator, src/llama-kv-xattn.h) selects key blocks per
// (head, query block); this file is the ggml custom op that turns that selection into the -INFINITY tiles of the kq_mask flash_attn_ext consumes,
// and the iqk CPU flash-attention loop (ggml/src/iqk/fa/iqk_fa_templates.h, iqk_fa_tile_all_masked) skips a q_step x k_step tile whose mask is
// all -inf, so a deselected block costs neither its QK product nor its V accumulation.
//
// SPEC (behaviour read from openvinotoolkit/openvino @ b0086aa9c, Apache-2.0, own implementation, no code copied):
//   executor_pa.cpp:2385-2440 runs the estimator only on a single-sequence prefill with more than one token, and :900-1075 skips the QK GEMM of a
//   deselected block (`if (!sparse_attention_mask[...]) continue;`), then zeroes the block in the softmax.  Here the same skip is expressed through
//   the mask: a deselected block becomes -inf for every row of the query block.
//
// THE MASK IS SHARED BY ALL HEADS ([n_kv, n_rows]; per-head selection cannot be expressed through it, same constraint as offload-attention's
// mask_from_kq_op).  A key block is therefore kept for a query block if ANY head selects it: a union, which is never less than what each head
// selected on its own (it can only over-include) and which skips less than the per-head reference does.
//
// WHEN IT ACTS (everything else is copied through unchanged and counted -- a wrong mask corrupts every answer, an untouched one is today's result):
//   * the mask of every real query row t must be exactly the causal prefix [0, base + t] with base = (last visible key of the last row) + 1 - n_tok,
//     i.e. one sequence, cells in position order, no sliding window, no other sequence (parallel slots / multi-sequence batches fail this check);
//   * n_tok and base are multiples of the block (the query block then sits on the key-block grid, as the estimator requires);
//   * K and Q are F32 or F16 host tensors (a quantised cache is not scored; the op passes through);
//   * 0 < threshold < 1 (1.0 keeps every causal block: the build site does not splice the op at all).
// Work is split by query block (unit = `block` mask rows), so every dst row is copied and masked by exactly one thread.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "llama-kv-xattn.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace justician_kv {

struct xattn_op_cfg {
    float   threshold = 0.0f;   // 0 = off
    int32_t stride    = 8;
    int32_t block     = 128;
};

struct xattn_op_stats {
    std::atomic<uint64_t> calls{0};          // op invocations (one per layer per ubatch)
    std::atomic<uint64_t> units_masked{0};   // query blocks that went through selection
    std::atomic<uint64_t> units_pass{0};     // query blocks copied through (shape/causality check failed)
    std::atomic<uint64_t> blocks_causal{0};  // causal (query block, key block) pairs seen by selection
    std::atomic<uint64_t> blocks_dropped{0}; // of those, deselected for every head (set to -inf)
    std::atomic<bool>     traced{false};
};

inline xattn_op_stats & xattn_stats() {
    static xattn_op_stats s;
    return s;
}

namespace xattn_detail {

inline float elem2(const ggml_tensor * t, int64_t i0, int64_t i1, int64_t i2) {
    const char * p = (const char *) t->data + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2];
    return t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*(const ggml_fp16_t *) p) : *(const float *) p;
}

inline bool host_typed(const ggml_tensor * t) {
    return t && t->data && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) &&
           (!t->buffer || ggml_backend_buffer_is_host(t->buffer));
}

inline bool is_neg_inf(const ggml_tensor * m, int64_t s, int64_t t) {
    const float v = elem2(m, s, t, 0);
    return std::isinf(v) && v < 0.0f;
}

inline void put_row_range(ggml_tensor * dst, int64_t t, int64_t s0, int64_t s1, float v) {
    char * row = (char *) dst->data + t * dst->nb[1];
    if (dst->type == GGML_TYPE_F16) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(v);
        for (int64_t s = s0; s < s1; ++s) ((ggml_fp16_t *) row)[s] = h;
    } else {
        for (int64_t s = s0; s < s1; ++s) ((float *) row)[s] = v;
    }
}

inline void copy_row(ggml_tensor * dst, const ggml_tensor * a, int64_t t) {
    if (dst->data == a->data) return;
    const int64_t n = a->ne[0];
    char * drow = (char *) dst->data + t * dst->nb[1];
    if (dst->type == a->type && a->nb[0] == (size_t) ggml_type_size(a->type)) {
        std::copy((const char *) a->data + t * a->nb[1], (const char *) a->data + t * a->nb[1] + n * ggml_type_size(a->type), drow);
        return;
    }
    for (int64_t s = 0; s < n; ++s) {
        const float v = elem2(a, s, t, 0);
        if (dst->type == GGML_TYPE_F16) ((ggml_fp16_t *) drow)[s] = ggml_fp32_to_fp16(v);
        else                            ((float *) drow)[s] = v;
    }
}

} // namespace xattn_detail

// ggml_custom3_op_t: dst = a with deselected (query block, key block) tiles set to -inf.
//   a = kq_mask [n_kv, n_rows (padded)]        (F16 or F32)
//   b = k       [head_dim, >= n_kv, n_head_kv] (any strides: the cache view interleaves heads per cell)
//   c = q       [head_dim, n_tok, n_head]      (any strides: the graph passes a permuted view)
inline void xattn_mask_op(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c,
                          int ith, int nth, void * userdata) {
    using namespace xattn_detail;
    const xattn_op_cfg * cfg = (const xattn_op_cfg *) userdata;
    xattn_op_stats & st = xattn_stats();
    if (ith == 0) st.calls.fetch_add(1, std::memory_order_relaxed);

    const int64_t n_kv   = a->ne[0];
    const int64_t n_rows = a->ne[1];
    const int64_t hd     = b ? b->ne[0] : 0;
    const int64_t n_hkv  = b ? b->ne[2] : 0;
    const int64_t n_tok  = c ? c->ne[1] : 0;
    const int64_t n_head = c ? c->ne[2] : 0;
    const int     B      = cfg ? cfg->block : 0;

    bool usable = cfg && host_typed(a) && host_typed(dst) && host_typed(b) && host_typed(c) &&
                  cfg->threshold > 0.0f && cfg->threshold < 1.0f && B > 0 && cfg->stride > 0 && B % cfg->stride == 0 &&
                  hd > 0 && c->ne[0] == hd && n_hkv > 0 && n_head > 0 && n_head % n_hkv == 0 &&
                  n_tok > 0 && n_tok <= n_rows && b->ne[1] >= n_kv && n_tok % B == 0;

    // Geometry from the LAST real row: its last visible key is the last query's own position.
    int64_t base = -1;
    if (usable) {
        int64_t last = -1;
        for (int64_t s = n_kv - 1; s >= 0; --s) {
            if (!is_neg_inf(a, s, n_tok - 1)) { last = s; break; }
        }
        base = last + 1 - n_tok;
        usable = last >= 0 && base >= 0 && base % B == 0;
    }

    const int64_t n_units = (n_rows + B - 1) / (B > 0 ? B : 1);
    const int64_t unit    = B > 0 ? B : n_rows;
    const int64_t gqa     = (usable && n_hkv > 0) ? n_head / n_hkv : 1;

    std::vector<float>   Kh, Qh;
    std::vector<uint8_t> keep;
    xattn_cfg ecfg;
    if (cfg) { ecfg.stride = cfg->stride; ecfg.block = cfg->block; ecfg.threshold = cfg->threshold; }

    for (int64_t u = ith; u < (B > 0 ? n_units : (ith == 0 ? 1 : 0)); u += nth) {
        const int64_t r0 = u * unit, r1 = std::min(n_rows, r0 + unit);
        for (int64_t t = r0; t < r1; ++t) copy_row(dst, a, t);
        if (!usable || r0 >= n_tok) continue;   // padded rows: copied, never selected for

        // causality check for every row of this query block
        bool causal = true;
        for (int64_t t = r0; t < r1 && causal; ++t) {
            const int64_t lim = base + t;      // last visible key of row t
            for (int64_t s = 0; s < n_kv; ++s) {
                if (is_neg_inf(a, s, t) != (s > lim)) { causal = false; break; }
            }
        }
        if (!causal) { st.units_pass.fetch_add(1, std::memory_order_relaxed); continue; }

        const int64_t qb   = u;
        const int64_t kb_n = base / B + qb + 1;     // causal key blocks of this query block (diagonal included)
        const int64_t nk   = kb_n * B;
        if (kb_n < 2) { st.units_masked.fetch_add(1, std::memory_order_relaxed); st.blocks_causal.fetch_add(1, std::memory_order_relaxed); continue; }

        keep.assign((size_t) kb_n, 0);
        Kh.resize((size_t) (nk * hd));
        Qh.resize((size_t) (B * hd));
        bool ok = true;
        for (int64_t kh = 0; kh < n_hkv && ok; ++kh) {
            for (int64_t s = 0; s < nk; ++s) {
                for (int64_t d = 0; d < hd; ++d) Kh[(size_t) (s * hd + d)] = elem2(b, d, s, kh);
            }
            for (int64_t g = 0; g < gqa && ok; ++g) {
                const int64_t h = kh * gqa + g;   // consecutive query heads share a kv head (ggml flash-attn ik2 = iq2 / rk2)
                for (int64_t i = 0; i < B; ++i) {
                    for (int64_t d = 0; d < hd; ++d) Qh[(size_t) (i * hd + d)] = elem2(c, d, r0 + i, h);
                }
                const std::vector<uint8_t> m = xattn_block_mask(Qh.data(), Kh.data(), 1, B, (int) nk, (int) hd, ecfg);
                if (m.size() != (size_t) kb_n) { ok = false; break; }
                for (int64_t kb = 0; kb < kb_n; ++kb) keep[(size_t) kb] |= m[(size_t) kb];
            }
        }
        if (!ok) { st.units_pass.fetch_add(1, std::memory_order_relaxed); continue; }

        uint64_t dropped = 0;
        for (int64_t kb = 0; kb < kb_n; ++kb) {
            if (keep[(size_t) kb]) continue;
            ++dropped;
            for (int64_t t = r0; t < r1; ++t) put_row_range(dst, t, kb * B, (kb + 1) * B, -INFINITY);
        }
        st.units_masked.fetch_add(1, std::memory_order_relaxed);
        st.blocks_causal.fetch_add((uint64_t) kb_n, std::memory_order_relaxed);
        st.blocks_dropped.fetch_add(dropped, std::memory_order_relaxed);
    }

    if (ith == 0 && !st.traced.exchange(true)) {
        fprintf(stderr, "xattn op: n_kv=%lld n_tok=%lld heads=%lld/%lld block=%d stride=%d threshold=%.3f %s\n",
                (long long) n_kv, (long long) n_tok, (long long) n_head, (long long) n_hkv, B, cfg ? cfg->stride : 0,
                cfg ? (double) cfg->threshold : 0.0, usable ? "ACTIVE" : "PASS-THROUGH (shape/type/geometry check failed)");
    }
}

// Process-exit summary: how much of the causal block area the union selection actually dropped.
struct xattn_exit_report {
    // Construct the stats first so they are destroyed after this reporter (reverse order of construction).
    xattn_exit_report() { (void) xattn_stats(); }
    ~xattn_exit_report() {
        xattn_op_stats & st = xattn_stats();
        const uint64_t calls = st.calls.load();
        if (calls == 0) return;
        const uint64_t bc = st.blocks_causal.load(), bd = st.blocks_dropped.load();
        fprintf(stderr, "xattn summary: calls=%llu query_blocks_selected=%llu query_blocks_passthrough=%llu causal_blocks=%llu dropped=%llu (%.2f%%)\n",
                (unsigned long long) calls, (unsigned long long) st.units_masked.load(), (unsigned long long) st.units_pass.load(),
                (unsigned long long) bc, (unsigned long long) bd, bc ? 100.0 * (double) bd / (double) bc : 0.0);
    }
};

} // namespace justician_kv
