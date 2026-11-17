// offload_attention_mask_op.cpp — offload-attention for the FLASH-ATTENTION path (#220).
//
// WHY A SECOND OP. The non-FA path exposes kq (the scores) between ggml_mul_mat and the softmax,
// so mask_op() can rank pages by their actual scores. ggml_flash_attn_ext computes the scores
// INTERNALLY and never materialises them -- there is nothing to intercept. The only lever it
// offers is the kq_mask it consumes, so selection here has to be made from the KEYS and the
// QUERY, which is the original QUEST formulation (2406.10774) rather than the score-max shortcut
// the non-FA op could afford.
//
// A SEMANTIC DIFFERENCE, stated because it changes the result and is not a detail:
//   k, v      are per-head 3D views  [head_dim, n_kv, n_head_kv]
//   kq_mask   is 2D                  [n_kv, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD)] -- SHARED by
//             every head.
// So masking a page hides it from ALL heads at once. Per-head selection cannot be expressed
// through this tensor. This op therefore keeps a page if ANY head ranks it in the top-k for that
// token -- a union, which is strictly MORE conservative than per-head selection: at the same
// page_budget the FA path attends at least as many pages as the non-FA path, never fewer. That is
// the safe direction (it can only over-include), but it means the two paths are NOT numerically
// identical at equal budget and must not be compared as if they were.
//
// PADDED ROWS. Rows from n_tokens up to the padded height are not real queries. They are left
// exactly as found -- never selected for, never written -- because the padding is what the Vulkan
// coopmat2 shaders read unclamped (llama.cpp PR #12720), and writing into it is precisely the
// memory-safety hazard that kept this splice sequenced behind the GGML_KQ_MASK_PAD fix.
//
// FAILS OPEN and COUNTS IT, same contract as mask_op(): anything unsupported copies the mask
// through unchanged, because a wrong mask corrupts every answer while an absent mask is merely
// today's behaviour.

#include "ggml.h"
#include "ggml-backend.h"
#include "offload_attention.h"
#include "offload_select_share.h"   // pure layer/step selection-reuse helpers (not yet called here; keeps the header in a compiled TU)

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

namespace ik_offload {

std::atomic<uint64_t> g_maskop_calls{0};
std::atomic<uint64_t> g_maskop_masked{0};
std::atomic<uint64_t> g_maskop_passthrough{0};
std::atomic<uint64_t> g_maskop_last_keep{0};
std::atomic<uint64_t> g_maskop_last_npages{0};
// See the matching note in offload_attention_op.cpp: the counters are reachable only from the
// offload-attn-test example, so keep/nP were invisible during serving.
std::atomic<bool> g_maskop_traced{false};

// ACTUAL kept pages, summed over every masked (token row), so the nominal `keep` in the trace line can be compared with what
// the head-combination rule really attended. Printed once at process exit when GGML_OFFLOAD_ATTN_TRACE_KEPT=1.
std::atomic<uint64_t> g_kept_pages_sum{0};
std::atomic<uint64_t> g_kept_rows{0};
std::atomic<uint64_t> g_kept_npages_sum{0};
struct KeptReport {
    ~KeptReport() {
        const char * e = std::getenv("GGML_OFFLOAD_ATTN_TRACE_KEPT");
        const uint64_t rows = g_kept_rows.load();
        if (e && e[0] == '1' && rows > 0) {
            fprintf(stderr, "offload-attn kept: rows=%llu mean_kept_pages=%.3f mean_pages=%.3f mean_kept_fraction=%.4f\n",
                    (unsigned long long) rows, (double) g_kept_pages_sum.load() / (double) rows,
                    (double) g_kept_npages_sum.load() / (double) rows, (double) g_kept_pages_sum.load() / (double) g_kept_npages_sum.load());
        }
    }
} g_kept_report;

// Every q.k product the FA op computes (exact and bisect scans alike), so the cost of a scan mode is measured, not assumed.
std::atomic<uint64_t> g_keys_scored{0};

// MASK REFRESH (OffloadConfig::refresh_interval / GGML_OFFLOAD_ATTN_REFRESH = R). The page selection of the last full scan is reused for
// single-token decode steps until it is R steps old, so the scan -- the whole cost of this op -- runs once per R tokens per layer. Behaviour
// spec from HiP (DeepAuto-AI/hip-attention hip_attn/v1_2/mask_refresh_interval.py:9-30: refresh when the decode index is a multiple of the
// interval, default intervals [64, 16, 8]; FSL-1.1-MIT, behaviour only, no code copied). Here the chain is checked, not assumed: a step
// reuses only if the previous step's attendable keys are all still attendable and exactly one key was added (same sequence, next token);
// anything else (a new prompt, a second sequence, a multi-token batch) triggers a full scan. Keyed by (config, key-view base address):
// the graph shares one config across layers, but each layer's K view has its own base address.
struct RefreshState {
    std::vector<char> topk;      // pages chosen by the last full scan (before the window additions)
    std::vector<char> att;       // attendable keys of the previous step's row
    int64_t n_att = 0;
    int64_t age   = 0;           // steps since the last full scan
};
std::mutex g_refresh_mu;
std::map<std::pair<const void *, const void *>, RefreshState> g_refresh;
std::atomic<uint64_t> g_refresh_full{0};
std::atomic<uint64_t> g_refresh_reused{0};

// DECODE KEY SCAN ACROSS THREADS (n_tok == 1, exact scan). Rows are partitioned over threads, so a one-row decode left every
// thread but ith 0 idle while ith 0 scored all heads serially (S69-ENGREV-HIPATTENTION-MASKOP-PARALLEL-SCAN; idea only from
// InfiniteHiP App. A / FlashDecode key-sequence parallelism, no code copied). Here the heads are split over ALL threads; each writes
// its heads' page scores into a shared buffer. ggml runs a custom op's threads with NO barrier between them, so the reduction
// (per-head top-k union or max log-softmax, then the mask write) is done by the LAST thread to arrive: an acq_rel counter
// orders every score write (and thread 0's copy of row 0) before it. State is keyed by dst->data, so concurrent contexts do
// not share a buffer.
struct ParScan {
    std::atomic<int> arrived{0};
    std::vector<float> buf;      // [n_head][nP] page scores
};
std::mutex g_par_mu;
std::map<const void *, std::shared_ptr<ParScan>> g_par;
struct ScanReport {
    ~ScanReport() {
        const char * e = std::getenv("GGML_OFFLOAD_ATTN_TRACE_KEPT");
        if (!(e && e[0] == '1')) return;
        if (g_refresh_full.load() + g_refresh_reused.load() > 0) {
            fprintf(stderr, "offload-attn refresh: full_scans=%llu reused=%llu\n",
                    (unsigned long long) g_refresh_full.load(), (unsigned long long) g_refresh_reused.load());
        }
        if (g_keys_scored.load() > 0) fprintf(stderr, "offload-attn scan: keys_scored=%llu\n", (unsigned long long) g_keys_scored.load());
    }
} g_scan_report;

namespace {

inline float elem(const void * base, ggml_type t, int64_t idx) {
    return t == GGML_TYPE_F16
         ? ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(base)[idx])
         : static_cast<const float *>(base)[idx];
}

// Element (i0, i1, i2) of a possibly NON-contiguous F32/F16 tensor, addressed through nb[]. q and k reach this op as VIEWS --
// q = ggml_permute(q_cur, 0, 2, 1, 3) and k = a strided ggml_view_3d of k_cache (src/llama-build-context.cpp, FA branch of
// llm_build_kqv) -- so a flat (h * n_tok + t) * hd + d index reads the wrong (token, head) / (cell, kv head) pairs.
inline float elem3(const ::ggml_tensor * x, int64_t i0, int64_t i1, int64_t i2) {
    const char * p = static_cast<const char *>(x->data) + i0 * x->nb[0] + i1 * x->nb[1] + i2 * x->nb[2];
    return x->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(p))
                                    : *reinterpret_cast<const float *>(p);
}

inline void put(void * base, ggml_type t, int64_t idx, float v) {
    if (t == GGML_TYPE_F16) static_cast<ggml_fp16_t *>(base)[idx] = ggml_fp32_to_fp16(v);
    else                    static_cast<float *>(base)[idx] = v;
}

inline bool host_ok(const ::ggml_tensor * t) {
    return t && t->data && t->buffer && ggml_backend_buffer_is_host(t->buffer);
}

inline bool typed(const ::ggml_tensor * t) {
    return t && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
}

// K may also be a QUANTISED cache (--cache-type-k q8_0 etc.). Before, typed(b) refused it and the op passed the mask through
// untouched and silently (only g_maskop_passthrough moved), so the flag did nothing for exactly the cache types people use to fit
// long contexts. A key row is dequantised with the type's own to_float. Needs the row to be whole blocks (hd % blck == 0) and the
// element stride to be one block (nb[0] == type size), the layout a k_cache view has.
inline bool k_typed(const ::ggml_tensor * t) {
    if (typed(t)) return true;
    if (!t || !ggml_is_quantized(t->type)) return false;
    const ggml_type_traits * tr = ggml_get_type_traits(t->type);
    return tr && tr->to_float && t->ne[0] % ggml_blck_size(t->type) == 0 && t->nb[0] == ggml_type_size(t->type);
}

// Key (s, kv head kh) of b as hd floats in out. F32/F16 go through elem3; a quantised row is dequantised once.
inline void k_row(const ::ggml_tensor * b, int64_t s, int64_t kh, float * out) {
    const int64_t hd = b->ne[0];
    if (typed(b)) {
        for (int64_t d = 0; d < hd; ++d) out[d] = elem3(b, d, s, kh);
        return;
    }
    const char * p = static_cast<const char *>(b->data) + s * b->nb[1] + kh * b->nb[2];
    ggml_get_type_traits(b->type)->to_float(p, out, hd);
}

// HIERARCHICAL PAGE SCAN (GGML_OFFLOAD_ATTN_SCAN=bisect). Behaviour spec from HiP (DeepAuto-AI/hip-attention, arXiv 2406.09827 and
// 2502.08910; FSL-1.1-MIT, behaviour only, no code copied): split the keys into chunks; inside each chunk descend by halves, scoring each
// half by ONE representative key (its midpoint) and following the better half, so a chunk is ranked from 2*log2(chunk/page) keys instead
// of all of them; keep the best chunks; then score pages EXACTLY inside the kept chunks only. An APPROXIMATION: a key that is never a
// sampled midpoint cannot lift its chunk, so recall is measured (offload-attn-recall), not assumed.
// Writes score[p] for one head: -INFINITY for pages outside the kept chunks, the exact max of attendable keys otherwise.
template <typename DotFn, typename AttFn>
void bisect_page_scores(int64_t n_kv, int64_t P, int64_t nP, int32_t keep, int64_t chunk, int64_t chunk_mult, int32_t sink,
                        int64_t qpos, int64_t recent_pages, DotFn dot, AttFn attendable, std::vector<float> & score, std::vector<float> & est,
                        uint64_t & keys_scored) {
    const int64_t n_chunks = (n_kv + chunk - 1) / chunk;
    std::vector<float> cs((size_t) n_chunks, -INFINITY);
    const int64_t pages_per_chunk = std::max<int64_t>(1, chunk / P);
    int iters = 0;
    while ((int64_t(1) << iters) < pages_per_chunk) ++iters;
    for (int64_t ch = 0; ch < n_chunks; ++ch) {
        int64_t lo = ch * chunk, hi = std::min(n_kv, lo + chunk);
        while (lo < hi && !attendable(lo)) ++lo;          // the chunk's attendable span; a fully forbidden chunk keeps -inf
        while (hi > lo && !attendable(hi - 1)) --hi;
        if (lo >= hi) continue;
        float best = -INFINITY;
        for (int it = 0; it < iters && hi - lo > 1; ++it) {
            const int64_t mid = lo + (hi - lo) / 2;
            const int64_t ml = lo + (mid - lo) / 2, mr = mid + (hi - mid) / 2;
            const float sl = attendable(ml) ? dot(ml) : -INFINITY;
            const float sr = attendable(mr) ? dot(mr) : -INFINITY;
            keys_scored += (uint64_t) attendable(ml) + (uint64_t) attendable(mr);
            best = std::max(best, std::max(sl, sr));
            if (sl >= sr) hi = mid; else lo = mid;
        }
        if (!(best > -INFINITY)) { best = dot(lo); ++keys_scored; }   // lo is attendable (span trimmed above)
        cs[(size_t) ch] = best;
    }
    const int64_t k1 = std::min<int64_t>(n_chunks, std::max<int64_t>(1, (chunk_mult * keep * P + chunk - 1) / chunk));
    std::vector<char> kc((size_t) n_chunks, 0);
    std::vector<int64_t> ord((size_t) n_chunks);
    std::iota(ord.begin(), ord.end(), 0);
    std::partial_sort(ord.begin(), ord.begin() + k1, ord.end(), [&cs](int64_t x, int64_t y) { return cs[(size_t) x] > cs[(size_t) y]; });
    for (int64_t i = 0; i < k1; ++i) kc[(size_t) ord[(size_t) i]] = 1;
    for (int64_t p = 0; p < sink && p < nP; ++p) kc[(size_t) (p * P / chunk)] = 1;          // pinned pages stay exactly scored
    if (recent_pages > 0 && qpos >= 0) {
        const int64_t pg = std::min<int64_t>(nP - 1, qpos / P);
        for (int64_t p = std::max<int64_t>(0, pg - recent_pages + 1); p <= pg; ++p) kc[(size_t) (p * P / chunk)] = 1;
    }
    std::fill(score.begin(), score.end(), -INFINITY);
    // est = what MAX mode normalises over: the exact page score where one was computed, else the page's chunk's sampled score.
    est.assign((size_t) nP, -INFINITY);
    for (int64_t p = 0; p < nP; ++p) est[(size_t) p] = cs[(size_t) std::min<int64_t>(n_chunks - 1, p * P / chunk)];
    for (int64_t p = 0; p < nP; ++p) {
        if (!kc[(size_t) (p * P / chunk)]) continue;
        float best = -INFINITY;
        for (int64_t s = p * P; s < std::min(n_kv, (p + 1) * P); ++s) {
            if (!attendable(s)) continue;
            best = std::max(best, dot(s));
            ++keys_scored;
        }
        score[(size_t) p] = best;
        est[(size_t) p]   = best;
    }
}

}  // namespace

// ggml_custom3_op_t: dst = masked(a), selecting from b (keys) and c (query).
//   a = kq_mask [n_kv, n_tokens_padded]
//   b = k       [head_dim, n_kv, n_head_kv]
//   c = q       [head_dim, n_tokens, n_head]
// a must be contiguous; b and c are graph VIEWS (permuted q, strided k_cache view) and are read through nb[] (elem3).
void mask_from_kq_op(::ggml_tensor * dst, const ::ggml_tensor * a, const ::ggml_tensor * b,
                     const ::ggml_tensor * c, int ith, int nth, void * userdata) {
    g_maskop_calls.fetch_add(1, std::memory_order_relaxed);

    const OffloadConfig * cfg = static_cast<const OffloadConfig *>(userdata);
    const int64_t n_kv   = a->ne[0];
    const int64_t n_rows = a->ne[1];          // PADDED height
    const int64_t hd     = b->ne[0];
    const int64_t n_head = c ? c->ne[2] : 0;
    const int64_t n_tok  = c ? c->ne[1] : 0;

    const bool usable = cfg && typed(a) && typed(dst) && k_typed(b) && typed(c) &&
                        host_ok(a) && host_ok(dst) && host_ok(b) && host_ok(c) &&
                        // a and dst are walked with flat indices (copy loop, t * n_kv + s), so they must be contiguous;
                        // b and c are read through nb[] and may be views
                        ggml_is_contiguous(a) && ggml_is_contiguous(dst) && c->ne[0] == hd &&
                        n_kv > 0 && hd > 0 && n_head > 0 && n_tok > 0 && n_tok <= n_rows &&
                        b->ne[1] >= n_kv &&
                        // the attention kernel itself asserts n_head % n_kv_head == 0; pass the mask through
                        // (fail open, counted) instead of inventing a head mapping for a shape it cannot run
                        b->ne[2] > 0 && n_head % b->ne[2] == 0;

    // ONE ROW PARTITION for the copy AND the masking: thread ith owns the padded rows [r0, r1), copies exactly those and masks
    // only the real query rows among them. The copy used to be split by ELEMENT and the masking by TOKEN, two different
    // partitions, and ggml runs a custom op's threads with no barrier between them -- so a thread could copy the original mask
    // over a row another thread had already masked, silently un-masking it (fail OPEN, uncounted). Every element of dst is still
    // written exactly once, by the thread that owns its row.
    const int64_t per_r = (n_rows + nth - 1) / nth;
    const int64_t r0 = std::min(n_rows, (int64_t) ith * per_r);
    const int64_t r1 = std::min(n_rows, r0 + per_r);
    const int64_t e0 = r0 * n_kv;
    const int64_t e1 = r1 * n_kv;

    if (!usable) {
        if (ith == 0) g_maskop_passthrough.fetch_add(1, std::memory_order_relaxed);
        if (dst->data != a->data) {
            for (int64_t i = e0; i < e1; ++i) put(dst->data, dst->type, i, elem(a->data, a->type, i));
        }
        return;
    }

    // Page count and kept count come from pages_kept() so this op cannot drift from the non-FA op
    // or from the graph-build site that decides whether to splice it at all.
    const int32_t P  = cfg->page_size > 0 ? cfg->page_size : 16;
    int64_t nP = 0;
    const int32_t keep = pages_kept(*cfg, n_kv, &nP);

    // Leading KV pages pinned regardless of score -- see the matching comment in
    // offload_attention_op.cpp. THIS FILE IS mask_from_kq_op = the FLASH-ATTENTION path, and it
    // is the one that actually executes when the server runs with -fa on (MEASURED 2026-08-21).
    // Both graph paths need the pin independently, because each carries its own top-k rather
    // than sharing one.
    int32_t sink = cfg->sink_pages;
    if (sink < 0)    sink = 0;
    if (sink > keep) sink = keep;

    const bool identity = keep >= (int32_t) nP;

    // Emitted BEFORE the identity short-circuit: the no-op case is exactly the one that must stay
    // visible, since an op that returns early without a trace is indistinguishable from an op
    // that never ran -- the ambiguity that made the first sink A/B unreadable.
    if (ith == 0 && !g_maskop_traced.exchange(true)) {
        fprintf(stderr,
                "offload-attn op(FA/mask_from_kq_op): n_kv=%lld page=%d nP=%lld keep=%d sink=%d"
                " min_tokens=%d -> %.1f%% of KV kept%s\n",
                (long long) n_kv, (int) P, (long long) nP, (int) keep, (int) sink,
                (int) cfg->min_kept_tokens,
                100.0 * (double) keep / (double) (nP ? nP : 1),
                identity ? "  [IDENTITY: nothing masked, op short-circuited]" : "");
    }

    // Copy everything first, including the padded rows, so nothing is left uninitialised and the
    // padding keeps whatever the builder put there.
    if (dst->data != a->data) {
        for (int64_t i = e0; i < e1; ++i) put(dst->data, dst->type, i, elem(a->data, a->type, i));
    }

    // NOTHING WOULD BE MASKED. keep == nP means every page survives, so the mask is the identity
    // and the per-head scoring loop -- which recomputes q.k dot products for EVERY key, the most
    // expensive thing this op does -- is pure cost. A token floor large enough to protect short
    // prompts makes keep == nP for every short prompt, so without this the flag charges a
    // MEASURED 7.4 vs 38 tok/s for provably no change in output. The copy above has already run,
    // so dst is complete and returning here is safe.
    if (identity) {
        if (ith == 0) g_maskop_passthrough.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    if (ith == 0) {
        g_maskop_masked.fetch_add(1, std::memory_order_relaxed);
        g_maskop_last_keep.store((uint64_t) keep, std::memory_order_relaxed);
        g_maskop_last_npages.store((uint64_t) nP, std::memory_order_relaxed);
    }

    // Then mask only the REAL query rows of this thread's own row range (the same partition the copy used).
    int64_t t0 = std::min(r0, n_tok);
    int64_t t1 = std::min(r1, n_tok);

    const float inv = 1.0f / std::sqrt((float) hd);
    std::vector<float> score((size_t) nP);
    std::vector<float> est;   // bisect only: the MAX-mode normaliser (see bisect_page_scores)
    std::vector<int32_t> order((size_t) nP);
    std::vector<char> keep_page((size_t) nP);
    std::vector<float> agg((size_t) nP);
    std::vector<float> qrow((size_t) hd);
    std::vector<float> krow((size_t) hd);

    // 0 = union of per-head top-k, 1 = max over heads of log-softmax page scores then one top-k (see OffloadConfig::head_reduce).
    // Score only attendable keys (default ON). GGML_OFFLOAD_ATTN_SCORE_ALL=1 restores the old behaviour of scoring every key, kept so the two can be measured against each other.
    const char * sa_env = std::getenv("GGML_OFFLOAD_ATTN_SCORE_ALL");
    const bool score_attendable_only = !(sa_env && sa_env[0] == '1');
    int64_t recent_pages = cfg->recent_pages;
    if (const char * rp = std::getenv("GGML_OFFLOAD_ATTN_RECENT_PAGES")) {
        if (rp[0] != 0) recent_pages = std::max<int64_t>(0, std::atoll(rp));
    }
    int head_reduce = cfg->head_reduce;
    if (const char * e = std::getenv("GGML_OFFLOAD_ATTN_HEAD_REDUCE")) {
        if (std::strcmp(e, "max") == 0)        head_reduce = 1;
        else if (std::strcmp(e, "union") == 0) head_reduce = 0;
    }

    // Scan mode: exact (default) scores every attendable key; bisect = the hierarchical scan above (OffloadConfig::scan_mode or
    // GGML_OFFLOAD_ATTN_SCAN=bisect|exact; GGML_OFFLOAD_ATTN_SCAN_CHUNK = first-stage chunk in keys, default 256).
    int scan_mode = cfg->scan_mode;
    if (const char * e = std::getenv("GGML_OFFLOAD_ATTN_SCAN")) {
        if (std::strcmp(e, "bisect") == 0)     scan_mode = 1;
        else if (std::strcmp(e, "exact") == 0) scan_mode = 0;
    }
    int64_t scan_chunk = 256;
    if (const char * e = std::getenv("GGML_OFFLOAD_ATTN_SCAN_CHUNK")) {
        if (e[0] != 0) scan_chunk = std::max<int64_t>(P, std::atoll(e));
    }
    int64_t refresh = cfg->refresh_interval;
    if (const char * e = std::getenv("GGML_OFFLOAD_ATTN_REFRESH")) {
        if (e[0] != 0) refresh = std::max<int64_t>(1, std::atoll(e));
    }
    const std::pair<const void *, const void *> rkey(static_cast<const void *>(cfg), b->data);
    if (refresh > 1 && n_tok != 1 && ith == 0) {
        // a multi-token batch (prefill, several sequences) breaks the decode chain: forget this layer's selection
        std::lock_guard<std::mutex> lk(g_refresh_mu);
        g_refresh.erase(rkey);
    }
    const bool refresh_on = refresh > 1 && n_tok == 1;
    uint64_t keys_scored = 0;

    // Exact page scores of query row t for one head (qr already holds that head's query): best attendable q.k per page.
    // ONE body for the serial and the parallel path, so they cannot drift.
    auto score_exact = [&](int64_t t, int64_t kh, const std::vector<float> & qr, std::vector<float> & kr, float * sc, uint64_t & ks) {
        for (int64_t p = 0; p < nP; ++p) {
            const int64_t s0 = p * P;
            const int64_t s1 = std::min<int64_t>(s0 + P, n_kv);
            float best = -INFINITY;
            for (int64_t s = s0; s < s1; ++s) {
                // A key the attention mask already forbids for this query row (causal future inside a prefill batch, a sliding-window
                // exclusion, a padded slot) can never be attended, so it must not be able to win the page's score and spend keep budget.
                if (score_attendable_only && elem(a->data, a->type, t * n_kv + s) == -INFINITY) continue;
                ++ks;
                k_row(b, s, kh, kr.data());      // converted once per key, then a plain float dot
                double dot = 0.0;
                for (int64_t d = 0; d < hd; ++d) dot += (double) qr[(size_t) d] * (double) kr[(size_t) d];
                best = std::max(best, (float) dot * inv);
            }
            sc[p] = best;
        }
    };

    // PARALLEL DECODE SCAN: one real row, exact scan, no refresh chain. GGML_OFFLOAD_ATTN_SCAN_PAR=0 keeps the serial scan (reference).
    bool par_scan = n_tok == 1 && nth > 1 && scan_mode == 0 && !refresh_on && n_head > 1;
    if (const char * e = std::getenv("GGML_OFFLOAD_ATTN_SCAN_PAR")) {
        if (e[0] == '0') par_scan = false;
    }
    std::shared_ptr<ParScan> ps;
    if (par_scan) {
        {
            std::lock_guard<std::mutex> lk(g_par_mu);
            std::shared_ptr<ParScan> & slot = g_par[dst->data];
            if (!slot) {
                slot = std::make_shared<ParScan>();
                slot->buf.assign((size_t) n_head * (size_t) nP, -INFINITY);
            }
            ps = slot;
        }
        const int64_t ph = (n_head + nth - 1) / nth;
        const int64_t h0 = std::min<int64_t>(n_head, (int64_t) ith * ph);
        const int64_t h1 = std::min<int64_t>(n_head, h0 + ph);
        for (int64_t h = h0; h < h1; ++h) {
            const int64_t kh = h / (n_head / b->ne[2]);
            for (int64_t d = 0; d < hd; ++d) qrow[(size_t) d] = elem3(c, d, 0, h);
            score_exact(0, kh, qrow, krow, ps->buf.data() + (size_t) h * (size_t) nP, keys_scored);
        }
        // thread 0 owns row 0: its copy ran above, so this arrival also publishes it
        if (ps->arrived.fetch_add(1, std::memory_order_acq_rel) + 1 < nth) {
            g_keys_scored.fetch_add(keys_scored, std::memory_order_relaxed);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(g_par_mu);
            g_par.erase(dst->data);
        }
        t0 = 0;   // the LAST arrival reduces and masks row 0, whichever thread it is
        t1 = 1;
    }

    for (int64_t t = t0; t < t1; ++t) {
        std::fill(keep_page.begin(), keep_page.end(), (char) 0);
        std::fill(agg.begin(), agg.end(), -INFINITY);

        // The query's position is its LAST ATTENDABLE key, read from the mask row. n_kv is the PADDED cache height
        // (GGML_PAD(used cells, 256) under flash attention), so n_kv - n_tok + t lands on padding pages in decode and the
        // recent window then missed the query's own page. A row with no attendable key has nothing to keep.
        int64_t qpos = -1;
        for (int64_t s = n_kv - 1; s >= 0; --s) {
            if (elem(a->data, a->type, t * n_kv + s) != -INFINITY) { qpos = s; break; }
        }
        auto attendable = [&](int64_t s) { return elem(a->data, a->type, t * n_kv + s) != -INFINITY; };

        // REFRESH reuse: only single-token rows get here with refresh_on, so this is row 0 and one thread.
        std::vector<char> att_now;
        int64_t n_att_now = 0;
        bool reused = false;
        if (refresh_on) {
            att_now.assign((size_t) n_kv, 0);
            for (int64_t s = 0; s < n_kv; ++s) {
                const bool ok = attendable(s);
                att_now[(size_t) s] = ok;
                n_att_now += ok;
            }
            std::lock_guard<std::mutex> lk(g_refresh_mu);
            auto it = g_refresh.find(rkey);
            if (it != g_refresh.end()) {
                RefreshState & st = it->second;
                bool chain = st.age < refresh && n_att_now == st.n_att + 1 && (int64_t) st.topk.size() == nP;
                for (size_t s = 0; chain && s < st.att.size(); ++s) {
                    if (st.att[s] && (s >= (size_t) n_kv || !att_now[s])) chain = false;   // every old key must still be attendable
                }
                if (chain) {
                    for (int64_t p = 0; p < nP; ++p) keep_page[(size_t) p] = st.topk[(size_t) p];
                    for (int64_t s = 0; s < n_kv; ++s) {
                        // the one newly attendable key (the token just written) is kept with its page
                        if (att_now[(size_t) s] && (s >= (int64_t) st.att.size() || !st.att[(size_t) s])) keep_page[(size_t) (s / P)] = 1;
                    }
                    for (int64_t p = 0; p < sink && p < nP; ++p) keep_page[(size_t) p] = 1;
                    st.att = att_now;
                    st.n_att = n_att_now;
                    st.age += 1;
                    reused = true;
                }
            }
        }

        if (!reused) {
        // UNION over heads: a page survives if ANY head wants it. The mask is shared, so this is
        // the only selection that cannot silently starve a head of the keys it needed.
        for (int64_t h = 0; h < n_head; ++h) {
            // GQA: consecutive query heads share one kv head -- the SAME mapping the attention kernel uses
            // (ggml.c flash-attn `ik2 = iq2 / rk2`, rk2 = n_head / n_kv_head; iqk_flash_attn.cpp likewise).
            // This was `h % n_kv_head` (interleaved), which scored a query head against a kv head it never
            // attends to whenever n_head != n_kv_head. Identical for MHA, which is why nothing noticed.
            // `usable` guarantees n_head % n_kv_head == 0.
            const int64_t kh = h / (n_head / b->ne[2]);
            for (int64_t d = 0; d < hd; ++d) qrow[(size_t) d] = elem3(c, d, t, h);
            std::fill(score.begin(), score.end(), -INFINITY);
            if (par_scan) {
                std::copy(ps->buf.begin() + (size_t) h * (size_t) nP, ps->buf.begin() + (size_t) (h + 1) * (size_t) nP, score.begin());
            } else if (scan_mode == 1) {
                // q and k are VIEWS: read them through nb[] (qrow, k_row), not a flat (h * n_tok + t) * hd index, which read
                // the wrong (head, token) / (kv head, cell) pairs for any non-contiguous layout.
                auto dot_s = [&](int64_t s) -> float {
                    k_row(b, s, kh, krow.data());
                    double dot = 0.0;
                    for (int64_t d = 0; d < hd; ++d) dot += (double) qrow[(size_t) d] * (double) krow[(size_t) d];
                    return (float) dot * inv;
                };
                bisect_page_scores(n_kv, P, nP, keep, scan_chunk, 4, sink, qpos, recent_pages, dot_s, attendable, score, est, keys_scored);
            }
            if (!par_scan && scan_mode != 1) score_exact(t, kh, qrow, krow, score.data(), keys_scored);
            if (head_reduce == 1) {
                // MAX mode: fold this head into the per-page aggregate as log-softmax over pages, so heads are comparable.
                // The bisect scan normalises over est (every page's estimate), not over the few exactly scored pages: a head that
                // scored 16 pages would otherwise get inflated log-probs and outvote heads normalised over everything.
                const std::vector<float> & nrm = scan_mode == 1 ? est : score;
                float mx = -INFINITY;
                for (int64_t p = 0; p < nP; ++p) mx = std::max(mx, nrm[(size_t) p]);
                if (!(mx > -INFINITY)) continue;   // every key of this row is masked for this head: nothing to rank (also avoids -inf - -inf = NaN)
                double z = 0.0;
                for (int64_t p = 0; p < nP; ++p) z += std::exp((double) (nrm[(size_t) p] - mx));
                const float lse = mx + (float) std::log(z);
                for (int64_t p = 0; p < nP; ++p) agg[(size_t) p] = std::max(agg[(size_t) p], score[(size_t) p] - lse);
                continue;
            }
            // iota leaves order[0..sink) holding pages 0..sink-1, so ranking only the tail pins
            // the sink pages without a second pass. Done per head, INSIDE the union, so the
            // sink survives for every head rather than only the one that happened to rank it.
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin() + sink, order.begin() + keep, order.end(),
                              [&score](int32_t x, int32_t y) {
                                  return score[(size_t) x] > score[(size_t) y];
                              });
            for (int32_t i = 0; i < keep; ++i) keep_page[(size_t) order[(size_t) i]] = 1;
        }

        if (head_reduce == 1) {
            // ONE top-k over the max-aggregated scores; the sink pages are pinned the same way as in the union path.
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin() + sink, order.begin() + keep, order.end(),
                              [&agg](int32_t x, int32_t y) {
                                  return agg[(size_t) x] > agg[(size_t) y];
                              });
            for (int32_t i = 0; i < keep; ++i) keep_page[(size_t) order[(size_t) i]] = 1;
        }

        if (refresh_on) {
            std::lock_guard<std::mutex> lk(g_refresh_mu);
            RefreshState & st = g_refresh[rkey];
            st.topk.assign(keep_page.begin(), keep_page.end());
            st.att   = att_now;
            st.n_att = n_att_now;
            st.age   = 1;
        }
        }   // !reused
        if (refresh_on) (reused ? g_refresh_reused : g_refresh_full).fetch_add(1, std::memory_order_relaxed);

        if (recent_pages > 0 && qpos >= 0) {
            const int64_t pg = std::min<int64_t>(nP - 1, qpos / P);
            for (int64_t p = std::max<int64_t>(0, pg - recent_pages + 1); p <= pg; ++p) keep_page[(size_t) p] = 1;
        }
        {
            uint64_t kp = 0;
            for (int64_t p = 0; p < nP; ++p) kp += keep_page[(size_t) p] ? 1 : 0;
            g_kept_pages_sum.fetch_add(kp, std::memory_order_relaxed);
            g_kept_npages_sum.fetch_add((uint64_t) nP, std::memory_order_relaxed);
            g_kept_rows.fetch_add(1, std::memory_order_relaxed);
        }
        for (int64_t s = 0; s < n_kv; ++s) {
            if (!keep_page[(size_t) (s / P)]) {
                put(dst->data, dst->type, t * n_kv + s, -INFINITY);
            }
        }
    }
    g_keys_scored.fetch_add(keys_scored, std::memory_order_relaxed);
}

uint64_t maskop_calls()       { return g_maskop_calls.load(std::memory_order_relaxed); }
uint64_t maskop_masked()      { return g_maskop_masked.load(std::memory_order_relaxed); }
uint64_t maskop_passthrough() { return g_maskop_passthrough.load(std::memory_order_relaxed); }
uint64_t maskop_last_keep()   { return g_maskop_last_keep.load(std::memory_order_relaxed); }
uint64_t maskop_last_npages() { return g_maskop_last_npages.load(std::memory_order_relaxed); }
uint64_t maskop_keys_scored() { return g_keys_scored.load(std::memory_order_relaxed); }
uint64_t maskop_refresh_full()   { return g_refresh_full.load(std::memory_order_relaxed); }
uint64_t maskop_refresh_reused() { return g_refresh_reused.load(std::memory_order_relaxed); }

}  // namespace ik_offload

extern "C" {
uint64_t offload_attn_maskop_calls()       { return ik_offload::maskop_calls(); }
uint64_t offload_attn_maskop_masked()      { return ik_offload::maskop_masked(); }
uint64_t offload_attn_maskop_passthrough() { return ik_offload::maskop_passthrough(); }
uint64_t offload_attn_maskop_last_keep()   { return ik_offload::maskop_last_keep(); }
uint64_t offload_attn_maskop_last_npages() { return ik_offload::maskop_last_npages(); }
}
