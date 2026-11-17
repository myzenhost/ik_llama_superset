// offload_attention_op.cpp — the EXECUTION-TIME half of #220.
//
// WHY AN OP AND NOT A CALL SITE. llama-build-context.cpp is graph CONSTRUCTION: ggml_mul_mat and
// ggml_soft_max_ext create nodes, and the K/V cache contents do not exist there -- they are filled
// when the graph runs. A host-side call to offload_attend_step() in the builder would read an
// unpopulated cache. So the selection has to happen inside an operator that executes with real
// data, which is what ggml_map_custom2 provides.
//
// WHAT IT DOES. Spliced between kq = ggml_mul_mat(k, q) and the softmax, it masks the attention
// scores of pages the selector did not choose to -INFINITY. Softmax then gives them exactly zero
// weight. This mirrors the harness reference (cortex/inference/offload_attention.py
// attach_to_hf_model), which also masks rather than gathers -- masking keeps the graph shape
// identical, so nothing downstream needs to change.
//
// Masking is not the memory win on its own; the win comes when the gather kernel stops reading
// unselected pages (#220 kernel compose / #93 ggml_paged_attention). This is the correctness
// stage: it makes the selector's decision affect the model's output, on a path that runs today.
//
// FAIL-OPEN, DELIBERATELY. Any condition the op cannot handle -- unsupported dtype, a device
// pointer, a shape it does not recognise -- copies the scores through UNCHANGED. A wrong mask
// silently corrupts every answer; an absent mask is merely the current behaviour. The op counts
// how often it passed through and why, and that counter is readable, because a fail-open path
// that is never observed is how "enabled" becomes indistinguishable from "off".

#include "ggml.h"
#include "ggml-backend.h"
#include "offload_attention.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace ik_offload {

// Observability. Fail-open without a counter is the failure this whole task is about.
std::atomic<uint64_t> g_op_calls{0};
std::atomic<uint64_t> g_op_masked{0};
std::atomic<uint64_t> g_op_passthrough{0};
// The budget produced no OUTPUT change across 0.90/0.10/0.02. Output text cannot say
// whether the budget reached the op at all, so record what the op actually computed.
std::atomic<uint64_t> g_last_keep{0};
std::atomic<uint64_t> g_last_npages{0};
std::atomic<uint64_t> g_last_budget_milli{0};
// One-shot execution trace. The counters above are readable only through an accessor that ONLY
// the offload-attn-test example calls, so during serving keep/nP were invisible -- and keep is
// the number that decides whether this is attention-offload or a lobotomy.
std::atomic<bool> g_op_traced{false};

namespace {

inline float elem_at(const void * base, KVType t, int64_t idx) {
    if (t == KVType::F16) {
        return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(base)[idx]);
    }
    return static_cast<const float *>(base)[idx];
}

inline bool host_readable(const struct ggml_tensor * t) {
    return t && t->data && t->buffer && ggml_backend_buffer_is_host(t->buffer);
}

inline bool dtype_of(const struct ggml_tensor * t, KVType & out) {
    if (t->type == GGML_TYPE_F32) { out = KVType::F32; return true; }
    if (t->type == GGML_TYPE_F16) { out = KVType::F16; return true; }
    return false;
}

}  // namespace

// ggml_custom2_op_t: dst = masked(a), using b (the key cache) to score pages.
//   a  = kq scores, ne = [n_kv, n_tokens, n_head, 1], F32
//   b  = key cache rows, ne[0] = row stride in elements
//   userdata = const OffloadConfig *
static void mask_op_impl(::ggml_tensor * dst, const ::ggml_tensor * a, const ::ggml_tensor * b, const ::ggml_tensor * kq_mask,
                         int ith, int nth, void * userdata) {
    g_op_calls.fetch_add(1, std::memory_order_relaxed);

    const OffloadConfig * cfg = static_cast<const OffloadConfig *>(userdata);
    float * out = static_cast<float *>(dst->data);
    const float * scores = static_cast<const float *>(a->data);

    const int64_t n_kv     = a->ne[0];
    const int64_t n_tokens = a->ne[1];
    const int64_t n_head   = a->ne[2];
    const int64_t total    = n_kv * n_tokens * n_head;

    KVType ktype;
    const bool usable =
        cfg && a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        host_readable(a) && host_readable(dst) && host_readable(b) &&
        // a and dst are walked with flat indices below; a view input (permuted / strided) would be read at the wrong
        // elements -- the defect mask_from_kq_op had (S69-FORK-OFFLOAD-ATTN-FA-MASKOP-IGNORES-VIEW-STRIDES). Fail open.
        ggml_is_contiguous(a) && ggml_is_contiguous(dst) &&
        dtype_of(b, ktype) && n_kv > 0 && b->ne[0] > 0;

    if (!usable) {
        // PASS THROUGH unchanged. Copying is the whole behaviour: the graph keeps its shape and
        // the model computes exactly what it would have without this op.
        if (ith == 0) g_op_passthrough.fetch_add(1, std::memory_order_relaxed);
        if (out != scores) {
            const int64_t per = (total + nth - 1) / nth;
            const int64_t lo = std::min(total, (int64_t) ith * per);
            const int64_t hi = std::min(total, lo + per);
            for (int64_t i = lo; i < hi; ++i) out[i] = scores[i];
        }
        return;
    }

    // Page count and kept count come from pages_kept() so this op cannot drift from the FA op or
    // from the graph-build site that decides whether to splice it at all.
    const int32_t P  = cfg->page_size > 0 ? cfg->page_size : 16;
    int64_t nP = 0;
    const int32_t keep = pages_kept(*cfg, n_kv, &nP);

    // Leading KV pages pinned regardless of score. An attention sink attracts attention mass
    // WITHOUT being semantically relevant (StreamingLLM, arXiv 2309.17453v3), so ANY relevance
    // ranking discards it by construction. The sink spends part of the budget rather than adding
    // to it, so the kept page count is unchanged and this remains a pure quality knob.
    // THIS FILE IS mask_op = the NON-flash-attention path, spliced on the materialised kq.
    // (The similarly named offload_attention_mask_op.cpp holds mask_from_kq_op, which is the FA
    // path -- it recomputes scores from k and q because flash-attn never materialises kq. The
    // file names invite exactly the swap that mislabelled the first execution trace.)
    // select_pages() in offload_attention.cpp carries the same pinning, but the graph never
    // calls it -- see the note at common/common.h:323. A knob wired only there is dormant in
    // serving, which is how the first sink A/B returned byte-identical output at sink=0 and 1.
    int32_t sink = cfg->sink_pages;
    if (sink < 0)    sink = 0;
    if (sink > keep) sink = keep;

    const bool identity = keep >= (int32_t) nP;

    // Printed ONCE, not per call: this fires per layer per token otherwise. The engagement trace
    // at graph build proves the op was SPLICED; only this proves what it COMPUTES. It is emitted
    // BEFORE the identity short-circuit below, because the no-op case is exactly the one that
    // must stay visible -- an op that returns early without a trace is indistinguishable from an
    // op that never ran, and that ambiguity is what made the first sink A/B unreadable.
    if (ith == 0 && !g_op_traced.exchange(true)) {
        fprintf(stderr,
                "offload-attn op(non-FA/mask_op): n_kv=%lld page=%d nP=%lld keep=%d sink=%d"
                " min_tokens=%d -> %.1f%% of KV kept%s\n",
                (long long) n_kv, (int) P, (long long) nP, (int) keep, (int) sink,
                (int) cfg->min_kept_tokens,
                100.0 * (double) keep / (double) (nP ? nP : 1),
                identity ? "  [IDENTITY: nothing masked, op short-circuited]" : "");
    }

    // NOTHING WOULD BE MASKED. keep == nP means every page survives, so the mask is the identity
    // and the per-row page scoring plus partial_sort are pure cost. This is not a rare corner: a
    // token floor large enough to protect short prompts makes keep == nP for every short prompt,
    // and without this the flag would charge a MEASURED 7.4 vs 38 tok/s for provably no change
    // in output. Take the same pass-through the unsupported path takes, and count it, so
    // "engaged but free" stays observable rather than looking like the op never ran.
    if (identity) {
        if (ith == 0) g_op_passthrough.fetch_add(1, std::memory_order_relaxed);
        if (out != scores) {
            const int64_t per = (total + nth - 1) / nth;
            const int64_t lo = std::min(total, (int64_t) ith * per);
            const int64_t hi = std::min(total, lo + per);
            for (int64_t i = lo; i < hi; ++i) out[i] = scores[i];
        }
        return;
    }

    if (ith == 0) {
        g_op_masked.fetch_add(1, std::memory_order_relaxed);
        g_last_keep.store((uint64_t) keep, std::memory_order_relaxed);
        g_last_npages.store((uint64_t) nP, std::memory_order_relaxed);
        g_last_budget_milli.store((uint64_t) (cfg->page_budget * 1000.0f),
                                  std::memory_order_relaxed);
    }

    // Rows are (token, head) pairs; split them across threads.
    const int64_t n_rows = n_tokens * n_head;
    const int64_t per    = (n_rows + nth - 1) / nth;
    const int64_t r0     = std::min(n_rows, (int64_t) ith * per);
    const int64_t r1     = std::min(n_rows, r0 + per);

    std::vector<float> page_score((size_t) nP);
    std::vector<int32_t> order((size_t) nP);

    // The non-FA path had no recent window: --offload-attn-recent reached only the FA op (48857eab3), so -fa off silently ignored it.
    int64_t recent_pages = cfg->recent_pages;
    if (const char * rp = std::getenv("GGML_OFFLOAD_ATTN_RECENT_PAGES")) {
        if (rp[0] != 0) recent_pages = std::max<int64_t>(0, std::atoll(rp));
    }

    // THE ATTENTION MASK (mask_op_kqmask only). kq here is the RAW score: the causal / sliding-window / padding mask is applied
    // later, inside the softmax (ggml_soft_max_ext(kq, kq_mask, ...)). Without it this op ranked pages by keys the query can never
    // attend -- a causal-future key inside a prefill batch, a padded slot -- which spent keep budget on nothing, and it placed the
    // recent window at n_kv - n_tokens + t, which is a PADDING page whenever n_kv is padded (always, outside a full batch). With it,
    // only attendable keys are scored and the window sits at the query's last attendable key -- the same rule mask_from_kq_op uses.
    // Rows of the mask are tokens, SHARED by every head ([n_kv, n_tokens_padded, 1, 1]). Any other shape: old behaviour.
    const bool mask_ok = kq_mask && (kq_mask->type == GGML_TYPE_F32 || kq_mask->type == GGML_TYPE_F16) && host_readable(kq_mask) &&
                         kq_mask->ne[0] == n_kv && kq_mask->ne[1] >= n_tokens && kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1;
    const KVType mtype = (kq_mask && kq_mask->type == GGML_TYPE_F16) ? KVType::F16 : KVType::F32;
    auto attendable = [&](int64_t t, int64_t s) -> bool {
        return elem_at(kq_mask->data, mtype, t * n_kv + s) != -INFINITY;
    };

    for (int64_t r = r0; r < r1; ++r) {
        const float * srow = scores + r * n_kv;
        float * drow       = out    + r * n_kv;
        const int64_t tok  = r % n_tokens;

        // Score each page by the MAXIMUM score of its members. Scoring the pages from the kq row
        // itself, rather than re-deriving them from the keys, keeps this op consistent with the
        // attention it is masking: the same numbers the model is about to soft-max decide which
        // pages survive. (QUEST's min/max bound is an ESTIMATE of exactly this quantity, used when
        // the true scores are not yet available; here they are.)
        for (int64_t p = 0; p < nP; ++p) {
            const int64_t t0 = p * P;
            const int64_t t1 = std::min<int64_t>(t0 + P, n_kv);
            float m = -INFINITY;
            for (int64_t t = t0; t < t1; ++t) {
                if (mask_ok && !attendable(tok, t)) continue;
                m = std::max(m, srow[t]);
            }
            page_score[(size_t) p] = m;
        }

        // iota leaves order[0..sink) holding pages 0..sink-1, so ranking only the tail pins the
        // sink pages at the front of the kept set without a second pass.
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin() + sink, order.begin() + keep, order.end(),
                          [&page_score](int32_t x, int32_t y) {
                              return page_score[(size_t) x] > page_score[(size_t) y];
                          });

        std::vector<bool> keep_page((size_t) nP, false);
        for (int32_t i = 0; i < keep; ++i) keep_page[(size_t) order[(size_t) i]] = true;

        // RECENT LOCAL WINDOW, same rule as mask_from_kq_op: the `recent_pages` pages ending at the query row's own page are kept
        // regardless of score, IN ADDITION to the top-k. kq is [n_kv, n_tokens, n_head], so row r is token r % n_tokens.
        if (recent_pages > 0) {
            int64_t qpos = std::max<int64_t>(0, n_kv - n_tokens) + tok;   // no mask: assume an unpadded KV (the old rule)
            if (mask_ok) {
                qpos = -1;
                for (int64_t s = n_kv - 1; s >= 0; --s) {
                    if (attendable(tok, s)) { qpos = s; break; }
                }
            }
            if (qpos >= 0) {
                const int64_t pg = std::min<int64_t>(nP - 1, qpos / P);
                for (int64_t p = std::max<int64_t>(0, pg - recent_pages + 1); p <= pg; ++p) keep_page[(size_t) p] = true;
            }
        }

        for (int64_t t = 0; t < n_kv; ++t) {
            drow[t] = keep_page[(size_t) (t / P)] ? srow[t] : -INFINITY;
        }
    }
    (void) b;      // reserved: the key cache is the input the gather kernel will need (#220)
    (void) ktype;
}

void mask_op(::ggml_tensor * dst, const ::ggml_tensor * a, const ::ggml_tensor * b,
             int ith, int nth, void * userdata) {
    mask_op_impl(dst, a, b, nullptr, ith, nth, userdata);
}

void mask_op_kqmask(::ggml_tensor * dst, const ::ggml_tensor * a, const ::ggml_tensor * b, const ::ggml_tensor * c,
                    int ith, int nth, void * userdata) {
    mask_op_impl(dst, a, b, c, ith, nth, userdata);
}

uint64_t op_calls()       { return g_op_calls.load(std::memory_order_relaxed); }
uint64_t op_masked()      { return g_op_masked.load(std::memory_order_relaxed); }
uint64_t op_passthrough() { return g_op_passthrough.load(std::memory_order_relaxed); }
uint64_t op_last_keep()   { return g_last_keep.load(std::memory_order_relaxed); }
uint64_t op_last_npages() { return g_last_npages.load(std::memory_order_relaxed); }
uint64_t op_last_budget_milli() { return g_last_budget_milli.load(std::memory_order_relaxed); }

}  // namespace ik_offload

extern "C" {
// C surface so the counters can be read from the server/CLI without pulling in the C++ header.
uint64_t offload_attn_op_calls()       { return ik_offload::op_calls(); }
uint64_t offload_attn_op_masked()      { return ik_offload::op_masked(); }
uint64_t offload_attn_op_passthrough() { return ik_offload::op_passthrough(); }
uint64_t offload_attn_op_last_keep()   { return ik_offload::op_last_keep(); }
uint64_t offload_attn_op_last_npages() { return ik_offload::op_last_npages(); }
uint64_t offload_attn_op_last_budget_milli() { return ik_offload::op_last_budget_milli(); }
}
