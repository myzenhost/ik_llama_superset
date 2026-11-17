// offload_attention.h — ik_llama fork SUPPORT for offload-attention (full KV off-VRAM + per-step sparse select
// + gather + attend). Training-free, weight-preserving: NO GGUF change; splices into the existing custom attention
// bridge (triattention-bridge.cpp) + llama-build-context.cpp KV path + ggml/ggml-cuda.cu flash-attn op.
//
// This header is the SUPPORT-LEVEL contract (user directive 2026-07-06: "these systems should be in our forked
// ik_llama even if just support for it"). The reference implementation that this port must MATCH bit-for-bit
// (CPU-proven cosine=1.0 vs dense @ 2% page budget, 3 selectors) lives in the harness:
//   cortex/cortex/inference/offload_attention.py  (OffloadKVStore + select_pages + offload_attend)
// DEC-S69-OFFLOAD-ATTENTION-INJECT-INTO-IKLLAMA. Tasks #219 (retattn OOD selector), #220 (kernel compose),
// #93 (ggml_paged_attention substrate), #89 (KVFlash pool-mask prereq — shared paged read path).
//
// NO ARBITRARY CONTEXT CAP (rule 3.8 + user directive): n_ctx is a runtime property of the loaded KV, never a
// compile-time limit. page_budget is a FRACTION or explicit int, per-model. Our memory/context/token systems
// EXTEND the effective window per model — nothing here truncates or caps context.
#pragma once
#include <cstdint>
#include <vector>

struct ggml_tensor;      // global scope on purpose -- see the note on mask_op below

namespace ik_offload {

// Which per-step critical-token selector to use (cheapest -> highest-recall), matching the harness reference.
enum class Selector : int {
    QUEST    = 0, // per-page element-wise min/max key upper-bound (ICML'24 2406.10774). No SVD/index. Cheapest.
    SHADOWKV = 1, // 8-tok chunk landmark (mean) + rank-r low-rank pre-RoPE key recon (2410.21465). Low VRAM.
    RETATTN  = 2, // attention-aware OOD-corrected selector (RetrievalAttention 2409.10516). Highest recall.
};

struct OffloadConfig {
    Selector selector   = Selector::QUEST;
    int32_t  page_size  = 16;     // tokens per KV page (block-table granularity)
    float    page_budget = 0.02f; // fraction of pages attended per step (>=1.0 => interpret as absolute page count)
    int32_t  rank       = 160;    // ShadowKV low-rank pre-RoPE key rank
    bool     use_nvme   = false;  // false=CPU-RAM tier (fast), true=NVMe mmap tier (capacity floor)
    // InfiniGen prefetch: speculate layer-i critical KV from layer-(i-1) attn input; overlap CPU->GPU copy.
    bool     prefetch_speculate = true;
    // ATTENTION SINK. Number of leading KV pages pinned into the kept set REGARDLESS of
    // score. DEFAULT 0, which reproduces the reference behaviour bit for bit -- this
    // header's contract is that it mirrors Quest / ShadowKV / RetAttn, and the E2E parity
    // tests check that, so the sink is opt-in rather than a silent change of default.
    //
    // WHY IT IS NEEDED AT ALL: select_pages ranks pages by RELEVANCE and takes the top k.
    // An attention sink is by definition a token that attracts attention mass WITHOUT
    // being semantically relevant, so a relevance ranking discards it by construction.
    // arXiv 2309.17453v3 (StreamingLLM, ICLR 2024): "the model collapses once the
    // sequence length exceeds the cache size, i.e. even just evicting the KV of the first
    // token"; four initial tokens suffice. page_size defaults to 16, so sink_pages = 1
    // covers those four with room to spare.
    int32_t  sink_pages = 0;
    // MINIMUM KEPT KV, expressed in TOKENS rather than pages so it is page_size independent.
    // keep = (int32_t)(n_pages * page_budget) truncates toward zero and is then floored to 1,
    // so at short context the default budget keeps a SINGLE page. MEASURED 2026-08-21 on the
    // served FA path: n_kv=256, n_pages=16, budget=0.02 -> keep=1 (6.2% of KV) and repetition
    // 0.789 against a 0.545 no-flag baseline, while keep=4 reproduced the baseline output
    // EXACTLY (similarity 1.000). QUEST, H2O and SnapKV all state the KV budget as an absolute
    // token count for this reason -- a bare fraction degenerates precisely where there is least
    // to gain. Below this floor the feature becomes a NO-OP instead of a lobotomy. 0 disables.
    int32_t  min_kept_tokens = 256;
    // HOW THE PER-HEAD PAGE SELECTIONS ARE COMBINED INTO THE ONE SHARED MASK (FlashAttention path, mask_from_kq_op).
    //   0 = UNION (default, unchanged): every head takes its own top-`keep` pages and a page survives if ANY head wants it.
    //       The kept set can reach n_head * keep pages, so the nominal "keep" fraction in the trace understates what is attended.
    //   1 = MAX: each head's page scores are normalised to log-softmax over pages (so a head with large |q.k| cannot outvote
    //       the rest), the per-page MAX over heads is taken, and ONE top-`keep` is cut. Exactly `keep` pages survive
    //       (HiP Attention, arXiv 2406.09827 / 2502.02770, reduces across heads before selecting).
    // Env GGML_OFFLOAD_ATTN_HEAD_REDUCE=max|union overrides this when it is set. Which default is better is a recall question
    // (S69-HIPATTENTION-PICK-ATTENTION-MASS-RECALL-HARNESS-2026-10-05-01); until measured the default stays UNION.
    int32_t  head_reduce = 0;
    // RECENT LOCAL WINDOW (mask_from_kq_op, FA path): the `recent_pages` pages ending at the query row's own page are kept REGARDLESS of score, in
    // ADDITION to the top-`keep` (so the actual kept count is keep + up to recent_pages). A relevance ranking tends to drop the tokens just before the
    // query, which carry much of the attention mass; HiP and StreamingLLM both pin a local window. DEFAULT 0 = old behaviour. Env
    // GGML_OFFLOAD_ATTN_RECENT_PAGES=N overrides. The query position of a row is its LAST ATTENDABLE key in kq_mask (n_kv is padded, so
    // n_kv - n_tok + t is not a position); mask_op without a mask input falls back to n_kv - n_tok + t.
    int32_t  recent_pages = 0;
    // MASK REFRESH (mask_from_kq_op, FA path): R > 1 reuses a layer's page selection for single-token decode steps until it is R steps old
    // (plus the newly written key's page and the sink), so the scan runs once per R tokens. DEFAULT 1 = scan every step (old behaviour).
    // Env GGML_OFFLOAD_ATTN_REFRESH=R overrides. HiP (mask_refresh_interval) refreshes every 8-64 steps.
    int32_t  refresh_interval = 1;
    // PAGE SCAN (mask_from_kq_op): 0 = exact (score every attendable key, default), 1 = bisect (hierarchical: rank chunks of
    // GGML_OFFLOAD_ATTN_SCAN_CHUNK keys, default 256, by 2 sampled keys per halving level, score pages exactly only inside the best chunks).
    // Env GGML_OFFLOAD_ATTN_SCAN=bisect|exact overrides. Bisect is an approximation; its recall is measured by offload-attn-recall.
    int32_t  scan_mode = 0;
    float    alpha      = 5.0f;   // InfiniGen threshold: keep tokens with score > (max - alpha)
};

// SINGLE SOURCE OF TRUTH for how many pages survive selection. Four call sites need this answer:
// mask_op (non-FA), mask_from_kq_op (FA), and the two graph-build sites that decide whether to
// splice an op at all. Each of those carried its OWN copy of the top-k arithmetic, which is how
// the two ops drifted apart -- one honoured a config field the other ignored, and the difference
// was invisible because both compiled. Anything that needs the kept count calls THIS.
// Returns the page count in *out_n_pages when non-null. keep == *out_n_pages means the selection
// is the identity: every page survives and masking is a no-op.
inline int32_t pages_kept(const OffloadConfig & cfg, int64_t n_kv, int64_t * out_n_pages = nullptr) {
    const int32_t P  = cfg.page_size > 0 ? cfg.page_size : 16;
    const int64_t nP = (n_kv + P - 1) / P;
    if (out_n_pages) *out_n_pages = nP;
    int32_t keep = (cfg.page_budget >= 1.0f)
                 ? (int32_t) cfg.page_budget
                 : (int32_t) ((float) nP * cfg.page_budget);
    if (keep < 1)   keep = 1;
    if (keep > nP)  keep = (int32_t) nP;
    // Token floor -- see min_kept_tokens above for the measurement that forced it.
    if (cfg.min_kept_tokens > 0) {
        int32_t min_pages = (int32_t) ((cfg.min_kept_tokens + P - 1) / P);
        if (min_pages > nP)   min_pages = (int32_t) nP;
        if (keep < min_pages) keep = min_pages;
    }
    return keep;
}

// Full K,V held off the compute device (CPU-RAM or NVMe mmap = slot-save #149 backing). Only selector-chosen
// pages are gathered to the GPU compute buffer per decode step (the ggml_paged_attention op / #93 substrate).
// Element type of the backing buffers. F16 is here because it is what the fork ACTUALLY uses:
// llama_context_default_params sets type_k/type_v = GGML_TYPE_F16 (src/llama.cpp:4906-4907), and
// an F32 KV cache is not a reachable configuration -- MEASURED 2026-08-19 on SmolLM2-135M:
//   type_v=F32, flash_attn=off -> "llama_init_from_model: V cache quantization requires flash_attn"
//   type_v=F32, flash_attn=on  -> iqk_flash_attn.cpp:184 GGML_ABORT; F32 is absent from
//                                 supported_kv_types()
// An F32-only implementation would therefore refuse on every real model while looking correct.
enum class KVType : int { F32 = 0, F16 = 1 };

struct OffloadKVStore {
    int64_t n_tokens = 0;
    int64_t head_dim = 0;
    int32_t n_kv_head = 0;
    // backing pointers (host RAM or mmap'd NVMe); populated at prefill, appended per decode.
    void*   k_backing = nullptr;
    void*   v_backing = nullptr;
    // Element type of k_backing / v_backing. Defaults to F32 so existing callers are unchanged.
    KVType  k_type = KVType::F32;
    KVType  v_type = KVType::F32;
};

// Return the token indices of the `n_keep_pages` critical pages for query q (one head). Contract mirrors
// harness select_pages(); the kernel fills these in per selector.
std::vector<int32_t> select_pages(const float* q, const OffloadKVStore& store,
                                  const OffloadConfig& cfg, int32_t n_keep_pages);

// One decode step: select -> gather selected KV CPU/NVMe->GPU (async, overlapped) -> flash-attn over the
// gathered set -> write output. Must match the harness reference output within fp tolerance.
// Returns 0 on success. IMPLEMENTED BY: the #220 kernel port (this header is the support-level contract).
int offload_attend_step(const float* q, const OffloadKVStore& store, const OffloadConfig& cfg,
                        float* out /*[head_dim]*/);

// EXECUTION-TIME masking op (#220). Spliced between kq = ggml_mul_mat(k, q) and the softmax via
// ggml_map_custom2, it sets the scores of unselected KV pages to -INFINITY so softmax gives them
// zero weight. It exists because the graph BUILDER has no KV contents to select from -- selection
// has to happen where the data is real.
//
// FAILS OPEN: any unsupported dtype/shape/residency copies the scores through unchanged, because a
// wrong mask corrupts every answer while an absent mask is merely today's behaviour. The
// pass-through is COUNTED, since a fail-open path nobody observes is how "enabled" becomes
// indistinguishable from "off".
// NOTE: ggml_tensor is forward-declared at GLOBAL scope above, not here. Declaring it inside this
// namespace creates ik_offload::ggml_tensor, a DIFFERENT incomplete type that shadows the real
// one -- the compiler then reports "use of undefined type" at every member access, pointing at
// this header rather than at the mistake.
void mask_op(::ggml_tensor * dst, const ::ggml_tensor * a, const ::ggml_tensor * b,
             int ith, int nth, void * userdata);
// ggml_custom3_op_t: mask_op with the attention mask as a third input, c = kq_mask [n_kv, n_tokens_padded] (F32 or F16). The mask states which
// keys each query row may attend, so a key it forbids (causal future inside a prefill batch, a padded KV cell, a sliding-window exclusion)
// cannot win a page's score, and the recent window is placed at the row's real position (its last attendable key) rather than at
// n_kv - n_tokens + t, which is wrong whenever n_kv is padded. Same scoring rule as mask_from_kq_op.
void mask_op_kqmask(::ggml_tensor * dst, const ::ggml_tensor * a, const ::ggml_tensor * b, const ::ggml_tensor * c,
                    int ith, int nth, void * userdata);

uint64_t op_calls();
uint64_t op_masked();
uint64_t op_passthrough();
uint64_t op_last_keep();
uint64_t op_last_npages();
uint64_t op_last_budget_milli();

// FLASH-ATTENTION variant (#220). ggml_flash_attn_ext computes scores internally, so
// there is nothing to intercept between mul_mat and softmax -- the only lever is the
// kq_mask it consumes. Selection is therefore made from the KEYS and QUERY. The mask is
// SHARED across heads, so a page is kept if ANY head wants it: strictly more conservative
// than per-head selection, and NOT numerically identical to the non-FA path at the same
// budget.
//
// CORRECTION 2026-08-20. This comment used to say "QUEST proper". It does not run QUEST.
// QUEST, as offload_attention.cpp defines it, is
//     score_p = sum_d max(qs_d*Kmin_pd, qs_d*Kmax_pd)
// over precomputed per-page min/max BOUNDS -- O(d) per page from cached bounds. This op takes,
// for each page, the maximum EXACT (q . k)/sqrt(head_dim) over every key in the page, which is
// O(P*d) per page. Exact max-dot is the quantity QUEST's bound estimates, so it is at least as
// accurate and strictly more expensive -- but it is not the same rule, and the difference
// matters to anyone comparing these results against the 2406.10774 numbers.
//
// AND IT IGNORES cfg.selector, as does mask_op. select_pages() in offload_attention.cpp
// implements QUEST / SHADOWKV / RETATTN and is reached only through offload_attend_step (the
// bridge), which the graph path never calls. MEASURED on served output: selectors 0/1/2 give
// byte-identical text on both -fa 1 and -fa 0 (similarity 1.0000, Llama-3.2-1B-Q8_0, temp 0,
// fixed seed); budget DOES separate. Tracked: S69-OFFLOAD-ATTN-SELECTOR-INERT-IN-GRAPH-OP-01.
void mask_from_kq_op(::ggml_tensor * dst, const ::ggml_tensor * a,
                     const ::ggml_tensor * b, const ::ggml_tensor * c,
                     int ith, int nth, void * userdata);

uint64_t maskop_calls();
uint64_t maskop_masked();
uint64_t maskop_passthrough();
uint64_t maskop_last_keep();
uint64_t maskop_last_npages();
uint64_t maskop_keys_scored();     // q.k products computed by mask_from_kq_op (exact and bisect scans), cumulative
uint64_t maskop_refresh_full();    // single-token steps that ran a full scan with refresh on
uint64_t maskop_refresh_reused();  // single-token steps that reused the previous selection

} // namespace ik_offload
