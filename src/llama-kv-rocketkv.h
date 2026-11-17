// llama-kv-rocketkv.h — RocketKV stage-2 "Hybrid Sparse Attention" (HSA) page selector for the
// JusticianAI ik_llama fork. S69-IKLLAMA-SNAPKV-H2O-01 follow-on (M4 of DEC-S69-KVFLASH-ADOPT-GATHER-OR-EVICT).
//
// Grounded by a FULL-READ (rule 1.23) of RocketKV (arxiv 2502.14051v3, "RocketKV: Accelerating Long-Context
// LLM Inference via Two-Stage KV Cache Compression"). Measured in-paper: Llama3.1-8B 256-token budget
// LongBench Full-KV 52.2 -> RocketKV 51.1 (-1.1pp) vs SnapKV 42.6 / Quest 17.8; NIAH 100%; RULER-96K 73.2 vs
// Exact-TopK 72.9; up to 3.7x e2e speedup, 32.6% peak-mem save, 400x compression.
//
// TWO-STAGE PLACEMENT:
//   Stage 1 = SnapKV permanent eviction  -> already in justician_kv::snapkv_keep_plan (llama-kv-snapkv.*).
//   Stage 2 = HSA dynamic top-k (THIS).  Per-decode: summarize remaining KV into pages, score pages by a
//             query-dependent UPPER BOUND, keep top pages, attend only their tokens. Lossy-but-bounded.
//
// DESIGN (mirrors snapkv): the page-summary build + page scoring + top-page selection are PURE HOST C++
// (no ggml/CUDA dep) so they compile + unit-test standalone (_rocketkv_selftest, GATEs A..D). The caller
// (llm_build_kqv) supplies the already-GQA-group-reduced query row + the per-page K max/min summaries it
// materialized from kv.k_l, and applies the returned candidate-token set as the sparse-attention index set
// (ggml_get_rows gather, same apply-path as the snapkv keep-plan). TRAINING-FREE: no weights, no gguf change.
//
// HSA bound (Quest/RocketKV step-2): for a page P with per-dim element-wise max[] / min[] over its K vectors,
//   page_score(q,P) = sum_d ( q[d] >= 0 ? q[d]*max[d] : q[d]*min[d] )   >=   max_{k in P} (q . k).
// It is a true UPPER BOUND on the best dot product in the page, so selecting top pages by it never demotes
// the page holding the global-argmax key below a page whose bound is smaller than that argmax score.
#pragma once
#include <cstdint>
#include <vector>

namespace justician_kv {

struct RocketKVConfig {
    int  page_size      = 16;    // tokens per page (HSA "consecutive pages along sequence dim")
    int  n_pages_keep   = 16;    // top pages retained per decode step (budget = n_pages_keep*page_size tokens)
    int  head_dim_k1    = 0;     // if >0, score using only the k1 largest-|q| head dims (HSA seq/head reduction);
                                 // 0 = use ALL dims = exact upper bound (no head-dim approximation)
    int  recent_pages   = 2;     // always-keep most-recent pages (causality/local-window safety)
    bool enabled        = false; // fires only when n_kv > (n_pages_keep+recent_pages)*page_size, else keep-all
};

// per-page element-wise max/min summary of the page's K vectors. len(max)==len(min)==head_dim.
struct PageSummary {
    std::vector<float> kmax;
    std::vector<float> kmin;
};

// Build page summaries from a row-major [n_kv x head_dim] key buffer (logical decode order).
// pages = ceil(n_kv / page_size). The last page may be short.
std::vector<PageSummary> build_page_summaries(const std::vector<float>& K, int n_kv, int head_dim, int page_size);

// HSA upper-bound score of one (group-reduced) query row against one page summary.
// If head_dim_k1>0, restrict the sum to the k1 dims with largest |q| (approximation).
float page_score(const std::vector<float>& q, const PageSummary& p, int head_dim_k1);

// Stage-2 selection: returns the SORTED candidate KV-token index set = tokens of {top n_pages_keep pages by
// page_score} UNION {recent_pages most-recent pages}. Empty result => caller keeps all (no-op).
std::vector<int> hsa_select_plan(const std::vector<float>& q,
                                 const std::vector<PageSummary>& pages,
                                 int n_kv, const RocketKVConfig& cfg);

int _rocketkv_selftest();   // returns 0 on pass; pure-host, no ggml/CUDA needed

} // namespace justician_kv
