// llama-kv-snapkv.h — training-free KV-cache eviction (SnapKV + H2O) for the JusticianAI ik_llama fork.
// S69-IKLLAMA-CPP-GAPS-01 / DEC-S69-IKLLAMA-SRC-LOCATED. Grounded by temp/research/findings/
// S69_ENGINE_KERNEL_REF_2026-06-16.md (SnapKV arxiv 2404.14469 + H2O arxiv 2306.14048, both VERIFIED-FROM-REF).
//
// DESIGN: the *scoring + selection* math is pure host C++ (no ggml dep) so it compiles + unit-tests
// standalone (KVSnap::_selftest). It consumes attention scores the graph already materializes (the
// post-softmax `kq` row-sums tagged `cb(kq,"kq",il)` @ llm-build-context.cpp:1646) and emits a KEEP-PLAN
// (which KV positions to retain per layer). The caller compacts kv.k_l/kv.v_l by the plan
// (ggml_get_rows gather) + fixes cells/head. TRAINING-FREE: no new weights, no gguf change.
//
// Wire-points (VERIFIED on disk, findings §0/§2):
//   - score capture : llm_build_kqv graph callback @ src/llama-build-context.cpp:1646 -> host buffer
//   - evict apply   : llama_kv_cache_seq_rm() @ include/llama.h:805  (per dropped pos) OR compact_layer
//   - flags         : common/common.cpp @717  --snapkv-cap/--snapkv-window/--h2o-heavy/--h2o-recent
#pragma once
#include <cstdint>
#include <vector>

namespace justician_kv {

// ---- SnapKV: one-shot PROMPT compression at end-of-prefill (better for long prompts) ----
struct SnapKVConfig {
    int  window_size        = 64;     // last-W queries = the observation window (always kept)
    int  max_capacity_prompt= 2048;   // target KV budget after compression
    int  kernel_size        = 5;      // pooling kernel (keeps contiguous info clusters)
    bool maxpool            = true;   // maxpool (true) vs avgpool — SnapKV default maxpool
    bool enabled            = false;
    // fires only when n_kv >= max_capacity_prompt (else no-op, returns keep-all)
};

// ---- H2O: streaming heavy-hitter eviction DURING decode (better for long generation) ----
struct H2OConfig {
    int  hh_size     = 64;    // heavy-hitter budget (top attention-mass positions kept)
    int  recent_size = 512;   // always-keep recent window
    bool enabled     = false;
    int  cache_size() const { return hh_size + recent_size; }
};

// 1-D pooling over a host score vector (kernel k, pad k/2, stride 1), max or avg. SnapKV step 3.
std::vector<float> pool1d(const std::vector<float>& scores, int kernel, bool maxpool);

// indices of the top-k values in `scores` (descending), bounded to scores.size().
std::vector<int> topk_idx(const std::vector<float>& scores, int k);

// SnapKV: given per-position observation-window attention scores for ONE layer (length n_kv),
// return the SORTED keep-set = topk(pooled scores over [0,n_kv-W)) UNION the recent window [n_kv-W,n_kv).
// q_len = prompt length; fires only when q_len >= cfg.max_capacity_prompt, else keep-all.
std::vector<int> snapkv_keep_plan(const std::vector<float>& obs_scores, int q_len, const SnapKVConfig& cfg);

// H2O running heavy-hitter accumulator (one per layer). Call accumulate() each decode step with the
// post-softmax attention column-sums; evict_plan() returns the keep-set when seq_len > cache_size.
struct H2OAccumulator {
    std::vector<float> hh_score;                 // running column-sum of attention per KV position
    void reset(int n_kv = 0) { hh_score.assign(n_kv, 0.0f); }
    void accumulate(const std::vector<float>& attn_col_sums);   // += per-position mass (auto-grows)
    // returns SORTED keep-set = topk(hh_size over [0,seq_len-recent)) UNION recent window. Empty => keep all.
    std::vector<int> evict_plan(int seq_len, const H2OConfig& cfg) const;
};

int _selftest();   // returns 0 on pass; pure-host, no ggml/CUDA needed

} // namespace justician_kv
