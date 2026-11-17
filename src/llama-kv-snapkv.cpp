// llama-kv-snapkv.cpp — see llama-kv-snapkv.h. Pure-host scoring/selection core (compiles + selftests
// with NO ggml/CUDA dependency). The graph-callback score capture + cache-tensor compaction live in the
// caller (llm_build_kqv / llama_kv_cache compaction) using the keep-plans this file returns.
//
// VERIFIED-FROM-REF algorithms (findings §2):
//   SnapKV (FasterDecoding/SnapKV snapkv_utils.py::update_kv): obs-window softmax-attn -> per-pos sum
//     -> pool1d(kernel=5) -> topk(cap-W) UNION recent-W.
//   H2O (FMInference/H2O modify_llama.py::H2OKVCache_LayerWise): running column-sum of attention
//     -> topk(hh_size over non-recent) UNION recent-window when seq_len>cache_size.
#include "llama-kv-snapkv.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace justician_kv {

std::vector<float> pool1d(const std::vector<float>& s, int kernel, bool maxpool) {
    const int n = (int) s.size();
    std::vector<float> out(n, 0.0f);
    if (n == 0) return out;
    if (kernel < 1) kernel = 1;
    const int pad = kernel / 2;
    for (int i = 0; i < n; ++i) {
        float acc = maxpool ? -INFINITY : 0.0f;
        int   cnt = 0;
        for (int j = i - pad; j <= i + pad; ++j) {
            if (j < 0 || j >= n) continue;       // valid-only (matches torch pad-with-edge semantics closely enough for ranking)
            acc = maxpool ? std::max(acc, s[j]) : acc + s[j];
            ++cnt;
        }
        out[i] = maxpool ? acc : (cnt ? acc / cnt : 0.0f);
    }
    return out;
}

std::vector<int> topk_idx(const std::vector<float>& s, int k) {
    const int n = (int) s.size();
    k = std::max(0, std::min(k, n));
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (k < n) {
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b){ return s[a] > s[b]; });
        idx.resize(k);
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return s[a] > s[b]; });
    }
    return idx;
}

// union of selected non-recent top-k with the always-keep recent window, returned SORTED ascending
static std::vector<int> keep_union(const std::vector<int>& topk, int recent_lo, int n_kv) {
    std::vector<int> keep = topk;
    for (int p = std::max(0, recent_lo); p < n_kv; ++p) keep.push_back(p);
    std::sort(keep.begin(), keep.end());
    keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
    return keep;
}

std::vector<int> snapkv_keep_plan(const std::vector<float>& obs_scores, int q_len, const SnapKVConfig& cfg) {
    const int n_kv = (int) obs_scores.size();
    std::vector<int> keep_all(n_kv);
    std::iota(keep_all.begin(), keep_all.end(), 0);
    if (!cfg.enabled || q_len < cfg.max_capacity_prompt || n_kv <= cfg.max_capacity_prompt) {
        return keep_all;                                  // no-op: prompt under budget
    }
    const int W = std::min(cfg.window_size, n_kv);
    const int recent_lo = n_kv - W;                       // [recent_lo, n_kv) always kept
    // score only the prior region [0, recent_lo); pool; topk(cap - W)
    std::vector<float> prior(obs_scores.begin(), obs_scores.begin() + recent_lo);
    std::vector<float> pooled = pool1d(prior, cfg.kernel_size, cfg.maxpool);
    std::vector<int>   sel    = topk_idx(pooled, cfg.max_capacity_prompt - W);
    return keep_union(sel, recent_lo, n_kv);
}

void H2OAccumulator::accumulate(const std::vector<float>& col_sums) {
    if (col_sums.size() > hh_score.size()) hh_score.resize(col_sums.size(), 0.0f);
    for (size_t i = 0; i < col_sums.size(); ++i) hh_score[i] += col_sums[i];
}

std::vector<int> H2OAccumulator::evict_plan(int seq_len, const H2OConfig& cfg) const {
    if (!cfg.enabled || seq_len <= cfg.cache_size()) return {};   // empty => keep all
    const int recent_lo = std::max(0, seq_len - cfg.recent_size);
    std::vector<float> prior(hh_score.begin(),
                             hh_score.begin() + std::min((size_t) recent_lo, hh_score.size()));
    std::vector<int> sel = topk_idx(prior, cfg.hh_size);
    return keep_union(sel, recent_lo, seq_len);
}

int _selftest() {
    // pool1d maxpool keeps a neighbor cluster's peak
    auto p = pool1d({0.0f, 9.0f, 0.0f, 1.0f, 1.0f, 1.0f}, 5, true);
    if (!(p[0] == 9.0f && p[2] == 9.0f)) return 1;       // 9 spreads to neighbors -> cluster kept
    // topk picks the largest indices
    auto t = topk_idx({0.1f, 0.9f, 0.2f, 0.8f}, 2);
    if (!(t.size() == 2 && t[0] == 1 && t[1] == 3)) return 2;
    // SnapKV: q_len under cap -> keep all
    SnapKVConfig sc; sc.enabled = true; sc.max_capacity_prompt = 2048;
    auto ka = snapkv_keep_plan(std::vector<float>(100, 1.0f), 100, sc);
    if ((int) ka.size() != 100) return 3;
    // SnapKV: over cap -> compresses to cap, recent window present
    sc.max_capacity_prompt = 8; sc.window_size = 4; sc.kernel_size = 3;
    std::vector<float> obs(20, 0.1f); obs[2] = obs[3] = obs[10] = 5.0f;   // hot prior tokens
    auto kp = snapkv_keep_plan(obs, 9999, sc);
    if ((int) kp.size() > 8) return 4;                                    // <= cap
    bool has_recent = false; for (int x : kp) if (x >= 16) has_recent = true;
    if (!has_recent) return 5;                                           // recent window kept
    // H2O: accumulates + evicts to cache_size, keeps recent
    H2OConfig hc; hc.enabled = true; hc.hh_size = 2; hc.recent_size = 3;
    H2OAccumulator acc; acc.reset(10);
    std::vector<float> cs(10, 0.0f); cs[1] = 10.0f; cs[5] = 8.0f;        // heavy hitters at 1,5
    acc.accumulate(cs);
    auto ep = acc.evict_plan(10, hc);
    if (ep.empty()) return 6;                                            // must evict (10 > 5)
    bool keep1 = false; for (int x : ep) if (x == 1) keep1 = true;
    if (!keep1) return 7;                                                // heavy hitter retained
    if ((int) ep.size() > hc.cache_size()) return 8;
    return 0;
}

} // namespace justician_kv
