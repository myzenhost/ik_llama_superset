// llama-kv-rocketkv.cpp — see llama-kv-rocketkv.h. Pure-host RocketKV stage-2 (HSA) page selector.
// Compiles + selftests with NO ggml/CUDA dependency. The page-summary materialization from kv.k_l and the
// sparse-attention gather live in the caller (llm_build_kqv), same split as llama-kv-snapkv.cpp.
//
// VERIFIED-FROM-REF: RocketKV (arxiv 2502.14051v3) Hybrid Sparse Attention, step 1 (per-page max/min
// auxiliary storage) + step 2 (query-dependent page-score approximation via the Quest upper bound) +
// step 3 (sparse attention over selected pages). Head-dim reduction (k1 largest-|q| dims) per §3.2.
#include "llama-kv-rocketkv.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace justician_kv {

std::vector<PageSummary> build_page_summaries(const std::vector<float>& K, int n_kv, int head_dim, int page_size) {
    std::vector<PageSummary> pages;
    if (n_kv <= 0 || head_dim <= 0) return pages;
    if (page_size < 1) page_size = 1;
    const int n_pages = (n_kv + page_size - 1) / page_size;
    pages.reserve(n_pages);
    for (int pg = 0; pg < n_pages; ++pg) {
        const int lo = pg * page_size;
        const int hi = std::min(lo + page_size, n_kv);
        PageSummary s;
        s.kmax.assign(head_dim,  -INFINITY);
        s.kmin.assign(head_dim,   INFINITY);
        for (int t = lo; t < hi; ++t) {
            const float* k = &K[(size_t) t * head_dim];
            for (int d = 0; d < head_dim; ++d) {
                if (k[d] > s.kmax[d]) s.kmax[d] = k[d];
                if (k[d] < s.kmin[d]) s.kmin[d] = k[d];
            }
        }
        pages.push_back(std::move(s));
    }
    return pages;
}

float page_score(const std::vector<float>& q, const PageSummary& p, int head_dim_k1) {
    const int d = (int) std::min(q.size(), p.kmax.size());
    if (d <= 0) return 0.0f;
    // contribution of each dim to the per-page UPPER BOUND: q>=0 picks kmax, q<0 picks kmin.
    auto contrib = [&](int i) -> float { return q[i] >= 0.0f ? q[i] * p.kmax[i] : q[i] * p.kmin[i]; };
    if (head_dim_k1 <= 0 || head_dim_k1 >= d) {
        float acc = 0.0f;
        for (int i = 0; i < d; ++i) acc += contrib(i);
        return acc;                                       // exact upper bound (no head-dim approximation)
    }
    // head-dim reduction: sum only over the k1 dims with largest |q| (RocketKV §3.2 seq/head reduction)
    std::vector<int> idx(d);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + head_dim_k1, idx.end(),
                      [&](int a, int b){ return std::fabs(q[a]) > std::fabs(q[b]); });
    float acc = 0.0f;
    for (int i = 0; i < head_dim_k1; ++i) acc += contrib(idx[i]);
    return acc;
}

std::vector<int> hsa_select_plan(const std::vector<float>& q,
                                 const std::vector<PageSummary>& pages,
                                 int n_kv, const RocketKVConfig& cfg) {
    const int n_pages = (int) pages.size();
    if (n_pages == 0 || n_kv <= 0) return {};
    const int budget_tokens = (cfg.n_pages_keep + cfg.recent_pages) * cfg.page_size;
    if (n_kv <= budget_tokens) return {};                 // under budget => caller keeps all (no-op)

    // score every page; select top n_pages_keep by score, UNION the most-recent recent_pages.
    std::vector<float> ps(n_pages);
    for (int pg = 0; pg < n_pages; ++pg) ps[pg] = page_score(q, pages[pg], cfg.head_dim_k1);

    std::vector<int> order(n_pages);
    std::iota(order.begin(), order.end(), 0);
    const int keep = std::min(cfg.n_pages_keep, n_pages);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [&](int a, int b){ return ps[a] > ps[b]; });

    std::vector<char> sel(n_pages, 0);
    for (int i = 0; i < keep; ++i) sel[order[i]] = 1;
    for (int pg = std::max(0, n_pages - cfg.recent_pages); pg < n_pages; ++pg) sel[pg] = 1;  // recent safety

    // expand selected pages -> sorted candidate token indices (the stage-2 sparse-attention set).
    std::vector<int> cand;
    cand.reserve((size_t)(keep + cfg.recent_pages) * cfg.page_size);
    for (int pg = 0; pg < n_pages; ++pg) {
        if (!sel[pg]) continue;
        const int lo = pg * cfg.page_size;
        const int hi = std::min(lo + cfg.page_size, n_kv);
        for (int t = lo; t < hi; ++t) cand.push_back(t);
    }
    // already ascending by construction (pages iterated in order); dedup is unnecessary but cheap-safe.
    cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    return cand;
}

// ----------------------------------------------------------------------------------------------------
// pure-host selftest — deterministic (fixed LCG), no ggml/CUDA. GATEs A..D.
// ----------------------------------------------------------------------------------------------------
int _rocketkv_selftest() {
    int fails = 0;
    const int head_dim = 32, page_size = 16, n_pages = 40, n_kv = n_pages * page_size;  // 640 tokens

    // deterministic pseudo-random K in [-1,1] via LCG (no <random>, reproducible across platforms).
    uint64_t st = 0x9E3779B97F4A7C15ull;
    auto nextf = [&]() -> float { st = st * 6364136223846793005ull + 1442695040888963407ull;
                                  return ((float)((st >> 33) & 0xFFFFFF) / (float)0x800000) - 1.0f; };
    std::vector<float> K((size_t) n_kv * head_dim);
    for (auto& v : K) v = nextf();

    // plant a HIGH-SIGNAL key aligned to query q (all-ones) in page 27, token 5 within it.
    std::vector<float> q(head_dim, 1.0f);
    const int plant_page = 27, plant_tok = plant_page * page_size + 5;
    for (int d = 0; d < head_dim; ++d) K[(size_t) plant_tok * head_dim + d] = 5.0f;  // q.k = 5*32 = 160, dominates

    auto pages = build_page_summaries(K, n_kv, head_dim, page_size);

    // GATE A — bound soundness: page_score(q,P) >= max_{k in P} q.k for EVERY page (exact-dim, k1=0).
    {
        bool ok = true;
        for (int pg = 0; pg < (int) pages.size(); ++pg) {
            float bound = page_score(q, pages[pg], 0);
            float truemax = -INFINITY;
            const int lo = pg * page_size, hi = std::min(lo + page_size, n_kv);
            for (int t = lo; t < hi; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q[d] * K[(size_t) t * head_dim + d];
                truemax = std::max(truemax, dot);
            }
            if (bound + 1e-3f < truemax) { ok = false; break; }   // bound must never under-estimate
        }
        if (!ok) { fails++; }   // GATE A
    }

    // GATE B — recall: with a tight budget the planted page IS selected and its token IS a candidate.
    {
        RocketKVConfig cfg; cfg.page_size = page_size; cfg.n_pages_keep = 4; cfg.recent_pages = 2; cfg.head_dim_k1 = 0;
        auto cand = hsa_select_plan(q, pages, n_kv, cfg);
        bool has_plant = std::find(cand.begin(), cand.end(), plant_tok) != cand.end();
        if (!has_plant) fails++;                                   // GATE B: must retain the argmax token
    }

    // GATE C — budget: |cand| <= (n_pages_keep+recent_pages)*page_size, and sorted+unique.
    {
        RocketKVConfig cfg; cfg.page_size = page_size; cfg.n_pages_keep = 4; cfg.recent_pages = 2;
        auto cand = hsa_select_plan(q, pages, n_kv, cfg);
        bool sorted = std::is_sorted(cand.begin(), cand.end());
        bool uniq   = (std::unique(cand.begin(), cand.end()) == cand.end());
        int cap = (cfg.n_pages_keep + cfg.recent_pages) * page_size;
        if (!sorted || !uniq || (int) cand.size() > cap) fails++;  // GATE C
    }

    // GATE D — head-dim reduction (k1=8 < 32) still recalls the planted page (dominant-signal preserved).
    {
        RocketKVConfig cfg; cfg.page_size = page_size; cfg.n_pages_keep = 4; cfg.recent_pages = 2; cfg.head_dim_k1 = 8;
        auto cand = hsa_select_plan(q, pages, n_kv, cfg);
        bool has_plant = std::find(cand.begin(), cand.end(), plant_tok) != cand.end();
        if (!has_plant) fails++;                                   // GATE D
    }

    // GATE E — under-budget no-op: n_kv <= budget => empty plan (caller keeps all).
    {
        RocketKVConfig cfg; cfg.page_size = page_size; cfg.n_pages_keep = 100; cfg.recent_pages = 100;
        auto cand = hsa_select_plan(q, pages, n_kv, cfg);
        if (!cand.empty()) fails++;                                // GATE E
    }
    return fails;
}

} // namespace justician_kv
