/*
 * triattention-bridge.cpp — C++ bridge for TriAttention KV scoring (ik_llama port)
 *
 * Adapted from domvox/llama.cpp-turboquant-hip feature/triattention-scoring.
 * Ported for ik_llama architecture:
 *   - llama_kv_cache_fork is a plain struct in llama-context.h (no vtable)
 *   - Methods added to llama_kv_cache_fork: get_k_raw, get_used_n_kv,
 *     get_cell_positions, triattention_compact, seq_pos_max
 *   - No llama_memory base class / dynamic_cast needed
 *   - llama_context directly contains kv_self (not returned by factory)
 */

#include "llama.h"
#include "llama-context.h"

#include <algorithm>
#include <numeric>
#include <vector>

/* Helper: extract llama_kv_cache_fork from context (ik_llama: kv_self is direct member) */
static llama_kv_cache_fork * get_kv(void * ctx_void) {
    if (!ctx_void) return nullptr;
    auto * ctx = static_cast<llama_context *>(ctx_void);
    return &ctx->kv_self;
}

extern "C" {
#include "triattention-runtime.h"

struct ggml_tensor * tria_get_k_tensor(void * ctx_void, int layer_idx) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return nullptr;
    return kv->get_k_raw(layer_idx);
}

int tria_get_n_kv(void * ctx_void) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    llama_pos pmax = kv->seq_pos_max(0);
    return (pmax >= 0) ? (int)(pmax + 1) : 0;
}

int tria_get_used_n_kv(void * ctx_void) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    return (int)kv->get_used_n_kv();
}

int tria_get_kv_positions(void * ctx_void, int * positions, int max_positions) {
    auto * ctx = static_cast<llama_context *>(ctx_void);
    if (!ctx || !positions || max_positions <= 0) {
        return 0;
    }

    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    std::vector<llama_pos> kv_positions;
    if (!kv->get_cell_positions(kv_positions)) {
        return 0;
    }

    const int n = std::min<int>((int)kv_positions.size(), max_positions);
    for (int i = 0; i < n; ++i) {
        positions[i] = (int)kv_positions[i];
    }

    return n;
}

int tria_compact_kv(struct tria_runtime * rt, void * ctx_void) {
    auto * ctx = static_cast<llama_context *>(ctx_void);
    if (!rt || !ctx || !rt->global_scores || rt->global_budget <= 0) {
        return 0;
    }

    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    const int n_kv = (int)kv->get_used_n_kv();
    const int n_old = n_kv - rt->window;
    if (n_old <= 0) {
        return 0;
    }

    int budget = rt->global_budget;
    budget = std::max(1, std::min(budget, n_old));

    /* Protect sink/prefix tokens from compaction */
    int prefix = rt->sink > 0 ? rt->sink : 128;
    if (prefix > n_old) prefix = n_old;

    /* Ensure budget covers at least the protected prefix */
    if (budget < prefix) budget = prefix;

    /* Build keep set: always keep prefix, then top-scoring from rest */
    std::vector<uint32_t> keep_positions;
    keep_positions.reserve(budget + rt->window);

    /* Always keep prefix tokens */
    for (int i = 0; i < prefix && i < budget; i++) {
        keep_positions.push_back((uint32_t)i);
    }

    /* Fill remaining budget from non-prefix tokens by score */
    int remaining_budget = budget - (int)keep_positions.size();
    if (remaining_budget > 0 && prefix < n_old) {
        std::vector<uint32_t> ranked;
        ranked.reserve(n_old - prefix);
        for (int i = prefix; i < n_old; i++) {
            ranked.push_back((uint32_t)i);
        }

        std::stable_sort(ranked.begin(), ranked.end(), [&](uint32_t a, uint32_t b) {
            if (rt->global_scores[a] == rt->global_scores[b]) return a < b;
            return rt->global_scores[a] > rt->global_scores[b];
        });

        int take = std::min(remaining_budget, (int)ranked.size());
        ranked.resize(take);
        std::sort(ranked.begin(), ranked.end());
        keep_positions.insert(keep_positions.end(), ranked.begin(), ranked.end());
    }

    /* Add window (recent) tokens */
    for (int pos = n_old; pos < n_kv; ++pos) {
        keep_positions.push_back((uint32_t)pos);
    }

    if ((int)keep_positions.size() >= n_kv) {
        return 0;
    }

    llama_synchronize(ctx);

    if (!kv->triattention_compact(keep_positions)) {
        return 0;
    }

    return n_kv - (int)keep_positions.size();
}

} /* extern "C" */
