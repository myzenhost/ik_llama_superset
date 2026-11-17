#pragma once

// SGMV-style heterogeneous batched LoRA, CPU reference (S69-LORAX-PICK-SGMV-HETEROGENEOUS-LORA-BATCHING-2026-09-29-01).
//   y[t] += scale_a * B_a (A_a x[t])    where a = ids[t]; ids[t] < 0 (or >= n_adapters, or a null A) means "no adapter": row t is left untouched.
// Semantics follow predibase/lorax (Apache-2.0) server/lorax_server/utils/punica.py + segments.py (SGMV: one launch, per-segment weight pair);
// independent implementation, no code copied. Grouping here is by adapter id over the WHOLE batch (stable, via a permutation) rather than only
// over contiguous runs, so interleaved ids still give one segment per adapter.
//
// Layouts (row-major): x [n_tokens][n_in], y [n_tokens][n_out], A [rank][n_in], B [n_out][rank].
// This is a reference/benchmark core, NOT wired into the graph: the fork applies LoRA per llama_context (llm_build_lora_mm) and the server batches
// only slots with equal LoRA lists (server_slot::can_batch_with). See docs/agent-reports/lorax-adapter-pool.md.

#include <cstddef>
#include <vector>

struct lora_sgmv_adapter {
    const float * A     = nullptr;   // rank x n_in
    const float * B     = nullptr;   // n_out x rank
    int           rank  = 0;
    float         scale = 1.0f;
};

struct lora_sgmv_segment {
    int adapter;   // index into the adapter table
    int start;     // first position in plan.perm
    int len;       // number of tokens (>= 1)
};

struct lora_sgmv_plan {
    std::vector<int>               perm;   // token indices, grouped by adapter (stable within an adapter); base-only tokens are absent
    std::vector<lora_sgmv_segment> segs;   // ascending adapter id
};

inline lora_sgmv_plan lora_sgmv_group(const int * ids, int n_tokens, const lora_sgmv_adapter * ad, int n_adapters) {
    lora_sgmv_plan p;
    std::vector<std::vector<int>> by(n_adapters > 0 ? n_adapters : 0);
    for (int t = 0; t < n_tokens; ++t) {
        const int a = ids[t];
        if (a >= 0 && a < n_adapters && ad[a].A && ad[a].B && ad[a].rank > 0) {
            by[a].push_back(t);
        }
    }
    for (int a = 0; a < n_adapters; ++a) {
        if (by[a].empty()) {
            continue;
        }
        p.segs.push_back({ a, (int) p.perm.size(), (int) by[a].size() });
        p.perm.insert(p.perm.end(), by[a].begin(), by[a].end());
    }
    return p;
}

// per-token loop: the semantic definition
inline void lora_sgmv_apply_naive(const float * x, float * y, const int * ids, int n_tokens, int n_in, int n_out,
                                  const lora_sgmv_adapter * ad, int n_adapters) {
    std::vector<float> v;
    for (int t = 0; t < n_tokens; ++t) {
        const int a = ids[t];
        if (a < 0 || a >= n_adapters || !ad[a].A || !ad[a].B || ad[a].rank <= 0) {
            continue;
        }
        const lora_sgmv_adapter & L = ad[a];
        v.assign(L.rank, 0.0f);
        for (int k = 0; k < L.rank; ++k) {
            float s = 0.0f;
            for (int i = 0; i < n_in; ++i) {
                s += L.A[(size_t) k * n_in + i] * x[(size_t) t * n_in + i];
            }
            v[k] = s;
        }
        for (int o = 0; o < n_out; ++o) {
            float s = 0.0f;
            for (int k = 0; k < L.rank; ++k) {
                s += L.B[(size_t) o * L.rank + k] * v[k];
            }
            y[(size_t) t * n_out + o] += L.scale * s;
        }
    }
}

// grouped: per segment, gather its rows, shrink all of them against A (A streamed once per segment), expand against B, scatter-add
inline void lora_sgmv_apply_grouped(const float * x, float * y, const lora_sgmv_plan & plan, int n_in, int n_out,
                                    const lora_sgmv_adapter * ad) {
    std::vector<float> xg, v;
    for (const lora_sgmv_segment & sg : plan.segs) {
        const lora_sgmv_adapter & L = ad[sg.adapter];
        const int len = sg.len, r = L.rank;
        xg.resize((size_t) len * n_in);
        v.assign((size_t) len * r, 0.0f);
        for (int j = 0; j < len; ++j) {
            const int t = plan.perm[sg.start + j];
            for (int i = 0; i < n_in; ++i) {
                xg[(size_t) j * n_in + i] = x[(size_t) t * n_in + i];
            }
        }
        for (int k = 0; k < r; ++k) {
            const float * a = L.A + (size_t) k * n_in;
            for (int j = 0; j < len; ++j) {
                const float * xr = xg.data() + (size_t) j * n_in;
                float s = 0.0f;
                for (int i = 0; i < n_in; ++i) {
                    s += a[i] * xr[i];
                }
                v[(size_t) j * r + k] = s;
            }
        }
        for (int o = 0; o < n_out; ++o) {
            const float * b = L.B + (size_t) o * r;
            for (int j = 0; j < len; ++j) {
                const float * vr = v.data() + (size_t) j * r;
                float s = 0.0f;
                for (int k = 0; k < r; ++k) {
                    s += b[k] * vr[k];
                }
                y[(size_t) plan.perm[sg.start + j] * n_out + o] += L.scale * s;
            }
        }
    }
}
