// llama-kv-diversity.h -- diversity-aware KV retention (AdaptiveR-KV style block diversity + KVCrush representative blocks) for the
// SnapKV / H2O keep-plan.  Pure host C++, no ggml dependency, header-only: compiles and unit-tests standalone
// (tests/test-kv-diversity.cpp).
// Rows: S69-openvino-PICK-KVCRUSH-DIVERSITY-RETENTION-2026-09-28-01.
//
// WHAT IS DIFFERENT FROM justician_kv::snapkv_keep_plan / H2OAccumulator (llama-kv-snapkv.h): those rank positions by attention mass only, so two
// near-duplicate regions that both score high are both kept and a lower-scored UNIQUE region is dropped. These functions add a redundancy term.
//
// SPEC READ FROM (behaviour only, no code copied; Apache-2.0, openvinotoolkit/openvino.genai @ 1445b60fe, own words below):
//   cache_eviction.cpp:659-737 (AdaptiveRKVBlockCalculator), kvcrush.cpp:15-174, and the reference diversity kernel
//   pa_adaptive_rkv_diversity_ref.cl:190-323 as summarised on the row:
//     (1) similarity is cosine similarity of K rows among the EVICTABLE tokens only (the first start_size "sink" tokens are excluded), per KV head;
//     (2) the diagonal is zeroed AFTER the matrix is built, the row mean is rowsum / n_evictable (the zero diagonal still counts in the denominator);
//     (3) entries BELOW their row mean are zeroed (only above-average similarity counts as redundancy);
//     (4) the per-head matrices are averaged over heads;
//     (5) a block's redundancy is the sum over its token rows; score = -redundancy so the most redundant block is the most negative.
//   The block-selection step: keep the blocks with the most attention mass until `attention_mass` of the total mass is covered (bounded by a
//   block budget), then fill the remaining budget with the MOST DIVERSE of the other blocks, where a block's diversity is its mean (negated)
//   redundancy against the tokens of the not-yet-kept candidate blocks.
//   KVCrush: one binary indicator per token (1 if the token is in the top half of the scores), blocks are compared by Hamming distance of their
//   indicator pattern to an anchor pattern, and `budget` representatives are taken uniformly spaced along the distance-sorted candidate list.
//
// OUR DEFECT GUARDS (the reference divides by zero when the budget or the eligible set is empty: kvcrush.cpp:136 step = size / n_retain): both are
// handled here and covered by the test.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace justician_kv {

// ---- token redundancy ------------------------------------------------------------------------------------------------------------------------
// K: [n_tokens][n_head_kv][head_dim] row-major floats (the cached keys of ONE layer, already dequantised).  Tokens [0, start_size) are sink tokens
// and are not part of the matrix.  Returns S, n_ev x n_ev row-major (n_ev = n_tokens - start_size), already thresholded and head-averaged, diagonal 0.
inline std::vector<float> kv_token_redundancy(const float * K, int n_tokens, int n_head_kv, int head_dim, int start_size) {
    const int n_ev = n_tokens - start_size;
    std::vector<float> out;
    if (n_ev <= 0 || n_head_kv <= 0 || head_dim <= 0) {
        return out;
    }
    out.assign((size_t) n_ev * n_ev, 0.0f);
    std::vector<float> kn((size_t) n_ev * head_dim);
    std::vector<float> row((size_t) n_ev);
    for (int h = 0; h < n_head_kv; ++h) {
        for (int i = 0; i < n_ev; ++i) {
            const float * src = K + ((size_t) (start_size + i) * n_head_kv + h) * head_dim;
            double nrm = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                nrm += (double) src[d] * src[d];
            }
            const float inv = nrm > 0.0 ? (float) (1.0 / std::sqrt(nrm)) : 0.0f;   // an all-zero key has similarity 0 to everything
            float * dst = kn.data() + (size_t) i * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                dst[d] = src[d] * inv;
            }
        }
        for (int i = 0; i < n_ev; ++i) {
            const float * a = kn.data() + (size_t) i * head_dim;
            double sum = 0.0;
            for (int j = 0; j < n_ev; ++j) {
                if (j == i) {
                    row[j] = 0.0f;                                  // diagonal zeroed (the reference sets it to 0 after building the matrix)
                    continue;
                }
                const float * b = kn.data() + (size_t) j * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += a[d] * b[d];
                }
                row[j] = dot;
                sum += dot;
            }
            const float mean = (float) (sum / n_ev);                // denominator counts the zero diagonal
            float * o = out.data() + (size_t) i * n_ev;
            for (int j = 0; j < n_ev; ++j) {
                if (row[j] >= mean) {                               // below-mean entries are zeroed
                    o[j] += row[j] / n_head_kv;                     // head average
                }
            }
        }
    }
    return out;
}

// ---- AdaptiveR-KV block selection ------------------------------------------------------------------------------------------------------------
struct AdaptiveRKVResult {
    std::vector<int> kept_blocks;      // sorted block indices (relative to the evictable area)
    int              n_mass_blocks = 0;
    int              n_diverse_blocks = 0;
};

// block_mass: attention mass per block of the evictable area (size n_blocks).  S: redundancy matrix from kv_token_redundancy over the same area
// (n_ev = n_blocks * block_size tokens; a trailing partial block is not supported: pass a block-aligned area).
// max_blocks: total number of blocks that may be kept.  attention_mass in (0, 1]: fraction of the total mass to preserve first.
// out_div (optional): the per-block diversity values (0 for blocks kept for their mass).  Ties are broken by attention mass (the reference breaks
// them by heap order, i.e. arbitrarily).
inline AdaptiveRKVResult adaptive_rkv_select(const std::vector<double> & block_mass, const std::vector<float> & S, int block_size,
                                             int max_blocks, double attention_mass, std::vector<double> * out_div = nullptr) {
    AdaptiveRKVResult res;
    const int n_blocks = (int) block_mass.size();
    const int n_ev = n_blocks * block_size;
    if (n_blocks == 0 || block_size <= 0 || max_blocks <= 0 || (int) S.size() != n_ev * n_ev) {
        return res;
    }
    max_blocks = std::min(max_blocks, n_blocks);

    // 1) mass-preserving blocks: highest mass first until `attention_mass` of the total is covered or the budget is used
    std::vector<int> order(n_blocks);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return block_mass[a] > block_mass[b]; });
    double total = 0.0;
    for (double m : block_mass) {
        total += m;
    }
    const double target = total * attention_mass;
    double acc = 0.0;
    std::vector<char> kept(n_blocks, 0);
    int n_kept = 0;
    size_t oi = 0;
    while (oi < order.size() && acc < target && n_kept < max_blocks) {
        kept[order[oi]] = 1;
        acc += block_mass[order[oi]];
        ++n_kept;
        ++oi;
    }
    res.n_mass_blocks = n_kept;

    // 2) the not-kept blocks are the diversity candidates; a block's diversity = -(mean redundancy of its token rows against the candidate tokens)
    std::vector<int> cand;
    for (int b = 0; b < n_blocks; ++b) {
        if (!kept[b]) {
            cand.push_back(b);
        }
    }
    const int fill = std::min(max_blocks - n_kept, (int) cand.size());
    if (fill > 0) {
        std::vector<double> div(n_blocks, 0.0);
        const double denom = (double) cand.size() * block_size;
        for (int b : cand) {
            double s = 0.0;
            for (int r = b * block_size; r < (b + 1) * block_size; ++r) {
                const float * srow = S.data() + (size_t) r * n_ev;
                for (int c : cand) {
                    for (int t = c * block_size; t < (c + 1) * block_size; ++t) {
                        s += srow[t];
                    }
                }
            }
            div[b] = -s / denom;
        }
        if (out_div) {
            *out_div = div;
        }
        std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) {   // closest to 0 (least redundant) first, then the heavier block
            return div[a] != div[b] ? div[a] > div[b] : block_mass[a] > block_mass[b];
        });
        for (int i = 0; i < fill; ++i) {
            kept[cand[i]] = 1;
        }
        res.n_diverse_blocks = fill;
    }
    for (int b = 0; b < n_blocks; ++b) {
        if (kept[b]) {
            res.kept_blocks.push_back(b);
        }
    }
    return res;
}

// ---- KVCrush ---------------------------------------------------------------------------------------------------------------------------------
enum kvcrush_anchor { KVCRUSH_ANCHOR_ZEROS = 0, KVCRUSH_ANCHOR_ONES, KVCRUSH_ANCHOR_ALTERNATING, KVCRUSH_ANCHOR_MEAN, KVCRUSH_ANCHOR_RANDOM };

// token_scores: importance per token of the evictable area (size n_blocks * block_size).  eligible: blocks the importance ranking has ALREADY chosen
// to evict (only those compete).  Returns up to `budget` of them (sorted by block index): the representatives of the eligible set.
inline std::vector<int> kvcrush_representatives(const std::vector<double> & token_scores, int block_size, const std::vector<int> & eligible,
                                                int budget, kvcrush_anchor anchor, uint32_t seed = 0) {
    std::vector<int> out;
    const int n_tokens = (int) token_scores.size();
    if (block_size <= 0 || n_tokens % block_size != 0 || budget <= 0 || eligible.empty()) {
        return out;                                              // the reference divides by zero here (kvcrush.cpp:136)
    }
    const int n_blocks = n_tokens / block_size;
    // 1) indicator: top half of the tokens by score
    std::vector<int> idx(n_tokens);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return token_scores[a] > token_scores[b]; });
    std::vector<int> ind(n_tokens, 0);
    for (int i = 0; i < n_tokens / 2; ++i) {
        ind[idx[i]] = 1;
    }
    // 2) anchor pattern of block_size bits
    std::vector<int> anc(block_size, 0);
    switch (anchor) {
        case KVCRUSH_ANCHOR_ZEROS: break;
        case KVCRUSH_ANCHOR_ONES: std::fill(anc.begin(), anc.end(), 1); break;
        case KVCRUSH_ANCHOR_ALTERNATING: for (int i = 0; i < block_size; ++i) anc[i] = i % 2; break;
        case KVCRUSH_ANCHOR_MEAN:
            for (int p = 0; p < block_size; ++p) {
                int s = 0;
                for (int b = 0; b < n_blocks; ++b) {
                    s += ind[b * block_size + p];
                }
                anc[p] = (2 * s > n_blocks) ? 1 : 0;           // mean > 0.5
            }
            break;
        case KVCRUSH_ANCHOR_RANDOM: {
            std::mt19937 rng(seed);
            std::uniform_int_distribution<int> d(0, 1);
            for (int & v : anc) v = d(rng);
        } break;
    }
    // 3) Hamming distance of every eligible block to the anchor
    std::vector<std::pair<int, int>> dist;   // (distance, block)
    for (int b : eligible) {
        if (b < 0 || b >= n_blocks) {
            continue;
        }
        int h = 0;
        for (int p = 0; p < block_size; ++p) {
            h += (ind[b * block_size + p] != anc[p]) ? 1 : 0;
        }
        dist.emplace_back(h, b);
    }
    if (dist.empty()) {
        return out;
    }
    std::stable_sort(dist.begin(), dist.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b) { return a.first < b.first; });
    // 4) uniformly spaced picks along the distance-sorted list
    const int n_ret = std::min<int>((int) dist.size(), budget);
    const int step = (int) dist.size() / n_ret;
    for (int i = 0; i < n_ret; ++i) {
        out.push_back(dist[(size_t) i * step].second);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ---- consumer glue: the SnapKV / H2O prompt prune (src/llama.cpp, after a multi-token prefill) ----------------------------------------------
// Rows: S69-openvino-PICK-KVCRUSH-DIVERSITY-RETENTION-2026-09-28-01.  The prune keeps a recent window and picks the rest of the budget from
// the older positions by attention mass alone.  With --kv-diversity-mass F (0 < F <= 1) the older positions are instead chosen block-wise by
// adaptive_rkv_select: blocks holding the top F of the mass first, the rest of the budget filled with the least redundant blocks.
//
// Geometry over n positions:  [0, sink) always kept (attention sinks)  |  n_blocks whole blocks of `block` positions (the evictable area)  |
// a short remainder of fewer than `block` positions, kept  |  [n - recent, n) the recent window, kept.
struct DiversityPlanGeom {
    int sink        = 0;
    int n_blocks    = 0;   // whole blocks in the evictable area
    int area_end    = 0;   // sink + n_blocks * block
    int tail_begin  = 0;   // n - recent
    int keep_blocks = 0;   // how many evictable blocks the budget leaves room for
    bool ok() const { return n_blocks > 0 && keep_blocks > 0 && keep_blocks < n_blocks; }
};

// budget: total positions to keep (sink + kept blocks + remainder + recent).  Returns a geometry whose ok() is false when the budget does not
// leave a whole block to choose, or already covers every block (the caller then keeps the mass-only plan).
inline DiversityPlanGeom diversity_plan_geometry(int n, int recent, int block, int sink, int budget) {
    DiversityPlanGeom g;
    if (n <= 0 || block <= 0 || recent < 0 || sink < 0 || sink + recent >= n) {
        return g;
    }
    g.sink       = sink;
    g.tail_begin = n - recent;
    g.n_blocks   = (g.tail_begin - sink) / block;
    g.area_end   = sink + g.n_blocks * block;
    const int fixed = sink + (g.tail_begin - g.area_end) + recent;   // positions kept whatever the selection does
    g.keep_blocks = budget > fixed ? (budget - fixed) / block : 0;
    return g;
}

// mass: per-position attention mass (size >= n).  S: kv_token_redundancy over the evictable area, i.e. computed on positions [0, area_end) with
// start_size = sink (n_ev = n_blocks * block), averaged over however many layers the caller sampled.  Returns the SORTED keep set, or an empty
// vector when the geometry is not ok() or S has the wrong size.
inline std::vector<int> diversity_keep_plan(const std::vector<float> & mass, int n, const std::vector<float> & S, const DiversityPlanGeom & g,
                                            int block, double attention_mass, AdaptiveRKVResult * out_res = nullptr) {
    std::vector<int> keep;
    const size_t n_ev = (size_t) g.n_blocks * (size_t) (block > 0 ? block : 0);
    if (!g.ok() || (int) mass.size() < n || S.size() != n_ev * n_ev) {
        return keep;
    }
    std::vector<double> block_mass((size_t) g.n_blocks, 0.0);
    for (int b = 0; b < g.n_blocks; ++b) {
        for (int p = g.sink + b * block; p < g.sink + (b + 1) * block; ++p) {
            block_mass[(size_t) b] += mass[(size_t) p];
        }
    }
    AdaptiveRKVResult r = adaptive_rkv_select(block_mass, S, block, g.keep_blocks, attention_mass);
    for (int p = 0; p < g.sink; ++p) {
        keep.push_back(p);
    }
    for (int b : r.kept_blocks) {
        for (int p = g.sink + b * block; p < g.sink + (b + 1) * block; ++p) {
            keep.push_back(p);
        }
    }
    for (int p = g.area_end; p < n; ++p) {      // remainder + recent window
        keep.push_back(p);
    }
    if (out_res) {
        *out_res = std::move(r);
    }
    return keep;                                // already sorted: sink < blocks (ascending) < remainder < recent
}

} // namespace justician_kv
