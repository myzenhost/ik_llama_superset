// spec-tree.h -- draft TREE builder + acceptance walk for tree speculative decoding (EAGLE-style top-K expansion with cumulative-logprob pruning).
// Header-only, no llama dependency beyond plain integers: a draft head gives (token, logprob) candidates, this builds the verify batch (tokens, position
// offsets, ancestry mask, root-to-leaf paths) and, after ONE target forward, picks the longest path the target agrees with.
// Row S69-MNN-PICK-EAGLE-DYNAMIC-DRAFT-TREE-2026-09-29-01  (step 1: the tree builder + acceptance; the tree-mask VERIFY batch that consumes it is the row
// S69-F1-PICK-TREE-MASK-VERIFY-BATCH-API-2026-10-05-01, which is why this file is not yet called from common/speculative.cpp).
//
// SPEC READ FROM (behaviour only, own implementation; Apache-2.0, alibaba/MNN @ 67450001, transformers/llm/engine/src/speculative_decoding/tokentree.hpp
// :24-213 and eagle.cpp:253-298, read in full):
//   * init(): the root's top-K children seed the K active leaves;
//   * grow(): every active leaf proposes K children, ALL candidates are ranked by cumulative log-probability and the best K become the next active leaves;
//     the draft model's attention mask for the next step is the parent's row extended by K columns with a self bit;
//   * finalize(): keep the best `max_nodes` nodes by cumulative log-probability, put them back in creation order, emit tokens (root = the sampled token at
//     index 0), position offsets (depth + 1, root 0), the ancestry mask and one root-to-leaf index path per leaf;
//   * acceptance: argmax of the target at every tree node; along each path keep walking while argmax(parent) == token(child); the longest path wins and the
//     argmax of its last node is the bonus token.
// DIFFERENCES FROM THE REFERENCE (deliberate, each covered by tests/test-spec-tree.cpp):
//   * the pruning order is (cumulative logprob desc, depth asc, creation id asc).  The reference sorts with an unstable std::sort on cumulative logprob only: on
//     a tie (a logprob of exactly 0) it can keep a child and drop its parent, leaving an orphan node whose mask row has no ancestors.  Here a parent always
//     outranks its child (log-probabilities are clamped to <= 0, so a child's cumulative value never exceeds its parent's, and ties go to the shallower node).
//   * a candidate with token < 0 is ignored (a draft head that returns fewer than K candidates).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct common_spec_tree_out {
    std::vector<int32_t>               tokens;      // [0] = the already sampled token (root), then the kept draft nodes in creation order
    std::vector<int32_t>               pos_offset;  // relative to the root position: root 0, a node at depth d (root children d = 0) gets d + 1
    std::vector<int32_t>               parent;      // index into `tokens` of the parent (root: -1, root's children: 0)
    std::vector<std::vector<uint8_t>>  mask;        // (n x n) row i = ancestors of i, i itself and the root
    std::vector<std::vector<int32_t>>  paths;       // one root-to-leaf list of indices into `tokens` per leaf (starts with 0)
};

struct common_spec_tree_accept_result {
    std::vector<int32_t> indices;   // indices into the tree's token array, root first; the LAST one is the node whose argmax is the bonus token
    std::vector<int32_t> tokens;    // the tokens to emit: argmax at every index (the accepted draft tokens followed by the bonus token)
    int32_t              n_draft_accepted = 0;   // accepted draft nodes (excludes the root and the bonus token)
};

class common_spec_tree {
public:
    explicit common_spec_tree(int top_k) : k(top_k > 0 ? top_k : 1) {
        mask.assign(k, std::vector<uint8_t>(k, 0));
        for (int i = 0; i < k; ++i) {
            mask[i][i] = 1;
        }
    }

    // k candidates for the first level (token ids and log-probabilities, best first is not required)
    void init(const int32_t * ids, const float * logp) {
        actives.clear();
        for (int i = 0; i < k; ++i) {
            if (ids[i] < 0) {
                continue;
            }
            actives.push_back(add_node(ids[i], logp[i], -1));
        }
        mask.assign(actives.size(), std::vector<uint8_t>(k, 0));   // one row per active leaf, k columns (a skipped candidate leaves its column empty)
        for (size_t i = 0; i < actives.size(); ++i) {
            mask[i][i] = 1;
        }
    }

    // k candidates for EACH active leaf, row-major [n_active][k]; returns the number of active leaves afterwards
    int grow(const int32_t * ids, const float * logp) {
        struct cand { int node; int from; };
        std::vector<cand> cands;
        const int n_act = (int) actives.size();
        for (int a = 0; a < n_act; ++a) {
            for (int j = 0; j < k; ++j) {
                const int32_t t = ids[(size_t) a * k + j];
                if (t < 0) {
                    continue;
                }
                cands.push_back({ add_node(t, logp[(size_t) a * k + j], actives[a]), a });
            }
        }
        std::stable_sort(cands.begin(), cands.end(), [&](const cand & x, const cand & y) {
            return nodes[x.node].cum > nodes[y.node].cum;
        });
        const size_t n_new = std::min<size_t>((size_t) k, cands.size());
        std::vector<std::vector<uint8_t>> new_mask;
        std::vector<int> new_act;
        const size_t old_w = mask.empty() ? 0 : mask[0].size();
        for (size_t i = 0; i < n_new; ++i) {
            std::vector<uint8_t> row(old_w + k, 0);
            const auto & prow = mask[cands[i].from];
            std::copy(prow.begin(), prow.end(), row.begin());
            row[old_w + i] = 1;
            new_mask.push_back(std::move(row));
            new_act.push_back(cands[i].node);
        }
        mask    = std::move(new_mask);
        actives = std::move(new_act);
        return (int) actives.size();
    }

    // tokens of the active leaves (the batch fed to the next draft step) and the draft model's attention mask rows for them
    std::vector<int32_t> active_tokens() const {
        std::vector<int32_t> r;
        for (int a : actives) {
            r.push_back(nodes[a].token);
        }
        return r;
    }
    const std::vector<std::vector<uint8_t>> & active_mask() const { return mask; }
    size_t n_nodes() const { return nodes.size(); }

    common_spec_tree_out finalize(int32_t root_token, int max_nodes) const {
        std::vector<int> order(nodes.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = (int) i;
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            if (nodes[a].cum != nodes[b].cum) {
                return nodes[a].cum > nodes[b].cum;
            }
            if (nodes[a].depth != nodes[b].depth) {
                return nodes[a].depth < nodes[b].depth;
            }
            return a < b;
        });
        if (max_nodes < 0) {
            max_nodes = 0;
        }
        if ((size_t) max_nodes < order.size()) {
            order.resize(max_nodes);
        }
        std::sort(order.begin(), order.end());   // creation order

        common_spec_tree_out out;
        const size_t n = order.size() + 1;
        out.tokens.reserve(n);
        out.pos_offset.reserve(n);
        out.parent.reserve(n);
        std::vector<int> idx_of(nodes.size(), -1);
        out.tokens.push_back(root_token);
        out.pos_offset.push_back(0);
        out.parent.push_back(-1);
        for (size_t i = 0; i < order.size(); ++i) {
            const node & nd = nodes[order[i]];
            idx_of[order[i]] = (int) i + 1;
            out.tokens.push_back(nd.token);
            out.pos_offset.push_back(nd.depth + 1);
            out.parent.push_back(nd.parent < 0 ? 0 : idx_of[nd.parent]);   // the parent has a smaller id, so it was already mapped (or pruned: -1)
        }
        out.mask.assign(n, std::vector<uint8_t>(n, 0));
        out.mask[0][0] = 1;
        for (size_t i = 1; i < n; ++i) {
            out.mask[i][i] = 1;
            const int p = out.parent[i];
            if (p >= 0) {
                for (size_t j = 0; j < n; ++j) {
                    if (out.mask[p][j]) {
                        out.mask[i][j] = 1;
                    }
                }
            }
        }
        std::vector<uint8_t> has_child(n, 0);
        for (size_t i = 1; i < n; ++i) {
            if (out.parent[i] >= 0) {
                has_child[out.parent[i]] = 1;
            }
        }
        for (size_t i = 1; i < n; ++i) {
            if (has_child[i]) {
                continue;
            }
            std::vector<int32_t> path;
            for (int c = (int) i; c >= 0; c = out.parent[c]) {
                path.push_back(c);
            }
            std::reverse(path.begin(), path.end());
            out.paths.push_back(std::move(path));
        }
        return out;
    }

private:
    struct node {
        int32_t token;
        double  cum;
        int     depth;
        int     parent;   // node id, -1 for a child of the root
    };

    int add_node(int32_t token, float logp, int parent) {
        node nd;
        nd.token  = token;
        const double lp = logp > 0.0f ? 0.0 : (double) logp;   // log-probabilities cannot be positive; the clamp keeps parent >= child
        nd.parent = parent;
        nd.depth  = parent < 0 ? 0 : nodes[parent].depth + 1;
        nd.cum    = (parent < 0 ? 0.0 : nodes[parent].cum) + lp;
        nodes.push_back(nd);
        return (int) nodes.size() - 1;
    }

    int                                k;
    std::vector<node>                  nodes;
    std::vector<int>                   actives;
    std::vector<std::vector<uint8_t>>  mask;
};

// target_argmax[i] = the target's argmax at tree node i (size == tree.tokens.size()).  Returns the longest agreeing path plus the bonus token.
inline common_spec_tree_accept_result common_spec_tree_accept(const common_spec_tree_out & tree, const std::vector<int32_t> & target_argmax) {
    common_spec_tree_accept_result res;
    if (tree.tokens.empty() || target_argmax.size() != tree.tokens.size()) {
        return res;
    }
    std::vector<int32_t> best;   // accepted node indices, root first (each is a node whose argmax matched its child)
    int32_t next = 0;
    for (const auto & path : tree.paths) {
        std::vector<int32_t> cand;
        int32_t nxt = 0;
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            if (target_argmax[path[i]] != tree.tokens[path[i + 1]]) {
                break;
            }
            cand.push_back(path[i]);
            nxt = path[i + 1];
        }
        if (cand.size() > best.size()) {
            best = cand;
            next = nxt;
        }
    }
    best.push_back(next);          // with no agreeing path this is the root: the bonus token is the target's own next token
    res.indices = best;
    for (int32_t i : best) {
        res.tokens.push_back(target_argmax[i]);
    }
    res.n_draft_accepted = (int32_t) best.size() - 1;
    return res;
}

// STOCHASTIC (target_only) LOSSLESS TREE VERIFY.  Row S69-ENGREV-DFLASHTFM-TREE-TARGET-ONLY-VERIFY-2026-10-27-01.
// SPEC READ FROM (behaviour only, own implementation; trymirai/sglang DFlash-TfM sha aeac03f0d4c8789559411be95c5c127bdff24d1c, Apache-2.0,
// speculative/tree_sampling.py 1-203; SOURCE-READ, no code copied): from the root, draw ONE token from the target distribution of the current node using a
// per-node uniform; if a child of the node holds that token, descend into it, otherwise the draw is the bonus token and the walk stops.  The draft tree is
// treated as deterministic (no draft probabilities), so the emitted stream is exactly target-distributed: the draw at every visited node is a plain sample
// of the target, and the tree only decides whether the next node's distribution (already computed in the same forward) can be reused.
//   dist_fn(node)    -> std::vector<double>: the target's sampling distribution at tree node `node` (index = token id; the caller has already applied the
//                       sampler chain: temperature, top-k, top-p ...; need not be normalised, negative entries count as 0)
//   uniform_fn(node) -> double in [0, 1): the uniform for that node; the caller keys it on (seed, absolute position, stream) so spec on == spec off output
// RESTRICTIONS (as in the engine, v1): penalties and grammar need sampler state per path, so the caller must refuse them in tree mode (or clone the sampler
// per path); this function only sees finished distributions.
// An unusable distribution (empty or no positive weight) at the node where a draw is needed ends the walk: at the root that gives an empty result.
template <class DistFn, class UniFn>
inline common_spec_tree_accept_result common_spec_tree_accept_sample(const common_spec_tree_out & tree, DistFn dist_fn, UniFn uniform_fn) {
    common_spec_tree_accept_result res;
    const int n = (int) tree.tokens.size();
    if (n == 0 || (int) tree.parent.size() != n) {
        return res;
    }
    std::vector<std::vector<int32_t>> kids(n);
    for (int i = 1; i < n; ++i) {
        if (tree.parent[i] >= 0 && tree.parent[i] < n) {
            kids[tree.parent[i]].push_back(i);
        }
    }
    int node = 0;
    for (;;) {
        const std::vector<double> d = dist_fn(node);
        double total = 0.0;
        int    last  = -1;
        for (size_t t = 0; t < d.size(); ++t) {
            if (d[t] > 0.0) {
                total += d[t];
                last = (int) t;
            }
        }
        if (last < 0 || !(total > 0.0)) {
            return res;   // nothing to draw from: stop here, keep what was accepted so far
        }
        double u = uniform_fn(node);
        if (!(u >= 0.0)) { u = 0.0; }
        const double target = u * total;
        double acc = 0.0;
        int    tok = last;   // a uniform at the float edge (>= 1) lands on the last positive cell
        for (size_t t = 0; t < d.size(); ++t) {
            if (d[t] <= 0.0) {
                continue;
            }
            acc += d[t];
            if (target < acc) {
                tok = (int) t;
                break;
            }
        }
        res.indices.push_back(node);
        res.tokens.push_back(tok);
        int next = -1;
        for (int c : kids[node]) {
            if (tree.tokens[c] == tok) {
                next = c;
                break;
            }
        }
        if (next < 0) {
            break;   // the sample is the bonus token
        }
        node = next;
    }
    res.n_draft_accepted = (int32_t) res.tokens.size() - 1;
    return res;
}
