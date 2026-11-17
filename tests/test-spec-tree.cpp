// common_spec_tree: draft tree builder + acceptance walk (common/spec-tree.h).  Row S69-MNN-PICK-EAGLE-DYNAMIC-DRAFT-TREE-2026-09-29-01 (step 1).
// GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: the header did not exist (the fork's draft interface returns one flat token vector, common/speculative.cpp:784-788).  The expected values
// are HAND-DERIVED from the algorithm (topK 2, depth 3), not produced by running the builder:
//   level 1: ids 10/11, logp -0.1/-0.7;  level 2: from 10: 20/21 (-0.2/-0.9), from 11: 22/23 (-0.1/-0.5);  level 3: from 20: 30/31 (-0.05/-2.0), from 22: 32/33 (-0.3/-0.4)
//   cumulative: n0 10:-0.1  n1 11:-0.7  n2 20:-0.3  n3 21:-1.0  n4 22:-0.8  n5 23:-1.2  n6 30:-0.35  n7 31:-2.3  n8 32:-1.1  n9 33:-1.2
//   active leaves after level 2 = n2 (parent slot 0), n4 (parent slot 1); after level 3 = n6, n8.
//   finalize(root 5, max 6) keeps n0 n2 n6 n1 n4 n3 (cumulative -0.1 -0.3 -0.35 -0.7 -0.8 -1.0) in creation order n0 n1 n2 n3 n4 n6.
#include "spec-tree.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    {
        common_spec_tree t(2);
        const int32_t l1i[2] = { 10, 11 };
        const float   l1p[2] = { -0.1f, -0.7f };
        t.init(l1i, l1p);
        CHECK((t.active_tokens() == std::vector<int32_t>{ 10, 11 }));
        const int32_t l2i[4] = { 20, 21, 22, 23 };
        const float   l2p[4] = { -0.2f, -0.9f, -0.1f, -0.5f };
        CHECK(t.grow(l2i, l2p) == 2);
        CHECK((t.active_tokens() == std::vector<int32_t>{ 20, 22 }));
        {
            // draft mask after level 2: slot 0 = child of n0, slot 1 = child of n1; columns = 2 (level 1) + 2 (level 2)
            const auto & m = t.active_mask();
            CHECK(m.size() == 2 && m[0].size() == 4);
            CHECK((m[0] == std::vector<uint8_t>{ 1, 0, 1, 0 }));
            CHECK((m[1] == std::vector<uint8_t>{ 0, 1, 0, 1 }));
        }
        const int32_t l3i[4] = { 30, 31, 32, 33 };
        const float   l3p[4] = { -0.05f, -2.0f, -0.3f, -0.4f };
        CHECK(t.grow(l3i, l3p) == 2);
        CHECK((t.active_tokens() == std::vector<int32_t>{ 30, 32 }));
        {
            const auto & m = t.active_mask();
            CHECK(m[0].size() == 6);
            CHECK((m[0] == std::vector<uint8_t>{ 1, 0, 1, 0, 1, 0 }));
            CHECK((m[1] == std::vector<uint8_t>{ 0, 1, 0, 1, 0, 1 }));
        }
        CHECK(t.n_nodes() == 10);

        const common_spec_tree_out o = t.finalize(5, 6);
        CHECK((o.tokens     == std::vector<int32_t>{ 5, 10, 11, 20, 21, 22, 30 }));
        CHECK((o.pos_offset == std::vector<int32_t>{ 0, 1, 1, 2, 2, 2, 3 }));
        CHECK((o.parent     == std::vector<int32_t>{ -1, 0, 0, 1, 1, 2, 3 }));
        CHECK(o.mask.size() == 7);
        CHECK((o.mask[0] == std::vector<uint8_t>{ 1, 0, 0, 0, 0, 0, 0 }));
        CHECK((o.mask[1] == std::vector<uint8_t>{ 1, 1, 0, 0, 0, 0, 0 }));
        CHECK((o.mask[2] == std::vector<uint8_t>{ 1, 0, 1, 0, 0, 0, 0 }));
        CHECK((o.mask[3] == std::vector<uint8_t>{ 1, 1, 0, 1, 0, 0, 0 }));
        CHECK((o.mask[4] == std::vector<uint8_t>{ 1, 1, 0, 0, 1, 0, 0 }));
        CHECK((o.mask[5] == std::vector<uint8_t>{ 1, 0, 1, 0, 0, 1, 0 }));
        CHECK((o.mask[6] == std::vector<uint8_t>{ 1, 1, 0, 1, 0, 0, 1 }));
        CHECK(o.paths.size() == 3);
        CHECK(o.paths.size() == 3 && (o.paths[0] == std::vector<int32_t>{ 0, 1, 4 }));
        CHECK(o.paths.size() == 3 && (o.paths[1] == std::vector<int32_t>{ 0, 2, 5 }));
        CHECK(o.paths.size() == 3 && (o.paths[2] == std::vector<int32_t>{ 0, 1, 3, 6 }));

        // acceptance: the deepest path agrees all the way, the first path diverges at its second node, the second at its first
        {
            const std::vector<int32_t> am = { 10, 20, 99, 30, 98, 97, 77 };
            const common_spec_tree_accept_result a = common_spec_tree_accept(o, am);
            CHECK((a.indices == std::vector<int32_t>{ 0, 1, 3, 6 }));
            CHECK((a.tokens  == std::vector<int32_t>{ 10, 20, 30, 77 }));
            CHECK(a.n_draft_accepted == 3);
        }
        // no path agrees: only the target's own token at the root is emitted (a plain decode step)
        {
            const std::vector<int32_t> am = { 42, 1, 1, 1, 1, 1, 1 };
            const common_spec_tree_accept_result a = common_spec_tree_accept(o, am);
            CHECK((a.indices == std::vector<int32_t>{ 0 }));
            CHECK((a.tokens  == std::vector<int32_t>{ 42 }));
            CHECK(a.n_draft_accepted == 0);
        }
        // the other branch wins when it agrees longer: root says 11, node 2 (tok 11) says 22 -> path [0,2,5]
        {
            const std::vector<int32_t> am = { 11, 0, 22, 0, 0, 55, 0 };
            const common_spec_tree_accept_result a = common_spec_tree_accept(o, am);
            CHECK((a.indices == std::vector<int32_t>{ 0, 2, 5 }));
            CHECK((a.tokens  == std::vector<int32_t>{ 11, 22, 55 }));
            CHECK(a.n_draft_accepted == 2);
        }
        // size mismatch -> empty result
        CHECK(common_spec_tree_accept(o, std::vector<int32_t>{ 1, 2 }).tokens.empty());
        // max_nodes larger than the tree keeps everything (10 nodes + root); 0 keeps only the root
        CHECK(t.finalize(5, 99).tokens.size() == 11);
        const common_spec_tree_out o0 = t.finalize(5, 0);
        CHECK(o0.tokens.size() == 1 && o0.paths.empty());
    }
    // ancestor closure on ties (the reference's unstable sort can drop a parent and keep its child when the cumulative values are equal):
    // top-1 chain whose log-probabilities are exactly 0, so every node ties at 0.  max_nodes 2 must keep the two SHALLOWEST nodes.
    {
        common_spec_tree t(1);
        const int32_t i1[1] = { 1 }; const float p0[1] = { 0.0f };
        t.init(i1, p0);
        const int32_t i2[1] = { 2 };
        t.grow(i2, p0);
        const int32_t i3[1] = { 3 };
        t.grow(i3, p0);
        const common_spec_tree_out o = t.finalize(9, 2);
        CHECK((o.tokens == std::vector<int32_t>{ 9, 1, 2 }));
        CHECK((o.parent == std::vector<int32_t>{ -1, 0, 1 }));
        CHECK(o.paths.size() == 1 && (o.paths[0] == std::vector<int32_t>{ 0, 1, 2 }));
        // a positive "logprob" is clamped to 0 (a malformed draft head cannot make a child outrank its parent)
        common_spec_tree u(1);
        const float pp[1] = { 5.0f };
        u.init(i1, pp);
        u.grow(i2, pp);
        const common_spec_tree_out ou = u.finalize(9, 1);
        CHECK((ou.tokens == std::vector<int32_t>{ 9, 1 }));
    }
    // a draft head with fewer than k candidates (token -1) is tolerated
    {
        common_spec_tree t(2);
        const int32_t a[2] = { 7, -1 }; const float p[2] = { -0.2f, 0.0f };
        t.init(a, p);
        CHECK((t.active_tokens() == std::vector<int32_t>{ 7 }));
        const int32_t b[2] = { 8, -1 }; const float q[2] = { -0.1f, 0.0f };
        CHECK(t.grow(b, q) == 1);
        const common_spec_tree_out o = t.finalize(1, 8);
        CHECK((o.tokens == std::vector<int32_t>{ 1, 7, 8 }));
    }
    // ---- stochastic target_only tree verify (row S69-ENGREV-DFLASHTFM-TREE-TARGET-ONLY-VERIFY-2026-10-27-01) ----
    // FAIL-BEFORE: common_spec_tree_accept_sample did not exist, so this block did not compile.
    // EXACT-DISTRIBUTION TEST: vocab 3, a branching tree (root 5; children 0,1; node1 children 0,2; node2 (tok 1) child 1; ...).  Every node has its own
    // fixed target distribution.  The walk draws ONE uniform per visited node; the uniforms of different nodes are independent, so the full product space of
    // uniform cells (one cell per tree node, midpoint of the cell) is enumerated and each cell combination weighted by the product of its cell widths.  The
    // summed weight per emitted token sequence must equal the exactly enumerated target sequence distribution (1e-10).
    {
        common_spec_tree t(2);
        const int32_t l1i[2] = { 0, 1 };
        const float   l1p[2] = { -0.3f, -1.0f };
        t.init(l1i, l1p);
        const int32_t l2i[4] = { 0, 2, 1, 2 };
        const float   l2p[4] = { -0.2f, -0.9f, -0.1f, -0.5f };
        t.grow(l2i, l2p);
        const int32_t l3i[4] = { 1, 2, 0, 1 };
        const float   l3p[4] = { -0.05f, -2.0f, -0.3f, -0.4f };
        t.grow(l3i, l3p);
        const common_spec_tree_out o = t.finalize(2, 99);
        const int n = (int) o.tokens.size();
        CHECK(n >= 5);

        // one fixed, non-degenerate target distribution per node (rows sum to 1, one zero entry to cover a zero-width cell)
        std::vector<std::vector<double>> dist(n);
        for (int i = 0; i < n; ++i) {
            const double a = 0.15 + 0.05 * (i % 5);
            const double b = 0.30 + 0.04 * ((i * 3) % 4);
            dist[i] = { a, b, 1.0 - a - b };
        }
        dist[n - 1] = { 0.5, 0.0, 0.5 };

        std::vector<std::vector<int32_t>> kids(n);
        for (int i = 1; i < n; ++i) {
            kids[o.parent[i]].push_back(i);
        }

        // exact target sequence distribution by recursion over the tree
        std::map<std::vector<int32_t>, double> exact;
        std::vector<int32_t> prefix;
        std::function<void(int, double)> rec = [&](int node, double w) {
            for (int tok = 0; tok < 3; ++tok) {
                const double pw = w * dist[node][tok];
                if (pw <= 0.0) {
                    continue;
                }
                prefix.push_back(tok);
                int child = -1;
                for (int c : kids[node]) {
                    if (o.tokens[c] == tok) { child = c; break; }
                }
                if (child >= 0) {
                    rec(child, pw);
                } else {
                    exact[prefix] += pw;
                }
                prefix.pop_back();
            }
        };
        rec(0, 1.0);

        // enumerate the product of one uniform cell per node
        std::map<std::vector<int32_t>, double> got;
        std::vector<int> cell(n, 0);
        long n_combo = 0;
        for (;;) {
            double w = 1.0;
            std::vector<double> u(n);
            for (int i = 0; i < n; ++i) {
                double lo = 0.0;
                for (int c = 0; c < cell[i]; ++c) { lo += dist[i][c]; }
                u[i] = lo + 0.5 * dist[i][cell[i]];
                w *= dist[i][cell[i]];
            }
            if (w > 0.0) {
                const common_spec_tree_accept_result r = common_spec_tree_accept_sample(
                    o, [&](int node) { return dist[node]; }, [&](int node) { return u[node]; });
                got[r.tokens] += w;
                CHECK(r.indices.size() == r.tokens.size());
                CHECK(r.n_draft_accepted == (int32_t) r.tokens.size() - 1);
                n_combo++;
            }
            int k = 0;
            while (k < n && ++cell[k] == 3) { cell[k] = 0; ++k; }
            if (k == n) { break; }
        }
        CHECK(n_combo > 1000);
        double total = 0.0;
        double worst = 0.0;
        for (const auto & kv : exact) {
            const auto it = got.find(kv.first);
            const double g = it == got.end() ? 0.0 : it->second;
            worst = std::max(worst, std::fabs(g - kv.second));
            total += g;
        }
        CHECK(got.size() == exact.size());
        CHECK(worst < 1e-10);
        CHECK(std::fabs(total - 1.0) < 1e-10);

        // greedy case: one-hot nodes must equal the argmax walker (tokens, indices, count)
        {
            const std::vector<int32_t> am = { 0, 0, 2, 1, 1, 2, 0, 1, 0, 2, 1, 0, 2, 2 };
            std::vector<int32_t> amv(am.begin(), am.begin() + n);
            const common_spec_tree_accept_result g = common_spec_tree_accept(o, amv);
            const common_spec_tree_accept_result s = common_spec_tree_accept_sample(
                o, [&](int node) { std::vector<double> d(3, 0.0); d[amv[node]] = 1.0; return d; }, [&](int) { return 0.37; });
            CHECK(g.tokens == s.tokens && g.indices == s.indices && g.n_draft_accepted == s.n_draft_accepted);
        }
        // an unusable root distribution (all zero / empty) yields an empty result, never a bogus token
        CHECK(common_spec_tree_accept_sample(o, [&](int) { return std::vector<double>{}; }, [&](int) { return 0.5; }).tokens.empty());
        CHECK(common_spec_tree_accept_sample(o, [&](int) { return std::vector<double>(3, 0.0); }, [&](int) { return 0.5; }).tokens.empty());
        // an uniform of exactly 1.0 (out of contract) clamps to the last positive cell instead of reading past the end
        CHECK(common_spec_tree_accept_sample(o, [&](int) { return std::vector<double>{ 0.0, 0.0, 1.0 }; }, [&](int) { return 1.0; }).tokens.size() >= 1);
    }
    if (n_fail) {
        fprintf(stderr, "test-spec-tree: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-spec-tree: all passed\n");
    return 0;
}
