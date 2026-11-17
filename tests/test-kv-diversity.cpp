// Diversity-aware KV retention (src/llama-kv-diversity.h): AdaptiveR-KV block diversity + KVCrush representatives.
// Row S69-openvino-PICK-KVCRUSH-DIVERSITY-RETENTION-2026-09-28-01.   GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: the header did not exist.  The row's own scenario is the partner: two near-duplicate blocks both score above the mass threshold,
// a unique lower-scored block exists.  A pure attention-mass ranking (what snapkv_keep_plan / H2OAccumulator do) keeps BOTH duplicates and evicts the
// unique block (asserted in A as the BEFORE behaviour); the diversity selection keeps neither duplicate and the unique block (asserted in A as AFTER).
//   A  scenario above, 8 blocks x 4 tokens, keys = distinct basis vectors (cosine 0) except block 2 = block 1 (cosine 1); expected diversity of the
//      duplicate blocks = -4/28 exactly, 0 for the rest; mass-only keep {1,2,3}, diverse keep {0,3,5}.
//   B  kv_token_redundancy: sink tokens are excluded (matrix is n_tokens - start_size wide), identical keys give 1 off the diagonal, the diagonal is
//      0, below-row-mean entries are zeroed, the head average of two identical heads equals one head.
//   C  controls: budget 0 / empty S / mismatched S give an empty plan; max_blocks >= n_blocks keeps everything; attention_mass 1.0 keeps by mass only.
//   D  KVCrush: block b has ones_b top-half tokens, anchor ZEROS, eight eligible blocks, budget 4 -> the uniformly spaced picks {0,2,4,6};
//      budget 0 and an empty eligible set return empty (the reference divides by zero there); budget above the eligible count returns all of them.
#include "../src/llama-kv-diversity.h"

#include <cstdio>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static bool has(const std::vector<int> & v, int x) {
    for (int e : v) {
        if (e == x) {
            return true;
        }
    }
    return false;
}

int main() {
    using namespace justician_kv;

    // ---- A -----------------------------------------------------------------------------------------------------------------------------------
    {
        const int block = 4, nb = 8, start = 2, dim = 64, nh = 1;
        const int n_ev = nb * block;
        const int n_tok = n_ev + start;
        std::vector<float> K((size_t) n_tok * nh * dim, 0.0f);
        // sink tokens 0,1: basis vectors 60,61 (never part of the matrix)
        K[(size_t) 0 * dim + 60] = 1.0f;
        K[(size_t) 1 * dim + 61] = 1.0f;
        for (int t = 0; t < n_ev; ++t) {
            int basis = t;
            if (t >= 2 * block && t < 3 * block) {
                basis = t - block;          // block 2 duplicates block 1
            }
            K[(size_t) (start + t) * dim + basis] = 1.0f;
        }
        const std::vector<float> S = kv_token_redundancy(K.data(), n_tok, nh, dim, start);
        CHECK((int) S.size() == n_ev * n_ev);
        const std::vector<double> mass = { 0.01, 0.10, 0.10, 0.50, 0.01, 0.05, 0.01, 0.01 };

        // BEFORE: pure mass ranking with a 3-block budget
        std::vector<int> order = { 0, 1, 2, 3, 4, 5, 6, 7 };
        for (int i = 0; i < nb; ++i) {
            for (int j = i + 1; j < nb; ++j) {
                if (mass[order[j]] > mass[order[i]]) {
                    std::swap(order[i], order[j]);
                }
            }
        }
        const std::vector<int> mass_only(order.begin(), order.begin() + 3);
        CHECK(has(mass_only, 3) && has(mass_only, 1) && has(mass_only, 2));   // both duplicates kept ...
        CHECK(!has(mass_only, 5));                                            // ... unique block evicted

        // AFTER
        std::vector<double> div;
        const AdaptiveRKVResult r = adaptive_rkv_select(mass, S, block, 3, 0.5, &div);
        CHECK(r.n_mass_blocks == 1);
        CHECK(r.n_diverse_blocks == 2);
        CHECK(r.kept_blocks.size() == 3);
        CHECK(has(r.kept_blocks, 3));          // the heaviest block, kept for its mass
        CHECK(!has(r.kept_blocks, 1) && !has(r.kept_blocks, 2));   // neither near-duplicate
        CHECK(has(r.kept_blocks, 5));          // the unique block survives (tie on diversity 0 -> heavier block first)
        CHECK(has(r.kept_blocks, 0));          // second zero-diversity block, lowest index
        CHECK((int) div.size() == nb);
        const double expect_dup = -4.0 / 28.0;
        CHECK(std::fabs(div[1] - expect_dup) < 1e-6 && std::fabs(div[2] - expect_dup) < 1e-6);
        CHECK(std::fabs(div[0]) < 1e-9 && std::fabs(div[5]) < 1e-9 && std::fabs(div[7]) < 1e-9);
        std::vector<int> sorted_ok = r.kept_blocks;
        for (size_t i = 1; i < sorted_ok.size(); ++i) {
            CHECK(sorted_ok[i - 1] < sorted_ok[i]);
        }

        // ---- C controls on the same data
        CHECK(adaptive_rkv_select(mass, S, block, 0, 0.5).kept_blocks.empty());
        CHECK(adaptive_rkv_select(mass, std::vector<float>(), block, 3, 0.5).kept_blocks.empty());
        CHECK(adaptive_rkv_select(mass, std::vector<float>(S.begin(), S.end() - 1), block, 3, 0.5).kept_blocks.empty());
        CHECK(adaptive_rkv_select(mass, S, block, 99, 0.5).kept_blocks.size() == (size_t) nb);
        const AdaptiveRKVResult rm = adaptive_rkv_select(mass, S, block, 3, 1.0);   // preserve 100% of the mass: budget caps at 3, all by mass
        CHECK(rm.n_mass_blocks == 3 && rm.n_diverse_blocks == 0);
        CHECK(has(rm.kept_blocks, 3) && has(rm.kept_blocks, 1) && has(rm.kept_blocks, 2));
    }

    // ---- B -----------------------------------------------------------------------------------------------------------------------------------
    {
        const int dim = 4, start = 1;
        // tokens: 0 sink (== token 1), 1 and 2 identical, 3 orthogonal.  two heads, both identical.
        const float k1[4] = { 1, 0, 0, 0 }, k3[4] = { 0, 1, 0, 0 };
        std::vector<float> K;
        auto push = [&](const float * v) { for (int h = 0; h < 2; ++h) K.insert(K.end(), v, v + dim); };
        push(k1); push(k1); push(k1); push(k3);
        const std::vector<float> S = kv_token_redundancy(K.data(), 4, 2, dim, start);
        CHECK(S.size() == 9);                                    // 3 x 3: the sink token is excluded
        CHECK(S[0] == 0.0f && S[4] == 0.0f && S[8] == 0.0f);     // diagonal
        CHECK(std::fabs(S[1] - 1.0f) < 1e-6f && std::fabs(S[3] - 1.0f) < 1e-6f);   // tokens (1,2) identical -> 1 (head average of two equal heads)
        CHECK(S[2] == 0.0f && S[5] == 0.0f && S[6] == 0.0f && S[7] == 0.0f);       // the orthogonal token: zero
        // below-mean zeroing: row 0 has sims {0 (diag), 1, 0}; mean = 1/3; the two zeros are below the mean (already 0) and the 1 survives
        // a row where the mean exceeds a small positive entry: tokens a, b, c with sims a-b = 0.1, a-c = 0.9: mean (0.1+0.9)/3 = 0.333, so 0.1 is zeroed
        const float a[2] = { 1.0f, 0.0f };
        const float b[2] = { 0.1f, 0.994987f };
        const float c[2] = { 0.9f, 0.435890f };
        std::vector<float> K2;
        K2.insert(K2.end(), a, a + 2); K2.insert(K2.end(), b, b + 2); K2.insert(K2.end(), c, c + 2);
        const std::vector<float> S2 = kv_token_redundancy(K2.data(), 3, 1, 2, 0);
        CHECK(S2.size() == 9);
        CHECK(S2[1] == 0.0f);                                    // a-b = 0.1 < row mean 0.333 -> zeroed
        CHECK(std::fabs(S2[2] - 0.9f) < 1e-4f);                  // a-c = 0.9 kept
        CHECK(kv_token_redundancy(K2.data(), 3, 1, 2, 3).empty());   // everything is sink
    }

    // ---- D -----------------------------------------------------------------------------------------------------------------------------------
    {
        const int block = 4, nb = 8;
        const int ones[nb] = { 0, 1, 1, 2, 2, 2, 4, 4 };          // sum 16 = half of 32 tokens
        std::vector<double> sc(nb * block);
        for (int b = 0; b < nb; ++b) {
            for (int p = 0; p < block; ++p) {
                sc[b * block + p] = (p < ones[b] ? 100.0 : 1.0) + 0.001 * (b * block + p);
            }
        }
        std::vector<int> elig(nb);
        for (int i = 0; i < nb; ++i) elig[i] = i;
        const std::vector<int> r = kvcrush_representatives(sc, block, elig, 4, KVCRUSH_ANCHOR_ZEROS);
        CHECK(r.size() == 4);
        CHECK(r.size() == 4 && r[0] == 0 && r[1] == 2 && r[2] == 4 && r[3] == 6);
        CHECK(kvcrush_representatives(sc, block, elig, 0, KVCRUSH_ANCHOR_ZEROS).empty());
        CHECK(kvcrush_representatives(sc, block, std::vector<int>(), 4, KVCRUSH_ANCHOR_ZEROS).empty());
        CHECK(kvcrush_representatives(sc, block, elig, 99, KVCRUSH_ANCHOR_ZEROS).size() == (size_t) nb);
        CHECK(kvcrush_representatives(sc, 5, elig, 4, KVCRUSH_ANCHOR_ZEROS).empty());   // 32 tokens are not a multiple of 5
        // anchor ONES reverses the distance order: block 7 (4 ones) is closest -> the first sorted entry; budget 1 picks index 0 of the sorted list
        const std::vector<int> r1 = kvcrush_representatives(sc, block, elig, 1, KVCRUSH_ANCHOR_ONES);
        CHECK(r1.size() == 1 && r1[0] == 6);                      // distance 0 for blocks 6 and 7 (all-ones pattern), stable order -> block 6
        // only the eligible blocks compete
        const std::vector<int> r2 = kvcrush_representatives(sc, block, std::vector<int>{ 3, 6 }, 1, KVCRUSH_ANCHOR_ZEROS);
        CHECK(r2.size() == 1 && r2[0] == 3);                      // distance 2 < 4
        // RANDOM anchor is reproducible for one seed
        CHECK(kvcrush_representatives(sc, block, elig, 3, KVCRUSH_ANCHOR_RANDOM, 7) == kvcrush_representatives(sc, block, elig, 3, KVCRUSH_ANCHOR_RANDOM, 7));
    }

    if (n_fail) {
        fprintf(stderr, "test-kv-diversity: %d FAILED\n", n_fail);
        return 1;
    }
    printf("test-kv-diversity: all passed\n");
    return 0;
}
