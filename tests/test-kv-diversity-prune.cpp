// test-kv-diversity-prune.cpp -- the SnapKV/H2O prompt prune with --kv-diversity-mass (diversity_keep_plan) vs the mass-only plan.
// Row: S69-openvino-PICK-KVCRUSH-DIVERSITY-RETENTION-2026-09-28-01.
//
// Scenario (the row's fail-before case): three near-duplicate blocks A1..A3 all carry high attention mass, one UNIQUE block U carries less,
// three filler blocks F1..F3 carry little.  The budget leaves room for two evictable blocks.  The mass-only plan (H2O evict_plan, the plan the
// prune uses today) keeps A1 and A2 -- two copies of the same keys -- and drops U.  The diversity plan keeps A1 (mass) and U (least redundant).
#include "../src/llama-kv-diversity.h"
#include "../src/llama-kv-snapkv.h"

#include <cstdio>
#include <set>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++g_fail; } else { std::printf("ok: " __VA_ARGS__); std::printf("\n"); } } while (0)

static bool has_block(const std::vector<int> & keep, int first, int block) {
    std::set<int> s(keep.begin(), keep.end());
    for (int p = first; p < first + block; ++p) {
        if (!s.count(p)) {
            return false;
        }
    }
    return true;
}

int main() {
    const int block = 4, sink = 4, recent = 4, hd = 64, n_head_kv = 2;
    // layout: [sink 0..3] A1 4..7  A2 8..11  A3 12..15  U 16..19  F1 20..23  F2 24..27  F3 28..31  [recent 32..35]
    const int A1 = 4, A2 = 8, A3 = 12, U = 16;
    const int n = 36;
    std::vector<float> mass(n, 0.1f);
    for (int p = 0; p < sink; ++p) mass[p] = 5.0f;
    for (int j = 0; j < block; ++j) { mass[A1 + j] = 1.02f; mass[A2 + j] = 1.01f; mass[A3 + j] = 1.00f; mass[U + j] = 0.6f; }
    for (int p = n - recent; p < n; ++p) mass[p] = 0.5f;
    const int budget = sink + recent + 2 * block;   // room for two evictable blocks

    // keys: one-hot directions; the three A blocks share theirs (near-duplicates), every other token has its own
    std::vector<float> K((size_t) n * n_head_kv * hd, 0.0f);
    for (int t = 0; t < n; ++t) {
        int dir = t;                                  // unique direction per token (t < 36 < 40)
        if (t >= A1 && t < A3 + block) dir = 40 + (t - A1) % block;
        for (int h = 0; h < n_head_kv; ++h) K[((size_t) t * n_head_kv + h) * hd + dir] = 1.0f;
    }

    // mass-only plan (today): H2O heavy hitters over [0, n - recent) + the recent window
    justician_kv::H2OConfig hc;
    hc.enabled = true; hc.hh_size = budget - recent; hc.recent_size = recent;
    justician_kv::H2OAccumulator acc;
    acc.accumulate(mass);
    const std::vector<int> mass_keep = acc.evict_plan(n, hc);
    CHECK((int) mass_keep.size() == budget, "mass-only plan keeps the budget (%d of %d)", (int) mass_keep.size(), budget);
    CHECK(has_block(mass_keep, A1, block) && has_block(mass_keep, A2, block), "mass-only plan keeps duplicates A1 and A2");
    CHECK(!has_block(mass_keep, U, block), "mass-only plan drops the unique block U (the defect)");

    // diversity plan
    const justician_kv::DiversityPlanGeom g = justician_kv::diversity_plan_geometry(n, recent, block, sink, budget);
    CHECK(g.ok() && g.n_blocks == 7 && g.keep_blocks == 2 && g.area_end == 32 && g.tail_begin == 32,
          "geometry: 7 blocks, keep 2, area_end 32 (got %d %d %d %d)", g.n_blocks, g.keep_blocks, g.area_end, g.tail_begin);
    const std::vector<float> S = justician_kv::kv_token_redundancy(K.data(), g.area_end, n_head_kv, hd, sink);
    justician_kv::AdaptiveRKVResult res;
    const std::vector<int> div_keep = justician_kv::diversity_keep_plan(mass, n, S, g, block, 0.25, &res);
    CHECK((int) div_keep.size() == budget, "diversity plan keeps the budget (%d of %d)", (int) div_keep.size(), budget);
    CHECK(has_block(div_keep, A1, block), "diversity plan keeps A1 (mass stage)");
    CHECK(has_block(div_keep, U, block), "diversity plan keeps the unique block U (FAILS with the mass-only plan)");
    CHECK(!has_block(div_keep, A2, block) && !has_block(div_keep, A3, block), "diversity plan drops the duplicates A2 and A3");
    CHECK(res.n_mass_blocks == 1 && res.n_diverse_blocks == 1, "one mass block + one diverse block (got %d + %d)", res.n_mass_blocks, res.n_diverse_blocks);
    CHECK(has_block(div_keep, 0, sink) && has_block(div_keep, n - recent, recent), "sink and recent window kept");
    bool sorted = true;
    for (size_t i = 1; i < div_keep.size(); ++i) sorted = sorted && div_keep[i - 1] < div_keep[i];
    CHECK(sorted, "keep set sorted and unique (the prune walks it in order)");

    // control: attention_mass = 1.0 spends the whole budget in the mass stage -> same blocks as the mass-only plan
    const std::vector<int> all_mass = justician_kv::diversity_keep_plan(mass, n, S, g, block, 1.0);
    CHECK(has_block(all_mass, A1, block) && has_block(all_mass, A2, block) && !has_block(all_mass, U, block),
          "control: attention_mass 1.0 reproduces the mass-only choice (A1, A2, not U)");

    // remainder: two extra positions between the last whole block and the recent window are kept, not selected
    {
        const int n2 = 38;
        std::vector<float> m2(mass); m2.insert(m2.begin() + 32, 2, 0.05f);
        const justician_kv::DiversityPlanGeom g2 = justician_kv::diversity_plan_geometry(n2, recent, block, sink, budget + 2);
        CHECK(g2.ok() && g2.n_blocks == 7 && g2.area_end == 32 && g2.tail_begin == 34 && g2.keep_blocks == 2,
              "remainder geometry (got %d %d %d %d)", g2.n_blocks, g2.area_end, g2.tail_begin, g2.keep_blocks);
        std::vector<float> K2(K); K2.resize((size_t) n2 * n_head_kv * hd, 0.0f);
        const std::vector<float> S2 = justician_kv::kv_token_redundancy(K2.data(), g2.area_end, n_head_kv, hd, sink);
        const std::vector<int> k2 = justician_kv::diversity_keep_plan(m2, n2, S2, g2, block, 0.25);
        CHECK((int) k2.size() == budget + 2 && has_block(k2, 32, 2) && has_block(k2, U, block), "remainder kept, U still kept (size %d)", (int) k2.size());
    }

    // fallback cases: the caller keeps the mass-only plan when the geometry is not ok()
    CHECK(!justician_kv::diversity_plan_geometry(n, recent, block, sink, n).ok(), "budget covering every block -> not ok (no prune choice)");
    CHECK(!justician_kv::diversity_plan_geometry(n, recent, block, sink, sink + recent + 3).ok(), "budget below one block -> not ok");
    CHECK(!justician_kv::diversity_plan_geometry(8, recent, block, sink, 6).ok(), "sink + recent cover everything -> not ok");
    CHECK(justician_kv::diversity_keep_plan(mass, n, std::vector<float>(3, 0.0f), g, block, 0.25).empty(), "wrong-size S -> empty plan");

    std::printf("%s (%d failure(s))\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
