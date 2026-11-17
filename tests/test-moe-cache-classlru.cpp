// MoE cache: ClassLRU policy and the CPU share (upstream llama.cpp c4fb57969 #31233, e95426f16 #30966), no model, no GPU.
//
//   A. ClassLRU plan(): hits and misses, the selected experts are never evicted, uncached = n_slots or later, a failed plan (more distinct
//      experts than slots) reports false.
//   B. CPU share: misses left to the CPU get a zero slot (n_slots..n_slots+n_zero-1), are not cached, take the lowest CRF first, and a zero
//      slot is used at most once per token.
//   C. MUL_MAT_ID on the CPU backend skips negative ids and writes zero rows, and  device part (zero slot for the CPU's experts) +
//      CPU part (-1 for the device's experts)  equals the plain mul_mat_id. Both directions.
//   D. Defaults: llama_context_default_params() keeps the SLRU policy and no CPU share, so the fork's behaviour without the flags is unchanged.

#include "llama-moe-cache-lru.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using llama_moe_cache_detail::moe_cache_lru;

static int g_fail = 0;

static void check(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) {
        g_fail++;
    }
}

static auto no_cpu = [](size_t) { return (size_t) 0; };

static void part_a() {
    printf("A. ClassLRU\n");
    const int n_expert = 16, n_slots = 6, n_used = 2;
    moe_cache_lru lru;
    lru.init(2, n_expert, n_slots);
    std::vector<int32_t> map0(n_expert, lru.empty), map1(n_expert, lru.empty);
    lru.slot_map[0] = map0.data();
    lru.slot_map[1] = map1.data();
    std::vector<moe_cache_lru::fill> fills, cpu;
    size_t hit = 0;

    int32_t ids1[] = { 3, 5 };
    check(lru.plan(0, ids1, 2, 1, n_used, 0, no_cpu, fills, cpu, hit) && hit == 0 && fills.size() == 2 && cpu.empty(), "cold start: 2 misses, 2 fills");
    check(map0[3] < n_slots && map0[5] < n_slots && map0[3] != map0[5] && map0[0] == n_slots, "slots assigned, uncached experts read n_slots");
    check(lru.plan(0, ids1, 2, 1, n_used, 0, no_cpu, fills, cpu, hit) && hit == 2 && fills.empty(), "repeat: 2 hits, nothing to upload");

    // fill the cache with layer 1, then ask layer 0 for experts that force evictions: the selected experts stay
    for (int e = 0; e + 1 < n_expert; e += 2) {
        int32_t ids[] = { e, e + 1 };
        check(lru.plan(1, ids, 2, 1, n_used, 0, no_cpu, fills, cpu, hit), "layer 1 plan succeeds");
    }
    int32_t ids2[] = { 9, 11 };
    check(lru.plan(0, ids2, 2, 1, n_used, 0, no_cpu, fills, cpu, hit) && fills.size() == 2, "layer 0, two new experts");
    check(map0[9] < n_slots && map0[11] < n_slots && map0[9] != map0[11], "the selected experts hold slots");
    int cached = 0;
    for (int e = 0; e < n_expert; ++e) cached += map0[e] < n_slots;
    for (int e = 0; e < n_expert; ++e) cached += map1[e] < n_slots;
    check(cached == n_slots, "the cache holds exactly n_slots experts, evictions emptied the others");
    check(lru.n_filled == n_slots, "n_filled counts the filled slots");

    int32_t big[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    check(!lru.plan(0, big, 8, 4, 2, 0, no_cpu, fills, cpu, hit), "more distinct experts than slots: plan() reports false");
}

static void part_b() {
    printf("B. CPU share\n");
    const int n_expert = 32, n_slots = 8, n_used = 4, n_zero = 4;
    moe_cache_lru lru;
    lru.init(1, n_expert, n_slots);
    std::vector<int32_t> map(n_expert, lru.empty);
    lru.slot_map[0] = map.data();
    std::vector<moe_cache_lru::fill> fills, cpu;
    size_t hit = 0;

    // the cache is empty, a share of 100% of the misses still goes through plan(): the caller (llama_moe_cache::copy) passes 0 until n_filled == n_slots
    int32_t a[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    check(lru.plan(0, a, 8, 2, n_used, n_zero, no_cpu, fills, cpu, hit) && fills.size() == 8 && lru.n_filled == n_slots, "fill the cache first: 8 uploads");

    // two tokens with distinct experts, all misses, half to the CPU
    int32_t b[] = { 10, 11, 12, 13, 20, 21, 22, 23 };
    auto half = [](size_t n) { return n/2; };
    check(lru.plan(0, b, 8, 2, n_used, n_zero, half, fills, cpu, hit) && cpu.size() == 4 && fills.size() == 4, "8 misses, share 1/2: 4 on the CPU, 4 uploaded");
    bool zero_ok = true;
    for (const auto & f : cpu) zero_ok &= f.slot >= n_slots && f.slot < n_slots + n_zero && map[f.expert] == f.slot;
    check(zero_ok, "the CPU's experts read zero slots");
    bool not_cached = true;
    for (const auto & f : fills) not_cached &= f.slot < n_slots;
    check(not_cached, "the uploaded experts hold real slots");

    // one token selecting 4 distinct misses, all left to the CPU: each takes a distinct zero slot
    moe_cache_lru l2;
    l2.init(1, n_expert, n_slots);
    std::vector<int32_t> m2(n_expert, l2.empty);
    l2.slot_map[0] = m2.data();
    l2.plan(0, a, 8, 2, n_used, n_zero, no_cpu, fills, cpu, hit);
    int32_t c[] = { 10, 11, 12, 13 };
    auto all = [](size_t n) { return n; };
    check(l2.plan(0, c, 4, 1, n_used, n_zero, all, fills, cpu, hit) && cpu.size() == 4 && fills.empty(), "4 misses of one token, all on the CPU");
    bool distinct = true;
    for (size_t i = 0; i < cpu.size(); ++i) for (size_t j = i + 1; j < cpu.size(); ++j) distinct &= cpu[i].slot != cpu[j].slot;
    check(distinct, "one token uses distinct zero slots (mul_mat_id does not support an expert selected twice by a token)");

    // an expert the CPU computed is a miss the next time, it was not cached
    check(l2.plan(0, c, 4, 1, n_used, n_zero, no_cpu, fills, cpu, hit) && hit == 0 && fills.size() == 4, "the CPU's experts are not cached: misses again");
    // more misses than zero slots in one token: the rest is uploaded
    moe_cache_lru l3;
    l3.init(1, n_expert, n_slots);
    std::vector<int32_t> m3(n_expert, l3.empty);
    l3.slot_map[0] = m3.data();
    l3.plan(0, a, 8, 2, n_used, n_zero, no_cpu, fills, cpu, hit);
    check(l3.plan(0, c, 4, 1, n_used, 2, all, fills, cpu, hit) && cpu.size() == 2 && fills.size() == 2, "n_zero = 2: two to the CPU, the rest uploaded");
}

static float run_mm_id(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids, std::vector<float> & out) {
    ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 2);
    out.resize(ggml_nelements(y));
    memcpy(out.data(), y->data, ggml_nbytes(y));
    return 0.0f;
}

static void part_c() {
    printf("C. MUL_MAT_ID on the CPU backend\n");
    const int k = 32, m = 8, n_as = 6, n_used = 3, n_tok = 2;
    ggml_init_params ip = { 16*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_as + 1); // the last expert is the zero slot
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n_tok);
    for (int64_t i = 0; i < ggml_nelements(w); ++i) ((float *) w->data)[i] = (i / (k*m) == n_as) ? 0.0f : 0.01f*(float) ((i*7919) % 101 - 50);
    for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float *) x->data)[i] = 0.1f*(float) ((i*31) % 17 - 8);
    ggml_tensor * ids_all = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * ids_dev = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok); // the CPU's experts read the zero expert
    ggml_tensor * ids_cpu = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok); // the device's experts are skipped: -1
    const int32_t all[n_tok][n_used] = { { 0, 2, 4 }, { 1, 2, 5 } };
    const bool on_cpu[n_tok][n_used] = { { false, true, false }, { true, false, true } };
    for (int t = 0; t < n_tok; ++t) {
        for (int u = 0; u < n_used; ++u) {
            ((int32_t *) ids_all->data)[t*n_used + u] = all[t][u];
            ((int32_t *) ids_dev->data)[t*n_used + u] = on_cpu[t][u] ? n_as : all[t][u];
            ((int32_t *) ids_cpu->data)[t*n_used + u] = on_cpu[t][u] ? all[t][u] : -1;
        }
    }
    std::vector<float> ref, dev, cpu;
    run_mm_id(ctx, w, x, ids_all, ref);
    run_mm_id(ctx, w, x, ids_dev, dev);
    run_mm_id(ctx, w, x, ids_cpu, cpu);
    bool zero_rows = true, sum_ok = true;
    for (int t = 0; t < n_tok; ++t) {
        for (int u = 0; u < n_used; ++u) {
            for (int r = 0; r < m; ++r) {
                const size_t i = ((size_t) t*n_used + u)*m + r;
                if (!on_cpu[t][u]) zero_rows &= cpu[i] == 0.0f;   // negative id: zero row
                else               zero_rows &= dev[i] == 0.0f;   // zero slot: zero row
                sum_ok &= std::fabs(dev[i] + cpu[i] - ref[i]) <= 1e-5f*(1.0f + std::fabs(ref[i]));
            }
        }
    }
    check(zero_rows, "negative id and zero slot both give zero rows");
    check(sum_ok, "device part + CPU part == plain mul_mat_id (both directions)");
    ggml_free(ctx);
}

static void part_d() {
    printf("D. defaults\n");
    const llama_context_params p = llama_context_default_params();
    check(p.moe_cache_policy == LLAMA_MOE_CACHE_POLICY_SLRU, "default policy is SLRU");
    check(p.moe_cache_cpu_share == 0.0f && p.moe_cache_size == 0, "default CPU share is 0, cache off");
}

int main() {
    part_a();
    part_b();
    part_c();
    part_d();
    printf(g_fail ? "FAILED: %d\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
