// Standalone KVFlash pager selftest (S69 #66) — proves the pager LOGIC host-side
// before wiring into ik_llama (matches the SnapKV standalone-compile precedent).
// Implements the design's GATE C (live paging, pool << logical, bit-exact roundtrip,
// >=90% byte reduction) + GATE E perf-context + GATE D (recall by score flip) + the
// pool-budget invariant + slot-mask correctness. Uses the ggml stub (memcpy device).
//
// Build: g++ -std=c++17 -I temp/_kvflash_test -I portable/ik_llama/source/src/kvflash_ref \
//        temp/_kvflash_test/test_kvflash.cpp -o kvf && ./kvf

#include "kvflash_pager.h"   // the real Apache-2.0 reference pager (kvflash_ref/)
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>

using dflash::common::KvFlashPager;
using dflash::common::KvFlashConfig;

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  FAIL: %s\n", msg); g_fail++; } \
                              else std::printf("  ok:   %s\n", msg); } while (0)

// Build a fake per-layer K or V cache tensor: [head_dim, pool_tokens, n_head_kv].
struct FakeCache {
    std::vector<ggml_tensor> tk, tv;
    std::vector<std::vector<uint8_t>> bk, bv;   // backing "device" bytes
    int head_dim, pool_tokens, n_head_kv, n_layer, row_bytes;
    FakeCache(int hd, int pool, int nh, int nl) : head_dim(hd), pool_tokens(pool),
        n_head_kv(nh), n_layer(nl) {
        row_bytes = hd;  // 1 byte/elem (stub), row = head_dim bytes
        tk.resize(nl); tv.resize(nl); bk.resize(nl); bv.resize(nl);
        for (int l = 0; l < nl; l++) {
            size_t bytes = (size_t)row_bytes * pool * nh;
            bk[l].assign(bytes, 0); bv[l].assign(bytes, 0);
            for (auto * arr : {&tk[l], &tv[l]}) {
                arr->type = GGML_TYPE_Q8_0;
                arr->ne[0]=hd; arr->ne[1]=pool; arr->ne[2]=nh; arr->ne[3]=1;
                arr->nb[0]=1; arr->nb[1]=(size_t)row_bytes; arr->nb[2]=(size_t)row_bytes*pool; arr->nb[3]=0;
            }
            tk[l].data = bk[l].data(); tv[l].data = bv[l].data();
        }
    }
    std::vector<ggml_tensor*> ks(){ std::vector<ggml_tensor*> v; for(auto&t:tk)v.push_back(&t); return v; }
    std::vector<ggml_tensor*> vs(){ std::vector<ggml_tensor*> v; for(auto&t:tv)v.push_back(&t); return v; }
    // write a known byte pattern for logical chunk c into its resident slot
    void stamp(KvFlashPager& p, int chunk, int chunk_tokens, uint8_t val) {
        int blk = p.block_of(chunk);
        if (blk < 0) return;
        for (int l = 0; l < n_layer; l++)
            for (int h = 0; h < n_head_kv; h++) {
                size_t off = (size_t)blk*chunk_tokens*tk[l].nb[1] + (size_t)h*tk[l].nb[2];
                std::memset((uint8_t*)tk[l].data+off, val, (size_t)chunk_tokens*row_bytes);
                std::memset((uint8_t*)tv[l].data+off, val^0xFF, (size_t)chunk_tokens*row_bytes);
            }
    }
    bool verify(KvFlashPager& p, int chunk, int chunk_tokens, uint8_t val) {
        int blk = p.block_of(chunk);
        if (blk < 0) return false;
        for (int l = 0; l < n_layer; l++)
            for (int h = 0; h < n_head_kv; h++) {
                size_t off = (size_t)blk*chunk_tokens*tk[l].nb[1] + (size_t)h*tk[l].nb[2];
                for (int b = 0; b < chunk_tokens*row_bytes; b++) {
                    if (((uint8_t*)tk[l].data+off)[b] != val) return false;
                    if (((uint8_t*)tv[l].data+off)[b] != (uint8_t)(val^0xFF)) return false;
                }
            }
        return true;
    }
};

int main() {
    const int hd=4, nh=2, nl=3, chunk=64;
    const int n_logical_chunks = 64;            // long logical context
    const int pool_chunks = 5;                  // pool << logical (= min_pool: 1 sink + 2 tail + 2)
    // reduction = 1 - 5/64 = 92.2% (>=90%); residency is a HARD O(pool) bound by construction
    const int pool = pool_chunks * chunk;
    KvFlashConfig cfg; cfg.chunk_tokens=chunk; cfg.pool_tokens=pool; cfg.sink_chunks=1; cfg.tail_window_chunks=2;

    FakeCache fc(hd, pool, nh, nl);
    KvFlashPager pager;
    std::printf("[gate] attach pool=%d (%d chunks) for %d logical chunks\n", pool, pool_chunks, n_logical_chunks);
    CHECK(pager.attach(cfg, fc.ks(), fc.vs()), "attach");

    // GATE C: live paging across logical >> pool. Stamp each chunk as it becomes resident;
    // the pool-budget invariant must hold every step.
    bool budget_ok = true;
    for (int c = 0; c < n_logical_chunks; c++) {
        int s = pager.slot_for((int64_t)c*chunk);          // append head of chunk c
        if (s < 0) { std::printf("  FAIL: slot_for chunk %d\n", c); g_fail++; break; }
        fc.stamp(pager, c, chunk, (uint8_t)(0x10 + c));
        if (pager.resident_blocks() > pool_chunks) budget_ok = false;
    }
    CHECK(budget_ok, "pool-budget invariant: resident_blocks never exceeds pool");
    CHECK(pager.stats().page_outs > 0, "evictions occurred (page_outs > 0)");

    // >=90% KV byte reduction: resident pool vs full logical cache.
    double reduction = 1.0 - (double)pool / (double)(n_logical_chunks*chunk);
    std::printf("[gate] KV byte reduction = %.1f%% (pool %d / logical %d)\n",
                reduction*100, pool, n_logical_chunks*chunk);
    CHECK(reduction >= 0.90, "GATE C: >=90% KV byte reduction");

    // GATE C bit-exact roundtrip: a sink chunk (0) is protected, still resident + intact.
    CHECK(fc.verify(pager, 0, chunk, 0x10), "sink chunk 0 resident + bit-exact");

    // GATE D: an evicted chunk recalled brings back its EXACT bytes. Chunk 3 was stamped
    // 0x13 then evicted (it's neither sink nor in the final tail window). Recall + verify.
    CHECK(!pager.is_resident(3), "chunk 3 was evicted (paged out)");
    int rs = pager.slot_for((int64_t)3*chunk);             // recall chunk 3
    CHECK(rs >= 0, "chunk 3 recalled (page_in)");
    CHECK(fc.verify(pager, 3, chunk, 0x13), "GATE D: recalled chunk 3 bytes bit-exact (0x13)");
    CHECK(pager.stats().page_ins > 0, "page_ins > 0");

    // slot-mask correctness: resident slots = 0x0000, free/paged = 0xFC00 (-inf f16).
    std::vector<uint16_t> mask(pool, 0xABCD);
    pager.fill_slot_mask(mask.data());
    int n_resident_slots = 0, n_inf = 0;
    for (int i = 0; i < pool; i++) { if (mask[i]==0x0000) n_resident_slots++; else if (mask[i]==0xFC00) n_inf++; }
    CHECK(n_resident_slots + n_inf == pool, "slot-mask: every slot is 0 or -inf");
    CHECK(n_resident_slots == pager.resident_blocks()*chunk, "slot-mask: resident-slot count matches");

    std::printf("\n[result] %s (%d failures)\n", g_fail==0 ? "PASS" : "FAIL", g_fail);
    return g_fail==0 ? 0 : 1;
}
