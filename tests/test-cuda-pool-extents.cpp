// Unit test for ggml/src/ggml-cuda/pool-extents.h (the free-order-agnostic bookkeeping of the CUDA VMM pool).
// CPU only: the header has no CUDA types.
//
// FAIL-BEFORE partner: StrictLifoModel below reproduces the old ggml_cuda_pool_vmm contract (free()
// asserts ptr == base + used). It REJECTS the non-LIFO sequence that ggml_cuda_pool_extents accepts.
// NEGATIVE controls: double free, free outside the range, and a mutation of the invariant checker.

#include "../ggml/src/ggml-cuda/pool-extents.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// the pre-change pool: bump allocator, free() valid only for the most recent live block
struct StrictLifoModel {
    size_t used = 0;
    size_t alloc(size_t size) { size_t o = used; used += size; return o; }
    bool   free(size_t off, size_t size) { used -= size; return off == used; } // the old GGML_ASSERT condition
};

struct Block { size_t off, size; };

// I1/I2/I3 + exact tiling of [0, top) by live and free bytes; returns false on any violation
static bool invariants_hold(const ggml_cuda_pool_extents & e, const std::vector<Block> & live, bool skip_i2 = false) {
    size_t prev_end = 0;
    bool   first    = true;
    for (const auto & f : e.free_map) {
        if (f.second == 0)                       return false;
        if (f.first + f.second > e.top)          return false;           // I3
        if (!first && f.first <= prev_end)       return false;           // I1: overlap or touching
        if (!skip_i2 && f.first + f.second == e.top) return false;      // I2
        prev_end = f.first + f.second;
        first    = false;
    }
    // live blocks must be disjoint, inside [0, top), and disjoint from every free extent
    std::vector<char> cover(e.top, 0);
    for (const auto & b : live) {
        if (b.off + b.size > e.top) return false;
        for (size_t i = 0; i < b.size; ++i) { if (cover[b.off + i]) return false; cover[b.off + i] = 1; }
    }
    for (const auto & f : e.free_map) {
        for (size_t i = 0; i < f.second; ++i) { if (cover[f.first + i]) return false; cover[f.first + i] = 2; }
    }
    for (size_t i = 0; i < e.top; ++i) { if (!cover[i]) return false; } // no byte lost
    return true;
}

static void test_lifo_identical_to_bump() {
    ggml_cuda_pool_extents e;
    StrictLifoModel        m;
    const size_t cap = 1 << 20;
    std::vector<Block> stack;
    std::mt19937 rng(7);
    for (int i = 0; i < 20000; ++i) {
        if (stack.empty() || (rng() % 3 != 0 && stack.size() < 64)) {
            const size_t sz = 128 * (1 + rng() % 64);
            size_t off = 0;
            const bool ok = e.alloc(sz, cap, &off);
            CHECK(ok);
            CHECK(off == m.alloc(sz));
            stack.push_back({off, sz});
        } else {
            Block b = stack.back(); stack.pop_back();
            CHECK(e.release(b.off, b.size));
            CHECK(m.free(b.off, b.size));
            CHECK(e.free_map.empty());           // pure LIFO never creates a free extent
            CHECK(e.top == m.used);
        }
    }
}

static void test_non_lifo_sequence() {
    // alloc A, alloc B, free A while B is live, alloc C
    ggml_cuda_pool_extents e;
    StrictLifoModel        m;
    const size_t cap = 1 << 20;
    size_t a = 0, b = 0, c = 0;
    CHECK(e.alloc(1024, cap, &a)); CHECK(e.alloc(2048, cap, &b));
    size_t ma = m.alloc(1024); (void) m.alloc(2048);
    CHECK(a == 0 && b == 1024);
    // FAIL-BEFORE: the old strict-LIFO pool rejects this free
    CHECK(!m.free(ma, 1024));
    // PASS-AFTER: the new bookkeeping accepts it and reuses the hole without growing
    CHECK(e.release(a, 1024));
    const size_t top_before = e.top;
    CHECK(e.alloc(512, cap, &c));
    CHECK(c == 0);                    // C lands inside A's freed extent
    CHECK(e.top == top_before);       // no growth
    CHECK(e.free_map.size() == 1 && e.free_map.begin()->first == 512 && e.free_map.begin()->second == 512);
    CHECK(e.used() == 512 + 2048);
}

static void test_coalescing_and_tail_return() {
    ggml_cuda_pool_extents e;
    const size_t cap = 1 << 20;
    size_t a, b, c, d;
    CHECK(e.alloc(256, cap, &a)); CHECK(e.alloc(256, cap, &b)); CHECK(e.alloc(256, cap, &c)); CHECK(e.alloc(256, cap, &d));
    CHECK(e.release(a, 256));
    CHECK(e.release(c, 256));
    CHECK(e.free_map.size() == 2);
    CHECK(e.release(b, 256));         // bridges a and c -> one extent [0, 768)
    CHECK(e.free_map.size() == 1);
    CHECK(e.free_map.begin()->first == 0 && e.free_map.begin()->second == 768);
    CHECK(e.top == 1024);
    CHECK(e.release(d, 256));         // the tail block returns the whole range to the bump region
    CHECK(e.free_map.empty());
    CHECK(e.top == 0);
    CHECK(e.used() == 0);
}

static void test_best_fit() {
    ggml_cuda_pool_extents e;
    const size_t cap = 1 << 20;
    size_t o[8];
    for (int i = 0; i < 7; ++i) CHECK(e.alloc(i % 2 ? 4096 : 1024, cap, &o[i])); // 1024,4096,1024,4096,1024,4096,1024
    CHECK(e.release(o[1], 4096));     // hole 4096 @ 1024
    CHECK(e.release(o[3], 4096));     // hole 4096 @ 6144
    CHECK(e.release(o[4], 1024));     // merges with o[3] -> 5120 @ 6144
    size_t got = 0;
    CHECK(e.alloc(3000, cap, &got));
    CHECK(got == o[1]);               // the 4096 hole (smaller leftover) beats the 5120 hole
    size_t got2 = 0;
    CHECK(e.alloc(5000, cap, &got2));
    CHECK(got2 == o[3]);              // only the 5120 hole fits
}

static void test_errors_and_growth() {
    ggml_cuda_pool_extents e;
    size_t a, b;
    CHECK(e.alloc(1024, 2048, &a));
    CHECK(e.alloc(1024, 2048, &b));
    size_t x = 0;
    CHECK(!e.alloc(1, 2048, &x));                 // full, no hole
    CHECK(e.shortfall(1, 2048) == 1);
    CHECK(e.shortfall(4096, 2048) == 4096);
    CHECK(e.release(a, 1024));
    CHECK(!e.release(a, 1024));                   // double free is refused ...
    CHECK(!e.release(a + 100, 200));              // ... and so is a partial overlap of a free extent
    CHECK(!e.release(1 << 20, 16));               // outside the range
    CHECK(!e.release(2000, 1000));                // runs past top
    CHECK(e.free_map.size() == 1);                // refused calls changed nothing
    CHECK(e.release(0, 0));                       // zero-size free of a zero-size alloc is a no-op
    CHECK(e.alloc(1024, 2048, &x) && x == a);     // the hole is reused instead of reporting full
}

static void test_random_differential() {
    for (unsigned seed = 1; seed <= 20; ++seed) {
        std::mt19937 rng(seed);
        ggml_cuda_pool_extents e;
        std::vector<Block> live;
        const size_t cap = 1 << 14;
        for (int step = 0; step < 4000; ++step) {
            if (live.empty() || rng() % 2) {
                const size_t sz = 128 * (1 + rng() % 8);
                size_t off = 0;
                if (e.alloc(sz, cap, &off)) live.push_back({off, sz});
            } else {
                const size_t k = rng() % live.size();
                CHECK(e.release(live[k].off, live[k].size));
                live[k] = live.back();
                live.pop_back();
            }
            if (!invariants_hold(e, live)) { std::printf("invariants broke at seed %u step %d\n", seed, step); ++g_fail; return; }
        }
        size_t sum = 0;
        for (const auto & b : live) sum += b.size;
        CHECK(e.used() == sum);
        while (!live.empty()) { CHECK(e.release(live.back().off, live.back().size)); live.pop_back(); }
        CHECK(e.top == 0 && e.free_map.empty());  // everything returned in an arbitrary order collapses to empty
    }
}

static void test_checker_catches_corruption() {
    // NEGATIVE control: the invariant checker must flag a hand-corrupted state, otherwise the
    // differential test above proves nothing.
    ggml_cuda_pool_extents e;
    size_t a, b, c;
    CHECK(e.alloc(256, 4096, &a)); CHECK(e.alloc(256, 4096, &b)); CHECK(e.alloc(256, 4096, &c));
    std::vector<Block> live = {{a, 256}, {c, 256}};
    CHECK(e.release(b, 256));
    CHECK(invariants_hold(e, live));                      // healthy state passes
    e.free_map[b + 8] = 8;                                // overlap with the real free extent
    CHECK(!invariants_hold(e, live));
    e.free_map.erase(b + 8);
    e.free_map.erase(b);
    CHECK(!invariants_hold(e, live));                     // a lost hole (bytes in no set) is flagged
    e.free_map[b] = 256;
    e.free_map[c + 256] = 0;
    CHECK(!invariants_hold(e, live));                     // zero-size extent flagged
}

int main() {
    test_lifo_identical_to_bump();
    test_non_lifo_sequence();
    test_coalescing_and_tail_return();
    test_best_fit();
    test_errors_and_growth();
    test_random_differential();
    test_checker_catches_corruption();
    if (g_fail) { std::printf("test-cuda-pool-extents: %d FAILURES\n", g_fail); return 1; }
    std::printf("test-cuda-pool-extents: PASS\n");
    return 0;
}
