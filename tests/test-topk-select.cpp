// test-topk-select.cpp -- runs the DSA indexer top-k SELECT kernel body (ggml/src/ggml-cuda/topk-select.cuh,
// topk_select_row) on the CPU, one std::thread per CUDA thread with a real barrier, against a stable-sort reference.
//
// Rows S69-EXLLAMAV3-PICK-DSATOPKRADIX-2026-09-28-01 and S69-GGML-PICK-INDEXER-TOPK-PARTIAL-SELECT-2026-09-29-01.
//
// What this proves: the algorithm (4-pass radix select, index-ordered tie compaction, bitonic sort) is correct for
// every (n, k, thread count) tried, including ties, -inf, NaN, +-0, sorted and constant rows. What it does NOT prove:
// the GPU launch, shared-memory size limits and cub::BlockScan wiring -- those are the parent's GPU run (row text).
//
// Controls: (a) a reference with the WRONG tie rule (highest index first) must disagree on tie-heavy data, so the
// comparison is sensitive to the property the kernel exists to guarantee; (b) a corrupted output must fail the check.

#include "../ggml/src/ggml-cuda/topk-select.cuh"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cstdio>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// ---- CPU executor of the block concept ------------------------------------------------------------------------------

struct emu_barrier {
    std::mutex m; std::condition_variable cv; int n, waiting = 0; unsigned gen = 0;
    explicit emu_barrier(int n_) : n(n_) {}
    void wait() {
        std::unique_lock<std::mutex> l(m);
        const unsigned g = gen;
        if (++waiting == n) { waiting = 0; ++gen; cv.notify_all(); }
        else cv.wait(l, [&] { return g != gen; });
    }
};

struct emu_shared_state {
    emu_barrier bar;
    std::vector<uint32_t> scratch;
    explicit emu_shared_state(int nt) : bar(nt), scratch(nt, 0) {}
};

struct emu_block {
    emu_shared_state & st; int t, nthreads;
    int tid() const { return t; }
    int nt() const { return nthreads; }
    void sync() { st.bar.wait(); }
    uint32_t atomic_add(uint32_t * p, uint32_t v) {
#ifdef _MSC_VER
        return (uint32_t) _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(p), (long) v); // full barrier on x86/x64: stronger
#else
        return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
#endif
    }
    uint32_t excl_scan(uint32_t v, uint32_t * total) {
        st.scratch[t] = v;
        sync();
        uint32_t ex = 0, tot = 0;
        for (int i = 0; i < nthreads; ++i) { if (i < t) ex += st.scratch[i]; tot += st.scratch[i]; }
        *total = tot;
        sync(); // scratch is reused by the next call
        return ex;
    }
};

static std::vector<int> run_kernel(const std::vector<float> & row, int k, int nthreads) {
    const int n = (int) row.size();
    const int k_pad = topk_pad_pow2(k);
    std::vector<int> out(k, -12345);
    topk_shared * sh = new topk_shared();
    emu_shared_state st(nthreads);
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) {
        th.emplace_back([&, t] {
            emu_block blk{st, t, nthreads};
            topk_select_row(blk, *sh, row.data(), n, k, out.data(), k_pad);
        });
    }
    for (auto & x : th) x.join();
    delete sh;
    return out;
}

// reference: stable descending order by value, NaN below everything, -0 == +0
static bool ref_before(const std::vector<float> & r, int a, int b, bool tie_low_first) {
    const float x = r[a], y = r[b];
    const bool xn = std::isnan(x), yn = std::isnan(y);
    if (xn != yn) return yn;                       // non-NaN before NaN
    if (!xn && x != y) return x > y;
    return tie_low_first ? a < b : a > b;
}

static std::vector<int> reference(const std::vector<float> & r, int k, bool tie_low_first = true) {
    std::vector<int> idx(r.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return ref_before(r, a, b, tie_low_first); });
    idx.resize(k);
    return idx;
}

static void check_case(const char * what, const std::vector<float> & row, int k, int nthreads) {
    const auto got = run_kernel(row, k, nthreads);
    const auto want = reference(row, k);
    if (got != want) {
        std::fprintf(stderr, "FAIL case '%s' n=%zu k=%d nt=%d\n", what, row.size(), k, nthreads);
        for (int i = 0; i < k && i < 12; ++i) std::fprintf(stderr, "   [%d] got %d want %d\n", i, got[i], want[i]);
        ++g_fail;
    }
}

static void test_edge_cases() {
    CHECK(topk_select_supported(10, 10) && topk_select_supported(10, 1));
    CHECK(!topk_select_supported(10, 11) && !topk_select_supported(10, 0) && !topk_select_supported(0, 0));
    CHECK(!topk_select_supported(100000, TOPK_SELECT_MAX_K + 1) && topk_select_supported(100000, TOPK_SELECT_MAX_K));
    CHECK(topk_pad_pow2(1) == 1 && topk_pad_pow2(2) == 2 && topk_pad_pow2(3) == 4 && topk_pad_pow2(2048) == 2048 && topk_pad_pow2(2049) == 4096);
    // ordered key is monotone in the float order
    const float ordered[] = {-INFINITY, -1e30f, -1.0f, -1e-30f, 0.0f, 1e-30f, 1.0f, 1e30f, INFINITY};
    for (size_t i = 1; i < sizeof(ordered) / sizeof(float); ++i) CHECK(topk_ordered_key(ordered[i - 1]) < topk_ordered_key(ordered[i]));
    CHECK(topk_ordered_key(-0.0f) == topk_ordered_key(0.0f));
    CHECK(topk_ordered_key(std::numeric_limits<float>::quiet_NaN()) < topk_ordered_key(-INFINITY));

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-5.f, 5.f);
    for (int nt : {32, 64, 96}) {
        for (auto nk : std::vector<std::pair<int, int>>{{1, 1}, {2, 1}, {2, 2}, {7, 3}, {33, 32}, {64, 64}, {65, 1}, {300, 17}, {1000, 1000}, {1500, 1024}, {1025, 513}}) {
            std::vector<float> row(nk.first);
            for (auto & x : row) x = u(rng);
            check_case("random", row, nk.second, nt);
        }
    }
}

static void test_ties_and_specials() {
    std::mt19937 rng(9);
    for (int nt : {32, 128}) {
        // heavy ties: only 5 distinct values
        std::vector<float> row(3000);
        for (auto & x : row) x = (float) (rng() % 5);
        for (int k : {1, 100, 599, 600, 601, 1500, 2999, 3000}) check_case("ties5", row, k, nt);
        // all equal -> the first k indices in order
        std::vector<float> eq(1234, 0.25f);
        for (int k : {1, 7, 1000, 1234}) check_case("all-equal", eq, k, nt);
        // -inf runs (masked positions) with a few live values
        std::vector<float> m(2000, -INFINITY);
        for (int i = 0; i < 2000; i += 37) m[i] = (float) (rng() % 100) / 10.f;
        for (int k : {1, 20, 54, 55, 300}) check_case("masked", m, k, nt);
        // NaN and signed zeros mixed in
        std::vector<float> w(1500);
        for (size_t i = 0; i < w.size(); ++i) {
            switch (rng() % 7) {
                case 0: w[i] = std::numeric_limits<float>::quiet_NaN(); break;
                case 1: w[i] = -0.0f; break;
                case 2: w[i] = 0.0f; break;
                case 3: w[i] = -INFINITY; break;
                case 4: w[i] = INFINITY; break;
                default: w[i] = (float) ((int) (rng() % 9) - 4); break;
            }
        }
        for (int k : {1, 10, 300, 900, 1500}) check_case("specials", w, k, nt);
        // monotone rows
        std::vector<float> up(1800), down(1800);
        for (int i = 0; i < 1800; ++i) { up[i] = (float) i; down[i] = (float) -i; }
        for (int k : {1, 64, 1000, 1800}) { check_case("ascending", up, k, nt); check_case("descending", down, k, nt); }
    }
}

static void test_large_k_and_n() {
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> u(-100.f, 100.f);
    std::vector<float> row(20000);
    for (auto & x : row) x = std::round(u(rng) * 8.f) / 8.f;       // many ties at 1/8 resolution
    for (int k : {2048, 4095, 4096}) check_case("large", row, k, 128);
}

static void test_controls() {
    // (a) the comparison is sensitive to the tie rule: lowest-index-first and highest-index-first references differ
    std::vector<float> row(500, 1.0f);
    CHECK(reference(row, 10, true) != reference(row, 10, false));
    // so a kernel that broke ties the other way would be caught: check the kernel agrees with the first, not the second
    CHECK(run_kernel(row, 10, 64) == reference(row, 10, true));
    CHECK(run_kernel(row, 10, 64) != reference(row, 10, false));
    // (b) a corrupted output must not equal the reference
    auto got = run_kernel(row, 10, 64);
    got[3] ^= 1;
    CHECK(got != reference(row, 10, true));
    // (c) thread count must not change the answer
    std::mt19937 rng(5);
    std::vector<float> r(2500);
    for (auto & x : r) x = (float) (rng() % 40);
    const auto a = run_kernel(r, 700, 32), b = run_kernel(r, 700, 160);
    CHECK(a == b);
}

int main() {
    test_edge_cases();
    test_ties_and_specials();
    test_large_k_and_n();
    test_controls();
    if (g_fail) { std::fprintf(stderr, "test-topk-select: %d FAILURES\n", g_fail); return 1; }
    std::printf("test-topk-select: PASS\n");
    return 0;
}
