// Exact top-k threshold-scan prefilter (common/topk-scan.h) against the REAL old path: llama_sample_top_k() on a vocabulary-sized array.
// Row S69-MNN-PICK-TOPK-RAW-LOGIT-THRESHOLD-PREFILTER-2026-10-05-01.
//   A  scan == llama_sample_top_k on thousands of random cases: same ids, same logits, same order, same size (ties by lower id)
//      shapes: random, heavy ties, 30% -inf, all equal, all -inf, ascending, descending; n 1..64 and 152064; k 1/20/40/64, k == n, k > n
//   B  common_argmax_scan == llama_sample_token_greedy on NaN-free data (first maximum wins ties); -1 when every logit is NaN
//   --bench   micro-benchmark old (copy to 12-byte array + llama_sample_top_k, then greedy) vs new at n = 152064 (1000 calls each)
#include "llama.h"
#include "topk-scan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { if (n_fail < 20) fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static void make(std::vector<float> & l, int n, int kind, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 3.0f);
    l.resize(n);
    for (int i = 0; i < n; ++i) {
        float x = nd(rng);
        switch (kind) {
            case 0: break;
            case 1: x = std::round(x * 2.0f) / 2.0f; break;
            case 2: if (rng() % 10 < 3) x = -INFINITY; break;
            case 3: x = 1.5f; break;
            case 4: x = -INFINITY; break;
            case 5: x = (float) i; break;
            case 6: x = (float) (n - i); break;
            case 7: x = (float) (rng() % 4); break;      // 4 distinct values: massive ties
        }
        l[i] = x;
    }
}

static void old_path(const std::vector<float> & l, int k, std::vector<llama_token_data> & out) {
    out.resize(l.size());
    for (size_t i = 0; i < l.size(); ++i) out[i] = { (llama_token) i, l[i], 0.0f };
    llama_token_data_array a = { out.data(), out.size(), false };
    llama_sample_top_k(nullptr, &a, k, 1);
    out.resize(a.size);
}

static llama_token old_greedy(const std::vector<float> & l) {
    std::vector<llama_token_data> c(l.size());
    for (size_t i = 0; i < l.size(); ++i) c[i] = { (llama_token) i, l[i], 0.0f };
    llama_token_data_array a = { c.data(), c.size(), false };
    return llama_sample_token_greedy(nullptr, &a);
}

static bool same(const std::vector<llama_token_data> & a, const llama_token_data * b, size_t nb) {
    if (a.size() != nb) return false;
    for (size_t i = 0; i < nb; ++i) {
        if (a[i].id != b[i].id || !(a[i].logit == b[i].logit)) return false;
    }
    return true;
}

static double now_ns() {
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int bench() {
    const int n = 152064, reps = 1000;
    std::mt19937 rng(5);
    std::vector<float> l;
    make(l, n, 0, rng);
    std::vector<llama_token_data> out(n), buf(1024);
    volatile long sink = 0;
    printf("bench n=%d reps=%d avx2=%d  (ns per call, mean over %d calls)\n", n, reps, (int) common_topk_have_avx2(), reps);
    for (int k : { 1, 20, 40, 64 }) {
        double t0 = now_ns();
        for (int r = 0; r < reps; ++r) {
            for (int i = 0; i < n; ++i) out[i] = { i, l[i], 0.0f };
            llama_token_data_array a = { out.data(), (size_t) n, false };
            llama_sample_top_k(nullptr, &a, k, 1);
            sink += a.data[0].id;
        }
        const double t_old = (now_ns() - t0) / reps;
        t0 = now_ns();
        for (int r = 0; r < reps; ++r) {
            sink += (long) common_topk_scan(l.data(), n, k, buf.data());
            sink += buf[0].id;
        }
        const double t_new = (now_ns() - t0) / reps;
        printf("top-k  k=%-3d old %12.0f ns   new %12.0f ns   speedup %.1fx\n", k, t_old, t_new, t_old / t_new);
    }
    double t0 = now_ns();
    for (int r = 0; r < reps; ++r) {
        for (int i = 0; i < n; ++i) out[i] = { i, l[i], 0.0f };
        llama_token_data_array a = { out.data(), (size_t) n, false };
        sink += llama_sample_token_greedy(nullptr, &a);
    }
    const double g_old = (now_ns() - t0) / reps;
    t0 = now_ns();
    for (int r = 0; r < reps; ++r) sink += common_argmax_scan(l.data(), n);
    const double g_new = (now_ns() - t0) / reps;
    printf("greedy      old %12.0f ns   new %12.0f ns   speedup %.1fx\n", g_old, g_new, g_old / g_new);
    return 0;
}

int main(int argc, char ** argv) {
    // llama_sample_top_k() reads ggml_time_us(); on Windows that divides by the QueryPerformanceFrequency that only ggml_time_init() sets
    // (llama_backend_init() calls it). Without this the first call dies with STATUS_INTEGER_DIVIDE_BY_ZERO (0xC0000094).
    llama_backend_init();
    if (argc > 1 && !strcmp(argv[1], "--bench")) return bench();
    std::mt19937 rng(2026);
    std::vector<float> l;
    std::vector<llama_token_data> ref, got(2048);
    int n_cases = 0;
    const int sizes[] = { 1, 2, 3, 7, 8, 9, 15, 16, 17, 40, 41, 63, 64, 65, 100, 1000, 4097, 152064 };
    for (int n : sizes) {
        const int reps = n >= 152064 ? 3 : (n >= 1000 ? 20 : 60);
        for (int kind = 0; kind < 8; ++kind) {
            for (int rep = 0; rep < reps; ++rep) {
                make(l, n, kind, rng);
                for (int k : { 1, 20, 40, 64, n, n + 1, n + 100, std::max(1, n / 2) }) {
                    const int kk = std::min(k, 1024);        // the sampler only prefilters k <= 1024
                    old_path(l, kk, ref);
                    const size_t g = common_topk_scan(l.data(), n, kk, got.data());
                    CHECK(same(ref, got.data(), g));
                    CHECK(g == std::min<size_t>(kk, n));
                    std::vector<llama_token_data> got_s(2048);
                    const size_t gs = common_topk_scan_scalar(l.data(), n, std::min(kk, n), got_s.data());
                    CHECK(same(ref, got_s.data(), gs));
                    ++n_cases;
                }
                // B
                const int32_t am = common_argmax_scan(l.data(), n);
                CHECK(am == old_greedy(l));
                CHECK(am == common_argmax_scan_scalar(l.data(), n));
                ++n_cases;
            }
        }
    }
    {   // n = 0 / k = 0 / all NaN
        CHECK(common_topk_scan(l.data(), 0, 5, got.data()) == 0);
        CHECK(common_topk_scan(l.data(), 10, 0, got.data()) == 0);
        std::vector<float> nan(100, NAN);
        CHECK(common_argmax_scan(nan.data(), 100) == -1);
        CHECK(common_topk_scan(nan.data(), 100, 5, got.data()) == 0);
    }
    printf("A/B  %d cases agree with the old path (llama_sample_top_k / llama_sample_token_greedy), avx2=%d\n", n_cases, (int) common_topk_have_avx2());
    if (n_fail) {
        fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }
    printf("test-topk-prefilter: all checks passed\n");
    return 0;
}
