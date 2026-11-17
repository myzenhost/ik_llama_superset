// CPU sparse down-projection: dense AXPY vs live-row AXPY at batch 1 vs batch 8
// (row S69-ENGREV-POWERINFER-AXPY-BATCH8-2026-10-27-01). Written from the row's description only, no third-party code read.
//
// The down projection is y[0..d) = sum_i x[i] * W[i][0..d) over rows i of a load-time TRANSPOSED matrix (one input
// activation per row, d contiguous outputs). When most x[i] are exactly zero (BitNet decode: zero fraction 0.58-0.61) only
// the live rows matter. Three kernels, all fp32 FMA, all adding rows to y in ascending row order:
//   dense    : every row, live or not                                  (the baseline)
//   sparse1  : live rows only, y loaded and stored once per live row
//   sparse8  : live rows only, 8 rows per pass over y (y loaded and stored once per 8 rows, the 8 FMAs chained in registers)
// Because each output element sees the same FMA chain in the same order, all three must be BIT-IDENTICAL on identical masks.
//
// default mode : small shapes, bit-exactness over several live fractions (0, 1/8, 0.4, 1, a live count that is not a multiple of
//                8), plus a negative control (a changed mask entry must change the output). Exit 0 on success.
// --bench      : timing table, per-live-row cost relative to a dense row (break-even recorded on the row: 2.57; target <= 2.0).
//                Single thread, fp32 weights (NOT the quantised IQ2_BN weights a real kernel would read).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define HAVE_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#define K_TARGET
static bool cpu_has_avx2_fma() {
    int r[4]; __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!(fma && osxsave && avx)) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(r, 7, 0);
    return (r[1] >> 5) & 1;
}
#else
#define K_TARGET __attribute__((target("avx2,fma")))
static bool cpu_has_avx2_fma() { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }
#endif
#else
#define HAVE_X86 0
#endif

#if HAVE_X86

K_TARGET static void axpy_row(float * y, const float * w, float xi, int d) {
    const __m256 vx = _mm256_set1_ps(xi);
    for (int j = 0; j < d; j += 8) {
        _mm256_storeu_ps(y + j, _mm256_fmadd_ps(vx, _mm256_loadu_ps(w + j), _mm256_loadu_ps(y + j)));
    }
}

// all rows, in order
K_TARGET static void down_dense(float * y, const float * W, const float * x, int n_rows, int d) {
    for (int i = 0; i < n_rows; ++i) axpy_row(y, W + (size_t) i * d, x[i], d);
}

// live rows only, one pass over y per live row
K_TARGET static void down_sparse1(float * y, const float * W, const float * x, const int * live, int n_live, int d) {
    for (int k = 0; k < n_live; ++k) axpy_row(y, W + (size_t) live[k] * d, x[live[k]], d);
}

// live rows only, 8 rows per pass over y; the 8 FMAs per element are chained in registers in ascending row order
K_TARGET static void down_sparse8(float * y, const float * W, const float * x, const int * live, int n_live, int d) {
    int k = 0;
    for (; k + 8 <= n_live; k += 8) {
        const float * r[8]; __m256 vx[8];
        for (int b = 0; b < 8; ++b) { r[b] = W + (size_t) live[k + b] * d; vx[b] = _mm256_set1_ps(x[live[k + b]]); }
        for (int j = 0; j < d; j += 8) {
            __m256 acc = _mm256_loadu_ps(y + j);
            acc = _mm256_fmadd_ps(vx[0], _mm256_loadu_ps(r[0] + j), acc);
            acc = _mm256_fmadd_ps(vx[1], _mm256_loadu_ps(r[1] + j), acc);
            acc = _mm256_fmadd_ps(vx[2], _mm256_loadu_ps(r[2] + j), acc);
            acc = _mm256_fmadd_ps(vx[3], _mm256_loadu_ps(r[3] + j), acc);
            acc = _mm256_fmadd_ps(vx[4], _mm256_loadu_ps(r[4] + j), acc);
            acc = _mm256_fmadd_ps(vx[5], _mm256_loadu_ps(r[5] + j), acc);
            acc = _mm256_fmadd_ps(vx[6], _mm256_loadu_ps(r[6] + j), acc);
            acc = _mm256_fmadd_ps(vx[7], _mm256_loadu_ps(r[7] + j), acc);
            _mm256_storeu_ps(y + j, acc);
        }
    }
    for (; k < n_live; ++k) axpy_row(y, W + (size_t) live[k] * d, x[live[k]], d);   // tail (< 8 rows)
}

struct problem {
    int n_rows, d;
    std::vector<float> W, x;
    std::vector<int> live;
};

static problem make_problem(int n_rows, int d, double live_frac, uint32_t seed, int force_live = -1) {
    problem p; p.n_rows = n_rows; p.d = d;
    p.W.resize((size_t) n_rows * d); p.x.assign(n_rows, 0.0f);
    uint32_t s = seed;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.0f / 16777216.0f); };   // [0,1)
    for (auto & w : p.W) w = rnd() * 2.0f - 1.0f;
    for (int i = 0; i < n_rows; ++i) {
        const bool on = force_live >= 0 ? (i % std::max(1, n_rows / std::max(1, force_live)) == 0 && (int) p.live.size() < force_live)
                                        : rnd() < live_frac;
        if (on) { p.x[i] = rnd() * 2.0f - 1.0f; if (p.x[i] == 0.0f) p.x[i] = 0.5f; p.live.push_back(i); }
    }
    return p;
}

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

static void run_all(const problem & p, std::vector<float> & yd, std::vector<float> & y1, std::vector<float> & y8) {
    yd.assign(p.d, 0.0f); y1.assign(p.d, 0.0f); y8.assign(p.d, 0.0f);
    down_dense(yd.data(), p.W.data(), p.x.data(), p.n_rows, p.d);
    down_sparse1(y1.data(), p.W.data(), p.x.data(), p.live.data(), (int) p.live.size(), p.d);
    down_sparse8(y8.data(), p.W.data(), p.x.data(), p.live.data(), (int) p.live.size(), p.d);
}

static int test_mode() {
    struct cfg { int n_rows, d; double frac; int force; const char * name; };
    const cfg cfgs[] = {
        {128, 96, 0.0, -1, "none live"}, {128, 96, 1.0, -1, "all live"}, {128, 96, 0.125, -1, "12.5% live"},
        {200, 96, 0.4, -1, "40% live"}, {200, 128, 0.0, 13, "13 live (tail of 5)"}, {200, 128, 0.0, 8, "exactly 8 live"},
        {200, 128, 0.0, 7, "7 live (no full batch)"},
    };
    for (const auto & c : cfgs) {
        problem p = make_problem(c.n_rows, c.d, c.frac, 777u + (uint32_t) c.n_rows, c.force);
        std::vector<float> yd, y1, y8;
        run_all(p, yd, y1, y8);
        CHECK(std::memcmp(yd.data(), y1.data(), sizeof(float) * c.d) == 0, "%s: sparse1 != dense bitwise (live %zu)", c.name, p.live.size());
        CHECK(std::memcmp(yd.data(), y8.data(), sizeof(float) * c.d) == 0, "%s: sparse8 != dense bitwise (live %zu)", c.name, p.live.size());
        if (c.force >= 0) CHECK((int) p.live.size() == c.force, "%s: built %zu live rows, wanted %d", c.name, p.live.size(), c.force);
    }
    // negative control: dropping one live row from the list must change the sparse output (the comparison is not vacuous)
    problem p = make_problem(200, 96, 0.4, 4242u);
    std::vector<float> yd, y1, y8;
    run_all(p, yd, y1, y8);
    problem q = p; q.live.erase(q.live.begin() + (long) q.live.size() / 2);
    std::vector<float> z(q.d, 0.0f);
    down_sparse8(z.data(), q.W.data(), q.x.data(), q.live.data(), (int) q.live.size(), q.d);
    CHECK(std::memcmp(yd.data(), z.data(), sizeof(float) * q.d) != 0, "negative control: a dropped live row was NOT detected");
    if (g_fail) { std::printf("test-sparse-downproj-axpy: %d FAILED\n", g_fail); return 1; }
    std::printf("test-sparse-downproj-axpy: OK (sparse1 and sparse8 bit-identical to dense over %zu masks, negative control detects)\n",
                sizeof(cfgs) / sizeof(cfgs[0]));
    return 0;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

static void bench_one(const char * label, int n_rows, int d, double live_frac, int reps) {
    problem p = make_problem(n_rows, d, live_frac, 99u);
    const int n_live = (int) p.live.size();
    std::vector<float> y(d), ref(d);
    std::vector<double> td, t1, t8;
    auto time_it = [&](auto && fn) {
        std::fill(y.begin(), y.end(), 0.0f);
        auto a = std::chrono::steady_clock::now();
        fn();
        auto b = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    for (int r = 0; r < reps + 2; ++r) {      // 2 warm-up reps, kernels interleaved so drift hits all alike
        const double a = time_it([&] { down_dense(y.data(), p.W.data(), p.x.data(), n_rows, d); });
        if (r == 0) ref = y;
        const double b = time_it([&] { down_sparse1(y.data(), p.W.data(), p.x.data(), p.live.data(), n_live, d); });
        const bool ok1 = std::memcmp(ref.data(), y.data(), sizeof(float) * d) == 0;
        const double c = time_it([&] { down_sparse8(y.data(), p.W.data(), p.x.data(), p.live.data(), n_live, d); });
        const bool ok8 = std::memcmp(ref.data(), y.data(), sizeof(float) * d) == 0;
        if (!ok1 || !ok8) { std::printf("BIT-EXACT FAILURE in bench (%s)\n", label); g_fail++; }
        if (r >= 2) { td.push_back(a); t1.push_back(b); t8.push_back(c); }
    }
    const double md = median(td), m1 = median(t1), m8 = median(t8);
    const double dense_row = md / n_rows;                 // us per dense row
    std::printf("%-26s rows %5d d %4d live %5d (%.0f%%) | dense %9.1f us | sparse1 %9.1f us | sparse8 %9.1f us | "
                "per-live-row cost vs dense row: b1 %.2fx  b8 %.2fx | b8 vs b1 %.2fx | whole-call speedup vs dense: b1 %.2fx b8 %.2fx\n",
                label, n_rows, d, n_live, 100.0 * n_live / n_rows, md, m1, m8,
                (m1 / n_live) / dense_row, (m8 / n_live) / dense_row, m1 / m8, md / m1, md / m8);
}

#endif // HAVE_X86

int main(int argc, char ** argv) {
#if HAVE_X86
    if (!cpu_has_avx2_fma()) { std::printf("test-sparse-downproj-axpy: SKIP (no AVX2+FMA on this CPU)\n"); return 0; }
    const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
    if (!bench) return test_mode();
    const int reps = argc > 2 ? std::atoi(argv[2]) : 15;
    std::printf("fp32 weights, single thread, median of %d reps; break-even recorded on the row = 2.57x, target <= 2.0x\n", reps);
    bench_one("mem-bound 61% zero",   6912, 2560, 0.39, reps);   // 70 MB of weights: streams from memory
    bench_one("mem-bound 90% zero",   6912, 2560, 0.10, reps);
    bench_one("cache-resident 61%",    128, 2560, 0.39, reps);   // 1.3 MB of weights: L2/L3 resident
    bench_one("cache-resident 90%",    128, 2560, 0.10, reps);
    bench_one("L2-small 61% zero",      32, 2560, 0.39, reps);
    return g_fail ? 1 : 0;
#else
    (void) argc; (void) argv;
    std::printf("test-sparse-downproj-axpy: SKIP (not x86)\n");
    return 0;
#endif
}
