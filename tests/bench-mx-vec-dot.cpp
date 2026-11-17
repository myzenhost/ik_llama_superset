// bench-mx-vec-dot: MXFP6/MXFP8 x Q8_0 dot product, scalar reference (ggml-quants.c) vs the CPU-backend entry point (AVX2+FMA when built with it).
// Usage: bench-mx-vec-dot [blocks=256] [calls_per_rep=20000] [reps=9]     Prints per-rep ns/block and mean +- sd over the reps (single thread).
// Not a test; the numbers depend on the box. BENCH-PENDING for any claim beyond the one run recorded in docs/agent-reports/wire-types-kv-gate.md.
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" {
void ggml_vec_dot_mxfp6_e2m3_q8_0_cpu(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_mxfp6_e3m2_q8_0_cpu(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_mxfp8_e4m3_q8_0_cpu(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_vec_dot_mxfp8_e5m2_q8_0_cpu(int, float *, size_t, const void *, size_t, const void *, size_t, int);
}
typedef void (*vd_fn)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*q_fn)(const float *, void *, int64_t);

int main(int argc, char ** argv) {
    const int nblk = argc > 1 ? atoi(argv[1]) : 256, calls = argc > 2 ? atoi(argv[2]) : 20000, reps = argc > 3 ? atoi(argv[3]) : 9;
    { ggml_init_params ip = { 1024, nullptr, true }; ggml_free(ggml_init(ip)); }
    ggml_cpu_init();
    struct F { const char * n; size_t bb; q_fn q; vd_fn ref, cpu; } fm[] = {
        { "mxfp6_e2m3", 25, quantize_row_mxfp6_e2m3, ggml_vec_dot_mxfp6_e2m3_q8_0, ggml_vec_dot_mxfp6_e2m3_q8_0_cpu },
        { "mxfp6_e3m2", 25, quantize_row_mxfp6_e3m2, ggml_vec_dot_mxfp6_e3m2_q8_0, ggml_vec_dot_mxfp6_e3m2_q8_0_cpu },
        { "mxfp8_e4m3", 33, quantize_row_mxfp8_e4m3, ggml_vec_dot_mxfp8_e4m3_q8_0, ggml_vec_dot_mxfp8_e4m3_q8_0_cpu },
        { "mxfp8_e5m2", 33, quantize_row_mxfp8_e5m2, ggml_vec_dot_mxfp8_e5m2_q8_0, ggml_vec_dot_mxfp8_e5m2_q8_0_cpu },
    };
    const int n = 32 * nblk;
    std::mt19937_64 rng(5);
    std::normal_distribution<double> N(0, 1);
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) { a[i] = (float) N(rng); b[i] = (float) N(rng); }
    std::vector<block_q8_0> qb(nblk);
    quantize_row_q8_0_ref(b.data(), qb.data(), n);
    printf("row = %d blocks, %d calls/rep, %d reps; ns per 32-element block\n", nblk, calls, reps);
    for (auto & f : fm) {
        std::vector<uint8_t> qa(f.bb * nblk);
        f.q(a.data(), qa.data(), n);
        for (int which = 0; which < 2; ++which) {
            vd_fn fn = which ? f.cpu : f.ref;
            std::vector<double> t;
            float sink = 0, s = 0;
            for (int r = 0; r < reps + 1; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                for (int c = 0; c < calls; ++c) { fn(n, &s, 0, qa.data(), 0, qb.data(), 0, 1); sink += s; }
                auto t1 = std::chrono::steady_clock::now();
                if (r > 0) t.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / ((double) calls * nblk)); // rep 0 = warm-up
            }
            double m = 0, v = 0; for (double x : t) m += x; m /= t.size(); for (double x : t) v += (x - m) * (x - m); v = std::sqrt(v / (t.size() - 1));
            printf("%-11s %-6s mean %7.2f ns/block  sd %5.2f  min %7.2f  max %7.2f   (sink %g)\n", f.n, which ? "cpu" : "scalar", m, v, *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()), sink);
        }
    }
    return 0;
}
