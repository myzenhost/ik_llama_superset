// Host-backend throughput of the Prism ternary dot products against a reference dot product, single thread.
//
// Row S69-GGML-QUANTS-C-BUILT-WITHOUT-AVX2-PRISM-TERNARY-DOTS-RUN-SCALAR-2026-10-04-01 says ggml-quants.c (ggml-base) is compiled without /arch:AVX2, so
// the AVX2/SSSE3 branches of ggml_vec_dot_pq2_0_q8_0 and ggml_vec_dot_ptq1_0_q8_0 are dead code and the scalar path runs. That row had no number. This prints
// one: giga-weights per second for each type, next to Q8_0 x Q8_0 (whose vec_dot lives in ggml-cpu, built with AVX2) as the speed a vectorised dot reaches here.
//
// SCOPE: one thread, K = 4096, one weight row dotted against one activation row, repeated until >= 0.5 s per type; the result is accumulated and printed so the
// compiler cannot drop the loop. It reports a ratio; it does not assert one (a pass/fail threshold would be invented). Exit 0 unless a type is missing.

#include "ggml.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// ggml-base entry points (ggml/src/ggml-quants.h, internal header; declared here by signature).
extern "C" {
void ggml_gemv_ptq1_0_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemm_ptq1_0_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
}

struct case_t { ggml_type w; const char * label; };

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    ggml_cpu_init();
    const int64_t K = 4096;
    const case_t cases[] = {
        { GGML_TYPE_Q8_0,   "Q8_0   (reference, vectorised in ggml-cpu)" },
        { GGML_TYPE_PQ2_0,  "PQ2_0  " },
        { GGML_TYPE_PTQ1_0, "PTQ1_0 " },
    };
    std::mt19937 rng(20261004);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> w(K), x(K);
    for (auto & v : w) v = nd(rng) * 0.02f;
    for (auto & v : x) v = nd(rng);

    double rate[3] = {0, 0, 0};
    int missing = 0;
    for (int c = 0; c < 3; c++) {
        const ggml_type_traits_cpu * wc = ggml_get_type_traits_cpu(cases[c].w);
        if (!wc || !wc->vec_dot || !wc->from_float) { printf("%-45s MISSING (no CPU vec_dot/from_float in this build)\n", cases[c].label); missing++; continue; }
        const ggml_type at = wc->vec_dot_type;
        const ggml_type_traits_cpu * ac = ggml_get_type_traits_cpu(at);
        if (!ac || !ac->from_float) { printf("%-45s MISSING activation type %s\n", cases[c].label, ggml_type_name(at)); missing++; continue; }
        std::vector<char> wq(ggml_row_size(cases[c].w, K) + 64), xq(ggml_row_size(at, K) + 64);
        wc->from_float(w.data(), wq.data(), K);
        ac->from_float(x.data(), xq.data(), K);
        float s = 0.0f, acc = 0.0f;
        {
            // Agreement check: the dot must equal sum(dequant(w) * dequant(x)) computed in double. A kernel that is fast and wrong must not pass.
            const ggml_type_traits * wt = ggml_get_type_traits(cases[c].w);
            const ggml_type_traits * at_t = ggml_get_type_traits(at);
            if (!wt || !wt->to_float || !at_t || !at_t->to_float) { printf("%-45s MISSING to_float for the agreement check\n", cases[c].label); missing++; continue; }
            std::vector<float> wd(K), xd(K);
            wt->to_float(wq.data(), wd.data(), K);
            at_t->to_float(xq.data(), xd.data(), K);
            double ref = 0.0, mag = 0.0;
            for (int64_t i = 0; i < K; i++) { ref += (double) wd[i] * xd[i]; mag += std::fabs((double) wd[i] * xd[i]); }
            float got = 0.0f;
            wc->vec_dot((int) K, &got, 0, wq.data(), 0, xq.data(), 0, 1);
            const double err = std::fabs((double) got - ref) / (mag > 0 ? mag : 1.0);
            printf("%-45s agreement: dot %.6f  reference %.6f  |diff|/sum|terms| %.2e\n", cases[c].label, (double) got, ref, err);
            if (!(err < 1e-5)) { printf("%-45s DISAGREES with its own dequantised reference\n", cases[c].label); missing++; continue; }
        }
        long long iters = 0;
        const auto t0 = std::chrono::steady_clock::now();
        double secs = 0.0;
        do {
            for (int i = 0; i < 2000; i++) {
                wc->vec_dot((int) K, &s, 0, wq.data(), 0, xq.data(), 0, 1);
                acc += s;
            }
            iters += 2000;
            secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } while (secs < 0.5);
        rate[c] = (double) iters * (double) K / secs / 1e9;
        printf("%-45s vec_dot_type %-6s  %8.3f G weights/s   (checksum %.4g)\n", cases[c].label, ggml_type_name(at), rate[c], (double) acc);
    }
    {
        // GEMV / GEMM agreement for PTQ1_0 (the tiled path decodes a block's codes once and reuses them across activation rows; a decode bug there
        // would not show in the single-row dot above). nr = 37 crosses the 32-row tile so the tail tile runs too.
        const int n = 512, nr = 37, nc = 3;
        const ggml_type_traits_cpu * wc = ggml_get_type_traits_cpu(GGML_TYPE_PTQ1_0);
        const ggml_type_traits_cpu * ac = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        const ggml_type_traits * wt = ggml_get_type_traits(GGML_TYPE_PTQ1_0);
        const ggml_type_traits * at_t = ggml_get_type_traits(GGML_TYPE_Q8_0);
        const size_t wrow = ggml_row_size(GGML_TYPE_PTQ1_0, n), arow = ggml_row_size(GGML_TYPE_Q8_0, n);
        std::vector<char> wq(wrow * nc), xq(arow * nr);
        std::vector<float> wf(n), xf(n), wd((size_t) n * nc), xd((size_t) n * nr);
        for (int r = 0; r < nc; r++) {
            for (auto & v : wf) v = nd(rng) * 0.02f;
            wc->from_float(wf.data(), wq.data() + r * wrow, n);
            wt->to_float(wq.data() + r * wrow, wd.data() + (size_t) r * n, n);
        }
        for (int r = 0; r < nr; r++) {
            for (auto & v : xf) v = nd(rng);
            ac->from_float(xf.data(), xq.data() + r * arow, n);
            at_t->to_float(xq.data() + r * arow, xd.data() + (size_t) r * n, n);
        }
        std::vector<float> sv((size_t) nr * nc, 0.0f), sm((size_t) nr * nc, 0.0f);
        ggml_gemv_ptq1_0_q8_0(n, sv.data(), nc, wq.data(), xq.data(), nr, nc);
        ggml_gemm_ptq1_0_q8_0(n, sm.data(), nc, wq.data(), xq.data(), nr, nc);
        double worst_v = 0.0, worst_m = 0.0;
        for (int iy = 0; iy < nr; iy++) for (int ix = 0; ix < nc; ix++) {
            double ref = 0.0, mag = 0.0;
            for (int k = 0; k < n; k++) { const double t = (double) wd[(size_t) ix * n + k] * xd[(size_t) iy * n + k]; ref += t; mag += std::fabs(t); }
            worst_v = std::fmax(worst_v, std::fabs((double) sv[(size_t) iy * nc + ix] - ref) / mag);
            worst_m = std::fmax(worst_m, std::fabs((double) sm[(size_t) iy * nc + ix] - ref) / mag);
        }
        printf("PTQ1_0 gemv/gemm agreement over %d x %d outputs: worst gemv %.2e, worst gemm %.2e (|diff|/sum|terms|)\n", nr, nc, worst_v, worst_m);
        if (!(worst_v < 1e-5) || !(worst_m < 1e-5)) { printf("PTQ1_0 gemv/gemm DISAGREE with the dequantised reference\n"); missing++; }
    }
    if (rate[0] > 0) {
        for (int c = 1; c < 3; c++) {
            if (rate[c] > 0) printf("%-45s = %.2fx of the Q8_0 reference rate\n", cases[c].label, rate[c] / rate[0]);
        }
    }
    return missing ? 1 : 0;
}
