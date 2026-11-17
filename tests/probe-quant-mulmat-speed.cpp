// MUL_MAT throughput of quantisation TYPES through the real CPU backend (so IQK kernels are used wherever the type has them).
//
// WHY. The narrow IQ1 types have an AVX2 vec_dot only; IQ1_S/IQ1_M ride the IQK GEMM. Whether that gap matters is a number, not a
// guess: this times  y = W x  for W = [K=2048, M=8192] (a Llama-3.2-1B FFN matrix) at N = 1 (decode) and N = 64 (prefill) columns,
// min of REPS runs, per type, with the SAME thread count, on whatever else the host is doing (it is shared -- the numbers carry that
// noise; the MIN of several runs is the least-noisy statistic available, and the ratio between types measured in one process
// back to back is more trustworthy than any absolute value).
//
// USAGE: probe-quant-mulmat-speed [threads=4] [reps=15] [types=iq1_xxxs,iq1_xxs,iq1_xs,iq1_s,iq1_m,iq2_xxs,q4_k,q8_0]

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#define strcasecmp _stricmp   // POSIX name; MSVC spells it _stricmp (the probe built under MinGW only)
#endif

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int threads = argc > 1 ? atoi(argv[1]) : 4;
    const int reps = argc > 2 ? atoi(argv[2]) : 15;
    std::string list = argc > 3 ? argv[3] : "iq1_xxxs,iq1_xxs,iq1_xs,iq1_s,iq1_m,iq2_xxs,q4_k,q8_0";
    const int64_t K = 2048, M = 8192;

    ggml_cpu_init();
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, threads);

    std::mt19937 rng(5);
    std::normal_distribution<float> Nd(0.0f, 0.05f);
    std::vector<float> wf(K*M), imp(K);
    for (auto & v : wf) v = Nd(rng);
    for (auto & v : imp) v = 0.5f + (float) (rng() % 1000) / 1000.0f;

    printf("threads=%d reps=%d  W=[%lld x %lld]\n", threads, reps, (long long) K, (long long) M);
    printf("%-10s %7s | %10s %10s | %10s %10s\n", "type", "bpw", "N=1 ms", "GB/s(w)", "N=64 ms", "GFLOP/s");

    std::stringstream ss(list);
    std::string name;
    while (std::getline(ss, name, ',')) {
        ggml_type t = GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const char * n = ggml_type_name((ggml_type) i);
            if (n && strcasecmp(n, name.c_str()) == 0) { t = (ggml_type) i; break; }
        }
        if (t == GGML_TYPE_COUNT) { printf("%-10s unknown type\n", name.c_str()); continue; }
        double ms[2] = { 0, 0 };
        const int64_t Ns[2] = { 1, 64 };
        for (int k = 0; k < 2; ++k) {
            ggml_init_params ip = { (size_t) 512*1024*1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_2d(ctx, t, K, M);
            ggml_quantize_chunk(t, wf.data(), W->data, 0, M, K, imp.data(), nullptr);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, Ns[k]);
            for (int64_t i = 0; i < K*Ns[k]; ++i) ((float *) X->data)[i] = Nd(rng);
            ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            if (!ggml_backend_supports_op(be, Y)) { printf("%-10s not claimed by the CPU backend\n", name.c_str()); ggml_free(ctx); break; }
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            double best = 1e30;
            for (int r = 0; r < reps + 2; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(be, gf);
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (r >= 2) best = std::min(best, s);   // 2 warm-up runs
            }
            ms[k] = best*1e3;
            ggml_free(ctx);
        }
        const double bytes = (double) ggml_row_size(t, K) * M;
        printf("%-10s %7.4f | %10.3f %10.2f | %10.3f %10.1f\n", name.c_str(), 8.0*ggml_type_size(t)/ggml_blck_size(t),
               ms[0], bytes/1e9/(ms[0]/1e3), ms[1], 2.0*K*M*64/1e9/(ms[1]/1e3));
    }
    ggml_backend_free(be);
    return 0;
}
