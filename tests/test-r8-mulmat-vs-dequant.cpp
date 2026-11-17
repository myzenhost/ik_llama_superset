// Row-interleaved (_r8) quant types: does the CPU MUL_MAT (vec_dot / IQK GEMM) agree with dequantize-then-float-dot?
//
// WHY. test-rowint-dequant-bounds checks that to_float round-trips; test-quantize-fns SKIPS row-interleaved types (its
// buffer is one row). Neither exercises the multiplication kernel. S69-PTQ1-0-R8-DEQUANT-READS-IQ1BN-ROW-HEADER fixed the
// dequantiser of ptq1_0_r8 (rel_rms 11759 -> 0.5558) but left "the vec_dot path was not implicated" unmeasured. A kernel that
// reads the layout differently from to_float shows up here as a large relative error against W_deq . x.
//
// y_ref = dequantise(W) . x in float; y = ggml_mul_mat(W, x) on the CPU backend (activations are quantised to the type's
// vec_dot_type, so a small error remains: bound REL_TOL). The NEGATIVE control compares y against a reference built from a
// shifted x and must be far worse -- otherwise the check could not have failed.
//
// USAGE: test-r8-mulmat-vs-dequant [types=ptq1_0_r8,q8_0_r8,iq4_xs_r8]   exit 0 = all listed types agree.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

static const double REL_TOL = 0.08;   // activation Q8 rounding on top of the weight quantisation both sides share

static double rel_rms(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = a[i] - b[i]; num += d*d; den += (double) b[i]*b[i]; }
    return den > 0 ? std::sqrt(num/den) : -1;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string list = argc > 1 ? argv[1] : "ptq1_0_r8,q8_0_r8,iq4_xs_r8";
    const int64_t K = 1024, M = 8;               // one 8-row interleaved group
    // Decode shape and a GEMM shape by default. R8MM_NS="1,8,64,512" adds prompt-sized batches: large N takes the
    // on-the-fly weight conversion paths (iqk_convert_*_q80_r8) that small N never reaches.
    std::vector<int64_t> Ns = { 1, 8 };
    if (const char * e = std::getenv("R8MM_NS")) {
        Ns.clear();
        std::stringstream ns(e);
        std::string tok;
        while (std::getline(ns, tok, ',')) { if (!tok.empty()) Ns.push_back(std::atoll(tok.c_str())); }
    }
    ggml_cpu_init();
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 2);

    std::mt19937 rng(11);
    std::normal_distribution<float> Nd(0.0f, 0.05f);
    std::vector<float> wf(K*M), imp(K, 1.0f);
    for (auto & v : wf) v = Nd(rng);

    int failed = 0, tested = 0;
    std::stringstream ss(list);
    std::string name;
    while (std::getline(ss, name, ',')) {
        ggml_type t = GGML_TYPE_COUNT;
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const char * n = ggml_type_name((ggml_type) i);
            if (n && strcasecmp(n, name.c_str()) == 0) { t = (ggml_type) i; break; }
        }
        if (t == GGML_TYPE_COUNT) { printf("%-12s UNKNOWN type (counted as failure)\n", name.c_str()); ++failed; continue; }
        const auto * tr = ggml_get_type_traits(t);
        if (!tr->to_float) { printf("%-12s no to_float\n", name.c_str()); ++failed; continue; }
        ++tested;

        ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * W = ggml_new_tensor_2d(ctx, t, K, M);
        const size_t want = ggml_row_size(t, K) * M;
        if (ggml_quantize_chunk(t, wf.data(), W->data, 0, M, K, imp.data(), nullptr) != want) {
            printf("%-12s quantize_chunk size mismatch\n", name.c_str()); ++failed; ggml_free(ctx); continue;
        }
        std::vector<float> wd(K*M);
        // A row-meta type (iq2_bn keeps a float scale at the head of EACH row) must be decoded one row at a time:
        // a K*M call reads it as one row, so every row after the first used row 0's scale at a 4-byte offset
        // (iq2_bn failed here at rel_rms 1.0-1.5 with the GEMM correct -- the oracle was wrong, not the kernel).
        // An interleaved _r8/_r4 type WITHOUT row meta must NOT be split: its to_float de-interleaves 8 rows per
        // call (q8_0_r8 failed at rel_rms 1.3-1.5 when split). An interleaved type WITH row meta (iq2_bn_r4) needs
        // a group-of-4 call and is not handled here -- such a type is reported, not silently mis-decoded.
        if (tr->row_meta_size > 0 && strstr(tr->type_name, "_r") != nullptr) {
            printf("%-12s interleaved row-meta type: this oracle cannot decode it (counted as failure)\n", name.c_str());
            ++failed; ggml_free(ctx); continue;
        }
        if (tr->row_meta_size > 0) {
            for (int64_t m = 0; m < M; ++m) tr->to_float((const char *) W->data + m*ggml_row_size(t, K), wd.data() + m*K, K);
        } else {
            tr->to_float(W->data, wd.data(), K*M);
        }

        for (size_t k = 0; k < Ns.size(); ++k) {
            const int64_t N = Ns[k];
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            std::vector<float> xf(K*N);
            for (auto & v : xf) v = Nd(rng);
            memcpy(X->data, xf.data(), xf.size()*sizeof(float));
            ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            if (!ggml_backend_supports_op(be, Y)) { printf("%-12s N=%lld not claimed by CPU backend\n", name.c_str(), (long long) N); ++failed; continue; }
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            ggml_backend_graph_compute(be, gf);
            std::vector<float> y((float *) Y->data, (float *) Y->data + M*N), ref(M*N), bad(M*N);
            for (int64_t n = 0; n < N; ++n) for (int64_t m = 0; m < M; ++m) {
                double s = 0, s2 = 0;
                for (int64_t i = 0; i < K; ++i) {
                    s  += (double) wd[m*K + i] * xf[n*K + i];
                    s2 += (double) wd[m*K + i] * xf[n*K + (i + 1) % K];   // negative control: x shifted by one
                }
                ref[n*M + m] = (float) s; bad[n*M + m] = (float) s2;
            }
            const double good = rel_rms(y, ref), ctrl = rel_rms(y, bad);
            const bool ok = good >= 0 && good < REL_TOL && ctrl > good * 3;
            // FNV-1a over the output bytes: two builds (e.g. AVX2 vs AVX-512) can be compared BIT FOR BIT by diffing logs.
            uint64_t h = 1469598103934665603ull;
            for (size_t b = 0; b < y.size() * sizeof(float); ++b) { h ^= ((const unsigned char *) y.data())[b]; h *= 1099511628211ull; }
            printf("%-12s N=%lld  rel_rms(mul_mat vs deq.x)=%.6f  negctl(shifted x)=%.4f  y_fnv=%016llx  %s\n", name.c_str(), (long long) N, good, ctrl,
                   (unsigned long long) h, ok ? "ok" : "FAIL");
            if (!ok) ++failed;
        }
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    printf("%d type(s) tested, %d failure(s)\n", tested, failed);
    return failed ? 1 : 0;
}
