// SGMV-style grouped LoRA vs the per-token loop (common/lora-sgmv.h). Row S69-LORAX-PICK-SGMV-HETEROGENEOUS-LORA-BATCHING-2026-09-29-01.
//   G1  random ids incl. -1 (base only) and an adapter with no weights: grouped == per-token naive, and base-only rows are bit-unchanged
//   G2  segment structure: runs of length 1, all-one-adapter, all-base, empty batch, mixed ranks per adapter
//   G3  independent check of the naive loop itself against a double-precision triple sum
//   G4  INFO (not asserted): CPU time grouped vs per-token for 64 tokens, 4 adapters, rank 16, d=2048; and the same shape through ggml MUL_MAT_ID
#include "lora-sgmv.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

struct fixture {
    int n_in, n_out, n_ad;
    std::vector<std::vector<float>> A, B;
    std::vector<lora_sgmv_adapter> ad;
    fixture(int n_in_, int n_out_, const std::vector<int> & ranks, std::mt19937 & rng) : n_in(n_in_), n_out(n_out_), n_ad((int) ranks.size()) {
        std::normal_distribution<float> nd(0.0f, 0.3f);
        A.resize(n_ad); B.resize(n_ad); ad.resize(n_ad);
        for (int a = 0; a < n_ad; ++a) {
            if (ranks[a] == 0) continue;           // adapter slot with no weights
            A[a].resize((size_t) ranks[a] * n_in); B[a].resize((size_t) n_out * ranks[a]);
            for (auto & v : A[a]) v = nd(rng);
            for (auto & v : B[a]) v = nd(rng);
            ad[a] = { A[a].data(), B[a].data(), ranks[a], 0.5f + 0.25f * a };
        }
    }
};

static double max_rel(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { double d = (double) a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

static void compare(const fixture & f, const std::vector<int> & ids, std::mt19937 & rng, const char * what) {
    const int T = (int) ids.size();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) T * f.n_in), y0((size_t) T * f.n_out);
    for (auto & v : x) v = nd(rng);
    for (auto & v : y0) v = nd(rng);                       // non-zero start so "+=" is exercised
    std::vector<float> yn = y0, yg = y0;
    lora_sgmv_apply_naive(x.data(), yn.data(), ids.data(), T, f.n_in, f.n_out, f.ad.data(), f.n_ad);
    lora_sgmv_plan plan = lora_sgmv_group(ids.data(), T, f.ad.data(), f.n_ad);
    lora_sgmv_apply_grouped(x.data(), yg.data(), plan, f.n_in, f.n_out, f.ad.data());
    const double e = max_rel(yg, yn);
    size_t seg_tokens = 0; for (auto & s : plan.segs) seg_tokens += s.len;
    CHECK(seg_tokens == plan.perm.size());
    CHECK(e < 1e-5);
    for (int t = 0; t < T; ++t) {                          // base-only rows untouched, bit for bit
        const int a = ids[t];
        const bool base = a < 0 || a >= f.n_ad || f.ad[a].A == nullptr;
        if (base) CHECK(std::memcmp(&yg[(size_t) t * f.n_out], &y0[(size_t) t * f.n_out], sizeof(float) * f.n_out) == 0);
    }
    printf("%-34s T=%3d segments=%zu  rel err grouped vs per-token = %.2e\n", what, T, plan.segs.size(), e);
}

int main() {
    std::mt19937 rng(11);
    {   // G1
        fixture f(40, 24, { 4, 8, 0, 3 }, rng);           // adapter 2 has no weights; ranks differ
        std::vector<int> ids(50);
        for (auto & v : ids) v = (int) (rng() % 7) - 2;    // -2..4: -2,-1 base, 0,1,3 real, 2 no weights, 4 out of range
        compare(f, ids, rng, "G1 random ids incl. base/none/OOR");
    }
    {   // G2
        fixture f(33, 17, { 5, 2 }, rng);
        compare(f, { 0 },                       rng, "G2 single token");
        compare(f, { -1 },                      rng, "G2 single base token");
        compare(f, { 0, 1 },                   rng, "G2 two segments of length 1");
        compare(f, { 0, 1, 0, 1, 0, 1, 0 },     rng, "G2 alternating ids (interleaved)");
        compare(f, { 1, 1, 1, 1, 1, 1 },        rng, "G2 all one adapter");
        compare(f, { -1, -1, -1 },              rng, "G2 all base");
        compare(f, {},                          rng, "G2 empty batch");
        compare(f, { 0, -1, 1, -1, 1, 0, 0, -1 }, rng, "G2 interleaved with base");
        std::vector<int> ids = { 1, 0, 1, 0, 1 };
        lora_sgmv_plan p = lora_sgmv_group(ids.data(), 5, f.ad.data(), f.n_ad);
        CHECK(p.segs.size() == 2 && p.segs[0].adapter == 0 && p.segs[0].len == 2 && p.segs[1].adapter == 1 && p.segs[1].len == 3);
        CHECK((p.perm == std::vector<int>{ 1, 3, 0, 2, 4 }));
    }
    {   // G3: naive loop vs double triple sum
        fixture f(9, 7, { 3 }, rng);
        std::normal_distribution<float> nd(0, 1);
        std::vector<float> x(9); for (auto & v : x) v = nd(rng);
        std::vector<float> y(7, 1.0f); int id = 0;
        lora_sgmv_apply_naive(x.data(), y.data(), &id, 1, 9, 7, f.ad.data(), 1);
        double worst = 0;
        for (int o = 0; o < 7; ++o) {
            double s = 0;
            for (int k = 0; k < 3; ++k) { double v = 0; for (int i = 0; i < 9; ++i) v += (double) f.A[0][k * 9 + i] * x[i]; s += (double) f.B[0][o * 3 + k] * v; }
            worst = std::max(worst, std::fabs(1.0 + f.ad[0].scale * s - y[o]));
        }
        CHECK(worst < 1e-5);
        printf("G3 naive vs double triple sum: max |err| %.2e\n", worst);
    }
    {   // G4 timing
        const int T = 64, NA = 4, R = 16, D = 2048, REPS = 30;
        fixture f(D, D, { R, R, R, R }, rng);
        std::vector<int> ids(T); for (int t = 0; t < T; ++t) ids[t] = (t * 5 + 1) % NA;   // fully interleaved: worst case for contiguous runs
        std::vector<float> x((size_t) T * D), y((size_t) T * D, 0.0f);
        std::normal_distribution<float> nd(0, 1); for (auto & v : x) v = nd(rng);
        auto time_us = [&](auto fn) {
            std::vector<double> s;
            for (int r = 0; r < REPS + 3; ++r) {
                std::fill(y.begin(), y.end(), 0.0f);
                auto t0 = std::chrono::steady_clock::now(); fn(); auto t1 = std::chrono::steady_clock::now();
                if (r >= 3) s.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(s.begin(), s.end()); return s[s.size() / 2];
        };
        const double t_naive = time_us([&] { lora_sgmv_apply_naive(x.data(), y.data(), ids.data(), T, D, D, f.ad.data(), NA); });
        std::vector<float> yn = y;
        lora_sgmv_plan plan = lora_sgmv_group(ids.data(), T, f.ad.data(), NA);
        const double t_group = time_us([&] { lora_sgmv_apply_grouped(x.data(), y.data(), plan, D, D, f.ad.data()); });
        printf("G4 d=%d T=%d adapters=%d rank=%d (median of %d): per-token %.0f us, grouped %.0f us, ratio per-token/grouped = %.2f; match rel err %.2e\n",
               D, T, NA, R, REPS, t_naive, t_group, t_naive / t_group, max_rel(y, yn));
        CHECK(max_rel(y, yn) < 1e-5);

        // same shape through the ggml ops the fork already has (MUL_MAT_ID with the adapter id as expert id)
        ggml_backend_t be = ggml_backend_cpu_init();
        if (be) {
            ggml_backend_cpu_set_n_threads(be, 1);   // like-for-like with the single-thread reference loops
            ggml_init_params ip = { 64u * 1024 * 1024, nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * a  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, R, NA);
            ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, R, D, NA);
            ggml_tensor * xs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, T);
            ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, T);
            ggml_tensor * out = ggml_mul_mat_id(ctx, b, ggml_mul_mat_id(ctx, a, xs, id), id);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
            std::vector<float> As, Bs;     // ggml layout: A[ad][k][i] (ne0 = D), B[ad][o][k] (ne0 = R) -- same row-major as ours
            for (int q = 0; q < NA; ++q) { As.insert(As.end(), f.A[q].begin(), f.A[q].end()); Bs.insert(Bs.end(), f.B[q].begin(), f.B[q].end()); }
            std::vector<int32_t> idv(ids.begin(), ids.end());
            ggml_backend_tensor_set(a, As.data(), 0, ggml_nbytes(a));
            ggml_backend_tensor_set(b, Bs.data(), 0, ggml_nbytes(b));
            ggml_backend_tensor_set(xs, x.data(), 0, ggml_nbytes(xs));
            ggml_backend_tensor_set(id, idv.data(), 0, ggml_nbytes(id));
            std::vector<double> s; std::vector<float> g((size_t) T * D);
            for (int r = 0; r < REPS + 3; ++r) {
                auto t0 = std::chrono::steady_clock::now(); ggml_backend_graph_compute(be, gf); auto t1 = std::chrono::steady_clock::now();
                if (r >= 3) s.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(s.begin(), s.end());
            ggml_backend_tensor_get(out, g.data(), 0, ggml_nbytes(out));
            // ours without the per-adapter scale for a like-for-like compare: rebuild y with scale 1
            std::vector<lora_sgmv_adapter> ad1 = f.ad; for (auto & q : ad1) q.scale = 1.0f;
            std::vector<float> y1((size_t) T * D, 0.0f);
            lora_sgmv_apply_grouped(x.data(), y1.data(), plan, D, D, ad1.data());
            const double e = max_rel(g, y1);
            printf("G4 ggml 2x MUL_MAT_ID (1 thread) median %.0f us; rel err vs grouped reference (scale 1) %.2e\n", s[s.size() / 2], e);
            CHECK(e < 1e-4);
            ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
        }
    }
    if (n_fail) { fprintf(stderr, "test-lora-sgmv-grouped: %d failure(s)\n", n_fail); return 1; }
    printf("test-lora-sgmv-grouped: all checks passed\n");
    return 0;
}
