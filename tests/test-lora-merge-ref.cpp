// Request-time LoRA merge, second test file for the row S69-LORAX-PICK-REQUEST-TIME-LORA-MERGE-TIES-DARE-2026-09-29-01.
// tests/test-lora-merge.cpp (already in the fork) checks hand-derived values; this one adds what it lacks:
//   R1  lora_delta_dense (B*A*scale) against a triple loop written here, tiny random A/B
//   R2  all four strategies on those deltas against NAIVE references written in this file (TIES: per-position sort-based election;
//       DARE: the documented draw order -- one bernoulli(density) per element, adapter-major, from mt19937_64(seed) -- then the same arithmetic)
//   R3  properties: DARE with drop probability p=0 (density 1) == plain weighted sum; TIES with ONE adapter and no trim == that adapter;
//       DARE-TIES p=0 == TIES; same seed same result
//   R4  merged delta == delta of nothing else: merging with weights (1,0) returns adapter 0 for linear
#include "lora-merge.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

typedef std::vector<float> vf;

static double max_diff(const vf & a, const vf & b) {
    double m = 0;
    if (a.size() != b.size()) return 1e30;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

static vf run(lora_merge_strategy s, float dens, const std::vector<vf> & d, const vf & w, lora_sign_method sm, uint64_t seed) {
    std::vector<const float *> p;
    for (auto & v : d) p.push_back(v.data());
    lora_merge_params mp; mp.strategy = s; mp.density = dens; mp.sign_method = sm; mp.seed = seed;
    vf out(d[0].size());
    std::string err;
    CHECK(lora_merge_deltas(mp, p.data(), w.data(), (int) d.size(), d[0].size(), out.data(), &err));
    return out;
}

// ---- naive references ----
static vf ref_trim(const vf & x, float dens) {           // keep floor(dens*n) largest |x|, lowest index wins ties
    size_t n = x.size(), k = (size_t) std::floor((double) dens * n);
    vf o(n, 0.0f);
    if (k >= n) return x;
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return std::fabs(x[a]) > std::fabs(x[b]); });
    for (size_t i = 0; i < k; ++i) o[idx[i]] = x[idx[i]];
    return o;
}
static vf ref_dare(const vf & x, float dens, std::mt19937_64 & rng) {
    vf o(x.size(), 0.0f);
    if (dens >= 1.0f) return x;
    std::bernoulli_distribution keep((double) dens);
    for (size_t i = 0; i < x.size(); ++i) if (keep(rng)) o[i] = x[i] * (1.0f / dens);
    return o;
}
static vf ref_combine(bool ties, lora_sign_method sm, const std::vector<vf> & pr, const vf & w) {
    size_t n = pr[0].size();
    vf out(n, 0.0f);
    for (size_t j = 0; j < n; ++j) {
        if (!ties) { for (size_t i = 0; i < pr.size(); ++i) out[j] += w[i] * pr[i][j]; continue; }
        double tot = 0, freq = 0;
        for (auto & p : pr) { tot += p[j]; freq += (p[j] > 0) - (p[j] < 0); }
        bool plus = (sm == LORA_SIGN_TOTAL ? tot : freq) >= 0;
        float num = 0; int cnt = 0;
        for (size_t i = 0; i < pr.size(); ++i) {
            float v = pr[i][j];
            if ((plus && v > 0) || (!plus && v < 0)) { num += w[i] * v; cnt++; }
        }
        out[j] = cnt ? num / cnt : 0.0f;
    }
    return out;
}

int main() {
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int rank = 3, n_in = 6, n_out = 5, N = 4;
    const size_t numel = (size_t) n_in * n_out;

    // R1
    std::vector<vf> A(N), B(N), D(N, vf(numel));
    const float scales[N] = { 1.0f, 0.5f, 2.0f, -1.5f };
    double r1 = 0;
    for (int a = 0; a < N; ++a) {
        A[a].resize((size_t) rank * n_in); B[a].resize((size_t) n_out * rank);
        for (auto & v : A[a]) v = nd(rng);
        for (auto & v : B[a]) v = nd(rng);
        CHECK(lora_delta_dense(A[a].data(), B[a].data(), rank, n_in, n_out, scales[a], D[a].data()));
        for (int o = 0; o < n_out; ++o) for (int i = 0; i < n_in; ++i) {
            double s = 0;
            for (int k = 0; k < rank; ++k) s += (double) B[a][o * rank + k] * A[a][k * n_in + i];
            r1 = std::max(r1, std::fabs(scales[a] * s - D[a][o * n_in + i]));
        }
    }
    CHECK(r1 < 1e-5);
    CHECK(!lora_delta_dense(nullptr, B[0].data(), rank, n_in, n_out, 1.0f, D[0].data()));
    CHECK(!lora_delta_dense(A[0].data(), B[0].data(), 0, n_in, n_out, 1.0f, D[0].data()));
    printf("R1 delta_dense max |err| vs triple loop = %.2e\n", r1);

    // R2
    const vf w = { 1.0f, 0.5f, 0.25f, 2.0f };
    double worst = 0;
    for (float dens : { 0.2f, 0.5f, 0.9f, 1.0f }) for (lora_sign_method sm : { LORA_SIGN_TOTAL, LORA_SIGN_FREQUENCY }) {
        // linear
        worst = std::max(worst, max_diff(run(LORA_MERGE_LINEAR, dens, D, w, sm, 0), ref_combine(false, sm, D, w)));
        // ties
        std::vector<vf> tr; for (auto & d : D) tr.push_back(ref_trim(d, dens));
        worst = std::max(worst, max_diff(run(LORA_MERGE_TIES, dens, D, w, sm, 0), ref_combine(true, sm, tr, w)));
        // dare variants
        for (uint64_t seed : { 1ull, 99ull }) {
            std::mt19937_64 g1(seed); std::vector<vf> dr; for (auto & d : D) dr.push_back(ref_dare(d, dens, g1));
            worst = std::max(worst, max_diff(run(LORA_MERGE_DARE_LINEAR, dens, D, w, sm, seed), ref_combine(false, sm, dr, w)));
            worst = std::max(worst, max_diff(run(LORA_MERGE_DARE_TIES,   dens, D, w, sm, seed), ref_combine(true,  sm, dr, w)));
        }
    }
    CHECK(worst < 1e-5);
    printf("R2 4 strategies x 4 densities x 2 sign methods vs naive reference: max |diff| = %.2e\n", worst);

    // R3 properties
    vf plain(numel, 0.0f);
    for (int a = 0; a < N; ++a) for (size_t j = 0; j < numel; ++j) plain[j] += w[a] * D[a][j];
    double p0 = max_diff(run(LORA_MERGE_DARE_LINEAR, 1.0f, D, w, LORA_SIGN_TOTAL, 5), plain);
    CHECK(p0 < 1e-6);
    CHECK(max_diff(run(LORA_MERGE_DARE_TIES, 1.0f, D, w, LORA_SIGN_TOTAL, 5), run(LORA_MERGE_TIES, 1.0f, D, w, LORA_SIGN_TOTAL, 0)) == 0.0);
    std::vector<vf> one = { D[2] };
    CHECK(max_diff(run(LORA_MERGE_TIES, 1.0f, one, vf{ 1.0f }, LORA_SIGN_TOTAL, 0), D[2]) == 0.0);
    CHECK(max_diff(run(LORA_MERGE_TIES, 1.0f, one, vf{ 1.0f }, LORA_SIGN_FREQUENCY, 0), D[2]) == 0.0);
    CHECK(max_diff(run(LORA_MERGE_DARE_LINEAR, 0.5f, D, w, LORA_SIGN_TOTAL, 8), run(LORA_MERGE_DARE_LINEAR, 0.5f, D, w, LORA_SIGN_TOTAL, 8)) == 0.0);
    CHECK(max_diff(run(LORA_MERGE_DARE_LINEAR, 0.5f, D, w, LORA_SIGN_TOTAL, 8), run(LORA_MERGE_DARE_LINEAR, 0.5f, D, w, LORA_SIGN_TOTAL, 9)) > 0.0);
    printf("R3 DARE(p=0) vs plain sum: %.2e; TIES(1 adapter, no trim) == adapter; DARE-TIES(p=0) == TIES; seed determinism ok\n", p0);

    // R4
    CHECK(max_diff(run(LORA_MERGE_LINEAR, 1.0f, D, vf{ 1, 0, 0, 0 }, LORA_SIGN_TOTAL, 0), D[0]) == 0.0);

    if (n_fail) { fprintf(stderr, "test-lora-merge-ref: %d failure(s)\n", n_fail); return 1; }
    printf("test-lora-merge-ref: all checks passed\n");
    return 0;
}
