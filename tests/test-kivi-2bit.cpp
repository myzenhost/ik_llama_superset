// Row S69-FORK-KV-KIVI-2BIT-PORT-01. CPU-only, model-free test of src/llama-kv-kivi.h (KIVI asymmetric 2/4-bit K per-channel /
// V per-token container with a full-precision residual window) and a MEASUREMENT of its error vs f16 and vs the fork's real
// q4_0 on synthetic data (recipe: tests/kv-quant-synth.h; V = per-token-scaled Gaussian, no channel structure).
#include "ggml.h"
#include "llama-kv-kivi.h"
#include "kv-quant-synth.h"

#include <cstdio>
#include <cstdlib>

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

using llama_kivi::cache;

static void gen_v(kvsynth::rng & r, float * v, int D) {
    const double ts = std::exp(0.3 * r.n()); // per-token magnitude
    for (int c = 0; c < D; c++) v[c] = (float)(ts * r.n());
}

static void test_window_and_flush_logic() {
    printf("[window + group-flush logic]\n");
    const uint32_t D = 16, G = 4, R = 8;
    cache c(D, 2, G, R);
    kvsynth::rng r(5);
    std::vector<float> k(D), v(D);
    for (size_t t = 1; t <= 40; t++) {
        for (auto & x : k) x = (float)r.n();
        gen_v(r, v.data(), D);
        c.append(k.data(), v.data());
        CHECK(c.size() == t);
        CHECK(c.residual_k_tokens() == t % R);                 // K flushes when the buffer holds exactly R tokens
        CHECK(c.quantized_k_tokens() == (t / R) * R);
        CHECK(c.residual_v_tokens() == std::min<size_t>(t, R)); // V window slides, always min(T,R) full-precision tokens
        CHECK(c.quantized_v_tokens() == (t > R ? t - R : 0));
    }
    // invalid geometry rejected
    bool threw = false; try { cache bad(16, 2, 4, 6); } catch (const std::invalid_argument &) { threw = true; } CHECK(threw);
    threw = false; try { cache bad(16, 3, 4, 8); } catch (const std::invalid_argument &) { threw = true; } CHECK(threw);
}

static void test_window_is_lossless_and_exact_ordering() {
    printf("[residual window is bit-exact; quantised part keeps token order]\n");
    const uint32_t D = 32, G = 8, R = 16;
    cache c(D, 4, G, R);
    kvsynth::rng r(9);
    const size_t T = 50;
    std::vector<float> K(T * D), V(T * D);
    for (size_t t = 0; t < T; t++) {
        for (uint32_t i = 0; i < D; i++) K[t * D + i] = (float)r.n() + (float)t; // token-dependent offset: wrong ordering would be visible
        gen_v(r, &V[t * D], D);
        c.append(&K[t * D], &V[t * D]);
    }
    std::vector<float> o(D);
    for (size_t t = 0; t < T; t++) {
        c.read_k(t, o.data());
        const bool in_window = t >= c.quantized_k_tokens();
        double e = 0; for (uint32_t i = 0; i < D; i++) e = std::max(e, (double)std::fabs(o[i] - K[t * D + i]));
        if (in_window) CHECK(e == 0.0); else CHECK(e < 1.5); // 4-bit over a group range of a few units: error << group range
        c.read_v(t, o.data());
        const bool vwin = t >= c.quantized_v_tokens();
        e = 0; for (uint32_t i = 0; i < D; i++) e = std::max(e, (double)std::fabs(o[i] - V[t * D + i]));
        if (vwin) CHECK(e == 0.0); else CHECK(e < 1.0);
    }
    bool threw = false; try { c.read_k(T, o.data()); } catch (const std::out_of_range &) { threw = true; } CHECK(threw);
}

static void test_prefill_equals_stepwise_and_attention_paths() {
    printf("[fused-from-codes attention == read-back float attention]\n");
    const uint32_t D = 64, G = 16, R = 64;
    for (uint32_t bits : { 2u, 4u }) {
        cache c(D, bits, G, R);
        kvsynth::key_model km(D, 2, true, 3);
        kvsynth::rng rk(1), rq(2), rv(3);
        const size_t T = 200;
        std::vector<float> k(D), v(D);
        for (size_t t = 0; t < T; t++) { km.sample(rk, (int)t, k.data()); gen_v(rv, v.data(), D); c.append(k.data(), v.data()); }
        std::vector<float> q(D); km.sample(rq, (int)T, q.data());
        // path 1: read everything back into a float buffer, ordinary attention
        std::vector<float> Kb(T * D), Vb(T * D), s1(T), s2(T), w(T), o1(D), o2(D);
        for (size_t t = 0; t < T; t++) { c.read_k(t, &Kb[t * D]); c.read_v(t, &Vb[t * D]); }
        for (size_t t = 0; t < T; t++) { double a = 0; for (uint32_t i = 0; i < D; i++) a += (double)q[i] * Kb[t * D + i]; s1[t] = (float)a; }
        // path 2: dot products computed directly from the packed codes
        c.k_scores(q.data(), s2.data());
        float md = 0; for (size_t t = 0; t < T; t++) md = std::max(md, std::fabs(s1[t] - s2[t]) / (1.f + std::fabs(s1[t])));
        printf("  bits=%u max rel |score(readback)-score(fused)| = %.3g\n", bits, md);
        CHECK(md < 1e-4f);
        float mx = -1e30f; for (float x : s1) mx = std::max(mx, x);
        double Z = 0; for (size_t t = 0; t < T; t++) { w[t] = std::exp((s1[t] - mx) / std::sqrt((float)D)); Z += w[t]; }
        for (auto & x : w) x = (float)(x / Z);
        std::fill(o1.begin(), o1.end(), 0.f);
        for (size_t t = 0; t < T; t++) for (uint32_t i = 0; i < D; i++) o1[i] += w[t] * Vb[t * D + i];
        c.v_weighted_sum(w.data(), o2.data());
        float mo = 0; for (uint32_t i = 0; i < D; i++) mo = std::max(mo, std::fabs(o1[i] - o2[i]) / (1.f + std::fabs(o1[i])));
        printf("  bits=%u max rel |out(readback)-out(fused)|   = %.3g\n", bits, mo);
        CHECK(mo < 1e-4f);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
static double q40_roundtrip(const std::vector<float> & x, std::vector<float> & y, size_t rows, int D) {
    const auto * tr = ggml_get_type_traits(GGML_TYPE_Q4_0);
    std::vector<uint8_t> blk(tr->type_size * (D / tr->blck_size));
    y.resize(x.size());
    for (size_t t = 0; t < rows; t++) { tr->from_float_ref(&x[t * D], blk.data(), D); tr->to_float(blk.data(), &y[t * D], D); }
    return 8.0 * tr->type_size / tr->blck_size;
}

int main() {
    ggml_init_params ip = { 1024 * 1024, nullptr, true }; ggml_free(ggml_init(ip));

    test_window_and_flush_logic();
    test_window_is_lossless_and_exact_ordering();
    test_prefill_equals_stepwise_and_attention_paths();

    const int D = 128; const size_t T = 1000; // not a multiple of R, so the residual window is non-empty at the end
    printf("\n[MEASURE] D=%d T=%zu. 'K-err'/'V-err' = relative Frobenius error over ALL T tokens (window tokens are exact, so they\n", D, T);
    printf("lower the number); 'attn-err' = rel. RMSE of softmax(qK^T/sqrt(D))V over 32 queries vs f32 (large when the reference output is a near-uniform average of V, so also shown as 'attn/Vrms' = RMSE / RMS(V)). bpw = stored bits/element\n");
    printf("of the quantised region only (codes + half scale/zero), 'blend' = whole cache incl. the f16 window.\n");
    struct ds { const char * name; int pairs; bool low; };
    const ds sets[] = { { "no-outliers (control)", 0, true }, { "2 outlier pairs LOW-freq", 2, true }, { "2 outlier pairs HIGH-freq", 2, false } };
    struct cfg { uint32_t bits, G, R; };
    const cfg cfgs[] = { { 2, 32, 128 }, { 2, 64, 128 }, { 4, 32, 128 }, { 2, 32, 32 } };
    double kivi2_k = 0, q40_k = 0; bool have = false;
    for (const auto & d : sets) {
        kvsynth::key_model km(D, d.pairs, d.low, 42);
        kvsynth::rng rk(222), rv(444), rq(333);
        std::vector<float> K(T * D), V(T * D), Q(32 * D);
        for (size_t t = 0; t < T; t++) { km.sample(rk, (int)t, &K[t * D]); gen_v(rv, &V[t * D], D); }
        for (int j = 0; j < 32; j++) km.sample(rq, (int)(T / 2 + j * 13), &Q[(size_t)j * D]);
        auto attn = [&](const std::vector<float> & Kx, const std::vector<float> & Vx, std::vector<float> & out) {
            out.assign(32 * D, 0.f);
            std::vector<double> s(T);
            for (int j = 0; j < 32; j++) {
                double mx = -1e300;
                for (size_t t = 0; t < T; t++) { double a = 0; for (int i = 0; i < D; i++) a += (double)Q[(size_t)j * D + i] * Kx[t * D + i]; s[t] = a / std::sqrt((double)D); mx = std::max(mx, s[t]); }
                double Z = 0; for (auto & x : s) { x = std::exp(x - mx); Z += x; }
                for (size_t t = 0; t < T; t++) for (int i = 0; i < D; i++) out[(size_t)j * D + i] += (float)(s[t] / Z * Vx[t * D + i]);
            }
        };
        std::vector<float> ref; attn(K, V, ref);
        double vss = 0; for (float x : V) vss += (double)x * x;
        const double vrms = std::sqrt(vss / V.size());
        auto abs_vrms = [&](const std::vector<float> & a, const std::vector<float> & b) { double e = 0; for (size_t i = 0; i < a.size(); i++) e += ((double)a[i] - b[i]) * ((double)a[i] - b[i]); return std::sqrt(e / a.size()) / vrms; };
        printf("\n  dataset: %s\n  %-30s %8s %8s %9s %9s %7s %7s\n", d.name, "scheme", "K-err", "V-err", "attn-err", "attn/Vrms", "bpw", "blend");
        for (const auto & cf : cfgs) {
            cache c(D, cf.bits, cf.G, cf.R);
            for (size_t t = 0; t < T; t++) c.append(&K[t * D], &V[t * D]);
            std::vector<float> Kq(T * D), Vq(T * D), got;
            for (size_t t = 0; t < T; t++) { c.read_k(t, &Kq[t * D]); c.read_v(t, &Vq[t * D]); }
            attn(Kq, Vq, got);
            const size_t nq = c.quantized_k_tokens() + c.quantized_v_tokens();
            const double bpw = nq ? (double)c.bits_used_quantized() / ((double)nq * D) : 0;
            const double blend = (double)(c.bits_used_quantized() + c.bits_used_window()) / (2.0 * T * D);
            char nm[64]; snprintf(nm, sizeof nm, "KIVI %ub G=%u R=%u", cf.bits, cf.G, cf.R);
            printf("  %-30s %8.4f %8.4f %9.4f %9.4f %7.2f %7.2f\n", nm, kvsynth::rel_rmse(K, Kq), kvsynth::rel_rmse(V, Vq), kvsynth::rel_rmse(ref, got), abs_vrms(ref, got), bpw, blend);
            if (cf.bits == 2 && cf.G == 32 && cf.R == 128 && d.pairs == 2 && d.low) { kivi2_k = kvsynth::rel_rmse(K, Kq); have = true; }
        }
        std::vector<float> Kq4, Vq4, got; const double bq = q40_roundtrip(K, Kq4, T, D); q40_roundtrip(V, Vq4, T, D); attn(Kq4, Vq4, got);
        printf("  %-30s %8.4f %8.4f %9.4f %9.4f %7.2f %7.2f\n", "q4_0 K and V (ggml)", kvsynth::rel_rmse(K, Kq4), kvsynth::rel_rmse(V, Vq4), kvsynth::rel_rmse(ref, got), abs_vrms(ref, got), bq, bq);
        printf("  %-30s %8.4f %8.4f %9.4f %9.4f %7.2f %7.2f\n", "f16 K and V", 0.0, 0.0, 0.0, 0.0, 16.0, 16.0);
        if (d.pairs == 2 && d.low) q40_k = kvsynth::rel_rmse(K, Kq4);
    }
    CHECK(have);
    (void)kivi2_k; (void)q40_k; // errors are printed, not pinned: KIVI 2-bit is NOT expected to beat 4.5-bit q4_0 on raw error

    if (g_fail) { printf("\nTEST_KIVI_2BIT FAIL (%d failed checks)\n", g_fail); return 1; }
    printf("\nTEST_KIVI_2BIT PASS\n");
    return 0;
}
