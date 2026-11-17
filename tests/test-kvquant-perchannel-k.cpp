// Row S69-KVQUANT-PICK-STATIC-PERCHANNEL-K-POSTROPE-2026-09-29-01.
// CPU-only, model-free test of src/llama-kv-kvquant.h (static calibrated per-channel K quantisation) plus a MEASUREMENT of
// its reconstruction / q.k-logit error against the fork's real q4_0 (ggml traits, with and without a 128-wide Hadamard,
// the fork's attn_rot_k) on synthetic post-RoPE keys that carry channel outliers (data recipe: tests/kv-quant-synth.h).
#include "ggml.h"
#include "llama-kv-kvquant.h"
#include "kv-quant-synth.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

using namespace llama_kvq;

static void test_f16() {
    printf("[f16 helper]\n");
    const float v[] = { 0.f, 1.f, -1.f, 0.333333f, 123.456f, -0.0078125f, 65504.f, 6.1e-5f, 3e-6f };
    for (float x : v) {
        const float r = round_f16(x);
        const float tol = std::fabs(x) * 1.0f / 1024.f + 6e-8f; // half ulp ~2^-11 relative, absolute floor for subnormals
        CHECK(std::fabs(r - x) <= tol);
    }
    CHECK(f32_to_f16(1.0f) == 0x3C00);
    CHECK(f32_to_f16(-2.0f) == 0xC000);
    CHECK(f32_to_f16(1e9f) == 0x7C00);
}

static void test_calibrator_quantiles() {
    printf("[calibrator quantiles]\n");
    // channel 0: values 0..1000 (n=1001). dense_frac 0.98 -> tail 1% -> a = round(0.01*1000)=10, b=990.
    calibrator cal(1, 1, 4, 4, 0.98f);
    for (int i = 1000; i >= 0; i--) { float r[4] = { (float)i, (float)(2 * i), -(float)i, 5.f }; cal.add_row(0, 0, r); }
    table t = cal.finish();
    CHECK(t.valid());
    CHECK(t.lo[0] == 10.f && t.hi[0] == 990.f);
    CHECK(t.lo[1] == 20.f && t.hi[1] == 1980.f);
    CHECK(t.lo[2] == -990.f && t.hi[2] == -10.f);
    CHECK(t.lo[3] == 5.f && t.hi[3] == 5.f);
}

static void test_roundtrip_bound_and_outliers() {
    printf("[quantise bound + outliers]\n");
    const uint32_t D = 16;
    for (uint32_t bits : { 2u, 4u, 8u }) {
        table t; t.n_layer = 1; t.n_kv_head = 1; t.head_dim = D; t.bits = bits; t.dense_frac = 0.99f;
        t.lo.assign(D, -1.f); t.hi.assign(D, 1.f);
        CHECK(t.valid());
        kvsynth::rng r(7 + bits);
        float k[D], o[D];
        for (int rep = 0; rep < 50; rep++) {
            for (uint32_t i = 0; i < D; i++) k[i] = (float)(r.u() * 2 - 1);
            k[3] = 7.5f; k[9] = -4.25f; // outliers
            row q = quantize_row(t, 0, 0, k);
            CHECK(q.outl.size() == 2);
            dequantize_row(t, 0, 0, q, o);
            const float step = 2.f / (float)((1u << bits) - 1);
            for (uint32_t i = 0; i < D; i++) {
                if (i == 3 || i == 9) CHECK(std::fabs(o[i] - k[i]) <= 1e-3f * std::fabs(k[i]));
                else                  CHECK(std::fabs(o[i] - k[i]) <= 0.5f * step + 1e-6f);
            }
            float qv[D]; for (uint32_t i = 0; i < D; i++) qv[i] = (float)r.n();
            float ref = 0; for (uint32_t i = 0; i < D; i++) ref += qv[i] * o[i];
            CHECK(std::fabs(dot_row(t, 0, 0, q, qv) - ref) <= 1e-4f * (1.f + std::fabs(ref)));
        }
    }
}

static void test_table_file() {
    printf("[table file format]\n");
    table t; t.n_layer = 2; t.n_kv_head = 3; t.head_dim = 8; t.bits = 4; t.dense_frac = 0.99f;
    const size_t n = 2 * 3 * 8;
    for (size_t i = 0; i < n; i++) { t.lo.push_back(-(float)i - 0.5f); t.hi.push_back((float)i * 1.25f + 0.1f); }
    auto buf = t.serialize();
    CHECK(buf.size() == 28 + n * 8);
    CHECK(buf[0] == 'K' && buf[1] == 'V' && buf[2] == 'Q' && buf[3] == 'T');
    table u; CHECK(u.load(buf.data(), buf.size()));
    CHECK(u.n_layer == 2 && u.n_kv_head == 3 && u.head_dim == 8 && u.bits == 4 && u.dense_frac == t.dense_frac);
    CHECK(u.lo == t.lo && u.hi == t.hi);
    CHECK(u.idx(1, 2, 7) == n - 1);
    table v;
    CHECK(!v.load(buf.data(), buf.size() - 1));                           // truncated
    { auto b = buf; b.push_back(0); CHECK(!v.load(b.data(), b.size())); }  // trailing byte
    { auto b = buf; b[0] = 'X'; CHECK(!v.load(b.data(), b.size())); }      // magic
    { auto b = buf; b[4] = 2; CHECK(!v.load(b.data(), b.size())); }        // version
    { auto b = buf; b[20] = 3; CHECK(!v.load(b.data(), b.size())); }       // bits=3 unsupported
    { auto b = buf; b[8] = 0; b[9] = 0; b[10] = 0; b[11] = 0; CHECK(!v.load(b.data(), b.size())); } // n_layer=0
    CHECK(!v.load(buf.data(), 10));
}

static void test_multi_layer_head_independent() {
    printf("[per layer/head tables are independent]\n");
    const uint32_t D = 8;
    calibrator cal(2, 2, D, 4, 1.0f);
    for (int s = 0; s < 64; s++) for (uint32_t l = 0; l < 2; l++) for (uint32_t h = 0; h < 2; h++) {
        float r[D]; for (uint32_t c = 0; c < D; c++) r[c] = (float)(1 + 10 * l + 100 * h) * (float)((s % 9) - 4) * (c + 1);
        cal.add_row(l, h, r);
    }
    table t = cal.finish();
    for (uint32_t l = 0; l < 2; l++) for (uint32_t h = 0; h < 2; h++) {
        const float sc = (float)(1 + 10 * l + 100 * h);
        for (uint32_t c = 0; c < D; c++) CHECK(t.lo[t.idx(l, h, c)] == -4.f * sc * (c + 1) && t.hi[t.idx(l, h, c)] == 4.f * sc * (c + 1));
    }
    // a store bound to (1,1) must use (1,1)'s range: value 4*sc*(c+1) is exactly hi there -> reconstructs exactly, no outlier
    k_store st(t, 1, 1);
    float r[D]; for (uint32_t c = 0; c < D; c++) r[c] = 4.f * 111.f * (c + 1);
    st.append(r);
    float o[D]; st.read(0, o);
    for (uint32_t c = 0; c < D; c++) CHECK(std::fabs(o[c] - r[c]) <= 1e-3f * r[c]);
}

// ---------------------------------------------------------------------------------------------------------------------
struct metrics { double k_err, logit_err, bpw; };

static metrics eval_static(const kvsynth::key_model & m, int D, uint32_t bits, float frac, int n_cal, int n_test, bool & ok_dot) {
    kvsynth::rng rc(111), rt(222), rq(333);
    calibrator cal(1, 1, D, bits, frac);
    std::vector<float> row(D);
    for (int i = 0; i < n_cal; i++) { m.sample(rc, (int)(rc.u() * 4096), row.data()); cal.add_row(0, 0, row.data()); }
    table t = cal.finish();
    // table round-trips through the file format before use, so the measurement exercises the on-disk path
    auto buf = t.serialize(); table t2; ok_dot = t2.load(buf.data(), buf.size());
    k_store st(t2, 0, 0);
    std::vector<float> K((size_t)n_test * D), Kq((size_t)n_test * D), Q((size_t)n_test * D);
    for (int i = 0; i < n_test; i++) { m.sample(rt, i, &K[(size_t)i * D]); st.append(&K[(size_t)i * D]); st.read(i, &Kq[(size_t)i * D]); m.sample(rq, i, &Q[(size_t)i * D]); }
    std::vector<float> lr, lg, sc(n_test);
    for (int j = 0; j < 64; j++) {
        const float * q = &Q[(size_t)(j * 7 % n_test) * D];
        st.scores(q, sc.data());
        for (int i = 0; i < n_test; i++) {
            float ref = 0; for (int c = 0; c < D; c++) ref += q[c] * K[(size_t)i * D + c];
            lr.push_back(ref); lg.push_back(sc[i]);
            float dq = 0; for (int c = 0; c < D; c++) dq += q[c] * Kq[(size_t)i * D + c];
            if (std::fabs(dq - sc[i]) > 1e-3f * (1.f + std::fabs(dq))) ok_dot = false;
        }
    }
    return { kvsynth::rel_rmse(K, Kq), kvsynth::rel_rmse(lr, lg), (double)st.bits_used() / ((double)n_test * D) };
}

static metrics eval_q40(const kvsynth::key_model & m, int D, bool hadamard, int n_test) {
    const auto * tr = ggml_get_type_traits(GGML_TYPE_Q4_0);
    kvsynth::rng rt(222), rq(333);
    std::vector<float> K((size_t)n_test * D), Kq((size_t)n_test * D), Q((size_t)n_test * D);
    std::vector<uint8_t> blk(tr->type_size * (D / tr->blck_size));
    for (int i = 0; i < n_test; i++) {
        float * k = &K[(size_t)i * D];
        m.sample(rt, i, k); m.sample(rq, i, &Q[(size_t)i * D]);
        std::vector<float> w(k, k + D);
        if (hadamard) kvsynth::wht(w.data(), D);
        tr->from_float_ref(w.data(), blk.data(), D);
        tr->to_float(blk.data(), &Kq[(size_t)i * D], D);
        if (hadamard) kvsynth::wht(&Kq[(size_t)i * D], D); // orthonormal and self-inverse
    }
    std::vector<float> lr, lg;
    for (int j = 0; j < 64; j++) {
        const float * q = &Q[(size_t)(j * 7 % n_test) * D];
        for (int i = 0; i < n_test; i++) {
            float a = 0, b = 0; for (int c = 0; c < D; c++) { a += q[c] * K[(size_t)i * D + c]; b += q[c] * Kq[(size_t)i * D + c]; }
            lr.push_back(a); lg.push_back(b);
        }
    }
    return { kvsynth::rel_rmse(K, Kq), kvsynth::rel_rmse(lr, lg), 8.0 * tr->type_size / tr->blck_size };
}

struct ds { const char * name; int pairs; bool low; };

int main() {
    ggml_init_params ip = { 1024 * 1024, nullptr, true }; ggml_free(ggml_init(ip)); // fp16 tables for q4_0 dequant

    test_f16();
    test_calibrator_quantiles();
    test_roundtrip_bound_and_outliers();
    test_table_file();
    test_multi_layer_head_independent();

    const int D = 128, NCAL = 4096, NTEST = 1024;
    printf("\n[MEASURE] D=%d  calib rows=%d (random pos in [0,4096))  test rows=%d (pos 0..%d, independent seed)\n", D, NCAL, NTEST, NTEST - 1);
    printf("metric 'K' = ||K-K^||_F/||K||_F ; metric 'logit' = RMS(q.K^ - q.K)/RMS(q.K) over 64 queries x %d keys\n", NTEST);
    printf("bpw = stored bits per K element (codes + 32 bits per sparse outlier; static table is per-model, 0 per token)\n");
    const ds sets[] = { { "no-outliers (control)", 0, true }, { "2 outlier pairs, LOW-freq (stable)", 2, true }, { "2 outlier pairs, HIGH-freq (oscillating)", 2, false } };
    for (const auto & d : sets) {
        kvsynth::key_model m(D, d.pairs, d.low, 42);
        printf("\n  dataset: %s\n  %-34s %8s %9s %6s\n", d.name, "scheme", "K-err", "logit-err", "bpw");
        bool ok1, ok2, ok3, ok4;
        metrics a = eval_static(m, D, 4, 1.00f, NCAL, NTEST, ok1);
        metrics b = eval_static(m, D, 4, 0.99f, NCAL, NTEST, ok2);
        metrics c = eval_static(m, D, 4, 0.985f, NCAL, NTEST, ok3);
        metrics e = eval_static(m, D, 2, 0.99f, NCAL, NTEST, ok4);
        metrics q = eval_q40(m, D, false, NTEST), qh = eval_q40(m, D, true, NTEST);
        CHECK(ok1 && ok2 && ok3 && ok4);
        printf("  %-34s %8.4f %9.4f %6.2f\n", "static per-channel 4b, minmax", a.k_err, a.logit_err, a.bpw);
        printf("  %-34s %8.4f %9.4f %6.2f\n", "static per-channel 4b, 99% dense", b.k_err, b.logit_err, b.bpw);
        printf("  %-34s %8.4f %9.4f %6.2f\n", "static per-channel 4b, 98.5% dense", c.k_err, c.logit_err, c.bpw);
        printf("  %-34s %8.4f %9.4f %6.2f\n", "static per-channel 2b, 99% dense", e.k_err, e.logit_err, e.bpw);
        printf("  %-34s %8.4f %9.4f %6.2f\n", "q4_0 (ggml)", q.k_err, q.logit_err, q.bpw);
        printf("  %-34s %8.4f %9.4f %6.2f\n", "q4_0 + Hadamard128 (fork attn_rot_k)", qh.k_err, qh.logit_err, qh.bpw);
        printf("  VERDICT (pc4 99%% dense @%.2f bpw vs q4_0 @%.2f bpw): K-err %s q4_0, logit-err %s q4_0; vs q4_0+Hadamard: K-err %s, logit-err %s\n",
               b.bpw, q.bpw, b.k_err < q.k_err ? "BEATS" : "loses to", b.logit_err < q.logit_err ? "BEATS" : "loses to",
               b.k_err < qh.k_err ? "beats" : "loses", b.logit_err < qh.logit_err ? "beats" : "loses");
        if (d.pairs == 2) { CHECK(b.bpw < q.bpw); CHECK(b.k_err < q.k_err); } // established by the run above for both outlier datasets
    }

    if (g_fail) { printf("\nTEST_KVQUANT_PERCHANNEL_K FAIL (%d failed checks)\n", g_fail); return 1; }
    printf("\nTEST_KVQUANT_PERCHANNEL_K PASS\n");
    return 0;
}
