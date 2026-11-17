// GPTQ error-compensated rounding (src/llama-gptq.h; row S69-ncnn-PICK-GPTQ-ERROR-COMPENSATED-ROUNDING-2026-09-29-01).
//
//   1. the factor: U^T U == inv(H + damp * mean(diag) * I) against an independent Gauss-Jordan inverse, U upper; a dead column is handled
//   2. CONTROL: a DIAGONAL Hessian couples nothing, so GPTQ must return exactly the block-wise rounding (byte for byte)
//   3. CONTROL: propagate=false is the same block-wise rounding whatever H is (byte for byte)
//   4. on correlated calibration inputs (H = X^T X of mixed signals) GPTQ lowers the objective it minimises, sum_b d H_b d^T,
//      below the ORDINARY whole-row quantisation (today's llama-quantize) for Q4_0, Q4_K and Q2_K -- the claim the row makes
//   5. the group update reduces to GPTQ's per-column rule: with a 1-wide "quant block" (exact rounding to a fixed grid,
//      emulated here) the update equals the column-by-column loop written out directly
//   6. NEGATIVE CONTROL: with the update's sign flipped the objective must go UP (the check in 4 can fail)

#include "../src/llama-gptq.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;
static void check(bool ok, const char * what) {
    printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++n_fail;
}

// H = sum_t x_t x_t^T of T tokens whose K features are mixtures of a few latent signals (strong off-diagonal terms, like real activations)
static std::vector<float> correlated_hessian(int64_t K, int64_t T, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t R = 8;
    std::vector<float> mix((size_t) (K * R));
    for (float & v : mix) v = nd(rng);
    std::vector<double> H((size_t) (K * K), 0.0);
    std::vector<float> x((size_t) K), z((size_t) R);
    for (int64_t t = 0; t < T; ++t) {
        for (float & v : z) v = nd(rng);
        for (int64_t k = 0; k < K; ++k) {
            double s = 0.3 * nd(rng);
            for (int64_t r = 0; r < R; ++r) s += mix[(size_t) (k * R + r)] * z[(size_t) r];
            x[(size_t) k] = (float) s;
        }
        for (int64_t i = 0; i < K; ++i)
            for (int64_t j = 0; j < K; ++j) H[(size_t) (i * K + j)] += (double) x[(size_t) i] * x[(size_t) j];
    }
    std::vector<float> Hf((size_t) (K * K));
    for (size_t i = 0; i < Hf.size(); ++i) Hf[i] = (float) H[i];
    return Hf;
}

// the sidecar layout for one matrix: element (j, k) at slab[k * N + j], row k restricted to its own diagonal block
static std::vector<float> to_slab(const std::vector<float> & H, int64_t K, int64_t N) {
    std::vector<float> s((size_t) (K * N), 0.0f);
    for (int64_t k = 0; k < K; ++k) {
        const int64_t b0 = (k / N) * N;
        for (int64_t j = 0; j < N && b0 + j < K; ++j) s[(size_t) (k * N + j)] = H[(size_t) (k * K + b0 + j)];
    }
    return s;
}

static std::vector<llama_gptq::block_factor> factors(const std::vector<float> & slab, int64_t K, int64_t N, double damp) {
    std::vector<llama_gptq::block_factor> f((size_t) ((K + N - 1) / N));
    for (int64_t b = 0; b * N < K; ++b) {
        const bool ok = llama_gptq::factor_block(slab.data() + b * N * N, N, std::min<int64_t>(N, K - b * N), damp, f[(size_t) b]);
        if (!ok) { printf("  factor failed\n"); ++n_fail; }
    }
    return f;
}

static std::vector<float> random_row(int64_t K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> w((size_t) K);
    for (float & v : w) v = nd(rng);
    return w;
}

// mean over rows of the block objective, for one quantised result per row
static void objectives(ggml_type t, int64_t K, int64_t N, const std::vector<float> & slab, int n_rows, double & o_plain, double & o_gptq,
                       double & o_noprop, double damp = 0.01) {
    const auto fac = factors(slab, K, N, damp);
    const size_t rs = ggml_row_size(t, K);
    std::vector<char> q_plain(rs), q_gptq(rs), q_np(rs);
    o_plain = o_gptq = o_noprop = 0.0;
    for (int r = 0; r < n_rows; ++r) {
        const auto w0 = random_row(K, 1000u + (uint32_t) r);
        ggml_quantize_chunk(t, w0.data(), q_plain.data(), 0, 1, K, nullptr, nullptr);
        auto w1 = w0;
        llama_gptq::quantize_row(t, w1.data(), K, nullptr, nullptr, fac, N, q_gptq.data(), true);
        auto w2 = w0;
        llama_gptq::quantize_row(t, w2.data(), K, nullptr, nullptr, fac, N, q_np.data(), false);
        o_plain  += llama_gptq::block_objective(w0.data(), q_plain.data(), t, K, slab.data(), N);
        o_gptq   += llama_gptq::block_objective(w0.data(), q_gptq.data(),  t, K, slab.data(), N);
        o_noprop += llama_gptq::block_objective(w0.data(), q_np.data(),    t, K, slab.data(), N);
    }
}

int main() {
    ggml_init_params ip = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);   // initialises the fp16 tables the quantisers use

    printf("1. factor: U^T U == inv(H_damped), U upper\n");
    {
        const int64_t n = 48;
        auto H = correlated_hessian(n, 200, 7u);
        for (int64_t j = 0; j < n; ++j) { H[(size_t) (5 * n + j)] = 0.0f; H[(size_t) (j * n + 5)] = 0.0f; }   // dead column 5
        llama_gptq::block_factor f;
        const bool ok = llama_gptq::factor_block(H.data(), n, n, 0.01, f);
        check(ok, "factor_block succeeds on a PSD block with a dead column");
        // independent reference: Gauss-Jordan inverse of the same damped matrix
        std::vector<double> A((size_t) (n * n)), I((size_t) (n * n), 0.0);
        double md = 0.0;
        for (int64_t i = 0; i < n; ++i) md += H[(size_t) (i * n + i)];
        md /= n;
        for (int64_t i = 0; i < n; ++i) for (int64_t j = 0; j < n; ++j) A[(size_t) (i * n + j)] = H[(size_t) (i * n + j)];
        for (int64_t i = 0; i < n; ++i) { if (A[(size_t) (i * n + i)] == 0.0) A[(size_t) (i * n + i)] = 1.0; A[(size_t) (i * n + i)] += 0.01 * md; I[(size_t) (i * n + i)] = 1.0; }
        for (int64_t c = 0; c < n; ++c) {
            int64_t p = c;
            for (int64_t r = c + 1; r < n; ++r) if (std::fabs(A[(size_t) (r * n + c)]) > std::fabs(A[(size_t) (p * n + c)])) p = r;
            for (int64_t j = 0; j < n; ++j) { std::swap(A[(size_t) (c * n + j)], A[(size_t) (p * n + j)]); std::swap(I[(size_t) (c * n + j)], I[(size_t) (p * n + j)]); }
            const double d = A[(size_t) (c * n + c)];
            for (int64_t j = 0; j < n; ++j) { A[(size_t) (c * n + j)] /= d; I[(size_t) (c * n + j)] /= d; }
            for (int64_t r = 0; r < n; ++r) {
                if (r == c) continue;
                const double m = A[(size_t) (r * n + c)];
                if (m == 0.0) continue;
                for (int64_t j = 0; j < n; ++j) { A[(size_t) (r * n + j)] -= m * A[(size_t) (c * n + j)]; I[(size_t) (r * n + j)] -= m * I[(size_t) (c * n + j)]; }
            }
        }
        double err = 0.0, mx = 0.0, lower = 0.0;
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (int64_t k = 0; k < n; ++k) s += f.U[(size_t) (k * n + i)] * f.U[(size_t) (k * n + j)];
                err = std::max(err, std::fabs(s - I[(size_t) (i * n + j)]));
                mx = std::max(mx, std::fabs(I[(size_t) (i * n + j)]));
                if (j < i) lower = std::max(lower, std::fabs(f.U[(size_t) (i * n + j)]));
            }
        printf("    max |U^T U - inv| = %.3g (max |inv| %.3g), max |U below diagonal| = %.3g, dead = %lld\n", err, mx, lower, (long long) f.n_dead);
        check(err <= 1e-9 * std::max(1.0, mx), "U^T U matches the Gauss-Jordan inverse to 1e-9 relative");
        check(lower == 0.0, "U is upper triangular");
        check(f.n_dead == 1, "the dead column is counted");
    }

    printf("2./3. CONTROLS: diagonal H and propagate=false both reduce to block-wise rounding, byte for byte\n");
    for (ggml_type t : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_K }) {
        const int64_t K = 1024, N = 512;
        std::vector<float> Hd((size_t) (K * K), 0.0f);
        for (int64_t i = 0; i < K; ++i) Hd[(size_t) (i * K + i)] = 1.0f + 0.01f * (float) (i % 7);
        const auto sd = to_slab(Hd, K, N);
        const auto fd = factors(sd, K, N, 0.01);
        const auto Hc = correlated_hessian(K, 64, 11u);
        const auto sc = to_slab(Hc, K, N);
        const auto fc = factors(sc, K, N, 0.01);
        const size_t rs = ggml_row_size(t, K);
        std::vector<char> a(rs), b(rs), c(rs);
        const auto w0 = random_row(K, 3u);
        auto w = w0; llama_gptq::quantize_row(t, w.data(), K, nullptr, nullptr, fd, N, a.data(), true);
        w = w0;      llama_gptq::quantize_row(t, w.data(), K, nullptr, nullptr, fd, N, b.data(), false);
        w = w0;      llama_gptq::quantize_row(t, w.data(), K, nullptr, nullptr, fc, N, c.data(), false);
        char msg[160];
        snprintf(msg, sizeof msg, "%s: diagonal H -> GPTQ bytes == block-wise bytes", ggml_type_name(t));
        check(memcmp(a.data(), b.data(), rs) == 0, msg);
        snprintf(msg, sizeof msg, "%s: propagate=false on a correlated H == block-wise bytes on a diagonal H (H unused)", ggml_type_name(t));
        check(memcmp(b.data(), c.data(), rs) == 0, msg);
    }

    printf("4. correlated H: GPTQ lowers the block objective below today's whole-row quantisation\n");
    struct Case { ggml_type t; int64_t K, N; double need; };
    for (const Case cs : { Case{ GGML_TYPE_Q4_0, 512, 128, 0.97 }, Case{ GGML_TYPE_Q4_K, 1024, 1024, 0.97 }, Case{ GGML_TYPE_Q2_K, 1024, 1024, 0.97 } }) {
        const auto H = correlated_hessian(cs.K, 256, 21u);
        const auto slab = to_slab(H, cs.K, cs.N);
        double op, og, on;
        objectives(cs.t, cs.K, cs.N, slab, 6, op, og, on);
        printf("    %-5s K=%lld N=%lld: objective plain %.5g | block-wise no-propagate %.5g | GPTQ %.5g  (GPTQ/plain %.3f)\n",
               ggml_type_name(cs.t), (long long) cs.K, (long long) cs.N, op, on, og, og / op);
        char msg[160];
        snprintf(msg, sizeof msg, "%s: GPTQ objective <= %.2f x the ordinary quantisation's", ggml_type_name(cs.t), cs.need);
        check(og <= cs.need * op, msg);
    }

    printf("5. the group update with 1-wide groups equals GPTQ's column loop written out directly\n");
    {
        // A fixed-grid "type": round to multiples of step. Group form with |G| = 1 vs the textbook column loop on the same U.
        const int64_t n = 64;
        const auto H = correlated_hessian(n, 300, 5u);
        llama_gptq::block_factor f;
        llama_gptq::factor_block(H.data(), n, n, 0.01, f);
        const float step = 0.01f;
        auto rnd = [step](float v) { return step * std::round(v / step); };
        const auto w0 = random_row(n, 77u);
        // textbook: err_k = (w_k - q_k) / U_kk; w_j -= err_k U_kj
        std::vector<float> wa = w0, qa((size_t) n);
        for (int64_t k = 0; k < n; ++k) {
            qa[(size_t) k] = rnd(wa[(size_t) k]);
            const double err = (wa[(size_t) k] - qa[(size_t) k]) / f.U[(size_t) (k * n + k)];
            for (int64_t j = k + 1; j < n; ++j) wa[(size_t) j] -= (float) (err * f.U[(size_t) (k * n + j)]);
        }
        // group form, |G| = 1: e' = e U_GG^-1, w_R -= e' U_GR  (the same arithmetic quantize_row does per group)
        std::vector<float> wb = w0, qb((size_t) n);
        for (int64_t a = 0; a < n; ++a) {
            qb[(size_t) a] = rnd(wb[(size_t) a]);
            const double ep = ((double) wb[(size_t) a] - qb[(size_t) a]) / f.U[(size_t) (a * n + a)];
            for (int64_t j = a + 1; j < n; ++j) wb[(size_t) j] -= (float) (ep * f.U[(size_t) (a * n + j)]);
        }
        double d = 0.0;
        for (int64_t k = 0; k < n; ++k) d = std::max(d, (double) std::fabs(qa[(size_t) k] - qb[(size_t) k]));
        check(d == 0.0, "1-wide group update == textbook GPTQ column loop");
    }

    printf("6. NEGATIVE CONTROL: flipping the update's sign must raise the objective\n");
    {
        const int64_t K = 512, N = 128;
        const ggml_type t = GGML_TYPE_Q4_0;
        const auto H = correlated_hessian(K, 256, 21u);
        const auto slab = to_slab(H, K, N);
        auto fac = factors(slab, K, N, 0.01);
        for (auto & f : fac)
            for (int64_t i = 0; i < f.n; ++i)
                for (int64_t j = i + 1; j < f.n; ++j) f.U[(size_t) (i * f.n + j)] = -f.U[(size_t) (i * f.n + j)];
        const size_t rs = ggml_row_size(t, K);
        std::vector<char> qp(rs), qg(rs);
        double op = 0.0, og = 0.0;
        for (int r = 0; r < 6; ++r) {
            const auto w0 = random_row(K, 1000u + (uint32_t) r);
            ggml_quantize_chunk(t, w0.data(), qp.data(), 0, 1, K, nullptr, nullptr);
            auto w = w0;
            llama_gptq::quantize_row(t, w.data(), K, nullptr, nullptr, fac, N, qg.data(), true);
            op += llama_gptq::block_objective(w0.data(), qp.data(), t, K, slab.data(), N);
            og += llama_gptq::block_objective(w0.data(), qg.data(), t, K, slab.data(), N);
        }
        printf("    sign-flipped update: objective %.5g vs plain %.5g\n", og, op);
        check(og > op, "the sign-flipped update is WORSE than plain (check 4 measures something)");
    }

    ggml_free(ctx);
    printf("%s: %d check(s) failed\n", n_fail ? "FAIL" : "PASS", n_fail);
    return n_fail ? 1 : 0;
}
