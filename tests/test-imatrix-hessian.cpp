// llama-imatrix --hessian-block: block-diagonal X^T X accumulation, sidecar file round trip and CLI bounds.
// Row S69-F1-PICK-IMATRIX-BLOCK-HESSIAN-OUTPUT-2026-10-06-01. CPU only, no model, deterministic.
//
//   test-imatrix-hessian                     the test (registered in tests/CMakeLists.txt); exit 0 = pass
//   test-imatrix-hessian --baseline-diag-only runs the SAME checks against the accumulation llama-imatrix
//                                             had before this row (diagonal only, tools/imatrix/imatrix.cpp
//                                             :564-569: acc[j] += x[j]*x[j]); it must FAIL (fail-before).
//
// Checks: (1) a hand-computed 5-wide case with block 3 (one full and one partial block), exact;
// (2) seeded random inputs, K = 300, N = 128, over several calls and 1 vs 4 threads, against a naive double
// X^T X block; every block exactly symmetric and PSD (smallest eigenvalue >= -1e-4 * largest, Jacobi);
// the block diagonal equals the existing imatrix diagonal accumulation; 3 routed matrices (MUL_MAT_ID shape);
// (3) the sidecar save/load round trip is bit exact and the layout keys are present; a wrong version is refused;
// (4) negative controls: a corrupted reference, an asymmetric block and an indefinite block must each be caught;
// (5) --hessian-block / --hessian-max-mib parse for llama-imatrix and out-of-range values are refused.

#include "arg.h"
#include "common.h"
#include "imatrix-hessian.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
static bool g_diag_only = false;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

// the accumulation under test, or (--baseline-diag-only) the pre-row imatrix diagonal-only accumulation
static void accumulate(float * acc, int64_t K, int64_t N, const float * const * rows, int64_t n, int nt) {
    if (!g_diag_only) {
        imatrix_hessian_accumulate(acc, K, N, rows, n, nt);
        return;
    }
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t k = 0; k < K; ++k) {
            acc[k * N + (k % N)] += rows[r][k] * rows[r][k];
        }
    }
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static float frand() { // deterministic, roughly N(0,1)-sized
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    const double u = (double) (g_rng >> 11) * (1.0 / 9007199254740992.0);
    return (float) ((u - 0.5) * 3.4);
}

// naive double reference in the file layout
static std::vector<double> ref_blocks(const std::vector<std::vector<float>> & xs, int64_t K, int64_t N) {
    std::vector<double> H((size_t) (K * N), 0.0);
    for (const auto & x : xs) {
        for (int64_t k = 0; k < K; ++k) {
            const int64_t b0 = (k / N) * N;
            for (int64_t j = 0; j < N && b0 + j < K; ++j) {
                H[k * N + j] += (double) x[k] * (double) x[b0 + j];
            }
        }
    }
    return H;
}

// |got - ref| <= rtol * sqrt(ref_kk * ref_jj) + atol, per entry
static bool matches_ref(const float * got, const std::vector<double> & ref, int64_t K, int64_t N, double rtol, std::string * why) {
    for (int64_t k = 0; k < K; ++k) {
        const int64_t b0 = (k / N) * N;
        for (int64_t j = 0; j < N; ++j) {
            const double r = ref[k * N + j];
            double scale = 0.0;
            if (b0 + j < K) {
                scale = std::sqrt(std::fabs(ref[k * N + (k - b0)]) * std::fabs(ref[(b0 + j) * N + j]));
            }
            if (std::fabs((double) got[k * N + j] - r) > rtol * scale + 1e-6) {
                if (why) {
                    char buf[160];
                    snprintf(buf, sizeof(buf), "entry (k=%lld, j=%lld): got %.9g ref %.9g", (long long) k, (long long) j, got[k * N + j], r);
                    *why = buf;
                }
                return false;
            }
        }
    }
    return true;
}

static bool blocks_symmetric(const float * h, int64_t K, int64_t N) {
    for (int64_t k = 0; k < K; ++k) {
        const int64_t b0 = (k / N) * N;
        for (int64_t j = 0; j < N && b0 + j < K; ++j) {
            const float a = h[k * N + j];
            const float b = h[(b0 + j) * N + (k - b0)];
            if (std::fabs(a - b) > 1e-6f * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)))) {
                return false;
            }
        }
    }
    return true;
}

// cyclic Jacobi eigenvalues of a symmetric n x n matrix (double)
static std::vector<double> sym_eig(std::vector<double> a, int n) {
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) for (int q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
        if (off < 1e-22) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
            }
        }
    }
    std::vector<double> ev(n);
    for (int i = 0; i < n; ++i) ev[i] = a[i * n + i];
    return ev;
}

// every block PSD: smallest eigenvalue >= -1e-4 * largest |eigenvalue|
static bool blocks_psd(const float * h, int64_t K, int64_t N, double * worst_ratio = nullptr) {
    double worst = 0.0;
    for (int64_t b0 = 0; b0 < K; b0 += N) {
        const int nb = (int) std::min<int64_t>(N, K - b0);
        std::vector<double> a((size_t) nb * nb);
        for (int i = 0; i < nb; ++i) for (int j = 0; j < nb; ++j) a[i * nb + j] = h[(b0 + i) * N + j];
        const auto ev = sym_eig(a, nb);
        double mn = ev[0], mx = 0.0;
        for (double v : ev) { mn = std::min(mn, v); mx = std::max(mx, std::fabs(v)); }
        if (mx > 0.0) worst = std::min(worst, mn / mx);
        if (mn < -1e-4 * mx) {
            if (worst_ratio) *worst_ratio = mn / mx;
            return false;
        }
    }
    if (worst_ratio) *worst_ratio = worst;
    return true;
}

static std::vector<const float *> ptrs(const std::vector<std::vector<float>> & xs, size_t a, size_t b) {
    std::vector<const float *> p;
    for (size_t i = a; i < b; ++i) p.push_back(xs[i].data());
    return p;
}

static void test_hand_case() {
    const int fail0 = g_fail;
    // x1..x4, K = 5, N = 3: block 0 = columns 0..2, block 1 = columns 3..4 (partial)
    const std::vector<std::vector<float>> xs = {
        { 1,  2, 0,  1, -1 },
        { 0,  1, 3,  2,  0 },
        { 2, -1, 1,  0,  1 },
        { 1,  1, 1, -1,  2 },
    };
    const float expect[5][3] = { // hand sums, stored layout [N=3, K=5]
        {  6,  1,  3 },
        {  1,  7,  3 },
        {  3,  3, 11 },
        {  6, -3,  0 },
        { -3,  6,  0 },
    };
    std::vector<float> acc(5 * 3, 0.0f);
    auto p = ptrs(xs, 0, xs.size());
    accumulate(acc.data(), 5, 3, p.data(), (int64_t) p.size(), 1);
    for (int k = 0; k < 5; ++k) for (int j = 0; j < 3; ++j) {
        CHECK(acc[k * 3 + j] == expect[k][j], "hand case (k=%d, j=%d): got %g expected %g", k, j, acc[k * 3 + j], expect[k][j]);
    }
    // diagonal == the existing imatrix value (sum of x^2 per column)
    const float imat_diag[5] = { 6, 7, 11, 6, 6 };
    for (int k = 0; k < 5; ++k) {
        CHECK(acc[k * 3 + (k % 3)] == imat_diag[k], "hand case diagonal k=%d: %g vs imatrix %g", k, acc[k * 3 + (k % 3)], imat_diag[k]);
    }
    printf("hand case K=5 N=3: %s\n", g_fail > fail0 ? "FAILED" : "ok");
}

static void test_random(int64_t K, int64_t N, int64_t n_tok) {
    const int fail0 = g_fail;
    std::vector<std::vector<float>> xs((size_t) n_tok, std::vector<float>((size_t) K));
    for (auto & x : xs) {
        for (int64_t k = 0; k < K; ++k) x[k] = frand() * (1.0f + (float) (k % 7)); // uneven column scales
        x[(size_t) (K / 3)] = 0.0f; // a dead column must stay all-zero and still PSD
    }
    // correlated columns so the off-diagonal terms are far from zero
    for (auto & x : xs) for (int64_t k = 1; k < K; k += 2) x[k] += 0.75f * x[k - 1];

    const auto ref = ref_blocks(xs, K, N);

    std::vector<float> h1((size_t) (K * N), 0.0f), h4((size_t) (K * N), 0.0f);
    // several calls of uneven size, like ubatches
    const size_t nt = (size_t) n_tok;
    const size_t cuts[] = { 0, nt / 24, nt / 4, nt / 4 + 1, (nt * 5) / 8, nt }; // uneven, incl. a 1-row call
    for (size_t c = 0; c + 1 < sizeof(cuts) / sizeof(cuts[0]); ++c) {
        auto p = ptrs(xs, cuts[c], cuts[c + 1]);
        accumulate(h1.data(), K, N, p.data(), (int64_t) p.size(), 1);
        accumulate(h4.data(), K, N, p.data(), (int64_t) p.size(), 4);
    }
    std::string why;
    CHECK(matches_ref(h1.data(), ref, K, N, 2e-5, &why), "K=%lld N=%lld vs double reference: %s", (long long) K, (long long) N, why.c_str());
    CHECK(std::memcmp(h1.data(), h4.data(), h1.size() * sizeof(float)) == 0, "1 thread and 4 threads differ");
    CHECK(blocks_symmetric(h1.data(), K, N), "blocks not symmetric");
    double ratio = 0.0;
    CHECK(blocks_psd(h1.data(), K, N, &ratio), "a block is not PSD (min/max eigenvalue %g)", ratio);

    // diagonal == the imatrix diagonal accumulation (tools/imatrix/imatrix.cpp:564-569 order), f32 order tolerance
    std::vector<float> diag((size_t) K, 0.0f);
    for (const auto & x : xs) for (int64_t j = 0; j < K; ++j) diag[j] += x[j] * x[j];
    for (int64_t k = 0; k < K; ++k) {
        const float hk = h1[k * N + (k % N)];
        CHECK(std::fabs(hk - diag[k]) <= 1e-5f * std::max(1.0f, diag[k]), "diag k=%lld: %g vs imatrix %g", (long long) k, hk, diag[k]);
    }
    // tail of the partial last block stays 0
    const int64_t b_last = ((K - 1) / N) * N;
    for (int64_t k = b_last; k < K; ++k) for (int64_t j = K - b_last; j < N; ++j) {
        CHECK(h1[k * N + j] == 0.0f, "partial-block tail (k=%lld, j=%lld) not zero", (long long) k, (long long) j);
    }
    printf("random K=%lld N=%lld tokens=%lld (min eig / max eig %.3g): %s\n", (long long) K, (long long) N,
           (long long) n_tok, ratio, g_fail > fail0 ? "FAILED" : "ok");
}

static void test_routed() {
    // MUL_MAT_ID shape: 3 matrices, every token routed to 2 of them
    const int fail0 = g_fail;
    const int64_t K = 96, N = 32, n_mat = 3, n_tok = 150;
    std::vector<std::vector<float>> xs((size_t) n_tok, std::vector<float>((size_t) K));
    for (auto & x : xs) for (auto & v : x) v = frand();
    imatrix_hessian_entry e;
    imatrix_hessian_entry_init(e, K, N, n_mat);
    std::vector<std::vector<std::vector<float>>> per((size_t) n_mat);
    std::vector<std::vector<const float *>> rows((size_t) n_mat);
    for (int64_t t = 0; t < n_tok; ++t) {
        const int64_t m0 = t % n_mat;
        const int64_t m1 = (m0 + 1 + (t / n_mat) % (n_mat - 1)) % n_mat; // a different second expert
        for (int64_t m : { m0, m1 }) {
            rows[(size_t) m].push_back(xs[(size_t) t].data());
            per[(size_t) m].push_back(xs[(size_t) t]);
        }
    }
    for (int64_t m = 0; m < n_mat; ++m) {
        accumulate(e.blocks.data() + m * K * N, K, N, rows[m].data(), (int64_t) rows[m].size(), 2);
        e.counts[m] += (int64_t) rows[m].size();
    }
    for (int64_t m = 0; m < n_mat; ++m) {
        std::string why;
        const auto ref = ref_blocks(per[m], K, N);
        CHECK(matches_ref(e.blocks.data() + m * K * N, ref, K, N, 2e-5, &why), "routed matrix %lld: %s", (long long) m, why.c_str());
    }
    printf("routed n_mat=3 K=96 N=32: %s\n", g_fail > fail0 ? "FAILED" : "ok");
}

static void test_file_round_trip() {
    const int fail0 = g_fail;
    imatrix_hessian_file f;
    f.block_size  = 24;
    f.chunk_count = 7;
    f.chunk_size  = 512;
    f.datasets    = { "calib.txt" };
    auto & d = f.entries["blk.0.attn_q.weight"];
    imatrix_hessian_entry_init(d, 40, 24, 1);
    auto & x = f.entries["blk.0.ffn_up_exps.weight"];
    imatrix_hessian_entry_init(x, 50, 24, 4);
    for (auto & v : d.blocks) v = frand();
    for (auto & v : x.blocks) v = frand();
    d.counts = { 123456789012ll };
    x.counts = { 1, 2, 3, 40000000000ll };

    const std::string path = "test-imatrix-hessian.tmp.gguf";
    CHECK(imatrix_hessian_save(path, f), "save failed");
    imatrix_hessian_file g;
    std::string err;
    CHECK(imatrix_hessian_load(path, g, &err), "load failed: %s", err.c_str());
    CHECK(g.block_size == 24 && g.version == IMATRIX_HESSIAN_VERSION && g.chunk_count == 7 && g.chunk_size == 512,
          "header mismatch (block %d version %u chunks %d size %d)", g.block_size, g.version, g.chunk_count, g.chunk_size);
    CHECK(g.datasets.size() == 1 && g.datasets[0] == "calib.txt", "datasets mismatch");
    CHECK(g.entries.size() == 2, "entry count %zu", g.entries.size());
    for (const auto & [name, e] : f.entries) {
        auto it = g.entries.find(name);
        CHECK(it != g.entries.end(), "entry %s missing", name.c_str());
        if (it == g.entries.end()) continue;
        CHECK(it->second.n_in == e.n_in && it->second.n_mat == e.n_mat, "%s shape mismatch", name.c_str());
        CHECK(it->second.blocks.size() == e.blocks.size() &&
              std::memcmp(it->second.blocks.data(), e.blocks.data(), e.blocks.size() * sizeof(float)) == 0, "%s blocks differ", name.c_str());
        CHECK(it->second.counts == e.counts, "%s counts differ", name.c_str());
    }
    // the file names its layout and is typed, so a GPTQ reader can refuse anything else
    {
        struct gguf_init_params gp = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
        struct gguf_context * gg = gguf_init_from_file(path.c_str(), gp);
        CHECK(gg != nullptr, "raw open failed");
        if (gg) {
            CHECK(gguf_find_key(gg, LLM_KV_IMATRIX_HESSIAN_LAYOUT) >= 0, "layout key missing");
            const int64_t ti = gguf_find_tensor(gg, "blk.0.ffn_up_exps.weight.hess_blocks");
            CHECK(ti >= 0 && gguf_get_tensor_type(gg, ti) == GGML_TYPE_F32, "hess_blocks tensor missing or not F32");
            gguf_free(gg);
        }
    }
    // a different version must be refused
    {
        struct ggml_context * ctx = nullptr;
        struct gguf_init_params gp = { /* .no_alloc = */ false, /* .ctx = */ &ctx };
        struct gguf_context * gg = gguf_init_from_file(path.c_str(), gp);
        if (gg) {
            // copy into a fresh context (KV + the ctx tensors), as gguf-split does; writing a context read from a
            // file back out is not safe (its tensor infos are not fully initialised, see the report for this row)
            struct gguf_context * out = gguf_init_empty();
            gguf_set_kv(out, gg);
            gguf_set_val_u32(out, LLM_KV_IMATRIX_HESSIAN_VER, IMATRIX_HESSIAN_VERSION + 1);
            for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                // the loader's context also holds its own data-blob tensor, which is not a GGUF tensor
                if (gguf_find_tensor(gg, ggml_get_name(t)) >= 0) {
                    gguf_add_tensor(out, t);
                }
            }
            const std::string bad = "test-imatrix-hessian.badver.tmp.gguf";
            CHECK(gguf_write_to_file(out, bad.c_str(), false), "writing the wrong-version file failed");
            gguf_free(out);
            gguf_free(gg);
            ggml_free(ctx);
            imatrix_hessian_file h;
            CHECK(!imatrix_hessian_load(bad, h, &err), "a version %u file was accepted", IMATRIX_HESSIAN_VERSION + 1);
            std::remove(bad.c_str());
        }
    }
    std::remove(path.c_str());
    printf("sidecar save/load round trip: %s\n", g_fail > fail0 ? "FAILED" : "ok");
}

static void test_negative_controls() {
    // the checkers must reject damaged inputs, or the passes above mean nothing
    const int fail0 = g_fail;
    const int64_t K = 40, N = 16;
    std::vector<std::vector<float>> xs(60, std::vector<float>((size_t) K));
    for (auto & x : xs) for (auto & v : x) v = frand();
    auto ref = ref_blocks(xs, K, N);
    std::vector<float> h((size_t) (K * N));
    for (size_t i = 0; i < h.size(); ++i) h[i] = (float) ref[i];

    CHECK(matches_ref(h.data(), ref, K, N, 2e-5, nullptr), "control: an exact copy of the reference was rejected");
    auto bad_ref = ref;
    bad_ref[3 * N + 5] += 0.05 * std::sqrt(ref[3 * N + 3] * ref[5 * N + 5]); // one off-diagonal entry, 5 percent
    CHECK(!matches_ref(h.data(), bad_ref, K, N, 2e-5, nullptr), "negative control: a corrupted reference was accepted");

    auto asym = h;
    asym[2 * N + 7] += 1.0f;
    CHECK(!blocks_symmetric(asym.data(), K, N), "negative control: an asymmetric block was accepted");

    auto indef = h;
    indef[(N + 4) * N + 4] = -10.0f * std::fabs(indef[(N + 4) * N + 4]) - 1.0f; // negative diagonal entry
    CHECK(!blocks_psd(indef.data(), K, N), "negative control: an indefinite block was accepted");
    printf("negative controls (corrupted reference, asymmetric block, indefinite block all rejected): %s\n",
           g_fail > fail0 ? "FAILED" : "ok");
}

static void test_cli() {
    const int fail0 = g_fail;
    auto parse = [](std::vector<std::string> args, common_params & p) {
        std::vector<char *> argv;
        for (auto & a : args) argv.push_back(const_cast<char *>(a.data()));
        return common_params_parse((int) argv.size(), argv.data(), p, LLAMA_EXAMPLE_IMATRIX);
    };
    {
        common_params p;
        CHECK(p.imat_hessian_block == 0, "default --hessian-block is %d, expected 0 (off)", p.imat_hessian_block);
        // llama-imatrix's parser requires -m; the file is not opened by the parser
        CHECK(parse({ "llama-imatrix", "-m", "unused.gguf", "--hessian-block", "128", "--hessian-out", "h.gguf", "--hessian-max-mib", "512" }, p),
              "--hessian-block 128 --hessian-out h.gguf --hessian-max-mib 512 rejected");
        CHECK(p.imat_hessian_block == 128 && p.imat_hessian_out == "h.gguf" && p.imat_hessian_max_mib == 512,
              "parsed values wrong: block %d out '%s' max %d", p.imat_hessian_block, p.imat_hessian_out.c_str(), p.imat_hessian_max_mib);
    }
    for (const char * v : { "1", "1025", "-3", "abc" }) {
        common_params p;
        CHECK(!parse({ "llama-imatrix", "-m", "unused.gguf", "--hessian-block", v }, p), "--hessian-block %s accepted", v);
    }
    {
        common_params p;
        CHECK(!parse({ "llama-imatrix", "-m", "unused.gguf", "--hessian-max-mib", "0" }, p), "--hessian-max-mib 0 accepted");
    }
    {
        common_params p; // positive control for the refusals above: the same argv with a valid value parses
        CHECK(parse({ "llama-imatrix", "-m", "unused.gguf", "--hessian-block", "2", "--hessian-max-mib", "1" }, p) &&
              p.imat_hessian_block == 2 && p.imat_hessian_max_mib == 1, "--hessian-block 2 --hessian-max-mib 1 rejected");
    }
    printf("CLI bounds: %s\n", g_fail > fail0 ? "FAILED" : "ok");
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0); // progress survives a crash when stdout is redirected
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--baseline-diag-only") == 0) {
            g_diag_only = true;
        }
    }
    if (g_diag_only) {
        printf("MODE: baseline-diag-only (the pre-row imatrix accumulation); this run is expected to FAIL\n");
    }
    test_hand_case();
    test_random(300, 128, 900);
    test_random(64, 24, 300);
    test_routed();
    test_negative_controls();
    if (!g_diag_only) {
        test_file_round_trip();
        test_cli();
    }
    if (g_fail) {
        fprintf(stderr, "test-imatrix-hessian: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    printf("test-imatrix-hessian: all checks passed\n");
    return 0;
}
