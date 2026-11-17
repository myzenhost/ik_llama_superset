// Per-sequence LoRA (llama_set_seq_adapters_lora): one decode batch whose sequences use DIFFERENT adapters equals running each sequence alone.
// Needs a model + adapters; scripts/run-with-tiny-lora-assets.py builds random-weight ones and passes the directory.
//   P1  every token's logits in the heterogeneous batch == the same sequence decoded alone under its own context-wide adapter set
//   P2  the adapters really matter: base != adapter A != adapter B (otherwise P1 proves nothing)
//   P3  a sequence with no adapter in a batch where others have adapters == the base model
//   P4  a sequence that lists two adapters == both set context-wide
//   P5  OFF is identical: after the per-sequence lists are cleared the logits equal the base logits bit for bit
//   P6  changing only the per-sequence SCALES between decodes (same adapter union -> the cached graph is reused) takes effect
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Relative tolerance for "batched == alone". The two graphs add the same low-rank terms in a different order (global branch: scale(B(Ax)) per adapter;
// per-sequence branch: alpha/rank scale, then the 0/s mask multiply), so fp32 rounding differs; the largest difference seen is ~5e-5 of the logit range
// while the adapters move the logits by ~1 of that range (P2), i.e. the tolerance is 0.05% of the effect it has to resolve.
static const double TOL = 3e-4;

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

struct seq_def {
    std::vector<llama_token> toks;
    std::vector<std::pair<llama_adapter_lora *, float>> ad;
};

static llama_context * g_ctx = nullptr;
static int32_t g_n_vocab = 0;

// decode `defs` as ONE batch (sequence i = seq id i); return per-sequence per-token logits
// last_only: logits only for the last token of each sequence (the shape a server batch has), so the last layer's rows are pruned to the outputs
static std::vector<std::vector<float>> decode_batch(const std::vector<seq_def> & defs, const std::vector<int> & seq_ids, bool last_only = false) {
    llama_memory_clear(llama_get_memory(g_ctx), true);
    int n = 0;
    for (const auto & d : defs) n += (int) d.toks.size();
    llama_batch b = llama_batch_init(n, 0, 1);
    b.n_tokens = 0;
    std::vector<int> first(defs.size());
    for (size_t s = 0; s < defs.size(); ++s) {
        first[s] = b.n_tokens;
        for (size_t i = 0; i < defs[s].toks.size(); ++i) {
            const int k = b.n_tokens++;
            b.token[k] = defs[s].toks[i];
            b.pos[k] = (llama_pos) i;
            b.n_seq_id[k] = 1;
            b.seq_id[k][0] = seq_ids[s];
            b.logits[k] = (!last_only || i + 1 == defs[s].toks.size()) ? 1 : 0;
        }
    }
    if (getenv("LORA_TEST_TRACE")) fprintf(stderr, "[decode: %zu seqs, %d tokens]\n", defs.size(), n);
    const int rc = llama_decode(g_ctx, b);
    CHECK(rc == 0);
    std::vector<std::vector<float>> out(defs.size());
    for (size_t s = 0; s < defs.size(); ++s) {
        for (size_t i = 0; i < defs[s].toks.size(); ++i) {
            if (last_only && i + 1 != defs[s].toks.size()) continue;
            const float * l = llama_get_logits_ith(g_ctx, first[s] + (int) i);
            out[s].insert(out[s].end(), l, l + g_n_vocab);
        }
    }
    llama_batch_free(b);
    return out;
}

static void set_global(const std::vector<std::pair<llama_adapter_lora *, float>> & ad) {
    std::vector<llama_adapter_lora *> p; std::vector<float> sc;
    for (auto & a : ad) { p.push_back(a.first); sc.push_back(a.second); }
    llama_set_adapters_lora(g_ctx, p.data(), p.size(), sc.data());
}

static void set_seq(int seq, const std::vector<std::pair<llama_adapter_lora *, float>> & ad) {
    std::vector<llama_adapter_lora *> p; std::vector<float> sc;
    for (auto & a : ad) { p.push_back(a.first); sc.push_back(a.second); }
    llama_set_seq_adapters_lora(g_ctx, seq, p.data(), p.size(), sc.data());
}

// the sequence alone, under a context-wide adapter set (every per-sequence list cleared first, so nothing but d.ad is applied)
static std::vector<float> alone(const seq_def & d) {
    for (int sq = 0; sq < 4; ++sq) llama_set_seq_adapters_lora(g_ctx, sq, nullptr, 0, nullptr);
    set_global(d.ad);
    auto r = decode_batch({ d }, { 0 });
    set_global({});
    return r[0];
}

// largest |a - b|, relative to the largest |b| (logits are O(10); the batched and the lone graph add the same terms in a different order)
static double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0, scale = 1.0;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, (double) std::fabs(a[i] - b[i])); scale = std::max(scale, (double) std::fabs(b[i])); }
    return m / scale;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.gguf lora-a.gguf lora-b.gguf\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 2; }
    llama_context_params cp = llama_context_default_params();
    const char * fa = getenv("LORA_TEST_FA"); // "0" disabled, "1" enabled, unset = the library default
    if (fa) cp.flash_attn_type = atoi(fa) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 4; cp.n_threads = 2; cp.n_threads_batch = 2;
    g_ctx = llama_init_from_model(model, cp);
    if (!g_ctx) { fprintf(stderr, "cannot create context\n"); return 2; }
    llama_adapter_lora * A = llama_adapter_lora_init(model, argv[2]);
    llama_adapter_lora * B = llama_adapter_lora_init(model, argv[3]);
    if (!A || !B) { fprintf(stderr, "cannot load adapters\n"); return 2; }
    g_n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const std::vector<llama_token> t0 = { 1, 270, 280, 290, 262, 271 };
    const std::vector<llama_token> t1 = { 1, 265, 290, 275, 291, 266, 280, 272, 259 };
    const std::vector<llama_token> t2 = { 1, 261, 298, 263, 288 };
    const std::vector<llama_token> t3 = { 1, 284, 277, 296, 260, 281, 269 };

    seq_def s0{ t0, { { A, 0.7f } } };
    seq_def s1{ t1, { { B, 1.0f } } };
    seq_def s2{ t2, {} };
    seq_def s3{ t3, { { A, 0.5f }, { B, 0.5f } } };

    const auto ref0 = alone(s0), ref1 = alone(s1), ref2 = alone(s2), ref3 = alone(s3);

    // P2: the adapters matter
    seq_def base0{ t0, {} };
    const auto b0 = alone(base0);
    const double dA = max_abs(ref0, b0);
    seq_def a_on_t0{ t0, { { B, 1.0f } } };
    const double dAB = max_abs(ref0, alone(a_on_t0));
    printf("P2 adapter effect: |A(0.7) - base| max %.3e, |A(0.7) - B(1.0)| max %.3e\n", dA, dAB);
    CHECK(dA > 1e-2);
    CHECK(dAB > 1e-2);

    // P1/P3/P4: one heterogeneous batch of four sequences
    set_global({});
    set_seq(0, s0.ad); set_seq(1, s1.ad); set_seq(2, s2.ad); set_seq(3, s3.ad);
    auto het = decode_batch({ s0, s1, s2, s3 }, { 0, 1, 2, 3 });
    const double e0 = max_abs(het[0], ref0), e1 = max_abs(het[1], ref1), e2 = max_abs(het[2], ref2), e3 = max_abs(het[3], ref3);
    printf("P1 heterogeneous batch vs alone, max |logit diff| / max|logit|: seq0(A) %.2e  seq1(B) %.2e  seq2(none) %.2e  seq3(A+B) %.2e\n", e0, e1, e2, e3);
    CHECK(e0 < TOL); CHECK(e1 < TOL); CHECK(e2 < TOL); CHECK(e3 < TOL);
    // P3 is e2 (a sequence without adapter inside an adapter batch == the base model alone); P4 is e3 (A+B on one sequence == both set context-wide)

    // P6: only the scales change (same union {A,B}): the cached graph is reused and must pick up the new scales
    s0.ad = { { A, 0.2f } }; s1.ad = { { B, 0.4f } }; s3.ad = { { A, 1.0f } };
    const auto ref0b = alone(s0), ref1b = alone(s1), ref3b = alone(s3);
    set_seq(0, s0.ad); set_seq(1, s1.ad); set_seq(3, s3.ad);
    auto het2 = decode_batch({ s0, s1, s2, s3 }, { 0, 1, 2, 3 });
    const double f0 = max_abs(het2[0], ref0b), f1 = max_abs(het2[1], ref1b), f3 = max_abs(het2[3], ref3b);
    printf("P6 scales changed, same adapter union: seq0 %.2e seq1 %.2e seq3 %.2e; changed vs previous batch %.2e\n", f0, f1, f3, max_abs(het2[0], het[0]));
    CHECK(f0 < TOL); CHECK(f1 < TOL); CHECK(f3 < TOL);
    CHECK(max_abs(het2[0], het[0]) > 1e-3);

    // P7: the shape a server batch has: logits only for the last token of each sequence (the last layer's rows are pruned to the outputs)
    {
        for (int sq = 0; sq < 4; ++sq) set_seq(sq, {});
        set_seq(0, s0.ad); set_seq(1, s1.ad); set_seq(3, s3.ad);
        const auto lo = decode_batch({ s0, s1, s2, s3 }, { 0, 1, 2, 3 }, true);
        auto tail = [&](const std::vector<float> & full) { return std::vector<float>(full.end() - g_n_vocab, full.end()); };
        const double g0 = max_abs(lo[0], tail(ref0b)), g1 = max_abs(lo[1], tail(ref1b)), g2 = max_abs(lo[2], tail(ref2)), g3 = max_abs(lo[3], tail(ref3b));
        printf("P7 last-token-only logits (pruned rows): seq0 %.2e seq1 %.2e seq2 %.2e seq3 %.2e\n", g0, g1, g2, g3);
        CHECK(g0 < TOL); CHECK(g1 < TOL); CHECK(g2 < TOL); CHECK(g3 < TOL);
    }

    // P5: OFF. Clear every per-sequence list; the base logits must be bit-identical to those of a context that never used the API.
    for (int s = 0; s < 4; ++s) set_seq(s, {});
    set_global({});
    const auto off = alone(base0);
    CHECK(off.size() == b0.size() && std::memcmp(off.data(), b0.data(), off.size() * sizeof(float)) == 0);
    printf("P5 after clearing the per-sequence lists, base logits identical to before the API was used: %s\n",
           (off.size() == b0.size() && std::memcmp(off.data(), b0.data(), off.size() * sizeof(float)) == 0) ? "bit-identical" : "DIFFERENT");

    llama_adapter_lora_free(A);
    llama_adapter_lora_free(B);
    llama_free(g_ctx);
    llama_model_free(model);
    llama_backend_free();
    if (n_fail) { fprintf(stderr, "test-lora-per-seq: %d failure(s)\n", n_fail); return 1; }
    printf("test-lora-per-seq: all checks passed\n");
    return 0;
}
