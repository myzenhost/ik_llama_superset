// S69 P1: the Inkling compute graph agrees with the reference implementation.
//
// Usage: test-inkling-parity [<inkling-f32.gguf> <ref_tokens.txt> <ref_logits.bin> [tol]]
//   no arguments -> prints SKIP and exits 0 (safe in a suite on a machine with no fixture)
//
// Tolerance: the default 1e-4 is 7x the measured F32 agreement (1.4e-5). It is tight ON PURPOSE: removing the log-N query scale moves the
// logits by only 1.5e-3, so a looser bound (the first draft used 2e-3) lets that defect through; see the mutation record in the commit message.
// Fixture: tests/inkling-arch-check/make_tiny_inkling.py (random weights) converted to F32, plus
// tests/inkling-arch-check/ref_inkling_logits.py, which runs the Apache-2.0 transformers InklingForCausalLM in eager fp32
// on the same weights and writes the logits of a fixed 40-token sequence. Everything here is fp32 end to end (F32 K/V cache), so the
// only expected difference is summation order.
//
// What each phase proves (every one compares ALL 40 positions against the reference):
//   A  one 40-token batch                         -- the whole graph: attention + relative bias + sliding window + log-N scale,
//                                                    four short convs, sigmoid router with shared-expert sink, padded-vocab mask
//   B  one token per decode                       -- the conv histories and K/V cells carry across ubatches
//   C  chunks of 7                                -- a chunk boundary inside the 4-wide conv history and inside the window
//   D  two sequences in one batch                 -- state rows are per sequence (a shared row would corrupt the second)
//   E  sequence removed, then decoded again       -- the in-graph reset at position 0: no history of the previous run leaks in
// A phase that agrees with the reference but is fed a graph that ignored state would still fail B/C/D/E; that is the point.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// Number of logits per position in the reference file (the UNPADDED vocab for Inkling: 262 of 320). Derived from the file, so the harness is
// arch-agnostic: the GDN l2-norm A/B (tests/gdn-l2norm-check) runs a qwen35 fixture through the same binary, where it equals the full vocab.
static int V_UNPADDED = 262;

struct entry { llama_token tok; llama_pos pos; llama_seq_id seq; bool out; };

// decode `es` as one batch; copy each requested row of logits into rows[seq][pos] (n_vocab floats each)
static bool decode(llama_context * ctx, const std::vector<entry> & es, int n_vocab,
                   std::vector<std::vector<std::vector<float>>> & rows) {
    llama_batch b = llama_batch_init((int32_t) es.size(), 0, 1);
    for (size_t i = 0; i < es.size(); ++i) {
        b.token[i]     = es[i].tok;
        b.pos[i]       = es[i].pos;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = es[i].seq;
        b.logits[i]    = es[i].out ? 1 : 0;
    }
    b.n_tokens = (int32_t) es.size();
    const int rc = llama_decode(ctx, b);
    if (rc == 0) {
        for (size_t i = 0; i < es.size(); ++i) {
            if (!es[i].out) {
                continue;
            }
            const float * l = llama_get_logits_ith(ctx, (int32_t) i);
            if (!l) {
                llama_batch_free(b);
                return false;
            }
            auto & r = rows[es[i].seq];
            if ((int) r.size() <= es[i].pos) {
                r.resize(es[i].pos + 1);
            }
            r[es[i].pos].assign(l, l + n_vocab);
        }
    } else {
        std::printf("llama_decode returned %d\n", rc);
    }
    llama_batch_free(b);
    return rc == 0;
}

// ctl_min is the NEGATIVE CONTROL: the smallest difference between a position's logits and the NEXT position's reference. It must
// be far above the tolerance, or "within tolerance" would not tell a right graph from a wrong one.
struct stats { double max_abs = 0, ref_absmax = 0, ctl_min = 1e30; int argmax_bad = 0, pad_bad = 0, n = 0, missing = 0; };

static stats compare(const std::vector<std::vector<float>> & got, const std::vector<float> & ref, int n_tok, int n_vocab) {
    stats s;
    for (int j = 0; j < n_tok; ++j) {
        if (j >= (int) got.size() || (int) got[j].size() != n_vocab) {
            ++s.missing;
            continue;
        }
        const float * g = got[j].data();
        const float * r = ref.data() + (size_t) j * V_UNPADDED;
        int ag = 0, ar = 0;
        for (int i = 0; i < V_UNPADDED; ++i) {
            s.max_abs    = std::max(s.max_abs, (double) std::fabs(g[i] - r[i]));
            s.ref_absmax = std::max(s.ref_absmax, (double) std::fabs(r[i]));
            if (g[i] > g[ag]) ag = i;
            if (r[i] > r[ar]) ar = i;
        }
        s.argmax_bad += ag != ar;
        if (j + 1 < n_tok) {
            const float * rn = ref.data() + (size_t) (j + 1) * V_UNPADDED;
            double d = 0;
            for (int i = 0; i < V_UNPADDED; ++i) d = std::max(d, (double) std::fabs(g[i] - rn[i]));
            s.ctl_min = std::min(s.ctl_min, d);
        }
        for (int i = V_UNPADDED; i < n_vocab; ++i) {
            if (!(std::isinf(g[i]) && g[i] < 0)) {
                ++s.pad_bad;
            }
        }
        ++s.n;
    }
    return s;
}

static void report(const char * name, const stats & s, double tol, int n_tok) {
    std::printf("  %-34s positions %2d/%d  max|d| %.3e (ref absmax %.3f)  argmax mismatches %d  padded rows not -inf %d\n",
                name, s.n, n_tok, s.max_abs, s.ref_absmax, s.argmax_bad, s.pad_bad);
    CHECK(s.missing == 0, "%s: %d positions produced no logits", name, s.missing);
    CHECK(s.max_abs <= tol, "%s: max abs logit difference %.3e exceeds %.3e", name, s.max_abs, tol);
    CHECK(s.ctl_min > 20.0*tol, "%s: negative control too weak: neighbouring positions differ by only %.3e (tolerance %.3e)", name, s.ctl_min, tol);
    CHECK(s.argmax_bad == 0, "%s: argmax disagrees at %d positions", name, s.argmax_bad);
    CHECK(s.pad_bad == 0, "%s: %d padded-vocab logits are not -inf", name, s.pad_bad);
}

int main(int argc, char ** argv) {
    if (argc < 4) { std::printf("SKIP: no fixture given (usage: %s <inkling-f32.gguf> <ref_tokens.txt> <ref_logits.bin> [tol])\n", argv[0]); return 0; }
    const double tol = argc > 4 ? std::atof(argv[4]) : 1e-4;

    std::vector<llama_token> toks;
    {
        std::ifstream f(argv[2]);
        long v;
        while (f >> v) toks.push_back((llama_token) v);
    }
    const int n_tok = (int) toks.size();
    std::vector<float> ref;
    {
        std::ifstream f(argv[3], std::ios::binary);
        f.seekg(0, std::ios::end);
        const size_t nb = (size_t) f.tellg();
        f.seekg(0);
        ref.resize(nb / sizeof(float));
        f.read((char *) ref.data(), (std::streamsize) nb);
    }
    if (n_tok == 0 || ref.size() % (size_t) n_tok != 0) {
        std::printf("FAIL: reference mismatch: %d tokens, %zu floats\n", n_tok, ref.size());
        return 2;
    }
    V_UNPADDED = (int) (ref.size() / (size_t) n_tok);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::printf("FAIL: model load returned null\n"); return 2; }
    const int n_vocab = llama_n_vocab(model);

    auto make_ctx = [&](uint32_t n_seq, uint32_t n_batch) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx      = 256 * n_seq;
        cp.n_batch    = n_batch;
        cp.n_ubatch   = n_batch;
        cp.n_seq_max  = n_seq;
        cp.type_k     = GGML_TYPE_F32;
        cp.type_v     = GGML_TYPE_F32;
        cp.n_threads  = 4;
        cp.n_threads_batch = 4;
        return llama_init_from_model(model, cp);
    };

    std::printf("inkling parity: %d tokens, tolerance %.1e\n", n_tok, tol);

    // ---- A, B, C, E on one single-sequence context ----
    {
        llama_context * ctx = make_ctx(1, 64);
        CHECK(ctx != nullptr, "context creation");
        if (ctx) {
            std::vector<std::vector<std::vector<float>>> rows(1);
            auto run_chunks = [&](int chunk) {
                rows.assign(1, {});
                bool ok = true;
                for (int s = 0; s < n_tok && ok; s += chunk) {
                    std::vector<entry> es;
                    for (int j = s; j < std::min(n_tok, s + chunk); ++j) es.push_back({ toks[j], j, 0, true });
                    ok = decode(ctx, es, n_vocab, rows);
                }
                return ok;
            };

            CHECK(run_chunks(n_tok), "A decode");
            report("A  one batch", compare(rows[0], ref, n_tok, n_vocab), tol, n_tok);

            llama_memory_clear(llama_get_memory(ctx), true);
            CHECK(run_chunks(1), "B decode");
            report("B  one token per decode", compare(rows[0], ref, n_tok, n_vocab), tol, n_tok);

            llama_memory_clear(llama_get_memory(ctx), true);
            CHECK(run_chunks(7), "C decode");
            report("C  chunks of 7", compare(rows[0], ref, n_tok, n_vocab), tol, n_tok);

            // E: drop the sequence but keep the buffers as they are (what a server does between requests), decode again
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1), "E seq_rm of the whole sequence is accepted");
            CHECK(run_chunks(n_tok), "E decode");
            report("E  re-decode after seq_rm", compare(rows[0], ref, n_tok, n_vocab), tol, n_tok);

            llama_free(ctx);
        }
    }

    // ---- D: two sequences, one batch each token pair-interleaved per sequence block ----
    {
        llama_context * ctx = make_ctx(2, 128);
        CHECK(ctx != nullptr, "2-sequence context creation");
        if (ctx) {
            std::vector<std::vector<std::vector<float>>> rows(2);
            std::vector<entry> es;
            for (int s = 0; s < 2; ++s) {
                for (int j = 0; j < n_tok; ++j) es.push_back({ toks[j], j, (llama_seq_id) s, true });
            }
            CHECK(decode(ctx, es, n_vocab, rows), "D decode");
            report("D  two sequences, seq 0", compare(rows[0], ref, n_tok, n_vocab), tol, n_tok);
            report("D  two sequences, seq 1", compare(rows[1], ref, n_tok, n_vocab), tol, n_tok);
            llama_free(ctx);
        }
    }

    llama_model_free(model);
    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
