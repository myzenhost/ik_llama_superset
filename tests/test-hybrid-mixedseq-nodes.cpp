// Hybrid (delta-net / recurrent) multi-slot decode with mixed sequences in one batch.
//
//  (a) N slots decoding at once = one batch of N tokens with N DISTINCT seq ids. delta_net::build_layer_attn_linear then builds one
//      complete linear-attention sub-graph PER TOKEN (src/llama-delta-net.cpp), so the graph grows with n_seq_max * n_recurrent_layers,
//      which the node budget max(n_tokens*40, 32*n_tensors) did not cover: ggml_new_object returns NULL when the meta pool is
//      exhausted (the assert is compiled out in release) and the caller dereferences it -> SIGSEGV. The budget must reserve for it.
//  (b) a batch with a repeated seq id AND a different one (one slot's prompt chunk + other slots' decode tokens) was cut to 1-token
//      ubatches for the whole window; it must be cut at sequence-run boundaries so the chunk stays one ubatch.
//
// Rows S69-ENGREV-THIREUS-HYBRID-MIXEDSEQ-MAXNODES-2026-10-27-01; upstream Thireus/ik_llama.cpp c2e3eba2 (MIT).
//
// usage: test-hybrid-mixedseq-nodes <hybrid.gguf> [n_seq=16] [n_ubatch=64]
//   hybrid.gguf = any delta-net model (qwen35 / qwen3next), e.g. Qwen3.5-0.8B-Q8_0.gguf.
// PASS: (a) the N-distinct-seq decode returns 0 and its logits equal a single-sequence run of the same token within 1e-3 relative;
//       (b) the 4-token chunk + 3 decode tokens batch returns 0, the log shows the run-boundary split (not the 1-token fallback),
//           and its logits equal the single-sequence references.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::string g_log;
static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        g_log += text;
    }
}

struct row { llama_token tok; llama_pos pos; llama_seq_id seq; bool out; };

static int decode_rows(llama_context * ctx, const std::vector<row> & rows) {
    llama_batch b = llama_batch_init((int32_t) rows.size(), 0, 1);
    b.n_tokens = (int32_t) rows.size();
    for (size_t i = 0; i < rows.size(); ++i) {
        b.token[i]     = rows[i].tok;
        b.pos[i]       = rows[i].pos;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = rows[i].seq;
        b.logits[i]    = rows[i].out ? 1 : 0;
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static llama_token tok_of(int seq, int p, int n_vocab) {
    return (llama_token) ((1000 + seq * 37 + p * 11) % (n_vocab > 5000 ? 5000 : n_vocab));
}

// rel_tol: 1e-3 for batches of up to 16 tokens, where the mixed decode is bit-identical to the single-sequence run on this CPU
// (MEASURED 2026-11-07, Qwen3.5-0.8B-Q8_0: worst rel diff 0 at 4 and 16 slots). From 32 tokens in one ubatch the CPU matmul takes a
// different quantized kernel path (rel diff 0.0137 for the SAME token at 32 and at 64 slots, i.e. independent of the sequence
// handling), so wide batches are held to the top-1 token plus 5e-2.
static bool close_enough(const float * a, const float * b, int n, double * worst, double rel_tol) {
    double mx = 0, ref = 0;
    int ia = 0, ib = 0;
    for (int i = 0; i < n; ++i) {
        mx  = std::fmax(mx, std::fabs((double) a[i] - b[i]));
        ref = std::fmax(ref, std::fabs((double) b[i]));
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
    }
    *worst = mx / (ref > 1e-6 ? ref : 1.0);
    return *worst < rel_tol && ia == ib;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <hybrid.gguf> [n_seq] [n_ubatch]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);   // a crash in the decode must not swallow the lines printed before it
    const int n_seq    = argc > 2 ? atoi(argv[2]) : 16;
    const int n_ubatch = argc > 3 ? atoi(argv[3]) : 64;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 2;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_hist  = 4;   // history tokens per sequence before the mixed decode

    auto make_ctx = [&](int n_seq_max) {
        llama_context_params cp = llama_context_default_params();
        cp.n_seq_max = n_seq_max;
        cp.n_ctx     = 128 * n_seq_max;
        cp.n_batch   = 512;
        cp.n_ubatch  = n_ubatch;
        return llama_init_from_model(model, cp);
    };

    int fails = 0;

    // reference: one sequence, one context -> logits of the token at position n_hist after n_hist history tokens
    auto reference = [&](int seq, int extra_tokens, std::vector<float> & out) -> bool {
        llama_context * rc = make_ctx(1);
        if (!rc) return false;
        std::vector<row> hist;
        for (int p = 0; p < n_hist; ++p) hist.push_back({ tok_of(seq, p, n_vocab), p, 0, false });
        bool ok = decode_rows(rc, hist) == 0;
        std::vector<row> last;
        for (int p = n_hist; p < n_hist + extra_tokens; ++p) last.push_back({ tok_of(seq, p, n_vocab), p, 0, p == n_hist + extra_tokens - 1 });
        ok = ok && decode_rows(rc, last) == 0;
        if (ok) {
            const float * l = llama_get_logits_ith(rc, -1);
            out.assign(l, l + n_vocab);
        }
        llama_free(rc);
        return ok;
    };

    // ---- (a) N distinct seq ids in one batch
    {
        g_log.clear();
        llama_context * ctx = make_ctx(n_seq);
        if (!ctx) { printf("FAIL (a) context creation\n"); return 1; }
        bool ok = true;
        for (int s = 0; s < n_seq && ok; ++s) {
            std::vector<row> hist;
            for (int p = 0; p < n_hist; ++p) hist.push_back({ tok_of(s, p, n_vocab), p, (llama_seq_id) s, false });
            ok = decode_rows(ctx, hist) == 0;
        }
        std::vector<row> mixed;
        for (int s = 0; s < n_seq; ++s) mixed.push_back({ tok_of(s, n_hist, n_vocab), n_hist, (llama_seq_id) s, true });
        const int rc = ok ? decode_rows(ctx, mixed) : -99;
        printf("(a) %d distinct seq ids in one batch (n_ubatch %d): decode rc=%d\n", n_seq, n_ubatch, rc);
        bool pass = rc == 0;
        for (int s = 0; pass && s < n_seq && s < 3; ++s) {
            std::vector<float> ref;
            if (!reference(s, 1, ref)) { pass = false; break; }
            double w = 0;
            const bool eq = close_enough(llama_get_logits_ith(ctx, s), ref.data(), n_vocab, &w, n_seq <= 16 ? 1e-3 : 5e-2);
            printf("    seq %d logits vs single-seq run: worst rel diff %.3g %s\n", s, w, eq ? "ok" : "MISMATCH");
            pass = pass && eq;
        }
        printf("%s (a) mixed decode of %d slots\n", pass ? "PASS" : "FAIL", n_seq);
        fails += pass ? 0 : 1;
        llama_free(ctx);
    }

    // ---- (b) one slot's prompt chunk (repeated seq id) + other slots' decode tokens
    {
        g_log.clear();
        llama_context * ctx = make_ctx(4);
        if (!ctx) { printf("FAIL (b) context creation\n"); return 1; }
        bool ok = true;
        for (int s = 1; s < 4 && ok; ++s) {
            std::vector<row> hist;
            for (int p = 0; p < n_hist; ++p) hist.push_back({ tok_of(s, p, n_vocab), p, (llama_seq_id) s, false });
            ok = decode_rows(ctx, hist) == 0;
        }
        std::vector<row> mixed;
        for (int p = 0; p < 4; ++p) mixed.push_back({ tok_of(0, p, n_vocab), p, 0, p == 3 });                        // seq 0: 4-token prompt chunk
        for (int s = 1; s < 4; ++s) mixed.push_back({ tok_of(s, n_hist, n_vocab), n_hist, (llama_seq_id) s, true }); // seqs 1..3: one decode token each
        const int rc = ok ? decode_rows(ctx, mixed) : -99;
        const bool fallback = g_log.find("falling back to single-token chunking") != std::string::npos;
        const bool split    = g_log.find("splitting it at sequence-run boundaries") != std::string::npos;
        printf("(b) 4-token chunk + 3 decode tokens: decode rc=%d, 1-token fallback logged=%s, run-boundary split logged=%s\n",
               rc, fallback ? "yes" : "no", split ? "yes" : "no");
        bool pass = rc == 0 && !fallback && split;
        for (int k = 0; pass && k < 4; ++k) {
            // output rows: row 3 (seq 0, pos 3), rows 4..6 (seq 1..3)
            const int out_row = k == 0 ? 3 : 3 + k;
            std::vector<float> ref;
            const bool got = k == 0 ? [&] {   // seq 0 reference: 4 tokens, logits at the last one
                llama_context * r1 = make_ctx(1);
                std::vector<row> v;
                for (int p = 0; p < 4; ++p) v.push_back({ tok_of(0, p, n_vocab), p, 0, p == 3 });
                const bool o = r1 && decode_rows(r1, v) == 0;
                if (o) { const float * l = llama_get_logits_ith(r1, -1); ref.assign(l, l + n_vocab); }
                if (r1) llama_free(r1);
                return o;
            }() : reference(k, 1, ref);
            double w = 0;
            const float * l = llama_get_logits_ith(ctx, out_row);
            const bool eq = got && l && close_enough(l, ref.data(), n_vocab, &w, 1e-3);
            printf("    seq %d logits vs single-seq run: worst rel diff %.3g %s\n", k, w, eq ? "ok" : "MISMATCH");
            pass = pass && eq;
        }
        printf("%s (b) chunk + decode tokens split at sequence-run boundaries\n", pass ? "PASS" : "FAIL");
        fails += pass ? 0 : 1;
        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();
    printf("%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
