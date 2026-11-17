// snapkv-rebase-ab -- does re-basing the surviving prompt KV to contiguous positions after a SnapKV prune change the accuracy hit?
// Row: S69-F1-PICK-SNAPKV-CONTIGUOUS-REBASE-AB-2026-10-06-01.
//
// For each document (a disjoint window of the input text) and each cap, four arms are scored on the SAME next n_score tokens by teacher forcing
// (sum of -log p, one token per decode call so the prune never fires during scoring):
//   A  full cache (snapkv_cap 0)                         -- reference
//   B  pruned, surviving cells keep their positions      -- today (snapkv_rebase 0)
//   C  pruned, survivors re-based to 0..keep-1            -- snapkv_rebase 1
//   W  pruned, survivors shifted the WRONG way (+len)     -- snapkv_rebase -1, control that the shift reaches the cache
//      (first run 2026-10-20: W is NOT measurably worse than B, W-B -0.004..+0.013 nats. RESEARCH-DONE: RoFormer, arXiv 2104.09864,
//      Section 3.2 eq. (16): the rotated q_m.k_n depends only on m-n. +len keeps every query-to-tail distance and only widens the
//      head-to-tail gap, so "worse than B" was not a valid criterion. Re-basing (C) is the same operation in the other direction:
//      at cap 512 C-A was below 0.7*(B-A) on 1 of 5 documents, at cap 1024 on 0 of 3.)
// Scoring continues at llama_kv_cache_seq_pos_max + 1, so every arm's next token follows its own last cached position.
// The prefill is ONE ubatch (n_ubatch = n_prefill), so exactly one prune happens, right after it.
//
// usage: llama-snapkv-rebase-ab -m model.gguf -f text.txt [--prefill 2048] [--score 256] [--docs 5] [--caps 512,1024] [-t 4]
// output: one line per (doc, cap, arm) plus a DECISION line computed with the rule fixed on the row before running:
//   adopt only if (C-A) < 0.7 * (B-A) on at least 4 of 5 documents at both caps.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) v.push_back(std::atoi(tok.c_str()));
    }
    return v;
}

static double score_arm(llama_model * model, const std::vector<llama_token> & toks, int n_prefill, int n_score, int cap, int rebase,
                        int n_threads, int * out_kept) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) (n_prefill + n_score + 64);
    cp.n_batch         = (uint32_t) n_prefill;
    cp.n_ubatch        = (uint32_t) n_prefill;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.flash_attn      = false;                                  // the SnapKV capture reads the materialised softmax
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.snapkv_cap      = cap;
    cp.snapkv_window   = 64;
    cp.snapkv_rebase   = rebase;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "context creation failed\n");
        std::exit(2);
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_batch b = llama_batch_init(n_prefill, 0, 1);
    for (int i = 0; i < n_prefill; ++i) {
        b.token[i] = toks[(size_t) i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = (i == n_prefill - 1);
    }
    b.n_tokens = n_prefill;
    if (llama_decode(ctx, b) != 0) {
        std::fprintf(stderr, "prefill decode failed\n");
        std::exit(3);
    }
    llama_batch_free(b);
    *out_kept = llama_kv_cache_seq_pos_max(ctx, 0) + 1;          // after a re-base: the number of survivors; otherwise n_prefill

    double nll = 0.0;
    const float * logits = llama_get_logits_ith(ctx, -1);
    llama_pos pos = llama_kv_cache_seq_pos_max(ctx, 0) + 1;
    for (int i = 0; i < n_score; ++i) {
        const llama_token target = toks[(size_t) (n_prefill + i)];
        double mx = -INFINITY;
        for (int v = 0; v < n_vocab; ++v) mx = logits[v] > mx ? logits[v] : mx;
        double se = 0.0;
        for (int v = 0; v < n_vocab; ++v) se += std::exp((double) logits[v] - mx);
        nll += -(((double) logits[target] - mx) - std::log(se));
        if (i + 1 == n_score) break;
        llama_batch one = llama_batch_init(1, 0, 1);
        one.token[0] = target; one.pos[0] = pos++; one.n_seq_id[0] = 1; one.seq_id[0][0] = 0; one.logits[0] = 1; one.n_tokens = 1;
        if (llama_decode(ctx, one) != 0) {
            std::fprintf(stderr, "scoring decode failed at %d\n", i);
            std::exit(4);
        }
        llama_batch_free(one);
        logits = llama_get_logits_ith(ctx, -1);
    }
    llama_free(ctx);
    return nll / n_score;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * text_path  = nullptr;
    int n_prefill = 2048, n_score = 256, n_docs = 5, n_threads = 4;
    std::vector<int> caps = {512, 1024};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); } return argv[++i]; };
        if (a == "-m") model_path = next();
        else if (a == "-f") text_path = next();
        else if (a == "--prefill") n_prefill = std::atoi(next());
        else if (a == "--score") n_score = std::atoi(next());
        else if (a == "--docs") n_docs = std::atoi(next());
        else if (a == "--caps") caps = parse_list(next());
        else if (a == "-t") n_threads = std::atoi(next());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (!model_path || !text_path) {
        std::fprintf(stderr, "usage: %s -m model.gguf -f text.txt [--prefill N] [--score N] [--docs N] [--caps a,b] [-t N]\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 2; }

    std::ifstream in(text_path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> all(text.size() + 16);
    const int n_all = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), all.data(), (int32_t) all.size(), false, false);
    if (n_all < 0) { std::fprintf(stderr, "tokenize failed\n"); return 2; }
    all.resize((size_t) n_all);
    const int doc_len = n_prefill + n_score;
    if (n_all < n_docs * doc_len) { std::fprintf(stderr, "text too short: %d tokens < %d\n", n_all, n_docs * doc_len); return 2; }
    std::printf("tokens %d, docs %d x %d (prefill %d + score %d), caps", n_all, n_docs, doc_len, n_prefill, n_score);
    for (int c : caps) std::printf(" %d", c);
    std::printf("\n");

    bool all_caps_adopt = true, controls_ok = true;
    for (int cap : caps) {
        int wins = 0;
        for (int d = 0; d < n_docs; ++d) {
            const size_t off = (size_t) d * (size_t) (n_all / n_docs);
            std::vector<llama_token> toks(all.begin() + (long) off, all.begin() + (long) off + doc_len);
            int kA = 0, kB = 0, kC = 0, kW = 0;
            const double A = score_arm(model, toks, n_prefill, n_score, 0,   0,  n_threads, &kA);
            const double B = score_arm(model, toks, n_prefill, n_score, cap, 0,  n_threads, &kB);
            const double C = score_arm(model, toks, n_prefill, n_score, cap, 1,  n_threads, &kC);
            const double W = score_arm(model, toks, n_prefill, n_score, cap, -1, n_threads, &kW);
            const bool win = (C - A) < 0.7 * (B - A);
            wins += win;
            // W control (revised after the 2026-10-20 run, see the header): the wrong-way shift must REACH the cache -- its next
            // position moves past B's and its score differs from B. "W worse than B" was the original criterion and is not one RoPE
            // supports: +len keeps every query-to-tail distance, so W differed from B by -0.004..+0.013 nats.
            controls_ok = controls_ok && (kW > kB) && (W != B) && (kC == cap);
            if (cap == caps.front()) {   // control: re-base on with a cap that drops nothing must reproduce A
                int kZ = 0;
                const double Z = score_arm(model, toks, n_prefill, n_score, n_prefill + 1, 1, n_threads, &kZ);
                const bool z_ok = std::fabs(Z - A) <= 1e-5 && kZ == n_prefill;
                controls_ok = controls_ok && z_ok;
                std::printf("control doc %d: cap %d drops nothing, |Z-A| = %.2e, next pos %d: %s\n", d, n_prefill + 1, std::fabs(Z - A), kZ,
                            z_ok ? "ok" : "FAILED");
            }
            std::printf("cap %5d doc %d  A %.5f  B %.5f (B-A %+.5f)  C %.5f (C-A %+.5f, next pos %d)  W %.5f (W-B %+.5f)  %s\n",
                        cap, d, A, B, B - A, C, C - A, kC, W, W - B, win ? "C-better" : "no");
            std::fflush(stdout);
        }
        std::printf("cap %d: C beats 0.7*(B-A) on %d of %d documents\n", cap, wins, n_docs);
        all_caps_adopt = all_caps_adopt && wins >= (n_docs * 4 + 4) / 5;
    }
    std::printf("CONTROLS %s (W's wrong-way shift reached the cache on every doc; C re-based to exactly cap positions; a no-drop cap reproduces A)\n", controls_ok ? "OK" : "FAILED");
    std::printf("DECISION %s\n", all_caps_adopt ? "ADOPT re-basing" : "KEEP GAPS (evidenced negative)");
    llama_model_free(model);
    llama_backend_free();
    return controls_ok ? 0 : 5;
}
