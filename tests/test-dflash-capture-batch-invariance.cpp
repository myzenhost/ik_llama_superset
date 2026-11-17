// Installing the DFlash target-feature capture must not change the target's logits, for one sequence or several in one batch.
//
// Row S69-SPEC-POOL-DFLASH-CONCURRENT-VERIFY-CORRUPTS-TARGET-2026-11-05-01 (measured 2026-11-05): llama-server -np 2 with
// --spec-type dflash --spec-draft-n-max 8 emits text the target itself scores ~9 nats lower; -np 2 with a draft MODEL, with
// suffix_automaton, or without speculation is clean, single-slot DFlash is clean. The only target-side difference DFlash adds is
// llama_set_dflash_capture_layers (src/llama-spec-features-dflash.cpp), an eval callback that observes l_out-N and makes the
// scheduler split the graph at every captured node. This test isolates that: no draft model, no server.
//
//   PLAIN   context n_seq_max 2, no capture: decode both prompts in one batch, then R rounds of a 2 x W verify batch (logits on every row)
//   CAP     the same context with the capture installed (layers from DFL_CAP_LAYERS, default 1,9,17,25,33 = Qwen3-8B DFlash-b16)
//   SOLO    per sequence alone (n_seq_max 1, no capture), the reference for the batch-variance baseline
// Reported per row: max |logit| diff and argmax agreement, CAP vs PLAIN (must be ~0: the capture only reads) and PLAIN vs SOLO
// (the batch-variance floor). A control row compares against a different token so the comparator is shown able to see a change.
// Exit 0 = CAP agrees with PLAIN on every row; 1 = a row differs beyond tolerance or its argmax flips; 2 = setup failure.
// Env: DFL_CAP_LAYERS, DFL_W (verify width, default 9 = n_max 8 + id_last), DFL_ROUNDS (default 3), DFL_KV_UNIFIED (default 1),
//      DFL_ACCEPT (default 0 = keep every verified row): k > 0 emulates speculative REJECTION -- after each verify round only the
//      first k rows of each sequence are kept (llama_memory_seq_rm of the rest), the next round continues from there, in every
//      context alike. Many rejected rows per round across two sequences is what the 2026-11-05 corrupt server arm had (drafts
//      of one repeated token); this arm asks whether the TARGET's rollback alone diverges, with no draft involved. With k > 0 the
//      PLAIN-vs-SOLO comparison is the gate as well (argmax per row), since there is no capture difference to measure.
// usage: test-dflash-capture-batch-invariance -m <target.gguf> [-t N]
// GATE-CLAIM-OK: unit test, no resource claims.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "../src/llama-spec-features-dflash.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<int32_t> parse_layers(const char * s) {
    std::vector<int32_t> out;
    std::string cur;
    for (const char * p = s; ; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) {
                out.push_back((int32_t) std::atoi(cur.c_str()));
            }
            cur.clear();
            if (*p == '\0') {
                break;
            }
        } else {
            cur.push_back(*p);
        }
    }
    return out;
}

static int env_int(const char * name, int def) {
    const char * v = std::getenv(name);
    return v ? std::atoi(v) : def;
}

struct row_cmp { double max_diff; double scale; int argmax_a; int argmax_b; };

static row_cmp compare(const float * a, const float * b, int n_vocab) {
    row_cmp r = { 0.0, 1.0, 0, 0 };
    for (int i = 0; i < n_vocab; ++i) {
        r.max_diff = std::max(r.max_diff, (double) std::fabs(a[i] - b[i]));
        r.scale    = std::max(r.scale, (double) std::fabs(b[i]));
        if (a[i] > a[r.argmax_a]) r.argmax_a = i;
        if (b[i] > b[r.argmax_b]) r.argmax_b = i;
    }
    return r;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_seq, bool unified) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq;
    cparams.n_ctx      = 1024;
    cparams.n_batch    = 128;
    cparams.n_ubatch   = 128;
    cparams.kv_unified = unified;
    return llama_init_from_model(model, cparams);
}

// tokens of each sequence: prompt [0, n_prompt) then R windows of W tokens; positions are contiguous per sequence
struct seq_text { std::vector<llama_token> tok; int n_prompt; };

// logits[round][seq][row] -> n_vocab floats
using logit_table = std::vector<std::vector<std::vector<std::vector<float>>>>;

// rounds advance by `step` positions (W when every row is kept, the accept count k otherwise); after each round the rows past
// p0 + step are removed (the emulated rejection)
static int g_step = 0;
static int g_accept = 0;

static void rollback(llama_context * ctx, llama_seq_id s, llama_pos keep_end) {
    if (g_accept > 0 && !llama_memory_seq_rm(llama_get_memory(ctx), s, keep_end, -1)) {
        fprintf(stderr, "FAIL: llama_memory_seq_rm(seq %d, %d, -1) refused\n", (int) s, (int) keep_end);
        std::exit(2);
    }
}

static bool run_batched(llama_context * ctx, const std::vector<seq_text> & seqs, int W, int R, int n_vocab, logit_table & out) {
    const int n_seq = (int) seqs.size();
    llama_batch batch = llama_batch_init(512, 0, n_seq);
    bool ok = true;
    // prompts of every sequence in ONE batch, as the server processes concurrent prompts
    batch.n_tokens = 0;
    for (int s = 0; s < n_seq && ok; ++s) {
        for (int i = 0; i < seqs[s].n_prompt; ++i) {
            common_batch_add(batch, seqs[s].tok[i], i, { s }, i + 1 == seqs[s].n_prompt);
        }
    }
    ok = llama_decode(ctx, batch) == 0;
    out.assign(R, std::vector<std::vector<std::vector<float>>>(n_seq));
    for (int r = 0; r < R && ok; ++r) {
        batch.n_tokens = 0;
        for (int s = 0; s < n_seq; ++s) {
            const int p0 = seqs[s].n_prompt + r * g_step;
            for (int i = 0; i < W; ++i) {
                common_batch_add(batch, seqs[s].tok[p0 + i], p0 + i, { s }, true);
            }
        }
        ok = llama_decode(ctx, batch) == 0;
        for (int s = 0; s < n_seq && ok; ++s) {
            for (int i = 0; i < W; ++i) {
                const float * lg = llama_get_logits_ith(ctx, s * W + i);
                out[r][s].emplace_back(lg, lg + n_vocab);
            }
            rollback(ctx, s, seqs[s].n_prompt + r * g_step + g_step);
        }
    }
    llama_batch_free(batch);
    return ok;
}

static bool run_solo(llama_context * ctx, const seq_text & st, int W, int R, int n_vocab, std::vector<std::vector<std::vector<float>>> & out) {
    llama_batch batch = llama_batch_init(512, 0, 1);
    batch.n_tokens = 0;
    for (int i = 0; i < st.n_prompt; ++i) {
        common_batch_add(batch, st.tok[i], i, { 0 }, i + 1 == st.n_prompt);
    }
    bool ok = llama_decode(ctx, batch) == 0;
    out.assign(R, {});
    for (int r = 0; r < R && ok; ++r) {
        batch.n_tokens = 0;
        const int p0 = st.n_prompt + r * g_step;
        for (int i = 0; i < W; ++i) {
            common_batch_add(batch, st.tok[p0 + i], p0 + i, { 0 }, true);
        }
        ok = llama_decode(ctx, batch) == 0;
        for (int i = 0; i < W && ok; ++i) {
            const float * lg = llama_get_logits_ith(ctx, i);
            out[r].emplace_back(lg, lg + n_vocab);
        }
        if (ok) {
            rollback(ctx, 0, p0 + g_step);
        }
    }
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 2;
    }
    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 2;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    const int  W       = env_int("DFL_W", 9);
    const int  R       = env_int("DFL_ROUNDS", 3);
    const bool unified = env_int("DFL_KV_UNIFIED", 1) != 0;
    g_accept = std::max(0, std::min(env_int("DFL_ACCEPT", 0), W));
    g_step   = g_accept > 0 ? g_accept : W;
    const std::vector<int32_t> layers = parse_layers(std::getenv("DFL_CAP_LAYERS") ? std::getenv("DFL_CAP_LAYERS") : "1,9,17,25,33");

    llama_context * ctx_plain = make_ctx(params, model, 2, unified);
    llama_context * ctx_cap   = make_ctx(params, model, 2, unified);
    llama_context * ctx_solo  = make_ctx(params, model, 1, false);
    if (!ctx_plain || !ctx_cap || !ctx_solo) {
        fprintf(stderr, "FAIL: context creation\n");
        return 2;
    }
    if (!llama_set_dflash_capture_layers(ctx_cap, layers.data(), (int32_t) layers.size())) {
        fprintf(stderr, "FAIL: llama_set_dflash_capture_layers refused %zu layers\n", layers.size());
        return 2;
    }

    // two real texts (the e2e test's prompt shapes: prose and code), each long enough for the prompt plus R windows
    const char * texts[2] = {
        "The printing press was invented by Johannes Gutenberg around 1440. It changed how knowledge spread across Europe, "
        "making books cheaper and more common. Before the press, scribes copied every page by hand, which took months. "
        "After it, a single shop could print hundreds of copies in a week, and literacy rose in the cities that had one.",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n\ndef factorial(n):\n    \"\"\"Return n!.\"\"\"\n"
        "    if n < 2:\n        return 1\n    return n * factorial(n - 1)\n\n\ndef ackermann(m, n):\n    if m == 0:\n",
    };
    std::vector<seq_text> seqs(2);
    for (int s = 0; s < 2; ++s) {
        seqs[s].tok = common_tokenize(ctx_plain, texts[s], true);
        const int need = (R - 1) * g_step + W; // the last round's window ends at n_prompt + (R-1)*step + W
        seqs[s].n_prompt = (int) seqs[s].tok.size() - need;
        if (seqs[s].n_prompt < 8) {
            fprintf(stderr, "FAIL: text %d has %zu tokens, needs > %d\n", s, seqs[s].tok.size(), need + 8);
            return 2;
        }
    }
    printf("W=%d R=%d accept=%d kv_unified=%d capture layers=%zu n_prompt=%d/%d\n", W, R, g_accept, unified ? 1 : 0, layers.size(),
           seqs[0].n_prompt, seqs[1].n_prompt);

    logit_table plain, cap;
    std::vector<std::vector<std::vector<std::vector<float>>>> solo(2);
    if (!run_batched(ctx_plain, seqs, W, R, n_vocab, plain) || !run_batched(ctx_cap, seqs, W, R, n_vocab, cap)) {
        fprintf(stderr, "FAIL: batched decode\n");
        return 2;
    }
    for (int s = 0; s < 2; ++s) {
        llama_memory_clear(llama_get_memory(ctx_solo), true);
        if (!run_solo(ctx_solo, seqs[s], W, R, n_vocab, solo[s])) {
            fprintf(stderr, "FAIL: solo decode seq %d\n", s);
            return 2;
        }
    }

    // measured tolerance: PLAIN vs SOLO is the batch-variance floor; CAP may differ from PLAIN by no more than 1e-3 x the logit scale
    constexpr double tol_rel = 1e-3;
    int fails = 0;
    double worst_cap = 0.0, worst_solo = 0.0;
    for (int r = 0; r < R; ++r) {
        for (int s = 0; s < 2; ++s) {
            for (int i = 0; i < W; ++i) {
                const row_cmp c = compare(cap[r][s][i].data(), plain[r][s][i].data(), n_vocab);
                const row_cmp b = compare(plain[r][s][i].data(), solo[s][r][i].data(), n_vocab);
                // rejection arm: the 2-sequence target must stay on the 1-sequence path; a flipped argmax counts unless SOLO's
                // own top-2 gap at that row is within the batch-variance noise (0.5)
                const float * so = solo[s][r][i].data();
                const bool solo_flip = g_accept > 0 && b.argmax_a != b.argmax_b && (so[b.argmax_b] - so[b.argmax_a]) > 0.5f;
                const bool bad = c.max_diff > tol_rel * c.scale || c.argmax_a != c.argmax_b || solo_flip;
                worst_cap  = std::max(worst_cap,  c.max_diff / c.scale);
                worst_solo = std::max(worst_solo, b.max_diff / b.scale);
                printf("%s round %d seq %d row %d: CAP-vs-PLAIN max|d| %.5f (argmax %d vs %d)  PLAIN-vs-SOLO max|d| %.5f (argmax %d vs %d)\n",
                       bad ? "FAIL" : "ok  ", r, s, i, c.max_diff, c.argmax_a, c.argmax_b, b.max_diff, b.argmax_a, b.argmax_b);
                fails += bad ? 1 : 0;
            }
        }
    }
    // control: the comparator must see a change -- a row of seq 0 against a row of seq 1
    const row_cmp ctl = compare(plain[0][0][0].data(), plain[0][1][0].data(), n_vocab);
    const bool ctl_ok = ctl.max_diff > 10 * tol_rel * ctl.scale;
    printf("%s CONTROL: seq 0 row 0 vs seq 1 row 0 max|d| %.5f (must exceed %.5f)\n", ctl_ok ? "ok  " : "FAIL", ctl.max_diff, 10 * tol_rel * ctl.scale);
    printf("worst CAP-vs-PLAIN rel %.3g, worst PLAIN-vs-SOLO rel %.3g, failing rows %d of %d\n", worst_cap, worst_solo, fails, 2 * R * W);

    llama_free(ctx_plain);
    llama_free(ctx_cap);
    llama_free(ctx_solo);
    if (!ctl_ok) {
        return 2;
    }
    printf("RESULT: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
