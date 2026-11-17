// legacy-state-check: does saved state actually RESTORE what decodes?
//
// Row S69-FORK-LEGACY-WHOLE-STATE-AND-STATE-FILE-API-SERIALIZE-UNUSED-MEMORY-2026-09-26-01. On a legacy-builder
// model the cache that decodes is kv_self; before the routing fix, llama_state_get/set_data and the seq FILE pair
// (what llama-server's /slots?action=save|restore uses) serialized ctx->memory instead, so a restore changed
// nothing that decodes. This check measures that directly instead of trusting a return code:
//
//   1. decode the prompt (seq 0), save  (a) the seq-0 state FILE  and  (b) the whole-context state BUFFER
//   2. decode one more token t at position n -> reference logits A
//   3. clear memory, restore (a) from file, decode t at position n -> logits B
//   4. clear memory, restore (b) from buffer, decode t at position n -> logits C
//   5. CONTROL: clear memory, restore NOTHING, decode t at position n -> logits Z (no context)
//
// PASS: max|A-B| and max|A-C| are small (restore worked) AND max|A-Z| is large (the check can tell a restore that
// did nothing from one that worked -- without this control a model insensitive to context would pass vacuously).
// Exit code: 0 pass, 1 fail, 2 setup error.
//
// usage: llama-legacy-state-check -m model.gguf [-p prompt] [-f seqfile]

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, int pos0, bool logits_last) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos0 + (int) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = logits_last && i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(l, l + n_vocab);
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char ** argv) {
    std::string model_path, prompt = "The committee decided that the budget deadline had passed because the treasurer", seqfile = "legacy-state-check.seq";
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-m")) model_path = argv[i + 1];
        else if (!strcmp(argv[i], "-p")) prompt = argv[i + 1];
        else if (!strcmp(argv[i], "-f")) seqfile = argv[i + 1];
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf [-p prompt] [-f seqfile]\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "SETUP: model load failed\n"); return 2; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "SETUP: context creation failed\n"); return 2; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> toks(512);
    const int n = llama_tokenize(model, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n <= 1) { fprintf(stderr, "SETUP: tokenize gave %d tokens\n", n); return 2; }
    toks.resize(n);

    // 1. prompt, then save both forms
    if (decode_tokens(ctx, toks, 0, true) != 0) { fprintf(stderr, "SETUP: prompt decode failed\n"); return 2; }
    const std::vector<float> after_prompt = last_logits(ctx, n_vocab);
    llama_token t = 0;
    for (int i = 1; i < n_vocab; ++i) if (after_prompt[i] > after_prompt[t]) t = i;

    const size_t saved_file = llama_state_seq_save_file(ctx, seqfile.c_str(), 0, toks.data(), toks.size());
    std::vector<uint8_t> buf(llama_state_get_size(ctx));
    const size_t saved_buf = llama_state_get_data(ctx, buf.data(), buf.size());
    printf("saved: seq file %zu bytes, whole-state buffer %zu of %zu bytes\n", saved_file, saved_buf, buf.size());

    // 2. reference
    if (decode_tokens(ctx, {t}, n, true) != 0) { fprintf(stderr, "SETUP: reference decode failed\n"); return 2; }
    const std::vector<float> A = last_logits(ctx, n_vocab);

    llama_memory_t mem = llama_get_memory(ctx);

    // 3. restore from seq file
    llama_memory_clear(mem, true);
    std::vector<llama_token> tok_back(512);
    size_t n_back = 0;
    const size_t loaded_file = llama_state_seq_load_file(ctx, seqfile.c_str(), 0, tok_back.data(), tok_back.size(), &n_back);
    if (decode_tokens(ctx, {t}, n, true) != 0) { fprintf(stderr, "decode after seq-file restore failed\n"); return 1; }
    const std::vector<float> B = last_logits(ctx, n_vocab);

    // 4. restore from whole-state buffer
    llama_memory_clear(mem, true);
    const size_t loaded_buf = llama_state_set_data(ctx, buf.data(), buf.size());
    if (decode_tokens(ctx, {t}, n, true) != 0) { fprintf(stderr, "decode after buffer restore failed\n"); return 1; }
    const std::vector<float> C = last_logits(ctx, n_vocab);

    // 5. control: no restore
    llama_memory_clear(mem, true);
    if (decode_tokens(ctx, {t}, n, true) != 0) { fprintf(stderr, "decode for the control failed\n"); return 1; }
    const std::vector<float> Z = last_logits(ctx, n_vocab);

    const double dB = max_abs_diff(A, B), dC = max_abs_diff(A, C), dZ = max_abs_diff(A, Z);
    printf("loaded: seq file %zu bytes (%zu tokens back), buffer %zu bytes\n", loaded_file, n_back, loaded_buf);
    printf("max|A-B| seq-file restore   = %.6g\n", dB);
    printf("max|A-C| buffer restore     = %.6g\n", dC);
    printf("max|A-Z| NO restore (ctrl)  = %.6g\n", dZ);

    const double tol = 1e-3 * std::max(1.0, dZ);
    const bool control_ok = dZ > 0.5;
    const bool pass = control_ok && dB <= tol && dC <= tol && loaded_file > 0 && loaded_buf > 0;
    printf("control discriminates: %s | RESULT: %s\n", control_ok ? "yes" : "NO (check is vacuous)", pass ? "PASS" : "FAIL");

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    std::remove(seqfile.c_str());
    return pass ? 0 : 1;
}
