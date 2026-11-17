// KV state basis tag: a sequence state saved under k_cache_hadamard must not restore into a context
// without it (and the reverse), because the cached K rows are in a rotated basis.
//
// Before the tag, the restore passed every type / row-size / v_trans check, and the next token's
// logits after the restore differed from the reference: wrong attention with no error. With the tag
// the cross-basis restore is refused, and a same-basis restore still reproduces the reference logits.
// The second check is the positive control, so a refusal of EVERY restore cannot pass.
//
// RESEARCH-DONE: PrismML llama.cpp fork, kv_mean_center.k_rot GGUF key: "the loader REFUSES a bias whose
// basis does not match the inference-time rotation state" (design read, no code copied; row
// S69-PRISMML-FORK-SOURCE-REVIEW-ADOPT-01).
//
// Usage: test-state-kv-basis [model.gguf]  (or env LLAMA_TEST_STATE_BASIS_MODEL)
// Exit 77 (SKIPPED) when no model is available. Needs a model whose K head size allows a Hadamard
// transform (power of two >= 64, e.g. gemma-3-270m head 256).

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * k_default_model = "models/gguf/gemma3-270m/gemma-3-270m-Q8_0.gguf"; // portability-ok: optional local model, test skips when absent

static llama_context * make_ctx(llama_model * model, bool khad) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx            = 512;
    cp.n_batch          = 512;
    cp.n_ubatch         = 512;
    cp.n_threads        = 4;
    cp.n_threads_batch  = 4;
    cp.type_k           = GGML_TYPE_Q8_0;   // the K Hadamard transform applies only to a quantized K cache (llama.cpp context-param checks)
    cp.k_cache_hadamard = khad;
    return llama_init_from_model(model, cp);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos p0) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i]     = toks[i];
        b.pos[i]       = p0 + (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(l, l + n_vocab);
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::fmax(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}

int main(int argc, char ** argv) {
    std::string path = argc > 1 ? argv[1] : "";
    if (path.empty() && std::getenv("LLAMA_TEST_STATE_BASIS_MODEL")) path = std::getenv("LLAMA_TEST_STATE_BASIS_MODEL");
    if (path.empty()) path = k_default_model;
    if (FILE * f = std::fopen(path.c_str(), "rb")) { std::fclose(f); } else {
        std::printf("SKIP: no model at %s\n", path.c_str());
        return 77;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) { std::printf("FAIL: model load\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // fixed token ids: no tokenizer dependence, long enough that attention over the cache matters
    std::vector<llama_token> prompt;
    for (int i = 0; i < 96; ++i) prompt.push_back((llama_token) (1000 + (i * 37) % 5000));
    const std::vector<llama_token> next = { (llama_token) 4242 };

    int failures = 0;
    auto check = [&](bool ok, const char * what) {
        std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };

    // writer: Hadamard ON. Reference = logits of `next` decoded on the intact sequence.
    llama_context * src = make_ctx(model, true);
    if (!src || !decode_tokens(src, prompt, 0)) { std::printf("FAIL: source decode\n"); return 1; }
    std::vector<uint8_t> state(llama_state_seq_get_size(src, 0, 0));
    const size_t n_saved = llama_state_seq_get_data(src, state.data(), state.size(), 0, 0);
    check(n_saved == state.size() && n_saved > 0, "saved the Hadamard-on sequence state");
    if (!decode_tokens(src, next, (llama_pos) prompt.size())) { std::printf("FAIL: reference decode\n"); return 1; }
    const std::vector<float> ref = last_logits(src, n_vocab);

    // positive control: same basis restores and reproduces the reference
    {
        llama_context * dst = make_ctx(model, true);
        const size_t n = llama_state_seq_set_data(dst, state.data(), state.size(), 0, 0);
        check(n > 0, "same-basis restore (khad on -> on) is accepted");
        if (n > 0 && decode_tokens(dst, next, (llama_pos) prompt.size())) {
            const double d = max_abs_diff(ref, last_logits(dst, n_vocab));
            std::printf("       same-basis max|dlogit| = %.6g\n", d);
            check(d < 1e-3, "same-basis restore reproduces the reference logits");
        } else {
            check(false, "same-basis restore decodes");
        }
        llama_free(dst);
    }

    // the defect: cross-basis restore must be refused (before the tag it was accepted, and the logits drifted)
    {
        llama_context * dst = make_ctx(model, false);
        const size_t n = llama_state_seq_set_data(dst, state.data(), state.size(), 0, 0);
        if (n > 0 && decode_tokens(dst, next, (llama_pos) prompt.size())) {
            std::printf("       cross-basis restore ACCEPTED; max|dlogit| vs reference = %.6g\n",
                    max_abs_diff(ref, last_logits(dst, n_vocab)));
        }
        check(n == 0, "cross-basis restore (khad on -> off) is refused");
        llama_free(dst);
    }

    // and the reverse direction
    {
        llama_context * w = make_ctx(model, false);
        if (!w || !decode_tokens(w, prompt, 0)) { std::printf("FAIL: plain decode\n"); return 1; }
        std::vector<uint8_t> st(llama_state_seq_get_size(w, 0, 0));
        llama_state_seq_get_data(w, st.data(), st.size(), 0, 0);
        llama_context * dst = make_ctx(model, true);
        const size_t n = llama_state_seq_set_data(dst, st.data(), st.size(), 0, 0);
        check(n == 0, "cross-basis restore (khad off -> on) is refused");
        llama_context * dst2 = make_ctx(model, false);
        check(llama_state_seq_set_data(dst2, st.data(), st.size(), 0, 0) > 0, "control: khad off -> off is accepted");
        llama_free(dst); llama_free(dst2); llama_free(w);
    }

    llama_free(src);
    llama_model_free(model);
    llama_backend_free();
    std::printf("RESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
