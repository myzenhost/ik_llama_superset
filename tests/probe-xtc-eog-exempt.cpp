// Probe: can the XTC sampler remove an end-of-generation token? (row S69-EXLLAMAV2-PICK-XTC-EXEMPT-EOG-NEWLINE-2026-09-29-01)
// Candidates: the model's EOS at p = 0.6, one other token at p = 0.3, the rest tiny. xtc_probability 1, threshold 0.1, so both cross.
// Unguarded XTC keeps only the LAST crossing token and removes the first, i.e. it removes the EOS. With the EOG exemption the EOS must
// survive in every one of 200 seeds. Controls: (a) two non-EOG tokens crossing are still cut (the sampler still does its job),
// (b) the check can fail: before the exemption this probe reports EOS removed in 200/200.
// Not in CMake on purpose -- links the static archives directly. Build (mingw, from the fork root, build-static):
//   g++ -O1 -std=c++17 -fopenmp -I include -I ggml/include tests/probe-xtc-eog-exempt.cpp -o probe-xtc.exe
//       -Wl,--start-group build-static/src/libllama.a build-static/ggml/src/ggml.a build-static/ggml/src/ggml-cpu.a
//       build-static/ggml/src/ggml-base.a -Wl,--end-group -fopenmp -lstdc++
// Model with a vocab: python tests/lfm2-forward-check/make_tiny_vocab_gguf.py <fork-root> <out-dir> (dense.gguf; make_tiny_gguf.py output has NO vocab and asserts).
// Usage: probe-xtc.exe <out-dir>/dense.gguf      Expected after the change: "5 checks, 0 failed".
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, what) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: %s\n", what); } } while (0)

static bool contains(const llama_token_data_array & a, llama_token id) {
    for (size_t i = 0; i < a.size; ++i) {
        if (a.data[i].id == id) return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("usage: %s model.gguf\n", argv[0]); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::printf("model load failed\n"); return 2; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::printf("context failed\n"); return 2; }

    const int32_t n_vocab = llama_n_vocab(model);
    const llama_token eos = llama_token_eos(model);
    const llama_token other = (eos == 5) ? 6 : 5;
    const llama_token other2 = (eos == 7 || other == 7) ? 8 : 7;
    CHECK(llama_token_is_eog(model, eos), "the model's EOS token is an EOG token");

    auto make = [&](llama_token a, float pa, llama_token b, float pb, std::vector<llama_token_data> & store) {
        store.assign(n_vocab, {0, -30.0f, 0.0f});
        for (int32_t i = 0; i < n_vocab; ++i) store[i].id = i;
        store[a].logit = std::log(pa);
        store[b].logit = std::log(pb);
        return llama_token_data_array{store.data(), store.size(), -1, false};
    };

    // first token whose decoded piece contains a newline (exempt like EOG)
    // SOURCE-READ: upstream/exllamav2/exllamav2/generator/sampler.py:277-302 (default XTC mask: pieces containing a newline + special tokens; MIT, behaviour only)
    llama_token nl_tok = -1;
    for (llama_token t = 0; t < n_vocab && nl_tok < 0; ++t) {
        char b[64];
        const int32_t n = llama_token_to_piece(model, t, b, (int32_t) sizeof(b), 0, true);
        for (int32_t k = 0; k < n; ++k) {
            if (b[k] == '\n') { nl_tok = t; break; }
        }
    }
    CHECK(nl_tok >= 0 && nl_tok != eos && nl_tok != other && nl_tok != other2, "the vocab has a distinct token whose piece contains a newline");

    int eos_removed = 0, ctrl_cut = 0, nl_removed = 0;
    const int N = 200;
    for (int seed = 0; seed < N; ++seed) {
        llama_set_rng_seed(ctx, 1000 + seed);
        std::vector<llama_token_data> st1, st2;
        llama_token_data_array a1 = make(eos, 0.6f, other, 0.3f, st1);
        llama_sample_xtc(ctx, &a1, 1.0f, 0.1f, 1);
        eos_removed += !contains(a1, eos);
        // control: two NON-EOG tokens crossing; the more probable one is removed, the less probable stays
        llama_token_data_array a2 = make(other, 0.6f, other2, 0.3f, st2);
        llama_sample_xtc(ctx, &a2, 1.0f, 0.1f, 1);
        ctrl_cut += !contains(a2, other) && contains(a2, other2);
        // newline token at 0.6 with another token at 0.3: the newline token must survive
        if (nl_tok >= 0) {
            std::vector<llama_token_data> st3;
            llama_token_data_array a3 = make(nl_tok, 0.6f, other, 0.3f, st3);
            llama_sample_xtc(ctx, &a3, 1.0f, 0.1f, 1);
            nl_removed += !contains(a3, nl_tok);
        }
    }
    std::printf("newline token (id %d) removed in %d / %d seeds\n", (int) nl_tok, nl_removed, N);
    CHECK(nl_removed == 0, "a newline-piece token survives XTC in every seed");
    std::printf("EOS removed in %d / %d seeds; control (two non-EOG tokens) cut as expected in %d / %d\n", eos_removed, N, ctrl_cut, N);
    CHECK(eos_removed == 0, "EOS survives XTC in every seed");
    CHECK(ctrl_cut == N, "control: two non-EOG crossing tokens are still cut");

    std::printf("%d checks, %d failed\n", g_checks, g_fails);
    llama_free(ctx);
    llama_model_free(model);
    return g_fails ? 1 : 0;
}
