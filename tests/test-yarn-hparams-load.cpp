// Loader test for the YaRN keys <arch>.rope.scaling.yarn_truncate and <arch>.rope.scaling.yarn_mscale.
// Row S69-EXO-YARN-TRUNCATE-FALSE-AND-MSCALE-RATIO-2026-11-06-01.
//
// usage: test-yarn-hparams-load <yarn_keys.gguf> <yarn_nokeys.gguf>      (made by tests/yarn-truncate-check/make_tiny_yarn_gguf.py)
//   no argument -> SKIP, exit 0.
// The files are tiny zero-weight models, fully loaded (vocab_only returns before the rope keys are read): this proves the loader reads the keys into llama_hparams and that an old GGUF
// without them keeps the historical defaults (truncate = true, mscale = 0 = "assume 1.0 / mscale_all_dim").
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"
#include "../src/llama-yarn.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static llama_model * load(const char * path) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    return llama_model_load_from_file(path, mp);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::printf("SKIP: usage %s <yarn_keys.gguf> <yarn_nokeys.gguf>\n", argv[0]);
        return 0;
    }
    llama_backend_init();

    llama_model * with = load(argv[1]);
    CHECK(with != nullptr, "load %s", argv[1]);
    if (with) {
        const llama_hparams & hp = with->hparams;
        CHECK(with->arch == LLM_ARCH_LLAMA, "arch llama");
        std::printf("sanity: n_embd=%u n_layer=%u n_ctx_train=%u rope_freq_base_train=%g\n", hp.n_embd, hp.n_layer, hp.n_ctx_train, hp.rope_freq_base_train);
        CHECK(hp.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_YARN, "scaling type yarn (%d)", (int) hp.rope_scaling_type_train);
        CHECK(std::fabs(hp.rope_freq_scale_train - 1.0f / 32.0f) < 1e-7f, "freq scale %g", hp.rope_freq_scale_train);
        CHECK(!hp.rope_yarn_truncate, "yarn_truncate=false must be read (got %d)", (int) hp.rope_yarn_truncate);
        CHECK(std::fabs(hp.rope_yarn_mscale - 0.707f) < 1e-6f, "yarn_mscale 0.707 must be read (got %g)", hp.rope_yarn_mscale);
        std::printf("with keys   : truncate=%d mscale=%g factor=%g\n", (int) hp.rope_yarn_truncate, hp.rope_yarn_mscale, 1.0f / hp.rope_freq_scale_train);
        llama_model_free(with);
    }

    llama_model * without = load(argv[2]);
    CHECK(without != nullptr, "load %s", argv[2]);
    if (without) {
        const llama_hparams & hp = without->hparams;
        CHECK(hp.rope_yarn_truncate, "no key -> truncate must default to true (got %d)", (int) hp.rope_yarn_truncate);
        CHECK(hp.rope_yarn_mscale == 0.0f, "no key -> mscale must default to 0 (got %g)", hp.rope_yarn_mscale);
        std::printf("without keys: truncate=%d mscale=%g\n", (int) hp.rope_yarn_truncate, hp.rope_yarn_mscale);
        llama_model_free(without);
    }

    // the attention factor the context derives from these hparams (llama-context.cpp), with mscale_all_dim = 1.0
    CHECK(std::fabs(llama_yarn_attn_factor_ratio(32.0f, 0.707f, 1.0f) -
                    (0.1f * 0.707f * std::log(32.0f) + 1.0f) / (0.1f * std::log(32.0f) + 1.0f)) < 1e-6f, "ratio from loaded values");

    llama_backend_free();
    std::printf(g_fails ? "test-yarn-hparams-load: %d FAILED\n" : "test-yarn-hparams-load: OK\n", g_fails);
    return g_fails ? 1 : 0;
}
