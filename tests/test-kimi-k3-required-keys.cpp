// Kimi-K3 refuses a GGUF that lacks (or zeroes) the hyperparameters the graph cannot run without.
//
// Usage: test-kimi-k3-required-keys [<fixture-dir>]
//   no argument -> prints SKIP and exits 0 (safe in a suite on a machine with no fixture)
//   <dir>       -> <dir> holds the files made by tests/make_tiny_kimi_k3_meta.py (metadata only, no weights):
//                    k3_no_latent.gguf   (expert_latent_length absent)  -> load refused, the log names expert_latent_length
//                    k3_zero_latent.gguf (expert_latent_length = 0)     -> load refused, the log names expert_latent_length
//                    k3_zero_block.gguf  (attn_res.block_size = 0)      -> load refused, the log names attn_res.block_size
//
// BEFORE the change the key was optional with a silent default, so the load went on to the vocab and tensor stage and was
// refused there with a message about something else (or, with a full file, loaded and produced garbage). The refusal text is
// the assertion, not just the null model.
#include "llama.h"

#include <cstdio>
#include <string>

static std::string g_log;
static void capture_log(ggml_log_level, const char * text, void *) { g_log += text; }

static int g_fails = 0;

static void expect_refusal(const std::string & dir, const char * file, const char * must_name) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_log.clear();
    llama_log_set(capture_log, nullptr);
    llama_model * model = llama_model_load_from_file((dir + "/" + file).c_str(), mp);
    llama_log_set(nullptr, nullptr);
    const bool refused = model == nullptr;
    const bool named   = g_log.find(must_name) != std::string::npos;
    std::printf("%-20s refused=%d names '%s'=%d\n", file, (int) refused, must_name, (int) named);
    if (!refused || !named) {
        ++g_fails;
        std::printf("FAIL: %s: log was:\n%s\n", file, g_log.c_str());
    }
    if (model) { llama_model_free(model); }
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("SKIP: no fixture dir given (usage: %s <dir with k3_*.gguf>)\n", argv[0]); return 0; }
    llama_backend_init();
    const std::string dir = argv[1];
    // phrases that appear only in the refusal, NOT in the loader's metadata dump (which names every key present)
    expect_refusal(dir, "k3_no_latent.gguf",   "key not found in model: kimi-k3.expert_latent_length");
    expect_refusal(dir, "k3_zero_latent.gguf", "Kimi-K3 requires expert_latent_length > 0");
    expect_refusal(dir, "k3_zero_block.gguf",  "Kimi-K3 requires attn_res.block_size > 0");
    std::printf("%d failed\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
