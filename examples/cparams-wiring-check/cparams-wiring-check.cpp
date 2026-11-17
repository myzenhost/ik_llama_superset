// cparams-wiring-check: does every fork context parameter reach llama_cparams?
//
// WIRE-68 (0940691c3) moved context creation into llama_context's constructor, which copied only
// upstream's fields. 32 fork fields of llama_cparams were then assigned nowhere in src/ and stayed
// zero from `cparams = {}`, so the CLI accepted -khad, -fmoe, -mla, -ser, ... and the context ignored
// them. This check sets each such field in llama_context_params to a NON-DEFAULT value, creates a
// context, and reads what arrived via llama_context::get_cparams().
//
// Every expectation is either the value that was set, or -- for the documented fixups -- the value
// the fixup must produce on this model (the fixup is itself under test):
//   mla_attn            -> 0     (non-MLA model)
//   kvflash_pool_tokens -> 0     (refused until the paged-KV integration is finished)
//   mtp                 -> false (arch not on the MTP allowlist)
//
// Exit code: 0 all fields arrived, 1 at least one did not, 2 setup error.
// usage: llama-cparams-wiring-check -m model.gguf   (a small non-MLA, non-MTP model, e.g. qwen2.5-0.5b)

#include "llama.h"
#include "llama-context.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

static int n_fail = 0;
static int n_pass = 0;

template <typename T>
static void expect(const char * name, T got, T want) {
    const bool ok = got == want;
    std::printf("  %-28s got %-12s want %-12s %s\n", name, std::to_string(got).c_str(), std::to_string(want).c_str(),
                ok ? "ok" : "MISMATCH");
    (ok ? n_pass : n_fail)++;
}

static void expect_f(const char * name, float got, float want) {
    const bool ok = std::fabs(got - want) < 1e-6f;
    std::printf("  %-28s got %-12g want %-12g %s\n", name, got, want, ok ? "ok" : "MISMATCH");
    (ok ? n_pass : n_fail)++;
}

static void expect_t(const char * name, ggml_type got, ggml_type want) {
    const bool ok = got == want;
    std::printf("  %-28s got %-12s want %-12s %s\n", name, ggml_type_name(got), ggml_type_name(want), ok ? "ok" : "MISMATCH");
    (ok ? n_pass : n_fail)++;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "-m") == 0) model_path = argv[i + 1];
    }
    if (!model_path) {
        std::fprintf(stderr, "usage: %s -m model.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        std::fprintf(stderr, "failed to load %s\n", model_path);
        return 2;
    }

    static const char cuda_params_str[] = "fusion=1";

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                       = 512;
    cp.n_batch                     = 512;
    cp.n_ubatch                    = 512;
    cp.n_threads                   = 4;
    cp.n_threads_batch             = 4;
    cp.flash_attn_type             = LLAMA_FLASH_ATTN_TYPE_ENABLED;  // quantized V needs FA
    cp.type_k                      = GGML_TYPE_Q4_0;                 // khad needs a quantized K
    cp.type_v                      = GGML_TYPE_Q4_0;                 // vhad needs a quantized V
    // every field below is set to a value different from llama_context_default_params()
    cp.defrag_thold                = 0.25f;
    cp.snapkv_cap                  = 64;
    cp.snapkv_window               = 16;
    cp.h2o_heavy                   = 32;
    cp.h2o_recent                  = 8;
    cp.kvflash_pool_tokens         = 128;
    cp.offload_attn_page           = 64;
    cp.offload_attn_budget         = 0.5f;
    cp.offload_attn_selector       = 1;
    cp.offload_attn_sink           = 2;
    cp.offload_attn_recent         = 3;
    cp.offload_attn_dense_layers   = 5;
    cp.offload_attn_refresh        = 16;
    cp.offload_attn_scan           = 1;
    cp.offload_attn_min_tokens     = 32;
    cp.offload_attn_head_reduce    = 1;
    cp.mla_attn                    = 2;
    cp.attn_max_batch              = 128;
    cp.fused_moe_up_gate           = false;
    cp.grouped_expert_routing      = true;
    cp.fused_up_gate               = false;
    cp.fused_mmad                  = false;
    cp.rope_cache                  = true;
    cp.graph_reuse                 = false;
    cp.dsa                         = true;
    cp.fused_idx_topk              = false;
    cp.dsa_top_k                   = 7;
    cp.k_cache_hadamard            = true;
    cp.v_cache_hadamard            = true;
    cp.split_mode_graph_scheduling = true;
    cp.scheduler_async             = true;
    cp.min_experts                 = 3;
    cp.thresh_experts              = 0.5f;
    cp.cuda_params                 = (void *) cuda_params_str;
    cp.mtp                         = true;
    cp.type_reduce                 = GGML_TYPE_BF16;
    cp.type_graph_attn             = GGML_TYPE_F32;
    cp.idx_type_k                  = GGML_TYPE_Q8_0;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 2;
    }
    const llama_cparams & c = ctx->get_cparams();

    std::printf("cparams-wiring-check: %s\n", model_path);
    expect_f("defrag_thold",               c.defrag_thold, 0.25f);
    expect("snapkv_cap",                   c.snapkv_cap, 64);
    expect("snapkv_window",                c.snapkv_window, 16);
    expect("h2o_heavy",                    c.h2o_heavy, 32);
    expect("h2o_recent",                   c.h2o_recent, 8);
    expect("kvflash_pool_tokens (fixup)",  c.kvflash_pool_tokens, 0);
    expect("offload_attn_page",            c.offload_attn_page, 64);
    expect_f("offload_attn_budget",        c.offload_attn_budget, 0.5f);
    expect("offload_attn_selector",        c.offload_attn_selector, 1);
    expect("offload_attn_sink",            c.offload_attn_sink, 2);
    expect("offload_attn_recent",          c.offload_attn_recent, 3);
    expect("offload_attn_dense_layers",    c.offload_attn_dense_layers, 5);
    expect("offload_attn_refresh",         c.offload_attn_refresh, 16);
    expect("offload_attn_scan",            c.offload_attn_scan, 1);
    expect("offload_attn_min_tokens",      c.offload_attn_min_tokens, 32);
    expect("offload_attn_head_reduce",     c.offload_attn_head_reduce, 1);
    expect("mla_attn (fixup)",             c.mla_attn, 0);
    expect("attn_max_batch",               c.attn_max_batch, 128);
    expect("fused_moe_up_gate",            (int) c.fused_moe_up_gate, 0);
    expect("grouped_expert_routing",       (int) c.grouped_expert_routing, 1);
    expect("fused_up_gate",                (int) c.fused_up_gate, 0);
    expect("fused_mmad",                   (int) c.fused_mmad, 0);
    expect("rope_cache",                   (int) c.rope_cache, 1);
    expect("graph_reuse",                  (int) c.graph_reuse, 0);
    expect("dsa",                          (int) c.dsa, 1);
    expect("fused_idx_topk",               (int) c.fused_idx_topk, 0);
    expect("dsa_top_k",                    c.dsa_top_k, 7);
    expect("k_cache_hadamard",             (int) c.k_cache_hadamard, 1);
    expect("v_cache_hadamard",             (int) c.v_cache_hadamard, 1);
    expect("split_mode_graph_scheduling",  (int) c.split_mode_graph_scheduling, 1);
    expect("scheduler_async",              (int) c.scheduler_async, 1);
    expect("min_experts",                  c.min_experts, 3);
    expect_f("thresh_experts",             c.thresh_experts, 0.5f);
    expect("cuda_params (pointer)",        (int) (c.cuda_params == (void *) cuda_params_str), 1);
    expect("mtp (fixup)",                  (int) c.mtp, 0);
    expect_t("reduce_type",                c.reduce_type, GGML_TYPE_BF16);
    expect_t("graph_attn_precision",       c.graph_attn_precision, GGML_TYPE_F32);
    expect_t("idx_type_k",                 c.idx_type_k, GGML_TYPE_Q8_0);

    std::printf("RESULT: %d fields arrived, %d did not -> %s\n", n_pass, n_fail, n_fail == 0 ? "PASS" : "FAIL");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}
