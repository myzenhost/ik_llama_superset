// A KV-cache type that carries per-row metadata (ggml type traits row_meta_size > 0; today only q8_KV is accepted by the
// -ctk/-ctv parsers) must be REFUSED at context creation with a clear message, not accepted and then crash at graph build
// (V side: GGML_ASSERT data_size + view_offs <= ggml_nbytes(view_src)) or at the first decode (K side: SIGSEGV in
// quantize_row_q8_KV), because the per-head sub-row views of the cache are sized with ggml_row_size(type, n_embd_head), which
// repeats the metadata block for every head.
//
// Row S69-ENGREV-THIREUS-ROWMETA-KV-TYPE-REFUSE-2026-10-27-01; upstream Thireus/ik_llama.cpp f228595b (MIT).
//
// usage: test-kv-row-meta-refuse <model.gguf>
//   any small GGUF works; tests/make_tiny_llama_gguf.py <tree-root> out.gguf builds one without a download.
// PASS criteria: every q8_KV combination returns a null context and logs "cannot be used as a K/V-cache type"; the controls
// (f16/f16 with FA off and on, bf16 K) still create a context. A block-quantized control (q8_0) needs a model whose head size
// is a multiple of 32, which the tiny GGUF is not; the llama-completion runs in the findings file cover q8_0/q8_0 on Qwen3-0.6B.
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>

static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        g_log += text;
    }
}

struct kv_case {
    const char *            name;
    ggml_type               k;
    ggml_type               v;
    llama_flash_attn_type   fa;
    bool                    expect_refused;
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 2;
    }

    const kv_case cases[] = {
        { "k=q8_KV v=f16 fa=off",    GGML_TYPE_Q8_KV, GGML_TYPE_F16,  LLAMA_FLASH_ATTN_TYPE_DISABLED, true  },
        { "k=q8_KV v=f16 fa=on",     GGML_TYPE_Q8_KV, GGML_TYPE_F16,  LLAMA_FLASH_ATTN_TYPE_ENABLED,  true  },
        { "k=f16 v=q8_KV fa=on",     GGML_TYPE_F16,   GGML_TYPE_Q8_KV, LLAMA_FLASH_ATTN_TYPE_ENABLED, true  },
        { "k=q8_KV v=q8_KV fa=on",   GGML_TYPE_Q8_KV, GGML_TYPE_Q8_KV, LLAMA_FLASH_ATTN_TYPE_ENABLED, true  },
        { "k=q8_KV v=q8_KV fa=auto", GGML_TYPE_Q8_KV, GGML_TYPE_Q8_KV, LLAMA_FLASH_ATTN_TYPE_AUTO,    true  },
        { "control f16/f16 fa=off",  GGML_TYPE_F16,   GGML_TYPE_F16,  LLAMA_FLASH_ATTN_TYPE_DISABLED, false },
        { "control f16/f16 fa=on",   GGML_TYPE_F16,   GGML_TYPE_F16,  LLAMA_FLASH_ATTN_TYPE_ENABLED,  false },
        { "control bf16/f16 fa=off", GGML_TYPE_BF16,  GGML_TYPE_F16,  LLAMA_FLASH_ATTN_TYPE_DISABLED, false },
    };

    int fails = 0;
    for (const auto & c : cases) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 256;
        cp.n_batch         = 64;
        cp.n_ubatch        = 64;
        cp.type_k          = c.k;
        cp.type_v          = c.v;
        cp.flash_attn_type = c.fa;
        g_log.clear();
        llama_context * ctx = llama_init_from_model(model, cp);
        const bool refused = ctx == nullptr;
        const bool msg     = g_log.find("cannot be used as a K/V-cache type") != std::string::npos;
        bool ok = true;
        if (c.expect_refused) {
            ok = refused && msg;
        } else {
            ok = !refused;
        }
        printf("%-4s %-34s ctx=%s msg=%s\n", ok ? "PASS" : "FAIL", c.name, refused ? "null" : "created", msg ? "yes" : "no");
        if (!ok) {
            fails++;
            if (!g_log.empty()) {
                printf("       log: %s\n", g_log.c_str());
            }
        }
        if (ctx) {
            llama_free(ctx);
        }
    }

    llama_model_free(model);
    llama_backend_free();
    printf("%s (%d failing)\n", fails ? "FAILED" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
