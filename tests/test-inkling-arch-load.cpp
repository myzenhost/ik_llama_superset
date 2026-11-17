// S69 P4: the Inkling architecture LOADS (hparams + tensors + vocab) from a converted random-weight GGUF.
//
// Usage: test-inkling-arch-load [<inkling.gguf> [--try-decode]]
//   no argument  -> prints SKIP and exits 0 (safe in a suite on a machine with no fixture)
//   <gguf>       -> loads it, checks every hparam, every layer's tensors (presence, shape, type) and the vocab
//   --try-decode -> additionally creates a context and decodes one token (P1 landed the graph; before it, this aborted
//                   with the "inkling: ... not implemented yet" message).
//
// The gguf is made by tests/inkling-arch-check/make_tiny_inkling.py (random weights, NEVER the real Inkling weights). The
// constants below must match that script; check_inkling_gguf.py checks the same facts from the Python side.
// Works on the F32 file and on llama-quantize outputs of it (shortconv_* / attn_rel_proj / ffn_gscale must stay F32 in both).
#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, ...) do { ++g_checks; if (!(cond)) { ++g_fails; std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// ---- fixture constants (make_tiny_inkling.py) ----
static const int      N_LAYER = 4, N_EMBD = 256, N_HEAD = 4, HEAD_DIM = 64, D_REL = 16, K = 4;
static const int      REL_EXTENT = 48, REL_EXTENT_SWA = 24, N_DENSE = 1;
static const int      N_EXPERT = 2, N_EXPERT_USED = 1, N_SHEXP = 1, FF_DENSE = 512, FF_EXP = 256;
static const int      V_PADDED = 320, V_UNPADDED = 262, EOS_ID = 261;
static const bool     LOCAL[N_LAYER]   = { true, true, false, true };
static const int      N_HEAD_KV[N_LAYER] = { 4, 4, 2, 4 };

// captures llama.cpp log lines so a refusal's reason can be asserted, not just its nullptr
static std::string g_log;
static void capture_log(ggml_log_level, const char * text, void *) { g_log += text; }

static bool is_f32(const ggml_tensor * t) { return t && t->type == GGML_TYPE_F32; }

static void check_shape(const ggml_tensor * t, const char * what, int il, int64_t n0, int64_t n1, int64_t n2) {
    if (!t) { CHECK(false, "blk.%d %s is missing", il, what); return; }
    CHECK(t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2 && t->ne[3] == 1,
          "blk.%d %s ne = {%lld,%lld,%lld,%lld}, want {%lld,%lld,%lld,1}", il, what,
          (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
          (long long) n0, (long long) n1, (long long) n2);
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::printf("SKIP: no gguf given (usage: %s <inkling.gguf> [--try-decode])\n", argv[0]); return 0; }
    const bool try_decode = argc > 2 && std::strcmp(argv[2], "--try-decode") == 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::printf("FAIL: model load returned null\n"); return 2; }

    // ---- arch + hparams ----
    char buf[256];
    CHECK(model->arch == LLM_ARCH_INKLING, "arch is LLM_ARCH_INKLING");
    CHECK(std::strcmp(llama_model_arch_name(model->arch), "inkling") == 0, "arch name is 'inkling'");
    const llama_hparams & hp = model->hparams;
    CHECK(hp.n_layer == N_LAYER, "n_layer %u", hp.n_layer);
    CHECK(hp.n_embd == N_EMBD, "n_embd %u", hp.n_embd);
    CHECK(hp.n_vocab == V_PADDED, "n_vocab %u (the PADDED size)", hp.n_vocab);
    CHECK(hp.inkling_unpadded_n_vocab == V_UNPADDED, "unpadded vocab %u", hp.inkling_unpadded_n_vocab);
    CHECK(hp.inkling_d_rel == D_REL && hp.inkling_rel_extent == REL_EXTENT && hp.inkling_rel_extent_swa == REL_EXTENT_SWA,
          "rel keys %u %u %u", hp.inkling_d_rel, hp.inkling_rel_extent, hp.inkling_rel_extent_swa);
    CHECK(hp.n_shortconv_l_cache == K, "shortconv kernel %u", hp.n_shortconv_l_cache);
    CHECK(hp.n_layer_dense_lead == N_DENSE, "dense blocks %u", hp.n_layer_dense_lead);
    CHECK(hp.n_expert == N_EXPERT && hp.n_expert_used == N_EXPERT_USED && hp.n_expert_shared == N_SHEXP,
          "experts %u/%u shared %u", hp.n_expert, hp.n_expert_used, hp.n_expert_shared);
    CHECK(hp.n_ff_exp == FF_EXP, "n_ff_exp %u", hp.n_ff_exp);
    CHECK(hp.inkling_log_n_floor == 32 && std::fabs(hp.inkling_log_alpha - 0.1f) < 1e-6f, "log scaling %u %g", hp.inkling_log_n_floor, hp.inkling_log_alpha);
    CHECK(std::fabs(hp.f_logit_scale - 1.0f / 8.0f) < 1e-7f, "f_logit_scale %g == 1/logit_scale_denom", hp.f_logit_scale);
    CHECK(std::fabs(hp.expert_weights_scale - 2.5f) < 1e-6f, "route scale %g", hp.expert_weights_scale);
    CHECK(hp.expert_gating_func == LLM_EXPERT_GATING_FUNC_SIGMOID, "sigmoid gating");
    CHECK(hp.n_swa == REL_EXTENT_SWA && hp.swa_type == LLAMA_SWA_TYPE_STANDARD, "sliding window %u type %d", hp.n_swa, (int) hp.swa_type);
    CHECK(hp.rope_type == LLAMA_ROPE_TYPE_NONE, "rope type NONE (is %d)", (int) hp.rope_type);
    CHECK(hp.n_embd_head_k(0) == HEAD_DIM && hp.n_embd_head_v(0) == HEAD_DIM, "head dim");
    CHECK(hp.n_ff(0) == FF_DENSE, "dense ff %u", hp.n_ff(0));
    for (int il = 0; il < N_LAYER; ++il) {
        CHECK(hp.is_swa(il) == LOCAL[il], "layer %d swa flag", il);
        CHECK(hp.n_head(il) == N_HEAD && hp.n_head_kv(il) == (uint32_t) N_HEAD_KV[il], "layer %d heads %u kv %u", il, hp.n_head(il), hp.n_head_kv(il));
    }

    // ---- tensors, per layer ----
    CHECK(model->tok_embd && model->tok_embd->ne[0] == N_EMBD && model->tok_embd->ne[1] == V_PADDED, "token_embd");
    check_shape(model->tok_norm, "token_embd_norm", -1, N_EMBD, 1, 1);
    CHECK(is_f32(model->tok_norm), "token_embd_norm is F32");
    check_shape(model->output_norm, "output_norm", -1, N_EMBD, 1, 1);
    CHECK(model->output && model->output->ne[0] == N_EMBD && model->output->ne[1] == V_PADDED, "output.weight is its own tensor [n_embd, n_vocab]");
    CHECK(model->layers.size() == (size_t) N_LAYER, "layers.size %zu", model->layers.size());
    for (int il = 0; il < N_LAYER && il < (int) model->layers.size(); ++il) {
        const llama_layer & L = model->layers[il];
        const int64_t kvw = (int64_t) N_HEAD_KV[il] * HEAD_DIM;
        const int64_t rel = LOCAL[il] ? REL_EXTENT_SWA : REL_EXTENT;
        check_shape(L.attn_norm, "attn_norm", il, N_EMBD, 1, 1);
        check_shape(L.wq, "attn_q", il, N_EMBD, N_HEAD * HEAD_DIM, 1);
        check_shape(L.wk, "attn_k", il, N_EMBD, kvw, 1);
        check_shape(L.wv, "attn_v", il, N_EMBD, kvw, 1);
        check_shape(L.wr, "attn_r", il, N_EMBD, N_HEAD * D_REL, 1);
        check_shape(L.wo, "attn_output", il, N_HEAD * HEAD_DIM, N_EMBD, 1);
        check_shape(L.attn_q_norm, "attn_q_norm", il, HEAD_DIM, 1, 1);
        check_shape(L.attn_k_norm, "attn_k_norm", il, HEAD_DIM, 1, 1);
        check_shape(L.attn_rel_proj, "attn_rel_proj", il, rel, D_REL, 1);
        check_shape(L.shortconv_k, "shortconv_k", il, K, kvw, 1);
        check_shape(L.shortconv_v, "shortconv_v", il, K, kvw, 1);
        check_shape(L.shortconv_attn, "shortconv_attn", il, K, N_EMBD, 1);
        check_shape(L.shortconv_mlp, "shortconv_mlp", il, K, N_EMBD, 1);
        check_shape(L.ffn_norm, "ffn_norm", il, N_EMBD, 1, 1);
        check_shape(L.ffn_gscale, "ffn_gscale", il, 1, 1, 1);
        // the F32-only tensors stay F32 whatever the file type (they are quantisation exemptions)
        CHECK(is_f32(L.attn_rel_proj) && is_f32(L.shortconv_k) && is_f32(L.shortconv_v) && is_f32(L.shortconv_attn) && is_f32(L.shortconv_mlp) && is_f32(L.ffn_gscale),
              "layer %d rel-proj / shortconv / gscale tensors are F32", il);
        if (il < N_DENSE) {
            check_shape(L.ffn_gate, "ffn_gate", il, N_EMBD, FF_DENSE, 1);
            check_shape(L.ffn_up, "ffn_up", il, N_EMBD, FF_DENSE, 1);
            check_shape(L.ffn_down, "ffn_down", il, FF_DENSE, N_EMBD, 1);
            CHECK(!L.ffn_gate_exps && !L.ffn_gate_inp && !L.ffn_gate_shexp, "dense layer %d has no MoE tensors", il);
        } else {
            check_shape(L.ffn_gate_inp, "ffn_gate_inp", il, N_EMBD, N_EXPERT + N_SHEXP, 1);
            check_shape(L.ffn_exp_probs_b, "exp_probs_b", il, N_EXPERT, 1, 1);
            check_shape(L.ffn_gate_exps, "ffn_gate_exps", il, N_EMBD, FF_EXP, N_EXPERT);
            check_shape(L.ffn_up_exps, "ffn_up_exps", il, N_EMBD, FF_EXP, N_EXPERT);
            check_shape(L.ffn_down_exps, "ffn_down_exps", il, FF_EXP, N_EMBD, N_EXPERT);
            check_shape(L.ffn_gate_shexp, "ffn_gate_shexp", il, N_EMBD, FF_EXP, N_SHEXP);
            check_shape(L.ffn_up_shexp, "ffn_up_shexp", il, N_EMBD, FF_EXP, N_SHEXP);
            check_shape(L.ffn_down_shexp, "ffn_down_shexp", il, FF_EXP, N_EMBD, N_SHEXP);
            CHECK(!L.ffn_gate && !L.ffn_up_gate_exps, "MoE layer %d has no dense ffn and no merged gate/up bank", il);
            CHECK(is_f32(L.ffn_exp_probs_b), "layer %d exp_probs_b is F32", il);
        }
    }

    // ---- vocab ----
    CHECK(llama_n_vocab(model) == V_PADDED, "llama_n_vocab %d", llama_n_vocab(model));
    CHECK(llama_token_eos(model) == EOS_ID, "eos id %d", (int) llama_token_eos(model));
    CHECK(llama_token_is_eog(model, EOS_ID), "<|content_model_end_sampling|> is EOG");
    CHECK(!llama_token_is_eog(model, EOS_ID - 1), "<|end_message|> (an intra-turn separator) is NOT EOG");
    CHECK(llama_add_bos_token(model) == 0, "no BOS is prepended");
    CHECK(llama_model_meta_val_str(model, "tokenizer.ggml.pre", buf, sizeof(buf)) > 0 && std::strcmp(buf, "inkling") == 0, "tokenizer.ggml.pre = inkling (%s)", buf);
    {
        llama_token toks[32];
        const char * text = "that the cat";
        const int n = llama_tokenize(model, text, (int) std::strlen(text), toks, 32, false, false);
        bool in_range = n > 0;
        for (int i = 0; i < n; ++i) { in_range = in_range && toks[i] >= 0 && toks[i] < V_UNPADDED; }
        CHECK(n > 0 && in_range, "tokenize('%s') -> %d tokens, all inside the unpadded vocab", text, n);
    }

    std::printf("%d checks, %d failed\n", g_checks, g_fails);

    if (try_decode && g_fails == 0) {
        // P1 landed the graph and the memory: a context is created and one token decodes. (Numerical agreement with the reference
        // is test-inkling-parity's job; this only proves the load-to-decode path no longer stops.)
        std::printf("creating a context and decoding one token\n");
        std::fflush(stdout);
        // a quantized V cache is refused at context init with the reason (Inkling cannot run flash attention, which a
        // quantized V needs); before this check the only message was the generic "requires Flash Attention".
        {
            llama_context_params qp = llama_context_default_params();
            qp.n_ctx  = 64;
            qp.type_v = GGML_TYPE_Q8_0;
            g_log.clear();
            llama_log_set(capture_log, nullptr);
            llama_context * qctx = llama_init_from_model(model, qp);
            llama_log_set(nullptr, nullptr);
            CHECK(qctx == nullptr, "context with a q8_0 V cache must be refused");
            CHECK(g_log.find("Inkling does not support a quantized V cache") != std::string::npos,
                  "refusal names Inkling and the quantized V cache; log was: %s", g_log.c_str());
            if (qctx) { llama_free(qctx); }
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 64;
        llama_context * ctx = llama_init_from_model(model, cp);
        CHECK(ctx != nullptr, "context creation");
        if (ctx) {
            llama_token tok = 1;
            llama_batch b = llama_batch_get_one(&tok, 1);
            const int rc = llama_decode(ctx, b);
            CHECK(rc == 0, "llama_decode of one token returned %d", rc);
            const float * l = rc == 0 ? llama_get_logits_ith(ctx, 0) : nullptr;
            CHECK(l != nullptr && std::isfinite(l[0]), "first logit is finite");
            llama_free(ctx);
        }
        std::printf("%d checks, %d failed\n", g_checks, g_fails);
        llama_model_free(model);
        return g_fails == 0 ? 0 : 1;
    }

    llama_model_free(model);
    return g_fails == 0 ? 0 : 1;
}
