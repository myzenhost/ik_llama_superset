#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_modernbert() {
    struct ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    // modernbert-family RoPE constants (config invariants; see SPEC ground-truth)
    const float modernbert_theta_global = 160000.0f;
    const float modernbert_theta_local  = 10000.0f;
    const int   modernbert_global_every = 3;

    struct ggml_tensor * inp_pos = build_inp_pos();
    struct ggml_tensor * inpL    = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // embeddings.norm (LayerNorm, no bias)
    inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, NULL, LLM_NORM, cb, -1);
    cb(inpL, "inp_norm", -1);

    struct ggml_tensor * KQ_mask = build_inp_KQ_mask(false);   // bidirectional

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * residual = inpL;

        // PRE-norm before attention (layer 0 attn_norm is Identity -> skip)
        struct ggml_tensor * cur = inpL;
        if (model.layers[il].attn_norm) {
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_norm, NULL, LLM_NORM, cb, il);
        }

        // packed QKV
        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
        cb(cur, "wqkv", il);

        struct ggml_tensor * Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd)));
        struct ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd)));
        struct ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa)));

        // per-layer RoPE theta: global (full-attn) layers every 3rd, else local
        const float freq_base_il = (il % modernbert_global_every == 0) ? modernbert_theta_global : modernbert_theta_local;

        Qcur = ggml_rope_ext(ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_il, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        Kcur = ggml_rope_ext(ctx0, ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base_il, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur, "Kcur", il);

        struct ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));
        struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_soft_max_ext(ctx0, kq, KQ_mask, 1.0f/sqrtf(float(n_embd_head)), 0.0f);
        cb(kq, "kq_soft_max_ext", il);

        struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
        struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
        struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
        ggml_build_forward_expand(gf, cur);

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);   // attn output (no bias)
        cb(cur, "kqv_out", il);

        // residual after attention
        cur = ggml_add(ctx0, cur, residual);
        struct ggml_tensor * ffn_residual = cur;

        // PRE-norm before MLP
        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM, cb, il);

        // GeGLU: Wi -> [input | gate]; gelu(input) * gate -> Wo
        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_up, cur);   // [2*n_ff, n_tokens]
        const int64_t n_ff_half = cur->ne[0] / 2;
        struct ggml_tensor * g_input = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_ff_half, n_tokens, cur->nb[1], 0));
        struct ggml_tensor * g_gate  = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_ff_half, n_tokens, cur->nb[1], n_ff_half*sizeof(float)));
        cur = ggml_mul(ctx0, ggml_gelu(ctx0, g_input), g_gate);
        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_down, cur);
        cb(cur, "ffn_out", il);

        // residual after MLP
        inpL = ggml_add(ctx0, cur, ffn_residual);
        cb(inpL, "l_out", il);
    }

    // final norm
    struct ggml_tensor * cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM, cb, -1);
    cb(cur, "result_embd", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
