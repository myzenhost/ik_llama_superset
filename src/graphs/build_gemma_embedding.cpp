#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// S69-OLLAMA-FORMAT-EMBEDDINGGEMMA-316-VS-314-TENSORS-2026-09-28-01.
//
// Legacy-path port. The new-generation implementation, src/models/gemma-embedding.cpp, is
// structurally unreachable (llama_model::build_graph() unconditionally returns nullptr and no
// factory ever constructs a llama_model_base subclass -- memory/project-fork-models-wired-but-no-dispatcher.md,
// findings/S69_EMBEDGEMMA_MTP_BAILING_2026-09-29.md). This file is therefore a from-scratch port
// of that source's field-by-field math into the legacy llm_build_context graph builder, not a
// call into it -- every formula below is cited to where it was read, not reinvented.
//
// SOURCE-READ: src/models/gemma-embedding.cpp:75-178 (new-gen graph -- scale-by-sqrt(n_embd) input,
// per-layer Q/K RMS-norm before RoPE, Gemma Q attention-scale multiply, GELU-gated FFN, pre/post
// norm sandwich, no lm_head, stops at result_norm).
// SOURCE-READ: src/graphs/build_gemma3.cpp:1-123 (legacy sibling arch -- per-layer SWA/global
// split via a local sliding-window pattern, llm_build_ffn's embedded pre-norm call convention,
// lctx.cvec.apply_to for control vectors).
// SOURCE-READ: src/graphs/build_modernbert.cpp:1-94 and src/graphs/build_bert.cpp:1-191 (legacy
// non-causal / no-KV-cache attention: build_inp_KQ_mask(false), manual permute/mul_mat/soft_max_ext/
// mul_mat attention with no llm_build_kv call, and the pooling_type-gated inp_out_ids trim at
// build_bert.cpp:128 -- kept here rather than gemma-embedding.cpp's unconditional trim because MEAN/
// CLS pooling need every token's hidden state, and the legacy build_inp_out_ids() never returns
// null the way the new path's apparently can).
ggml_cgraph * llm_build_context::build_gemma_embedding() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image embeddings)
    if (batch.token) {
        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);
    }

    struct ggml_tensor * inp_pos = build_inp_pos();

    // KQ_mask (mask for 1 head, broadcast to all heads) -- bidirectional, no KV cache.
    // gemma-embedding requires a different mask for the sliding-window (SWA) layers, same as
    // gemma3, except both masks here are non-causal (n_tokens x n_tokens, not n_kv x n_tokens).
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask(false);
    struct ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa(false);

    for (int il = 0; il < n_layer; ++il) {
        // hparams.is_swa(il) reads is_swa_impl, populated by set_swa_pattern() in the
        // LLM_ARCH_GEMMA_EMBEDDING case of llama-hparams.cpp (paired with swa_type there, per the
        // GEMMA3 case's own comment on why one without the other asserts).
        const bool is_sliding = hparams.is_swa(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        struct ggml_tensor * KQ_mask_l = is_sliding ? KQ_mask_swa : KQ_mask;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention (non-causal, no KV cache)
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    nullptr, nullptr,
                    nullptr, nullptr,
                    model.layers[il].wq, nullptr, model.layers[il].wk, nullptr, model.layers[il].wv, nullptr,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0.0f, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // ref: https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/model.py#L315
            // Applied to Q directly (not as a softmax scale) so the manual attention below passes
            // scale=1.0 to soft_max_ext -- same split gemma-embedding.cpp's new-gen graph uses.
            Qcur = ggml_scale(ctx0, Qcur, hparams.f_attention_scale);

            struct ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor * k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            struct ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            kq = ggml_soft_max_ext(ctx0, kq, KQ_mask_l, 1.0f, 0.0f);
            cb(kq, "kq_soft_max_ext", il);

            struct ggml_tensor * v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_tokens)));
            cb(v, "v", il);

            struct ggml_tensor * kqv = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, v, n_tokens, n_embd_head, n_head_kv), kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor * kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            // merged width is n_head * n_embd_head (the Q side), NOT n_embd_gqa (the K/V side):
            // embeddinggemma has 3 query heads over 1 KV head, so the two differ (768 vs 256).
            cur = ggml_cont_2d(ctx0, kqv_merged, n_embd_head * n_head, n_tokens);
            cb(cur, "kqv_merged_cont", il);

            ggml_build_forward_expand(gf, cur);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            cb(cur, "kqv_out", il);
        }

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_post_norm", il);

        // pooled embeddings need every token's hidden state -- only trim to the requested output
        // rows when nothing downstream pools (matches build_bert.cpp:128, not gemma-embedding.cpp's
        // unconditional trim; see the file header note).
        if (il == n_layer - 1 && pooling_type == LLAMA_POOLING_TYPE_NONE) {
            struct ggml_tensor * inp_out_ids = build_inp_out_ids();
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        struct ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        // feed-forward network (llm_build_ffn applies the pre-norm internally)
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, sa_out,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, -1);
        cb(cur, "ffn_post_norm", -1);

        cur = ggml_add(ctx0, cur, sa_out);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // No lm_head: embeddinggemma ships none, and nothing downstream of this graph needs one.
    // append_pooling() (llama-build-context.cpp:815, invoked generically after the arch dispatch
    // at :3412) finds this "result_norm" node, pools it, and -- for this arch -- applies the
    // dense_2 -> dense_3 SentenceTransformers projection when those tensors are present.
    ggml_build_forward_expand(gf, cur);

    return gf;
}
