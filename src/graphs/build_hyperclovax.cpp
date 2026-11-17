#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// HyperCLOVA X text graph (S69-EXLLAMAV3-PICK-HYPERCLOVAX-2026-10-20-01).
// SOURCE-READ: transformers HyperCLOVAXDecoderLayer / HyperCLOVAXModel / HyperCLOVAXForCausalLM (modular_hyperclovax.py, Apache-2.0) and
// exllamav3 architecture/hyperclovax.py (MIT). Behaviour re-implemented on this fork's graph helpers; no code copied.
//
//   x = embed(tokens) * embedding_multiplier
//   per layer:  h = x + residual_multiplier * post_norm1( attn( input_layernorm(x) ) )
//               x = h + residual_multiplier * post_norm2( mlp ( post_attention_layernorm(h) ) )
//   logits = lm_head( norm(x) ) * logits_scaling
//
// The three differences from the Granite path (build_llama.cpp):
//   * the Peri-LN norms sit between each sub-layer and its residual add, so the scale and the add cannot be fused into the sub-layer;
//   * attention scales q.k by attention_multiplier (f_attention_scale) and never by 1/sqrt(head_dim) unless the key is absent;
//   * the logits are MULTIPLIED by logits_scaling. The converter stores 1/logits_scaling in the GGUF logit_scale key, so the division
//     below is the same one every Granite-family graph does (see llama-hparams.cpp, LLM_ARCH_HYPERCLOVAX).
// A checkpoint with use_post_norm=false has no post-norm tensors; the norm is then skipped (HF nn.Identity).
ggml_cgraph * llm_build_context::build_hyperclovax() {
    ggml_cgraph * gf = new_graph_custom();
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // applies hparams.f_embedding_scale for token inputs (llama-build-context.cpp, llm_build_inp_embd)
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * KQ_mask = build_inp_KQ_mask();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // self-attention (full head_dim NEOX rope, no q/k norm)
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, model.layers[il].bqkv,
                    model.layers[il].wqk,  model.layers[il].bqk,
                    model.layers[il].wq,   model.layers[il].bq,
                    model.layers[il].wk,   model.layers[il].bk,
                    model.layers[il].wv,   model.layers[il].bv,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Peri-LN: norm the sub-layer output, scale it, THEN add the residual
        if (model.layers[il].attn_post_norm) {
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "attn_post_norm", il);
        }
        if (hparams.f_residual_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (llm_build_ffn norms ffn_inp with ffn_norm itself and does NOT add the residual here)
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        if (model.layers[il].ffn_post_norm) {
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_post_norm", il);
        }
        if (hparams.f_residual_scale != 0.0f) {
            cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

    // logits_scaling multiplies; the GGUF stores its reciprocal (see above)
    if (hparams.f_logit_scale != 0.0f) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
    }
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
