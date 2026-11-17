#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-model.h"

#include <cmath>

//
// Inkling (S69 P1): legacy-graph-path builder.
//
// SOURCE-READ: upstream/unsloth-llama-inkling-raw/src__models__inkling.cpp:122-666 (the class-path graph of
// unslothai/llama.cpp branch inkling-25731-b10775, MIT) and the Apache-2.0 reference InklingForCausalLM
// (transformers/models/inkling/modeling_inkling.py: InklingAttention, InklingRelativeLogits, InklingShortConvolution,
// InklingTopkRouter, InklingSharedExperts). The structure follows the MIT graph; what is this fork's own is how it meets
// the legacy path, which has no memory-context objects: see "WHERE THIS DEPARTS" below.
//
// WHAT ONE LAYER COMPUTES
//     h  += attn_sconv( o_proj( attn( attn_norm(h) ) ) )
//     h  += mlp_sconv ( mlp( mlp_norm(h) ) )
//   attn: no RoPE. q = q_norm(Wq x), k = k_norm(sconv_k(Wk x)), v = sconv_v(Wv x). Position enters ONLY through an
//         additive per-head bias: bias[h, q, kv] = (Wr x_q -> d_rel per head) @ proj[d_rel, E] indexed at the backward
//         distance pos_q - pos_kv, zero outside [0, E). Scores are q.k / head_dim (NOT sqrt) + bias, then the usual mask.
//         Global layers also scale q AND the bias by tau = 1 + alpha*ln(max((pos+1)/n_floor, 1)).
//   mlp:  dense SwiGLU x global_scale on the leading blocks; afterwards a sigmoid router with a selection-only bias,
//         top-k weights = softmax(logsigmoid(raw logits of the chosen experts ++ the shared-expert sink logits)) x
//         route_scale x global_scale, routed experts weighted by the first k, shared experts by the trailing gammas.
//   sconv(x) = x + causal_depthwise_conv1d(x), kernel K; the carried state is the last K-1 INPUTS.
//
// WHERE THIS DEPARTS FROM THE MIT GRAPH (each is forced by this fork's graph path, not a preference)
//   1. STATE. The MIT graph keeps the four conv histories in a recurrent memory context. Here they are one F32 row per
//      sequence in kv_self.s_l[il] (llama_kv_cache_init allocates it from hparams.n_embd_attn_layer_state), packed
//      [k | v | attn | mlp], each stream channel-major [K-1, width] -- the layout this fork's ggml_ssm_conv takes. Because
//      the same layer also owns K/V cells, the arch is registered hybrid but NO layer is recurrent (is_recurrent() is
//      false everywhere), exactly the openPangu shape; unlike openPangu the rows are per sequence (recurrent_runs).
//   2. ATTENTION. No banded flash-attention op exists in this tree, so attention is the explicit chain with the bias folded
//      into the mask: soft_max_ext(q.k * scale + mask_h) where mask_h = KQ_mask (causal / window, -inf) + the gathered
//      bias, one [n_kv, n_tokens] plane PER HEAD. The mask input broadcasts over heads in ggml_soft_max_ext, and the scale
//      multiplies only the scores, so the bias stays unscaled. The cost is an n_kv x n_tokens x n_head F32 plane per layer:
//      fine for a parity fixture and short contexts, and the reason the banded op exists as a follow-up for long ones.
//      flash_attn is therefore forced off for this arch (llama_fork_prepare_context_params).
//      DISABLE-TEMPORARY: S69-INKLING-FATTN-BANDED-OP-FOR-P1-GRAPH-2026-10-01-01 (that row is the route back to flash attention)
//   3. SHARED EXPERTS go through mul_mat_id with a constant id input, as the MIT graph does: a 2D view into a quantised
//      or repacked 3D bank is not a valid weight.
//

ggml_cgraph * llm_build_context::build_inkling() {
    ggml_cgraph * gf = new_graph_custom();

    GGML_ASSERT(!flash_attn && "inkling: flash attention is not available (no banded op yet); the context params force it off");

    const int64_t d_rel   = hparams.inkling_d_rel;
    const int64_t kernel  = hparams.n_shortconv_l_cache;
    const int64_t d_conv  = kernel - 1;                 // carried columns
    const int64_t n_shexp = hparams.n_expert_shared;

    GGML_ASSERT(d_conv >= 1 && d_rel > 0);

    bool has_global = false;
    bool has_local  = false;
    for (int il = 0; il < n_layer; ++il) {
        (hparams.is_swa(il) ? has_local : has_global) = true;
    }

    // ---- inputs ------------------------------------------------------------------------------------------------------
    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    // mtmd embedding rows arrive already normalised; the embedding norm applies to text-token lookups only
    if (batch.token) {
        inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, NULL, LLM_NORM_RMS, cb, -1);
        cb(inpL, "inkling_embd_norm", -1);
    }

    ggml_tensor * KQ_mask     = has_global ? build_inp_KQ_mask(true)     : nullptr;
    ggml_tensor * KQ_mask_swa = has_local  ? build_inp_KQ_mask_swa(true) : nullptr;

    // the per-token sequence-slot input every hybrid state user shares (zeros: each run addresses its own single-row view)
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    if (hparams.inkling_log_n_floor > 0 && has_global) {
        lctx.inp_inkling_tau = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        ggml_set_input(lctx.inp_inkling_tau);
        cb(lctx.inp_inkling_tau, "inkling_tau", -1);
    }
    if (has_global) {
        lctx.inp_inkling_rel_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
        ggml_set_input(lctx.inp_inkling_rel_idx);
        cb(lctx.inp_inkling_rel_idx, "inkling_rel_idx", -1);
    }
    if (has_local) {
        lctx.inp_inkling_rel_idx_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
        ggml_set_input(lctx.inp_inkling_rel_idx_swa);
        cb(lctx.inp_inkling_rel_idx_swa, "inkling_rel_idx_swa", -1);
    }
    const int64_t n_vocab = model.output->ne[1];
    if (!cparams.embeddings && hparams.inkling_unpadded_n_vocab > 0 && (int64_t) hparams.inkling_unpadded_n_vocab < n_vocab) {
        lctx.inp_inkling_vocab_mask = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_vocab);
        ggml_set_input(lctx.inp_inkling_vocab_mask);
        cb(lctx.inp_inkling_vocab_mask, "inkling_vocab_mask", -1);
    }
    if (n_shexp > 0 && (uint32_t) n_layer > hparams.n_layer_dense_lead) {
        lctx.inp_inkling_shexp_idx = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_shexp, n_tokens);
        ggml_set_input(lctx.inp_inkling_shexp_idx);
        cb(lctx.inp_inkling_shexp_idx, "inkling_shexp_idx", -1);
    }
    ggml_tensor * tau        = lctx.inp_inkling_tau;
    ggml_tensor * shexp_idx  = lctx.inp_inkling_shexp_idx;

    const std::vector<recurrent_run> runs = recurrent_runs();

    // sconv(x) = x + causal depthwise conv; `off` is the stream's offset (in floats) inside the layer's state row
    auto build_sconv = [&](ggml_tensor * x, ggml_tensor * kernel_w, int64_t off, int il) -> ggml_tensor * {
        ggml_tensor * state_all = kv_self.s_l[il];
        GGML_ASSERT(state_all != nullptr && state_all->type == GGML_TYPE_F32 && "inkling: layer has no conv-history row");
        const int64_t w = x->ne[0];
        GGML_ASSERT(off + d_conv*w <= state_all->ne[0]);

        const size_t ss = ggml_element_size(state_all);

        // ggml_ssm_conv wants an F32, contiguous [kernel, width] kernel
        ggml_tensor * kern = kernel_w->type == GGML_TYPE_F32 ? kernel_w : ggml_cast(ctx0, kernel_w, GGML_TYPE_F32);
        kern = ggml_reshape_2d(ctx0, kern, kernel, w);

        ggml_tensor * conv_out = nullptr;
        for (const auto & run : runs) {
            const size_t row_off = (size_t) run.seq * state_all->nb[1];

            ggml_tensor * st = ggml_view_3d(ctx0, state_all, d_conv, w, 1,
                    d_conv*ss, d_conv*w*ss, row_off + (size_t) off*ss);
            if (run.reset) {
                st = ggml_scale(ctx0, st, 0.0f);     // a run that starts its sequence convolves against zero history
            }
            ggml_tensor * x_r = ggml_view_2d(ctx0, x, w, run.len, x->nb[1], (size_t) run.start*x->nb[1]);
            ggml_tensor * sq_r = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, run.len,
                    lctx.inp_s_seq_qnext->nb[1], (size_t) run.start*lctx.inp_s_seq_qnext->nb[1]);

            // result is 1D: w*len conv outputs, then kernel*w state floats, whose last d_conv of each kernel-group are
            // the new carried columns (hence the one-element offset and the stride of `kernel`)
            ggml_tensor * x_conv = ggml_ssm_conv(ctx0, st, x_r, kern, sq_r, nullptr);
            const size_t ces = ggml_element_size(x_conv);

            ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0,
                        ggml_view_2d(ctx0, x_conv, d_conv, w, kernel*ces, (1 + w*(int64_t) run.len)*ces),
                        ggml_view_2d(ctx0, state_all, d_conv, w, d_conv*ss, row_off + (size_t) off*ss)));

            ggml_tensor * out_r = ggml_view_2d(ctx0, x_conv, w, run.len, w*ces, 0);
            conv_out = conv_out ? ggml_concat(ctx0, conv_out, out_r, 1) : out_r;
        }
        GGML_ASSERT(conv_out != nullptr);

        ggml_tensor * y = ggml_add(ctx0, x, conv_out);   // built-in residual, no activation
        cb(y, "inkling_sconv", il);
        return y;
    };

    auto build_attn_block = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];

        const bool    is_swa    = hparams.is_swa(il);
        const int64_t hd        = hparams.n_embd_head_k(il);
        const int64_t nh        = hparams.n_head(il);
        const int64_t nkv       = hparams.n_head_kv(il);
        const int64_t kw        = hparams.n_embd_k_gqa(il);
        const int64_t rel_ext   = is_swa ? hparams.inkling_rel_extent_swa : hparams.inkling_rel_extent;
        GGML_ASSERT(hparams.n_embd_head_v(il) == hd);

        ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq, cur);
        ggml_tensor * k = llm_build_lora_mm(lctx, ctx0, layer.wk, cur);
        ggml_tensor * v = llm_build_lora_mm(lctx, ctx0, layer.wv, cur);
        ggml_tensor * r = llm_build_lora_mm(lctx, ctx0, layer.wr, cur);
        cb(q, "inkling_attn_q", il);
        cb(k, "inkling_attn_k", il);
        cb(v, "inkling_attn_v", il);
        cb(r, "inkling_attn_r", il);

        // key / value short convs act on the flat projections, before the head reshape
        k = build_sconv(k, layer.shortconv_k, 0,          il);
        v = build_sconv(v, layer.shortconv_v, d_conv*kw,  il);

        q = ggml_reshape_3d(ctx0, q, hd, nh,  n_tokens);
        k = ggml_reshape_3d(ctx0, k, hd, nkv, n_tokens);
        v = ggml_reshape_3d(ctx0, v, hd, nkv, n_tokens);

        q = llm_build_norm(ctx0, q, hparams, layer.attn_q_norm, NULL, LLM_NORM_RMS, cb, il);
        k = llm_build_norm(ctx0, k, hparams, layer.attn_k_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(q, "inkling_attn_q_norm", il);
        cb(k, "inkling_attn_k_norm", il);

        // log-N scaling: global layers only, after the q norm
        if (tau && !is_swa) {
            q = ggml_mul(ctx0, q, tau);
        }

        // relative-position bias table: [E, n_head, n_tokens] = (r viewed as [d_rel, n_head*n_tokens]) contracted with proj[d_rel, E]
        ggml_tensor * proj = layer.attn_rel_proj;                         // gguf ne = [E, d_rel]
        if (proj->type != GGML_TYPE_F32) {
            proj = ggml_cast(ctx0, proj, GGML_TYPE_F32);
        }
        proj = ggml_cont(ctx0, ggml_transpose(ctx0, proj));               // [d_rel, E]
        ggml_tensor * r2  = ggml_reshape_2d(ctx0, r, d_rel, nh*n_tokens);
        ggml_tensor * rel = ggml_mul_mat(ctx0, proj, r2);                 // [E, n_head*n_tokens]
        ggml_mul_mat_set_prec(rel, GGML_PREC_F32);
        rel = ggml_reshape_3d(ctx0, rel, rel_ext, nh, n_tokens);
        if (tau && !is_swa) {
            rel = ggml_mul(ctx0, rel, tau);
        }
        cb(rel, "inkling_rel_logits", il);

        // gather: pad a zero column at index E, lay the table out as rows of n_head, pick row idx[q, cell]
        rel = ggml_pad(ctx0, rel, 1, 0, 0, 0);                            // [E+1, n_head, n_tokens]
        rel = ggml_cont(ctx0, ggml_permute(ctx0, rel, 1, 0, 2, 3));       // [n_head, E+1, n_tokens]
        rel = ggml_reshape_2d(ctx0, rel, nh, (rel_ext + 1)*n_tokens);

        ggml_tensor * idx  = is_swa ? lctx.inp_inkling_rel_idx_swa : lctx.inp_inkling_rel_idx;   // [n_kv, n_tokens]
        GGML_ASSERT(idx != nullptr);
        ggml_tensor * idx1 = ggml_reshape_1d(ctx0, idx, n_kv*n_tokens);

        ggml_tensor * kq_b = ggml_get_rows(ctx0, rel, idx1);              // [n_head, n_kv*n_tokens]
        kq_b = ggml_reshape_3d(ctx0, kq_b, nh, n_kv, n_tokens);
        kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));     // [n_kv, n_tokens, n_head]
        cb(kq_b, "inkling_kq_b", il);

        // fold the causal / window mask in: one [n_kv, n_tokens] plane per head (the mask is padded in rows, hence the view)
        ggml_tensor * mask  = is_swa ? KQ_mask_swa : KQ_mask;
        ggml_tensor * mask_v = ggml_view_2d(ctx0, mask, n_kv, n_tokens, mask->nb[1], 0);
        ggml_tensor * mask_h = ggml_add(ctx0, kq_b, mask_v);
        cb(mask_h, "inkling_mask_h", il);

        // scale 1/head_dim multiplies q.k only; the bias rides in the mask and stays unscaled
        cur = llm_build_kv(ctx0, lctx, kv_self, gf, layer.wo, nullptr,
                k, v, q, mask_h, n_tokens, kv_head, n_kv, 1.0f/float(hd), cb, il);
        cb(cur, "inkling_attn_o", il);
        return cur;
    };

    auto build_dense_ffn = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];
        cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, false);
        cur = ggml_mul(ctx0, cur, layer.ffn_gscale);
        cb(cur, "inkling_dense_ffn_out", il);
        return cur;
    };

    // Custom routing (not expressible through the shared MoE builder): select by top-k(sigmoid(logits) + bias), but WEIGHT by
    // softmax(logsigmoid(raw logits)) over the chosen experts together with the shared-expert sink logits.
    auto build_moe = [&](ggml_tensor * cur, int il) -> ggml_tensor * {
        const auto & layer = model.layers[il];

        ggml_tensor * logits = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_inp, cur);   // [n_expert + n_shexp, n_tokens]
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        cb(logits, "inkling_moe_logits", il);

        const size_t lsz = ggml_element_size(logits);

        ggml_tensor * routed = ggml_cont(ctx0, ggml_view_2d(ctx0, logits, n_expert, n_tokens, logits->nb[1], 0));
        ggml_tensor * shared_logits = ggml_cont(ctx0, ggml_view_2d(ctx0, logits, n_shexp, n_tokens, logits->nb[1], n_expert*lsz));

        // the bias steers selection only, never the weights
        ggml_tensor * scores = ggml_sigmoid(ctx0, routed);
        scores = ggml_add(ctx0, scores, layer.ffn_exp_probs_b);
        cb(scores, "inkling_moe_scores", il);

        ggml_tensor * selected = ggml_argsort_top_k(ctx0, scores, n_expert_used);        // I32 [n_expert_used, n_tokens]
        cb(selected, "inkling_moe_topk", il);

        ggml_tensor * routed3     = ggml_reshape_3d(ctx0, routed, 1, n_expert, n_tokens);
        ggml_tensor * topk_logits = ggml_get_rows(ctx0, routed3, selected);               // [1, n_expert_used, n_tokens]
        topk_logits = ggml_reshape_2d(ctx0, topk_logits, n_expert_used, n_tokens);

        ggml_tensor * all_logits = ggml_concat(ctx0, topk_logits, shared_logits, 0);      // [n_expert_used + n_shexp, n_tokens]

        // logsigmoid(x) = -softplus(-x)
        ggml_tensor * w = ggml_neg(ctx0, ggml_softplus(ctx0, ggml_neg(ctx0, all_logits)));
        w = ggml_soft_max(ctx0, w);
        w = ggml_scale(ctx0, w, hparams.expert_weights_scale);
        w = ggml_mul(ctx0, w, layer.ffn_gscale);                                          // the gate's global_scale, F32 [1]
        cb(w, "inkling_moe_weights", il);

        const size_t wsz = ggml_element_size(w);

        ggml_tensor * weights = ggml_cont(ctx0, ggml_view_2d(ctx0, w, n_expert_used, n_tokens, w->nb[1], 0));
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);

        ggml_tensor * xr   = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        ggml_tensor * gate = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_gate_exps, xr, selected);   // [n_ff_exp, n_expert_used, n_tokens]
        ggml_tensor * up   = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_up_exps,   xr, selected);
        ggml_tensor * h    = ggml_swiglu_split(ctx0, gate, up);

        ggml_tensor * experts = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_down_exps, h, selected); // [n_embd, n_expert_used, n_tokens]
        experts = ggml_mul(ctx0, experts, weights);

        ggml_tensor * moe_out = nullptr;
        for (int64_t i = 0; i < n_expert_used; ++i) {
            ggml_tensor * e = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
            moe_out = moe_out ? ggml_add(ctx0, moe_out, e) : e;
        }

        // shared experts: mul_mat_id with constant ids (a 2D view into a quantised / repacked bank is not a valid weight)
        GGML_ASSERT(shexp_idx != nullptr);
        ggml_tensor * gs = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_gate_shexp, xr, shexp_idx);   // [n_ff_exp, n_shexp, n_tokens]
        ggml_tensor * us = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_up_shexp,   xr, shexp_idx);
        ggml_tensor * hs = ggml_swiglu_split(ctx0, gs, us);

        // the gammas (the trailing n_shexp weights) scale the activation BEFORE the down projection, as the reference does
        ggml_tensor * gammas = ggml_cont(ctx0, ggml_view_2d(ctx0, w, n_shexp, n_tokens, w->nb[1], n_expert_used*wsz));
        hs = ggml_mul(ctx0, hs, ggml_reshape_3d(ctx0, gammas, 1, n_shexp, n_tokens));
        ggml_tensor * ds = llm_build_lora_mm_id(lctx, ctx0, layer.ffn_down_shexp, hs, shexp_idx);   // [n_embd, n_shexp, n_tokens]

        for (int64_t s = 0; s < n_shexp; ++s) {
            ggml_tensor * e = ggml_view_2d(ctx0, ds, n_embd, n_tokens, ds->nb[2], s*ds->nb[1]);
            moe_out = ggml_add(ctx0, moe_out, e);
        }
        cb(moe_out, "inkling_moe_out", il);

        return moe_out;
    };

    ggml_tensor * cur = inpL;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int64_t kw = hparams.n_embd_k_gqa(il);
        const int64_t vw = hparams.n_embd_v_gqa(il);

        // h += attn_sconv(attn(attn_norm(h)))
        ggml_tensor * attn_in  = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(attn_in, "inkling_attn_norm", il);
        ggml_tensor * attn_out = build_attn_block(attn_in, il);
        attn_out = build_sconv(attn_out, layer.shortconv_attn, d_conv*(kw + vw), il);
        cb(attn_out, "inkling_attn_sconv", il);

        cur = ggml_add(ctx0, cur, attn_out);

        // h += mlp_sconv(mlp(mlp_norm(h)))
        ggml_tensor * ffn_in  = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(ffn_in, "inkling_ffn_norm", il);
        ggml_tensor * ffn_out = (uint32_t) il < hparams.n_layer_dense_lead ? build_dense_ffn(ffn_in, il) : build_moe(ffn_in, il);
        ffn_out = build_sconv(ffn_out, layer.shortconv_mlp, d_conv*(kw + vw + n_embd), il);
        cb(ffn_out, "inkling_ffn_sconv", il);

        cur = ggml_add(ctx0, cur, ffn_out);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
    }

    // every layer's conv histories need ALL tokens of the ubatch, so outputs are trimmed only after the whole stack
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    if (!cparams.embeddings) {
        cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        // padded vocab rows are made unreachable so a sampler can never emit one
        if (lctx.inp_inkling_vocab_mask) {
            cur = ggml_add(ctx0, cur, lctx.inp_inkling_vocab_mask);
        }
        cb(cur, "result_output", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}
