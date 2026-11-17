#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-model.h"

//
// LFM2 / LFM2.5 (LiquidAI): SHORT-CONVOLUTION layers interleaved with conventional attention.
//
// head_count_kv is written per layer and is 0 on the convolution layers -- [0,0,8,0,0,8,...] for
// LFM2.5-350M -- so the array itself says which layers are which. That is the same convention
// bailingmoe3 and kimi-k3 already use in this fork, applied to a different mechanism: those mark
// KDA/SSM layers, these mark short convolutions.
//
// WHAT A SHORT-CONVOLUTION LAYER COMPUTES, per mainline llama.cpp src/models/lfm2.cpp:
//     bcx      = in_proj(x)                 [3*n_embd, T] -- three equal chunks b, c, x
//     bx       = b * x                      elementwise
//     conv_out = causal_conv1d(bx, kernel)  over the last l_cache positions
//     y        = c * conv_out
//     out      = out_proj(y)
// It is a gated convolution: b gates the input, c gates the convolved result.
//
// TWO PLACES THIS DEPARTS FROM MAINLINE'S FILE, both because this fork is built differently:
//
//  1. ggml_ssm_conv HAS A DIFFERENT SIGNATURE HERE. Mainline concatenates the carried state onto
//     the input itself and calls a 3-argument ggml_ssm_conv(ctx, bx, kernel). This fork's is
//     ggml_ssm_conv(ctx, s, x, c, sq, saved_steps) -- it takes the state SEPARATELY and returns
//     the new state packed into the tail of the result. Transcribing mainline's call would not
//     compile, and the idiom below is copied from this fork's own build_mamba.cpp instead.
//
//  2. RECURRENT STATE LIVES IN s_l, NOT k_l. build_mamba stores conv state in kv_self.k_l[il],
//     which works for a model where EVERY layer is recurrent. LFM2 is hybrid: its attention
//     layers need k_l/v_l for a real K/V cache, so the two uses would collide. This fork already
//     solves that for hybrid architectures -- llama.cpp allocates kv_self.s_l[il] for recurrent
//     layers and pushes nullptr into k_l/v_l for them -- and LFM2 is registered in
//     llm_arch_is_hybrid so it takes that path.
//
// THE STATE-SIZE FIT, which is worth spelling out because one field is set to a value that reads
// oddly. The hybrid allocator sizes s_l[il] by hparams.n_embd_v_s(), which for a non-qwen3next
// model is ssm_d_state * ssm_d_inner. A short convolution carries (l_cache - 1) * n_embd floats.
// So the hparam case sets ssm_d_conv = l_cache, ssm_d_inner = n_embd, ssm_d_state = l_cache - 1,
// which makes both n_embd_v_s() and n_embd_k_s() equal exactly that figure. ssm_d_state is not a
// state dimension in any SSM sense here; it is the carried-column count, chosen so the existing
// allocator produces the right size without a new code path. Anyone changing l_cache handling
// should re-check that identity.
//

ggml_cgraph * llm_build_context::build_lfm2() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t d_conv      = hparams.ssm_d_conv;        // l_cache, 3 for LFM2.5
    const int64_t d_inner     = hparams.ssm_d_inner;       // == n_embd for this family
    const int64_t n_embd_head = hparams.n_embd_head_k(0);  // 64
    const float   kq_scale    = 1.0f / sqrtf(float(n_embd_head));

    GGML_ASSERT(d_conv > 1 && "lfm2: shortconv.l_cache must exceed 1");
    GGML_ASSERT(d_inner == n_embd && "lfm2: shortconv operates at n_embd width");

    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_pos     = build_inp_pos();
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();

    // The per-token sequence-slot input the hybrid recurrent path uses. build_inp_s_mask /
    // build_inp_s_seq are the MAMBA-family inputs and are deliberately not used here -- see the
    // note in the shortconv block on why token-space indexing is wrong for a hybrid model.
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    ggml_tensor * cur = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * prev = inpL;

        // "operator_norm": one norm ahead of whichever operator this layer runs
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
        cb(cur, "operator_norm", il);

        if (hparams.is_recurrent(il)) {
            // ---- short convolution ------------------------------------------------------
            //
            // ggml_ssm_conv's contract (ggml.c:11005-11020) is exact and it is what dictates the
            // shapes here:
            //     s  3D  [d_conv-1, d_inner, n_kv]
            //     x      [d_inner,  n_tokens]
            //     c      [d_conv,   d_inner]
            //     sq I32 [n_kv,     n_tokens]      <- sq->ne[0] MUST equal s->ne[2]
            //
            // build_mamba indexes its state with kv_head / n_kv, which are TOKEN-space offsets
            // into the attention cache. That is correct for mamba because every layer there is
            // recurrent, so the KV cache IS the sequence-slot array. LFM2 is hybrid: s_l holds
            // per-SEQUENCE slots while n_kv counts cached tokens, and using a token offset to
            // index it runs off the end -- measured, as a view-bounds assert at ggml.c:5477.
            //
            // ONE STATE ROW PER SEQUENCE. s_l[il] is [n_embd_v_s, slots] with slots >= n_seq_max
            // (llama.cpp allocates it that way for every hybrid arch); row r holds sequence r's
            // carried columns. Each contiguous run of same-sequence tokens (recurrent_runs) is
            // convolved against its own row and writes its new state back to that row, and a run
            // that starts its sequence (pos == 0) begins from zero state.
            //
            // FIXED 2026-09-26: this block used row 0 for every sequence and never zeroed it. The
            // state of whatever was decoded last leaked into the next request: under llama-server
            // the second request on a slot produced garbage ("Water boils at a temperature of" ->
            // "a"), and -np > 1 mixed all slots into one state. llama-perplexity and llama-cli
            // looked fine only because they clear the buffers (memory_clear with data) or run once.
            //
            // Per run, sq is the matching slice of inp_s_seq_qnext (all zeros): with n_kv == 1 it
            // addresses the single state row the view below exposes.
            GGML_ASSERT(lctx.inp_s_seq_qnext != nullptr);

            const int64_t n_state_kv = 1;
            const size_t  ss         = ggml_element_size(kv_self.s_l[il]);
            const size_t  row_bytes  = kv_self.s_l[il]->nb[1];

            ggml_tensor * bcx = llm_build_lora_mm(lctx, ctx0,
                    model.layers[il].shortconv_in_proj, cur);
            cb(bcx, "shortconv_in_proj", il);

            // three equal chunks along ne0: b, c, x -- the order mainline uses
            const size_t es = ggml_element_size(bcx);
            ggml_tensor * b = ggml_view_2d(ctx0, bcx, d_inner, bcx->ne[1], bcx->nb[1], 0);
            ggml_tensor * c = ggml_view_2d(ctx0, bcx, d_inner, bcx->ne[1], bcx->nb[1], es * d_inner);
            ggml_tensor * x = ggml_view_2d(ctx0, bcx, d_inner, bcx->ne[1], bcx->nb[1], es * d_inner * 2);

            // b gates the input before the convolution
            ggml_tensor * bx = ggml_cont(ctx0, ggml_mul(ctx0, b, x));

            ggml_tensor * conv_out = nullptr;
            for (const auto & run : recurrent_runs()) {
                const size_t row_off = (size_t) run.seq * row_bytes;
                ggml_tensor * conv_states = ggml_view_3d(ctx0, kv_self.s_l[il],
                        d_conv - 1, d_inner, n_state_kv,
                        (d_conv - 1) * ss, (d_conv - 1) * d_inner * ss, row_off);
                if (run.reset) {
                    conv_states = ggml_scale(ctx0, conv_states, 0.0f);
                }
                ggml_tensor * bx_r = ggml_view_2d(ctx0, bx, d_inner, run.len, bx->nb[1],
                                                  (size_t) run.start * bx->nb[1]);
                ggml_tensor * sq_r = ggml_view_2d(ctx0, lctx.inp_s_seq_qnext, 1, run.len,
                                                  lctx.inp_s_seq_qnext->nb[1],
                                                  (size_t) run.start * lctx.inp_s_seq_qnext->nb[1]);

                // this fork's ggml_ssm_conv takes the carried state separately and returns the new
                // state in the tail of the result -- see the note at the top of this file
                ggml_tensor * x_conv = ggml_ssm_conv(ctx0, conv_states, bx_r,
                        model.layers[il].shortconv_conv, sq_r, nullptr);
                const size_t ces = ggml_element_size(x_conv);

                // The result is 1D and packs two regions (ggml.c:11036): the output,
                // d_inner*len floats, then the state region, d_conv*d_inner*n_kv floats. The new
                // carried state is the LAST d_conv-1 of each d_conv-length group, which is why the
                // view below starts one element into the state region and strides by d_conv.
                ggml_build_forward_expand(gf,
                        ggml_cpy(ctx0,
                            ggml_view_2d(ctx0, x_conv, d_conv - 1, d_inner * n_state_kv,
                                         d_conv * ces, (1 + d_inner * (int64_t) run.len) * ces),
                            ggml_view_1d(ctx0, kv_self.s_l[il],
                                         (d_conv - 1) * d_inner * n_state_kv, row_off)));

                ggml_tensor * out_r = ggml_view_2d(ctx0, x_conv, d_inner, run.len, d_inner * ces, 0);
                conv_out = conv_out ? ggml_concat(ctx0, conv_out, out_r, 1) : out_r;
            }
            GGML_ASSERT(conv_out != nullptr);
            cb(conv_out, "shortconv_conv", il);

            // c gates the convolved result. No bias and no SiLU here -- mainline's lfm2 applies
            // neither, unlike mamba's conv block which does both.
            ggml_tensor * y = ggml_mul(ctx0, c, conv_out);
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].shortconv_out_proj, y);
            cb(cur, "shortconv_out_proj", il);
        } else {
            // ---- conventional attention with q/k norms ----------------------------------
            ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
            ggml_tensor * k = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
            ggml_tensor * v = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
            cb(q, "wq", il);
            cb(k, "wk", il);
            cb(v, "wv", il);

            q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
            k = ggml_reshape_3d(ctx0, k, n_embd_head, hparams.n_head_kv(il), n_tokens);

            // the norms are per HEAD DIMENSION here -- attn_q_norm/attn_k_norm are [64], not
            // [n_embd] -- so they are applied AFTER the reshape to head layout
            q = llm_build_norm(ctx0, q, hparams, model.layers[il].attn_q_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            k = llm_build_norm(ctx0, k, hparams, model.layers[il].attn_k_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(q, "attn_q_norm", il);
            cb(k, "attn_k_norm", il);

            q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

            cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, nullptr,
                    k, v, q, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur, inp_out_ids);
            prev = ggml_get_rows(ctx0, prev, inp_out_ids);
        }

        cur = ggml_add(ctx0, prev, cur);
        ggml_tensor * ffn_inp = cur;

        // dense FFN on BOTH layer kinds of the dense family; LFM2MOE (ik_llama #2530) routes every block past
        // n_layer_dense_lead through the standard MoE FFN (sigmoid router + selection bias, renormalised top-k).
        // The residual add stays below for both, so add_input is false here.
        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, false);
        } else {
            GGML_ASSERT(model.layers[il].ffn_down_exps != nullptr);
            cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                    model.layers[il].ffn_gate_inp,  nullptr,
                    model.layers[il].ffn_up_exps,   nullptr,
                    model.layers[il].ffn_gate_exps, nullptr,
                    model.layers[il].ffn_down_exps, nullptr,
                    model.layers[il].ffn_exp_probs_b,
                    nullptr, nullptr,
                    nullptr, nullptr,
                    nullptr, nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale != 0.0f, hparams.expert_weights_scale,
                    (llm_expert_gating_func_type) hparams.expert_gating_func,
                    LLM_FFN_SILU, cb, il, gf, false,
                    model.layers[il].ffn_up_gate_exps, nullptr);
        }
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, ffn_inp, cur);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    // the final norm is "token_embd_norm" in this family; create_lfm2_tensors already put it in
    // model.output_norm, and model.output is the tied embedding when no output.weight exists
    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);

    return gf;
}
