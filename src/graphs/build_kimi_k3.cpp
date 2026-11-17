#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"

//
// Kimi-K3: hybrid KDA (linear) + MLA (full) attention, as in bailingmoe3 / kimi-linear.
// What this architecture adds on top of that, and where each piece lives here:
//
//   1. cross-layer residual attention  -> k3_res_push / k3_res_mix below
//   2. latent MoE                      -> k3_build_latent_moe below
//   3. situ activation                 -> llm_build_context::llm_build_situ (shared with the
//                                         dense FFN switch and the MoE expert path)
//   4. MLA output gate                 -> already in build_mla_output_gate; engages by itself
//                                         because create_kimi_k3_tensors sets layer.wqkv_gate,
//                                         and that helper now picks per-head vs per-head-dim
//                                         granularity from the projection's own width
//   5. full-rank KDA gate              -> no code here: the model's single full-rank SSM_G is
//                                         stored into layer.ssm_g_a, the slot build_kda_qkvz
//                                         already reads, and the shape matches
//
// Transcribed against mainline llama.cpp's src/models/kimi-k3.cpp, which is the reference for
// the maths. The structure differs because our KDA and MLA machinery is factored differently
// (delta_net + build_deepseek2_layer_attention do most of the work), not because the maths does.
//

// One checkpoint of the residual stream, appended. The stack is [n_embd, n_ckpt, n_tokens];
// append-only, deliberately: re-folding the whole chain each layer would change the values.
static void k3_res_push(ggml_context * ctx0, ggml_tensor *& stack, ggml_tensor * cur,
                        int64_t n_embd, int64_t n_tokens) {
    ggml_tensor * ckpt = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    stack = stack ? ggml_concat(ctx0, stack, ckpt, 1) : ckpt;
}

// Softmax over {banked checkpoints, current stream}, then recombine.
//
// mainline folds the stack with ggml_dsv4_hc_pre, which this fork does not carry. The same
// value is produced here from ops it does have: transpose so the checkpoint axis is ne0,
// broadcast-multiply by the per-checkpoint probabilities, and sum_rows over that axis. Written
// out rather than adding a new op because it is a weighted sum, and a new op would need a
// kernel per backend to earn its place.
static ggml_tensor * k3_res_mix(ggml_context * ctx0, const llama_hparams & hparams,
                                ggml_tensor * stack, ggml_tensor * cur, ggml_tensor * score_w,
                                int64_t n_embd, int64_t n_tokens, int il,
                                const llm_build_cb & cb) {
    if (!stack) {
        return cur;   // layer 0: nothing banked yet
    }

    const int64_t n_ckpt = stack->ne[1];
    const float   eps    = hparams.f_norm_rms_eps;

    // the scores use NORMALISED values while the recombination below uses the raw ones --
    // mainline does the same, and the two must not be collapsed
    ggml_tensor * sc_src = ggml_rms_norm(ctx0, stack, eps);
    sc_src = ggml_mul(ctx0, sc_src, score_w);
    sc_src = ggml_sum_rows(ctx0, sc_src);                          // [1, n_ckpt, n_tokens]
    sc_src = ggml_reshape_2d(ctx0, sc_src, n_ckpt, n_tokens);

    ggml_tensor * sc_cur = ggml_rms_norm(ctx0, cur, eps);
    sc_cur = ggml_mul(ctx0, sc_cur, score_w);
    sc_cur = ggml_sum_rows(ctx0, sc_cur);                          // [1, n_tokens]

    ggml_tensor * scores = ggml_concat(ctx0, sc_src, sc_cur, 0);   // [n_ckpt+1, n_tokens]
    ggml_tensor * probs  = ggml_soft_max(ctx0, scores);
    cb(probs, "res_probs", il);

    ggml_tensor * p_src = ggml_cont(ctx0,
            ggml_view_2d(ctx0, probs, n_ckpt, n_tokens, probs->nb[1], 0));
    ggml_tensor * p_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, probs, 1, n_tokens, probs->nb[1], probs->nb[0] * n_ckpt));

    // weighted sum over the checkpoint axis, with that axis moved to ne0 so sum_rows applies
    ggml_tensor * s = ggml_cont(ctx0, ggml_permute(ctx0, stack, 1, 0, 2, 3)); // [n_ckpt, n_embd, T]
    s = ggml_mul(ctx0, s, ggml_reshape_3d(ctx0, p_src, n_ckpt, 1, n_tokens));
    s = ggml_sum_rows(ctx0, s);                                    // [1, n_embd, n_tokens]
    ggml_tensor * out = ggml_reshape_2d(ctx0, s, n_embd, n_tokens);

    out = ggml_add(ctx0, out, ggml_mul(ctx0, cur, p_cur));
    cb(out, "res_mix", il);
    return out;
}

// Routed experts run in the latent space; the shared expert stays at n_embd and reads the
// un-projected input. The router also scores the FULL-width input, so its logits are computed
// here and handed to llm_build_moe_ffn rather than derived from the tensor the experts see.
static ggml_tensor * k3_build_latent_moe(ggml_context * ctx0, llama_context & lctx,
                                         const llama_layer & layer, ggml_tensor * normed,
                                         int64_t n_expert, int64_t n_expert_used,
                                         int il, ggml_cgraph * gf, const llm_build_cb & cb) {
    const auto & hparams = lctx.model.hparams;

    ggml_tensor * routed_in = layer.ffn_routed_down
        ? llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_routed_down, normed)
        : normed;
    cb(routed_in, "ffn_routed_in", il);

    ggml_tensor * logits = llm_build_context::llm_build_lora_mm(lctx, ctx0,
            layer.ffn_gate_inp, normed);
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * moe_out = llm_build_context::llm_build_moe_ffn(ctx0, lctx, routed_in,
            /*gate_inp   =*/ nullptr,          // unused: input_logits below is authoritative
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, n_expert_used,
            LLM_FFN_SITU, hparams.expert_weights_norm,
            /*scale_w    =*/ true, hparams.expert_weights_scale,
            (llm_expert_gating_func_type) hparams.expert_gating_func,
            cb, il, gf, /*add_input=*/ false,
            /*up_gate_exps=*/ nullptr, /*up_gate_exps_b=*/ nullptr,
            /*input_logits=*/ logits);
    cb(moe_out, "ffn_moe_out", il);

    if (layer.ffn_routed_norm) {
        moe_out = llm_build_context::llm_build_norm(ctx0, moe_out, hparams,
                layer.ffn_routed_norm, nullptr, LLM_NORM_RMS, cb, il);
    }
    if (layer.ffn_routed_up) {
        moe_out = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_routed_up, moe_out);
    }

    if (layer.ffn_gate_shexp) {
        // built from primitives rather than llm_build_ffn: that helper always applies its own
        // norm, and `normed` is already normalised here
        ggml_tensor * g = llm_build_context::llm_build_lora_mm(lctx, ctx0,
                layer.ffn_gate_shexp, normed);
        ggml_tensor * u = llm_build_context::llm_build_lora_mm(lctx, ctx0,
                layer.ffn_up_shexp, normed);
        ggml_tensor * sh = llm_build_context::llm_build_situ(ctx0, g, u,
                hparams.situ_beta, hparams.situ_linear_beta);
        sh = llm_build_context::llm_build_lora_mm(lctx, ctx0, layer.ffn_down_shexp, sh);
        cb(sh, "ffn_shexp", il);
        moe_out = ggml_add(ctx0, moe_out, sh);
    }

    cb(moe_out, "ffn_out", il);
    return moe_out;
}

ggml_cgraph * llm_build_context::build_kimi_k3() {
#ifdef GGML_USE_VULKAN
    const bool use_f32_attn_precision = true;
#else
    const bool use_f32_attn_precision = lctx.cparams.graph_attn_precision == GGML_TYPE_F32;
#endif

    ggml_cgraph * gf = new_graph_custom();
    delta_net delta(lctx, batch);

    auto inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto inp_out_ids = build_inp_out_ids();
    auto KQ_mask     = build_inp_KQ_mask();

    // K3's MLA is nope-only, so nothing in this graph consumes a position input. Building one
    // anyway leaves a graph input that no node references, the scheduler never allocates it,
    // and llama_set_inputs then writes into it -- "tensor buffer not set" from ggml-backend.
    // Cleared rather than merely skipped, because lctx keeps this pointer across graph builds
    // and a stale one from an earlier graph would be written to just the same.
    lctx.inp_pos = nullptr;

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k_full));
    const bool  pp_opt   = n_tokens >= 128 && lctx.cparams.mla_attn > 1;

    // K3's MLA is nope-only, so no rope cache is built and no positions are consumed by it.
    const uint32_t res_bs       = hparams.attn_res_block_size;
    const bool     use_attn_res = res_bs > 0;

    ggml_tensor * resi_stack = nullptr;
    ggml_tensor * cur        = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // the residual stream. Banked on checkpoint layers and then RESTARTED from the
        // attention output alone, rather than accumulated -- that restart is the whole point
        // of the scheme and is easy to lose when adapting a conventional builder.
        ggml_tensor * prefix_sum = inpL;

        cur = use_attn_res
            ? k3_res_mix(ctx0, hparams, resi_stack, prefix_sum, layer.attn_res_score,
                         n_embd, n_tokens, il, cb)
            : prefix_sum;

        bool banked = false;
        if (use_attn_res && (uint32_t) il % res_bs == 0) {
            k3_res_push(ctx0, resi_stack, prefix_sum, n_embd, n_tokens);  // banks the RAW input
            banked = true;
        }

        if (hparams.is_recurrent(il)) {
            // normalised here because the KDA helper for this architecture takes an already
            // normalised input and adds no residual of its own
            ggml_tensor * normed = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(normed, "attn_norm", il);
            // NOT narrowed to inp_out_ids on the last layer, unlike the other builders in this
            // tree. The residual stack is [n_embd, n_ckpt, n_tokens] and ggml_get_rows does not
            // mean on a 3-D tensor what it means on a 2-D one, so narrowing mid-loop would
            // leave the stack and the stream at different lengths. Everything stays full length
            // and is narrowed ONCE after the loop, which is also what mainline does.
            cur = delta.build_layer_attn_kda_k3(ctx0, gf, normed, nullptr, il, cb);
        } else {
            // this helper norms internally with layer.attn_norm and returns the projected
            // attention output WITHOUT a residual, which is exactly the contract needed here.
            // The K3 output gate is applied inside it, driven by layer.wqkv_gate.
            cur = build_deepseek2_layer_attention(gf, il, cur, KQ_mask, /*inp_pos=*/nullptr,
                    /*rope_cache=*/nullptr,
                    kq_scale, 1.0f, use_f32_attn_precision, hparams.n_lora_q == 0, pp_opt);
        }

        prefix_sum = banked ? cur : ggml_add(ctx0, prefix_sum, cur);
        cb(prefix_sum, "prefix_sum_attn", il);

        cur = use_attn_res
            ? k3_res_mix(ctx0, hparams, resi_stack, prefix_sum, layer.ffn_res_score,
                         n_embd, n_tokens, il, cb)
            : prefix_sum;

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // llm_build_ffn applies layer.ffn_norm itself, and nothing else here needs the
            // normalised value on a dense layer
            cur = llm_build_ffn(ctx0, lctx, layer.ffn_norm, cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SITU, LLM_FFN_PAR, cb, il, gf, false);
        } else {
            ggml_tensor * normed = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(normed, "ffn_norm", il);
            cur = k3_build_latent_moe(ctx0, lctx, layer, normed,
                    n_expert, n_expert_used, il, gf, cb);
        }

        prefix_sum = ggml_add(ctx0, prefix_sum, cur);
        prefix_sum = lctx.cvec.apply_to(ctx0, prefix_sum, il);
        cb(prefix_sum, "l_out", il);
        inpL = prefix_sum;
    }

    cur = inpL;

    // final mix at FULL length, then narrow to the requested output tokens -- this is the one
    // place narrowing happens, which is why nothing above passes inp_out_ids down
    if (use_attn_res) {
        cur = k3_res_mix(ctx0, hparams, resi_stack, cur, model.output_res_score,
                         n_embd, n_tokens, -1, cb);
    }
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
