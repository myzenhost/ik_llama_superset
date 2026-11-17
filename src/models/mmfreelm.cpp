#include "models.h"

#include "llama-memory-recurrent.h"

// MatMul-free LM (ridger/MMfreeLM-*, "Scalable MatMul-free Language Modeling", arXiv 2406.02528, Apache-2.0 reference
// code github.com/ridgerchu/matmulfreellm). Written from a full read of the reference (layers/hgrn_bit.py,
// models/hgrn_bit/modeling_hgrn_bit.py, ops/fusedbitnet.py, modules/fused_norm_gate.py); no code copied.
//
//   h -> attn_norm -> MLGRU token mixer -> + residual -> mlp_norm -> GLU channel mixer -> + residual
//   BitLinear(x)  = W_ternary . actquant8(RMSNorm_w(x))   (weights stored as TQ2_0; ggml quantizes the activation
//                   to 8 bits inside the TQ2_0 x Q8_K dot, so the explicit per-token fake-quant is not repeated)
//   MLGRU:  i = BitLin(x), f = sigmoid(BitLin(x)), g = BitLin(x)
//           f = lb + (1-lb)*f           (layers > 0; lb = cumulative-softmax lower bound, precomputed by the converter)
//           i = silu(i) * (1-f)
//           h_t = f_t * h_{t-1} + i_t   (elementwise linear recurrence)
//           o = RMSNorm_w(g) * h * sigmoid(h);  y = BitLin(o)
//   GLU:    y = BitLin_down( silu(BitLin_gate(x)) * BitLin_up(x) )
//   head:   logits = BitLin(RMSNorm(h))
//
// The recurrence runs on GGML_OP_RWKV_WKV7 with head_size 1 (one head per channel): its update is
// S_t = S_{t-1}*w + v*k + (S_{t-1}.a)*b and out_t = S_t*r, so r=k=1, v=i, w=f, a=b=0 gives exactly h_t = f_t*h_{t-1} + i_t.
// That op already has CPU/CUDA/Vulkan/Metal kernels and the recurrent-state plumbing (llama_memory_recurrent).

void llama_model_mmfreelm::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // head_size 1 selects the RWKV state sizing in llama_hparams::n_embd_s() (= n_embd * 1); the shift buffer is unused
    hparams.wkv_head_size     = 1;
    hparams.token_shift_count = 1;

    switch (hparams.n_layer_eff()) {
        case 24:
            switch (hparams.n_embd) {
                case 1024: type = LLM_TYPE_450M; break; // MMfreeLM-370M (the enum has no 370M bucket)
                case 2048: type = LLM_TYPE_1_3B; break;
                default:   type = LLM_TYPE_UNKNOWN;
            } break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_mmfreelm::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff0 = hparams.n_ff_arr[0]; // LLAMA_LOAD_LOCALS already binds n_ff (= hparams.n_ff()): MSVC C2374 on the redefinition

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,     "weight"), {n_embd, n_vocab}, 0);
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), {n_embd}, 0);
    // lm_head is a BitLinear too: its own input norm + ternary matrix
    output_in_norm  = create_tensor(tn(LLM_TENSOR_OUTPUT_IN_NORM, "weight"), {n_embd}, 0);
    output          = create_tensor(tn(LLM_TENSOR_OUTPUT,         "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.hgrn_i      = create_tensor(tn(LLM_TENSOR_HGRN_I,      "weight", i), {n_embd, n_embd}, 0);
        layer.hgrn_f      = create_tensor(tn(LLM_TENSOR_HGRN_F,      "weight", i), {n_embd, n_embd}, 0);
        layer.hgrn_g      = create_tensor(tn(LLM_TENSOR_HGRN_G,      "weight", i), {n_embd, n_embd}, 0);
        layer.hgrn_i_norm = create_tensor(tn(LLM_TENSOR_HGRN_I_NORM, "weight", i), {n_embd}, 0);
        layer.hgrn_f_norm = create_tensor(tn(LLM_TENSOR_HGRN_F_NORM, "weight", i), {n_embd}, 0);
        layer.hgrn_g_norm = create_tensor(tn(LLM_TENSOR_HGRN_G_NORM, "weight", i), {n_embd}, 0);
        layer.hgrn_o_norm = create_tensor(tn(LLM_TENSOR_HGRN_O_NORM, "weight", i), {n_embd}, 0);
        layer.hgrn_gnorm  = create_tensor(tn(LLM_TENSOR_HGRN_GNORM,  "weight", i), {n_embd}, 0);
        layer.hgrn_lower_bound = create_tensor(tn(LLM_TENSOR_HGRN_LOWER_BOUND, "weight", i), {n_embd}, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_in_norm      = create_tensor(tn(LLM_TENSOR_FFN_IN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_down_in_norm = create_tensor(tn(LLM_TENSOR_FFN_DOWN_IN_NORM, "weight", i), {n_ff0}, 0);
        layer.ffn_gate         = create_tensor(tn(LLM_TENSOR_FFN_GATE,         "weight", i), {n_embd, n_ff0}, 0);
        layer.ffn_up           = create_tensor(tn(LLM_TENSOR_FFN_UP,           "weight", i), {n_embd, n_ff0}, 0);
        layer.ffn_down         = create_tensor(tn(LLM_TENSOR_FFN_DOWN,         "weight", i), {n_ff0, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_mmfreelm::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_mmfreelm::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const auto * mctx_cur   = static_cast<const llama_memory_recurrent_context *>(mctx);
    const auto   n_seqs     = ubatch.n_seqs;
    const auto   kv_head    = mctx_cur->get_head();
    const int64_t n_embd_l  = hparams.n_embd;

    // BitLinear: RMSNorm(x; w_norm) -> ternary matmul
    auto bit = [&](ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w, int il) {
        ggml_tensor * n = build_norm(x, w_norm, nullptr, LLM_NORM_RMS, il);
        return build_lora_mm(w, n);
    };

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    auto * rs_inp = build_rs_inp();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer & layer = model.layers[il];

        ggml_tensor * x = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(x, "attn_norm", il);

        // ---- MLGRU token mixer ----
        ggml_tensor * i_p = bit(x, layer.hgrn_i_norm, layer.hgrn_i, il);
        ggml_tensor * f_p = bit(x, layer.hgrn_f_norm, layer.hgrn_f, il);
        ggml_tensor * g_p = bit(x, layer.hgrn_g_norm, layer.hgrn_g, il);

        ggml_tensor * f = ggml_sigmoid(ctx0, f_p);
        if (il > 0) {
            // f = lb + (1 - lb) * f
            ggml_tensor * one_minus_lb = ggml_scale_bias(ctx0, layer.hgrn_lower_bound, -1.0f, 1.0f);
            f = ggml_add(ctx0, ggml_mul(ctx0, f, one_minus_lb), layer.hgrn_lower_bound);
        }
        ggml_tensor * i_g = ggml_mul(ctx0, ggml_silu(ctx0, i_p), ggml_scale_bias(ctx0, f, -1.0f, 1.0f));

        const int64_t n_tokens_l = n_tokens;
        ggml_tensor * ones  = ggml_scale_bias(ctx0, f, 0.0f, 1.0f);
        ggml_tensor * zeros = ggml_scale(ctx0, f, 0.0f);

        // one head per channel (head_size 1)
        ggml_tensor * r3 = ggml_reshape_3d(ctx0, ones,  1, n_embd_l, n_tokens_l);
        ggml_tensor * w3 = ggml_reshape_3d(ctx0, f,     1, n_embd_l, n_tokens_l);
        ggml_tensor * k3 = r3;
        ggml_tensor * v3 = ggml_reshape_3d(ctx0, i_g,   1, n_embd_l, n_tokens_l);
        ggml_tensor * a3 = ggml_reshape_3d(ctx0, zeros, 1, n_embd_l, n_tokens_l);

        ggml_tensor * state = build_rs(rs_inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);

        ggml_tensor * out = ggml_rwkv_wkv7(ctx0, r3, w3, k3, v3, a3, a3, state);
        ggml_tensor * h   = ggml_view_1d(ctx0, out, n_embd_l * n_tokens_l, 0);
        ggml_tensor * new_state = ggml_view_1d(ctx0, out, n_embd_l * n_seqs, n_embd_l * n_tokens_l * sizeof(float));

        ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs,
                             hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il)))));

        h = ggml_reshape_2d(ctx0, h, n_embd_l, n_tokens_l);

        // o = RMSNorm_w(g) * h * sigmoid(h)
        ggml_tensor * g_n = build_norm(g_p, layer.hgrn_gnorm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * o   = ggml_mul(ctx0, ggml_mul(ctx0, g_n, h), ggml_sigmoid(ctx0, h));
        cb(o, "mlgru_out", il);

        cur = bit(o, layer.hgrn_o_norm, layer.wo, il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        if (il == n_layer - 1 && inp_out_ids) {
            ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
        }

        // ---- GLU channel mixer ----
        ggml_tensor * y  = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(y, "ffn_norm", il);
        ggml_tensor * yn = build_norm(y, layer.ffn_in_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * gate = build_lora_mm(layer.ffn_gate, yn);
        ggml_tensor * up   = build_lora_mm(layer.ffn_up,   yn);
        ggml_tensor * act  = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
        cur = bit(act, layer.ffn_down_in_norm, layer.ffn_down, il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = bit(cur, model.output_in_norm, model.output, -1);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
