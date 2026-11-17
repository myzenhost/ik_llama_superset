// GPT-J (EleutherAI, GPTJForCausalLM). Class model reached through llama_model_create_for_arch (src/llama-model-create.cpp).
//
// SOURCE-READ: HF transformers 4.57.6 models/gptj/modeling_gptj.py (Apache-2.0), no code copied, graph written from the maths:
//   GPTJBlock.forward :447-460  residual; h = ln_1(x); x = attn(h) + mlp(h) + residual           -> ONE norm, parallel attention + FFN
//   GPTJAttention     :114-117  q/k/v/out_proj are bias-free Linear                              -> no wq_b/wk_b/wv_b/wo_b
//   apply_rotary_pos_emb :73-83 rotate_every_two over x[..., ::2] / x[..., 1::2]                 -> adjacent-pair RoPE (GGML rope mode NORM)
//   forward           :221-235  only the first rotary_dim channels of each head rotate           -> n_rot = rotary_dim, rest passes through
//   create_sinusoidal_positions :62-65  inv_freq = 10000^(-2i/rotary_dim)                         -> freq_base 10000, base on rotary_dim
//   _attn             :159-169  scores / sqrt(head_dim), softmax                                  -> kq_scale 1/sqrt(n_embd_head)
//   GPTJMLP           :414-423  fc_in -> gelu_new (tanh GELU) -> fc_out, both with bias           -> LLM_FFN_GELU
//   GPTJForCausalLM   :884      lm_head = nn.Linear(n_embd, vocab) WITH bias                      -> output_b
//   GPTJModel         :556,730  final LayerNorm ln_f (weight+bias)
// llama.cpp upstream has only the arch NAME for GPT-J (a one-tensor table, no builder), so this is a new implementation, not a port.
// Position handling: llama_rope_type() (src/llama.cpp) maps GPTJ to LLAMA_ROPE_TYPE_NORM.
#include "models.h"

void llama_model_gptj::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    switch (hparams.n_layer_eff()) {
        case 28: type = LLM_TYPE_6B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gptj::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output (GPT-J does not tie lm_head to wte, and lm_head has a bias)
    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);
    output        = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
    output_b      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, 0);

        layer.ffn_up     = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gptj::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gptj::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        // ln_1: the single norm feeding BOTH attention and the MLP
        ggml_tensor * attn_norm_output = build_norm(inpL,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, il);
        cb(attn_norm_output, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], attn_norm_output,
                    n_embd_head, n_head, n_head_kv, il);

            // rotate_every_two over the first n_rot (= rotary_dim) channels of each head: rope_type NORM
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur              = ggml_get_rows(ctx0,              cur, inp_out_ids);
            inpL             = ggml_get_rows(ctx0,             inpL, inp_out_ids);
            attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
        }

        // MLP on the SAME normed input (parallel residual)
        ggml_tensor * ffn_output = build_ffn(attn_norm_output,
                model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                NULL,                      NULL,                        NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, il);
        cb(ffn_output, "ffn_out", il);

        // x = attn(h) + mlp(h) + residual
        cur = ggml_add(ctx0, cur, ffn_output);
        cur = ggml_add(ctx0, cur, inpL);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = build_norm(inpL,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output_no_bias", -1);

    cur = ggml_add(ctx0, cur, model.output_b);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
