#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// S69 #5/#32d — EAGLE3 draft-head graph (PR #18039 port). Single decoder layer.
// Re-staged onto the upstream-main rebase 2026-06-22 (graph extracted to graphs/; the target-feature
// capture + draft loop now wire into upstream's NATIVE hidden-state spec API —
// llama_set_draft_input_hidden_state_copy / llama_spec_get_hidden_feature_view — see
// DEC-S69-EAGLE3-RESTAGE-RESCOPE-UPSTREAM-NATIVE. This file is the draft-head FORWARD only, our IP.)
// ENCODER: 3 concatenated target hidden states [3*n_embd_tgt, n_tokens] (input "eagle3_target_feats",
//   filled by the spec-decode draft path) -> fc -> g [n_embd, n_tokens].
// DECODER: norm(draft tok_embd) + norm(g) -> concat[2*n_embd] -> q/k/v -> rope -> attn -> +residual
//   (norm_before_residual picks g_norm vs raw g) -> ffn(SILU/PAR) -> +residual -> output_norm -> output.
ggml_cgraph * llm_build_context::build_eagle3() {
    struct ggml_cgraph * gf = new_graph_custom();   // rebased-tree helper (matches build_modernbert et al.)

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    // An EAGLE3 draft head has no token embeddings of its own — it reuses the TARGET model's
    // (set by the spec loop via draft_external_tok_embd). During the load-time WARMUP the target is not
    // attached yet (tok_embd_src null) — use a throwaway zero embedding just to SIZE the graph.
    struct ggml_tensor * tok_embd_src = lctx.draft_external_tok_embd
        ? const_cast<struct ggml_tensor *>(lctx.draft_external_tok_embd)
        : model.tok_embd;

    struct ggml_tensor * inpL;
    if (tok_embd_src) {
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, tok_embd_src, cb);
    } else {
        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inpL);
        cb(inpL, "eagle3_warmup_embd", -1);
    }

    // EAGLE-3 g (arxiv 2503.01840 §3.1): step 1 -> g = FC(concat 3 target feats); steps 2+ (recurrence) ->
    // g = the draft's OWN previous hidden `a`, fed directly (no fc) via eagle3_g_direct.
    struct ggml_tensor * g;
    if (lctx.eagle3_g_direct) {
        struct ggml_tensor * inp_gd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp_gd);
        ggml_set_name(inp_gd, "eagle3_g_direct");
        lctx.inp_eagle3_g_direct = inp_gd;
        lctx.inp_eagle3_feats    = nullptr;
        g = inp_gd;
    } else {
        const int64_t n_embd_inp = hparams.n_embd_tgt > 0 ? 3*(int64_t)hparams.n_embd_tgt : 3*(int64_t)n_embd;
        struct ggml_tensor * inp_g = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp_g);
        ggml_set_name(inp_g, "eagle3_target_feats");
        lctx.inp_eagle3_feats     = inp_g;
        lctx.inp_eagle3_g_direct  = nullptr;
        g = llm_build_lora_mm(lctx, ctx0, model.fc, inp_g);  // -> [n_embd, n_tokens]
    }
    cb(g, "eagle3_g", -1);

    struct ggml_tensor * inp_pos = build_inp_pos();
    struct ggml_tensor * KQ_mask = build_inp_KQ_mask();

    auto & layer = model.layers[0];

    // S69 #33 fix (E2E root-caused): the fused RMS-norm kernel asserts the norm WEIGHT is F32
    // (ggml_compute_forward_fused_rms_norm_f32: src1->nb[0]==sizeof(float)). The EAGLE3 draft gguf may store
    // norm weights at model precision (non-F32) — cast them to F32 here. No-op when already F32 (the canonical
    // gguf case); the proper long-term fix is re-converting with F32 norms (DEC-S69-EAGLE3-FIX-IS-RECONVERT-F32-NORMS).
    struct ggml_tensor * attn_norm_f32   = ggml_cast(ctx0, layer.attn_norm,   GGML_TYPE_F32);
    struct ggml_tensor * attn_norm_2_f32 = ggml_cast(ctx0, layer.attn_norm_2, GGML_TYPE_F32);
    struct ggml_tensor * ffn_norm_f32    = ggml_cast(ctx0, layer.ffn_norm,    GGML_TYPE_F32);
    struct ggml_tensor * output_norm_f32 = ggml_cast(ctx0, model.output_norm, GGML_TYPE_F32);

    struct ggml_tensor * embd_norm = llm_build_norm(ctx0, inpL, hparams, attn_norm_f32,   NULL, LLM_NORM_RMS, cb, 0);
    struct ggml_tensor * g_norm    = llm_build_norm(ctx0, g,    hparams, attn_norm_2_f32, NULL, LLM_NORM_RMS, cb, 0);
    // EAGLE-3 residual (verified vs vLLM llama_eagle3.py): residual is always the fused feature g,
    // NOT the token embedding. norm_before_residual=true -> residual = g_norm; false -> raw g.
    struct ggml_tensor * inpSA     = hparams.norm_before_residual ? g_norm : g;

    // CANONICAL (PR#18039 src/models/eagle3.cpp:217): cur = concat(embd_norm, g_norm). Last swap to [g;e]
    // was wrong (mis-read of paper). Canonical reference confirms [embd; g].
    struct ggml_tensor * cur = ggml_concat(ctx0, embd_norm, g_norm, 0);  // [2*n_embd, n_tokens]

    struct ggml_tensor * Qcur = llm_build_lora_mm(lctx, ctx0, layer.wq, cur);
    struct ggml_tensor * Kcur = llm_build_lora_mm(lctx, ctx0, layer.wk, cur);
    struct ggml_tensor * Vcur = llm_build_lora_mm(lctx, ctx0, layer.wv, cur);
    Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "Qcur", 0); cb(Kcur, "Kcur", 0);

    cur = llm_build_kv(ctx0, lctx, kv_self, gf, layer.wo, layer.bo,
            Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f/sqrtf(float(n_embd_head)), cb, 0);

    // 2026-10-18 (engsrv-1018): skip the rows that produce no output, as the other single-layer graphs here do. The
    // per-sequence pool writes a whole prompt's (token, feature) pairs in one decode with logits only on the last row,
    // and llama_set_inputs asserts inp_out_ids exists whenever n_outputs < n_tokens (llama.cpp:6920). Measured before
    // this: llama-speculative-simple and llama-server aborted on that assert at the first draft batch.
    if (n_tokens > 1 && n_outputs < n_tokens) {
        struct ggml_tensor * inp_out_ids = build_inp_out_ids();
        cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
        inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
    }

    struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
    cb(ffn_inp, "ffn_inp", 0);

    cur = llm_build_ffn(ctx0, lctx, ffn_norm_f32, ffn_inp,
            layer.ffn_up,   NULL, NULL,
            layer.ffn_gate, NULL, NULL,
            layer.ffn_down, NULL, NULL,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, cb, 0, gf, true);
    cur = ggml_add(ctx0, cur, ffn_inp);
    cb(cur, "l_out", 0);
    // tag the draft hidden `a` (pre-output-head) so the eval callback can copy the last-token value into
    // the next recurrence step's g (steps 2+).
    cb(cur, "eagle3_a", 0);

    cur = build_output(lctx, ctx0, cur, model.output, output_norm_f32, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
